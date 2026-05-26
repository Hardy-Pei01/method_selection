import os, re, glob, json, time
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from moocore import hypervolume as _exact_hv
from moocore import is_nondominated as _moo_is_nd

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from two_lake import TwoLakeEnv

# ----------------------------------------------------------------------
# Experiment scope
# ----------------------------------------------------------------------
INPUT_ROOT = '../data/lake_data_2/'
OUTPUT_ROOT = '../data/lake_data_2'
SETTING = 'robust'

EVAL_SCENARIOS_PATH = '../lakes/lake_scenarios_eval.npy'  # (1000,) struct

run_moea_method = {'NSGAII': 1, 'IBEA': 1, 'MOEAD': 1}
run_policy = {'intertemporal': 1, 'dps': 1}
run_scenario_method = {'multi': 1, 'moro': 1}
run_n_obj = {2: 1, 6: 1}

MC_SAMPLES_6OBJ = 50_000
MC_SEED = 12345

# Parallelism: 0 = auto (os.cpu_count()), 1 = serial (debug), N>0 = N workers
N_WORKERS = 0

FOLDER_RE = re.compile(
    r'^(intertemporal|dps)_(NSGAII|IBEA|MOEAD)_(multi|moro)_(\d+)$')

# ----------------------------------------------------------------------
# Worker-global state
# ----------------------------------------------------------------------
_EVAL_SCENARIOS = None  # list of dicts (broadcast once per worker)


def _worker_init(eval_scenarios):
    global _EVAL_SCENARIOS
    _EVAL_SCENARIOS = eval_scenarios


# ----------------------------------------------------------------------
# Scenario + env construction
# ----------------------------------------------------------------------
def _scenario_dict_from_struct(s):
    return {
        'b1': float(s['b1']), 'q1': float(s['q1']),
        'b2': float(s['b2']), 'q2': float(s['q2']),
        'inflow_seed1': int(s['inflow_seed1']),
        'inflow_seed2': int(s['inflow_seed2']),
        'Pcrit1': float(s['Pcrit1']) if s['Pcrit1'] is not None else None,
        'Pcrit2': float(s['Pcrit2']) if s['Pcrit2'] is not None else None,
    }


def _build_env(scenario_dict, n_obj):
    return TwoLakeEnv(
        b1=scenario_dict['b1'], q1=scenario_dict['q1'],
        b2=scenario_dict['b2'], q2=scenario_dict['q2'],
        inflow_seed1=scenario_dict['inflow_seed1'],
        inflow_seed2=scenario_dict['inflow_seed2'],
        Pcrit1=scenario_dict.get('Pcrit1'),
        Pcrit2=scenario_dict.get('Pcrit2'),
        num_obj=n_obj,
    )


def _load_eval_scenarios(path):
    arr = np.load(path)
    return [_scenario_dict_from_struct(s) for s in arr]


# ----------------------------------------------------------------------
# Policy rollouts (MAX-conv)
# ----------------------------------------------------------------------
def _rollout_intertemporal(env, decisions_u1, decisions_u2):
    env.reset()
    total = np.zeros(env.num_obj, dtype=np.float64)
    for t in range(env.n_gym_steps):
        _, reward, _, _, _ = env.step(
            np.array([decisions_u1[t], decisions_u2[t]], dtype=np.int64))
        total += np.asarray(reward, dtype=np.float64)
    return total


def _emission_from_rbf(xt, c1, c2, r1, r2, w1):
    rule = w1 * (abs(xt - c1) / r1) ** 3 + (1 - w1) * (abs(xt - c2) / r2) ** 3
    u = float(np.clip(rule, 0.0, 0.10))
    return int(round(u / 0.02))


def _rollout_dps(env, c1_1, c2_1, r1_1, r2_1, w1_1,
                 c1_2, c2_2, r1_2, r2_2, w1_2):
    env.reset()
    total = np.zeros(env.num_obj, dtype=np.float64)
    for _ in range(env.n_gym_steps):
        X1, X2 = env.X1, env.X2
        u1 = _emission_from_rbf(X1, c1_1, c2_1, r1_1, r2_1, w1_1)
        u2 = _emission_from_rbf(X2, c1_2, c2_2, r1_2, r2_2, w1_2)
        _, reward, _, _, _ = env.step(np.array([u1, u2], dtype=np.int64))
        total += np.asarray(reward, dtype=np.float64)
    return total


def _eval_policy_on_scenarios(row, policy_kind, scenarios, n_obj,
                              u1_keys=None, u2_keys=None):
    """Replay one policy across all `scenarios`; return MEAN per-obj
    realised return (MAX-conv)."""
    sums = np.zeros(n_obj, dtype=np.float64)
    n = len(scenarios)
    if policy_kind == 'intertemporal':
        u1 = [int(row[c]) for c in u1_keys]
        u2 = [int(row[c]) for c in u2_keys]
        for s in scenarios:
            env = _build_env(s, n_obj)
            sums += _rollout_intertemporal(env, u1, u2)
    elif policy_kind == 'dps':
        c1_1, c2_1 = float(row['c1_1']), float(row['c2_1'])
        r1_1, r2_1 = float(row['r1_1']), float(row['r2_1'])
        w1_1 = float(row['w1_1'])
        c1_2, c2_2 = float(row['c1_2']), float(row['c2_2'])
        r1_2, r2_2 = float(row['r1_2']), float(row['r2_2'])
        w1_2 = float(row['w1_2'])
        for s in scenarios:
            env = _build_env(s, n_obj)
            sums += _rollout_dps(env, c1_1, c2_1, r1_1, r2_1, w1_1,
                                 c1_2, c2_2, r1_2, r2_2, w1_2)
    else:
        raise ValueError(f'unknown policy_kind {policy_kind!r}')
    return sums / n


def _reevaluate_archive(archive_path, policy_kind, n_obj, scenarios):
    """Returns (n_policies, n_obj) array of MEAN realised returns in
    MIN-conv (matching MOEA archive sign)."""
    df = pd.read_csv(archive_path)
    n_pol = len(df)
    if n_pol == 0:
        return np.empty((0, n_obj))
    u1_keys = u2_keys = None
    if policy_kind == 'intertemporal':
        u1_keys = sorted(
            [c for c in df.columns if re.fullmatch(r'u1_\d+', c)],
            key=lambda c: int(re.search(r'\d+', c).group()))
        u2_keys = sorted(
            [c for c in df.columns if re.fullmatch(r'u2_\d+', c)],
            key=lambda c: int(re.search(r'\d+', c).group()))
    out_min = np.empty((n_pol, n_obj), dtype=np.float64)
    for i, row in df.iterrows():
        mean_pos = _eval_policy_on_scenarios(
            row, policy_kind, scenarios, n_obj,
            u1_keys=u1_keys, u2_keys=u2_keys)
        out_min[i] = -mean_pos  # MAX-conv -> MIN-conv
    return out_min


def _nd_filter_max(F_max):
    if len(F_max) <= 1:
        return F_max
    mask = _moo_is_nd(-F_max)
    return F_max[mask]


# ----------------------------------------------------------------------
# Worker task: re-evaluate one (config, seed) cell
# ----------------------------------------------------------------------
def _process_one_seed(task):
    sd_dir = task['sd_dir']
    k = task['seed']
    policy_kind = task['policy']
    scenm = task['scenm']
    n_obj = task['n_obj']
    out_csv_seed = task['out_csv_seed']
    meta = task['meta']
    scenarios = _EVAL_SCENARIOS

    t0 = time.time()
    archives = sorted(
        f for f in glob.glob(os.path.join(sd_dir, 'archives_*.csv'))
        if not f.endswith('_evaluated.csv'))
    if not archives:
        return {**meta, 'seed': k, 'front_max': None,
                'dt': 0.0, 'note': 'no archive files'}

    per_archive_min = []
    for archpath in archives:
        arr = _reevaluate_archive(
            archpath, policy_kind, n_obj, scenarios)
        per_archive_min.append(arr)

    if not per_archive_min or sum(len(a) for a in per_archive_min) == 0:
        return {**meta, 'seed': k, 'front_max': None,
                'dt': time.time() - t0, 'note': 'no policies parsed'}

    if scenm == 'multi':
        pooled_min = np.vstack(per_archive_min)
        front_max = _nd_filter_max(-pooled_min)
    else:  # moro
        front_max = _nd_filter_max(-per_archive_min[0])

    os.makedirs(os.path.dirname(out_csv_seed), exist_ok=True)
    pd.DataFrame(
        -front_max, columns=[f're_o{j + 1}' for j in range(n_obj)]
    ).to_csv(out_csv_seed, index=False)

    return {**meta, 'seed': k, 'front_max': front_max,
            'dt': time.time() - t0,
            'n_archives': len(archives),
            'n_total_pol': sum(len(a) for a in per_archive_min),
            'n_dom': sum(len(a) for a in per_archive_min) - len(front_max),
            'note': 'computed'}


# ----------------------------------------------------------------------
# HV machinery — IDENTICAL to lake_morl_reeval.py / lake_moea_reeval.py
# ----------------------------------------------------------------------
def _hv_exact(front_max, ref_min):
    if len(front_max) == 0:
        return 0.0
    return float(_exact_hv(-np.asarray(front_max, float), ref=ref_min))


def _hv_mc(front_max, lo, hi, samples):
    if len(front_max) == 0:
        return 0.0
    F = np.ascontiguousarray(front_max, float)
    N = samples.shape[0]
    dom = np.zeros(N, dtype=bool)
    blk = 2000
    for i in range(0, N, blk):
        S = samples[i:i + blk]
        dom[i:i + blk] = (F[None, :, :] >= S[:, None, :]).all(axis=2).any(axis=1)
    return float(dom.mean() * np.prod(hi - lo))


def _fixed_box(n_obj):
    from params_config import (lake_box_robust_dim2, lake_box_robust_dim6)
    box = {2: lake_box_robust_dim2, 6: lake_box_robust_dim6}[n_obj]
    nadir = np.asarray(box['nadir'], dtype=float)
    ideal = np.asarray(box['ideal'], dtype=float)
    if not np.all(ideal > nadir):
        raise ValueError(f'Degenerate box for lake dim={n_obj}')
    return nadir, ideal


def _panel_machinery(cells, n_obj):
    nadir, ideal = _fixed_box(n_obj)
    box_volume = float(np.prod(ideal - nadir))
    if n_obj == 2:
        ref_min = -nadir
        hv = lambda F: _hv_exact(np.clip(F, nadir, ideal), ref_min)
        meta = dict(estimator='exact')
    else:
        rng = np.random.default_rng(MC_SEED)
        samples = rng.uniform(nadir, ideal, size=(MC_SAMPLES_6OBJ, n_obj))
        hv = lambda F: _hv_mc(np.clip(F, nadir, ideal), nadir, ideal, samples)
        meta = dict(estimator='monte_carlo', mc_samples=MC_SAMPLES_6OBJ,
                    mc_seed=MC_SEED)
    union = np.vstack([c[-1] for c in cells])
    best_known_hv = hv(union)
    meta.update(box_nadir=nadir.tolist(), box_ideal=ideal.tolist(),
                box_volume=box_volume, best_known_hv=best_known_hv,
                n_union_points=int(len(union)))
    return hv, box_volume, meta


# ----------------------------------------------------------------------
# Discovery + task planning
# ----------------------------------------------------------------------
def _seed_idx(p):
    m = re.search(r'seed(\d+)', p)
    return int(m.group(1)) if m else -1


def _enabled_stems():
    moea_base = os.path.join(INPUT_ROOT, SETTING, 'moea')
    if not os.path.isdir(moea_base):
        return
    for d in sorted(os.listdir(moea_base)):
        m = FOLDER_RE.match(d)
        if not m:
            continue
        policy, algo, scenm, n_obj_s = m.groups()
        n_obj = int(n_obj_s)
        if not run_moea_method.get(algo, 0): continue
        if not run_policy.get(policy, 0): continue
        if not run_scenario_method.get(scenm, 0): continue
        if not run_n_obj.get(n_obj, 0): continue
        yield algo, policy, scenm, n_obj, os.path.join(moea_base, d)


def _build_tasks(reeval_root):
    tasks = []
    prebuilt_cells = []  # (algo, policy, scenm, n_obj, seed, F_max)
    skipped = 0
    for algo, policy, scenm, n_obj, stem_dir in _enabled_stems():
        stem_name = os.path.basename(stem_dir)
        cfg_out_dir = os.path.join(reeval_root, stem_name)
        seeds = sorted(glob.glob(os.path.join(stem_dir, 'seed*')))
        for sd_dir in seeds:
            k = _seed_idx(sd_dir)
            out_csv_seed = os.path.join(cfg_out_dir, f'seed{k}.csv')

            if os.path.exists(out_csv_seed):
                prev = pd.read_csv(out_csv_seed)
                rec = [c for c in prev.columns
                       if re.fullmatch(r're_o\d+', c)]
                if rec:
                    F_min = prev[rec].values
                    prebuilt_cells.append(
                        (algo, policy, scenm, n_obj, k, -F_min))
                    skipped += 1
                    continue

            tasks.append({
                'sd_dir': sd_dir,
                'seed': k,
                'policy': policy,
                'scenm': scenm,
                'n_obj': n_obj,
                'out_csv_seed': out_csv_seed,
                'meta': {'algo': algo, 'policy': policy,
                         'scenm': scenm, 'n_obj': n_obj},
            })
    return tasks, prebuilt_cells, skipped


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == '__main__':
    eval_scenarios = _load_eval_scenarios(EVAL_SCENARIOS_PATH)
    print(f'loaded {len(eval_scenarios)} evaluation scenarios '
          f'from {EVAL_SCENARIOS_PATH}')

    out_root = os.path.join(OUTPUT_ROOT, SETTING, 'moea')
    reeval_root = os.path.join(out_root, 'reeval_archives')
    os.makedirs(reeval_root, exist_ok=True)

    tasks, cells, skipped = _build_tasks(reeval_root)
    n_total = len(tasks) + skipped
    n_workers = (os.cpu_count() if N_WORKERS == 0 else N_WORKERS) or 1
    n_workers = min(n_workers, max(1, len(tasks)))

    print(f'\n=== Re-evaluating archives ({SETTING}) ===')
    print(f'   {n_total} cells total; {skipped} already checkpointed, '
          f'{len(tasks)} to compute with {n_workers} worker(s)')

    t0_all = time.time()
    if tasks:
        if n_workers == 1:
            _worker_init(eval_scenarios)
            for i, task in enumerate(tasks, 1):
                res = _process_one_seed(task)
                if res.get('front_max') is not None:
                    cells.append((res['algo'], res['policy'],
                                  res['scenm'], res['n_obj'],
                                  res['seed'], res['front_max']))
                lbl = (f"{res['policy']}_{res['algo']}_{res['scenm']}_"
                       f"{res['n_obj']}/seed{res['seed']}")
                if res.get('note') == 'computed':
                    print(f"   [{i:3d}/{len(tasks)}] {lbl:50s} "
                          f"{res['dt']:6.1f}s  {res['n_archives']} arch, "
                          f"{res['n_total_pol']}→{len(res['front_max'])} pols "
                          f"({res['n_dom']} dom)")
                else:
                    print(f"   [{i:3d}/{len(tasks)}] {lbl:50s}  {res['note']}")
        else:
            with ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_worker_init,
                    initargs=(eval_scenarios,)) as pool:
                futures = {pool.submit(_process_one_seed, t): t for t in tasks}
                for i, fut in enumerate(as_completed(futures), 1):
                    res = fut.result()
                    if res.get('front_max') is not None:
                        cells.append((res['algo'], res['policy'],
                                      res['scenm'], res['n_obj'],
                                      res['seed'], res['front_max']))
                    lbl = (f"{res['policy']}_{res['algo']}_{res['scenm']}_"
                           f"{res['n_obj']}/seed{res['seed']}")
                    if res.get('note') == 'computed':
                        print(f"   [{i:3d}/{len(tasks)}] {lbl:50s} "
                              f"{res['dt']:6.1f}s  {res['n_archives']} arch, "
                              f"{res['n_total_pol']}→{len(res['front_max'])} pols "
                              f"({res['n_dom']} dom)")
                    else:
                        print(f"   [{i:3d}/{len(tasks)}] {lbl:50s}  {res['note']}")

    print(f'\n   re-evaluation phase done in {time.time() - t0_all:.1f}s')

    if not cells:
        raise SystemExit('no MOEA cells available — nothing to score')

    print('\n=== Computing HV on re-evaluated fronts ===')
    by_nobj = defaultdict(list)
    for c in cells:
        by_nobj[c[3]].append(c)  # c[3] is n_obj

    rows = []
    meta = {'setting': SETTING,
            'kind': 'reevaluation_moea',
            'n_eval_scenarios': len(eval_scenarios),
            'aggregation': 'arithmetic_mean_over_scenarios',
            'multi_rule': 'pool_per_scenario_archives_then_nd_filter',
            'n_workers_used': n_workers,
            'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'panels': {}}

    for n_obj, panel_cells in sorted(by_nobj.items()):
        hv, box_volume, pmeta = _panel_machinery(panel_cells, n_obj)
        meta['panels'][str(n_obj)] = pmeta
        print(f'\n n_obj={n_obj}: {len(panel_cells)} re-evaluated cells, '
              f"{pmeta['estimator']} HV, box_vol={box_volume:.5g}, "
              f"best-known HV={pmeta['best_known_hv']:.5g} "
              f"(={pmeta['best_known_hv'] / box_volume:.4f} of box)")
        for algo, policy, scenm, no, k, F in panel_cells:
            h = hv(F)
            cond = f'{policy}_{scenm}'
            rows.append(dict(
                paradigm='MOEA', method=algo, condition=cond,
                policy=policy, scenario_method=scenm,
                n_obj=no, seed=k, n_solutions=int(len(F)),
                hv=h, box_volume=box_volume,
                hv_ratio=(h / box_volume if box_volume > 0 else np.nan)))
        agg = defaultdict(list)
        for r in rows:
            if r['n_obj'] == n_obj:
                agg[(r['method'], r['condition'])].append(r['hv_ratio'])
        for (m, c), vs in sorted(agg.items()):
            print(f'   {m:8s} {c:32s} n={len(vs):3d} '
                  f'HVr={np.mean(vs):.4f}±{np.std(vs):.4f}')

    df = pd.DataFrame(rows).sort_values(
        ['n_obj', 'paradigm', 'method', 'condition', 'seed'])
    out_csv = os.path.join(out_root, 'metrics_long_reeval.csv')
    out_json = os.path.join(out_root, '_meta_reeval.json')
    df.to_csv(out_csv, index=False)
    with open(out_json, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'\n  wrote {out_csv}  ({len(df)} rows)')
    print(f'  wrote {out_json}')
    print(f'  re-evaluated archives under {reeval_root}/')
