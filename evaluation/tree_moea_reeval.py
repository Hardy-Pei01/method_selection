import os, re, glob, json, time
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fruit_tree import FruitTreeEnv

from utils import (
    MC_SAMPLES_6OBJ, MC_SEED, _hv_exact, _hv_mc,
    _nd_filter_max, _seed_idx, _spacing_norm,
)

INPUT_ROOT = '../data/tree_data_1/'
OUTPUT_ROOT = '../data/tree_data_1'
SETTING = 'deterministic'

TREE_DEPTH = 9
TREE_CSV = {2: '../trees/depth9_dim2.csv',
            6: '../trees/depth9_dim6.csv'}

run_moea_method = {'NSGAII': 1, 'IBEA': 1, 'MOEAD': 1}
run_policy = {'table': 1, 'intertemporal': 1}
run_observability = {'observable': 1, 'non_observable': 1}
run_n_obj = {2: 1, 6: 1}

# Parallelism: 0 = auto (os.cpu_count()), 1 = serial (debug), N>0 = N workers
N_WORKERS = 0

# Deterministic training folder: `{policy}_{method}_single_{n_obj}_{obs}`
FOLDER_RE = re.compile(
    r'^(intertemporal|table)_(NSGAII|IBEA|MOEAD)_single_(\d+)_'
    r'(observable|non_observable)$')

# ----------------------------------------------------------------------
# Worker-global state (set by _worker_init)
# ----------------------------------------------------------------------
_EVAL_SLIP = None


def _worker_init(eval_slip):
    """Broadcast the eval slip patterns once per worker."""
    global _EVAL_SLIP
    _EVAL_SLIP = eval_slip


def _eval_one_policy(decisions, *, policy_kind, depth, num_obj,
                     csv_path, observe, slip_patterns, env=None):
    """Replay one policy across all rows of `slip_patterns` and return
    the MEAN per-objective outcome (shape (num_obj,)).

    For deterministic re-evaluation, slip_patterns is always shape (1,
    n_internal), so the rollout runs exactly once per policy."""
    n_scen = slip_patterns.shape[0]
    sums = np.zeros(num_obj, dtype=float)
    own_env = env is None
    if own_env:
        env = FruitTreeEnv(depth=depth, reward_dim=num_obj,
                           csv_path=csv_path, observe=bool(observe))

    for s_idx in range(n_scen):
        env._slip_pattern = slip_patterns[s_idx]
        obs, _ = env.reset()
        total = np.zeros(num_obj)
        if policy_kind == 'intertemporal':
            for step in range(depth):
                _, r, term, _, _ = env.step(int(decisions[step]))
                total += r
                if term:
                    break
        elif policy_kind == 'table':
            for _ in range(depth):
                level, pos = obs
                node_id = int(2 ** level - 1) + pos
                action = int(decisions[node_id])
                obs, r, term, _, _ = env.step(action)
                total += r
                if term:
                    break
        else:
            raise ValueError(f'unknown policy_kind {policy_kind}')
        sums += total

    return sums / n_scen


def _decision_cols(df, policy_kind):
    if policy_kind == 'intertemporal':
        cols = [c for c in df.columns if re.fullmatch(r'l\d+', c)]
    elif policy_kind == 'table':
        cols = [c for c in df.columns if re.fullmatch(r'n\d+', c)]
    else:
        raise ValueError(policy_kind)
    return sorted(cols, key=lambda c: int(re.search(r'\d+', c).group()))


def _reevaluate_archive(archive_path, policy_kind, num_obj, slip_patterns,
                        observe):
    """Returns a (n_policies, num_obj) array of re-evaluated MEAN
    outcomes in MIN-conv (matches the archive's stored 'o*' sign)."""
    df = pd.read_csv(archive_path)
    cols = _decision_cols(df, policy_kind)
    if not cols:
        return None
    levers = df[cols].values
    csv_path = TREE_CSV[num_obj]
    env = FruitTreeEnv(depth=TREE_DEPTH, reward_dim=num_obj,
                       csv_path=csv_path, observe=bool(observe))
    out = np.empty((levers.shape[0], num_obj), dtype=float)
    for i, lev in enumerate(levers):
        mean_obj = _eval_one_policy(
            lev, policy_kind=policy_kind, depth=TREE_DEPTH,
            num_obj=num_obj, csv_path=csv_path, observe=observe,
            slip_patterns=slip_patterns, env=env)
        out[i] = -mean_obj
    return out


# ----------------------------------------------------------------------
# Worker task: re-evaluate one (config, seed) cell
# ----------------------------------------------------------------------
def _process_one_seed(task):
    sd_dir = task['sd_dir']
    k = task['seed']
    policy_kind = task['policy']
    obs_str = task['obs']
    num_obj = task['n_obj']
    out_csv_seed = task['out_csv_seed']
    meta = task['meta']
    slip_patterns = _EVAL_SLIP

    t0 = time.time()
    archives = sorted(
        f for f in glob.glob(os.path.join(sd_dir, 'archives_*.csv'))
        if not f.endswith('_evaluated.csv'))
    if not archives:
        return {**meta, 'seed': k, 'front_max': None,
                'dt': 0.0, 'note': 'no archive files'}

    observe = (obs_str == 'observable')
    per_archive_min = []
    for archpath in archives:
        arr = _reevaluate_archive(
            archpath, policy_kind, num_obj, slip_patterns, observe)
        if arr is not None:
            per_archive_min.append(arr)

    if not per_archive_min or sum(len(a) for a in per_archive_min) == 0:
        return {**meta, 'seed': k, 'front_max': None,
                'dt': time.time() - t0, 'note': 'no policies parsed'}

    # Deterministic = single scenario, so single archive per seed; pool
    # if there's somehow more than one (preserves robust-script symmetry).
    pooled_min = np.vstack(per_archive_min)
    front_max = _nd_filter_max(-pooled_min)

    n_archives = len(per_archive_min)
    n_total_pol = int(sum(len(a) for a in per_archive_min))
    n_dom = n_total_pol - int(len(front_max))

    os.makedirs(os.path.dirname(out_csv_seed), exist_ok=True)
    pd.DataFrame(
        -front_max, columns=[f're_o{j + 1}' for j in range(num_obj)]
    ).to_csv(out_csv_seed, index=False)

    side = out_csv_seed.replace('.csv', '_meta.json')
    with open(side, 'w') as f:
        json.dump({'n_archives': n_archives,
                   'n_total_pol': n_total_pol,
                   'n_dom': n_dom}, f)

    return {**meta, 'seed': k, 'front_max': front_max,
            'n_archives': n_archives,
            'n_total_pol': n_total_pol,
            'n_dom': n_dom,
            'dt': time.time() - t0, 'note': 'computed'}


# ----------------------------------------------------------------------
# HV machinery — same deterministic box used by tree_morl_reeval.
# ----------------------------------------------------------------------
def _fixed_box(n_obj):
    from params_config import tree_box_dim2, tree_box_dim6
    box = {2: tree_box_dim2, 6: tree_box_dim6}[n_obj]
    nadir = np.asarray(box['nadir'], dtype=float)
    ideal = np.asarray(box['ideal'], dtype=float)
    if not np.all(ideal > nadir):
        raise ValueError(
            f'Degenerate box for tree dim={n_obj}: '
            f'nadir={nadir.tolist()} ideal={ideal.tolist()}')
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
    union = np.vstack([c[5] for c in cells])  # c[5] is front_max
    best_known_hv = hv(union) if len(union) else 0.0
    meta.update(box_nadir=nadir.tolist(), box_ideal=ideal.tolist(),
                box_volume=box_volume, best_known_hv=best_known_hv,
                n_union_points=int(len(union)))
    return hv, box_volume, meta


# ----------------------------------------------------------------------
# Discovery + task planning
# ----------------------------------------------------------------------
def _enabled_stems():
    moea_base = os.path.join(INPUT_ROOT, SETTING, 'moea')
    if not os.path.isdir(moea_base):
        return
    for d in sorted(os.listdir(moea_base)):
        m = FOLDER_RE.match(d)
        if not m:
            continue
        policy, algo, n_obj_s, obs = m.groups()
        n_obj = int(n_obj_s)
        if not run_moea_method.get(algo, 0): continue
        if not run_policy.get(policy, 0): continue
        if not run_observability.get(obs, 0): continue
        if not run_n_obj.get(n_obj, 0): continue
        yield algo, policy, n_obj, obs, os.path.join(moea_base, d)


def _build_tasks(eval_slip, reeval_root):
    tasks = []
    prebuilt_cells = []  # (method, policy, obs, n_obj, seed, F,
    #  n_archives, n_total_pol, n_dom)
    skipped = 0
    for method, policy, n_obj, obs, stem_dir in _enabled_stems():
        stem_name = os.path.basename(stem_dir)
        cfg_out_dir = os.path.join(reeval_root, stem_name)
        seeds = sorted(glob.glob(os.path.join(stem_dir, 'seed*')))
        for sd_dir in seeds:
            k = _seed_idx(sd_dir)
            out_csv_seed = os.path.join(cfg_out_dir, f'seed{k}.csv')

            if os.path.exists(out_csv_seed):
                prev = pd.read_csv(out_csv_seed)
                rec = [c for c in prev.columns if re.fullmatch(r're_o\d+', c)]
                if rec:
                    front_max = -prev[rec].values
                    side = out_csv_seed.replace('.csv', '_meta.json')
                    if os.path.exists(side):
                        with open(side) as sf:
                            sd_meta = json.load(sf)
                        n_arc = int(sd_meta.get('n_archives', 0))
                        n_tot = int(sd_meta.get('n_total_pol', 0))
                        n_dm = int(sd_meta.get('n_dom', 0))
                    else:
                        n_arc = n_tot = n_dm = -1  # sentinel: unknown
                    prebuilt_cells.append(
                        (method, policy, obs, n_obj, k, front_max,
                         n_arc, n_tot, n_dm))
                    skipped += 1
                    continue

            tasks.append({
                'sd_dir': sd_dir,
                'seed': k,
                'policy': policy,
                'obs': obs,
                'n_obj': n_obj,
                'out_csv_seed': out_csv_seed,
                'meta': {'method': method, 'policy': policy,
                         'obs': obs, 'n_obj': n_obj},
            })
    return tasks, prebuilt_cells, skipped


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print('=' * 64)
    print(f'EVALUATING setting = {SETTING} (tree, MOEA — env replay)')
    print('=' * 64)

    # Default scenario: one no-slip pattern (same as deterministic MORL).
    n_internal = 2 ** TREE_DEPTH - 1
    eval_slip = np.zeros((1, n_internal), dtype=bool)
    print(f'SETTING=deterministic — using 1 no-slip pattern')

    out_root = os.path.join(OUTPUT_ROOT, SETTING, 'moea')
    reeval_root = os.path.join(out_root, 'reeval_archives')
    os.makedirs(reeval_root, exist_ok=True)

    tasks, cells, skipped = _build_tasks(eval_slip, reeval_root)
    n_total = len(tasks) + skipped
    n_workers = (os.cpu_count() if N_WORKERS == 0 else N_WORKERS) or 1
    n_workers = min(n_workers, max(1, len(tasks)))

    print(f'\n=== Re-evaluating archives ({SETTING}) ===')
    print(f'   {n_total} cells total; {skipped} already checkpointed, '
          f'{len(tasks)} to compute with {n_workers} worker(s)')

    t0_all = time.time()
    if tasks:
        if n_workers == 1:
            _worker_init(eval_slip)
            for i, task in enumerate(tasks, 1):
                res = _process_one_seed(task)
                if res['front_max'] is not None:
                    cells.append((res['method'], res['policy'],
                                  res['obs'], res['n_obj'], res['seed'],
                                  res['front_max'],
                                  res['n_archives'],
                                  res['n_total_pol'],
                                  res['n_dom']))
                    print(f'   [{i:3d}/{len(tasks)}] '
                          f"{res['policy']}_{res['n_obj']}_{res['obs']}/"
                          f"seed{res['seed']}  {res['dt']:6.2f}s  "
                          f"{res['n_archives']} arch, "
                          f"{res['n_total_pol']}→{len(res['front_max'])} pols "
                          f"({res['n_dom']} dom)")
                else:
                    print(f'   [{i:3d}/{len(tasks)}] '
                          f"{res['policy']}_{res['n_obj']}_{res['obs']}/"
                          f"seed{res['seed']}  {res['dt']:6.2f}s  "
                          f"{res['note']}")
        else:
            with ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_worker_init,
                    initargs=(eval_slip,)) as pool:
                futures = {pool.submit(_process_one_seed, t): t for t in tasks}
                for i, fut in enumerate(as_completed(futures), 1):
                    res = fut.result()
                    if res['front_max'] is not None:
                        cells.append((res['method'], res['policy'],
                                      res['obs'], res['n_obj'], res['seed'],
                                      res['front_max'],
                                      res['n_archives'],
                                      res['n_total_pol'],
                                      res['n_dom']))
                        print(f'   [{i:3d}/{len(tasks)}] '
                              f"{res['policy']}_{res['n_obj']}_{res['obs']}/"
                              f"seed{res['seed']}  {res['dt']:6.2f}s  "
                              f"{res['n_archives']} arch, "
                              f"{res['n_total_pol']}→{len(res['front_max'])} pols "
                              f"({res['n_dom']} dom)")
                    else:
                        print(f'   [{i:3d}/{len(tasks)}] '
                              f"{res['policy']}_{res['n_obj']}_{res['obs']}/"
                              f"seed{res['seed']}  {res['dt']:6.2f}s  "
                              f"{res['note']}")

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
            'n_eval_scenarios': 1,
            'aggregation': 'identity (single default scenario)',
            'scenario_source': 'no-slip pattern (matches deterministic MORL)',
            'n_workers_used': n_workers,
            'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'note': ('Deterministic MOEA HV scoring with env replay on '
                     'the default scenario. Matches the deterministic '
                     'MORL pipeline so HV comparisons are computed on '
                     'realised returns from the same eval scenario, '
                     'not on archive-stored o*.'),
            'panels': {}}

    for n_obj, panel_cells in sorted(by_nobj.items()):
        hv, box_volume, pmeta = _panel_machinery(panel_cells, n_obj)
        meta['panels'][str(n_obj)] = pmeta
        nadir = np.asarray(pmeta['box_nadir'], dtype=float)
        ideal = np.asarray(pmeta['box_ideal'], dtype=float)
        print(f'\n n_obj={n_obj}: {len(panel_cells)} re-evaluated cells, '
              f"{pmeta['estimator']} HV, box_vol={box_volume:.5g}, "
              f"best-known HV={pmeta['best_known_hv']:.5g} "
              f"(={pmeta['best_known_hv'] / box_volume:.4f} of box)")
        for method, policy, obs, no, k, F, n_arc, n_tot, n_dm in panel_cells:
            h = hv(F)
            s = _spacing_norm(F, nadir, ideal)
            cond = f'{policy}_{obs}'
            rows.append(dict(
                paradigm='MOEA', method=method, condition=cond,
                policy=policy, observability=obs,
                n_obj=no, seed=k,
                n_archives=int(n_arc) if n_arc >= 0 else np.nan,
                n_total_pol=int(n_tot) if n_tot >= 0 else np.nan,
                n_dom=int(n_dm) if n_dm >= 0 else np.nan,
                n_solutions=int(len(F)),
                hv=h, box_volume=box_volume,
                hv_ratio=(h / box_volume if box_volume > 0 else np.nan),
                spacing_norm=s))
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
