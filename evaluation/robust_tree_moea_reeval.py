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

# ----------------------------------------------------------------------
# Experiment scope
# ----------------------------------------------------------------------
INPUT_ROOT = '../data/tree_data_1/'
OUTPUT_ROOT = '../data/tree_data_1'
SETTING = 'robust'

TREE_DEPTH = 9
TREE_CSV = {2: '../trees/depth9_dim2.csv',
            6: '../trees/depth9_dim6.csv'}
EVAL_SLIP_PATH = '../trees/slip_patterns_depth9_eval.npy'

run_moea_method = {'NSGAII': 1, 'IBEA': 1, 'MOEAD': 1}
run_policy = {'table': 1, 'intertemporal': 1}
run_scenario_method = {'multi': 1, 'moro': 1}
run_observability = {'observable': 1, 'non_observable': 1}
run_n_obj = {2: 1, 6: 1}


# Parallelism: set to 1 for serial (debug), 0 for auto (os.cpu_count()),
# or a positive integer for a specific worker count.
N_WORKERS = 0

# ----------------------------------------------------------------------
# Worker-global state (set by _worker_init)
# ----------------------------------------------------------------------
_EVAL_SLIP = None


def _worker_init(eval_slip):
    """ProcessPoolExecutor initializer — broadcasts the eval slip
    patterns to every worker once, instead of pickling into every task."""
    global _EVAL_SLIP
    _EVAL_SLIP = eval_slip


# ----------------------------------------------------------------------
# Simulation primitives
# ----------------------------------------------------------------------
def _eval_one_policy(decisions, *, policy_kind, depth, num_obj,
                     csv_path, observe, slip_patterns, env=None):
    """Replay one policy across all rows of `slip_patterns` and return
    the MEAN per-objective outcome (shape (num_obj,)).

    If `env` is supplied (a prebuilt FruitTreeEnv with the same depth/
    csv/observe settings), we reuse it across scenarios and simply
    swap _slip_pattern — this avoids reading the tree CSV + rebuilding
    a 2**depth tree array on every scenario, which is the bottleneck."""
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


def _reevaluate_archive(archive_path, policy_kind, num_obj, slip_patterns):
    """Returns a (n_policies, num_obj) array of re-evaluated MEAN
    outcomes in MIN-conv (matches the archive's stored 'o*' sign)."""
    df = pd.read_csv(archive_path)
    cols = _decision_cols(df, policy_kind)
    if not cols:
        return None
    levers = df[cols].values
    csv_path = TREE_CSV[num_obj]
    observe = ('non_observable' not in archive_path)
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
    """Self-contained worker task. Takes a dict describing one
    (config, seed) cell, returns a dict with the re-evaluated front
    plus the cell metadata so the parent can assemble HV results."""
    sd_dir = task['sd_dir']
    k = task['seed']
    policy_kind = task['policy']
    scenm = task['scenm']
    num_obj = task['n_obj']
    out_csv_seed = task['out_csv_seed']
    meta = task['meta']  # (method, policy, scenm, obs, n_obj)
    slip_patterns = _EVAL_SLIP

    t0 = time.time()
    archives = sorted(glob.glob(os.path.join(sd_dir, 'archives_*.csv')))
    if not archives:
        return {**meta, 'seed': k, 'front_max': None, 'dt': 0.0,
                'note': 'no archive files found'}

    per_archive_min = []
    for archpath in archives:
        arr = _reevaluate_archive(archpath, policy_kind, num_obj,
                                  slip_patterns)
        if arr is not None:
            per_archive_min.append(arr)

    if not per_archive_min:
        return {**meta, 'seed': k, 'front_max': None,
                'dt': time.time() - t0, 'note': 'no policies parsed'}

    if scenm == 'multi':
        pooled_min = np.vstack(per_archive_min)
        front_max = _nd_filter_max(-pooled_min)
    else:  # moro
        front_max = _nd_filter_max(-per_archive_min[0])

    n_archives = len(per_archive_min)
    n_total_pol = int(sum(len(a) for a in per_archive_min))
    n_dom = n_total_pol - int(len(front_max))

    # Write checkpoint in MIN-conv (convention shared with the MORL
    # scripts and robust_lake_moea_reeval.py: `re_o*` columns are
    # MIN-conv; reload negates back to MAX-conv).
    os.makedirs(os.path.dirname(out_csv_seed), exist_ok=True)
    pd.DataFrame(
        -front_max, columns=[f're_o{j + 1}' for j in range(num_obj)]
    ).to_csv(out_csv_seed, index=False)

    # Sidecar JSON with per-cell aggregates, so a future rerun with this
    # seed's CSV checkpoint still recovers the diagnostic columns.
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
# HV machinery
# ----------------------------------------------------------------------
def _fixed_box(n_obj):
    from params_config import tree_box_dim2, tree_box_dim6
    box = {2: tree_box_dim2, 6: tree_box_dim6}[n_obj]
    nadir = np.asarray(box['nadir'], dtype=float)
    ideal = np.asarray(box['ideal'], dtype=float)
    if not np.all(ideal > nadir):
        raise ValueError(f'Degenerate box for tree dim={n_obj}')
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
    union = np.vstack([c[6] for c in cells])  # c[6] is front_max
    best_known_hv = hv(union)
    meta.update(box_nadir=nadir.tolist(), box_ideal=ideal.tolist(),
                box_volume=box_volume, best_known_hv=best_known_hv,
                n_union_points=int(len(union)))
    return hv, box_volume, meta


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------
def _enabled_stems():
    moea_base = os.path.join(INPUT_ROOT, SETTING, 'moea')
    for n_obj, on in run_n_obj.items():
        if not on: continue
        for method, m_on in run_moea_method.items():
            if not m_on: continue
            for policy, p_on in run_policy.items():
                if not p_on: continue
                for scenm, s_on in run_scenario_method.items():
                    if not s_on: continue
                    for obs, o_on in run_observability.items():
                        if not o_on: continue
                        stem = f'{policy}_{method}_{scenm}_{n_obj}_{obs}'
                        full = os.path.join(moea_base, stem)
                        if os.path.isdir(full):
                            yield method, policy, scenm, n_obj, obs, full


def _build_tasks(eval_slip, reeval_root):
    """Walk the dataset and yield one task dict per (config, seed) cell
    that does NOT already have a checkpoint file. Already-checkpointed
    cells get loaded directly into `cells` (no worker needed)."""
    tasks = []
    prebuilt_cells = []  # (method, policy, scenm, obs, n_obj, seed, F,
                         #  n_archives, n_total_pol, n_dom)
    skipped = 0
    for method, policy, scenm, n_obj, obs, stem_dir in _enabled_stems():
        stem_name = os.path.basename(stem_dir)
        cfg_out_dir = os.path.join(reeval_root, stem_name)
        seeds = sorted(glob.glob(os.path.join(stem_dir, 'seed*')))
        for sd_dir in seeds:
            k = _seed_idx(sd_dir)
            out_csv_seed = os.path.join(cfg_out_dir, f'seed{k}.csv')

            # Checkpoint hit?
            if os.path.exists(out_csv_seed):
                prev = pd.read_csv(out_csv_seed)
                rec = [c for c in prev.columns if re.fullmatch(r're_o\d+', c)]
                if rec:
                    # `re_o*` columns are MIN-conv on disk; negate to
                    # MAX-conv for the downstream HV pipeline.
                    front_max = -prev[rec].values
                    # Sidecar with per-cell diagnostics (n_archives,
                    # n_total_pol, n_dom). Missing sidecar -> NaN, which
                    # cleanly signals "checkpoint pre-dates the metric".
                    side = out_csv_seed.replace('.csv', '_meta.json')
                    if os.path.exists(side):
                        with open(side) as sf:
                            sd_meta = json.load(sf)
                        n_arc = int(sd_meta.get('n_archives', 0))
                        n_tot = int(sd_meta.get('n_total_pol', 0))
                        n_dm  = int(sd_meta.get('n_dom', 0))
                    else:
                        n_arc = n_tot = n_dm = -1  # sentinel: unknown
                    prebuilt_cells.append(
                        (method, policy, scenm, obs, n_obj, k, front_max,
                         n_arc, n_tot, n_dm))
                    skipped += 1
                    continue

            tasks.append({
                'sd_dir': sd_dir,
                'seed': k,
                'policy': policy,
                'scenm': scenm,
                'n_obj': n_obj,
                'out_csv_seed': out_csv_seed,
                'meta': {'method': method, 'policy': policy,
                         'scenm': scenm, 'obs': obs, 'n_obj': n_obj},
            })
    return tasks, prebuilt_cells, skipped


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == '__main__':
    eval_slip = np.load(EVAL_SLIP_PATH)
    assert eval_slip.shape == (200, 2 ** TREE_DEPTH - 1), \
        f'unexpected eval slip-pattern shape: {eval_slip.shape}'
    print(f'loaded {eval_slip.shape[0]} evaluation scenarios '
          f'(depth {TREE_DEPTH})')

    out_root = os.path.join(OUTPUT_ROOT, SETTING, 'moea')
    reeval_root = os.path.join(out_root, 'reeval_archives')
    os.makedirs(reeval_root, exist_ok=True)

    # Plan: build the task list and the prebuilt-cells list separately.
    tasks, cells, skipped = _build_tasks(eval_slip, reeval_root)
    n_total = len(tasks) + skipped
    n_workers = (os.cpu_count() if N_WORKERS == 0 else N_WORKERS) or 1
    n_workers = min(n_workers, max(1, len(tasks)))  # avoid empty pool
    print(f'\n=== Re-evaluating archives ===')
    print(f'   {n_total} cells total; {skipped} already checkpointed, '
          f'{len(tasks)} to compute with {n_workers} worker(s)')

    t0_all = time.time()
    if tasks:
        if n_workers == 1:
            # Serial path — easier to debug.
            _worker_init(eval_slip)
            for i, task in enumerate(tasks, 1):
                res = _process_one_seed(task)
                if res['front_max'] is not None:
                    cells.append((res['method'], res['policy'],
                                  res['scenm'], res['obs'],
                                  res['n_obj'], res['seed'],
                                  res['front_max'],
                                  res['n_archives'],
                                  res['n_total_pol'],
                                  res['n_dom']))
                    print(f'   [{i:3d}/{len(tasks)}] '
                          f"{res['policy']}_{res['scenm']}_{res['n_obj']}_"
                          f"{res['obs']}/seed{res['seed']}  "
                          f"{res['dt']:6.2f}s  {res['n_archives']} arch, "
                          f"{res['n_total_pol']}→{len(res['front_max'])} pols "
                          f"({res['n_dom']} dom)")
                else:
                    print(f'   [{i:3d}/{len(tasks)}] '
                          f"{res['policy']}_{res['scenm']}_{res['n_obj']}_"
                          f"{res['obs']}/seed{res['seed']}  "
                          f"{res['dt']:6.2f}s  {res['note']}")
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
                                      res['scenm'], res['obs'],
                                      res['n_obj'], res['seed'],
                                      res['front_max'],
                                      res['n_archives'],
                                      res['n_total_pol'],
                                      res['n_dom']))
                        print(f'   [{i:3d}/{len(tasks)}] '
                              f"{res['policy']}_{res['scenm']}_{res['n_obj']}_"
                              f"{res['obs']}/seed{res['seed']}  "
                              f"{res['dt']:6.2f}s  {res['n_archives']} arch, "
                              f"{res['n_total_pol']}→{len(res['front_max'])} pols "
                              f"({res['n_dom']} dom)")
                    else:
                        print(f'   [{i:3d}/{len(tasks)}] '
                              f"{res['policy']}_{res['scenm']}_{res['n_obj']}_"
                              f"{res['obs']}/seed{res['seed']}  "
                              f"{res['dt']:6.2f}s  {res['note']}")

    print(f'\n   re-evaluation phase done in {time.time() - t0_all:.1f}s')

    # ------------------------------------------------------------------
    # Fixed-box HV over re-evaluated fronts
    # ------------------------------------------------------------------
    print('\n=== Computing HV on re-evaluated fronts ===')
    by_nobj = defaultdict(list)
    for c in cells:
        by_nobj[c[4]].append(c)  # c[4] is n_obj

    rows = []
    meta = {'setting': SETTING,
            'kind': 'reevaluation',
            'n_eval_scenarios': int(eval_slip.shape[0]),
            'aggregation': 'arithmetic_mean_over_scenarios',
            'multi_rule': 'pool_5_per_scenario_archives_then_nd_filter',
            'n_workers_used': n_workers,
            'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
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
        for method, policy, scenm, obs, no, k, F, n_arc, n_tot, n_dm in panel_cells:
            h = hv(F)
            s = _spacing_norm(F, nadir, ideal)
            cond = f'{policy}_{scenm}_{obs}'
            rows.append(dict(
                paradigm='MOEA', method=method, condition=cond,
                policy=policy, scenario_method=scenm, observability=obs,
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
            print(f'   {m:8s} {c:36s} n={len(vs):3d} '
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
