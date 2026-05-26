import os, re, gzip, pickle, glob, json, time
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fruit_tree import FruitTreeEnv
from morl.pql import PQL

try:
    from moocore import is_nondominated as _moo_is_nd
except ImportError:
    _moo_is_nd = None

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
INPUT_ROOT = '../data/tree_data_1/'
OUTPUT_ROOT = '../data/tree_data_1'

# 'deterministic' or 'robust'
SETTING = 'robust'

TREE_DEPTH = 9
TREE_CSV = {2: '../trees/depth9_dim2.csv',
            6: '../trees/depth9_dim6.csv'}
# Path is only consulted when SETTING == 'robust'.
EVAL_SLIP_PATH = '../trees/slip_patterns_depth9_eval.npy'  # (200, 511)

# Per-setting scope. Scoring × scenario_method × n_obj cells.
run_scoring = {'pareto': 1, 'indicator': 1, 'decomposition': 1}
run_scenario_method_for_setting = {
    'deterministic': ['single'],
    'robust': ['multi', 'moro'],
}
run_n_obj = {2: 1, 6: 1}

# Parallelism: 0 = auto (os.cpu_count()), 1 = serial (debug), N>0 = N workers
N_WORKERS = 0

FOLDER_RE = re.compile(
    r'^(pareto|indicator|decomposition)_(single|multi|moro)_(\d+)$')
AGENT_RE = re.compile(r'^agent_(\d+)(?:_(\d+))?\.pkl$')

# ----------------------------------------------------------------------
# Worker-global state (set by ProcessPoolExecutor initializer)
# ----------------------------------------------------------------------
_EVAL_SLIP = None  # (n_eval, n_internal_nodes) bool array


def _worker_init(eval_slip):
    global _EVAL_SLIP
    _EVAL_SLIP = eval_slip


# ----------------------------------------------------------------------
# PQL agent loading
# ----------------------------------------------------------------------
def _load_agent(agent_path, n_obj, csv_path):
    """Construct a PQL agent and restore its Q-table from disk.
    PQL.load_q_table restores gamma, ref_point, scoring, etc. from the
    saved config. Robust agents need n_scenarios at construction; we
    peek at the saved config to learn it before instantiation."""
    with open(agent_path, 'rb') as f:
        magic = f.read(2)
    opener = gzip.open if magic == b'\x1f\x8b' else open
    with opener(agent_path, 'rb') as f:
        payload = pickle.load(f)
    cfg = payload['config']
    saved_robust = bool(cfg.get('robust', False))
    n_scenarios = payload.get('n_scenarios') if saved_robust else None

    env = FruitTreeEnv(
        depth=TREE_DEPTH, reward_dim=n_obj, csv_path=csv_path,
        observe=True, scenario_index=None, slip_patterns_path=None,
    )
    kwargs = dict(env=env, ref_point=np.asarray(cfg['ref_point']))
    if saved_robust and n_scenarios is not None:
        kwargs['robust'] = True
        kwargs['n_scenarios'] = n_scenarios
    agent = PQL(**kwargs)
    agent.load_q_table(agent_path)
    return agent


# ----------------------------------------------------------------------
# Q-cache + target-tracking rollout
# ----------------------------------------------------------------------
def _build_q_cache(agent, decomp):
    """Per-(state, action) cached (Q_raw, Qsa) arrays.
    Built once per agent; reused across every (target, scenario) rollout."""
    cache = {}
    gamma = agent.gamma
    nd_dict = agent.nd_decomp if decomp else agent.non_dominated
    for state, counts in agent.counts.items():
        per_action = {}
        for a in range(agent.num_actions):
            if counts[a] == 0:
                continue
            nd_set = nd_dict[state][a]
            if not nd_set:
                continue
            Q = np.array(list(nd_set), dtype=float)
            Qsa = gamma * Q + agent.avg_reward[state][a]
            per_action[a] = (Q, Qsa)
        if per_action:
            cache[state] = per_action
    return cache


def _pick_action_cached(cache, state_flat, target, num_actions):
    """L1 target match. Returns (action, next_target, found)."""
    per_action = cache.get(state_flat)
    if per_action is None:
        return 0, target, False

    best_action, best_dist, next_target = None, np.inf, target
    for a, (Q, Qsa) in per_action.items():
        dists = np.abs(Qsa - target).sum(axis=1)
        i = int(np.argmin(dists))
        if dists[i] < best_dist:
            best_dist = float(dists[i])
            best_action = a
            next_target = Q[i]
    if best_action is None:
        return 0, target, False
    return best_action, next_target, True


def _rollout(cache, env, target_vec, depth, env_shape,
             n_obj, num_actions):
    """One target-tracked rollout. Slip pattern already set on env.
    Returns the per-objective summed reward (MAX-conv)."""
    obs, _ = env.reset()
    target = np.array(target_vec, dtype=float)
    total = np.zeros(n_obj)
    for _ in range(depth):
        state_flat = int(np.ravel_multi_index(obs, env_shape))
        action, next_target, _ = _pick_action_cached(
            cache, state_flat, target, num_actions)
        obs, reward, terminal, _, _ = env.step(action)
        total += reward
        target = next_target
        if terminal:
            break
    return total


def _action_sequence(cache, env, target_vec, depth, env_shape,
                     num_actions):
    """Action sequence under no slip — diagnostic only."""
    obs, _ = env.reset()
    env._slip_pattern = np.zeros(2 ** depth - 1, dtype=bool)
    target = np.array(target_vec, dtype=float)
    actions = []
    for _ in range(depth):
        state_flat = int(np.ravel_multi_index(obs, env_shape))
        action, next_target, _ = _pick_action_cached(
            cache, state_flat, target, num_actions)
        actions.append(action)
        obs, _, terminal, _, _ = env.step(action)
        target = next_target
        if terminal:
            break
    return tuple(actions)


# ----------------------------------------------------------------------
# Per-agent evaluation
# ----------------------------------------------------------------------
def _evaluate_agent(agent, eval_patterns, n_obj, depth):
    """Returns (targets, means_pos, diag).
      targets  : (n_pol, n_obj) archive target vectors (MAX-conv)
      means_pos: (n_pol, n_obj) mean realised reward over eval patterns
                 (MAX-conv)
      diag     : {'n_unique_seq', 'n_archive', 'mean_corr'}
    """
    decomp = (agent.action_eval == 'decomposition')
    cache = _build_q_cache(agent, decomp)
    env = agent.env
    env_shape = agent.env_shape
    n_actions = agent.num_actions

    archive = [np.asarray(v, dtype=float) for v in agent.archive]
    if not archive:
        return (np.empty((0, n_obj)), np.empty((0, n_obj)),
                {'n_unique_seq': 0, 'n_archive': 0,
                 'mean_corr': float('nan')})

    targets = np.zeros((len(archive), n_obj))
    means_pos = np.zeros((len(archive), n_obj))
    seqs = set()
    for p, target_vec in enumerate(archive):
        seqs.add(_action_sequence(cache, env, target_vec, depth,
                                  env_shape, n_actions))
        scenario_returns = np.zeros((len(eval_patterns), n_obj))
        for s, pat in enumerate(eval_patterns):
            env._slip_pattern = pat
            scenario_returns[s] = _rollout(
                cache, env, target_vec, depth, env_shape,
                n_obj, n_actions)
        targets[p] = target_vec
        means_pos[p] = scenario_returns.mean(axis=0)

    corrs = []
    for j in range(n_obj):
        if (np.std(targets[:, j]) > 1e-9 and
                np.std(means_pos[:, j]) > 1e-9):
            corrs.append(
                np.corrcoef(targets[:, j], means_pos[:, j])[0, 1])
    mean_corr = float(np.nanmean(corrs)) if corrs else float('nan')

    return targets, means_pos, {
        'n_unique_seq': len(seqs), 'n_archive': len(archive),
        'mean_corr': mean_corr,
    }


# ----------------------------------------------------------------------
# ND filter (MIN-conv input)
# ----------------------------------------------------------------------
def _nd_filter_min(values):
    arr = np.asarray(values, dtype=float)
    if arr.shape[0] <= 1:
        return np.ones(arr.shape[0], dtype=bool)
    if _moo_is_nd is not None:
        return _moo_is_nd(arr)
    n = arr.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if np.all(arr[j] <= arr[i]) and np.any(arr[j] < arr[i]):
                keep[i] = False
                break
    return keep


# ----------------------------------------------------------------------
# HV machinery — IDENTICAL to robust_tree_moea_reeval.py to guarantee
# bit-for-bit comparable HV between MOEA and MORL given the same box.
# ----------------------------------------------------------------------
from moocore import hypervolume as _exact_hv

MC_SAMPLES_6OBJ = 50_000
MC_SEED = 12345


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
    union = np.vstack([c[-1] for c in cells])
    best_known_hv = hv(union)
    meta.update(box_nadir=nadir.tolist(), box_ideal=ideal.tolist(),
                box_volume=box_volume, best_known_hv=best_known_hv,
                n_union_points=int(len(union)))
    return hv, box_volume, meta


# ----------------------------------------------------------------------
# Worker: re-evaluate one (config, seed) cell
# ----------------------------------------------------------------------
def _process_one_seed(task):
    sd_dir = task['sd_dir']
    k = task['seed']
    scoring = task['scoring']
    scenm = task['scenm']
    n_obj = task['n_obj']
    out_csv_seed = task['out_csv_seed']
    meta = task['meta']
    eval_patterns = _EVAL_SLIP

    t0 = time.time()
    # Discover agent files in this seed folder.
    agent_files = []
    for fname in sorted(os.listdir(sd_dir)):
        am = AGENT_RE.match(fname)
        if not am:
            continue
        nfe, ref_num = am.groups()
        ref_num = int(ref_num) if ref_num is not None else None
        agent_files.append((os.path.join(sd_dir, fname), ref_num))
    if not agent_files:
        return {**meta, 'seed': k, 'front_min': None,
                'rows': None, 'dt': time.time() - t0,
                'note': 'no agent files'}

    csv_path = TREE_CSV[n_obj]
    all_targets, all_means, all_ref = [], [], []
    diag_lines = []
    for fpath, ref_num in agent_files:
        agent = _load_agent(fpath, n_obj=n_obj, csv_path=csv_path)
        if int(agent.num_objectives) != n_obj:
            raise ValueError(
                f'{os.path.basename(fpath)}: num_objectives '
                f'{agent.num_objectives} != folder n_obj {n_obj}')
        targets, means_pos, diag = _evaluate_agent(
            agent, eval_patterns, n_obj, TREE_DEPTH)

        ref_tag = f' ref{ref_num}' if ref_num is not None else ''
        corr_str = ('nan' if np.isnan(diag['mean_corr'])
                    else f"{diag['mean_corr']:.2f}")
        diag_lines.append(
            f"{ref_tag} {diag['n_unique_seq']}/{diag['n_archive']} "
            f"unique, corr={corr_str}")

        all_targets.append(targets)
        all_means.append(means_pos)
        ref_id = ref_num if ref_num is not None else 0
        all_ref.append(np.full(len(targets), ref_id, dtype=int))

    targets = np.vstack(all_targets)
    means_pos = np.vstack(all_means)
    agent_ref = np.concatenate(all_ref)
    means_min = -means_pos  # MIN-conv for HV pipeline

    # ND-filter on the realised returns (across the merged set for multi).
    keep = _nd_filter_min(means_min)
    targets_k = targets[keep]
    means_min_k = means_min[keep]
    means_pos_k = means_pos[keep]
    ref_k = agent_ref[keep]

    # Build the per-row output.
    out = {'policy_id': np.arange(int(keep.sum())),
           'agent_ref': ref_k}
    for j in range(n_obj):
        out[f'target_o{j + 1}'] = targets_k[:, j]
    for j in range(n_obj):
        out[f're_o{j + 1}'] = means_min_k[:, j]
    df_out = pd.DataFrame(out)

    os.makedirs(os.path.dirname(out_csv_seed), exist_ok=True)
    df_out.to_csv(out_csv_seed, index=False)

    return {**meta, 'seed': k,
            'front_min': means_min_k,
            'rows': len(df_out),
            'n_agents': len(agent_files),
            'n_total_pol': len(means_min),
            'n_dom': int((~keep).sum()),
            'dt': time.time() - t0,
            'diag': '; '.join(diag_lines),
            'note': 'computed'}


# ----------------------------------------------------------------------
# Discovery + task planning
# ----------------------------------------------------------------------
def _seed_idx(p):
    m = re.search(r'seed(\d+)', p)
    return int(m.group(1)) if m else -1


def _enabled_stems():
    """yield (scoring, scenm, n_obj, stem_dir)"""
    morl_base = os.path.join(INPUT_ROOT, SETTING, 'morl')
    if not os.path.isdir(morl_base):
        return
    allowed_scenms = set(run_scenario_method_for_setting[SETTING])
    for d in sorted(os.listdir(morl_base)):
        m = FOLDER_RE.match(d)
        if not m:
            continue
        scoring, scenm, n_obj_s = m.groups()
        n_obj = int(n_obj_s)
        if scenm not in allowed_scenms:
            continue
        if not run_scoring.get(scoring, 0):
            continue
        if not run_n_obj.get(n_obj, 0):
            continue
        yield scoring, scenm, n_obj, os.path.join(morl_base, d)


def _build_tasks(reeval_root):
    """Walk MORL folders, build per-(config, seed) task dicts, and
    pre-load any cells that are already checkpointed."""
    tasks = []
    prebuilt_cells = []  # (scoring, scenm, n_obj, seed, F_min)
    skipped = 0
    for scoring, scenm, n_obj, stem_dir in _enabled_stems():
        stem_name = os.path.basename(stem_dir)
        cfg_out_dir = os.path.join(reeval_root, stem_name)
        for sd in sorted(os.listdir(stem_dir)):
            sd_dir = os.path.join(stem_dir, sd)
            if not os.path.isdir(sd_dir):
                continue
            k = _seed_idx(sd_dir)
            out_csv_seed = os.path.join(cfg_out_dir, f'seed{k}.csv')

            if os.path.exists(out_csv_seed):
                prev = pd.read_csv(out_csv_seed)
                re_cols = [c for c in prev.columns
                           if re.fullmatch(r're_o\d+', c)]
                if re_cols:
                    F_min = prev[re_cols].values
                    prebuilt_cells.append(
                        (scoring, scenm, n_obj, k, F_min))
                    skipped += 1
                    continue

            tasks.append({
                'sd_dir': sd_dir,
                'seed': k,
                'scoring': scoring,
                'scenm': scenm,
                'n_obj': n_obj,
                'out_csv_seed': out_csv_seed,
                'meta': {'scoring': scoring, 'scenm': scenm,
                         'n_obj': n_obj},
            })
    return tasks, prebuilt_cells, skipped


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def _log_line(i, n_tasks, res):
    """Pretty-print one task's result."""
    label = f"{res['scoring']}_{res['scenm']}_{res['n_obj']}/seed{res['seed']}"
    if res.get('note') == 'computed':
        print(f"   [{i:3d}/{n_tasks}] {label:42s} "
              f"{res['dt']:6.1f}s  "
              f"{res['n_agents']} agent(s), "
              f"{res['n_total_pol']}→{res['rows']} pols "
              f"({res['n_dom']} dom)  diag: {res['diag']}")
    else:
        print(f"   [{i:3d}/{n_tasks}] {label:42s}  {res['note']}")


if __name__ == '__main__':
    # Choose evaluation slip patterns by setting.
    if SETTING == 'deterministic':
        n_internal = 2 ** TREE_DEPTH - 1
        eval_slip = np.zeros((1, n_internal), dtype=bool)
        print(f'SETTING=deterministic — using 1 no-slip pattern')
    elif SETTING == 'robust':
        eval_slip = np.load(EVAL_SLIP_PATH)
        assert eval_slip.shape[1] == 2 ** TREE_DEPTH - 1, \
            f'eval-slip width {eval_slip.shape[1]} != 2**{TREE_DEPTH}-1'
        print(f'SETTING=robust — loaded {eval_slip.shape[0]} eval '
              f'patterns from {EVAL_SLIP_PATH}')
    else:
        raise SystemExit(f'unknown SETTING {SETTING!r}')

    out_root = os.path.join(OUTPUT_ROOT, SETTING, 'morl')
    reeval_root = os.path.join(out_root, 'reeval_archives_morl')
    os.makedirs(reeval_root, exist_ok=True)

    tasks, prebuilt_cells, skipped = _build_tasks(reeval_root)
    n_total = len(tasks) + skipped
    n_workers = (os.cpu_count() if N_WORKERS == 0 else N_WORKERS) or 1
    n_workers = min(n_workers, max(1, len(tasks)))

    print(f'\n=== Re-evaluating MORL agents ({SETTING}) ===')
    print(f'   {n_total} cells total; {skipped} checkpointed, '
          f'{len(tasks)} to compute with {n_workers} worker(s)')

    cells = list(prebuilt_cells)  # (scoring, scenm, n_obj, seed, F_min)
    t0_all = time.time()
    if tasks:
        if n_workers == 1:
            _worker_init(eval_slip)
            for i, task in enumerate(tasks, 1):
                res = _process_one_seed(task)
                if res.get('front_min') is not None:
                    cells.append((res['scoring'], res['scenm'],
                                  res['n_obj'], res['seed'],
                                  res['front_min']))
                _log_line(i, len(tasks), res)
        else:
            with ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_worker_init,
                    initargs=(eval_slip,)) as pool:
                futures = {pool.submit(_process_one_seed, t): t for t in tasks}
                for i, fut in enumerate(as_completed(futures), 1):
                    res = fut.result()
                    if res.get('front_min') is not None:
                        cells.append((res['scoring'], res['scenm'],
                                      res['n_obj'], res['seed'],
                                      res['front_min']))
                    _log_line(i, len(tasks), res)

    print(f'\n   MORL re-evaluation done in {time.time() - t0_all:.1f}s')
    print(f'   re-evaluated archives under {reeval_root}/')

    # ------------------------------------------------------------------
    # Fixed-box HV over re-evaluated MORL fronts. Same machinery and box
    # as robust_tree_moea_reeval.py — guarantees comparable HV ratios
    # between paradigms.
    # ------------------------------------------------------------------
    if not cells:
        raise SystemExit('no MORL cells available — nothing to score')

    print('\n=== Computing HV on re-evaluated MORL fronts ===')
    # cells: list of (scoring, scenm, n_obj, seed, F_min)
    # _panel_machinery expects last element to be the front; convert
    # MIN-conv -> MAX-conv inline.
    panel_cells_max = [(sc, scen, n, k, -F_min) for sc, scen, n, k, F_min in cells]

    by_nobj = defaultdict(list)
    for c in panel_cells_max:
        by_nobj[c[2]].append(c)   # c[2] is n_obj

    rows = []
    meta = {'setting': SETTING,
            'kind': 'reevaluation_morl',
            'aggregation': 'arithmetic_mean_over_scenarios',
            'multi_rule': 'pool_5_per_seed_target_tracked_then_nd_filter',
            'n_workers_used': n_workers,
            'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'panels': {}}

    for n_obj, pcells in sorted(by_nobj.items()):
        hv, box_volume, pmeta = _panel_machinery(pcells, n_obj)
        meta['panels'][str(n_obj)] = pmeta
        print(f'\n n_obj={n_obj}: {len(pcells)} re-evaluated cells, '
              f"{pmeta['estimator']} HV, box_vol={box_volume:.5g}, "
              f"best-known HV={pmeta['best_known_hv']:.5g} "
              f"(={pmeta['best_known_hv'] / box_volume:.4f} of box)")
        for scoring, scenm, no, k, F in pcells:
            h = hv(F)
            cond = f'closed_loop_{scenm}'
            rows.append(dict(
                paradigm='MORL', method=scoring, condition=cond,
                scoring=scoring, scenario_method=scenm,
                n_obj=no, seed=k, n_solutions=int(len(F)),
                hv=h, box_volume=box_volume,
                hv_ratio=(h / box_volume if box_volume > 0 else np.nan)))
        agg = defaultdict(list)
        for r in rows:
            if r['n_obj'] == n_obj:
                agg[(r['method'], r['condition'])].append(r['hv_ratio'])
        for (m, c), vs in sorted(agg.items()):
            print(f'   {m:14s} {c:32s} n={len(vs):3d} '
                  f'HVr={np.mean(vs):.4f}±{np.std(vs):.4f}')

    df = pd.DataFrame(rows).sort_values(
        ['n_obj', 'paradigm', 'method', 'condition', 'seed'])
    out_csv  = os.path.join(out_root, 'metrics_long_reeval.csv')
    out_json = os.path.join(out_root, '_meta_reeval.json')
    df.to_csv(out_csv, index=False)
    with open(out_json, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'\n  wrote {out_csv}  ({len(df)} rows)')
    print(f'  wrote {out_json}')
