import os, re, gzip, pickle, glob, json, time
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from two_lake import TwoLakeEnv
from morl.pql import PQL

from utils import (
    MC_SAMPLES_6OBJ, MC_SEED, _hv_exact, _hv_mc,
    _nd_filter_min, _seed_idx, _spacing_norm, MAX_POLICIES_PER_AGENT
)

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
INPUT_ROOT = '../data/lake_data_2/'
OUTPUT_ROOT = '../data/lake_data_2'

# 'deterministic' or 'robust'
SETTING = 'deterministic'

EVAL_SCENARIOS_PATH = '../lakes/lake_scenarios_eval.npy'

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
_EVAL_SCENARIOS = None  # list of dicts with env params


def _worker_init(scenarios):
    global _EVAL_SCENARIOS
    _EVAL_SCENARIOS = scenarios


# ----------------------------------------------------------------------
# Scenario loading
# ----------------------------------------------------------------------
def _scenario_dict_from_struct(s):
    """Convert a structured-array record (or dict-like) to a plain dict
    of native Python types — pickles cheaply across workers."""
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


def _load_deterministic_morl_scenario():
    """MORL is trained at default_lake_scenario. Re-evaluate there too."""
    from params_config import default_lake_scenario
    return [_scenario_dict_from_struct(default_lake_scenario)]


def _load_robust_eval_scenarios(path):
    arr = np.load(path)
    return [_scenario_dict_from_struct(s) for s in arr]


# ----------------------------------------------------------------------
# PQL agent loading
# ----------------------------------------------------------------------
def _load_agent(agent_path, n_obj):
    with open(agent_path, 'rb') as f:
        magic = f.read(2)
    opener = gzip.open if magic == b'\x1f\x8b' else open
    with opener(agent_path, 'rb') as f:
        payload = pickle.load(f)
    cfg = payload['config']
    saved_robust = bool(cfg.get('robust', False))
    n_scenarios = payload.get('n_scenarios') if saved_robust else None

    env = TwoLakeEnv(num_obj=n_obj)
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


def _pick_action_cached(cache, state_flat, target):
    """L1 target match. Returns (action_flat, next_target, found)."""
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


def _rollout(cache, env, target_vec, n_obj, env_shape, action_nvec):
    """One target-tracked rollout. Returns the per-objective summed reward
    (MAX-conv, since env.step returns MAX-conv rewards)."""
    obs, _ = env.reset()
    target = np.array(target_vec, dtype=float)
    total = np.zeros(n_obj, dtype=np.float64)
    for _ in range(env.n_gym_steps):
        state_flat = int(np.ravel_multi_index(obs, env_shape))
        action_flat, next_target, _ = _pick_action_cached(
            cache, state_flat, target)
        action_nd = np.unravel_index(action_flat, action_nvec)
        obs, reward, terminated, truncated, _ = env.step(
            np.array(action_nd, dtype=np.int64))
        total += np.asarray(reward, dtype=np.float64)
        target = next_target
        if terminated or truncated:
            break
    return total


# ----------------------------------------------------------------------
# Per-agent evaluation
# ----------------------------------------------------------------------
def _evaluate_agent(agent, scenarios, n_obj):
    """Returns (targets, means_pos, diag).
      targets  : (n_pol, n_obj) archive target vectors (MAX-conv)
      means_pos: (n_pol, n_obj) mean realised reward over scenarios
                 (MAX-conv)
      diag     : {'n_archive', 'n_archive_orig', 'mean_corr'}
    """
    decomp = (agent.action_eval == 'decomposition')
    cache = _build_q_cache(agent, decomp)
    env_shape = tuple(int(x) for x in agent.env_shape)
    env_shape_arr = np.array(env_shape)
    action_nvec = tuple(int(x) for x in agent.env.action_space.nvec)

    raw_archive = list(agent.archive)
    n_archive_orig = len(raw_archive)
    if not raw_archive:
        return (np.empty((0, n_obj)), np.empty((0, n_obj)),
                {'n_archive': 0, 'n_archive_orig': 0,
                 'mean_corr': float('nan')})

    # Crowding-distance subsample BEFORE rollouts (saves env replay
    # on policies we'd drop). PQL._subsample_nd canonicalizes via
    # lexsort so the result is reproducible across orderings.
    if (MAX_POLICIES_PER_AGENT is not None and
            n_archive_orig > MAX_POLICIES_PER_AGENT):
        kept = agent._subsample_nd(set(agent.archive),
                                   target_size=MAX_POLICIES_PER_AGENT)
        raw_archive = list(kept)
    archive = [np.asarray(v, dtype=float) for v in raw_archive]

    # Build one env per scenario, reused across targets.
    envs = [_build_env(s, n_obj) for s in scenarios]

    targets = np.zeros((len(archive), n_obj))
    means_pos = np.zeros((len(archive), n_obj))
    for p, target_vec in enumerate(archive):
        scenario_returns = np.zeros((len(envs), n_obj))
        for s_idx, env in enumerate(envs):
            scenario_returns[s_idx] = _rollout(
                cache, env, target_vec, n_obj, env_shape_arr, action_nvec)
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
        'n_archive': len(archive),
        'n_archive_orig': n_archive_orig,
        'mean_corr': mean_corr}


# ----------------------------------------------------------------------
# HV machinery — IDENTICAL to lake_moea_reeval.py to guarantee
# bit-for-bit comparable HV between MOEA and MORL given the same box.
# ----------------------------------------------------------------------
def _fixed_box(n_obj):
    if SETTING == 'deterministic':
        from params_config import (lake_box_deterministic_dim2,
                                   lake_box_deterministic_dim6)
        box = {2: lake_box_deterministic_dim2,
               6: lake_box_deterministic_dim6}[n_obj]
    else:
        from params_config import (lake_box_robust_dim2,
                                   lake_box_robust_dim6)
        box = {2: lake_box_robust_dim2,
               6: lake_box_robust_dim6}[n_obj]
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
    union = np.vstack([c[4] for c in cells])  # c[4] is front_max
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
    scenarios = _EVAL_SCENARIOS

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

    all_targets, all_means, all_ref = [], [], []
    diag_lines = []
    corr_values = []
    for fpath, ref_num in agent_files:
        agent = _load_agent(fpath, n_obj=n_obj)
        if int(agent.num_objectives) != n_obj:
            raise ValueError(
                f'{os.path.basename(fpath)}: num_objectives '
                f'{agent.num_objectives} != folder n_obj {n_obj}')
        targets, means_pos, diag = _evaluate_agent(
            agent, scenarios, n_obj)

        if not np.isnan(diag['mean_corr']):
            corr_values.append(float(diag['mean_corr']))

        ref_tag = f' ref{ref_num}' if ref_num is not None else ''
        corr_str = ('nan' if np.isnan(diag['mean_corr'])
                    else f"{diag['mean_corr']:.2f}")
        if diag['n_archive_orig'] != diag['n_archive']:
            n_str = f"{diag['n_archive_orig']}→{diag['n_archive']} pols"
        else:
            n_str = f"{diag['n_archive']} pols"
        diag_lines.append(f"{ref_tag} {n_str}, corr={corr_str}")

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
    ref_k = agent_ref[keep]

    out = {'policy_id': np.arange(int(keep.sum())),
           'agent_ref': ref_k}
    for j in range(n_obj):
        out[f'target_o{j + 1}'] = targets_k[:, j]
    for j in range(n_obj):
        out[f're_o{j + 1}'] = means_min_k[:, j]
    df_out = pd.DataFrame(out)

    os.makedirs(os.path.dirname(out_csv_seed), exist_ok=True)
    df_out.to_csv(out_csv_seed, index=False)

    cell_mean_corr = (float(np.mean(corr_values))
                      if corr_values else float('nan'))
    n_total_pol = int(len(means_min))
    n_dom = int((~keep).sum())

    # Sidecar JSON with per-cell aggregates so a future rerun with this
    # seed's CSV checkpoint still recovers the diagnostic columns.
    side = out_csv_seed.replace('.csv', '_meta.json')
    with open(side, 'w') as f:
        json.dump({'n_agents': len(agent_files),
                   'n_total_pol': n_total_pol,
                   'n_dom': n_dom,
                   'mean_corr': cell_mean_corr}, f)

    return {**meta, 'seed': k,
            'front_min': means_min_k,
            'rows': len(df_out),
            'n_agents': len(agent_files),
            'n_total_pol': n_total_pol,
            'n_dom': n_dom,
            'mean_corr': cell_mean_corr,
            'dt': time.time() - t0,
            'diag': '; '.join(diag_lines),
            'note': 'computed'}


# ----------------------------------------------------------------------
# Discovery + task planning
# ----------------------------------------------------------------------
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
    tasks = []
    prebuilt_cells = []  # (scoring, scenm, n_obj, seed, F_min,
    #  n_total_pol, n_dom, mean_corr)
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
                    side = out_csv_seed.replace('.csv', '_meta.json')
                    n_tot = n_dm = -1
                    mc = float('nan')
                    if os.path.exists(side):
                        try:
                            with open(side) as sf:
                                sd_meta = json.load(sf)
                            n_tot = int(sd_meta.get('n_total_pol', -1))
                            n_dm = int(sd_meta.get('n_dom', -1))
                            mc = float(sd_meta.get('mean_corr',
                                                   float('nan')))
                        except Exception:
                            pass
                    prebuilt_cells.append(
                        (scoring, scenm, n_obj, k, F_min,
                         n_tot, n_dm, mc))
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
    if SETTING == 'deterministic':
        scenarios = _load_deterministic_morl_scenario()
        print(f'SETTING=deterministic — using 1 scenario '
              f'(default_lake_scenario)')
    elif SETTING == 'robust':
        scenarios = _load_robust_eval_scenarios(EVAL_SCENARIOS_PATH)
        print(f'SETTING=robust — loaded {len(scenarios)} eval scenarios '
              f'from {EVAL_SCENARIOS_PATH}')
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

    cells = list(prebuilt_cells)
    t0_all = time.time()
    if tasks:
        if n_workers == 1:
            _worker_init(scenarios)
            for i, task in enumerate(tasks, 1):
                res = _process_one_seed(task)
                if res.get('front_min') is not None:
                    cells.append((res['scoring'], res['scenm'],
                                  res['n_obj'], res['seed'],
                                  res['front_min'],
                                  res['n_total_pol'], res['n_dom'],
                                  res['mean_corr']))
                _log_line(i, len(tasks), res)
        else:
            with ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_worker_init,
                    initargs=(scenarios,)) as pool:
                futures = {pool.submit(_process_one_seed, t): t for t in tasks}
                for i, fut in enumerate(as_completed(futures), 1):
                    res = fut.result()
                    if res.get('front_min') is not None:
                        cells.append((res['scoring'], res['scenm'],
                                      res['n_obj'], res['seed'],
                                      res['front_min'],
                                      res['n_total_pol'], res['n_dom'],
                                      res['mean_corr']))
                    _log_line(i, len(tasks), res)

    print(f'\n   MORL re-evaluation done in {time.time() - t0_all:.1f}s')
    print(f'   re-evaluated archives under {reeval_root}/')

    if not cells:
        raise SystemExit('no MORL cells available — nothing to score')

    print('\n=== Computing HV on re-evaluated MORL fronts ===')
    panel_cells_max = [(sc, scen, n, k, -F_min, n_tot, n_dm, mc)
                       for sc, scen, n, k, F_min, n_tot, n_dm, mc in cells]
    by_nobj = defaultdict(list)
    for c in panel_cells_max:
        by_nobj[c[2]].append(c)

    rows = []
    meta = {'setting': SETTING,
            'kind': 'reevaluation_morl',
            'n_eval_scenarios': len(scenarios),
            'aggregation': 'arithmetic_mean_over_scenarios',
            'multi_rule': 'pool_5_per_seed_target_tracked_then_nd_filter',
            'n_workers_used': n_workers,
            'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'panels': {}}

    for n_obj, pcells in sorted(by_nobj.items()):
        hv, box_volume, pmeta = _panel_machinery(pcells, n_obj)
        meta['panels'][str(n_obj)] = pmeta
        nadir = np.asarray(pmeta['box_nadir'], dtype=float)
        ideal = np.asarray(pmeta['box_ideal'], dtype=float)
        print(f'\n n_obj={n_obj}: {len(pcells)} re-evaluated cells, '
              f"{pmeta['estimator']} HV, box_vol={box_volume:.5g}, "
              f"best-known HV={pmeta['best_known_hv']:.5g} "
              f"(={pmeta['best_known_hv'] / box_volume:.4f} of box)")
        for scoring, scenm, no, k, F, n_tot, n_dm, mc in pcells:
            h = hv(F)
            s = _spacing_norm(F, nadir, ideal)
            cond = f'closed_loop_{scenm}'
            rows.append(dict(
                paradigm='MORL', method=scoring, condition=cond,
                scoring=scoring, scenario_method=scenm,
                n_obj=no, seed=k,
                n_total_pol=int(n_tot) if n_tot >= 0 else np.nan,
                n_dom=int(n_dm) if n_dm >= 0 else np.nan,
                n_solutions=int(len(F)),
                mean_corr=float(mc),
                hv=h, box_volume=box_volume,
                hv_ratio=(h / box_volume if box_volume > 0 else np.nan),
                spacing_norm=s))
        agg = defaultdict(list)
        for r in rows:
            if r['n_obj'] == n_obj:
                agg[(r['method'], r['condition'])].append(r['hv_ratio'])
        for (m, c), vs in sorted(agg.items()):
            print(f'   {m:14s} {c:32s} n={len(vs):3d} '
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
