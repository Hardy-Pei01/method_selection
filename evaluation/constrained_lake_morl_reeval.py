import os, re, gzip, pickle, glob, json, time
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constrained_two_lake import ConstrainedTwoLakeEnv
from morl.pql import PQL

try:
    from moocore import is_nondominated as _moo_is_nd
except ImportError:
    _moo_is_nd = None

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
INPUT_ROOT  = '../data/constrained_data_1/'
OUTPUT_ROOT = '../data/constrained_data_1'

# 'deterministic' or 'robust'
SETTING = 'robust'

EVAL_SCENARIOS_PATH = '../lakes/lake_scenarios_eval.npy'   # (1000,) struct

run_scoring = {'pareto': 1, 'indicator': 1, 'decomposition': 1}
run_scenario_method_for_setting = {
    'deterministic': ['single'],
    'robust':        ['multi', 'moro'],
}
run_n_obj = {2: 1, 6: 1}

# Cap on archive size per agent. Crowding-distance subsample before
# rollouts. Set to None to disable.
MAX_POLICIES_PER_AGENT = 1000

# Parallelism: 0 = auto (os.cpu_count()), 1 = serial (debug), N>0 = N workers
N_WORKERS = 0

# Stem-folder pattern (mirrors lake_morl_reeval.py): seeds are
# subdirectories `seed*/` inside each stem.
FOLDER_RE = re.compile(
    r'^(pareto|indicator|decomposition)_(single|multi|moro)_(\d+)$')
AGENT_RE  = re.compile(r'^agent_(\d+)(?:_(\d+))?\.pkl$')


# ----------------------------------------------------------------------
# Worker-global state (set by ProcessPoolExecutor initializer)
# ----------------------------------------------------------------------
_EVAL_SCENARIOS = None


def _worker_init(scenarios):
    global _EVAL_SCENARIOS
    _EVAL_SCENARIOS = scenarios


# ----------------------------------------------------------------------
# Scenario loading
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
    return ConstrainedTwoLakeEnv(
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

    env = ConstrainedTwoLakeEnv(num_obj=n_obj)
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
    """One target-tracked rollout. Returns (total_reward, feasible).
    `feasible` reflects info['feasible'] at the LAST step of the
    episode — which by construction equals `_n_violations_*` == 0
    over the whole episode (the env accumulates and `reset()` zeros
    them, per the canonical contract)."""
    obs, _ = env.reset()
    target = np.array(target_vec, dtype=float)
    total = np.zeros(n_obj, dtype=np.float64)
    feasible = True
    for _ in range(env.n_gym_steps):
        state_flat = int(np.ravel_multi_index(obs, env_shape))
        action_flat, next_target, _ = _pick_action_cached(
            cache, state_flat, target)
        action_nd = np.unravel_index(action_flat, action_nvec)
        obs, reward, terminated, truncated, info = env.step(
            np.array(action_nd, dtype=np.int64))
        total += np.asarray(reward, dtype=np.float64)
        feasible = bool(info.get('feasible', True))
        target = next_target
        if terminated or truncated:
            break
    return total, feasible


# ----------------------------------------------------------------------
# Per-agent evaluation
# ----------------------------------------------------------------------
def _evaluate_agent(agent, scenarios, n_obj):
    """Re-evaluate one MORL agent against the eval scenarios. Drops any
    policy that violates the constraint in ANY scenario (the env's
    info['feasible'] flag turns False at episode end if any year of any
    step exceeded Pcrit). Short-circuits per policy: stops at the first
    violating scenario.

    Returns (targets_feas, means_feas, diag).
      targets_feas : (n_feas, n_obj) archive target vectors (MAX-conv)
      means_feas   : (n_feas, n_obj) mean realised reward across scenarios
                     (MAX-conv) — for feasible policies only
      diag         = {n_archive_orig, n_archive (post-subsample),
                      n_feasible, n_infeasible, mean_corr}
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
                 'n_feasible': 0, 'n_infeasible': 0,
                 'mean_corr': float('nan')})

    if (MAX_POLICIES_PER_AGENT is not None and
            n_archive_orig > MAX_POLICIES_PER_AGENT):
        kept = agent._subsample_nd(set(agent.archive),
                                   target_size=MAX_POLICIES_PER_AGENT)
        raw_archive = list(kept)
    archive = [np.asarray(v, dtype=float) for v in raw_archive]

    envs = [_build_env(s, n_obj) for s in scenarios]
    n_scen = len(envs)

    targets_feas, means_feas = [], []
    n_infeasible = 0
    for target_vec in archive:
        sums = np.zeros(n_obj, dtype=np.float64)
        feasible = True
        for env in envs:
            ret, feas = _rollout(
                cache, env, target_vec, n_obj, env_shape_arr, action_nvec)
            if not feas:
                feasible = False
                break             # short-circuit: any infeasible scenario → drop
            sums += ret
        if feasible:
            targets_feas.append(target_vec)
            means_feas.append(sums / n_scen)
        else:
            n_infeasible += 1

    if not targets_feas:
        return (np.empty((0, n_obj)), np.empty((0, n_obj)),
                {'n_archive': len(archive),
                 'n_archive_orig': n_archive_orig,
                 'n_feasible': 0,
                 'n_infeasible': n_infeasible,
                 'mean_corr': float('nan')})

    targets   = np.vstack([t.reshape(1, -1) for t in targets_feas])
    means_pos = np.vstack([m.reshape(1, -1) for m in means_feas])

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
        'n_feasible': len(targets_feas),
        'n_infeasible': n_infeasible,
        'mean_corr': mean_corr}


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
# HV machinery — IDENTICAL to constrained_lake_moea_reeval.py.
# Box loaded from params_config.lake_box_* (same constants as the
# unconstrained two-lake problem). This is valid because feasibility
# filtering above drops every policy with any penalty contribution —
# the surviving feasible policies' realised o* values fall in the same
# range as the unconstrained problem.
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
    union = np.vstack([c[4] for c in cells])
    best_known_hv = hv(union)
    meta.update(box_nadir=nadir.tolist(), box_ideal=ideal.tolist(),
                box_volume=box_volume, best_known_hv=best_known_hv,
                n_union_points=int(len(union)))
    return hv, box_volume, meta


# ----------------------------------------------------------------------
# Worker: re-evaluate one (config, seed) cell
# ----------------------------------------------------------------------
def _process_one_cell(task):
    sd_dir        = task['sd_dir']
    k             = task['seed']
    scoring       = task['scoring']
    scenm         = task['scenm']
    n_obj         = task['n_obj']
    out_csv_seed  = task['out_csv_seed']
    meta          = task['meta']
    scenarios     = _EVAL_SCENARIOS

    t0 = time.time()
    agent_files = []
    for fname in sorted(os.listdir(sd_dir)):
        am = AGENT_RE.match(fname)
        if not am:
            continue
        _nfe, ref_num = am.groups()
        ref_num = int(ref_num) if ref_num is not None else None
        agent_files.append((os.path.join(sd_dir, fname), ref_num))
    if not agent_files:
        return {**meta, 'seed': k, 'front_min': None,
                'rows': None, 'dt': time.time() - t0,
                'note': 'no agent files'}

    all_targets, all_means, all_ref = [], [], []
    diag_lines = []
    n_evaluated_cell  = 0   # sum of policies actually rolled out (post-subsample)
    n_infeasible_cell = 0   # sum of infeasible policies dropped
    for fpath, ref_num in agent_files:
        agent = _load_agent(fpath, n_obj=n_obj)
        if int(agent.num_objectives) != n_obj:
            raise ValueError(
                f'{os.path.basename(fpath)}: num_objectives '
                f'{agent.num_objectives} != folder n_obj {n_obj}')
        targets, means_pos, diag = _evaluate_agent(
            agent, scenarios, n_obj)

        n_evaluated_cell  += diag['n_archive']
        n_infeasible_cell += diag['n_infeasible']

        ref_tag = f' ref{ref_num}' if ref_num is not None else ''
        corr_str = ('nan' if np.isnan(diag['mean_corr'])
                    else f"{diag['mean_corr']:.2f}")
        if diag['n_archive_orig'] != diag['n_archive']:
            n_str = f"{diag['n_archive_orig']}→{diag['n_archive']} pols"
        else:
            n_str = f"{diag['n_archive']} pols"
        infeas_str = (f", infeas={diag['n_infeasible']}"
                      if diag['n_infeasible'] else '')
        diag_lines.append(f"{ref_tag} {n_str}, corr={corr_str}{infeas_str}")

        if len(targets) > 0:
            all_targets.append(targets)
            all_means.append(means_pos)
            ref_id = ref_num if ref_num is not None else 0
            all_ref.append(np.full(len(targets), ref_id, dtype=int))

    # If every policy was infeasible, write an empty CSV + sidecar so
    # checkpointing still treats this cell as done. Front is empty
    # (HV = 0 on the union).
    if not all_targets:
        os.makedirs(os.path.dirname(out_csv_seed), exist_ok=True)
        cols = ({'policy_id': [], 'agent_ref': []} |
                {f'target_o{j+1}': [] for j in range(n_obj)} |
                {f're_o{j+1}': [] for j in range(n_obj)})
        pd.DataFrame(cols).to_csv(out_csv_seed, index=False)
        side = out_csv_seed.replace('.csv', '_meta.json')
        with open(side, 'w') as f:
            json.dump({'n_evaluated': n_evaluated_cell,
                       'n_infeasible': n_infeasible_cell}, f)
        return {**meta, 'seed': k,
                'front_min': np.empty((0, n_obj)),
                'rows': 0,
                'n_agents': len(agent_files),
                'n_total_pol': 0,
                'n_dom': 0,
                'n_evaluated': n_evaluated_cell,
                'n_infeasible': n_infeasible_cell,
                'dt': time.time() - t0,
                'diag': '; '.join(diag_lines),
                'note': 'computed (all infeasible)'}

    targets   = np.vstack(all_targets)
    means_pos = np.vstack(all_means)
    agent_ref = np.concatenate(all_ref)
    means_min = -means_pos

    keep = _nd_filter_min(means_min)
    targets_k   = targets[keep]
    means_min_k = means_min[keep]
    ref_k       = agent_ref[keep]

    out = {'policy_id': np.arange(int(keep.sum())),
           'agent_ref': ref_k}
    for j in range(n_obj):
        out[f'target_o{j+1}'] = targets_k[:, j]
    for j in range(n_obj):
        out[f're_o{j+1}'] = means_min_k[:, j]
    df_out = pd.DataFrame(out)

    os.makedirs(os.path.dirname(out_csv_seed), exist_ok=True)
    df_out.to_csv(out_csv_seed, index=False)

    # Sidecar with per-cell feasibility counts (the per-seed CSV holds
    # only feasible policies, so these counts would be lost otherwise).
    side = out_csv_seed.replace('.csv', '_meta.json')
    with open(side, 'w') as f:
        json.dump({'n_evaluated': n_evaluated_cell,
                   'n_infeasible': n_infeasible_cell}, f)

    return {**meta, 'seed': k,
            'front_min': means_min_k,
            'rows': len(df_out),
            'n_agents': len(agent_files),
            'n_total_pol': len(means_min),
            'n_dom': int((~keep).sum()),
            'n_evaluated': n_evaluated_cell,
            'n_infeasible': n_infeasible_cell,
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
    """yield (scoring, scenm, n_obj, stem_dir) — mirrors lake_morl_reeval.py."""
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
        if scenm not in allowed_scenms: continue
        if not run_scoring.get(scoring, 0): continue
        if not run_n_obj.get(n_obj, 0): continue
        yield scoring, scenm, n_obj, os.path.join(morl_base, d)


def _build_tasks(reeval_root):
    """Walk <setting>/morl/<stem>/seed*, build per-cell task dicts."""
    tasks = []
    prebuilt_cells = []
    skipped = 0

    for scoring, scenm, n_obj, stem_dir in _enabled_stems():
        stem_name = os.path.basename(stem_dir)
        for sd in sorted(os.listdir(stem_dir)):
            sd_dir = os.path.join(stem_dir, sd)
            if not os.path.isdir(sd_dir):
                continue
            k = _seed_idx(sd_dir)
            if k < 0:
                continue
            out_csv_seed = os.path.join(reeval_root, stem_name, f'seed{k}.csv')

            if os.path.exists(out_csv_seed):
                prev = pd.read_csv(out_csv_seed)
                re_cols = [c for c in prev.columns
                           if re.fullmatch(r're_o\d+', c)]
                if re_cols:
                    F_min = prev[re_cols].values
                    # Recover infeasible count from sidecar; absent → 0
                    # (legacy CSVs without sidecar are treated as
                    # "no infeasible info recorded").
                    side = out_csv_seed.replace('.csv', '_meta.json')
                    n_eval = 0
                    n_infeas = 0
                    if os.path.exists(side):
                        try:
                            with open(side) as f:
                                sd_meta = json.load(f)
                            n_eval   = int(sd_meta.get('n_evaluated', 0))
                            n_infeas = int(sd_meta.get('n_infeasible', 0))
                        except Exception:
                            pass
                    prebuilt_cells.append(
                        (scoring, scenm, n_obj, k, F_min, n_eval, n_infeas))
                    skipped += 1
                    continue

            tasks.append({
                'sd_dir':    sd_dir,
                'seed':      k,
                'scoring':   scoring,
                'scenm':     scenm,
                'n_obj':     n_obj,
                'out_csv_seed': out_csv_seed,
                'meta': {'scoring': scoring, 'scenm': scenm, 'n_obj': n_obj},
            })
    return tasks, prebuilt_cells, skipped


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def _log_line(i, n_tasks, res):
    label = f"{res['scoring']}_{res['scenm']}_{res['n_obj']}/seed{res['seed']}"
    if res.get('note', '').startswith('computed'):
        infeas_str = (f", {res['n_infeasible']} infeas"
                      if res.get('n_infeasible', 0) else '')
        print(f"   [{i:3d}/{n_tasks}] {label:42s} "
              f"{res['dt']:6.1f}s  "
              f"{res['n_agents']} agent(s), "
              f"{res['n_total_pol']}→{res['rows']} pols "
              f"({res['n_dom']} dom{infeas_str})  diag: {res['diag']}")
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

    print(f'\n=== Re-evaluating MORL agents (constrained_lake, {SETTING}) ===')
    print(f'   {n_total} cells total; {skipped} checkpointed, '
          f'{len(tasks)} to compute with {n_workers} worker(s)')

    cells = list(prebuilt_cells)
    t0_all = time.time()
    if tasks:
        if n_workers == 1:
            _worker_init(scenarios)
            for i, task in enumerate(tasks, 1):
                res = _process_one_cell(task)
                if res.get('front_min') is not None:
                    cells.append((res['scoring'], res['scenm'],
                                  res['n_obj'], res['seed'],
                                  res['front_min'],
                                  res.get('n_evaluated', 0),
                                  res.get('n_infeasible', 0)))
                _log_line(i, len(tasks), res)
        else:
            with ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_worker_init,
                    initargs=(scenarios,)) as pool:
                futures = {pool.submit(_process_one_cell, t): t
                           for t in tasks}
                for i, fut in enumerate(as_completed(futures), 1):
                    res = fut.result()
                    if res.get('front_min') is not None:
                        cells.append((res['scoring'], res['scenm'],
                                      res['n_obj'], res['seed'],
                                      res['front_min'],
                                      res.get('n_evaluated', 0),
                                      res.get('n_infeasible', 0)))
                    _log_line(i, len(tasks), res)

    print(f'\n   MORL re-evaluation done in {time.time() - t0_all:.1f}s')
    print(f'   re-evaluated archives under {reeval_root}/')

    if not cells:
        raise SystemExit('no MORL cells available — nothing to score')

    print('\n=== Computing HV on re-evaluated MORL fronts ===')
    panel_cells_max = [(sc, scen, n, k, -F_min, ne, ni)
                       for sc, scen, n, k, F_min, ne, ni in cells]
    by_nobj = defaultdict(list)
    for c in panel_cells_max:
        by_nobj[c[2]].append(c)

    rows = []
    meta = {'setting': SETTING,
            'kind': 'reevaluation_morl',
            'problem': 'constrained_lake',
            'n_eval_scenarios': len(scenarios),
            'aggregation': 'arithmetic_mean_over_scenarios',
            'multi_rule': 'pool_5_per_seed_target_tracked_then_nd_filter',
            'feasibility_filter': ('drop policies with any violation in any '
                                   'eval scenario (info["feasible"] == False)'),
            'hv_box_source': ('params_config.lake_box_* — same constants as '
                              'unconstrained two-lake (valid because feasible '
                              'policies have zero penalty contribution)'),
            'n_workers_used': n_workers,
            'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'panels': {}}

    for n_obj, pcells in sorted(by_nobj.items()):
        # _panel_machinery only needs the front (5th tuple entry); pass through.
        hv, box_volume, pmeta = _panel_machinery(pcells, n_obj)
        # Per-panel infeasibility totals
        panel_n_eval     = int(sum(c[5] for c in pcells))
        panel_n_infeas   = int(sum(c[6] for c in pcells))
        pmeta['n_evaluated_total']  = panel_n_eval
        pmeta['n_infeasible_total'] = panel_n_infeas
        meta['panels'][str(n_obj)]  = pmeta
        print(f'\n n_obj={n_obj}: {len(pcells)} re-evaluated cells, '
              f"{pmeta['estimator']} HV, box_vol={box_volume:.5g}, "
              f"best-known HV={pmeta['best_known_hv']:.5g} "
              f"(={pmeta['best_known_hv'] / box_volume:.4f} of box)")
        if panel_n_eval:
            print(f'   infeasible: {panel_n_infeas}/{panel_n_eval} '
                  f'({panel_n_infeas / panel_n_eval:.1%}) across cells')
        for scoring, scenm, no, k, F, n_eval, n_infeas in pcells:
            h = hv(F)
            cond = f'closed_loop_{scenm}'
            rows.append(dict(
                paradigm='MORL', method=scoring, condition=cond,
                scoring=scoring, scenario_method=scenm,
                n_obj=no, seed=k, n_solutions=int(len(F)),
                n_evaluated=int(n_eval), n_infeasible=int(n_infeas),
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
