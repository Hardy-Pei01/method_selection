import os, re, gzip, pickle, glob, json, time
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constrained_two_lake import ConstrainedTwoLakeEnv
from morl.pql import PQL

from utils import (
    MC_SAMPLES_6OBJ, MC_SEED, _hv_exact, _hv_mc,
    _nd_filter_min, _seed_idx, _spacing_norm, MAX_POLICIES_PER_AGENT
)

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
INPUT_ROOT = '../data/constrained_data_1/'
OUTPUT_ROOT = '../data/constrained_data_1'

# 'deterministic' or 'robust'
SETTING = 'robust'

EVAL_SCENARIOS_PATH = '../lakes/lake_scenarios_eval.npy'  # (1000,) struct

run_scoring = {'pareto': 1, 'indicator': 1, 'decomposition': 1}
run_scenario_method_for_setting = {
    'deterministic': ['single'],
    'robust': ['multi', 'moro'],
}
run_n_obj = {2: 1, 6: 1}

# Parallelism: 0 = auto (os.cpu_count()), 1 = serial (debug), N>0 = N workers
N_WORKERS = 0

# Stem-folder pattern (mirrors lake_morl_reeval.py): seeds are
# subdirectories `seed*/` inside each stem.
FOLDER_RE = re.compile(
    r'^(pareto|indicator|decomposition)_(single|multi|moro)_(\d+)$')
AGENT_RE = re.compile(r'^agent_(\d+)(?:_(\d+))?\.pkl$')

# Tolerant mode: a policy passes if it is feasible (info['feasible']==True
# at episode end) in at least this fraction of eval scenarios. Strict mode
# requires 100% feasibility. The 80% threshold mirrors the 20th-percentile
# robustness criterion used during training: a policy that violates in up
# to 20% of scenarios still has its 20th-percentile fitness untouched.
# For deterministic (single scenario) the two modes coincide trivially.
FEASIBILITY_THRESHOLD = 0.80

# ----------------------------------------------------------------------
# Worker-global state (set by ProcessPoolExecutor initializer)
# ----------------------------------------------------------------------
_EVAL_SCENARIOS = None


def _worker_init(scenarios):
    global _EVAL_SCENARIOS
    _EVAL_SCENARIOS = scenarios


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
# Agent loading
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
# Per-agent evaluation (dual-mode)
# ----------------------------------------------------------------------
def _evaluate_agent(agent, scenarios, n_obj):
    """Re-evaluate one MORL agent against the eval scenarios. Rolls
    every archive policy across every scenario (no short-circuit), then
    derives both strict and tolerant per-mode results.

    Returns dict:
      targets    : (n_pol, n_obj) — archive target vectors (post-subsample)
      means_uncond : (n_pol, n_obj) — unconditional mean across all scenarios
                     (used for mean_corr; also used for strict survivors)
      means_cond : (n_pol, n_obj) — conditional mean over feasible scenarios
                   only (NaN for policies with 0 feasible scenarios)
      feas_rates : (n_pol,) — per-policy feasibility rate ∈ [0, 1]
      diag       : aggregate counts + mean_corr
    """
    decomp = (agent.action_eval == 'decomposition')
    cache = _build_q_cache(agent, decomp)
    env_shape = tuple(int(x) for x in agent.env_shape)
    env_shape_arr = np.array(env_shape)
    action_nvec = tuple(int(x) for x in agent.env.action_space.nvec)

    raw_archive = list(agent.archive)
    n_archive_orig = len(raw_archive)
    if not raw_archive:
        return {
            'targets': np.empty((0, n_obj)),
            'means_uncond': np.empty((0, n_obj)),
            'means_cond': np.empty((0, n_obj)),
            'feas_rates': np.empty((0,)),
            'diag': {'n_archive': 0, 'n_archive_orig': 0,
                     'n_infeasible_strict': 0,
                     'n_infeasible_tolerant': 0,
                     'mean_corr': float('nan')},
        }

    if (MAX_POLICIES_PER_AGENT is not None and
            n_archive_orig > MAX_POLICIES_PER_AGENT):
        kept = agent._subsample_nd(set(agent.archive),
                                   target_size=MAX_POLICIES_PER_AGENT)
        raw_archive = list(kept)
    archive = [np.asarray(v, dtype=float) for v in raw_archive]

    envs = [_build_env(s, n_obj) for s in scenarios]
    n_scen = len(envs)
    n_pol = len(archive)

    targets_arr     = np.vstack([t.reshape(1, -1) for t in archive])
    means_uncond    = np.empty((n_pol, n_obj), dtype=np.float64)
    means_cond      = np.full((n_pol, n_obj), np.nan, dtype=np.float64)
    feas_rates      = np.zeros(n_pol, dtype=np.float64)
    n_inf_strict    = 0
    n_inf_tolerant  = 0

    for i, target_vec in enumerate(archive):
        returns = np.empty((n_scen, n_obj), dtype=np.float64)
        feas_flags = np.zeros(n_scen, dtype=bool)
        for s_idx, env in enumerate(envs):
            ret, feas = _rollout(
                cache, env, target_vec, n_obj, env_shape_arr, action_nvec)
            returns[s_idx] = ret
            feas_flags[s_idx] = feas
        feas_rate = float(feas_flags.mean())
        feas_rates[i] = feas_rate

        # Unconditional mean — always defined (used for mean_corr and
        # for strict-mode survivors).
        means_uncond[i] = returns.mean(axis=0)

        # Conditional mean — defined only when at least one feasible
        # scenario exists.
        if feas_flags.any():
            means_cond[i] = returns[feas_flags].mean(axis=0)

        if feas_rate < 1.0 - 1e-12:
            n_inf_strict += 1
        if feas_rate < FEASIBILITY_THRESHOLD - 1e-12:
            n_inf_tolerant += 1

    # Per-agent mean_corr: correlate target vectors with unconditional
    # realised returns. Mode-independent — characterises how well PQL's
    # Q-targets predict realised performance, regardless of feasibility.
    corrs = []
    for j in range(n_obj):
        if (np.std(targets_arr[:, j]) > 1e-9 and
                np.std(means_uncond[:, j]) > 1e-9):
            corrs.append(
                np.corrcoef(targets_arr[:, j], means_uncond[:, j])[0, 1])
    mean_corr = float(np.nanmean(corrs)) if corrs else float('nan')

    return {
        'targets': targets_arr,
        'means_uncond': means_uncond,
        'means_cond': means_cond,
        'feas_rates': feas_rates,
        'diag': {
            'n_archive': n_pol,
            'n_archive_orig': n_archive_orig,
            'n_infeasible_strict': n_inf_strict,
            'n_infeasible_tolerant': n_inf_tolerant,
            'mean_corr': mean_corr,
        },
    }


# ----------------------------------------------------------------------
# HV machinery — mode-independent setup + per-mode best-known.
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
        raise ValueError(
            f'Degenerate box for constrained lake dim={n_obj}: '
            f'nadir={nadir.tolist()} ideal={ideal.tolist()}')
    return nadir, ideal


def _panel_setup(n_obj):
    nadir, ideal = _fixed_box(n_obj)
    box_volume = float(np.prod(ideal - nadir))
    if n_obj == 2:
        ref_min = -nadir
        hv = lambda F: _hv_exact(np.clip(F, nadir, ideal), ref_min)
        pmeta = dict(estimator='exact')
    else:
        rng = np.random.default_rng(MC_SEED)
        samples = rng.uniform(nadir, ideal, size=(MC_SAMPLES_6OBJ, n_obj))
        hv = lambda F: _hv_mc(np.clip(F, nadir, ideal), nadir, ideal, samples)
        pmeta = dict(estimator='monte_carlo',
                     mc_samples=MC_SAMPLES_6OBJ, mc_seed=MC_SEED)
    pmeta.update(box_nadir=nadir.tolist(), box_ideal=ideal.tolist(),
                 box_volume=box_volume)
    return hv, box_volume, nadir, ideal, pmeta


def _best_known(panel_cells, hv, mode):
    key = f'F_{mode}'
    fronts = [c[key] for c in panel_cells if len(c[key]) > 0]
    if not fronts:
        return 0.0, 0
    union = np.vstack(fronts)
    return hv(union), int(len(union))


# ----------------------------------------------------------------------
# Worker: re-evaluate one (config, seed) cell
# ----------------------------------------------------------------------
def _process_one_cell(task):
    sd_dir = task['sd_dir']
    k = task['seed']
    scoring = task['scoring']
    scenm = task['scenm']
    n_obj = task['n_obj']
    out_csv_strict = task['out_csv_strict']
    out_csv_tolerant = task['out_csv_tolerant']
    out_meta = task['out_meta']
    meta = task['meta']
    scenarios = _EVAL_SCENARIOS

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
        return {**meta, 'seed': k, 'F_strict': None, 'F_tolerant': None,
                'dt': time.time() - t0, 'note': 'no agent files'}

    # Per-cell accumulators (across agents).
    rows_strict = {'policy_id': [], 'agent_ref': [],
                   **{f'target_o{j + 1}': [] for j in range(n_obj)},
                   **{f're_o{j + 1}': [] for j in range(n_obj)}}
    rows_tolerant = {'policy_id': [], 'agent_ref': [],
                     **{f'target_o{j + 1}': [] for j in range(n_obj)},
                     **{f're_o{j + 1}': [] for j in range(n_obj)}}
    # Pre-ND counts per mode (number of survivors before ND-filter).
    n_total_strict_cell = 0
    n_total_tolerant_cell = 0
    n_evaluated_cell = 0           # post-subsample archive size, summed
    n_inf_strict_cell = 0
    n_inf_tolerant_cell = 0
    diag_lines = []
    corr_values = []

    # Per-agent loop: collect (target, mean) pairs into mode-specific
    # accumulators. We keep arrays per agent so we can stack and ND-filter
    # at the cell level.
    agg_strict_targets, agg_strict_means_min, agg_strict_refs = [], [], []
    agg_tolerant_targets, agg_tolerant_means_min, agg_tolerant_refs = [], [], []

    for fpath, ref_num in agent_files:
        agent = _load_agent(fpath, n_obj=n_obj)
        if int(agent.num_objectives) != n_obj:
            raise ValueError(
                f'{os.path.basename(fpath)}: num_objectives '
                f'{agent.num_objectives} != folder n_obj {n_obj}')
        ev = _evaluate_agent(agent, scenarios, n_obj)
        diag = ev['diag']

        n_evaluated_cell += diag['n_archive']
        n_inf_strict_cell += diag['n_infeasible_strict']
        n_inf_tolerant_cell += diag['n_infeasible_tolerant']
        if not np.isnan(diag['mean_corr']):
            corr_values.append(float(diag['mean_corr']))

        # Diag log line for this agent
        ref_tag = f' ref{ref_num}' if ref_num is not None else ''
        corr_str = ('nan' if np.isnan(diag['mean_corr'])
                    else f"{diag['mean_corr']:.2f}")
        if diag['n_archive_orig'] != diag['n_archive']:
            n_str = f"{diag['n_archive_orig']}→{diag['n_archive']} pols"
        else:
            n_str = f"{diag['n_archive']} pols"
        infeas_str = (f", infS={diag['n_infeasible_strict']}/"
                      f"infT={diag['n_infeasible_tolerant']}"
                      if (diag['n_infeasible_strict'] or
                          diag['n_infeasible_tolerant']) else '')
        diag_lines.append(f"{ref_tag} {n_str}, corr={corr_str}{infeas_str}")

        feas_rates = ev['feas_rates']
        targets = ev['targets']
        means_uncond = ev['means_uncond']
        means_cond = ev['means_cond']
        ref_id = ref_num if ref_num is not None else 0

        # Strict survivors
        mask_s = feas_rates >= 1.0 - 1e-12
        if mask_s.any():
            agg_strict_targets.append(targets[mask_s])
            agg_strict_means_min.append(-means_uncond[mask_s])  # MAX → MIN
            agg_strict_refs.append(
                np.full(int(mask_s.sum()), ref_id, dtype=int))

        # Tolerant survivors
        mask_t = feas_rates >= FEASIBILITY_THRESHOLD - 1e-12
        if mask_t.any():
            agg_tolerant_targets.append(targets[mask_t])
            agg_tolerant_means_min.append(-means_cond[mask_t])  # MAX → MIN
            agg_tolerant_refs.append(
                np.full(int(mask_t.sum()), ref_id, dtype=int))

    cell_mean_corr = (float(np.mean(corr_values))
                      if corr_values else float('nan'))

    def _finalize_mode(agg_targets, agg_means_min, agg_refs, rows_out):
        """Stack per-agent arrays, ND-filter, build CSV rows. Returns
        (front_min_filtered, n_pre_nd, n_dom)."""
        if not agg_targets:
            return np.empty((0, n_obj)), 0, 0
        T = np.vstack(agg_targets)
        M_min = np.vstack(agg_means_min)
        R = np.concatenate(agg_refs)
        n_pre_nd = int(len(M_min))
        keep = _nd_filter_min(M_min)
        T_k = T[keep]
        M_k = M_min[keep]
        R_k = R[keep]
        rows_out['policy_id'] = list(range(int(keep.sum())))
        rows_out['agent_ref'] = list(R_k)
        for j in range(n_obj):
            rows_out[f'target_o{j + 1}'] = list(T_k[:, j])
            rows_out[f're_o{j + 1}'] = list(M_k[:, j])
        n_dom = n_pre_nd - int(keep.sum())
        return M_k, n_pre_nd, n_dom

    front_strict_min, n_total_strict_cell, n_dom_strict = _finalize_mode(
        agg_strict_targets, agg_strict_means_min, agg_strict_refs,
        rows_strict)
    front_tolerant_min, n_total_tolerant_cell, n_dom_tolerant = _finalize_mode(
        agg_tolerant_targets, agg_tolerant_means_min, agg_tolerant_refs,
        rows_tolerant)

    # Write two CSVs + one sidecar.
    os.makedirs(os.path.dirname(out_csv_strict), exist_ok=True)
    pd.DataFrame(rows_strict).to_csv(out_csv_strict, index=False)
    pd.DataFrame(rows_tolerant).to_csv(out_csv_tolerant, index=False)

    with open(out_meta, 'w') as f:
        json.dump({
            'n_agents': len(agent_files),
            'n_evaluated': int(n_evaluated_cell),
            'mean_corr': cell_mean_corr,
            'n_total_pol_strict':    int(n_total_strict_cell),
            'n_total_pol_tolerant':  int(n_total_tolerant_cell),
            'n_infeasible_strict':   int(n_inf_strict_cell),
            'n_infeasible_tolerant': int(n_inf_tolerant_cell),
            'n_dom_strict':          int(n_dom_strict),
            'n_dom_tolerant':        int(n_dom_tolerant),
            'feasibility_threshold': FEASIBILITY_THRESHOLD,
        }, f)

    # Return both fronts in MAX-conv to align with the rest of the script.
    F_strict = -front_strict_min if len(front_strict_min) else np.empty((0, n_obj))
    F_tolerant = -front_tolerant_min if len(front_tolerant_min) else np.empty((0, n_obj))

    return {**meta, 'seed': k,
            'F_strict': F_strict, 'F_tolerant': F_tolerant,
            'n_agents': len(agent_files),
            'n_evaluated': int(n_evaluated_cell),
            'mean_corr': cell_mean_corr,
            'n_total_pol_strict':    int(n_total_strict_cell),
            'n_total_pol_tolerant':  int(n_total_tolerant_cell),
            'n_infeasible_strict':   int(n_inf_strict_cell),
            'n_infeasible_tolerant': int(n_inf_tolerant_cell),
            'n_dom_strict':          int(n_dom_strict),
            'n_dom_tolerant':        int(n_dom_tolerant),
            'dt': time.time() - t0,
            'diag': '; '.join(diag_lines),
            'note': 'computed'}


# ----------------------------------------------------------------------
# Discovery + task planning
# ----------------------------------------------------------------------
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
        if not run_scoring.get(scoring, 0): continue
        if scenm not in allowed_scenms: continue
        if not run_n_obj.get(n_obj, 0): continue
        yield scoring, scenm, n_obj, os.path.join(morl_base, d)


def _load_sidecar(out_meta):
    if not os.path.exists(out_meta):
        return {}
    try:
        with open(out_meta) as f:
            return json.load(f)
    except Exception:
        return {}


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
            out_csv_strict = os.path.join(reeval_root, stem_name,
                                           f'seed{k}_strict.csv')
            out_csv_tolerant = os.path.join(reeval_root, stem_name,
                                             f'seed{k}_tolerant.csv')
            out_meta = os.path.join(reeval_root, stem_name,
                                     f'seed{k}_meta.json')

            if (os.path.exists(out_csv_strict)
                    and os.path.exists(out_csv_tolerant)):
                prev_s = pd.read_csv(out_csv_strict)
                prev_t = pd.read_csv(out_csv_tolerant)
                re_s = [c for c in prev_s.columns if re.fullmatch(r're_o\d+', c)]
                re_t = [c for c in prev_t.columns if re.fullmatch(r're_o\d+', c)]
                if re_s and re_t:
                    F_strict = -prev_s[re_s].values
                    F_tolerant = -prev_t[re_t].values
                    sd_meta = _load_sidecar(out_meta)
                    cell = {
                        'scoring': scoring, 'scenm': scenm,
                        'n_obj': n_obj, 'seed': k,
                        'F_strict': F_strict, 'F_tolerant': F_tolerant,
                        'n_agents':              int(sd_meta.get('n_agents', -1)),
                        'n_evaluated':           int(sd_meta.get('n_evaluated', 0)),
                        'mean_corr':             float(sd_meta.get('mean_corr', float('nan'))),
                        'n_total_pol_strict':    int(sd_meta.get('n_total_pol_strict', -1)),
                        'n_total_pol_tolerant':  int(sd_meta.get('n_total_pol_tolerant', -1)),
                        'n_infeasible_strict':   int(sd_meta.get('n_infeasible_strict', -1)),
                        'n_infeasible_tolerant': int(sd_meta.get('n_infeasible_tolerant', -1)),
                        'n_dom_strict':          int(sd_meta.get('n_dom_strict', -1)),
                        'n_dom_tolerant':        int(sd_meta.get('n_dom_tolerant', -1)),
                    }
                    prebuilt_cells.append(cell)
                    skipped += 1
                    continue

            tasks.append({
                'sd_dir': sd_dir,
                'seed': k,
                'scoring': scoring,
                'scenm': scenm,
                'n_obj': n_obj,
                'out_csv_strict': out_csv_strict,
                'out_csv_tolerant': out_csv_tolerant,
                'out_meta': out_meta,
                'meta': {'scoring': scoring, 'scenm': scenm, 'n_obj': n_obj},
            })
    return tasks, prebuilt_cells, skipped


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def _log_line(i, n_tasks, res):
    label = f"{res['scoring']}_{res['scenm']}_{res['n_obj']}/seed{res['seed']}"
    if res.get('note', '').startswith('computed'):
        print(f"   [{i:3d}/{n_tasks}] {label:42s} "
              f"{res['dt']:6.1f}s  {res['n_agents']} agent(s); "
              f"strict {res['n_total_pol_strict']}"
              f"→{len(res['F_strict'])} "
              f"(infeas={res['n_infeasible_strict']}, "
              f"dom={res['n_dom_strict']});  "
              f"tolerant {res['n_total_pol_tolerant']}"
              f"→{len(res['F_tolerant'])} "
              f"(infeas={res['n_infeasible_tolerant']}, "
              f"dom={res['n_dom_tolerant']})  diag: {res['diag']}")
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

    print(f'FEASIBILITY_THRESHOLD={FEASIBILITY_THRESHOLD} '
          f'(deterministic: strict==tolerant by construction since n_scen=1)')

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

    def _record(res):
        if res.get('F_strict') is None:
            return
        cells.append({
            'scoring': res['scoring'], 'scenm': res['scenm'],
            'n_obj': res['n_obj'], 'seed': res['seed'],
            'F_strict': res['F_strict'], 'F_tolerant': res['F_tolerant'],
            'n_agents':              res['n_agents'],
            'n_evaluated':           res['n_evaluated'],
            'mean_corr':             res['mean_corr'],
            'n_total_pol_strict':    res['n_total_pol_strict'],
            'n_total_pol_tolerant':  res['n_total_pol_tolerant'],
            'n_infeasible_strict':   res['n_infeasible_strict'],
            'n_infeasible_tolerant': res['n_infeasible_tolerant'],
            'n_dom_strict':          res['n_dom_strict'],
            'n_dom_tolerant':        res['n_dom_tolerant'],
        })

    t0_all = time.time()
    if tasks:
        if n_workers == 1:
            _worker_init(scenarios)
            for i, task in enumerate(tasks, 1):
                res = _process_one_cell(task)
                _record(res)
                _log_line(i, len(tasks), res)
        else:
            with ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_worker_init,
                    initargs=(scenarios,)) as pool:
                futures = {pool.submit(_process_one_cell, t): t for t in tasks}
                for i, fut in enumerate(as_completed(futures), 1):
                    res = fut.result()
                    _record(res)
                    _log_line(i, len(tasks), res)

    print(f'\n   MORL re-evaluation done in {time.time() - t0_all:.1f}s')
    print(f'   re-evaluated archives under {reeval_root}/')

    if not cells:
        raise SystemExit('no MORL cells available — nothing to score')

    print('\n=== Computing HV on re-evaluated MORL fronts ===')
    by_nobj = defaultdict(list)
    for c in cells:
        by_nobj[c['n_obj']].append(c)

    rows = []
    meta = {'setting': SETTING,
            'kind': 'reevaluation_morl',
            'problem': 'constrained_lake',
            'n_eval_scenarios': len(scenarios),
            'aggregation': ('strict: mean over all scenarios; '
                            'tolerant: mean over feasible scenarios only'),
            'feasibility_threshold': FEASIBILITY_THRESHOLD,
            'feasibility_modes': {
                'strict': ("policy kept iff feasibility rate == 1.0; "
                           "mean over ALL scenarios"),
                'tolerant': (f"policy kept iff feasibility rate >= "
                             f"{FEASIBILITY_THRESHOLD}; mean over feasible "
                             f"scenarios only (avoids double-punishment via "
                             f"the env's violation penalty)"),
            },
            'multi_rule': 'pool_per_agent_targets_then_nd_filter_per_mode',
            'n_workers_used': n_workers,
            'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'panels': {}}

    for n_obj, panel_cells in sorted(by_nobj.items()):
        hv, box_volume, nadir, ideal, pmeta = _panel_setup(n_obj)
        bk_s, nu_s = _best_known(panel_cells, hv, 'strict')
        bk_t, nu_t = _best_known(panel_cells, hv, 'tolerant')
        pmeta['best_known_hv_strict']   = bk_s
        pmeta['best_known_hv_tolerant'] = bk_t
        pmeta['n_union_strict']         = nu_s
        pmeta['n_union_tolerant']       = nu_t
        panel_n_eval = int(sum(c['n_evaluated'] for c in panel_cells))
        panel_n_inf_s = int(sum(c['n_infeasible_strict']
                                for c in panel_cells if c['n_infeasible_strict'] >= 0))
        panel_n_inf_t = int(sum(c['n_infeasible_tolerant']
                                for c in panel_cells if c['n_infeasible_tolerant'] >= 0))
        pmeta['n_evaluated_total']           = panel_n_eval
        pmeta['n_infeasible_strict_total']   = panel_n_inf_s
        pmeta['n_infeasible_tolerant_total'] = panel_n_inf_t
        meta['panels'][str(n_obj)] = pmeta

        print(f'\n n_obj={n_obj}: {len(panel_cells)} re-evaluated cells, '
              f"{pmeta['estimator']} HV, box_vol={box_volume:.5g}")
        print(f"   best-known HV: strict={bk_s:.5g} ({nu_s} pts), "
              f"tolerant={bk_t:.5g} ({nu_t} pts)")
        if panel_n_eval:
            print(f'   infeasible across cells: '
                  f'strict {panel_n_inf_s}/{panel_n_eval} '
                  f'({panel_n_inf_s / panel_n_eval:.1%}); '
                  f'tolerant {panel_n_inf_t}/{panel_n_eval} '
                  f'({panel_n_inf_t / panel_n_eval:.1%})')

        for c in panel_cells:
            F_s = c['F_strict']
            F_t = c['F_tolerant']
            h_s = hv(F_s) if len(F_s) else 0.0
            h_t = hv(F_t) if len(F_t) else 0.0
            s_s = _spacing_norm(F_s, nadir, ideal)
            s_t = _spacing_norm(F_t, nadir, ideal)
            cond = f"closed_loop_{c['scenm']}"
            rows.append(dict(
                paradigm='MORL', method=c['scoring'], condition=cond,
                scoring=c['scoring'], scenario_method=c['scenm'],
                n_obj=c['n_obj'], seed=c['seed'],
                n_agents=int(c['n_agents']) if c['n_agents'] >= 0 else np.nan,
                n_evaluated=int(c['n_evaluated']),
                mean_corr=float(c['mean_corr']),
                # Strict mode columns
                n_total_pol_strict=(int(c['n_total_pol_strict'])
                                    if c['n_total_pol_strict'] >= 0 else np.nan),
                n_infeasible_strict=(int(c['n_infeasible_strict'])
                                     if c['n_infeasible_strict'] >= 0 else np.nan),
                n_dom_strict=(int(c['n_dom_strict'])
                              if c['n_dom_strict'] >= 0 else np.nan),
                n_solutions_strict=int(len(F_s)),
                hv_strict=h_s,
                hv_ratio_strict=(h_s / box_volume if box_volume > 0 else np.nan),
                spacing_norm_strict=s_s,
                # Tolerant mode columns
                n_total_pol_tolerant=(int(c['n_total_pol_tolerant'])
                                      if c['n_total_pol_tolerant'] >= 0 else np.nan),
                n_infeasible_tolerant=(int(c['n_infeasible_tolerant'])
                                       if c['n_infeasible_tolerant'] >= 0 else np.nan),
                n_dom_tolerant=(int(c['n_dom_tolerant'])
                                if c['n_dom_tolerant'] >= 0 else np.nan),
                n_solutions_tolerant=int(len(F_t)),
                hv_tolerant=h_t,
                hv_ratio_tolerant=(h_t / box_volume if box_volume > 0 else np.nan),
                spacing_norm_tolerant=s_t,
                box_volume=box_volume,
            ))

        agg_s = defaultdict(list)
        agg_t = defaultdict(list)
        for r in rows:
            if r['n_obj'] == n_obj:
                k = (r['method'], r['condition'])
                agg_s[k].append(r['hv_ratio_strict'])
                agg_t[k].append(r['hv_ratio_tolerant'])
        for k in sorted(agg_s):
            vs, vt = agg_s[k], agg_t[k]
            print(f'   {k[0]:14s} {k[1]:32s} n={len(vs):3d} '
                  f'HVr_strict={np.mean(vs):.4f}±{np.std(vs):.4f}  '
                  f'HVr_tolerant={np.mean(vt):.4f}±{np.std(vt):.4f}')

    df = pd.DataFrame(rows).sort_values(
        ['n_obj', 'paradigm', 'method', 'condition', 'seed'])
    out_csv = os.path.join(out_root, 'metrics_long_reeval.csv')
    out_json = os.path.join(out_root, '_meta_reeval.json')
    df.to_csv(out_csv, index=False)
    with open(out_json, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'\n  wrote {out_csv}  ({len(df)} rows)')
    print(f'  wrote {out_json}')
