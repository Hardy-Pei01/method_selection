from __future__ import annotations

import warnings
warnings.filterwarnings('ignore', message=r'pkg_resources is deprecated')

import os
import re
import glob
import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:                    # so `import _envs` works
    sys.path.insert(0, _HERE)
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:            # so `import fruit_tree`, etc. work
    sys.path.insert(0, _PROJECT_ROOT)

from utils import (
    MC_SAMPLES_6OBJ, MC_SEED, _hv_exact, _hv_mc,
    _nd_filter_min, _seed_idx, _spacing_norm, MAX_POLICIES_PER_AGENT,
)
import envs as _ENV_MOD


# ======================================================================
# Folder discovery — MORL folder regex is the same across all envs.
# (Tree-MORL is hard-coded observable=True; no obs token in path.)
# ======================================================================
FOLDER_RE = re.compile(
    r'^(pareto|indicator|decomposition)_(single|multi|moro)_(\d+)$')
AGENT_RE = re.compile(r'^agent_(\d+)(?:_(\d+))?\.pkl$')

_SCENMS_FOR = {
    'deterministic': {'single'},
    'robust': {'multi', 'moro'},
}


# ======================================================================
# Worker globals (set by initializer; broadcast once per worker).
# ======================================================================
_W_SCENARIOS = None
_W_ENV_NAME = None
_W_FEASIBILITY_THRESHOLD = None


def _worker_init(env_name, scenarios, feasibility_threshold):
    global _W_SCENARIOS, _W_ENV_NAME, _W_FEASIBILITY_THRESHOLD
    _W_SCENARIOS = scenarios
    _W_ENV_NAME = env_name
    _W_FEASIBILITY_THRESHOLD = feasibility_threshold


def _w_adapter():
    return _ENV_MOD.get(_W_ENV_NAME)


# ======================================================================
# Q-cache builder — env-independent (operates on PQL agent internals).
# ======================================================================
def _build_q_cache(agent, decomp):
    """Per-(state, action) cached (Q_raw, Qsa) arrays. Built once per
    agent; reused across every (target, scenario) rollout."""
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


# ======================================================================
# Generic per-agent evaluation. Dispatches to env-specific rollout
# / scenario-installation pattern via the adapter.
# ======================================================================
def _evaluate_agent(adapter, agent, scenarios, n_obj, feasibility_threshold):
    """Return dict with:
      targets         : (n_pol, n_obj) archive target vectors (MAX-conv)
      means_uncond_pos: (n_pol, n_obj) mean realised reward over scenarios
      means_cond_pos  : (n_pol, n_obj) — constrained only; mean over
                        feasible scenarios (NaN row if no feasible)
      feas_rates      : (n_pol,)      — constrained only
      diag            : dict with n_archive, n_archive_orig, mean_corr;
                        plus n_unique_seq (tree only);
                        plus n_infeasible_{strict,tolerant} (constrained only)

    Per-env env policy:
      * tree:           one env, swap _slip_pattern per scenario
      * lake/constrained: build one env per scenario, reuse across targets
    """
    decomp = (agent.action_eval == 'decomposition')
    cache = _build_q_cache(agent, decomp)
    action_meta = adapter.morl_action_meta(agent)

    raw_archive = list(agent.archive)
    n_archive_orig = len(raw_archive)

    # Subsample archive BEFORE rollouts (lake/constrained only).
    if (adapter.morl_uses_subsample
            and MAX_POLICIES_PER_AGENT is not None
            and n_archive_orig > MAX_POLICIES_PER_AGENT):
        kept = agent._subsample_nd(
            set(agent.archive), target_size=MAX_POLICIES_PER_AGENT)
        raw_archive = list(kept)
    archive = [np.asarray(v, dtype=float) for v in raw_archive]
    n_pol = len(archive)

    if n_pol == 0:
        diag = {'n_archive': 0, 'n_archive_orig': n_archive_orig,
                'mean_corr': float('nan')}
        if adapter.morl_has_action_sequence_diag:
            diag['n_unique_seq'] = 0
        out = {'targets': np.empty((0, n_obj)),
               'means_uncond_pos': np.empty((0, n_obj)),
               'diag': diag}
        if adapter.has_feasibility:
            out['means_cond_pos'] = np.empty((0, n_obj))
            out['feas_rates'] = np.empty((0,))
            diag['n_infeasible_strict'] = 0
            diag['n_infeasible_tolerant'] = 0
        return out

    n_scen = adapter.n_scenarios(scenarios)
    targets = np.zeros((n_pol, n_obj))
    means_uncond = np.zeros((n_pol, n_obj))
    seqs = set() if adapter.morl_has_action_sequence_diag else None

    if adapter.has_feasibility:
        means_cond = np.full((n_pol, n_obj), np.nan, dtype=np.float64)
        feas_rates = np.zeros(n_pol, dtype=np.float64)
    else:
        means_cond = None
        feas_rates = None

    if adapter.name == 'tree':
        env = agent.env  # already constructed by morl_load_agent
        for p, target_vec in enumerate(archive):
            if seqs is not None:
                seqs.add(adapter.morl_action_sequence(
                    cache, env, target_vec, action_meta))
            scen_returns = np.zeros((n_scen, n_obj))
            for s_idx, scen in enumerate(adapter.iter_scenarios(scenarios)):
                adapter.install_scenario(env, scen)
                r, _ = adapter.morl_rollout(
                    cache, env, target_vec, n_obj, action_meta)
                scen_returns[s_idx] = r
            targets[p] = target_vec
            means_uncond[p] = scen_returns.mean(axis=0)
    else:
        envs = [adapter.build_env(s, n_obj)
                for s in adapter.iter_scenarios(scenarios)]
        for p, target_vec in enumerate(archive):
            scen_returns = np.empty((n_scen, n_obj), dtype=np.float64)
            feas_flags = np.zeros(n_scen, dtype=bool)
            for s_idx, env in enumerate(envs):
                r, feas = adapter.morl_rollout(
                    cache, env, target_vec, n_obj, action_meta)
                scen_returns[s_idx] = r
                feas_flags[s_idx] = feas
            targets[p] = target_vec
            means_uncond[p] = scen_returns.mean(axis=0)
            if adapter.has_feasibility:
                fr = float(feas_flags.mean())
                feas_rates[p] = fr
                if feas_flags.any():
                    means_cond[p] = scen_returns[feas_flags].mean(axis=0)

    # Mean correlation between targets and unconditional realised returns.
    corrs = []
    for j in range(n_obj):
        if (np.std(targets[:, j]) > 1e-9 and
                np.std(means_uncond[:, j]) > 1e-9):
            corrs.append(np.corrcoef(targets[:, j], means_uncond[:, j])[0, 1])
    mean_corr = float(np.nanmean(corrs)) if corrs else float('nan')

    diag = {'n_archive': n_pol, 'n_archive_orig': n_archive_orig,
            'mean_corr': mean_corr}
    if adapter.morl_has_action_sequence_diag:
        diag['n_unique_seq'] = len(seqs)

    result = {'targets': targets,
              'means_uncond_pos': means_uncond,
              'diag': diag}
    if adapter.has_feasibility:
        result['means_cond_pos'] = means_cond
        result['feas_rates'] = feas_rates
        diag['n_infeasible_strict'] = int(
            np.sum(feas_rates < 1.0 - 1e-12))
        diag['n_infeasible_tolerant'] = int(
            np.sum(feas_rates < feasibility_threshold - 1e-12))
    return result


# ======================================================================
# Worker: single-mode cell (tree / lake)
# ======================================================================
def _process_one_cell_morl_single(task):
    adapter = _w_adapter()
    sd_dir = task['sd_dir']
    k = task['seed']
    scoring = task['scoring']
    scenm = task['scenm']
    n_obj = task['n_obj']
    out_csv_seed = task['out_csv_seed']
    meta = task['meta']
    scenarios = _W_SCENARIOS

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
    corr_values = []
    total_unique_seq = 0
    for fpath, ref_num in agent_files:
        agent = adapter.morl_load_agent(fpath, n_obj=n_obj)
        if int(agent.num_objectives) != n_obj:
            raise ValueError(
                f'{os.path.basename(fpath)}: num_objectives '
                f'{agent.num_objectives} != folder n_obj {n_obj}')
        ev = _evaluate_agent(adapter, agent, scenarios, n_obj,
                             _W_FEASIBILITY_THRESHOLD)
        diag = ev['diag']

        if not np.isnan(diag['mean_corr']):
            corr_values.append(float(diag['mean_corr']))
        if 'n_unique_seq' in diag:
            total_unique_seq += int(diag['n_unique_seq'])

        ref_tag = f' ref{ref_num}' if ref_num is not None else ''
        corr_str = ('nan' if np.isnan(diag['mean_corr'])
                    else f"{diag['mean_corr']:.2f}")
        if 'n_unique_seq' in diag:
            # tree format: "n_uniq/n_archive unique, corr=..."
            diag_lines.append(
                f"{ref_tag} {diag['n_unique_seq']}/{diag['n_archive']} "
                f"unique, corr={corr_str}")
        elif diag['n_archive_orig'] != diag['n_archive']:
            diag_lines.append(
                f"{ref_tag} {diag['n_archive_orig']}→{diag['n_archive']} "
                f"pols, corr={corr_str}")
        else:
            diag_lines.append(
                f"{ref_tag} {diag['n_archive']} pols, corr={corr_str}")

        all_targets.append(ev['targets'])
        all_means.append(ev['means_uncond_pos'])
        ref_id = ref_num if ref_num is not None else 0
        all_ref.append(np.full(len(ev['targets']), ref_id, dtype=int))

    targets = np.vstack(all_targets)
    means_pos = np.vstack(all_means)
    agent_ref = np.concatenate(all_ref)
    means_min = -means_pos  # MIN-conv for HV pipeline

    # ND-filter on realised returns across the merged set.
    keep = _nd_filter_min(means_min)
    targets_k = targets[keep]
    means_min_k = means_min[keep]
    ref_k = agent_ref[keep]

    out_dict = {'policy_id': np.arange(int(keep.sum())),
                'agent_ref': ref_k}
    for j in range(n_obj):
        out_dict[f'target_o{j + 1}'] = targets_k[:, j]
    for j in range(n_obj):
        out_dict[f're_o{j + 1}'] = means_min_k[:, j]
    df_out = pd.DataFrame(out_dict)
    os.makedirs(os.path.dirname(out_csv_seed), exist_ok=True)
    df_out.to_csv(out_csv_seed, index=False)

    cell_mean_corr = (float(np.mean(corr_values))
                      if corr_values else float('nan'))
    n_total_pol = int(len(means_min))
    n_dom = int((~keep).sum())

    side = out_csv_seed.replace('.csv', '_meta.json')
    sidecar = {'n_agents': len(agent_files),
               'n_total_pol': n_total_pol,
               'n_dom': n_dom}
    if adapter.morl_has_action_sequence_diag:
        sidecar['n_unique_seq'] = total_unique_seq
    sidecar['mean_corr'] = cell_mean_corr
    with open(side, 'w') as f:
        json.dump(sidecar, f)

    return {**meta, 'seed': k,
            'front_min': means_min_k,
            'rows': len(df_out),
            'n_agents': len(agent_files),
            'n_total_pol': n_total_pol,
            'n_dom': n_dom,
            'n_unique_seq': (total_unique_seq
                             if adapter.morl_has_action_sequence_diag
                             else None),
            'mean_corr': cell_mean_corr,
            'dt': time.time() - t0,
            'diag': '; '.join(diag_lines),
            'note': 'computed'}


# ======================================================================
# Worker: dual-mode cell (constrained_lake)
# ======================================================================
def _process_one_cell_morl_dual(task):
    adapter = _w_adapter()
    sd_dir = task['sd_dir']
    k = task['seed']
    scoring = task['scoring']
    scenm = task['scenm']
    n_obj = task['n_obj']
    out_csv_strict = task['out_csv_strict']
    out_csv_tolerant = task['out_csv_tolerant']
    out_meta = task['out_meta']
    meta = task['meta']
    scenarios = _W_SCENARIOS
    fthr = _W_FEASIBILITY_THRESHOLD

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

    agg_strict_targets, agg_strict_means_min, agg_strict_refs = [], [], []
    agg_tolerant_targets, agg_tolerant_means_min, agg_tolerant_refs = [], [], []
    n_evaluated_cell = 0
    n_inf_strict_cell = 0
    n_inf_tolerant_cell = 0
    corr_values = []
    diag_lines = []

    for fpath, ref_num in agent_files:
        agent = adapter.morl_load_agent(fpath, n_obj=n_obj)
        if int(agent.num_objectives) != n_obj:
            raise ValueError(
                f'{os.path.basename(fpath)}: num_objectives '
                f'{agent.num_objectives} != folder n_obj {n_obj}')
        ev = _evaluate_agent(adapter, agent, scenarios, n_obj, fthr)
        diag = ev['diag']
        n_evaluated_cell += diag['n_archive']
        n_inf_strict_cell += diag.get('n_infeasible_strict', 0)
        n_inf_tolerant_cell += diag.get('n_infeasible_tolerant', 0)
        if not np.isnan(diag['mean_corr']):
            corr_values.append(float(diag['mean_corr']))

        ref_tag = f' ref{ref_num}' if ref_num is not None else ''
        corr_str = ('nan' if np.isnan(diag['mean_corr'])
                    else f"{diag['mean_corr']:.2f}")
        if diag['n_archive_orig'] != diag['n_archive']:
            n_str = f"{diag['n_archive_orig']}→{diag['n_archive']} pols"
        else:
            n_str = f"{diag['n_archive']} pols"
        infeas_str = (f", infS={diag.get('n_infeasible_strict', 0)}/"
                      f"infT={diag.get('n_infeasible_tolerant', 0)}"
                      if (diag.get('n_infeasible_strict', 0) or
                          diag.get('n_infeasible_tolerant', 0)) else '')
        diag_lines.append(f"{ref_tag} {n_str}, corr={corr_str}{infeas_str}")

        feas_rates = ev['feas_rates']
        targets = ev['targets']
        means_uncond = ev['means_uncond_pos']
        means_cond = ev['means_cond_pos']
        ref_id = ref_num if ref_num is not None else 0

        mask_s = feas_rates >= 1.0 - 1e-12
        if mask_s.any():
            agg_strict_targets.append(targets[mask_s])
            agg_strict_means_min.append(-means_uncond[mask_s])
            agg_strict_refs.append(
                np.full(int(mask_s.sum()), ref_id, dtype=int))

        mask_t = feas_rates >= fthr - 1e-12
        if mask_t.any():
            agg_tolerant_targets.append(targets[mask_t])
            agg_tolerant_means_min.append(-means_cond[mask_t])
            agg_tolerant_refs.append(
                np.full(int(mask_t.sum()), ref_id, dtype=int))

    cell_mean_corr = (float(np.mean(corr_values))
                      if corr_values else float('nan'))

    def _finalize_mode(agg_T, agg_M, agg_R, out_csv):
        if not agg_T:
            empty = pd.DataFrame({'policy_id': [], 'agent_ref': [],
                                  **{f'target_o{j+1}': [] for j in range(n_obj)},
                                  **{f're_o{j+1}': [] for j in range(n_obj)}})
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            empty.to_csv(out_csv, index=False)
            return np.empty((0, n_obj)), 0, 0
        T = np.vstack(agg_T)
        M_min = np.vstack(agg_M)
        R = np.concatenate(agg_R)
        n_pre_nd = int(len(M_min))
        keep = _nd_filter_min(M_min)
        T_k = T[keep]
        M_k = M_min[keep]
        R_k = R[keep]
        rows = {'policy_id': list(range(int(keep.sum()))),
                'agent_ref': list(R_k)}
        for j in range(n_obj):
            rows[f'target_o{j + 1}'] = list(T_k[:, j])
        for j in range(n_obj):
            rows[f're_o{j + 1}'] = list(M_k[:, j])
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        return M_k, n_pre_nd, n_pre_nd - int(keep.sum())

    front_strict_min, n_total_strict, n_dom_strict = _finalize_mode(
        agg_strict_targets, agg_strict_means_min, agg_strict_refs,
        out_csv_strict)
    front_tolerant_min, n_total_tolerant, n_dom_tolerant = _finalize_mode(
        agg_tolerant_targets, agg_tolerant_means_min, agg_tolerant_refs,
        out_csv_tolerant)

    with open(out_meta, 'w') as f:
        json.dump({
            'n_agents': len(agent_files),
            'n_evaluated': int(n_evaluated_cell),
            'mean_corr': cell_mean_corr,
            'n_total_pol_strict': int(n_total_strict),
            'n_total_pol_tolerant': int(n_total_tolerant),
            'n_infeasible_strict': int(n_inf_strict_cell),
            'n_infeasible_tolerant': int(n_inf_tolerant_cell),
            'n_dom_strict': int(n_dom_strict),
            'n_dom_tolerant': int(n_dom_tolerant),
            'feasibility_threshold': fthr,
        }, f)

    F_strict = (-front_strict_min if len(front_strict_min)
                else np.empty((0, n_obj)))
    F_tolerant = (-front_tolerant_min if len(front_tolerant_min)
                  else np.empty((0, n_obj)))

    return {**meta, 'seed': k,
            'F_strict': F_strict, 'F_tolerant': F_tolerant,
            'n_agents': len(agent_files),
            'n_evaluated': int(n_evaluated_cell),
            'mean_corr': cell_mean_corr,
            'n_total_pol_strict': int(n_total_strict),
            'n_total_pol_tolerant': int(n_total_tolerant),
            'n_infeasible_strict': int(n_inf_strict_cell),
            'n_infeasible_tolerant': int(n_inf_tolerant_cell),
            'n_dom_strict': int(n_dom_strict),
            'n_dom_tolerant': int(n_dom_tolerant),
            'dt': time.time() - t0,
            'diag': '; '.join(diag_lines),
            'note': 'computed'}


# ======================================================================
# Discovery + task planning
# ======================================================================
def _enabled_stems(input_root, setting, run_scoring, run_n_obj):
    morl_base = os.path.join(input_root, setting, 'morl')
    if not os.path.isdir(morl_base):
        return
    allowed = _SCENMS_FOR[setting]
    for d in sorted(os.listdir(morl_base)):
        m = FOLDER_RE.match(d)
        if not m:
            continue
        scoring, scenm, n_obj_s = m.groups()
        n_obj = int(n_obj_s)
        if scenm not in allowed:
            continue
        if not run_scoring.get(scoring, 0):
            continue
        if not run_n_obj.get(n_obj, 0):
            continue
        yield (scoring, scenm, n_obj, os.path.join(morl_base, d))


def _build_tasks_single(input_root, setting, reeval_root,
                        run_scoring, run_n_obj, adapter):
    tasks = []
    cells = []
    skipped = 0
    for scoring, scenm, n_obj, stem_dir in _enabled_stems(
            input_root, setting, run_scoring, run_n_obj):
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
                rec = [c for c in prev.columns
                       if re.fullmatch(r're_o\d+', c)]
                if rec:
                    F_min = prev[rec].values
                    side = out_csv_seed.replace('.csv', '_meta.json')
                    sd_meta = {}
                    if os.path.exists(side):
                        try:
                            with open(side) as sf:
                                sd_meta = json.load(sf)
                        except Exception:
                            sd_meta = {}
                    cell = {
                        'scoring': scoring, 'scenm': scenm,
                        'n_obj': n_obj, 'seed': k, 'F_min': F_min,
                        'n_total_pol': int(sd_meta.get('n_total_pol', -1)),
                        'n_dom': int(sd_meta.get('n_dom', -1)),
                        'mean_corr': float(sd_meta.get('mean_corr',
                                                       float('nan'))),
                        'n_unique_seq': int(sd_meta.get('n_unique_seq', -1))
                        if adapter.morl_has_action_sequence_diag else None,
                    }
                    cells.append(cell)
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
    return tasks, cells, skipped


def _build_tasks_dual(input_root, setting, reeval_root,
                      run_scoring, run_n_obj, adapter):
    tasks = []
    cells = []
    skipped = 0
    for scoring, scenm, n_obj, stem_dir in _enabled_stems(
            input_root, setting, run_scoring, run_n_obj):
        stem_name = os.path.basename(stem_dir)
        cfg_out_dir = os.path.join(reeval_root, stem_name)
        for sd in sorted(os.listdir(stem_dir)):
            sd_dir = os.path.join(stem_dir, sd)
            if not os.path.isdir(sd_dir):
                continue
            k = _seed_idx(sd_dir)
            out_csv_strict = os.path.join(cfg_out_dir, f'seed{k}_strict.csv')
            out_csv_tolerant = os.path.join(cfg_out_dir, f'seed{k}_tolerant.csv')
            out_meta = os.path.join(cfg_out_dir, f'seed{k}_meta.json')

            if (os.path.exists(out_csv_strict) and
                    os.path.exists(out_csv_tolerant)):
                prev_s = pd.read_csv(out_csv_strict)
                prev_t = pd.read_csv(out_csv_tolerant)
                rec_s = [c for c in prev_s.columns
                         if re.fullmatch(r're_o\d+', c)]
                rec_t = [c for c in prev_t.columns
                         if re.fullmatch(r're_o\d+', c)]
                if rec_s and rec_t:
                    F_strict_min = prev_s[rec_s].values
                    F_tolerant_min = prev_t[rec_t].values
                    sd_meta = {}
                    if os.path.exists(out_meta):
                        try:
                            with open(out_meta) as f:
                                sd_meta = json.load(f)
                        except Exception:
                            sd_meta = {}
                    cell = {
                        'scoring': scoring, 'scenm': scenm,
                        'n_obj': n_obj, 'seed': k,
                        'F_strict_min': F_strict_min,
                        'F_tolerant_min': F_tolerant_min,
                        'n_evaluated': int(sd_meta.get('n_evaluated', 0)),
                        'mean_corr': float(sd_meta.get('mean_corr',
                                                       float('nan'))),
                        'n_total_pol_strict':
                            int(sd_meta.get('n_total_pol_strict', -1)),
                        'n_total_pol_tolerant':
                            int(sd_meta.get('n_total_pol_tolerant', -1)),
                        'n_infeasible_strict':
                            int(sd_meta.get('n_infeasible_strict', -1)),
                        'n_infeasible_tolerant':
                            int(sd_meta.get('n_infeasible_tolerant', -1)),
                        'n_dom_strict':
                            int(sd_meta.get('n_dom_strict', -1)),
                        'n_dom_tolerant':
                            int(sd_meta.get('n_dom_tolerant', -1)),
                    }
                    cells.append(cell)
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
                'meta': {'scoring': scoring, 'scenm': scenm,
                         'n_obj': n_obj},
            })
    return tasks, cells, skipped


# ======================================================================
# Cell ingestion (worker result → cell dict for HV pipeline)
# ======================================================================
def _append_cell_morl(res, adapter, cells):
    if adapter.has_feasibility:
        if res.get('F_strict') is None:
            return
        # store as MIN-conv for consistency with checkpoint load path
        cells.append({
            'scoring': res['scoring'], 'scenm': res['scenm'],
            'n_obj': res['n_obj'], 'seed': res['seed'],
            'F_strict_min':   -res['F_strict']   if len(res['F_strict'])   else np.empty((0, res['n_obj'])),
            'F_tolerant_min': -res['F_tolerant'] if len(res['F_tolerant']) else np.empty((0, res['n_obj'])),
            'n_evaluated':           res['n_evaluated'],
            'mean_corr':             res['mean_corr'],
            'n_total_pol_strict':    res['n_total_pol_strict'],
            'n_total_pol_tolerant':  res['n_total_pol_tolerant'],
            'n_infeasible_strict':   res['n_infeasible_strict'],
            'n_infeasible_tolerant': res['n_infeasible_tolerant'],
            'n_dom_strict':          res['n_dom_strict'],
            'n_dom_tolerant':        res['n_dom_tolerant'],
        })
    else:
        if res.get('front_min') is None:
            return
        cells.append({
            'scoring': res['scoring'], 'scenm': res['scenm'],
            'n_obj': res['n_obj'], 'seed': res['seed'],
            'F_min': res['front_min'],
            'n_total_pol': res['n_total_pol'],
            'n_dom':       res['n_dom'],
            'mean_corr':   res['mean_corr'],
            'n_unique_seq': res.get('n_unique_seq'),
        })


def _log_morl(i, n_tasks, res, adapter):
    lbl = f"{res['scoring']}_{res['scenm']}_{res['n_obj']}/seed{res['seed']}"
    if adapter.has_feasibility:
        if res.get('note') == 'computed':
            print(f"   [{i:3d}/{n_tasks}] {lbl:50s} "
                  f"{res['dt']:6.1f}s  {res['n_agents']} agent(s), "
                  f"strict: →{len(res['F_strict'])} pols "
                  f"({res['n_infeasible_strict']} infeas, "
                  f"{res['n_dom_strict']} dom);  "
                  f"tolerant: →{len(res['F_tolerant'])} pols "
                  f"({res['n_infeasible_tolerant']} infeas, "
                  f"{res['n_dom_tolerant']} dom)  diag: {res['diag']}")
        else:
            print(f"   [{i:3d}/{n_tasks}] {lbl:50s}  {res['note']}")
    else:
        if res.get('note') == 'computed':
            print(f"   [{i:3d}/{n_tasks}] {lbl:50s} "
                  f"{res['dt']:6.1f}s  {res['n_agents']} agent(s), "
                  f"{res['n_total_pol']}→{res['rows']} pols "
                  f"({res['n_dom']} dom)  diag: {res['diag']}")
        else:
            print(f"   [{i:3d}/{n_tasks}] {lbl:50s}  {res['note']}")


# ======================================================================
# HV panel machinery
# ======================================================================
def _panel_machinery(panel_cells, n_obj, adapter, setting, mode_key='F_min'):
    """`mode_key` selects which front column to union for best-known."""
    nadir, ideal = adapter.get_box(setting, n_obj)
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
    fronts_max = [(-c[mode_key]) for c in panel_cells if len(c[mode_key]) > 0]
    if fronts_max:
        union = np.vstack(fronts_max)
        best_known_hv = hv(union)
    else:
        union = np.empty((0, n_obj))
        best_known_hv = 0.0
    meta.update(box_nadir=nadir.tolist(), box_ideal=ideal.tolist(),
                box_volume=box_volume, best_known_hv=best_known_hv,
                n_union_points=int(len(union)))
    return hv, box_volume, nadir, ideal, meta


# ======================================================================
# Metrics writers
# ======================================================================
def _write_metrics_single(cells, adapter, setting, scenarios, out_root,
                          n_workers):
    by_nobj = defaultdict(list)
    for c in cells:
        by_nobj[c['n_obj']].append(c)

    rows = []
    meta = {
        'setting': setting,
        'kind': 'reevaluation_morl',
        'env': adapter.name,
        'aggregation': 'arithmetic_mean_over_scenarios',
        'n_workers_used': n_workers,
        'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'panels': {},
    }

    for n_obj, panel_cells in sorted(by_nobj.items()):
        hv, box_volume, nadir, ideal, pmeta = _panel_machinery(
            panel_cells, n_obj, adapter, setting, mode_key='F_min')
        meta['panels'][str(n_obj)] = pmeta
        print(f'\n n_obj={n_obj}: {len(panel_cells)} re-evaluated cells, '
              f"{pmeta['estimator']} HV, box_vol={box_volume:.5g}, "
              f"best-known HV={pmeta['best_known_hv']:.5g} "
              f"(={pmeta['best_known_hv'] / box_volume:.4f} of box)")

        for c in panel_cells:
            F_min = c['F_min']
            F_max = -F_min if len(F_min) else F_min
            h = hv(F_max) if len(F_max) else 0.0
            s = _spacing_norm(F_max, nadir, ideal)
            cond = f"closed_loop_{c['scenm']}"
            # Column order matching the originals:
            #   tree:  paradigm, method, condition, scoring, scenario_method,
            #          n_obj, seed, n_total_pol, n_dom, n_solutions,
            #          n_unique_seq, mean_corr, hv, box_volume, hv_ratio,
            #          spacing_norm
            #   lake:  paradigm, method, condition, scoring, scenario_method,
            #          n_obj, seed, n_agents, n_total_pol, n_dom, mean_corr,
            #          n_solutions, hv, box_volume, hv_ratio, spacing_norm
            row = {
                'paradigm': 'MORL',
                'method': c['scoring'],
                'condition': cond,
                'scoring': c['scoring'],
                'scenario_method': c['scenm'],
                'n_obj': c['n_obj'],
                'seed': c['seed'],
                'n_total_pol': (int(c['n_total_pol'])
                                if c['n_total_pol'] >= 0 else np.nan),
                'n_dom': (int(c['n_dom']) if c['n_dom'] >= 0 else np.nan),
                'n_solutions': int(len(F_min)),
            }
            if adapter.morl_has_action_sequence_diag:
                row['n_unique_seq'] = (int(c['n_unique_seq'])
                                       if c['n_unique_seq'] is not None
                                       and c['n_unique_seq'] >= 0 else np.nan)
            row.update({
                'mean_corr': float(c['mean_corr']),
                'hv': h,
                'box_volume': box_volume,
                'hv_ratio': (h / box_volume if box_volume > 0 else np.nan),
                'spacing_norm': s,
            })
            rows.append(row)

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


def _write_metrics_dual(cells, adapter, setting, scenarios,
                        feasibility_threshold, out_root, n_workers):
    by_nobj = defaultdict(list)
    for c in cells:
        by_nobj[c['n_obj']].append(c)

    rows = []
    meta = {
        'setting': setting,
        'kind': 'reevaluation_morl',
        'env': adapter.name,
        'aggregation': ('strict: mean over all scenarios; '
                        'tolerant: mean over feasible scenarios only'),
        'feasibility_threshold': feasibility_threshold,
        'n_workers_used': n_workers,
        'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'panels': {},
    }

    for n_obj, panel_cells in sorted(by_nobj.items()):
        hv_s, box_volume, nadir, ideal, pmeta = _panel_machinery(
            panel_cells, n_obj, adapter, setting, mode_key='F_strict_min')
        # Compute tolerant best-known separately
        fronts_t = [(-c['F_tolerant_min']) for c in panel_cells
                    if len(c['F_tolerant_min']) > 0]
        if fronts_t:
            union_t = np.vstack(fronts_t)
            bk_t = hv_s(union_t)
            nu_t = int(len(union_t))
        else:
            bk_t = 0.0
            nu_t = 0
        bk_s = pmeta['best_known_hv']
        nu_s = pmeta['n_union_points']
        pmeta['best_known_hv_strict'] = bk_s
        pmeta['best_known_hv_tolerant'] = bk_t
        pmeta['n_union_strict'] = nu_s
        pmeta['n_union_tolerant'] = nu_t
        pmeta.pop('best_known_hv', None)
        pmeta.pop('n_union_points', None)
        meta['panels'][str(n_obj)] = pmeta

        print(f'\n n_obj={n_obj}: {len(panel_cells)} re-evaluated cells, '
              f"{pmeta['estimator']} HV, box_vol={box_volume:.5g}")
        print(f'   best-known HV: strict={bk_s:.5g} ({nu_s} pts), '
              f'tolerant={bk_t:.5g} ({nu_t} pts)')

        for c in panel_cells:
            F_s_min = c['F_strict_min']
            F_t_min = c['F_tolerant_min']
            F_s_max = -F_s_min if len(F_s_min) else F_s_min
            F_t_max = -F_t_min if len(F_t_min) else F_t_min
            h_s = hv_s(F_s_max) if len(F_s_max) else 0.0
            h_t = hv_s(F_t_max) if len(F_t_max) else 0.0
            s_s = _spacing_norm(F_s_max, nadir, ideal)
            s_t = _spacing_norm(F_t_max, nadir, ideal)
            cond = f"closed_loop_{c['scenm']}"
            row = {
                'paradigm': 'MORL',
                'method': c['scoring'],
                'condition': cond,
                'scoring': c['scoring'],
                'scenario_method': c['scenm'],
                'n_obj': c['n_obj'],
                'seed': c['seed'],
                'n_evaluated': int(c['n_evaluated']),
                'mean_corr': float(c['mean_corr']),
                'n_total_pol_strict': (int(c['n_total_pol_strict'])
                                       if c['n_total_pol_strict'] >= 0
                                       else np.nan),
                'n_infeasible_strict': (int(c['n_infeasible_strict'])
                                        if c['n_infeasible_strict'] >= 0
                                        else np.nan),
                'n_dom_strict': (int(c['n_dom_strict'])
                                 if c['n_dom_strict'] >= 0 else np.nan),
                'n_solutions_strict': int(len(F_s_min)),
                'hv_strict': h_s,
                'hv_ratio_strict': (h_s / box_volume if box_volume > 0
                                    else np.nan),
                'spacing_norm_strict': s_s,
                'n_total_pol_tolerant': (int(c['n_total_pol_tolerant'])
                                         if c['n_total_pol_tolerant'] >= 0
                                         else np.nan),
                'n_infeasible_tolerant': (int(c['n_infeasible_tolerant'])
                                          if c['n_infeasible_tolerant'] >= 0
                                          else np.nan),
                'n_dom_tolerant': (int(c['n_dom_tolerant'])
                                   if c['n_dom_tolerant'] >= 0 else np.nan),
                'n_solutions_tolerant': int(len(F_t_min)),
                'hv_tolerant': h_t,
                'hv_ratio_tolerant': (h_t / box_volume if box_volume > 0
                                      else np.nan),
                'spacing_norm_tolerant': s_t,
                'box_volume': box_volume,
            }
            rows.append(row)

        agg_s = defaultdict(list)
        agg_t = defaultdict(list)
        for r in rows:
            if r['n_obj'] == n_obj:
                kkey = (r['method'], r['condition'])
                agg_s[kkey].append(r['hv_ratio_strict'])
                agg_t[kkey].append(r['hv_ratio_tolerant'])
        for kkey in sorted(agg_s):
            vs, vt = agg_s[kkey], agg_t[kkey]
            print(f'   {kkey[0]:14s} {kkey[1]:32s} n={len(vs):3d} '
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


# ======================================================================
# Main
# ======================================================================
if __name__ == '__main__':
    # ------------------------------------------------------------------
    # CONFIGURATION — edit these.
    # ------------------------------------------------------------------
    ENV = 'tree'                # 'tree' | 'lake' | 'constrained_lake'
    SETTING = 'deterministic'   # 'deterministic' | 'robust'

    INPUT_ROOT = f'../data/{ENV}/'
    OUTPUT_ROOT = f'../data/{ENV}'

    EVAL_SCENARIOS_PATH_TREE = '../trees/slip_patterns_depth9_eval.npy'
    EVAL_SCENARIOS_PATH_LAKE = '../lakes/lake_scenarios_eval.npy'

    run_scoring = {'pareto': 1, 'indicator': 1, 'decomposition': 1}
    run_n_obj = {2: 1, 6: 1}

    FEASIBILITY_THRESHOLD = 0.80

    N_WORKERS = 0

    # ------------------------------------------------------------------
    adapter = _ENV_MOD.get(ENV)
    print('=' * 64)
    print(f'EVALUATING setting={SETTING} env={ENV} (MORL — target-tracked)')
    print('=' * 64)

    if SETTING == 'deterministic':
        scenarios = adapter.load_default_scenarios()
        print(f'SETTING=deterministic — using 1 eval scenario')
    elif SETTING == 'robust':
        eval_path = (EVAL_SCENARIOS_PATH_TREE if ENV == 'tree'
                     else EVAL_SCENARIOS_PATH_LAKE)
        scenarios = adapter.load_eval_scenarios(eval_path)
        n_sc = adapter.n_scenarios(scenarios)
        print(f'SETTING=robust — loaded {n_sc} eval scenarios from {eval_path}')
    else:
        raise SystemExit(f'unknown SETTING {SETTING!r}')

    if adapter.has_feasibility:
        print(f'FEASIBILITY_THRESHOLD={FEASIBILITY_THRESHOLD}')

    out_root = os.path.join(OUTPUT_ROOT, SETTING, 'morl')
    reeval_root = os.path.join(out_root, 'reeval_archives_morl')
    os.makedirs(reeval_root, exist_ok=True)

    if adapter.has_feasibility:
        tasks, cells, skipped = _build_tasks_dual(
            INPUT_ROOT, SETTING, reeval_root, run_scoring, run_n_obj,
            adapter)
        worker = _process_one_cell_morl_dual
    else:
        tasks, cells, skipped = _build_tasks_single(
            INPUT_ROOT, SETTING, reeval_root, run_scoring, run_n_obj,
            adapter)
        worker = _process_one_cell_morl_single

    n_total = len(tasks) + skipped
    n_workers = (os.cpu_count() if N_WORKERS == 0 else N_WORKERS) or 1
    n_workers = min(n_workers, max(1, len(tasks)))

    print(f'\n=== Re-evaluating MORL agents ({SETTING}) ===')
    print(f'   {n_total} cells total; {skipped} checkpointed, '
          f'{len(tasks)} to compute with {n_workers} worker(s)')

    init_args = (ENV, scenarios, FEASIBILITY_THRESHOLD)
    t0_all = time.time()
    if tasks:
        if n_workers == 1:
            _worker_init(*init_args)
            for i, task in enumerate(tasks, 1):
                res = worker(task)
                _append_cell_morl(res, adapter, cells)
                _log_morl(i, len(tasks), res, adapter)
        else:
            with ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_worker_init,
                    initargs=init_args) as pool:
                futures = {pool.submit(worker, t): t for t in tasks}
                for i, fut in enumerate(as_completed(futures), 1):
                    res = fut.result()
                    _append_cell_morl(res, adapter, cells)
                    _log_morl(i, len(tasks), res, adapter)

    print(f'\n   MORL re-evaluation done in {time.time() - t0_all:.1f}s')
    print(f'   re-evaluated archives under {reeval_root}/')

    if not cells:
        raise SystemExit('no MORL cells available — nothing to score')

    print('\n=== Computing HV on re-evaluated MORL fronts ===')
    if adapter.has_feasibility:
        _write_metrics_dual(cells, adapter, SETTING, scenarios,
                            FEASIBILITY_THRESHOLD, out_root, n_workers)
    else:
        _write_metrics_single(cells, adapter, SETTING, scenarios,
                              out_root, n_workers)
