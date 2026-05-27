from __future__ import annotations

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
    _nd_filter_max, _seed_idx, _spacing_norm,
)
import envs as _ENV_MOD


# ======================================================================
# Worker-global state. Each worker process gets `scenarios` and a
# small bundle of run config via `_worker_init`.
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
# Per-policy evaluation primitive (env-agnostic via the adapter)
# ======================================================================
def _eval_policy_on_scenarios(adapter, policy_kind, decoded, scenarios, n_obj,
                              observe):
    """Replay one decoded policy across every scenario in `scenarios`.

    Returns:
      returns    : (n_scen, n_obj) per-scenario MAX-conv returns
      feas_flags : (n_scen,) bool — feasibility flag per scenario
                   (always True for non-constrained envs)

    Per-env env-management policy:
      * tree: build one env, swap _slip_pattern in-place per scenario
      * lake / constrained_lake: build a fresh env per scenario
    """
    n_scen = adapter.n_scenarios(scenarios)
    returns = np.empty((n_scen, n_obj), dtype=np.float64)
    feas_flags = np.zeros(n_scen, dtype=bool)

    if adapter.name == 'tree':
        env = adapter.build_env(None, n_obj, observe=observe)
        for s_idx, scen in enumerate(adapter.iter_scenarios(scenarios)):
            adapter.install_scenario(env, scen)
            ret, feas = adapter.moea_rollout(env, policy_kind, decoded, n_obj)
            returns[s_idx] = ret
            feas_flags[s_idx] = feas
    else:
        for s_idx, scen in enumerate(adapter.iter_scenarios(scenarios)):
            env = adapter.build_env(scen, n_obj, observe=observe)
            ret, feas = adapter.moea_rollout(env, policy_kind, decoded, n_obj)
            returns[s_idx] = ret
            feas_flags[s_idx] = feas

    return returns, feas_flags


# ======================================================================
# Per-archive aggregation. Two code paths:
#   * single-mode (non-constrained): one front per archive
#   * dual-mode (constrained): strict + tolerant fronts per archive
# Both consume the (returns, feas_flags) primitive above.
# ======================================================================
def _reevaluate_archive_single_mode(adapter, archive_path, policy_kind,
                                    n_obj, scenarios, observe):
    """Returns: (front_min, n_evaluated) where front_min is (n_pol, n_obj)
    of MEAN realised returns in MIN-conv (matching archive 'o*' sign)."""
    df = pd.read_csv(archive_path)
    cols = adapter.moea_decision_cols(df, policy_kind)
    if not cols and len(df) > 0:
        return None, 0  # parser failed
    n_pol = len(df)
    if n_pol == 0:
        return np.empty((0, n_obj)), 0
    out_min = np.empty((n_pol, n_obj), dtype=np.float64)
    for i, row in df.iterrows():
        decoded = adapter.moea_decode_row(row, policy_kind, cols)
        returns, _feas = _eval_policy_on_scenarios(
            adapter, policy_kind, decoded, scenarios, n_obj, observe)
        out_min[i] = -returns.mean(axis=0)  # MAX → MIN
    return out_min, n_pol


def _reevaluate_archive_dual_mode(adapter, archive_path, policy_kind, n_obj,
                                  scenarios, observe, feasibility_threshold):
    """Returns dict with strict_min, tolerant_min, plus per-mode counts.

    Strict mode: keep policies with feas_rate == 1.0; mean over ALL scenarios.
    Tolerant mode: keep policies with feas_rate >= threshold; mean over
                   FEASIBLE scenarios only."""
    df = pd.read_csv(archive_path)
    n_pol = len(df)
    if n_pol == 0:
        return {
            'strict_min': np.empty((0, n_obj)),
            'tolerant_min': np.empty((0, n_obj)),
            'n_evaluated': 0,
            'n_infeasible_strict': 0,
            'n_infeasible_tolerant': 0,
        }
    cols = adapter.moea_decision_cols(df, policy_kind)
    strict_rows = []
    tolerant_rows = []
    n_inf_strict = 0
    n_inf_tolerant = 0
    for _, row in df.iterrows():
        decoded = adapter.moea_decode_row(row, policy_kind, cols)
        returns, feas_flags = _eval_policy_on_scenarios(
            adapter, policy_kind, decoded, scenarios, n_obj, observe)
        feas_rate = float(feas_flags.mean())

        if feas_rate >= 1.0 - 1e-12:
            strict_rows.append(-returns.mean(axis=0))
        else:
            n_inf_strict += 1

        if feas_rate >= feasibility_threshold - 1e-12:
            tolerant_rows.append(-returns[feas_flags].mean(axis=0))
        else:
            n_inf_tolerant += 1

    return {
        'strict_min': (np.vstack(strict_rows) if strict_rows
                       else np.empty((0, n_obj))),
        'tolerant_min': (np.vstack(tolerant_rows) if tolerant_rows
                         else np.empty((0, n_obj))),
        'n_evaluated': n_pol,
        'n_infeasible_strict': n_inf_strict,
        'n_infeasible_tolerant': n_inf_tolerant,
    }


# ======================================================================
# Worker tasks. Two variants — single-mode and dual-mode.
# ======================================================================
def _process_one_cell_single(task):
    """Single-mode worker (tree / lake)."""
    adapter = _w_adapter()
    sd_dir = task['sd_dir']
    k = task['seed']
    policy_kind = task['policy']
    n_obj = task['n_obj']
    scenm = task['scenm']
    obs = task['obs']
    out_csv_seed = task['out_csv_seed']
    meta = task['meta']
    scenarios = _W_SCENARIOS
    # `obs` is the parsed folder token ('observable' | 'non_observable')
    # for envs with observability; None otherwise.
    observe = (obs == 'observable') if obs is not None else None

    t0 = time.time()
    archives = sorted(
        f for f in glob.glob(os.path.join(sd_dir, 'archives_*.csv'))
        if not f.endswith('_evaluated.csv'))
    if not archives:
        return {**meta, 'seed': k, 'front_max': None,
                'dt': 0.0, 'note': 'no archive files'}

    per_archive_min = []
    for archpath in archives:
        arr, _ = _reevaluate_archive_single_mode(
            adapter, archpath, policy_kind, n_obj, scenarios, observe)
        if arr is not None:
            per_archive_min.append(arr)

    if not per_archive_min or sum(len(a) for a in per_archive_min) == 0:
        return {**meta, 'seed': k, 'front_max': None,
                'dt': time.time() - t0, 'note': 'no policies parsed'}

    # Per-archive pooling:
    #   single/multi : pool all archives via vstack, then ND-filter
    #   moro         : ND-filter only the first archive (per-scenario runs)
    if scenm == 'moro':
        front_max = _nd_filter_max(-per_archive_min[0])
    else:
        pooled_min = np.vstack(per_archive_min)
        front_max = _nd_filter_max(-pooled_min)

    n_archives = len(per_archive_min)
    n_total_pol = int(sum(len(a) for a in per_archive_min))
    n_dom = n_total_pol - int(len(front_max))

    os.makedirs(os.path.dirname(out_csv_seed), exist_ok=True)
    pd.DataFrame(
        -front_max, columns=[f're_o{j + 1}' for j in range(n_obj)]
    ).to_csv(out_csv_seed, index=False)

    side = out_csv_seed.replace('.csv', '_meta.json')
    with open(side, 'w') as f:
        json.dump({'n_archives': n_archives,
                   'n_total_pol': n_total_pol,
                   'n_dom': n_dom}, f)

    return {**meta, 'seed': k, 'front_max': front_max,
            'dt': time.time() - t0,
            'n_archives': n_archives,
            'n_total_pol': n_total_pol,
            'n_dom': n_dom,
            'note': 'computed'}


def _process_one_cell_dual(task):
    """Dual-mode worker (constrained_lake)."""
    adapter = _w_adapter()
    sd_dir = task['sd_dir']
    k = task['seed']
    policy_kind = task['policy']
    n_obj = task['n_obj']
    scenm = task['scenm']
    out_csv_strict = task['out_csv_strict']
    out_csv_tolerant = task['out_csv_tolerant']
    out_meta = task['out_meta']
    meta = task['meta']
    scenarios = _W_SCENARIOS
    fthr = _W_FEASIBILITY_THRESHOLD
    observe = None  # constrained_lake has no observability

    t0 = time.time()
    archives = sorted(
        f for f in glob.glob(os.path.join(sd_dir, 'archives_*.csv'))
        if not f.endswith('_evaluated.csv'))
    if not archives:
        return {**meta, 'seed': k, 'F_strict': None, 'F_tolerant': None,
                'dt': 0.0, 'note': 'no archive files'}

    per_arch_strict = []
    per_arch_tolerant = []
    n_evaluated_cell = 0
    n_inf_strict_cell = 0
    n_inf_tolerant_cell = 0
    for archpath in archives:
        r = _reevaluate_archive_dual_mode(
            adapter, archpath, policy_kind, n_obj, scenarios, observe, fthr)
        per_arch_strict.append(r['strict_min'])
        per_arch_tolerant.append(r['tolerant_min'])
        n_evaluated_cell += r['n_evaluated']
        n_inf_strict_cell += r['n_infeasible_strict']
        n_inf_tolerant_cell += r['n_infeasible_tolerant']

    # Aggregate per scenm — for 'moro' keep only archives[0]; else pool all.
    def _agg(per_arch):
        if scenm == 'moro':
            return per_arch[0] if per_arch else np.empty((0, n_obj))
        return np.vstack(per_arch) if per_arch else np.empty((0, n_obj))

    agg_strict_min = _agg(per_arch_strict)
    agg_tolerant_min = _agg(per_arch_tolerant)

    n_total_strict = int(len(agg_strict_min))
    n_total_tolerant = int(len(agg_tolerant_min))

    F_strict = (_nd_filter_max(-agg_strict_min) if n_total_strict
                else np.empty((0, n_obj)))
    F_tolerant = (_nd_filter_max(-agg_tolerant_min) if n_total_tolerant
                  else np.empty((0, n_obj)))

    n_dom_strict = n_total_strict - int(len(F_strict))
    n_dom_tolerant = n_total_tolerant - int(len(F_tolerant))

    os.makedirs(os.path.dirname(out_csv_strict), exist_ok=True)
    cols = [f're_o{j + 1}' for j in range(n_obj)]
    pd.DataFrame(-F_strict, columns=cols).to_csv(out_csv_strict, index=False)
    pd.DataFrame(-F_tolerant, columns=cols).to_csv(out_csv_tolerant, index=False)

    with open(out_meta, 'w') as f:
        sidecar = {
            'n_archives': len(archives),
            'n_evaluated': int(n_evaluated_cell),
            'n_infeasible_strict': int(n_inf_strict_cell),
            'n_infeasible_tolerant': int(n_inf_tolerant_cell),
            'n_dom_strict': int(n_dom_strict),
            'n_dom_tolerant': int(n_dom_tolerant),
            'feasibility_threshold': fthr,
        }
        if scenm in ('multi', 'moro'):
            sidecar['n_total_pol_strict'] = n_total_strict
            sidecar['n_total_pol_tolerant'] = n_total_tolerant
            sidecar['scenm'] = scenm
        else:
            sidecar['n_total_pol'] = int(n_evaluated_cell)
        json.dump(sidecar, f)

    return {**meta, 'seed': k,
            'F_strict': F_strict, 'F_tolerant': F_tolerant,
            'n_archives': len(archives),
            'n_total_pol': int(n_evaluated_cell),
            'n_evaluated': int(n_evaluated_cell),
            'n_total_pol_strict': n_total_strict,
            'n_total_pol_tolerant': n_total_tolerant,
            'n_infeasible_strict': int(n_inf_strict_cell),
            'n_infeasible_tolerant': int(n_inf_tolerant_cell),
            'n_dom_strict': int(n_dom_strict),
            'n_dom_tolerant': int(n_dom_tolerant),
            'dt': time.time() - t0, 'note': 'computed'}


# ======================================================================
# HV panel machinery — shared with morl_reeval.py; identical formula.
# ======================================================================
def _panel_machinery(panel_cells_max, n_obj, adapter, setting):
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
    union = np.vstack([c['F'] for c in panel_cells_max])
    best_known_hv = hv(union) if len(union) else 0.0
    meta.update(box_nadir=nadir.tolist(), box_ideal=ideal.tolist(),
                box_volume=box_volume, best_known_hv=best_known_hv,
                n_union_points=int(len(union)))
    return hv, box_volume, nadir, ideal, meta


def _panel_machinery_dual(panel_cells, n_obj, adapter, setting):
    """Like _panel_machinery but for the dual-mode (constrained) case.
    Returns separate best-known HVs for strict and tolerant."""
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
    meta.update(box_nadir=nadir.tolist(), box_ideal=ideal.tolist(),
                box_volume=box_volume)
    return hv, box_volume, nadir, ideal, meta


# ======================================================================
# Discovery + task planning (single-mode)
# ======================================================================
def _enabled_stems(input_root, setting, adapter, run_method, run_policy,
                   run_scenario_method, run_obs, run_n_obj):
    moea_base = os.path.join(input_root, setting, 'moea')
    if not os.path.isdir(moea_base):
        return
    pattern = adapter.moea_folder_regex(setting)
    for d in sorted(os.listdir(moea_base)):
        m = pattern.match(d)
        if not m:
            continue
        info = adapter.moea_parse_match(m, setting)
        if not run_method.get(info['algo'], 0): continue
        if not run_policy.get(info['policy'], 0): continue
        if not run_scenario_method.get(info['scenm'], 0): continue
        if not run_n_obj.get(info['n_obj'], 0): continue
        if adapter.has_observability:
            if not run_obs.get(info['obs'], 0): continue
        yield info, os.path.join(moea_base, d)


def _build_tasks_single(input_root, setting, adapter, reeval_root,
                        run_method, run_policy, run_scenario_method,
                        run_obs, run_n_obj):
    tasks = []
    prebuilt_cells = []
    skipped = 0
    for info, stem_dir in _enabled_stems(
            input_root, setting, adapter,
            run_method, run_policy, run_scenario_method, run_obs, run_n_obj):
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
                    F_max = -prev[rec].values
                    side = out_csv_seed.replace('.csv', '_meta.json')
                    if os.path.exists(side):
                        with open(side) as sf:
                            sd_meta = json.load(sf)
                        n_arc = int(sd_meta.get('n_archives', 0))
                        n_tot = int(sd_meta.get('n_total_pol', 0))
                        n_dm = int(sd_meta.get('n_dom', 0))
                    else:
                        n_arc = n_tot = n_dm = -1
                    cell = dict(info, seed=k, F=F_max,
                                n_archives=n_arc, n_total_pol=n_tot, n_dom=n_dm)
                    prebuilt_cells.append(cell)
                    skipped += 1
                    continue
            tasks.append({
                'sd_dir': sd_dir,
                'seed': k,
                'policy': info['policy'],
                'scenm': info['scenm'],
                'obs': info['obs'],
                'n_obj': info['n_obj'],
                'out_csv_seed': out_csv_seed,
                'meta': info,
            })
    return tasks, prebuilt_cells, skipped


def _build_tasks_dual(input_root, setting, adapter, reeval_root,
                      run_method, run_policy, run_scenario_method,
                      run_obs, run_n_obj):
    tasks = []
    prebuilt_cells = []
    skipped = 0
    for info, stem_dir in _enabled_stems(
            input_root, setting, adapter,
            run_method, run_policy, run_scenario_method, run_obs, run_n_obj):
        stem_name = os.path.basename(stem_dir)
        cfg_out_dir = os.path.join(reeval_root, stem_name)
        seeds = sorted(glob.glob(os.path.join(stem_dir, 'seed*')))
        for sd_dir in seeds:
            k = _seed_idx(sd_dir)
            out_csv_strict = os.path.join(cfg_out_dir, f'seed{k}_strict.csv')
            out_csv_tolerant = os.path.join(cfg_out_dir, f'seed{k}_tolerant.csv')
            out_meta = os.path.join(cfg_out_dir, f'seed{k}_meta.json')
            if (os.path.exists(out_csv_strict)
                    and os.path.exists(out_csv_tolerant)):
                prev_s = pd.read_csv(out_csv_strict)
                prev_t = pd.read_csv(out_csv_tolerant)
                rec_s = [c for c in prev_s.columns
                         if re.fullmatch(r're_o\d+', c)]
                rec_t = [c for c in prev_t.columns
                         if re.fullmatch(r're_o\d+', c)]
                if rec_s and rec_t:
                    F_strict = -prev_s[rec_s].values
                    F_tolerant = -prev_t[rec_t].values
                    sd_meta = {}
                    if os.path.exists(out_meta):
                        try:
                            with open(out_meta) as f:
                                sd_meta = json.load(f)
                        except Exception:
                            sd_meta = {}
                    cell = dict(info, seed=k,
                                F_strict=F_strict, F_tolerant=F_tolerant,
                                n_archives=int(sd_meta.get('n_archives', -1)),
                                n_total_pol=int(sd_meta.get('n_total_pol', -1)),
                                n_evaluated=int(sd_meta.get('n_evaluated', 0)),
                                n_total_pol_strict=int(sd_meta.get('n_total_pol_strict', -1)),
                                n_total_pol_tolerant=int(sd_meta.get('n_total_pol_tolerant', -1)),
                                n_infeasible_strict=int(sd_meta.get('n_infeasible_strict', -1)),
                                n_infeasible_tolerant=int(sd_meta.get('n_infeasible_tolerant', -1)),
                                n_dom_strict=int(sd_meta.get('n_dom_strict', -1)),
                                n_dom_tolerant=int(sd_meta.get('n_dom_tolerant', -1)))
                    prebuilt_cells.append(cell)
                    skipped += 1
                    continue
            tasks.append({
                'sd_dir': sd_dir,
                'seed': k,
                'policy': info['policy'],
                'scenm': info['scenm'],
                'obs': info['obs'],
                'n_obj': info['n_obj'],
                'out_csv_strict': out_csv_strict,
                'out_csv_tolerant': out_csv_tolerant,
                'out_meta': out_meta,
                'meta': info,
            })
    return tasks, prebuilt_cells, skipped


# ======================================================================
# Condition-label / output-row helpers — match the originals exactly.
# ======================================================================
def _condition_label(adapter, info):
    """Output condition string used in metrics_long_reeval.csv.

    Matches each original script's format:
      det tree:        '{policy}_{obs}'
      robust tree:     '{policy}_{scenm}_{obs}'
      det lake/const:  '{policy}_single'
      robust lake/c.:  '{policy}_{scenm}'
    """
    policy = info['policy']
    scenm = info['scenm']
    obs = info.get('obs')
    if adapter.has_observability:
        if scenm == 'single':
            return f'{policy}_{obs}'
        return f'{policy}_{scenm}_{obs}'
    else:
        if scenm == 'single':
            return f'{policy}_single'
        return f'{policy}_{scenm}'


def _append_cell_from_result(res, adapter, cells):
    """Convert a worker result dict into a cell dict appended to `cells`.
    Returns None; mutates `cells` in place. Skips no-front results."""
    if adapter.has_feasibility:
        if res.get('F_strict') is None:
            return
        cell = {
            'algo': res['algo'], 'policy': res['policy'],
            'scenm': res['scenm'], 'obs': res.get('obs'),
            'n_obj': res['n_obj'], 'seed': res['seed'],
            'F_strict': res['F_strict'], 'F_tolerant': res['F_tolerant'],
            'n_archives':           res['n_archives'],
            'n_total_pol':          res['n_total_pol'],
            'n_evaluated':          res['n_evaluated'],
            'n_total_pol_strict':   res.get('n_total_pol_strict', -1),
            'n_total_pol_tolerant': res.get('n_total_pol_tolerant', -1),
            'n_infeasible_strict':  res['n_infeasible_strict'],
            'n_infeasible_tolerant':res['n_infeasible_tolerant'],
            'n_dom_strict':         res['n_dom_strict'],
            'n_dom_tolerant':       res['n_dom_tolerant'],
        }
    else:
        if res.get('front_max') is None:
            return
        cell = {
            'algo': res['algo'], 'policy': res['policy'],
            'scenm': res['scenm'], 'obs': res.get('obs'),
            'n_obj': res['n_obj'], 'seed': res['seed'],
            'F': res['front_max'],
            'n_archives':  res['n_archives'],
            'n_total_pol': res['n_total_pol'],
            'n_dom':       res['n_dom'],
        }
    cells.append(cell)


def _log(i, n_tasks, res, adapter):
    """Per-task progress line; format matches the originals closely."""
    bits = [res['policy'], res['algo']]
    if res['scenm'] != 'single':
        bits.append(res['scenm'])
    bits.append(str(res['n_obj']))
    if adapter.has_observability and res.get('obs'):
        bits.append(res['obs'])
    lbl = '_'.join(bits) + f"/seed{res['seed']}"

    if adapter.has_feasibility:
        if res.get('note') == 'computed':
            print(f"   [{i:3d}/{n_tasks}] {lbl:50s} "
                  f"{res['dt']:6.1f}s  {res['n_archives']} arch, "
                  f"strict: {res['n_total_pol']}→{len(res['F_strict'])} pols "
                  f"({res['n_infeasible_strict']} infeas, "
                  f"{res['n_dom_strict']} dom);  "
                  f"tolerant: {res['n_total_pol']}→{len(res['F_tolerant'])} pols "
                  f"({res['n_infeasible_tolerant']} infeas, "
                  f"{res['n_dom_tolerant']} dom)")
        else:
            print(f"   [{i:3d}/{n_tasks}] {lbl:50s}  {res['note']}")
    else:
        if res.get('note') == 'computed':
            print(f"   [{i:3d}/{n_tasks}] {lbl:50s} "
                  f"{res['dt']:6.1f}s  {res['n_archives']} arch, "
                  f"{res['n_total_pol']}→{len(res['front_max'])} pols "
                  f"({res['n_dom']} dom)")
        else:
            print(f"   [{i:3d}/{n_tasks}] {lbl:50s}  {res['note']}")


def _write_metrics_single(cells, adapter, setting, scenarios, out_root,
                          n_workers):
    """Write metrics_long_reeval.csv + _meta_reeval.json (single-mode)."""
    by_nobj = defaultdict(list)
    for c in cells:
        by_nobj[c['n_obj']].append(c)

    n_scen = adapter.n_scenarios(scenarios)
    rows = []
    meta = {
        'setting': setting,
        'kind': 'reevaluation_moea',
        'env': adapter.name,
        'n_eval_scenarios': n_scen,
        'aggregation': ('identity (single default scenario)' if n_scen == 1
                        else 'arithmetic_mean_over_scenarios'),
        'n_workers_used': n_workers,
        'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'panels': {},
    }

    for n_obj, panel_cells in sorted(by_nobj.items()):
        hv, box_volume, nadir, ideal, pmeta = _panel_machinery(
            panel_cells, n_obj, adapter, setting)
        meta['panels'][str(n_obj)] = pmeta
        print(f'\n n_obj={n_obj}: {len(panel_cells)} re-evaluated cells, '
              f"{pmeta['estimator']} HV, box_vol={box_volume:.5g}, "
              f"best-known HV={pmeta['best_known_hv']:.5g} "
              f"(={pmeta['best_known_hv'] / box_volume:.4f} of box)")

        for c in panel_cells:
            F = c['F']
            h = hv(F)
            s = _spacing_norm(F, nadir, ideal)
            # Build row in same column order as the original scripts:
            #   paradigm, method, condition, policy, [scenario_method,]
            #   [observability,] n_obj, seed, n_archives, n_total_pol,
            #   n_dom, n_solutions, hv, box_volume, hv_ratio, spacing_norm
            row = {'paradigm': 'MOEA',
                   'method': c['algo'],
                   'condition': _condition_label(adapter, c),
                   'policy': c['policy']}
            if c['scenm'] != 'single':
                row['scenario_method'] = c['scenm']
            if adapter.has_observability:
                row['observability'] = c['obs']
            row.update({
                'n_obj': c['n_obj'],
                'seed': c['seed'],
                'n_archives': int(c['n_archives']) if c['n_archives'] >= 0 else np.nan,
                'n_total_pol': int(c['n_total_pol']) if c['n_total_pol'] >= 0 else np.nan,
                'n_dom': int(c['n_dom']) if c['n_dom'] >= 0 else np.nan,
                'n_solutions': int(len(F)),
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


def _write_metrics_dual(cells, adapter, setting, scenarios,
                        feasibility_threshold, out_root, n_workers):
    """Write metrics_long_reeval.csv + _meta_reeval.json (dual-mode)."""
    by_nobj = defaultdict(list)
    for c in cells:
        by_nobj[c['n_obj']].append(c)

    n_scen = adapter.n_scenarios(scenarios)
    rows = []
    meta = {
        'setting': setting,
        'kind': 'reevaluation_moea',
        'env': adapter.name,
        'n_eval_scenarios': n_scen,
        'aggregation': ('strict: mean over all scenarios; '
                        'tolerant: mean over feasible scenarios only'),
        'feasibility_threshold': feasibility_threshold,
        'feasibility_modes': {
            'strict': ('policy kept iff feasibility rate == 1.0; '
                       'mean over ALL scenarios'),
            'tolerant': (f'policy kept iff feasibility rate >= '
                         f'{feasibility_threshold}; mean over feasible '
                         f'scenarios only'),
        },
        'n_workers_used': n_workers,
        'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'panels': {},
    }

    for n_obj, panel_cells in sorted(by_nobj.items()):
        hv, box_volume, nadir, ideal, pmeta = _panel_machinery_dual(
            panel_cells, n_obj, adapter, setting)
        def _bk(mode):
            key = f'F_{mode}'
            fronts = [c[key] for c in panel_cells if len(c[key]) > 0]
            if not fronts:
                return 0.0, 0
            union = np.vstack(fronts)
            return hv(union), int(len(union))
        bk_s, nu_s = _bk('strict')
        bk_t, nu_t = _bk('tolerant')
        pmeta['best_known_hv_strict'] = bk_s
        pmeta['best_known_hv_tolerant'] = bk_t
        pmeta['n_union_strict'] = nu_s
        pmeta['n_union_tolerant'] = nu_t
        panel_n_eval = int(sum(c['n_evaluated'] for c in panel_cells))
        panel_n_inf_s = int(sum(c['n_infeasible_strict']
                                for c in panel_cells if c['n_infeasible_strict'] >= 0))
        panel_n_inf_t = int(sum(c['n_infeasible_tolerant']
                                for c in panel_cells if c['n_infeasible_tolerant'] >= 0))
        pmeta['n_evaluated_total'] = panel_n_eval
        pmeta['n_infeasible_strict_total'] = panel_n_inf_s
        pmeta['n_infeasible_tolerant_total'] = panel_n_inf_t
        meta['panels'][str(n_obj)] = pmeta

        print(f'\n n_obj={n_obj}: {len(panel_cells)} re-evaluated cells, '
              f"{pmeta['estimator']} HV, box_vol={box_volume:.5g}")
        print(f'   best-known HV: strict={bk_s:.5g} ({nu_s} pts), '
              f'tolerant={bk_t:.5g} ({nu_t} pts)')

        for c in panel_cells:
            F_s = c['F_strict']
            F_t = c['F_tolerant']
            h_s = hv(F_s) if len(F_s) else 0.0
            h_t = hv(F_t) if len(F_t) else 0.0
            s_s = _spacing_norm(F_s, nadir, ideal)
            s_t = _spacing_norm(F_t, nadir, ideal)
            row = {'paradigm': 'MOEA',
                   'method': c['algo'],
                   'condition': _condition_label(adapter, c),
                   'policy': c['policy']}
            if c['scenm'] != 'single':
                row['scenario_method'] = c['scenm']
            row.update({
                'n_obj': c['n_obj'],
                'seed': c['seed'],
                'n_archives': int(c['n_archives']) if c['n_archives'] >= 0 else np.nan,
                'n_total_pol': int(c['n_total_pol']) if c['n_total_pol'] >= 0 else np.nan,
                'n_evaluated': int(c['n_evaluated']),
                'n_infeasible_strict': (int(c['n_infeasible_strict'])
                                        if c['n_infeasible_strict'] >= 0 else np.nan),
                'n_dom_strict': (int(c['n_dom_strict'])
                                 if c['n_dom_strict'] >= 0 else np.nan),
                'n_solutions_strict': int(len(F_s)),
                'hv_strict': h_s,
                'hv_ratio_strict': (h_s / box_volume if box_volume > 0 else np.nan),
                'spacing_norm_strict': s_s,
                'n_infeasible_tolerant': (int(c['n_infeasible_tolerant'])
                                          if c['n_infeasible_tolerant'] >= 0 else np.nan),
                'n_dom_tolerant': (int(c['n_dom_tolerant'])
                                   if c['n_dom_tolerant'] >= 0 else np.nan),
                'n_solutions_tolerant': int(len(F_t)),
                'hv_tolerant': h_t,
                'hv_ratio_tolerant': (h_t / box_volume if box_volume > 0 else np.nan),
                'spacing_norm_tolerant': s_t,
                'box_volume': box_volume,
            })
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
            print(f'   {kkey[0]:8s} {kkey[1]:32s} n={len(vs):3d} '
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
    ENV = 'tree'                  # 'tree' | 'lake' | 'constrained_lake'
    SETTING = 'deterministic'     # 'deterministic' | 'robust'

    INPUT_ROOT = '../data/tree_data_2/'
    OUTPUT_ROOT = '../data/tree_data_2'

    # Robust-only paths
    EVAL_SCENARIOS_PATH_TREE = '../trees/slip_patterns_depth9_eval.npy'
    EVAL_SCENARIOS_PATH_LAKE = '../lakes/lake_scenarios_eval.npy'

    # Run filters
    run_moea_method = {'NSGAII': 1, 'IBEA': 1, 'MOEAD': 1}
    run_policy = {'intertemporal': 1, 'table': 1, 'dps': 1}
    run_scenario_method = {'single': 1, 'multi': 1, 'moro': 1}
    run_observability = {'observable': 1, 'non_observable': 1}
    run_n_obj = {2: 1, 6: 1}

    # Feasibility threshold (constrained only). 0.8 mirrors the
    # 20th-percentile robustness criterion used during training.
    FEASIBILITY_THRESHOLD = 0.80

    # Parallelism: 0 = auto, 1 = serial (debug), N>0 = N workers
    N_WORKERS = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    adapter = _ENV_MOD.get(ENV)
    print('=' * 64)
    print(f'EVALUATING setting={SETTING} env={ENV} (MOEA — env replay)')
    print('=' * 64)

    if SETTING == 'deterministic':
        scenarios = adapter.load_default_scenarios()
        print(f'SETTING=deterministic — using 1 scenario')
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

    out_root = os.path.join(OUTPUT_ROOT, SETTING, 'moea')
    reeval_root = os.path.join(out_root, 'reeval_archives')
    os.makedirs(reeval_root, exist_ok=True)

    if adapter.has_feasibility:
        tasks, cells, skipped = _build_tasks_dual(
            INPUT_ROOT, SETTING, adapter, reeval_root,
            run_moea_method, run_policy, run_scenario_method,
            run_observability, run_n_obj)
        worker = _process_one_cell_dual
    else:
        tasks, cells, skipped = _build_tasks_single(
            INPUT_ROOT, SETTING, adapter, reeval_root,
            run_moea_method, run_policy, run_scenario_method,
            run_observability, run_n_obj)
        worker = _process_one_cell_single

    n_total = len(tasks) + skipped
    n_workers = (os.cpu_count() if N_WORKERS == 0 else N_WORKERS) or 1
    n_workers = min(n_workers, max(1, len(tasks)))

    print(f'\n=== Re-evaluating archives ({SETTING}) ===')
    print(f'   {n_total} cells total; {skipped} already checkpointed, '
          f'{len(tasks)} to compute with {n_workers} worker(s)')

    init_args = (ENV, scenarios, FEASIBILITY_THRESHOLD)
    t0_all = time.time()
    if tasks:
        if n_workers == 1:
            _worker_init(*init_args)
            for i, task in enumerate(tasks, 1):
                res = worker(task)
                _append_cell_from_result(res, adapter, cells)
                _log(i, len(tasks), res, adapter)
        else:
            with ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_worker_init,
                    initargs=init_args) as pool:
                futures = {pool.submit(worker, t): t for t in tasks}
                for i, fut in enumerate(as_completed(futures), 1):
                    res = fut.result()
                    _append_cell_from_result(res, adapter, cells)
                    _log(i, len(tasks), res, adapter)

    print(f'\n   re-evaluation phase done in {time.time() - t0_all:.1f}s')

    if not cells:
        raise SystemExit('no MOEA cells available — nothing to score')

    # ------------------------------------------------------------------
    # HV computation per n_obj panel + final CSV/JSON output
    # ------------------------------------------------------------------
    print('\n=== Computing HV on re-evaluated fronts ===')

    if adapter.has_feasibility:
        _write_metrics_dual(cells, adapter, SETTING, scenarios,
                            FEASIBILITY_THRESHOLD, out_root, n_workers)
    else:
        _write_metrics_single(cells, adapter, SETTING, scenarios,
                              out_root, n_workers)
