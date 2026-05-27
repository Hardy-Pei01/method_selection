import os, re, glob, json, time
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constrained_two_lake import ConstrainedTwoLakeEnv

from utils import (
    MC_SAMPLES_6OBJ, MC_SEED, _hv_exact, _hv_mc,
    _nd_filter_max, _seed_idx, _spacing_norm,
)

# ----------------------------------------------------------------------
# Experiment scope
# ----------------------------------------------------------------------
INPUT_ROOT = '../data/constrained_data_1/'
OUTPUT_ROOT = '../data/constrained_data_1'
SETTING = 'robust'

EVAL_SCENARIOS_PATH = '../lakes/lake_scenarios_eval.npy'

run_moea_method = {'NSGAII': 1, 'IBEA': 1, 'MOEAD': 1}
run_policy = {'intertemporal': 1, 'dps': 1}
run_scenario_method = {'multi': 1, 'moro': 1}
run_n_obj = {2: 1, 6: 1}


N_WORKERS = 0

FOLDER_RE = re.compile(
    r'^(intertemporal|dps)_(NSGAII|IBEA|MOEAD)_(multi|moro)_(\d+)$')

# Tolerant mode: a policy passes if it is feasible (info['feasible']==True
# at episode end) in at least this fraction of eval scenarios. Strict mode
# requires 100% feasibility. The 80% threshold mirrors the 20th-percentile
# robustness criterion used during training: a policy that violates in up
# to 20% of scenarios still has its 20th-percentile fitness untouched.
FEASIBILITY_THRESHOLD = 0.80

# ----------------------------------------------------------------------
# Worker-global state
# ----------------------------------------------------------------------
_EVAL_SCENARIOS = None


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
    return ConstrainedTwoLakeEnv(
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
# Policy rollouts (MAX-conv, with feasibility tracking)
# ----------------------------------------------------------------------
def _rollout_intertemporal(env, decisions_u1, decisions_u2):
    env.reset()
    total = np.zeros(env.num_obj, dtype=np.float64)
    feasible = True
    for t in range(env.n_gym_steps):
        _, reward, _, _, info = env.step(
            np.array([decisions_u1[t], decisions_u2[t]], dtype=np.int64))
        total += np.asarray(reward, dtype=np.float64)
        feasible = bool(info.get('feasible', True))
    return total, feasible


def _emission_from_rbf(xt, c1, c2, r1, r2, w1):
    rule = w1 * (abs(xt - c1) / r1) ** 3 + (1 - w1) * (abs(xt - c2) / r2) ** 3
    u = float(np.clip(rule, 0.0, 0.10))
    return int(round(u / 0.02))


def _rollout_dps(env, c1_1, c2_1, r1_1, r2_1, w1_1,
                 c1_2, c2_2, r1_2, r2_2, w1_2):
    env.reset()
    total = np.zeros(env.num_obj, dtype=np.float64)
    feasible = True
    for _ in range(env.n_gym_steps):
        X1, X2 = env.X1, env.X2
        u1 = _emission_from_rbf(X1, c1_1, c2_1, r1_1, r2_1, w1_1)
        u2 = _emission_from_rbf(X2, c1_2, c2_2, r1_2, r2_2, w1_2)
        _, reward, _, _, info = env.step(np.array([u1, u2], dtype=np.int64))
        total += np.asarray(reward, dtype=np.float64)
        feasible = bool(info.get('feasible', True))
    return total, feasible


def _eval_policy_on_scenarios(row, policy_kind, scenarios, n_obj,
                              u1_keys=None, u2_keys=None):
    """Roll one policy across EVERY scenario. Returns:
      returns    : (n_scen, n_obj) per-scenario MAX-conv returns
      feas_flags : (n_scen,) bool — info['feasible'] at episode end
    No short-circuit — callers apply strict/tolerant filtering."""
    n = len(scenarios)
    returns = np.empty((n, n_obj), dtype=np.float64)
    feas_flags = np.zeros(n, dtype=bool)
    if policy_kind == 'intertemporal':
        u1 = [int(row[c]) for c in u1_keys]
        u2 = [int(row[c]) for c in u2_keys]
        for i, s in enumerate(scenarios):
            env = _build_env(s, n_obj)
            ret, feas = _rollout_intertemporal(env, u1, u2)
            returns[i] = ret
            feas_flags[i] = feas
    elif policy_kind == 'dps':
        c1_1, c2_1 = float(row['c1_1']), float(row['c2_1'])
        r1_1, r2_1 = float(row['r1_1']), float(row['r2_1'])
        w1_1 = float(row['w1_1'])
        c1_2, c2_2 = float(row['c1_2']), float(row['c2_2'])
        r1_2, r2_2 = float(row['r1_2']), float(row['r2_2'])
        w1_2 = float(row['w1_2'])
        for i, s in enumerate(scenarios):
            env = _build_env(s, n_obj)
            ret, feas = _rollout_dps(env, c1_1, c2_1, r1_1, r2_1, w1_1,
                                     c1_2, c2_2, r1_2, r2_2, w1_2)
            returns[i] = ret
            feas_flags[i] = feas
    else:
        raise ValueError(f'unknown policy_kind {policy_kind!r}')
    return returns, feas_flags


def _reevaluate_archive(archive_path, policy_kind, n_obj, scenarios):
    """Build two per-archive fronts (MIN-conv) and per-mode counts.

    Strict mode: keep policies with feas_rate == 1.0; mean over all
    scenarios.
    Tolerant mode: keep policies with feas_rate >= FEASIBILITY_THRESHOLD;
    mean over feasible scenarios only (avoids double-punishment via
    the env's violation penalty).
    """
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
    u1_keys = u2_keys = None
    if policy_kind == 'intertemporal':
        u1_keys = sorted(
            [c for c in df.columns if re.fullmatch(r'u1_\d+', c)],
            key=lambda c: int(re.search(r'\d+', c).group()))
        u2_keys = sorted(
            [c for c in df.columns if re.fullmatch(r'u2_\d+', c)],
            key=lambda c: int(re.search(r'\d+', c).group()))

    strict_rows = []
    tolerant_rows = []
    n_inf_strict = 0
    n_inf_tolerant = 0
    for _, row in df.iterrows():
        returns, feas_flags = _eval_policy_on_scenarios(
            row, policy_kind, scenarios, n_obj,
            u1_keys=u1_keys, u2_keys=u2_keys)
        feas_rate = float(feas_flags.mean())

        if feas_rate >= 1.0 - 1e-12:
            strict_rows.append(-returns.mean(axis=0))
        else:
            n_inf_strict += 1

        if feas_rate >= FEASIBILITY_THRESHOLD - 1e-12:
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


# ----------------------------------------------------------------------
# Worker task: re-evaluate one (config, seed) cell
# ----------------------------------------------------------------------
def _process_one_cell(task):
    sd_dir = task['sd_dir']
    k = task['seed']
    policy_kind = task['policy']
    scenm = task['scenm']
    n_obj = task['n_obj']
    out_csv_strict = task['out_csv_strict']
    out_csv_tolerant = task['out_csv_tolerant']
    out_meta = task['out_meta']
    meta = task['meta']
    scenarios = _EVAL_SCENARIOS

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
        r = _reevaluate_archive(archpath, policy_kind, n_obj, scenarios)
        per_arch_strict.append(r['strict_min'])
        per_arch_tolerant.append(r['tolerant_min'])
        n_evaluated_cell += r['n_evaluated']
        n_inf_strict_cell += r['n_infeasible_strict']
        n_inf_tolerant_cell += r['n_infeasible_tolerant']

    # Aggregate per-archive fronts according to scenm. For 'multi', we
    # pool all archives; for 'moro', we keep only the first (the policy
    # families that emerged from per-scenario optimisation runs).
    def _aggregate(per_arch):
        if scenm == 'multi':
            return np.vstack(per_arch) if per_arch else np.empty((0, n_obj))
        return per_arch[0] if per_arch else np.empty((0, n_obj))

    agg_strict_min = _aggregate(per_arch_strict)
    agg_tolerant_min = _aggregate(per_arch_tolerant)

    n_total_strict = int(len(agg_strict_min))
    n_total_tolerant = int(len(agg_tolerant_min))

    F_strict = (_nd_filter_max(-agg_strict_min) if n_total_strict
                else np.empty((0, n_obj)))
    F_tolerant = (_nd_filter_max(-agg_tolerant_min) if n_total_tolerant
                  else np.empty((0, n_obj)))

    n_dom_strict = n_total_strict - int(len(F_strict))
    n_dom_tolerant = n_total_tolerant - int(len(F_tolerant))

    # Two CSVs + one sidecar.
    os.makedirs(os.path.dirname(out_csv_strict), exist_ok=True)
    cols = [f're_o{j + 1}' for j in range(n_obj)]
    pd.DataFrame(-F_strict, columns=cols).to_csv(out_csv_strict, index=False)
    pd.DataFrame(-F_tolerant, columns=cols).to_csv(out_csv_tolerant, index=False)

    with open(out_meta, 'w') as f:
        json.dump({
            'n_archives': len(archives),
            'n_total_pol_strict': n_total_strict,
            'n_total_pol_tolerant': n_total_tolerant,
            'n_evaluated': int(n_evaluated_cell),
            'n_infeasible_strict': int(n_inf_strict_cell),
            'n_infeasible_tolerant': int(n_inf_tolerant_cell),
            'n_dom_strict': int(n_dom_strict),
            'n_dom_tolerant': int(n_dom_tolerant),
            'feasibility_threshold': FEASIBILITY_THRESHOLD,
            'scenm': scenm,
        }, f)

    return {**meta, 'seed': k,
            'F_strict': F_strict, 'F_tolerant': F_tolerant,
            'n_archives': len(archives),
            'n_total_pol_strict': n_total_strict,
            'n_total_pol_tolerant': n_total_tolerant,
            'n_evaluated': int(n_evaluated_cell),
            'n_infeasible_strict': int(n_inf_strict_cell),
            'n_infeasible_tolerant': int(n_inf_tolerant_cell),
            'n_dom_strict': int(n_dom_strict),
            'n_dom_tolerant': int(n_dom_tolerant),
            'dt': time.time() - t0, 'note': 'computed'}


# ----------------------------------------------------------------------
# HV machinery — mode-independent setup + per-mode best-known.
# ----------------------------------------------------------------------
def _fixed_box(n_obj):
    from params_config import (lake_box_robust_dim2, lake_box_robust_dim6)
    box = {2: lake_box_robust_dim2,
           6: lake_box_robust_dim6}[n_obj]
    nadir = np.asarray(box['nadir'], dtype=float)
    ideal = np.asarray(box['ideal'], dtype=float)
    if not np.all(ideal > nadir):
        raise ValueError(f'Degenerate box for lake dim={n_obj}')
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
        policy, algo, scenm, n_obj_s = m.groups()
        n_obj = int(n_obj_s)
        if not run_moea_method.get(algo, 0): continue
        if not run_policy.get(policy, 0): continue
        if not run_scenario_method.get(scenm, 0): continue
        if not run_n_obj.get(n_obj, 0): continue
        yield algo, policy, scenm, n_obj, os.path.join(moea_base, d)


def _load_sidecar(out_meta):
    if not os.path.exists(out_meta):
        return {}
    try:
        with open(out_meta) as f:
            return json.load(f)
    except Exception:
        return {}


def _build_tasks(reeval_root):
    tasks = []
    prebuilt_cells = []
    skipped = 0
    for algo, policy, scenm, n_obj, stem_dir in _enabled_stems():
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
                    sd_meta = _load_sidecar(out_meta)
                    cell = {
                        'algo': algo, 'policy': policy, 'scenm': scenm,
                        'n_obj': n_obj, 'seed': k,
                        'F_strict': F_strict, 'F_tolerant': F_tolerant,
                        'n_archives':            int(sd_meta.get('n_archives', -1)),
                        'n_total_pol_strict':    int(sd_meta.get('n_total_pol_strict', -1)),
                        'n_total_pol_tolerant':  int(sd_meta.get('n_total_pol_tolerant', -1)),
                        'n_evaluated':           int(sd_meta.get('n_evaluated', 0)),
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
                'policy': policy,
                'scenm': scenm,
                'n_obj': n_obj,
                'out_csv_strict': out_csv_strict,
                'out_csv_tolerant': out_csv_tolerant,
                'out_meta': out_meta,
                'meta': {'algo': algo, 'policy': policy,
                         'scenm': scenm, 'n_obj': n_obj},
            })
    return tasks, prebuilt_cells, skipped


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print('=' * 64)
    print(f'EVALUATING setting = {SETTING} (constrained_lake, MOEA — env replay)')
    print('=' * 64)

    eval_scenarios = _load_eval_scenarios(EVAL_SCENARIOS_PATH)
    print(f'loaded {len(eval_scenarios)} eval scenarios from '
          f'{EVAL_SCENARIOS_PATH}')
    print(f'FEASIBILITY_THRESHOLD={FEASIBILITY_THRESHOLD}')

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

    def _record(res):
        if res.get('F_strict') is None:
            return
        cells.append({
            'algo': res['algo'], 'policy': res['policy'],
            'scenm': res['scenm'], 'n_obj': res['n_obj'],
            'seed': res['seed'],
            'F_strict': res['F_strict'], 'F_tolerant': res['F_tolerant'],
            'n_archives':            res['n_archives'],
            'n_total_pol_strict':    res['n_total_pol_strict'],
            'n_total_pol_tolerant':  res['n_total_pol_tolerant'],
            'n_evaluated':           res['n_evaluated'],
            'n_infeasible_strict':   res['n_infeasible_strict'],
            'n_infeasible_tolerant': res['n_infeasible_tolerant'],
            'n_dom_strict':          res['n_dom_strict'],
            'n_dom_tolerant':        res['n_dom_tolerant'],
        })

    def _print_log(i, n_tasks, res):
        lbl = (f"{res['policy']}_{res['algo']}_{res['scenm']}_"
               f"{res['n_obj']}/seed{res['seed']}")
        if res.get('note') == 'computed':
            print(f"   [{i:3d}/{n_tasks}] {lbl:50s} "
                  f"{res['dt']:6.1f}s  {res['n_archives']} arch; "
                  f"strict {res['n_total_pol_strict']}"
                  f"→{len(res['F_strict'])} "
                  f"(infeas={res['n_infeasible_strict']}, "
                  f"dom={res['n_dom_strict']});  "
                  f"tolerant {res['n_total_pol_tolerant']}"
                  f"→{len(res['F_tolerant'])} "
                  f"(infeas={res['n_infeasible_tolerant']}, "
                  f"dom={res['n_dom_tolerant']})")
        else:
            print(f"   [{i:3d}/{n_tasks}] {lbl:50s}  {res['note']}")

    t0_all = time.time()
    if tasks:
        if n_workers == 1:
            _worker_init(eval_scenarios)
            for i, task in enumerate(tasks, 1):
                res = _process_one_cell(task)
                _record(res)
                _print_log(i, len(tasks), res)
        else:
            with ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_worker_init,
                    initargs=(eval_scenarios,)) as pool:
                futures = {pool.submit(_process_one_cell, t): t
                           for t in tasks}
                for i, fut in enumerate(as_completed(futures), 1):
                    res = fut.result()
                    _record(res)
                    _print_log(i, len(tasks), res)

    print(f'\n   re-evaluation phase done in {time.time() - t0_all:.1f}s')

    if not cells:
        raise SystemExit('no MOEA cells available — nothing to score')

    print('\n=== Computing HV on re-evaluated fronts ===')
    by_nobj = defaultdict(list)
    for c in cells:
        by_nobj[c['n_obj']].append(c)

    rows = []
    meta = {'setting': SETTING,
            'kind': 'reevaluation_moea',
            'n_eval_scenarios': len(eval_scenarios),
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
            'scenm_rule': ('multi: pool all per-scenario archives; '
                           'moro: use the first archive only'),
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
            cond = f"{c['policy']}_{c['scenm']}"
            rows.append(dict(
                paradigm='MOEA', method=c['algo'], condition=cond,
                policy=c['policy'], scenario_method=c['scenm'],
                n_obj=c['n_obj'], seed=c['seed'],
                n_archives=int(c['n_archives']) if c['n_archives'] >= 0 else np.nan,
                n_evaluated=int(c['n_evaluated']),
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
            print(f'   {k[0]:8s} {k[1]:32s} n={len(vs):3d} '
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
    print(f'  re-evaluated archives under {reeval_root}/')
