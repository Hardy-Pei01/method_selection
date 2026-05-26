import os, re, glob, json, time, sys
import numpy as np
import pandas as pd
from collections import defaultdict
from moocore import hypervolume as _exact_hv

# Project root on sys.path so `from params_config import ...` works
# regardless of which directory the script is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ----------------------------------------------------------------------
# Experiment selection — flip 1/0
# ----------------------------------------------------------------------
INPUT_ROOT  = '../data/constrained_data_1/'
OUTPUT_ROOT = '../data/constrained_data_1'
SETTING = 'deterministic'   # robust uses robust_constrained_lake_moea_reeval.py

run_moea_method = {
    'NSGAII': 1,
    'IBEA':   1,
    'MOEAD':  1,
}
run_policy = {
    'intertemporal': 1,
    'dps':           1,
}
run_n_obj = {2: 1, 6: 1}

MC_SAMPLES_6OBJ = 50_000
MC_SEED = 12345


# ----------------------------------------------------------------------
# Front loading + feasibility filter — deterministic single-scenario MOEA.
# Layout (mirrors lake_moea_reeval.py):
#   <INPUT_ROOT>/<SETTING>/moea/<policy>_<algo>_single_<n_obj>/seed<k>/archives_*.csv
#
# Stored o* IS the realised return (penalty included in env reward).
# We do NOT re-run the env — deterministic policies are unaffected.
# Infeasibility is detected directly from the stored values:
#   A policy is feasible iff all MAX-conv axes lie within the lake_box
#   (i.e. >= nadir on every axis). The penalty (-10 per axis per
#   violating year, applied uniformly) drives infeasible policies far
#   below the nadir on every axis simultaneously, so the box-edge test
#   is sharp:
#     feasible upper bounds (MIN-conv):
#       max_P_step total       ≤ Pcrit  ≈ 0.7
#       -utility total         ≤ 0
#       inertia total          ≤ 1
#     vs box-derived MIN-conv ceiling (= -nadir):
#       max_P axis: 3,  utility: 0,  inertia: 1
#     while a single violation adds +10 to every MIN-conv axis — well
#     beyond all the ceilings above.
# ----------------------------------------------------------------------
import glob


def _obj_cols(df):
    return sorted(
        [c for c in df.columns if re.fullmatch(r'o\d+', c)],
        key=lambda c: int(re.search(r'\d+', c).group()))


def _fixed_box_consts(n_obj):
    """Return the box's nadir/ideal as numpy arrays (MAX-conv)."""
    from params_config import (lake_box_deterministic_dim2,
                               lake_box_deterministic_dim6)
    box = {2: lake_box_deterministic_dim2,
           6: lake_box_deterministic_dim6}[n_obj]
    nadir = np.asarray(box['nadir'], dtype=float)
    ideal = np.asarray(box['ideal'], dtype=float)
    return nadir, ideal


def _load_and_filter(path, n_obj):
    """Load one archive, return (front_max_feasible, n_evaluated,
    n_infeasible). MIN-conv -> MAX-conv via negation; feasibility from
    box-nadir threshold."""
    df = pd.read_csv(path)
    min_conv = df[_obj_cols(df)].values.astype(float)
    if min_conv.shape[0] == 0:
        return np.empty((0, n_obj)), 0, 0
    if min_conv.shape[1] != n_obj:
        raise ValueError(
            f'{path}: archive has {min_conv.shape[1]} objective columns, '
            f'expected {n_obj}')
    max_conv = -min_conv
    nadir, _ = _fixed_box_consts(n_obj)
    feasible_mask = (max_conv >= nadir).all(axis=1)
    n_evaluated  = int(max_conv.shape[0])
    n_infeasible = int((~feasible_mask).sum())
    return max_conv[feasible_mask], n_evaluated, n_infeasible


def _seed_dirs(base, stem):
    root = os.path.join(base, stem)
    if not os.path.isdir(root):
        return []
    return sorted(d for d in glob.glob(os.path.join(root, 'seed*'))
                  if os.path.isdir(d))


def _only_csv(folder, prefix):
    hits = [f for f in os.listdir(folder)
            if f.startswith(prefix) and f.endswith('.csv')
            and not f.endswith('_evaluated.csv')
            and not f.startswith('convergences_')]
    return os.path.join(folder, hits[0]) if hits else None


def _enabled_cells():
    """Yield (paradigm, method, condition, n_obj, seed_idx,
              front_maxconv_feasible, n_evaluated, n_infeasible)."""
    moea_base = os.path.join(INPUT_ROOT, SETTING, 'moea')
    for n_obj, on in run_n_obj.items():
        if not on:
            continue
        for method, m_on in run_moea_method.items():
            if not m_on:
                continue
            for policy, p_on in run_policy.items():
                if not p_on:
                    continue
                stem = f'{policy}_{method}_single_{n_obj}'
                sds = _seed_dirs(moea_base, stem)
                if not sds:
                    continue
                for sd in sds:
                    csv = _only_csv(sd, 'archives_')
                    if csv is None:
                        continue
                    k = int(re.search(r'seed(\d+)', sd).group(1))
                    F_feas, n_eval, n_infeas = _load_and_filter(csv, n_obj)
                    yield ('MOEA', method, f'{policy}_single', n_obj, k,
                           F_feas, n_eval, n_infeas)


# ----------------------------------------------------------------------
# HV machinery — IDENTICAL across the three constrained_lake files.
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
    nadir, ideal = _fixed_box_consts(n_obj)
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
    # front is at position 5 of the cell tuple
    union = np.vstack([c[5] for c in cells]) if cells else np.empty((0, n_obj))
    best_known_hv = hv(union) if len(union) else 0.0
    meta.update(box_nadir=nadir.tolist(), box_ideal=ideal.tolist(),
                box_volume=box_volume, best_known_hv=best_known_hv,
                n_union_points=int(len(union)))
    return hv, box_volume, meta


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print('=' * 64)
    print(f'EVALUATING setting = {SETTING} (constrained_lake, MOEA)')
    print('=' * 64)

    cells = list(_enabled_cells())
    if not cells:
        print(f'  no enabled+available runs for {SETTING} — '
              f'check INPUT_ROOT={INPUT_ROOT}')
        raise SystemExit(1)

    by_para = defaultdict(set)
    for paradigm, method, cond, n_obj, _, _, _, _ in cells:
        by_para[paradigm].add((method, cond, n_obj))
    print(f'\nfound {len(cells)} fronts:')
    for p in sorted(by_para):
        print(f'  {p}: {len(by_para[p])} (method, condition, n_obj) cells')

    by_nobj = defaultdict(list)
    for c in cells:
        by_nobj[c[3]].append(c)

    rows = []
    meta = {'setting': SETTING,
            'kind': 'evaluation_moea',
            'problem': 'constrained_lake',
            'source': 'archives_*.csv stored o* (no env replay)',
            'aggregation': 'identity (single-scenario archive values)',
            'feasibility_filter': ('drop rows whose MAX-conv values fall '
                                   'below the lake_box nadir on any axis '
                                   '(penalty signature, no env replay)'),
            'hv_box_source': ('params_config.lake_box_deterministic_* — '
                              'same constants as the unconstrained two-lake '
                              'problem (valid because feasible policies '
                              'have zero penalty contribution)'),
            'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'scope': {
                'moea':           [k for k, v in run_moea_method.items() if v],
                'policy':         [k for k, v in run_policy.items() if v],
                'n_obj':          [k for k, v in run_n_obj.items() if v],
            },
            'panels': {}}

    for n_obj, panel_cells in sorted(by_nobj.items()):
        hv, box_volume, pmeta = _panel_machinery(panel_cells, n_obj)
        panel_n_eval     = int(sum(c[6] for c in panel_cells))
        panel_n_infeas   = int(sum(c[7] for c in panel_cells))
        pmeta['n_evaluated_total']  = panel_n_eval
        pmeta['n_infeasible_total'] = panel_n_infeas
        meta['panels'][str(n_obj)]  = pmeta
        print(f'\n n_obj={n_obj}: {len(panel_cells)} runs, '
              f"{pmeta['estimator']} HV, box_vol={box_volume:.5g}, "
              f"best-known HV={pmeta['best_known_hv']:.5g} "
              f"(={pmeta['best_known_hv'] / box_volume:.4f} of box)")
        if panel_n_eval:
            print(f'   infeasible: {panel_n_infeas}/{panel_n_eval} '
                  f'({panel_n_infeas / panel_n_eval:.1%}) across cells')
        for paradigm, method, cond, no, seed, F, n_eval, n_infeas in panel_cells:
            h = hv(F) if len(F) else 0.0
            rows.append(dict(
                paradigm=paradigm, method=method, condition=cond,
                n_obj=no, seed=seed, n_solutions=int(len(F)),
                n_evaluated=int(n_eval), n_infeasible=int(n_infeas),
                hv=h, box_volume=box_volume,
                hv_ratio=(h / box_volume if box_volume > 0 else np.nan)))
        agg = defaultdict(list)
        for r in rows:
            if r['n_obj'] == n_obj:
                agg[(r['paradigm'], r['method'], r['condition'])].append(
                    r['hv_ratio'])
        for (p, m, c), vs in sorted(agg.items()):
            print(f'   {p:4s} {m:14s} {c:24s} n={len(vs):2d} '
                  f'HVr={np.mean(vs):.4f}±{np.std(vs):.4f}')

    out_dir = os.path.join(OUTPUT_ROOT, SETTING, 'moea')
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(
        ['n_obj', 'paradigm', 'method', 'condition', 'seed'])
    df.to_csv(os.path.join(out_dir, 'metrics_long_reeval.csv'), index=False)
    with open(os.path.join(out_dir, '_meta_reeval.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'\n  wrote {out_dir}/metrics_long_reeval.csv  ({len(df)} rows)')
    print(f'  wrote {out_dir}/_meta_reeval.json')
