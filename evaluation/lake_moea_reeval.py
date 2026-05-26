import os, re, glob, json, time
import numpy as np
import pandas as pd
from collections import defaultdict
from moocore import hypervolume as _exact_hv

# ----------------------------------------------------------------------
# Experiment selection — flip 1/0
# ----------------------------------------------------------------------
INPUT_ROOT = '../data/lake_data_2/'
OUTPUT_ROOT = '../data/lake_data_2'
SETTING = 'deterministic'  # robust uses robust_lake_moea_reeval.py

run_moea_method = {
    'NSGAII': 1,
    'IBEA': 1,
    'MOEAD': 1,
}
run_policy = {
    'intertemporal': 1,
    'dps': 1,
}
run_n_obj = {2: 1, 6: 1}

MC_SAMPLES_6OBJ = 50_000
MC_SEED = 12345


# ----------------------------------------------------------------------
# Front loading — deterministic single-scenario MOEA: stored o* IS the
# realised return, so we read it directly (no env replay).
# ----------------------------------------------------------------------
def _obj_cols(df):
    return sorted(
        [c for c in df.columns if re.fullmatch(r'o\d+', c)],
        key=lambda c: int(re.search(r'\d+', c).group()))


def _front_maxconv(path):
    """MOEA archives are MIN-conv -> negate to MAX-conv."""
    df = pd.read_csv(path)
    v = df[_obj_cols(df)].values.astype(float)
    return -v


def _seed_dirs(base, stem):
    root = os.path.join(base, stem)
    if not os.path.isdir(root):
        return []
    return sorted(d for d in glob.glob(os.path.join(root, 'seed*'))
                  if os.path.isdir(d))


def _only_csv(folder, prefix):
    hits = [f for f in os.listdir(folder)
            if f.startswith(prefix) and f.endswith('.csv')
            and not f.endswith('_evaluated.csv')]
    return os.path.join(folder, hits[0]) if hits else None


def _enabled_cells():
    """Yield (paradigm, method, condition, n_obj, seed_idx, front_maxconv)."""
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
                cond = f'{policy}_single'
                for sd in sds:
                    csv = _only_csv(sd, 'archives_')
                    if csv:
                        k = int(re.search(r'seed(\d+)', sd).group(1))
                        yield ('MOEA', method, cond, n_obj, k,
                               _front_maxconv(csv))


# ----------------------------------------------------------------------
# Hypervolume (MAX convention) — IDENTICAL to lake_morl_reeval.py
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
    from params_config import (lake_box_deterministic_dim2,
                               lake_box_deterministic_dim6)
    box = {2: lake_box_deterministic_dim2,
           6: lake_box_deterministic_dim6}[n_obj]
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
    union = np.vstack([c[-1] for c in cells])
    best_known_hv = hv(union)
    meta.update(box_nadir=nadir.tolist(), box_ideal=ideal.tolist(),
                box_volume=box_volume, best_known_hv=best_known_hv,
                n_union_points=int(len(union)))
    return hv, box_volume, meta


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print('=' * 64)
    print(f'EVALUATING setting = {SETTING} (lake, MOEA)')
    print('=' * 64)

    cells = list(_enabled_cells())
    if not cells:
        print(f'  no enabled+available runs for {SETTING} — '
              f'check INPUT_ROOT={INPUT_ROOT}')
        raise SystemExit(1)

    by_para = defaultdict(set)
    for paradigm, method, cond, n_obj, _, _ in cells:
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
            'source': 'archives_*.csv stored o* (no env replay)',
            'aggregation': 'identity (single-scenario archive values)',
            'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'scope': {
                'moea': [k for k, v in run_moea_method.items() if v],
                'policy': [k for k, v in run_policy.items() if v],
                'n_obj': [k for k, v in run_n_obj.items() if v],
            },
            'note': ('Deterministic MOEA HV scoring. Single-scenario '
                     'archives store realised o* directly (MIN-conv); '
                     'negated to MAX-conv. Fixed HV box from '
                     'params_config — identical to the box used for '
                     'MORL scoring in lake_morl_reeval.py, so HV ratios '
                     'are directly comparable across paradigms.'),
            'panels': {}}

    for n_obj, panel_cells in sorted(by_nobj.items()):
        hv, box_volume, pmeta = _panel_machinery(panel_cells, n_obj)
        meta['panels'][str(n_obj)] = pmeta
        print(f'\n n_obj={n_obj}: {len(panel_cells)} runs, '
              f"{pmeta['estimator']} HV, box_vol={box_volume:.5g}, "
              f"best-known HV={pmeta['best_known_hv']:.5g} "
              f"(={pmeta['best_known_hv'] / box_volume:.4f} of box)")
        for paradigm, method, cond, no, seed, F in panel_cells:
            h = hv(F)
            rows.append(dict(
                paradigm=paradigm, method=method, condition=cond,
                n_obj=no, seed=seed, n_solutions=int(len(F)),
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
