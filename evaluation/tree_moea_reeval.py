import os, re, glob, json, time
import numpy as np
import pandas as pd
from collections import defaultdict
from moocore import hypervolume as _exact_hv

# ----------------------------------------------------------------------
# Experiment selection — flip 1/0
# ----------------------------------------------------------------------
INPUT_ROOT = '../data/tree_data_1/'
OUTPUT_ROOT = '../data/tree_data_1'
SETTING = 'deterministic'

run_moea_method = {
    'NSGAII': 1,
    'IBEA': 1,
    'MOEAD': 1,
}
run_policy = {
    'table': 1,
    'intertemporal': 1,
}
run_observability = {
    'observable': 1,
    'non_observable': 1,
}
run_n_obj = {2: 1, 6: 1}

MC_SAMPLES_6OBJ = 50_000
MC_SEED = 12345


# ----------------------------------------------------------------------
# Front loading
# ----------------------------------------------------------------------
def _obj_cols(df):
    return [c for c in df.columns if re.match(r'^o\d+$', c)]


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
            if f.startswith(prefix) and f.endswith('.csv')]
    return os.path.join(folder, hits[0]) if hits else None


def _enabled_cells():
    """Yield (paradigm, method, condition, n_obj, seed_idx, front_maxconv)
    for every enabled+available MOEA run. Deterministic MOEA archives
    are single-scenario realised returns, so no re-evaluation is needed
    here — we read archives_*.csv directly."""

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
                for obs, o_on in run_observability.items():
                    if not o_on:
                        continue
                    stem = f'{policy}_{method}_single_{n_obj}_{obs}'
                    sds = _seed_dirs(moea_base, stem)
                    if not sds:
                        continue
                    cond = f'{policy}_{obs}'
                    for sd in sds:
                        csv = _only_csv(sd, 'archives_')
                        if csv:
                            k = int(re.search(r'seed(\d+)', sd).group(1))
                            yield ('MOEA', method, cond, n_obj, k,
                                   _front_maxconv(csv))


# ----------------------------------------------------------------------
# Hypervolume (MAX convention)
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
    """Problem-level box loaded from params_config — same for every
    paradigm/method/seed, ensuring HV ratios are comparable."""
    from params_config import tree_box_dim2, tree_box_dim6
    box = {2: tree_box_dim2, 6: tree_box_dim6}[n_obj]
    nadir = np.asarray(box['nadir'], dtype=float)
    ideal = np.asarray(box['ideal'], dtype=float)
    if not np.all(ideal > nadir):
        raise ValueError(
            f'Degenerate box for tree dim={n_obj}: '
            f'nadir={nadir.tolist()} ideal={ideal.tolist()}')
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

    union = np.vstack([f for *_, f in cells])
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
    print(f'EVALUATING setting = {SETTING} (tree)')
    print('=' * 64)

    cells = list(_enabled_cells())
    if not cells:
        print(f'  no enabled+available runs for {SETTING} — '
              f'check INPUT_ROOT={INPUT_ROOT}')
        raise SystemExit(1)

    # Report what was found
    by_para = defaultdict(set)
    for paradigm, method, cond, n_obj, _, _ in cells:
        by_para[paradigm].add((method, cond, n_obj))
    print(f'\nfound {len(cells)} fronts across paradigms:')
    for p in sorted(by_para):
        print(f'  {p}: {len(by_para[p])} (method, condition, n_obj) cells')

    by_nobj = defaultdict(list)
    for c in cells:
        by_nobj[c[3]].append(c)

    rows = []
    meta = {'setting': SETTING,
            'kind': 'evaluation_moea',
            'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'scope': {
                'moea': [k for k, v in run_moea_method.items() if v],
                'policy': [k for k, v in run_policy.items() if v],
                'observability': [k for k, v in run_observability.items() if v],
                'n_obj': [k for k, v in run_n_obj.items() if v],
            },
            'note': ('Deterministic MOEA HV scoring. Archives stored '
                     'MIN-conv (negated to MAX). Fixed HV box from '
                     'params_config — identical to the box used for MORL '
                     'scoring in tree_morl_reeval.py, so HV ratios are '
                     'directly comparable across paradigms.'),
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
