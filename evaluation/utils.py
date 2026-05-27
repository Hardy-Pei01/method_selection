import re
import numpy as np

from moocore import hypervolume as _exact_hv

try:
    from moocore import is_nondominated as _moo_is_nd
except ImportError:
    _moo_is_nd = None

# --- MC HV configuration --------------------------------------------------
MC_SAMPLES_6OBJ = 50000
MC_SEED = 12345

# Cap on archive size per agent. Larger archives are subsampled by
# crowding distance (via PQL._subsample_nd) BEFORE rollouts — this
# avoids wasted env replay on policies we'd drop. Set to None to
# disable subsampling. Recommended: 1000 for robust runs where multi
# pools 5 agents (5000 total before ND).
MAX_POLICIES_PER_AGENT = None


# --- HV -------------------------------------------------------------------
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


# --- Front-quality metric -------------------------------------------------
def _spacing_norm(F, nadir, ideal):
    """L1 nearest-neighbour spacing normalised by box L1-diagonal.
    Lower = more uniform. Complements HV: HV rewards extent and
    convergence; spacing_norm rewards even point distribution. A
    sparse front of corner points can score high HV but high (bad)
    spacing_norm. Returns nan for fronts with < 2 points."""
    F = np.asarray(F, dtype=float)
    if len(F) < 2:
        return float('nan')
    diff = np.abs(F[:, None, :] - F[None, :, :]).sum(axis=2)
    np.fill_diagonal(diff, np.inf)
    d_nn = diff.min(axis=1)
    box_diag_l1 = float(np.sum(ideal - nadir))
    if box_diag_l1 <= 0:
        return float('nan')
    return float(np.std(d_nn) / box_diag_l1)


# --- ND filters -----------------------------------------------------------
def _nd_filter_min(values):
    """Keep non-dominated rows under MINIMISATION."""
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


def _nd_filter_max(F_max):
    """Keep non-dominated rows under MAXIMISATION."""
    if len(F_max) <= 1:
        return F_max
    mask = _moo_is_nd(-F_max)
    return F_max[mask]


# --- Path utility ---------------------------------------------------------
def _seed_idx(p):
    m = re.search(r'seed(\d+)', p)
    return int(m.group(1)) if m else -1
