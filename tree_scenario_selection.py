"""
tree_scenario_selection.py
──────────────────────────
Fruit-tree counterpart of lake_scenario_selection.py: selects K reference
scenarios (slip-pattern indices) for multi-scenario optimisation following
Eker & Kwakkel (2018) / Bartholomew & Kwakkel (2020).

  Stage 1  Take the policies optimal in the default scenario (or N_pol
           random policies); evaluate each under every candidate
           slip pattern; keep the policy-relevant experiments (all outcomes
           in the undesired half, by mean or median).
  Stage 2  Normalise outcomes and choose the K experiments with maximum
           Carlsen et al. (2016) diversity (w = 0.5), exhaustively.
  Stage 3  Report the slip-pattern indices of the chosen experiments.

Fruit-tree specifics
  - Candidate scenarios: the training slip patterns (the uncertainty space
    is IntegerParameter('scenario_index', 0, tree_n_scenarios-1)). Pattern 0
    is slip-free, i.e. the default scenario, which is already part of the
    multi-scenario set, so it is excluded from the candidates.
  - Because there are only 49 candidates, several experiments can share a
    scenario; the K chosen experiments are required to have distinct
    scenarios.
  - Outcomes are the accumulated value vectors (maximised), so "undesired"
    means below the mean/median.
  - Test cases: n_obj ∈ {2, 6} × policy kind ∈ {inter, table-observable,
    table-non-observable}. Open-loop (intertemporal) policies do not depend
    on observability.

Usage
  python tree_scenario_selection.py                 # all test cases
  python tree_scenario_selection.py --n-obj 2 --kind inter
"""

import argparse

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist

from fruit_tree import FruitTreeEnv
from params_config import tree_depth, slip_patterns_path
from lake_scenario_selection import filter_mean, filter_median, normalise_outcomes

N_INTERNAL = 2 ** tree_depth - 1
KINDS = {  # policy kind -> observe flag used by the env
    "inter": True,
    "table-observable": True,
    "table-non-observable": False,
}


# ── Stage 1 ───────────────────────────────────────────────────────────────────

def sample_policies(kind, n_policies, rng):
    n_levers = tree_depth if kind == "inter" else N_INTERNAL
    return rng.integers(0, 2, size=(n_policies, n_levers))


def pareto_policies(kind, num_obj, rng):
    """Policies optimal in the default (slip-free) scenario: the paths to the
    ground-truth Pareto-optimal leaves, i.e. what multi-objective
    optimisation in the default scenario converges to (B&K 2020 re-evaluate
    such policies). For table policies, entries off the path are never
    visited in the default scenario and are filled at random; paths that a
    non-observable table cannot represent consistently are skipped."""
    df = pd.read_csv(f"./trees/depth{tree_depth}_dim{num_obj}.csv")
    leaves = df.loc[df.ground_truth_pareto == 1, "leaf_index"].values
    paths = [[(int(p) >> (tree_depth - 1 - i)) & 1 for i in range(tree_depth)]
             for p in leaves]
    if kind == "inter":
        return np.array(paths)
    observe = KINDS[kind]
    tables = []
    for path in paths:
        table = rng.integers(0, 2, size=N_INTERNAL)
        fixed = {}
        h = p = 0
        for a in path:
            node = (2 ** h - 1 + p) if observe else p
            if fixed.setdefault(node, a) != a:
                break
            table[node] = a
            h, p = h + 1, 2 * p + a
        else:
            tables.append(table)
    return np.array(tables)


def run_policy(env, kind, policy, num_obj):
    obs, _ = env.reset()
    total = np.zeros(num_obj)
    for step in range(tree_depth):
        if kind == "inter":
            action = int(policy[step])
        else:  # same table look-up as moea/fruit_tree_moea.py
            level, pos = obs
            action = int(policy[int(2 ** level - 1) + pos])
        obs, r, terminal, _, _ = env.step(action)
        total += r
        if terminal:
            break
    return total


def run_experiments(kind, policies, scenario_ids, num_obj):
    csv_path = f"./trees/depth{tree_depth}_dim{num_obj}.csv"
    rows = []
    for s in scenario_ids:
        env = FruitTreeEnv(depth=tree_depth, reward_dim=num_obj,
                           csv_path=csv_path, observe=KINDS[kind],
                           scenario_index=int(s),
                           slip_patterns_path=slip_patterns_path)
        for p_idx, pol in enumerate(policies):
            out = run_policy(env, kind, pol, num_obj)
            rows.append({"policy_id": p_idx, "scenario_index": int(s),
                         **{f"o{k+1}": out[k] for k in range(num_obj)}})
    return pd.DataFrame(rows)


# ── Stage 2 ───────────────────────────────────────────────────────────────────

def _carlsen(D, idx, w):
    d = D[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]
    return (1 - w) * d.min() + w * d.mean()


def _greedy_distinct(x, groups, k, w):
    """Farthest-point seeding + swap hill climb (same fallback as
    lake_scenario_selection.find_maxdiverse), keeping scenarios distinct."""
    D = squareform(pdist(x))
    i, j = np.unravel_index(np.argmax(np.where(
        groups[:, None] == groups[None, :], -1, D)), D.shape)
    chosen = [int(i), int(j)]
    while len(chosen) < k:
        ok = ~np.isin(groups, groups[chosen])
        min_d = np.where(ok, D[:, chosen].min(1), -1)
        chosen.append(int(np.argmax(min_d)))
    best = _carlsen(D, chosen, w)
    improved = True
    while improved:
        improved = False
        for slot in range(k):
            others = chosen[:slot] + chosen[slot + 1:]
            for cand in np.flatnonzero(~np.isin(groups, groups[others])):
                trial = others[:slot] + [int(cand)] + others[slot:]
                s = _carlsen(D, trial, w)
                if s > best + 1e-12:
                    chosen, best, improved = trial, s, True
    return chosen, best


def find_maxdiverse_distinct(x, groups, k=4, w=0.5, exhaustive_max_n=150):
    """Search over k-subsets of rows of `x` with pairwise-distinct `groups`,
    maximising (1-w)*min + w*mean of pairwise Euclidean distances.
    Exhaustive (vectorised, k = 4) when n <= exhaustive_max_n, otherwise
    greedy + hill climb."""
    n = len(x)
    if k != 4 or n > exhaustive_max_n:
        return _greedy_distinct(x, groups, k, w)
    D = squareform(pdist(x))
    same = groups[:, None] == groups[None, :]
    upper = np.triu(np.ones((n, n), dtype=bool), 1)
    best, best_idx = -np.inf, None
    for a in range(n):
        for b in range(a + 1, n):
            if same[a, b]:
                continue
            # candidate pairs (c, d) with b < c < d, all scenarios distinct
            valid = upper.copy()
            valid[: b + 1, :] = False
            valid[:, : b + 1] = False
            ok = ~(same[a] | same[b])
            valid &= ok[:, None] & ok[None, :] & ~same
            if not valid.any():
                continue
            dab = D[a, b]
            dac, dbc = D[a][:, None], D[b][:, None]
            dad, dbd = D[a][None, :], D[b][None, :]
            stack = np.broadcast_arrays(dab, dac, dad, dbc, dbd, D)
            stack = np.stack(stack)
            score = (1 - w) * stack.min(0) + w * stack.mean(0)
            score[~valid] = -np.inf
            c, d = np.unravel_index(np.argmax(score), score.shape)
            if score[c, d] > best:
                best, best_idx = score[c, d], [a, b, int(c), int(d)]
    return best_idx, best


# ── End-to-end ────────────────────────────────────────────────────────────────

def run(num_obj, kind, filter_type="mean", n_policies=10, k=4, w=0.5,
        seed=42, min_undesired=None, exclude_default=True, verbose=True,
        policy_source="pareto"):
    rng = np.random.default_rng(seed)
    n_patterns = len(np.load(slip_patterns_path))
    scenario_ids = np.arange(1 if exclude_default else 0, n_patterns)

    policies = (pareto_policies(kind, num_obj, rng)
                if policy_source == "pareto"
                else sample_policies(kind, n_policies, rng))
    exp = run_experiments(kind, policies, scenario_ids, num_obj)
    obj_cols = [f"o{i+1}" for i in range(num_obj)]

    if min_undesired is None:
        min_undesired = num_obj
    filt = filter_mean if filter_type == "mean" else filter_median
    mask = filt(exp, obj_cols, min_undesired=min_undesired)
    sub = exp[mask].reset_index(drop=True)
    n_sc = sub["scenario_index"].nunique()
    if verbose:
        print(f"[{num_obj}-obj, {kind}, {len(policies)} {policy_source} "
              f"policies] policy-relevant: {len(sub)}/"
              f"{len(exp)} experiments over {n_sc} scenarios "
              f"(min_undesired={min_undesired}/{num_obj})")
    if n_sc < k:
        raise RuntimeError(f"only {n_sc} distinct scenarios survive the filter")

    x = normalise_outcomes(sub[obj_cols].values)
    idx, score = find_maxdiverse_distinct(
        x, sub["scenario_index"].values, k=k, w=w)
    chosen = sorted(int(s) for s in sub.loc[idx, "scenario_index"])
    if verbose:
        print(f"  reference scenarios: {chosen}  (diversity {score:.4f})")
    return chosen, score, sub


def current_score(sub, num_obj, current, w=0.5):
    """Best diversity attainable using only the current reference scenarios
    (choosing, for each, its most favourable policy-relevant experiment).
    Returns None if any current scenario has no policy-relevant experiment."""
    obj_cols = [f"o{i+1}" for i in range(num_obj)]
    keep = sub["scenario_index"].isin(current).values
    if sub.loc[keep, "scenario_index"].nunique() < len(current):
        return None
    x = normalise_outcomes(sub[obj_cols].values)[keep]
    _, s = find_maxdiverse_distinct(
        x, sub.loc[keep, "scenario_index"].values, k=len(current), w=w)
    return s


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    p.add_argument("--n-obj", type=int, choices=[2, 6], default=None)
    p.add_argument("--kind", choices=list(KINDS), default=None)
    p.add_argument("--filter", choices=["mean", "median"], default="mean")
    p.add_argument("--policies", choices=["pareto", "random"],
                   default="pareto",
                   help="pareto: policies optimal in the default scenario "
                        "(as in B&K 2020); random: random policies (as in "
                        "lake_scenario_selection.py; uninformative for the fruit "
                        "tree, since slips only re-randomise random actions)")
    p.add_argument("--n-policies", type=int, default=10,
                   help="number of random policies (--policies random)")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--w", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-undesired", type=int, default=None)
    args = p.parse_args()

    from params_config import tree_reference_scenarios
    current = [d["scenario_index"] for d in tree_reference_scenarios]

    objs = [args.n_obj] if args.n_obj else [2, 6]
    kinds = [args.kind] if args.kind else list(KINDS)
    for m in objs:
        for kd in kinds:
            mu = args.min_undesired
            while True:
                try:
                    chosen, score, sub = run(m, kd, args.filter,
                                             args.n_policies, args.k, args.w,
                                             args.seed, mu,
                                             policy_source=args.policies)
                    break
                except RuntimeError as e:
                    mu = (m if mu is None else mu) - 1
                    print(f"  {e}; relaxing min_undesired to {mu}")
            cs = current_score(sub, m, current, args.w)
            print("  current " + str(current) + ": " +
                  ("not all policy-relevant" if cs is None
                   else f"diversity {cs:.4f} ({cs / score:.0%} of max)"))
