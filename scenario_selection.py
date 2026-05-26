"""
scenario_selection.py
─────────────────────
Faithful Python port of the original three-stage scenario-selection
procedure from Bartholomew & Kwakkel (2020), as implemented in
github.com/eebart/RobustDecisionSupportComparison/scenario_selection/.

Procedure (verified against the original notebooks):

  Stage 1 (was ScenarioSelection.ipynb):
    a. Sample N_pol random policies from the lever space.
    b. Sample N_sc deep-uncertainty scenarios via LHS.
    c. Run every (policy × scenario) experiment ⇒ N_pol × N_sc outcomes.
    d. Apply a *policy-relevance filter* to the per-experiment outcomes.
       Three options, matching the original notebook:
         · 'mean'   — keep experiments where every outcome is in the
                      undesired half relative to its mean across all
                      experiments.
         · 'median' — same with median (this is the option Eker–Kwakkel
                      recommend, but Bartholomew's thesis primarily uses
                      'prim').
         · 'prim'   — keep experiments inside a hardcoded PRIM box
                      (extracted from a prior scenario-discovery run).

  Stage 2 (was MaxDiverseSelect.py):
    e. Normalise the outcome columns of the filtered experiments to [0, 1]
       per outcome.
    f. For each (M choose K) subset of the filtered experiments compute
       pairwise Euclidean distance over normalised outcomes; aggregate via
       Carlsen et al. (2016)'s weighted min+mean diversity, w=0.5.
    g. Return the K-subset with maximum diversity. Exhaustive search.

  Stage 3 (was ScenarioVisualization.ipynb):
    h. Read the (b1, q1, b2, q2, inflow_seed1, inflow_seed2) uncertainty
       values for the chosen K experiments — these are the K reference
       scenarios for multi-scenario MORDM.

Lake-specific notes:
  - The original was built for a single-lake problem (5 deep uncertainties:
    b, q, mean, stdev, delta). Our two-lake env has 6 (b1, q1, b2, q2,
    inflow_seed1, inflow_seed2). The procedure is otherwise identical.
  - The 6 deep uncertainties drive the env via TwoLakeEnv constructor args.
    Pcrit values are derived from b, q (computed inside TwoLakeEnv if not
    provided), so they don't need to be sampled.

Usage:
  python scenario_selection.py --filter mean --n-scenarios 500 --n-policies 10 --k 4
  python scenario_selection.py --filter prim --prim-box prim_box.json
"""

import argparse
import json
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from two_lake import TwoLakeEnv
from constrained_two_lake import ConstrainedTwoLakeEnv
from params_config import total_years, years_per_action

# ── Lake configuration ────────────────────────────────────────────────────────

UNCERTAINTY_BOUNDS = {
    "b1":           (0.25, 0.45),
    "q1":           (2.5,  4.5),
    "b2":           (0.25, 0.45),
    "q2":           (2.5,  4.5),
    "inflow_seed1": (0,    10000),
    "inflow_seed2": (0,    10000),
}
INT_DIMS = {"inflow_seed1", "inflow_seed2"}

# Lake env constants
TOTAL_YEARS = 100
YEARS_PER_ACTION = 5
N_GYM_STEPS = TOTAL_YEARS // YEARS_PER_ACTION
N_ACTIONS_PER_LEVEL = 6  # 0..5 indexes into REDUCED_EMISSIONS

OUTCOME_NAMES_2OBJ = ["o1", "o2"]
OUTCOME_NAMES_6OBJ = ["o1", "o2", "o3", "o4", "o5", "o6"]


# ── Stage 1a: sample policies (intertemporal) ─────────────────────────────────

def sample_random_intertemporal_policies(n_policies: int,
                                          rng: np.random.Generator) -> np.ndarray:
    """Each policy is N_GYM_STEPS random emission decisions per lake, drawn
    uniformly from {0, 1, …, 5}. Returns shape (n_policies, n_steps, 2).

    Original (cell 4): `sample_levers(model, n_samples=nr_policies)` — i.e.
    samples from EMA Workbench's lever space. For our discrete intertemporal
    formulation this is uniform over the integer lever range.
    """
    return rng.integers(
        low=0, high=N_ACTIONS_PER_LEVEL,
        size=(n_policies, N_GYM_STEPS, 2),
        endpoint=False,
    )


# ── Stage 1b: LHS over deep uncertainties ────────────────────────────────────

def lhs(n_samples: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    """Latin hypercube sample on [0, 1]^n_dims, shape (n_samples, n_dims)."""
    cuts = np.linspace(0, 1, n_samples + 1)
    u = rng.uniform(cuts[:-1], cuts[1:], size=(n_dims, n_samples)).T
    for j in range(n_dims):
        rng.shuffle(u[:, j])
    return u


def generate_scenarios(n: int, rng: np.random.Generator) -> pd.DataFrame:
    u = lhs(n, len(UNCERTAINTY_BOUNDS), rng)
    cols = {}
    for j, (name, (lo, hi)) in enumerate(UNCERTAINTY_BOUNDS.items()):
        if name in INT_DIMS:
            cols[name] = (lo + u[:, j] * (hi - lo + 1)).astype(int)
        else:
            cols[name] = lo + u[:, j] * (hi - lo)
    return pd.DataFrame(cols)


# ── Stage 1c: run experiments (every policy × every scenario) ────────────────

def _run_episode(env, action_seq, num_obj: int) -> np.ndarray:
    env.reset()
    total = np.zeros(num_obj, dtype=np.float64)
    for u1, u2 in action_seq:
        _, r, _, _, _ = env.step(np.array([u1, u2]))
        total += r
    return total


def run_experiments(policies: np.ndarray, scenarios_df: pd.DataFrame,
                    num_obj: int, env_cls=TwoLakeEnv) -> pd.DataFrame:
    """Replay every policy under every scenario and return a long-form
    DataFrame: one row per (policy, scenario) experiment, columns are the
    deep-uncertainty values + outcome values (o1..oN) + indices.
    """
    n_pol = policies.shape[0]
    n_sc = len(scenarios_df)
    rows = []
    for s_idx in range(n_sc):
        s = scenarios_df.iloc[s_idx]
        env = env_cls(
            b1=float(s["b1"]), q1=float(s["q1"]),
            b2=float(s["b2"]), q2=float(s["q2"]),
            inflow_seed1=int(s["inflow_seed1"]),
            inflow_seed2=int(s["inflow_seed2"]),
            num_obj=num_obj,
            total_years=total_years,
            years_per_action=years_per_action,
        )
        for p_idx in range(n_pol):
            outcomes = _run_episode(env, policies[p_idx], num_obj)
            row = {
                "policy_id": p_idx,
                "scenario_id": s_idx,
                "b1": float(s["b1"]), "q1": float(s["q1"]),
                "b2": float(s["b2"]), "q2": float(s["q2"]),
                "inflow_seed1": int(s["inflow_seed1"]),
                "inflow_seed2": int(s["inflow_seed2"]),
            }
            for k in range(num_obj):
                row[f"o{k+1}"] = float(outcomes[k])
            rows.append(row)
    return pd.DataFrame(rows)


# ── Stage 1d: policy-relevance filter ─────────────────────────────────────────

def _undesired_half_mask(experiments, obj_cols, agg_func, min_undesired=None):
    """Return mask where at least `min_undesired` outcomes are in the
    undesired half. Defaults to ALL outcomes (matches the original).
    """
    if min_undesired is None:
        min_undesired = len(obj_cols)
    masks = []
    for col in obj_cols:
        vals = experiments[col].values
        threshold = agg_func(vals)
        masks.append(vals < threshold)
    stacked = np.stack(masks, axis=0)
    return stacked.sum(axis=0) >= min_undesired


def filter_mean(experiments, obj_cols, min_undesired=None):
    """Mean-based filter (notebook cell 6). All outcomes in undesired half
    by default; pass `min_undesired<len(obj_cols)` to relax."""
    return _undesired_half_mask(experiments, obj_cols, np.mean, min_undesired)


def filter_median(experiments, obj_cols, min_undesired=None):
    """Median-based filter (Eker–Kwakkel 2018, notebook cell 8)."""
    return _undesired_half_mask(experiments, obj_cols, np.median, min_undesired)


def filter_prim(experiments: pd.DataFrame, prim_box: dict) -> np.ndarray:
    """PRIM-box filter: keep experiments whose deep-uncertainty values fall
    inside the box defined by `prim_box`.
    """
    if not prim_box:
        raise ValueError("prim_box must be a non-empty dict.")
    masks = []
    for unc_name, (lo, hi) in prim_box.items():
        if unc_name not in experiments.columns:
            raise ValueError(f"Uncertainty {unc_name!r} not in experiments.")
        vals = experiments[unc_name].values
        masks.append((vals >= lo) & (vals <= hi))
    return np.all(np.stack(masks, axis=0), axis=0)


# ── Stage 2: normalise & diversity selection ─────────────────────────────────

def normalise_outcomes(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise each column independently to [0, 1].
    """
    mn, mx = arr.min(axis=0), arr.max(axis=0)
    rng = np.where(mx - mn == 0, 1.0, mx - mn)
    return (arr - mn) / rng


def diversity_score(subset_outcomes: np.ndarray, w: float = 0.5) -> float:
    """Carlsen et al. (2016) weighted diversity: D = (1-w)·min + w·mean of
    pairwise Euclidean distances.
    """
    if len(subset_outcomes) < 2:
        return 0.0
    distances = pdist(subset_outcomes, metric="euclidean")
    return (1 - w) * distances.min() + w * distances.mean()


def find_maxdiverse(outcomes_norm: np.ndarray, k: int, w: float = 0.5,
                    exhaustive_cap: int = 1_000_000):
    """Pick k indices maximising Carlsen diversity over normalised outcomes.
    """
    n = len(outcomes_norm)
    if n < k:
        raise ValueError(f"Need ≥{k} candidates; got {n}.")

    if comb(n, k) <= exhaustive_cap:
        best_score, best_idx = -np.inf, None
        for combo in combinations(range(n), k):
            d = diversity_score(outcomes_norm[list(combo)], w=w)
            if d > best_score:
                best_score, best_idx = d, combo
        return list(best_idx), best_score, "exhaustive"

    # Greedy farthest-point seeding + hill climb (kept identical to
    # the previous version of this script — same fallback strategy).
    pairwise = np.linalg.norm(
        outcomes_norm[:, None, :] - outcomes_norm[None, :, :], axis=-1
    )
    i, j = np.unravel_index(np.argmax(pairwise), pairwise.shape)
    chosen = [int(i), int(j)]
    while len(chosen) < k:
        rest = [x for x in range(n) if x not in chosen]
        min_d = pairwise[rest][:, chosen].min(axis=1)
        chosen.append(int(rest[int(np.argmax(min_d))]))

    best = diversity_score(outcomes_norm[chosen], w=w)
    improved = True
    while improved:
        improved = False
        for slot in range(k):
            for cand in range(n):
                if cand in chosen:
                    continue
                trial = chosen.copy()
                trial[slot] = cand
                d = diversity_score(outcomes_norm[trial], w=w)
                if d > best + 1e-12:
                    chosen, best = trial, d
                    improved = True
    return chosen, best, "greedy"


# ── Stage 3: extract reference scenarios from chosen experiments ──────────────

def extract_reference_scenarios(experiments: pd.DataFrame,
                                 chosen_idx: list) -> pd.DataFrame:
    """For each of the K chosen experiments, return the (b1, q1, b2, q2,
    inflow_seed1, inflow_seed2) values as one row of the output.
    """
    cols = list(UNCERTAINTY_BOUNDS.keys())
    selected = experiments.iloc[chosen_idx][cols].copy().reset_index(drop=True)
    selected.insert(0, "scenario_id", range(1, len(selected) + 1))
    return selected


# ── End-to-end ────────────────────────────────────────────────────────────────

_ENV_CLASSES = {
    "unconstrained": TwoLakeEnv,
    "constrained":   ConstrainedTwoLakeEnv,
}


def run(filter_type: str = "mean",
        prim_box: dict = None,
        n_scenarios: int = 500,
        n_policies: int = 10,
        k: int = 4,
        w: float = 0.5,
        seed: int = 42,
        exhaustive_cap: int = 1_000_000,
        n_obj: int = 6,
        min_undesired: int = None,
        env_type: str = "unconstrained",
        out_path: str = "lake_reference_scenarios.csv"):
    """Three-stage Bartholomew & Kwakkel (2020) scenario selection.

    `filter_type` ∈ {'mean', 'median', 'prim'}. PRIM requires `prim_box`,
    a dict mapping uncertainty name → (lower, upper).

    `min_undesired` — number of outcomes that must be in the undesired half
    for an experiment to count as policy-relevant. Defaults to all (matching
    the original strict filter). With 6 anti-correlated lake objectives the
    strict filter often empties the candidate set; pass e.g.
    `min_undesired=4` to relax.

    `env_type` ∈ {'unconstrained', 'constrained'} selects which environment
    drives the per-experiment outcomes used by the policy-relevance filter
    and the diversity selection. The two problems have materially different
    outcome landscapes — the constrained env carries a violation penalty
    that shifts which scenarios look "hard" — so the reference sets they
    produce are different. Pick the env that matches the problem you'll
    optimize for; pass each env separately to get a reference set per
    problem.

    Every LHS draw from the configured `UNCERTAINTY_BOUNDS` is u=0 feasible
    by construction (the bounds were tightened to eliminate the structurally
    infeasible corner of the canonical box), so no feasibility filtering is
    applied to the candidate pool.
    """
    if env_type not in _ENV_CLASSES:
        raise ValueError(f"env_type must be one of {list(_ENV_CLASSES)}; "
                         f"got {env_type!r}.")
    env_cls = _ENV_CLASSES[env_type]
    rng = np.random.default_rng(seed)

    # Stage 1a-b
    print(f"Stage 1: sampling {n_policies} random intertemporal policies...")
    policies = sample_random_intertemporal_policies(n_policies, rng)

    print(f"  generating {n_scenarios} LHS scenarios over deep uncertainties...")
    scenarios = generate_scenarios(n_scenarios, rng)

    # Stage 1c
    print(f"  running {n_policies}×{n_scenarios} = "
          f"{n_policies*n_scenarios} experiments under {env_cls.__name__}...")
    experiments = run_experiments(policies, scenarios,
                                  num_obj=n_obj, env_cls=env_cls)

    obj_cols = [f"o{k+1}" for k in range(n_obj)]

    # Stage 1c′ (constrained only): drop penalty-affected experiments before
    # the policy-relevance filter. The constrained env applies a uniform
    # −VIOLATION_PENALTY to every axis per violating year, so any infeasible
    # experiment has at least one axis ≤ ≈−8 (=−10+feasible_base_max); the
    # feasible region keeps every axis ≥ ≈−1. The −5 cut sits cleanly in
    # the gap.
    if env_type == "constrained":
        FEASIBILITY_CUT = -5.0
        obj_vals = experiments[obj_cols].values
        feasible_mask = (obj_vals >= FEASIBILITY_CUT).all(axis=1)
        n_drop = int((~feasible_mask).sum())
        print(f"  dropping {n_drop} penalty-affected experiments "
              f"(any outcome < {FEASIBILITY_CUT}); "
              f"{int(feasible_mask.sum())} feasible remain")
        experiments = experiments[feasible_mask].reset_index(drop=True)

    if min_undesired is None:
        min_undesired = n_obj  # strict: all outcomes in undesired half

    # Stage 1d
    print(f"  filtering by '{filter_type}' "
          f"(min_undesired={min_undesired}/{n_obj})...")
    if filter_type == "mean":
        mask = filter_mean(experiments, obj_cols, min_undesired=min_undesired)
    elif filter_type == "median":
        mask = filter_median(experiments, obj_cols, min_undesired=min_undesired)
    elif filter_type == "prim":
        if prim_box is None:
            raise ValueError(
                "filter_type='prim' requires a prim_box dict "
                "(use --prim-box file.json on the CLI)."
            )
        mask = filter_prim(experiments, prim_box)
    else:
        raise ValueError(f"Unknown filter_type: {filter_type!r}")

    M = int(mask.sum())
    print(f"  policy-relevant subset: M = {M} of {len(experiments)} experiments")
    if M < k:
        raise RuntimeError(
            f"M={M} < K={k}. The filter is too restrictive. With 6 anti-"
            f"correlated objectives the strict 'all-outcomes-undesired' "
            f"filter often retains 0 experiments. Try:\n"
            f"  - relax via --min-undesired N (default = n_obj = {n_obj}); "
            f"e.g. --min-undesired {max(1, n_obj-2)}\n"
            f"  - run with --n-obj 2 (selection valid for 6-obj too since "
            f"lakes are independent)\n"
            f"  - use --filter prim with a prior PRIM box"
        )

    filtered = experiments[mask].reset_index(drop=True)

    # Stage 2
    print(f"Stage 2: normalising outcomes and selecting K = {k} maximally "
          f"diverse experiments...")
    outcomes_norm = normalise_outcomes(filtered[obj_cols].values)

    n_combos = comb(M, k)
    mode_str = "exhaustive" if n_combos <= exhaustive_cap else "greedy + hill-climb"
    print(f"  ({M} choose {k}) = {n_combos:,} subsets — {mode_str}")
    chosen_local, score, mode = find_maxdiverse(
        outcomes_norm, k=k, w=w, exhaustive_cap=exhaustive_cap)
    print(f"  Carlsen diversity: {score:.4f}  ({mode})")

    # Stage 3
    print(f"Stage 3: extracting reference scenarios from chosen experiments...")
    ref = extract_reference_scenarios(filtered, chosen_local)
    print("\nSelected reference scenarios:")
    print(ref.to_string(index=False))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ref.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    return ref


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=("Bartholomew & Kwakkel (2020) scenario selection for "
                     "two-lake multi-scenario MORDM. "
                     "Faithful port of "
                     "github.com/eebart/RobustDecisionSupportComparison/"
                     "scenario_selection/."))
    p.add_argument("--filter", choices=["mean", "median", "prim"],
                   default="mean",
                   help="Policy-relevance filter (paper's thesis uses 'prim'; "
                        "our default is 'mean' for cases without prior PRIM).")
    p.add_argument("--prim-box", type=str, default=None,
                   help="JSON file with PRIM box {uncertainty: [lo, hi], ...}. "
                        "Required if --filter prim.")
    p.add_argument("--n-scenarios", type=int, default=5000,
                   help="LHS scenarios over deep uncertainties (paper: 500).")
    p.add_argument("--n-policies", type=int, default=10,
                   help="Random intertemporal policies (paper: 10).")
    p.add_argument("--k", type=int, default=4,
                   help="Number of reference scenarios to select (paper: 4).")
    p.add_argument("--w", type=float, default=0.5,
                   help="Carlsen diversity weight on mean (paper: 0.5).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exhaustive-cap", type=int, default=1_000_000,
                   help="(M choose K) above which to use greedy fallback.")
    p.add_argument("--n-obj", type=int, default=6, choices=[2, 6],
                   help="Number of objectives. 6 (default) gives a selection "
                        "valid for both 2-obj and 6-obj run grids since lakes "
                        "are independent.")
    p.add_argument("--min-undesired", type=int, default=None,
                   help="Minimum number of outcomes that must be in the "
                        "undesired half for an experiment to be policy-"
                        "relevant. Defaults to all (n_obj). Use a smaller "
                        "value to relax the filter when 6-obj joint filter "
                        "empties.")
    p.add_argument("--env-type",
                   choices=["unconstrained", "constrained"],
                   default="unconstrained",
                   help="Problem the references are selected for. "
                        "Unconstrained uses TwoLakeEnv (pure objective "
                        "outcomes); constrained uses ConstrainedTwoLakeEnv "
                        "(penalized outcomes). The two yield different "
                        "reference sets — pick separately per problem.")
    p.add_argument("--out", default=None,
                   help="Output CSV path. Default: "
                        "'lake_reference_scenarios.csv' for "
                        "--env-type unconstrained, "
                        "'constrained_lake_reference_scenarios.csv' "
                        "for --env-type constrained.")
    args = p.parse_args()

    out_path = args.out
    if out_path is None:
        out_path = ("constrained_lake_reference_scenarios.csv"
                    if args.env_type == "constrained"
                    else "lake_reference_scenarios.csv")

    prim_box = None
    if args.prim_box is not None:
        with open(args.prim_box) as f:
            raw = json.load(f)
        prim_box = {k: tuple(v) for k, v in raw.items()}

    run(filter_type=args.filter,
        prim_box=prim_box,
        n_scenarios=args.n_scenarios,
        n_policies=args.n_policies,
        k=args.k,
        w=args.w,
        seed=args.seed,
        exhaustive_cap=args.exhaustive_cap,
        n_obj=args.n_obj,
        min_undesired=args.min_undesired,
        env_type=args.env_type,
        out_path=out_path)
