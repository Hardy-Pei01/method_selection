"""Pareto Q-Learning."""
import gzip
import pickle
import numbers
from itertools import product
from typing import Callable
from collections import defaultdict

import gymnasium as gym
import numpy as np

from morl_baselines.common.morl_algorithm import MOAgent
from morl_baselines.common.performance_indicators import hypervolume
from morl_baselines.common.utils import linearly_decaying_value
from moocore import is_nondominated as _moocore_is_nd


# Faster pareto pruning via moocore
def _nd_mask(arr: np.ndarray) -> np.ndarray:
    """Boolean mask: True for non-dominated rows under maximisation."""
    if arr.shape[0] <= 1:
        return np.ones(arr.shape[0], dtype=bool)
    # moocore is min-convention; negate for max
    return _moocore_is_nd(-arr)


def get_non_dominated(candidates):
    """Drop-in replacement for morl_baselines.get_non_dominated."""
    if not candidates:
        return set()
    arr = np.asarray(list(candidates), dtype=float)
    if arr.shape[0] == 1:
        return {tuple(arr[0])}
    mask = _nd_mask(arr)
    return {tuple(arr[i]) for i in range(arr.shape[0]) if mask[i]}


def _additive_epsilon(A: np.ndarray, B: np.ndarray) -> float:
    """I_eps+(A, B) under MAXIMISATION: smallest eps s.t. every b in B is
    eps-dominated by some a in A. 0 if B empty; inf if A empty & B non-empty.

    Can be negative when every b is strictly dominated by some a (the later
    front shrank inside the earlier one) — this is intentional and lets the
    symmetric metric pass through zero rather than flooring at convergence.
    """
    if B.shape[0] == 0:
        return 0.0
    if A.shape[0] == 0:
        return np.inf
    diff = B[:, None, :] - A[None, :, :]      # b_i - a_i
    return float(diff.max(axis=2).min(axis=1).max())


def _additive_epsilon_mean(A: np.ndarray, B: np.ndarray) -> float:
    """Mean-aggregated companion to _additive_epsilon.

    Identical per-point computation — for each b in B, min_a max_o (b_o - a_o)
    — but the final reduction over front points is a mean rather than a max.
    This yields a bulk-sensitive, lower-variance signal that a few wandering
    tail points cannot dominate. Same empty/inf conventions as the max form.
    """
    if B.shape[0] == 0:
        return 0.0
    if A.shape[0] == 0:
        return np.inf
    diff = B[:, None, :] - A[None, :, :]      # b_i - a_i
    return float(diff.max(axis=2).min(axis=1).mean())


def pcs_shift(prev: set, curr: set, d: int) -> float:
    """Symmetric additive epsilon-indicator between two Pareto sets.
    """
    A = np.array(list(prev), float) if prev else np.empty((0, d))
    B = np.array(list(curr), float) if curr else np.empty((0, d))
    return max(_additive_epsilon(A, B), _additive_epsilon(B, A))


def pcs_shift_mean(prev: set, curr: set, d: int) -> float:
    """Symmetric mean-aggregated additive epsilon-indicator.

    Bulk-sensitive companion to pcs_shift: same per-point epsilon, averaged
    over front points rather than worst-cased. Reported alongside pcs_shift so
    convergence can be read both ways — max stays high if any tail point
    wanders; mean settles toward zero once the bulk of the front stabilises.
    """
    A = np.array(list(prev), float) if prev else np.empty((0, d))
    B = np.array(list(curr), float) if curr else np.empty((0, d))
    return max(_additive_epsilon_mean(A, B), _additive_epsilon_mean(B, A))


def generate_weights(num_objectives: int, num_divisions: int) -> np.ndarray:
    """Generate evenly spaced weight vectors on the unit simplex."""
    combs = product(range(num_divisions + 1), repeat=num_objectives)
    weights = np.array([c for c in combs if sum(c) == num_divisions])
    weights = weights / num_divisions
    # Avoid zero weights — replace with small epsilon for Chebyshev stability
    weights = np.maximum(weights, 1e-4)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return weights


class PQL(MOAgent):

    def __init__(
            self,
            env,
            ref_point,
            gamma=0.8,
            initial_epsilon=1.0,
            epsilon_decay_steps=100000,
            final_epsilon=0.1,
            seed=None,
            num_weight_divisions=5,
            nd_update_freq=1,
            robust=False,
            n_scenarios=None,
            max_nd_size=None,
            max_archive_size=None,
            verbose=False,
            tag="",
    ):

        super().__init__(env, seed=seed)

        # Learning parameters
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.final_epsilon = final_epsilon
        self.ref_point = ref_point
        self.robust = robust
        # Number of scenarios in the robust-MORO ensemble.
        self.n_scenarios = n_scenarios
        if self.robust and self.n_scenarios is None:
            raise ValueError(
                "PQL(robust=True, ...) requires n_scenarios=<count>"
            )
        self.max_nd_size = max_nd_size
        # Optional cap on the returned start-state Pareto front.
        self.max_archive_size = max_archive_size
        self.verbose = verbose
        self.tag = tag

        # Action space
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.num_actions = self.env.action_space.n
        elif isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
            self.num_actions = np.prod(self.env.action_space.nvec)
        else:
            raise Exception("PQL only supports (multi)discrete action spaces.")

        # Observation space
        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            self.env_shape = (self.env.observation_space.n,)
        elif isinstance(self.env.observation_space, gym.spaces.MultiDiscrete):
            self.env_shape = self.env.observation_space.nvec
        elif (
                isinstance(self.env.observation_space, gym.spaces.Box)
                and self.env.observation_space.is_bounded(manner="both")
                and issubclass(self.env.observation_space.dtype.type, numbers.Integral)
        ):
            low_bound = np.array(self.env.observation_space.low)
            high_bound = np.array(self.env.observation_space.high)
            self.env_shape = high_bound - low_bound + 1
        else:
            raise Exception("PQL only supports discretizable observation spaces.")

        self.num_states = np.prod(self.env_shape)
        self.num_objectives = self.env.unwrapped.reward_space.shape[0]

        # Q-learning data structures for pareto and indicator variants.
        # non_dominated[s][a]: full Pareto-pruned set of Q-vectors.
        self.counts = defaultdict(lambda: np.zeros(self.num_actions))

        self.non_dominated = defaultdict(
            lambda: [{tuple(np.zeros(self.num_objectives))} for _ in range(self.num_actions)]
        )
        self.avg_reward = defaultdict(
            lambda: np.zeros((self.num_actions, self.num_objectives))
        )
        if self.robust:
            # Per-scenario running mean of immediate rewards at each (s, a),
            # plus a per-scenario visit count. Aggregation across scenarios
            # uses np.percentile(20, axis=0) over the visited subset.
            n_s = self.n_scenarios
            n_a = self.num_actions
            n_o = self.num_objectives
            self.scenario_reward_means = defaultdict(
                lambda: np.zeros((n_s, n_a, n_o))
            )
            self.scenario_visit_counts = defaultdict(
                lambda: np.zeros((n_s, n_a), dtype=np.int64)
            )
        self.nd_decomp = defaultdict(
            lambda: [{tuple(np.zeros(self.num_objectives))} for _ in range(self.num_actions)]
        )

        # Decomposition structures.
        self.weights = generate_weights(self.num_objectives, num_weight_divisions)
        # Cursor into self.weights.
        self._weight_cursor = 0
        # Initialise to -inf so the first update always takes effect.
        self.ideal_point = np.full(self.num_objectives, -np.inf)
        self.nd_update_freq = nd_update_freq
        self.archive = set()

    # ------------------------------------------------------------------
    # Ideal point
    # ------------------------------------------------------------------

    def _update_ideal_point_global(self, decomp: bool = False):
        """Update the ideal point globally from Q-sets across all visited
        state-action pairs.
        """
        for s in self.counts:
            for a in range(self.num_actions):
                if self.counts[s][a] == 0:
                    continue
                qa = self.get_q_array(s, a, decomp=decomp)
                if qa.shape[0]:
                    self.ideal_point = np.maximum(self.ideal_point,
                                                  qa.max(axis=0))

    # ------------------------------------------------------------------
    # Scoring functions
    # ------------------------------------------------------------------

    def score_pareto_cardinality(self, state: int):
        """Pareto-based scoring: count non-dominated contributions per action."""
        q_arrays = []
        action_of_row = []
        for a in range(self.num_actions):
            qa = self.get_q_array(state, a)
            if qa.shape[0] == 0:
                continue
            q_arrays.append(qa)
            action_of_row.extend([a] * qa.shape[0])
        if not q_arrays:
            return np.ones(self.num_actions)
        Q = np.vstack(q_arrays)
        action_of_row = np.array(action_of_row)
        mask = _nd_mask(Q)
        scores = np.zeros(self.num_actions)
        if mask.any():
            np.add.at(scores, action_of_row[mask], 1)
        return scores

    def score_hypervolume(self, state: int):
        """Indicator-based scoring: hypervolume contribution per action."""
        scores = np.zeros(self.num_actions)
        for a in range(self.num_actions):
            scores[a] = hypervolume(self.ref_point,
                                    self.get_q_array(state, a))
        return scores

    def score_decomposition(self, state: int):
        """Decomposition-based scoring"""
        q_arrays = []
        action_of_row = []
        for a in range(self.num_actions):
            if self.counts[state][a] == 0:
                continue
            qa = self.get_q_array(state, a, decomp=True)
            if qa.shape[0] == 0:
                continue
            q_arrays.append(qa)
            action_of_row.extend([a] * qa.shape[0])
        if not q_arrays:
            return np.ones(self.num_actions)
        Q = np.vstack(q_arrays)                         # (V, d)
        action_of_row = np.array(action_of_row)         # (V,)
        dev = self.ideal_point - Q                      # (V, d)

        # Pick the next subproblem (weight) — round-robin cursor.
        w_i = self._weight_cursor
        self._weight_cursor = (self._weight_cursor + 1) % len(self.weights)
        w = self.weights[w_i]                           # (d,)

        # Chebyshev scalarisation: smaller = better for this subproblem.
        scalarised = np.max(w * dev, axis=1)            # (V,)
        best_row = int(np.argmin(scalarised))

        # One-hot: only the action owning the winning row gets +1.
        scores = np.zeros(self.num_actions)
        scores[action_of_row[best_row]] = 1.0
        return scores

    # ------------------------------------------------------------------
    # Shared Q-learning machinery
    # ------------------------------------------------------------------

    def get_q_set(self, state: int, action: int, decomp: bool = False):
        nd = self.nd_decomp[state][action] if decomp else self.non_dominated[state][action]
        nd_array = np.array(list(nd))
        q_array = self.avg_reward[state][action] + self.gamma * nd_array
        return {tuple(vec) for vec in q_array}

    def get_q_array(self, state: int, action: int, decomp: bool = False) -> np.ndarray:
        nd = self.nd_decomp[state][action] if decomp else self.non_dominated[state][action]
        if not nd:
            return np.empty((0, self.num_objectives))
        nd_array = np.array(list(nd))
        return self.avg_reward[state][action] + self.gamma * nd_array

    def select_action(self, state: int, score_func: Callable):
        if self.np_random.uniform(0, 1) < self.epsilon:
            return self.np_random.integers(self.num_actions)
        else:
            action_scores = score_func(state)
            return self.np_random.choice(
                np.argwhere(action_scores == np.max(action_scores)).flatten()
            )

    def _subsample_nd(self, nd: set, target_size: int = None) -> set:
        """Subsample a non-dominated set to target_size via crowding
        distance.
        """
        if target_size is None:
            target_size = self.max_nd_size
        if target_size is None or len(nd) <= target_size:
            return nd
        arr = np.array(list(nd))
        n, d = arr.shape
        # Canonical row order: depends only on the set contents.
        canon = np.lexsort(tuple(arr[:, c] for c in range(d - 1, -1, -1)))
        arr = arr[canon]
        crowd = np.zeros(n)
        for obj in range(d):
            order = np.argsort(arr[:, obj], kind='stable')
            crowd[order[0]] = crowd[order[-1]] = np.inf
            obj_range = arr[order[-1], obj] - arr[order[0], obj]
            if obj_range == 0:
                continue
            s = arr[order, obj]
            crowd[order[1:-1]] += (s[2:] - s[:-2]) / obj_range
        # Keep largest crowding; ties broken by canonical coordinates.
        keys = tuple(arr[:, c] for c in range(d - 1, -1, -1)) + (crowd,)
        keep = np.lexsort(keys)[-target_size:]
        return {tuple(arr[i]) for i in keep}

    def calc_non_dominated(self, state: int):
        # Stack per-action Q-arrays and prune directly.
        parts = []
        for a in range(self.num_actions):
            qa = self.get_q_array(state, a)
            if qa.shape[0]:
                parts.append(qa)
        if not parts:
            return {tuple(np.zeros(self.num_objectives))}
        Q = np.vstack(parts)
        if Q.shape[0] == 1:
            return {tuple(Q[0])}
        mask = _nd_mask(Q)
        nd = {tuple(Q[i]) for i in range(Q.shape[0]) if mask[i]}
        if self.max_nd_size is not None:
            return self._subsample_nd(nd)
        return nd

    def calc_decomp_best(self, state: int) -> set:
        q_vecs = []
        for a in range(self.num_actions):
            if self.counts[state][a] == 0:
                continue
            q_vecs.extend(self.get_q_set(state, a, decomp=True))
        if not q_vecs:
            return {tuple(np.zeros(self.num_objectives))}

        Q = np.array(q_vecs, dtype=float)
        dev = self.ideal_point - Q
        # scalarised[w, p] = max_o ( weights[w, o] * dev[p, o] )
        # Running max over the d objectives avoids materialising the full
        # (n_weights, P, d) product.
        scal = np.full((self.weights.shape[0], Q.shape[0]), -np.inf)
        for o in range(self.weights.shape[1]):
            np.maximum(scal, np.outer(self.weights[:, o], dev[:, o]),
                       out=scal)
        best_idx = np.argmin(scal, axis=1)
        global_set = {tuple(Q[i]) for i in best_idx}
        return global_set if global_set else {tuple(np.zeros(self.num_objectives))}

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
            self,
            total_timesteps: int,
            action_eval: str = "indicator",
            log_every: int = 1000,
    ):
        if action_eval == "indicator":
            score_func = self.score_hypervolume
        elif action_eval == "pareto":
            score_func = self.score_pareto_cardinality
        elif action_eval == "decomposition":
            score_func = self.score_decomposition
        else:
            raise Exception(
                f"Unknown action_eval: '{action_eval}'. "
                f"Choose from 'pareto', 'indicator', 'decomposition'."
            )

        is_decomp = (action_eval == "decomposition")
        self.action_eval = action_eval

        # Determine the canonical start state (where Bellman bootstrap
        # propagates Q-values back to). For envs whose reset doesn't yield
        # state index 0, hardcoding state=0 would give an empty front.
        _reset_obs, _ = self.env.reset()
        self._start_state = int(np.ravel_multi_index(_reset_obs, self.env_shape))

        convergence_log = []
        next_log_step = log_every
        prev_pcs = None

        while self.global_step < total_timesteps:
            state, _ = self.env.reset()
            state = int(np.ravel_multi_index(state, self.env_shape))
            terminated = False
            truncated = False

            while not (terminated or truncated) and self.global_step < total_timesteps:
                action = self.select_action(state, score_func)
                if isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
                    action_nd = np.unravel_index(action, self.env.action_space.nvec)
                    next_state, reward, terminated, truncated, _ = self.env.step(action_nd)
                else:
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.global_step += 1
                next_state = int(np.ravel_multi_index(next_state, self.env_shape))

                self.counts[state][action] += 1

                if self.global_step % self.nd_update_freq == 0:
                    if terminated or truncated:
                        zero = {tuple(np.zeros(self.num_objectives))}
                        if is_decomp:
                            self.nd_decomp[state][action] = zero
                        else:
                            self.non_dominated[state][action] = zero
                    elif is_decomp:
                        self.nd_decomp[state][action] = self.calc_decomp_best(next_state)
                    else:
                        self.non_dominated[state][action] = self.calc_non_dominated(next_state)

                if self.robust:
                    # Per-scenario running mean: update only the active
                    # scenario's slot at this (s, a). Cheap O(d) update.
                    sc_idx = int(self.env.unwrapped._current_idx)
                    self.scenario_visit_counts[state][sc_idx, action] += 1
                    n = self.scenario_visit_counts[state][sc_idx, action]
                    self.scenario_reward_means[state][sc_idx, action] += (
                        (reward - self.scenario_reward_means[state][sc_idx, action]) / n
                    )
                    if self.global_step % self.nd_update_freq == 0:
                        # Aggregate across scenarios visited so far at this
                        # cell. Take 20th percentile across the visited-
                        # scenario subset; if only 1 scenario has been seen,
                        # use that scenario's mean directly.
                        visited_mask = self.scenario_visit_counts[state][:, action] > 0
                        if visited_mask.any():
                            visited_means = self.scenario_reward_means[state][visited_mask, action]
                            if visited_means.shape[0] > 1:
                                self.avg_reward[state][action] = np.percentile(
                                    visited_means, 20, axis=0
                                )
                            else:
                                self.avg_reward[state][action] = visited_means[0]
                else:
                    self.avg_reward[state][action] += (
                            (reward - self.avg_reward[state][action]) / self.counts[state][action]
                    )

                state = next_state

                if self.global_step >= next_log_step:
                    pcs = self.get_local_pcs(state=self._start_state, decomp=is_decomp)

                    if prev_pcs is None:
                        shift = None
                        shift_mean = None
                    else:
                        shift = pcs_shift(prev_pcs, pcs, self.num_objectives)
                        shift_mean = pcs_shift_mean(prev_pcs, pcs, self.num_objectives)
                    prev_pcs = set(pcs)

                    convergence_log.append({
                        "timestep": self.global_step,
                        "pcs_size": len(pcs),
                        "pcs_shift_eps": shift,
                        "pcs_shift_eps_mean": shift_mean,
                        "epsilon": self.epsilon,
                    })
                    if self.verbose:
                        prefix = f'[{self.tag}] ' if self.tag else '  '
                        s = '—' if shift is None else f'{shift:.4g}'
                        sm = '—' if shift_mean is None else f'{shift_mean:.4g}'
                        print(
                            f'{prefix}step {self.global_step}/{total_timesteps} '
                            f'|archive|={len(pcs)} Δpcs(max)={s} Δpcs(mean)={sm} '
                            f'eps={self.epsilon:.3f}',
                            flush=True,
                        )
                    next_log_step += log_every

            self._update_ideal_point_global(decomp=is_decomp)

            self.epsilon = linearly_decaying_value(
                self.initial_epsilon,
                self.epsilon_decay_steps,
                self.global_step,
                0,
                self.final_epsilon,
            )

        self.archive = self.get_local_pcs(state=self._start_state, decomp=is_decomp)
        if (self.max_archive_size is not None
                and len(self.archive) > self.max_archive_size):
            self.archive = self._subsample_nd(
                self.archive, target_size=self.max_archive_size
            )
        return self.archive, convergence_log

    # ------------------------------------------------------------------
    # Result extraction
    # ------------------------------------------------------------------

    def get_local_pcs(self, state: int = 0, decomp: bool = False):
        """Return the Pareto coverage set at the given state."""
        q_sets = [
            self.get_q_set(state, a, decomp=decomp)
            for a in range(self.num_actions)
        ]
        candidates = set().union(*q_sets)
        if not candidates:
            return set()
        if len(candidates) == 1:
            return candidates
        return get_non_dominated(candidates)

    def get_config(self):
        env_unwrapped = self.env.unwrapped
        spec = getattr(env_unwrapped, "spec", None)
        env_id = spec.id if spec is not None else type(env_unwrapped).__name__
        return {
            "env_id": env_id,
            "gamma": self.gamma,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "final_epsilon": self.final_epsilon,
            "num_states": self.num_states,
            "num_actions": self.num_actions,
            "num_objectives": self.num_objectives,
            "ref_point": self.ref_point.tolist(),
            "num_weights": len(self.weights),
            "nd_update_freq": self.nd_update_freq,
            "robust": self.robust,
            "max_nd_size": self.max_nd_size,
            "max_archive_size": self.max_archive_size,
        }

    # ------------------------------------------------------------------
    # Q-table persistence
    # ------------------------------------------------------------------

    def save_q_table(self, path: str):
        """Pickle the four Q-learning structures plus archive and metadata."""
        payload = {
            "version": 1,
            "config": self.get_config(),
            "action_eval": getattr(self, "action_eval", None),
            "global_step": self.global_step,
            "ideal_point": self.ideal_point.copy(),
            "archive": set(self.archive),
            "weight_cursor": int(self._weight_cursor),
            "counts": {s: arr.copy() for s, arr in self.counts.items()},
            "non_dominated": {
                s: [set(per_a) for per_a in cells]
                for s, cells in self.non_dominated.items()
            },
            "nd_decomp": {
                s: [set(per_a) for per_a in cells]
                for s, cells in self.nd_decomp.items()
            },
            "avg_reward": {s: arr.copy() for s, arr in self.avg_reward.items()},
        }
        if self.robust and hasattr(self, "scenario_reward_means"):
            payload["scenario_reward_means"] = {
                s: arr.copy()
                for s, arr in self.scenario_reward_means.items()
            }
            payload["scenario_visit_counts"] = {
                s: arr.copy()
                for s, arr in self.scenario_visit_counts.items()
            }
            payload["n_scenarios"] = self.n_scenarios
        with gzip.open(path, "wb", compresslevel=6) as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_q_table(self, path: str):
        """Restore Q-table structures from a save_q_table file."""
        with open(path, "rb") as f:
            magic = f.read(2)
        if magic == b'\x1f\x8b':
            with gzip.open(path, "rb") as f:
                payload = pickle.load(f)
        else:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        if payload.get("version") != 1:
            raise ValueError(f"Unsupported q-table format version: {payload.get('version')}")

        cfg = payload["config"]
        if cfg["num_objectives"] != self.num_objectives:
            raise ValueError(
                f"num_objectives mismatch: saved={cfg['num_objectives']}, "
                f"current={self.num_objectives}"
            )
        if cfg["num_actions"] != self.num_actions:
            raise ValueError(
                f"num_actions mismatch: saved={cfg['num_actions']}, "
                f"current={self.num_actions}"
            )

        self.gamma = float(cfg["gamma"])
        self.initial_epsilon = float(cfg["initial_epsilon"])
        self.epsilon_decay_steps = int(cfg["epsilon_decay_steps"])
        self.final_epsilon = float(cfg["final_epsilon"])
        self.ref_point = np.asarray(cfg["ref_point"])
        self.nd_update_freq = int(cfg["nd_update_freq"])
        self.max_nd_size = cfg["max_nd_size"]
        self.max_archive_size = cfg["max_archive_size"]

        d = self.num_objectives
        a = self.num_actions
        counts_factory = lambda: np.zeros(a)
        nd_factory = lambda: [{tuple(np.zeros(d))} for _ in range(a)]
        avg_factory = lambda: np.zeros((a, d))

        self.counts = defaultdict(counts_factory, payload["counts"])
        self.non_dominated = defaultdict(nd_factory, payload["non_dominated"])
        self.nd_decomp = defaultdict(nd_factory, payload["nd_decomp"])
        self.avg_reward = defaultdict(avg_factory, payload["avg_reward"])

        if "scenario_reward_means" in payload:
            n_s = payload.get("n_scenarios", self.n_scenarios)
            n_a = a
            n_o = d
            srm_factory = lambda: np.zeros((n_s, n_a, n_o))
            svc_factory = lambda: np.zeros((n_s, n_a), dtype=np.int64)
            self.scenario_reward_means = defaultdict(
                srm_factory, payload["scenario_reward_means"]
            )
            self.scenario_visit_counts = defaultdict(
                svc_factory, payload["scenario_visit_counts"]
            )

        self.archive = set(payload["archive"])
        self.ideal_point = np.asarray(payload["ideal_point"]).copy()
        self.global_step = payload["global_step"]
        self.action_eval = payload.get("action_eval")
        # Older pickles may not have weight_cursor; default to 0.
        self._weight_cursor = int(payload.get("weight_cursor", 0))
