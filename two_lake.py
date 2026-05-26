import numpy as np
import gymnasium as gym
from scipy.optimize import brentq

LAKE_BINS = np.array([0.0, 0.05, 0.10, 0.20, 0.35, 0.55, 0.75, 0.90, 1.5, 5.0])
REDUCED_EMISSIONS = np.array([0.00, 0.02, 0.04, 0.06, 0.08, 0.10])


class TwoLakeEnv(gym.Env):

    def __init__(
            self,
            # Lake 1 parameters
            b1=0.42,
            q1=2.5,
            # Lake 2 parameters — deliberately different to ensure objective independence
            b2=0.35,
            q2=3.0,
            # Shared inflow parameters
            mean=0.02,
            stdev=0.0017,
            # Economic parameters
            alpha=0.4,
            delta=0.98,
            # Time structure
            total_years=100,
            years_per_action=5,
            # Seed for natural inflow generation — fixed for the lifetime of the env,
            # independent of the episode seed passed to reset()
            inflow_seed1=0,
            inflow_seed2=0,
            Pcrit1=None,
            Pcrit2=None,
            num_obj=2,
            # Number of gym_step bins added to the observation to mitigate
            # state aliasing under tabular Q-learning.
            n_step_bins=5,
    ):
        super().__init__()

        # --- Lake parameters ---
        self.b1, self.q1 = b1, q1
        self.b2, self.q2 = b2, q2

        # --- Inflow parameters ---
        self.mean = mean
        self.stdev = stdev
        # Pre-compute log-normal transform once
        self._ln_mean = np.log(mean ** 2 / np.sqrt(stdev ** 2 + mean ** 2))
        self._ln_sigma = np.sqrt(np.log(1.0 + stdev ** 2 / mean ** 2))

        # --- Economic parameters ---
        self.alpha = alpha
        self.delta = delta

        # --- Time structure ---
        self.total_years = total_years
        self.years_per_action = years_per_action
        self.n_gym_steps = total_years // years_per_action  # e.g. 10 steps
        self.inflow_seed1 = inflow_seed1
        self.inflow_seed2 = inflow_seed2
        self.num_obj = num_obj

        # --- Step-bin configuration ---
        if self.n_gym_steps % n_step_bins != 0:
            raise ValueError(
                f"n_step_bins ({n_step_bins}) must divide n_gym_steps "
                f"({self.n_gym_steps}) evenly."
            )
        self.n_step_bins = n_step_bins
        self._step_bin_size = self.n_gym_steps // n_step_bins

        # --- Critical thresholds (solved once at construction) ---
        self.Pcrit1 = Pcrit1 if Pcrit1 is not None else \
            brentq(lambda x: x ** self.q1 / (1 + x ** self.q1) - self.b1 * x, 0.01, 1.5)
        self.Pcrit2 = Pcrit2 if Pcrit2 is not None else \
            brentq(lambda x: x ** self.q2 / (1 + x ** self.q2) - self.b2 * x, 0.01, 1.5)

        # --- Natural inflows — generated once at construction using seed ---
        inflow_rng1 = np.random.default_rng(self.inflow_seed1)
        self._inflows1 = inflow_rng1.lognormal(
            self._ln_mean, self._ln_sigma, size=total_years)
        inflow_rng2 = np.random.default_rng(self.inflow_seed2)
        self._inflows2 = inflow_rng2.lognormal(
            self._ln_mean, self._ln_sigma, size=total_years)

        # --- Spaces ---
        # Actions: emission level for each lake, held for years_per_action years.
        # 6 discrete levels mapped to REDUCED_EMISSIONS = [0.00, 0.02, ..., 0.10].
        self.action_space = gym.spaces.MultiDiscrete([6, 6])

        # Observations: [P_lake1_bin, P_lake2_bin, gym_step_bin]
        n_bins = len(LAKE_BINS) - 1
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0], dtype=np.int32),
            high=np.array([n_bins - 1, n_bins - 1, n_step_bins - 1], dtype=np.int32),
            dtype=np.int32,
        )

        # Reward space (6 objectives) — informational, used by MORL libraries
        self.reward_space = gym.spaces.Box(
            low=np.full(num_obj, -np.inf, dtype=np.float32),
            high=np.full(num_obj, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

        # Internal state (initialised in reset)
        self.X1 = None
        self.X2 = None
        self.gym_step = None
        # nan sentinel — inertia is not counted for the first gym step,
        # matching EMA workbench where np.diff starts from decisions[1]-decisions[0]
        self.prev_u1 = np.nan
        self.prev_u2 = np.nan

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.X1 = 0.0
        self.X2 = 0.05
        self.gym_step = 0
        self.prev_u1 = np.nan
        self.prev_u2 = np.nan

        return self._obs(), {}

    def step(self, action):
        u1 = REDUCED_EMISSIONS[int(action[0])]
        u2 = REDUCED_EMISSIONS[int(action[1])]

        # --- Simulate years_per_action years of lake dynamics ---
        X1_traj, X2_traj, X1_new, X2_new = self._simulate(u1, u2)

        # --- Update internal state ---
        self.X1 = float(X1_new)
        self.X2 = float(X2_new)

        # --- Compute 6 objectives ---
        # Absolute year indices for discounting
        year_start = self.gym_step * self.years_per_action
        years = np.arange(year_start, year_start + self.years_per_action)
        discount = np.power(self.delta, years)

        utility1 = float(np.sum(self.alpha * u1 * discount))
        utility2 = float(np.sum(self.alpha * u2 * discount))
        # Per-step safety contribution: -max(X) over this step's 5-year window,
        # scaled by years_per_action / total_years.
        max_P1_step = float(np.max(X1_traj)) * self.years_per_action / self.total_years
        max_P2_step = float(np.max(X2_traj)) * self.years_per_action / self.total_years
        inertia1 = (float(not np.isnan(self.prev_u1) and abs(u1 - self.prev_u1) > 0.02)
                    * self.years_per_action / self.total_years)
        inertia2 = (float(not np.isnan(self.prev_u2) and abs(u2 - self.prev_u2) > 0.02)
                    * self.years_per_action / self.total_years)

        rewards = np.array([
            -max_P1_step,
            -max_P2_step,
            utility1,
            utility2,
            -inertia1,
            -inertia2,
        ], dtype=np.float32)

        # --- Advance step counter ---
        self.prev_u1 = u1
        self.prev_u2 = u2
        self.gym_step += 1

        terminated = self.gym_step >= self.n_gym_steps

        if self.num_obj == 2:
            rewards = rewards[[0, 2]]  # (-max_P1_step, utility1)

        return self._obs(), rewards, terminated, False, {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        n_bins = len(LAKE_BINS) - 1
        x1_bin = int(np.clip(np.digitize(self.X1, LAKE_BINS) - 1, 0, n_bins - 1))
        x2_bin = int(np.clip(np.digitize(self.X2, LAKE_BINS) - 1, 0, n_bins - 1))
        # Clip to last bin: at terminal, gym_step == n_gym_steps would overflow.
        step_bin = int(np.clip(self.gym_step // self._step_bin_size,
                               0, self.n_step_bins - 1))
        return np.array([x1_bin, x2_bin, step_bin], dtype=np.int32)

    def _simulate(self, u1: float, u2: float):
        """
        Run years_per_action steps of both lake dynamics, reading
        pre-computed inflows from the episode arrays.
        """
        year_start = self.gym_step * self.years_per_action
        inflows1 = self._inflows1[year_start:year_start + self.years_per_action]
        inflows2 = self._inflows2[year_start:year_start + self.years_per_action]

        n = self.years_per_action
        X1_traj = np.empty(n)
        X2_traj = np.empty(n)

        X1, X2 = self.X1, self.X2
        b1, q1 = self.b1, self.q1
        b2, q2 = self.b2, self.q2

        # First element: pre-step state (X[0]=0 for step 0, X[10] for step 1, ...)
        # This ensures trajectories span X[0..total_years-1], matching EMA's convention.
        X1_traj[0] = X1
        X2_traj[0] = X2

        # n-1 updates stored in trajectory  (inflows[0..n-2])
        for i in range(1, n):
            X1 = ((1 - b1) * X1
                  + X1 ** q1 / (1 + X1 ** q1)
                  + u1
                  + inflows1[i - 1])
            X2 = ((1 - b2) * X2
                  + X2 ** q2 / (1 + X2 ** q2)
                  + u2
                  + inflows2[i - 1])
            X1_traj[i] = X1
            X2_traj[i] = X2

        # Final update for new env state — uses inflows[n-1], not stored in trajectory
        X1_new = ((1 - b1) * X1
                  + X1 ** q1 / (1 + X1 ** q1)
                  + u1
                  + inflows1[n - 1])
        X2_new = ((1 - b2) * X2
                  + X2 ** q2 / (1 + X2 ** q2)
                  + u2
                  + inflows2[n - 1])

        return X1_traj, X2_traj, X1_new, X2_new

    @property
    def unwrapped(self):
        return self
