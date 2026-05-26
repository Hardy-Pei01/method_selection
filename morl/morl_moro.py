import os
import re
import time
import numpy as np
import pandas as pd
from fruit_tree import FruitTreeEnv
from two_lake import TwoLakeEnv
from constrained_two_lake import ConstrainedTwoLakeEnv
from morl.pql import PQL
from params_config import slip_patterns_path, lake_scenarios_path, \
    nd_size_cap_lake, nd_update_freq_tree, nd_update_freq_lake, \
    archive_cap_tree, archive_cap_lake, total_years, years_per_action, \
    gamma_tree, gamma_lake, tree_n_scenarios, lake_n_scenarios


class MoroFruitTreeEnv(FruitTreeEnv):
    def __init__(self, depth, reward_dim, csv_path, observe,
                 patterns_path, seed=42):
        self._patterns = np.load(patterns_path)  # load once
        self._scenario_rng = np.random.default_rng(seed)
        # Tracked so the PQL agent can read the active scenario per step.
        self._current_idx = 0
        super().__init__(depth=depth, reward_dim=reward_dim,
                         csv_path=csv_path, observe=observe,
                         scenario_index=0,
                         slip_patterns_path=patterns_path)

    def reset(self, *, seed=None, options=None):
        idx = int(self._scenario_rng.integers(len(self._patterns)))
        self._current_idx = idx
        self._slip_pattern = self._patterns[idx]
        return super().reset(seed=seed, options=options)


class MoroLakeEnv(TwoLakeEnv):
    def __init__(self, n_obj, scenarios_path, seed=42):
        scenarios = np.load(scenarios_path)  # load once
        self._scenarios = scenarios
        self._scenario_rng = np.random.default_rng(seed)
        # Initialise with first scenario
        s = scenarios[0]
        self._current_idx = -1
        super().__init__(
            b1=float(s['b1']), q1=float(s['q1']),
            b2=float(s['b2']), q2=float(s['q2']),
            inflow_seed1=int(s['inflow_seed1']),
            inflow_seed2=int(s['inflow_seed2']),
            Pcrit1=float(s['Pcrit1']),
            Pcrit2=float(s['Pcrit2']),
            num_obj=n_obj,
            total_years=total_years,
            years_per_action=years_per_action,
        )

    def reset(self, *, seed=None, options=None):
        idx = int(self._scenario_rng.integers(len(self._scenarios)))

        # Only regenerate inflows and Pcrit when scenario actually changes
        if idx != self._current_idx:
            self._current_idx = idx
            s = self._scenarios[idx]
            self.b1, self.q1 = float(s['b1']), float(s['q1'])
            self.b2, self.q2 = float(s['b2']), float(s['q2'])
            self.inflow_seed1 = int(s['inflow_seed1'])
            self.inflow_seed2 = int(s['inflow_seed2'])
            inflow_rng1 = np.random.default_rng(self.inflow_seed1)
            self._inflows1 = inflow_rng1.lognormal(
                self._ln_mean, self._ln_sigma, size=self.total_years)
            inflow_rng2 = np.random.default_rng(self.inflow_seed2)
            self._inflows2 = inflow_rng2.lognormal(
                self._ln_mean, self._ln_sigma, size=self.total_years)
            self.Pcrit1 = float(s['Pcrit1'])
            self.Pcrit2 = float(s['Pcrit2'])

        return super().reset(seed=seed, options=options)


def _get_depth(csv_path):
    """Extract tree depth from a CSV filename containing 'depth{d}'."""
    m = re.search(r'depth(\d+)', csv_path)
    if m:
        return int(m.group(1))
    raise ValueError(f'Cannot infer tree depth from csv_path: {csv_path}')


# ── Main runner ───────────────────────────────────────────────────────────────

def run_moro(
        scoring,
        timesteps,
        ref_point,
        n_obj,
        csv_path,
        num_weight_divisions,
        output_folder,
        file_end,
        start_time=None,
        seed=None,
):
    os.makedirs(output_folder, exist_ok=True)

    depth = _get_depth(csv_path)
    env_kwargs = dict(
        depth=depth, reward_dim=n_obj,
        csv_path=csv_path, observe=True,
        patterns_path=slip_patterns_path,
    )
    if seed is not None:
        env_kwargs['seed'] = seed
    env = MoroFruitTreeEnv(**env_kwargs)

    agent = PQL(
        env=env,
        ref_point=ref_point,
        gamma=gamma_tree,
        initial_epsilon=1.0,
        epsilon_decay_steps=timesteps,
        final_epsilon=0.05,
        seed=seed,
        num_weight_divisions=num_weight_divisions,
        nd_update_freq=nd_update_freq_tree,
        robust=True,
        n_scenarios=tree_n_scenarios,
        max_nd_size=None,
        max_archive_size=archive_cap_tree,
        verbose=True,
        tag=file_end,
    )

    pcs, conv_log = agent.train(
        total_timesteps=timesteps,
        action_eval=scoring,
        log_every=max(1, timesteps // 10),
    )

    # ── Build convergence dataframe ───────────────────────────────────────
    conv_df = pd.DataFrame(conv_log)
    if start_time is not None:
        elapsed = int(time.time() - start_time)
        conv_df['time'] = time.strftime('%H:%M:%S', time.gmtime(elapsed))

    # ── Persist ───────────────────────────────────────────────────────────
    # Convergence trace + Q-table. Decision sequences are not saved here;
    # they can be re-extracted from the agent on demand, and robust
    # re-evaluation is done post-training by evaluation/tree_morl_reeval.py.
    conv_df.to_csv(f'{output_folder}/convergence_{file_end}.csv', index=False)
    agent.save_q_table(f'{output_folder}/agent_{file_end}.pkl')

    return len(agent.archive)


def run_moro_lake(
        scoring, timesteps, ref_point, n_obj,
        num_weight_divisions,
        output_folder, file_end, start_time=None,
        seed=None,
):
    os.makedirs(output_folder, exist_ok=True)

    env_kwargs = dict(n_obj=n_obj, scenarios_path=lake_scenarios_path)
    if seed is not None:
        env_kwargs['seed'] = seed
    env = MoroLakeEnv(**env_kwargs)

    agent = PQL(
        env=env, ref_point=ref_point, gamma=gamma_lake,
        initial_epsilon=1.0, epsilon_decay_steps=timesteps,
        final_epsilon=0.1, seed=seed,
        num_weight_divisions=num_weight_divisions,
        nd_update_freq=nd_update_freq_lake, robust=True,
        n_scenarios=lake_n_scenarios,
        max_nd_size=nd_size_cap_lake,
        max_archive_size=archive_cap_lake,
        verbose=True,
        tag=file_end,
    )

    pcs, conv_log = agent.train(
        total_timesteps=timesteps, action_eval=scoring,
        log_every=max(1, timesteps // 10),
    )

    conv_df = pd.DataFrame(conv_log)
    if start_time is not None:
        elapsed = int(time.time() - start_time)
        conv_df['time'] = time.strftime('%H:%M:%S', time.gmtime(elapsed))

    # ── Persist ───────────────────────────────────────────────────────────
    # Convergence trace + Q-table. Decision sequences are not saved here;
    # they can be re-extracted from the agent on demand, and robust
    # re-evaluation is done post-training by evaluation/lake_morl_reeval.py.
    conv_df.to_csv(f'{output_folder}/convergence_{file_end}.csv', index=False)
    agent.save_q_table(f'{output_folder}/agent_{file_end}.pkl')

    return len(agent.archive)


# ================================================================
# Constrained two-lake MORO — parallel infrastructure
# ================================================================

class ConstrainedMoroLakeEnv(ConstrainedTwoLakeEnv):
    def __init__(self, n_obj, scenarios_path, seed=42):
        scenarios = np.load(scenarios_path)
        self._scenarios = scenarios
        self._scenario_rng = np.random.default_rng(seed)
        s = scenarios[0]
        self._current_idx = -1
        super().__init__(
            b1=float(s['b1']), q1=float(s['q1']),
            b2=float(s['b2']), q2=float(s['q2']),
            inflow_seed1=int(s['inflow_seed1']),
            inflow_seed2=int(s['inflow_seed2']),
            Pcrit1=float(s['Pcrit1']),
            Pcrit2=float(s['Pcrit2']),
            num_obj=n_obj,
            total_years=total_years,
            years_per_action=years_per_action,
        )

    def reset(self, *, seed=None, options=None):
        idx = int(self._scenario_rng.integers(len(self._scenarios)))
        if idx != self._current_idx:
            self._current_idx = idx
            s = self._scenarios[idx]
            self.b1, self.q1 = float(s['b1']), float(s['q1'])
            self.b2, self.q2 = float(s['b2']), float(s['q2'])
            self.inflow_seed1 = int(s['inflow_seed1'])
            self.inflow_seed2 = int(s['inflow_seed2'])
            inflow_rng1 = np.random.default_rng(self.inflow_seed1)
            self._inflows1 = inflow_rng1.lognormal(
                self._ln_mean, self._ln_sigma, size=self.total_years)
            inflow_rng2 = np.random.default_rng(self.inflow_seed2)
            self._inflows2 = inflow_rng2.lognormal(
                self._ln_mean, self._ln_sigma, size=self.total_years)
            self.Pcrit1 = float(s['Pcrit1'])
            self.Pcrit2 = float(s['Pcrit2'])

        return super().reset(seed=seed, options=options)


def run_moro_lake_constrained(
        scoring, timesteps, ref_point, n_obj,
        num_weight_divisions,
        output_folder, file_end, start_time=None,
        seed=None,
):
    os.makedirs(output_folder, exist_ok=True)

    env_kwargs = dict(n_obj=n_obj, scenarios_path=lake_scenarios_path)
    if seed is not None:
        env_kwargs['seed'] = seed
    env = ConstrainedMoroLakeEnv(**env_kwargs)

    agent = PQL(
        env=env, ref_point=ref_point, gamma=gamma_lake,
        initial_epsilon=1.0, epsilon_decay_steps=timesteps,
        final_epsilon=0.1, seed=seed,
        num_weight_divisions=num_weight_divisions,
        nd_update_freq=nd_update_freq_lake, robust=True,
        n_scenarios=lake_n_scenarios,
        max_nd_size=nd_size_cap_lake,
        max_archive_size=archive_cap_lake,
        verbose=True,
        tag=file_end,
    )

    pcs, conv_log = agent.train(
        total_timesteps=timesteps, action_eval=scoring,
        log_every=max(1, timesteps // 10),
    )

    conv_df = pd.DataFrame(conv_log)
    if start_time is not None:
        elapsed = int(time.time() - start_time)
        conv_df['time'] = time.strftime('%H:%M:%S', time.gmtime(elapsed))

    conv_df.to_csv(f'{output_folder}/convergence_{file_end}.csv', index=False)
    agent.save_q_table(f'{output_folder}/agent_{file_end}.pkl')

    return len(agent.archive)