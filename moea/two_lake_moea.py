import numpy as np
from two_lake import TwoLakeEnv
from constrained_two_lake import ConstrainedTwoLakeEnv


def get_emission(xt, c1, c2, r1, r2, w1):
    """Single-lake cubic RBF policy."""
    rule = w1 * (abs(xt - c1) / r1) ** 3 + (1 - w1) * (abs(xt - c2) / r2) ** 3
    u = float(np.clip(rule, 0.0, 0.10))
    # Snap to nearest of {0.00, 0.02, 0.04, 0.06, 0.08, 0.10}; return the index.
    return int(round(u / 0.02))


def _run_episode(env, actions):
    """Run a full episode given a list of (u1, u2) actions."""
    env.reset()
    total_rewards = np.zeros(env.num_obj, dtype=np.float32)
    for u1, u2 in actions:
        action = np.array([u1, u2], dtype=np.int64)
        _, rewards, _, _, _ = env.step(action)
        total_rewards -= np.array(rewards, dtype=np.float32)
    return {f'o{i + 1}': float(total_rewards[i]) for i in range(env.num_obj)}


def two_lake_inter(num_obj, alpha, delta, total_years, years_per_action, **kwargs):
    n_steps = total_years // years_per_action
    env = TwoLakeEnv(
        alpha=alpha, delta=delta,
        total_years=total_years,
        years_per_action=years_per_action,
        num_obj=num_obj,
    )
    actions = [(kwargs[f'u1_{i}'], kwargs[f'u2_{i}']) for i in range(n_steps)]
    return _run_episode(env, actions)


def two_lake_dps(num_obj, alpha, delta, total_years, years_per_action, **kwargs):
    env = TwoLakeEnv(
        alpha=alpha, delta=delta,
        total_years=total_years,
        years_per_action=years_per_action,
        num_obj=num_obj,
    )
    env.reset()

    # Independent RBF parameters for each lake
    c1_1, c2_1 = kwargs['c1_1'], kwargs['c2_1']
    r1_1, r2_1 = kwargs['r1_1'], kwargs['r2_1']
    w1_1 = kwargs['w1_1']

    c1_2, c2_2 = kwargs['c1_2'], kwargs['c2_2']
    r1_2, r2_2 = kwargs['r1_2'], kwargs['r2_2']
    w1_2 = kwargs['w1_2']

    total_rewards = np.zeros(num_obj, dtype=np.float32)
    for _ in range(env.n_gym_steps):
        X1, X2 = env.X1, env.X2
        u1 = get_emission(X1, c1_1, c2_1, r1_1, r2_1, w1_1)
        u2 = get_emission(X2, c1_2, c2_2, r1_2, r2_2, w1_2)
        _, rewards, _, _, _ = env.step(np.array([u1, u2], dtype=np.int64))
        total_rewards -= np.array(rewards, dtype=np.float32)

    return {f'o{i + 1}': float(total_rewards[i]) for i in range(num_obj)}


def two_lake_inter_robust(num_obj, alpha, delta, total_years, years_per_action,
                          b1=0.42, q1=2.5, b2=0.35, q2=3.0,
                          inflow_seed1=None, inflow_seed2=None,
                          Pcrit1=None, Pcrit2=None, **kwargs):
    n_steps = total_years // years_per_action
    env = TwoLakeEnv(
        b1=b1, q1=q1, b2=b2, q2=q2,
        alpha=alpha, delta=delta,
        total_years=total_years,
        years_per_action=years_per_action,
        inflow_seed1=inflow_seed1,
        inflow_seed2=inflow_seed2,
        Pcrit1=Pcrit1,
        Pcrit2=Pcrit2,
        num_obj=num_obj,
    )
    actions = [(kwargs[f'u1_{i}'], kwargs[f'u2_{i}']) for i in range(n_steps)]
    return _run_episode(env, actions)


def two_lake_dps_robust(num_obj, alpha, delta, total_years, years_per_action,
                        b1=0.42, q1=2.5, b2=0.35, q2=3.0,
                        inflow_seed1=None, inflow_seed2=None,
                        Pcrit1=None, Pcrit2=None, **kwargs):
    env = TwoLakeEnv(
        b1=b1, q1=q1, b2=b2, q2=q2,
        alpha=alpha, delta=delta,
        total_years=total_years,
        years_per_action=years_per_action,
        inflow_seed1=inflow_seed1,
        inflow_seed2=inflow_seed2,
        Pcrit1=Pcrit1,
        Pcrit2=Pcrit2,
        num_obj=num_obj,
    )
    env.reset()

    c1_1, c2_1 = kwargs['c1_1'], kwargs['c2_1']
    r1_1, r2_1 = kwargs['r1_1'], kwargs['r2_1']
    w1_1 = kwargs['w1_1']

    c1_2, c2_2 = kwargs['c1_2'], kwargs['c2_2']
    r1_2, r2_2 = kwargs['r1_2'], kwargs['r2_2']
    w1_2 = kwargs['w1_2']

    total_rewards = np.zeros(num_obj, dtype=np.float32)
    for _ in range(env.n_gym_steps):
        X1, X2 = env.X1, env.X2
        u1 = get_emission(X1, c1_1, c2_1, r1_1, r2_1, w1_1)
        u2 = get_emission(X2, c1_2, c2_2, r1_2, r2_2, w1_2)
        _, rewards, _, _, _ = env.step(np.array([u1, u2], dtype=np.int64))
        total_rewards -= np.array(rewards, dtype=np.float32)

    return {f'o{i + 1}': float(total_rewards[i]) for i in range(num_obj)}


# ================================================================
# Constrained two-lake wrappers — parallel infrastructure
# ================================================================

def _run_constrained_episode(env, actions):
    """Run an intertemporal episode; return MIN-convention objectives."""
    env.reset()
    total_rewards = np.zeros(env.num_obj, dtype=np.float32)
    for u1, u2 in actions:
        action = np.array([u1, u2], dtype=np.int64)
        _, rewards, _, _, _ = env.step(action)
        total_rewards -= np.array(rewards, dtype=np.float32)

    return {f'o{i + 1}': float(total_rewards[i]) for i in range(env.num_obj)}


def _run_constrained_dps_episode(env, c1_1, c2_1, r1_1, r2_1, w1_1,
                                 c1_2, c2_2, r1_2, r2_2, w1_2):
    """Run a DPS episode; return MIN-convention objectives. Penalty stays in."""
    total_rewards = np.zeros(env.num_obj, dtype=np.float32)
    for _ in range(env.n_gym_steps):
        X1, X2 = env.X1, env.X2
        u1 = get_emission(X1, c1_1, c2_1, r1_1, r2_1, w1_1)
        u2 = get_emission(X2, c1_2, c2_2, r1_2, r2_2, w1_2)
        _, rewards, _, _, _ = env.step(np.array([u1, u2], dtype=np.int64))
        total_rewards -= np.array(rewards, dtype=np.float32)

    return {f'o{i + 1}': float(total_rewards[i]) for i in range(env.num_obj)}


def constrained_two_lake_inter(num_obj, alpha, delta, total_years,
                               years_per_action, **kwargs):
    n_steps = total_years // years_per_action
    env = ConstrainedTwoLakeEnv(
        alpha=alpha, delta=delta,
        total_years=total_years,
        years_per_action=years_per_action,
        num_obj=num_obj,
    )
    actions = [(kwargs[f'u1_{i}'], kwargs[f'u2_{i}']) for i in range(n_steps)]
    return _run_constrained_episode(env, actions)


def constrained_two_lake_dps(num_obj, alpha, delta, total_years,
                             years_per_action, **kwargs):
    env = ConstrainedTwoLakeEnv(
        alpha=alpha, delta=delta,
        total_years=total_years,
        years_per_action=years_per_action,
        num_obj=num_obj,
    )
    env.reset()
    return _run_constrained_dps_episode(
        env,
        kwargs['c1_1'], kwargs['c2_1'], kwargs['r1_1'], kwargs['r2_1'], kwargs['w1_1'],
        kwargs['c1_2'], kwargs['c2_2'], kwargs['r1_2'], kwargs['r2_2'], kwargs['w1_2'],
    )


def constrained_two_lake_inter_robust(num_obj, alpha, delta, total_years,
                                      years_per_action,
                                      b1=0.42, q1=2.5, b2=0.35, q2=3.0,
                                      inflow_seed1=None, inflow_seed2=None,
                                      Pcrit1=None, Pcrit2=None, **kwargs):
    n_steps = total_years // years_per_action
    env = ConstrainedTwoLakeEnv(
        b1=b1, q1=q1, b2=b2, q2=q2,
        alpha=alpha, delta=delta,
        total_years=total_years, years_per_action=years_per_action,
        inflow_seed1=inflow_seed1, inflow_seed2=inflow_seed2,
        Pcrit1=Pcrit1, Pcrit2=Pcrit2,
        num_obj=num_obj,
    )
    actions = [(kwargs[f'u1_{i}'], kwargs[f'u2_{i}']) for i in range(n_steps)]
    return _run_constrained_episode(env, actions)


def constrained_two_lake_dps_robust(num_obj, alpha, delta, total_years,
                                    years_per_action,
                                    b1=0.42, q1=2.5, b2=0.35, q2=3.0,
                                    inflow_seed1=None, inflow_seed2=None,
                                    Pcrit1=None, Pcrit2=None, **kwargs):
    env = ConstrainedTwoLakeEnv(
        b1=b1, q1=q1, b2=b2, q2=q2,
        alpha=alpha, delta=delta,
        total_years=total_years, years_per_action=years_per_action,
        inflow_seed1=inflow_seed1, inflow_seed2=inflow_seed2,
        Pcrit1=Pcrit1, Pcrit2=Pcrit2,
        num_obj=num_obj,
    )
    env.reset()
    return _run_constrained_dps_episode(
        env,
        kwargs['c1_1'], kwargs['c2_1'], kwargs['r1_1'], kwargs['r2_1'], kwargs['w1_1'],
        kwargs['c1_2'], kwargs['c2_2'], kwargs['r1_2'], kwargs['r2_2'], kwargs['w1_2'],
    )
