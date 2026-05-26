import os
import numpy as np
from scipy.optimize import brentq
from params_config import lake_n_scenarios
from scenario_selection import generate_scenarios

SCENARIO_SEED = 42


def _solve_pcrit(b, q):
    return brentq(lambda x: x ** q / (1 + x ** q) - b * x, 0.01, 1.5)


def generate_lake_scenarios(n_scenarios=lake_n_scenarios, seed=SCENARIO_SEED):
    """Latin-hypercube sample over the lake uncertainty space."""
    rng = np.random.default_rng(seed)
    df = generate_scenarios(n_scenarios, rng)

    dt = np.dtype([
        ('b1', np.float64), ('q1', np.float64),
        ('b2', np.float64), ('q2', np.float64),
        ('inflow_seed1', np.int64), ('inflow_seed2', np.int64),
        ('Pcrit1', np.float64), ('Pcrit2', np.float64),
    ])
    scenarios = np.empty(n_scenarios, dtype=dt)
    for c in ('b1', 'q1', 'b2', 'q2',
              'inflow_seed1', 'inflow_seed2'):
        scenarios[c] = df[c].to_numpy()
    for i in range(n_scenarios):
        scenarios['Pcrit1'][i] = _solve_pcrit(
            scenarios['b1'][i], scenarios['q1'][i])
        scenarios['Pcrit2'][i] = _solve_pcrit(
            scenarios['b2'][i], scenarios['q2'][i])
    return scenarios


def _summarise(scenarios, indices, out_path):
    print(f"Generated {len(scenarios)} lake scenarios -> {out_path}")
    for i in indices:
        s = scenarios[i]
        print(f"  scenario {i:4d}: b1={s['b1']:.3f}, q1={s['q1']:.3f}, "
              f"b2={s['b2']:.3f}, q2={s['q2']:.3f}, "
              f"inflow_seeds=({s['inflow_seed1']},{s['inflow_seed2']})")


def main():
    os.makedirs('lakes', exist_ok=True)

    # Training set
    train = generate_lake_scenarios(lake_n_scenarios, seed=SCENARIO_SEED)
    train_path = 'lakes/lake_scenarios.npy'
    np.save(train_path, train)
    _summarise(train, [0, len(train) // 2, len(train) - 1], train_path)

    # Eval set — a separate seed so the two sets are independent draws.
    eval_ = generate_lake_scenarios(1000, seed=SCENARIO_SEED + 1)
    eval_path = 'lakes/lake_scenarios_eval.npy'
    np.save(eval_path, eval_)
    _summarise(eval_, [0, 199, 399, 599, 799, 999], eval_path)


if __name__ == '__main__':
    main()