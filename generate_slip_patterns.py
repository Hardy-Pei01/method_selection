import os
import numpy as np
from params_config import tree_depth

N_SCENARIOS = 200
SLIP_PROB_MIN = 0.0
SLIP_PROB_MAX = 0.2
PATTERN_SEED = 42  # fixed seed for full reproducibility


def generate_slip_patterns(depth, n_scenarios=N_SCENARIOS,
                           slip_prob_min=SLIP_PROB_MIN,
                           slip_prob_max=SLIP_PROB_MAX,
                           seed=PATTERN_SEED):
    n_internal = 2 ** depth - 1
    slip_probs = np.linspace(slip_prob_min, slip_prob_max, n_scenarios)
    patterns = np.zeros((n_scenarios, n_internal), dtype=bool)

    for i, prob in enumerate(slip_probs):
        rng = np.random.default_rng(seed + i)
        patterns[i] = rng.random(n_internal) < prob

    return patterns, slip_probs


def main(depth):
    patterns, slip_probs = generate_slip_patterns(depth)

    # os.makedirs('trees', exist_ok=True)
    out_path = f'trees/slip_patterns_depth{depth}_eval.npy'
    # np.save(out_path, patterns)

    print(f"depth={depth}: {patterns.shape[0]} scenarios, "
          f"{patterns.shape[1]} internal nodes")
    for i in [0, 49, 99, 149, 199]:
        print(f"  scenario {i:2d}: slip_prob={slip_probs[i]:.3f}, "
              f"nodes_slip={patterns[i].sum()}/{patterns.shape[1]}")
    print(f"  Saved → {out_path}")


if __name__ == '__main__':

    main(tree_depth)
