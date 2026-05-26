from ema_workbench import (RealParameter, IntegerParameter, ScalarOutcome, Constant)
from scipy.optimize import brentq


def _pcrit(b, q):
    return brentq(lambda x: x ** q / (1 + x ** q) - b * x, 0.01, 1.5)


seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
tree_depth = 9
tree_multi_obj = 2
tree_many_obj = 6
slip_patterns_path = f'./trees/slip_patterns_depth{tree_depth}.npy'
tree_n_scenarios = 50
nd_update_freq_tree = 1
archive_cap_tree = None
gamma_tree = 1.0
lake_multi_obj = 2
lake_many_obj = 6
total_years = 100
years_per_action = 5
lake_scenarios_path = './lakes/lake_scenarios.npy'
lake_n_scenarios = 50
nd_size_cap_lake = 48
nd_update_freq_lake = 1
archive_cap_lake = None
gamma_lake = 0.95

tree_box_dim2 = {
    'nadir': [0.0] * 2,
    'ideal': [10.0] * 2,  # exact per-axis leaf maxes
}
tree_box_dim6 = {
    'nadir': [0.0] * 6,
    'ideal': [10.0] * 6,
}
lake_box_deterministic_dim2 = {
    'nadir': [-3.0, 0.0],
    'ideal': [0.0, 2.0],
}
lake_box_deterministic_dim6 = {
    'nadir': [-3.0, -3.0, 0.0, 0.0, -1.0, -1.0],
    'ideal': [0.0, 0.0, 2.0, 2.0, 0.0, 0.0],
}
lake_box_robust_dim2 = {
    'nadir': [-5.0, 0.0],
    'ideal': [0.0, 2.0],
}
lake_box_robust_dim6 = {
    'nadir': [-5.0, -5.0, 0.0, 0.0, -1.0, -1.0],
    'ideal': [0.0, 0.0, 2.0, 2.0, 0.0, 0.0],
}

multi_objs_tree_params = {
    'depth': tree_depth,

    'uncertainties': [IntegerParameter('scenario_index', 0, tree_n_scenarios - 1)],

    'outcomes': [ScalarOutcome(f'o{i + 1}', kind=ScalarOutcome.MINIMIZE) for i in range(tree_multi_obj)],

    'constants': [
        Constant("depth", tree_depth),
        Constant("num_obj", tree_multi_obj),
        Constant("csv_path", f"./trees/depth{tree_depth}_dim{tree_multi_obj}.csv"),
        Constant("observe", 1),
        Constant("slip_patterns_path", slip_patterns_path),
    ]
}

many_objs_tree_params = {
    'depth': tree_depth,

    'uncertainties': [IntegerParameter('scenario_index', 0, tree_n_scenarios - 1)],

    'outcomes': [ScalarOutcome(f'o{i + 1}', kind=ScalarOutcome.MINIMIZE) for i in range(tree_many_obj)],

    'constants': [
        Constant("depth", tree_depth),
        Constant("num_obj", tree_many_obj),
        Constant("csv_path", f"./trees/depth{tree_depth}_dim{tree_many_obj}.csv"),
        Constant("observe", 1),
        Constant("slip_patterns_path", slip_patterns_path),
    ]
}

default_tree_scenario = {'scenario_index': None}
default_tree_scenario_robust = {'scenario_index': 0}

tree_reference_scenarios = [
    {'scenario_index': 12},
    {'scenario_index': 24},
    {'scenario_index': 37},
    {'scenario_index': 49},
]

non_observable_constants_multi = [
    Constant("depth", tree_depth),
    Constant("num_obj", tree_multi_obj),
    Constant("csv_path", f"./trees/depth{tree_depth}_dim{tree_multi_obj}.csv"),
    Constant("observe", 0),
]

non_observable_constants_many = [
    Constant("depth", tree_depth),
    Constant("num_obj", tree_many_obj),
    Constant("csv_path", f"./trees/depth{tree_depth}_dim{tree_many_obj}.csv"),
    Constant("observe", 0),
]

# ------------------------------------------------------------------
# Two-lake model params
# ------------------------------------------------------------------
multi_objs_lake_params = {
    'uncertainties': [
        RealParameter('b1', 0.25, 0.45),
        RealParameter('q1', 2.5, 4.5),
        RealParameter('b2', 0.25, 0.45),
        RealParameter('q2', 2.5, 4.5),
        IntegerParameter('inflow_seed1', 0, 10000),
        IntegerParameter('inflow_seed2', 0, 10000),
    ],

    'outcomes': [ScalarOutcome(f'o{i + 1}', kind=ScalarOutcome.MINIMIZE)
                 for i in range(lake_multi_obj)],

    'constants': [
        Constant("num_obj", lake_multi_obj),
        Constant("alpha", 0.4),
        Constant("delta", 0.98),
        Constant("total_years", total_years),
        Constant("years_per_action", years_per_action),
    ]
}

many_objs_lake_params = {
    'uncertainties': [
        RealParameter('b1', 0.25, 0.45),
        RealParameter('q1', 2.5, 4.5),
        RealParameter('b2', 0.25, 0.45),
        RealParameter('q2', 2.5, 4.5),
        IntegerParameter('inflow_seed1', 0, 10000),
        IntegerParameter('inflow_seed2', 0, 10000),
    ],

    'outcomes': [ScalarOutcome(f'o{i + 1}', kind=ScalarOutcome.MINIMIZE)
                 for i in range(lake_many_obj)],

    'constants': [
        Constant("num_obj", lake_many_obj),
        Constant("alpha", 0.4),
        Constant("delta", 0.98),
        Constant("total_years", total_years),
        Constant("years_per_action", years_per_action),
    ]
}

default_lake_scenario = {
    'b1': 0.42, 'q1': 2.5,
    'b2': 0.35, 'q2': 3.0,
    'inflow_seed1': 0,
    'inflow_seed2': 0,
    'Pcrit1': _pcrit(0.42, 2.5),
    'Pcrit2': _pcrit(0.35, 3.0)
}

lake_reference_scenarios = [
    {'b1': 0.443, 'q1': 2.554,
     'b2': 0.252, 'q2': 2.823,
     'inflow_seed1': 1284, 'inflow_seed2': 1357,
     'Pcrit1': _pcrit(0.443, 2.554),
     'Pcrit2': _pcrit(0.252, 2.823)},
    {'b1': 0.252, 'q1': 2.544,
     'b2': 0.427, 'q2': 2.651,
     'inflow_seed1': 6900, 'inflow_seed2': 8408,
     'Pcrit1': _pcrit(0.252, 2.544),
     'Pcrit2': _pcrit(0.427, 2.651)},
    {'b1': 0.340, 'q1': 3.513,
     'b2': 0.361, 'q2': 3.415,
     'inflow_seed1': 2903, 'inflow_seed2': 6424,
     'Pcrit1': _pcrit(0.340, 3.513),
     'Pcrit2': _pcrit(0.361, 3.415)},
    {'b1': 0.258, 'q1': 2.712,
     'b2': 0.260, 'q2': 2.629,
     'inflow_seed1': 9443, 'inflow_seed2': 9837,
     'Pcrit1': _pcrit(0.258, 2.712),
     'Pcrit2': _pcrit(0.260, 2.629)},
]

# ────────────────────────────────────────────────────────────────────────────
# Constrained two-lake problem — parallel infrastructure
# ────────────────────────────────────────────────────────────────────────────

constrained_multi_objs_lake_params = {
    'uncertainties': [
        RealParameter('b1', 0.25, 0.45),
        RealParameter('q1', 2.5, 4.5),
        RealParameter('b2', 0.25, 0.45),
        RealParameter('q2', 2.5, 4.5),
        IntegerParameter('inflow_seed1', 0, 10000),
        IntegerParameter('inflow_seed2', 0, 10000),
    ],

    'outcomes': [
        ScalarOutcome(f'o{i + 1}', kind=ScalarOutcome.MINIMIZE)
        for i in range(lake_multi_obj)
    ],

    'constants': [
        Constant("num_obj", lake_multi_obj),
        Constant("alpha", 0.4),
        Constant("delta", 0.98),
        Constant("total_years", total_years),
        Constant("years_per_action", years_per_action),
    ]
}

# 6-obj MOEA model params.
constrained_many_objs_lake_params = {
    'uncertainties': [
        RealParameter('b1', 0.25, 0.45),
        RealParameter('q1', 2.5, 4.5),
        RealParameter('b2', 0.25, 0.45),
        RealParameter('q2', 2.5, 4.5),
        IntegerParameter('inflow_seed1', 0, 10000),
        IntegerParameter('inflow_seed2', 0, 10000),
    ],

    'outcomes': [
        ScalarOutcome(f'o{i + 1}', kind=ScalarOutcome.MINIMIZE)
        for i in range(lake_many_obj)
    ],

    'constants': [
        Constant("num_obj", lake_many_obj),
        Constant("alpha", 0.4),
        Constant("delta", 0.98),
        Constant("total_years", total_years),
        Constant("years_per_action", years_per_action),
    ]
}

constrained_lake_reference_scenarios = [
    {'b1': 0.351, 'q1': 4.065,
     'b2': 0.376, 'q2': 4.222,
     'inflow_seed1': 1522, 'inflow_seed2': 2307,
     'Pcrit1': _pcrit(0.351, 4.065),
     'Pcrit2': _pcrit(0.376, 4.222)},
    {'b1': 0.377, 'q1': 3.800,
     'b2': 0.399, 'q2': 3.350,
     'inflow_seed1': 8772, 'inflow_seed2': 955,
     'Pcrit1': _pcrit(0.377, 3.800),
     'Pcrit2': _pcrit(0.399, 3.350)},
    {'b1': 0.250, 'q1': 4.338,
     'b2': 0.375, 'q2': 3.314,
     'inflow_seed1': 8282, 'inflow_seed2': 3230,
     'Pcrit1': _pcrit(0.250, 4.338),
     'Pcrit2': _pcrit(0.375, 3.314)},
    {'b1': 0.396, 'q1': 2.807,
     'b2': 0.275, 'q2': 4.132,
     'inflow_seed1': 3329, 'inflow_seed2': 3111,
     'Pcrit1': _pcrit(0.396, 2.807),
     'Pcrit2': _pcrit(0.275, 4.132)},
]
