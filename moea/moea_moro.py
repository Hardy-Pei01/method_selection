import os
import random
import time

from ema_workbench import (ScalarOutcome, MultiprocessingEvaluator, Scenario)
from ema_workbench.em_framework.optimization import EpsilonProgress
from params_config import lake_scenarios_path, slip_patterns_path

import numpy as np


def lower_tail_20(outcomes):
    return np.percentile(outcomes, 80)


def build_robustness_functions(num_obj):
    return [
        ScalarOutcome(
            f'p20_o{i + 1}',
            kind=ScalarOutcome.MINIMIZE,
            variable_name=f'o{i + 1}',
            function=lower_tail_20
        )
        for i in range(num_obj)
    ]


def build_optimization_scenarios(problem):
    if problem == 'tree':
        patterns = np.load(slip_patterns_path)
        return [Scenario(f'scenario_{i}', scenario_index=i)
                for i in range(len(patterns))]
    else:
        scenarios = np.load(lake_scenarios_path)
        return [Scenario(f'scenario_{i}',
                         b1=float(s['b1']), q1=float(s['q1']),
                         b2=float(s['b2']), q2=float(s['q2']),
                         inflow_seed1=int(s['inflow_seed1']),
                         inflow_seed2=int(s['inflow_seed2']),
                         Pcrit1=float(s['Pcrit1']),
                         Pcrit2=float(s['Pcrit2']))
                for i, s in enumerate(scenarios)]


def run_moea(model, params, file_end, start_time, problem):
    archiveName = f'archives_{file_end}'
    convergenceName = f'convergences_{file_end}'

    if not os.path.exists(params.output_folder):
        os.makedirs(params.output_folder)

    if params.seed is not None:
        random.seed(params.seed)
        np.random.seed(params.seed)

    scenarios = build_optimization_scenarios(problem)
    num_obj = sum(1 for o in model.outcomes if o.kind != ScalarOutcome.INFO)
    robustness_functions = build_robustness_functions(num_obj)

    with MultiprocessingEvaluator(model, n_processes=32) as evaluator:
        arch, conv = evaluator.robust_optimize(
            robustness_functions,
            scenarios,
            algorithm=params.algorithm,
            nfe=params.nfe,
            epsilons=params.epsilons,
            convergence=[EpsilonProgress()],
            population_size=150
        )

        elapsed = int(time.time() - start_time)
        conv['time'] = time.strftime("%H:%M:%S", time.gmtime(elapsed))

        arch.to_csv(f'{params.output_folder}/{archiveName}.csv')
        conv.to_csv(f'{params.output_folder}/{convergenceName}.csv')

        return arch, conv
