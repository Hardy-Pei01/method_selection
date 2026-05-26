from params_config import (default_tree_scenario, default_tree_scenario_robust,
                           default_lake_scenario, tree_multi_obj, tree_many_obj,
                           tree_reference_scenarios, lake_reference_scenarios,
                           constrained_lake_reference_scenarios)
import moea.moea_single as multi
import moea.moea_moro as moro
from ema_workbench import Scenario
from moea.algos import NSGAII, IBEA, MOEAD


class base_params(object):
    def __init__(self, name, nfe, algo, root_folder, robust, seed=None):
        self.name = name
        self.nfe = nfe
        self.algo_name = algo
        if self.algo_name == "NSGAII":
            self.algorithm = NSGAII
        elif self.algo_name == "IBEA":
            self.algorithm = IBEA
        else:
            self.algorithm = MOEAD
        self.output_folder = f'{root_folder}/{name}'
        self.robust = robust
        self.seed = seed


class base_tree_params(base_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust, seed=None):
        super().__init__(name, nfe, algo, root_folder, robust, seed)
        if many_obj:
            self.epsilons = [0.001] * tree_many_obj
        else:
            self.epsilons = [0.001] * tree_multi_obj


class multi_tree_params(base_tree_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust, seed=None):
        super().__init__(name, nfe, algo, root_folder, many_obj, robust, seed)

        self.references = tree_reference_scenarios


class moro_tree_params(base_tree_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust, seed=None):
        super().__init__(name, nfe, algo, root_folder, many_obj, robust, seed)


class base_lake_params(base_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust, seed=None):
        super().__init__(name, nfe, algo, root_folder, robust, seed)
        if many_obj:
            self.epsilons = [0.05, 0.05, 0.25, 0.25, 0.05, 0.05]
        else:
            self.epsilons = [0.01, 0.05]


class multi_lake_params(base_lake_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust, seed=None):
        super().__init__(name, nfe, algo, root_folder, many_obj, robust, seed)
        self.references = lake_reference_scenarios


class moro_lake_params(base_lake_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust, seed=None):
        super().__init__(name, nfe, algo, root_folder, many_obj, robust, seed)


def output_file_end(model, params):
    return f'{model.name}_{params.algo_name}_{params.nfe}'


def moea_multi(model, params, start_time, problem):
    base = default_tree_scenario if problem == 'tree' else default_lake_scenario
    base_robust = default_tree_scenario_robust if problem == 'tree' else default_lake_scenario

    if not params.robust:
        refs = [base]
    else:
        refs = params.references + [base_robust]

    file_end = output_file_end(model, params)
    for idx, ref in enumerate(refs):
        print('Reference scenario', idx)
        # Inner run_moea writes archive + convergence CSVs to
        # params.output_folder; we don't need to retain them here.
        multi.run_moea(model, params=params,
                       file_end=file_end,
                       reference=Scenario('reference', **ref),
                       ref_num=idx,
                       start_time=start_time)


def moea_moro(model, params, start_time, problem):
    file_end = output_file_end(model, params)
    # Inner run_moea writes archive + convergence CSVs.
    moro.run_moea(model, params=params,
                  file_end=file_end,
                  start_time=start_time,
                  problem=problem)


# ================================================================
# Constrained two-lake method config — parallel infrastructure
# ================================================================

class base_constrained_lake_params(base_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust, seed=None):
        super().__init__(name, nfe, algo, root_folder, robust, seed)
        if many_obj:
            self.epsilons = [0.05, 0.05, 0.25, 0.25, 0.05, 0.05]
        else:
            self.epsilons = [0.01, 0.05]


class constrained_multi_lake_params(base_constrained_lake_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust, seed=None):
        super().__init__(name, nfe, algo, root_folder, many_obj, robust, seed)
        self.references = constrained_lake_reference_scenarios
        self.many_obj = many_obj


class constrained_moro_lake_params(base_constrained_lake_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust, seed=None):
        super().__init__(name, nfe, algo, root_folder, many_obj, robust, seed)
        self.many_obj = many_obj


def constrained_moea_multi(model, params, start_time, problem):
    """Constrained multi/MORDM: identical to moea_multi."""
    base = default_lake_scenario
    if not params.robust:
        refs = [base]
    else:
        refs = params.references + [base]

    file_end = output_file_end(model, params)
    for idx, ref in enumerate(refs):
        print('Reference scenario', idx)
        multi.run_moea(model, params=params,
                       file_end=file_end,
                       reference=Scenario('reference', **ref),
                       ref_num=idx,
                       start_time=start_time)


def constrained_moea_moro(model, params, start_time, problem):
    """Constrained MORO: identical to moea_moro."""
    file_end = output_file_end(model, params)
    moro.run_moea(model, params=params,
                  file_end=file_end,
                  start_time=start_time,
                  problem=problem)