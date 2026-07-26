from ema_workbench import ema_logging
from params_config import (multi_objs_tree_params, many_objs_tree_params,
                           tree_depth, tree_multi_obj, tree_many_obj, seeds)
from moea.moea_method_config import multi_tree_params, moro_tree_params, moea_moro, moea_multi
from moea.model_builder import (inter_tree_model, inter_robust_tree_model, table_tree_model,
                                table_robust_tree_model, table_many_objs_partially_observable_tree_model,
                                table_multi_objs_partially_observable_tree_model)
from collections import defaultdict
import time

activate_logging = 1
root_folder = f'./tree_ea_5'

run_policy = {
    'intertemporal': 1,
    'table': 1
}
run_evo_method = {
    'NSGAII': 1,
    'IBEA': 1,
    'MOEAD': 1
}
run_scenario_method = {
    'single': 1,
    'multi': 1,
    'moro': 1
}

obj_uncertain = {
    'multi_obj': 1,
    'many_obj': 1
}

param_uncertain = {
    'deterministic': 1,
    'robust': 1
}

observability = {
    'observable': 1,
    'non_observable': 1
}


def tree():
    return defaultdict(tree)


nfe_settings = tree()
nfe_settings['intertemporal']['single']['multi_obj']['deterministic'] = 25000
nfe_settings['intertemporal']['single']['many_obj']['deterministic'] = 25000
nfe_settings['table']['single']['multi_obj']['deterministic'] = 25000
nfe_settings['table']['single']['many_obj']['deterministic'] = 25000
nfe_settings['intertemporal']['multi']['multi_obj']['robust'] = 25000
nfe_settings['intertemporal']['multi']['many_obj']['robust'] = 25000
nfe_settings['table']['multi']['multi_obj']['robust'] = 25000
nfe_settings['table']['multi']['many_obj']['robust'] = 25000
nfe_settings['intertemporal']['moro']['multi_obj']['robust'] = 60000
nfe_settings['intertemporal']['moro']['many_obj']['robust'] = 60000
nfe_settings['table']['moro']['multi_obj']['robust'] = 60000
nfe_settings['table']['moro']['many_obj']['robust'] = 150000

model_settings = tree()
model_settings['intertemporal']['multi_obj']['deterministic']['observable'] = (inter_tree_model, 'interMulti')
model_settings['intertemporal']['many_obj']['deterministic']['observable'] = (inter_tree_model, 'interMany')
model_settings['intertemporal']['multi_obj']['robust']['observable'] = (inter_robust_tree_model, 'interMultiRobust')
model_settings['intertemporal']['many_obj']['robust']['observable'] = (inter_robust_tree_model, 'interManyRobust')
model_settings['table']['multi_obj']['deterministic']['observable'] = (table_tree_model, 'tableMulti')
model_settings['table']['many_obj']['deterministic']['observable'] = (table_tree_model, 'tableMany')
model_settings['table']['multi_obj']['robust']['observable'] = (table_robust_tree_model, 'tableMultiRobust')
model_settings['table']['many_obj']['robust']['observable'] = (table_robust_tree_model, 'tableManyRobust')
model_settings['table']['multi_obj']['deterministic']['non_observable'] = (
    table_multi_objs_partially_observable_tree_model, 'tableMultiNonObs')
model_settings['table']['many_obj']['deterministic']['non_observable'] = (
    table_many_objs_partially_observable_tree_model, 'tableManyNonObs')

num_objectives = {
    'multi_obj': tree_multi_obj,
    'many_obj': tree_many_obj,
}

if __name__ == '__main__':
    if (activate_logging):
        ema_logging.log_to_stderr(ema_logging.INFO)

    # ── Replication loop over random seeds ───────────
    for seed in seeds:

        for key_1, value_1 in run_policy.items():
            if not value_1:
                continue

            for key_2, value_2 in run_evo_method.items():
                if not value_2:
                    continue

                for key_3, value_3 in run_scenario_method.items():
                    if not value_3:
                        continue

                    for key_4, value_4 in obj_uncertain.items():
                        if not value_4:
                            continue

                        for key_5, value_5 in param_uncertain.items():
                            if not value_5:
                                continue

                            for key_6, value_6 in observability.items():
                                if not value_6:
                                    continue

                                # Guard 1: single only with deterministic,
                                #          multi/moro only with robust
                                if key_3 == 'single' and key_5 == 'robust':
                                    continue
                                if key_3 in ('multi', 'moro') and key_5 == 'deterministic':
                                    continue

                                # Guard 2: non-observable only for table+single+deterministic
                                if key_6 == 'non_observable' and not (
                                        key_1 == 'table' and
                                        key_3 == 'single' and
                                        key_5 == 'deterministic'):
                                    continue

                                num_obj = num_objectives[key_4]
                                base_name = f'{key_1}_{key_2}_{key_3}_{num_obj}_{key_6}'
                                nfe = nfe_settings[key_1][key_3][key_4][key_5]

                                robust = (key_5 == 'robust')
                                many_obj = (key_4 == 'many_obj')
                                model_params = many_objs_tree_params if many_obj else multi_objs_tree_params

                                model_func, model_name = model_settings[key_1][key_4][key_5][key_6]
                                model = model_func(model_params, model_name)

                                name = f'{base_name}/seed{seed}'
                                print('--------------------------------------------------------------------')
                                print(f"This experiment is {name}, with depth={tree_depth}, num_obj={num_obj}")
                                print('--------------------------------------------------------------------')

                                start_time = time.time()

                                if key_3 == "moro":
                                    method_params = moro_tree_params(name=name, nfe=nfe, algo=key_2,
                                                                     root_folder=root_folder, many_obj=many_obj,
                                                                     robust=robust, seed=seed)
                                    moea_moro(model, method_params, start_time, problem='tree')
                                else:
                                    method_params = multi_tree_params(name=name, nfe=nfe, algo=key_2,
                                                                      root_folder=root_folder, many_obj=many_obj,
                                                                      robust=robust, seed=seed)
                                    moea_multi(model, method_params, start_time, problem='tree')
