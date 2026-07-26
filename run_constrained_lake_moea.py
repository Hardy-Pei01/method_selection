from ema_workbench import ema_logging
from params_config import (constrained_multi_objs_lake_params,
                           constrained_many_objs_lake_params,
                           lake_multi_obj, lake_many_obj, seeds)
from moea.moea_method_config import (constrained_multi_lake_params,
                                     constrained_moro_lake_params,
                                     constrained_moea_moro,
                                     constrained_moea_multi)
from moea.model_builder import (constrained_inter_lake_model,
                                constrained_inter_robust_lake_model,
                                constrained_dps_lake_model,
                                constrained_dps_robust_lake_model)
from collections import defaultdict
import time

activate_logging = 1
root_folder = f'./constrained_lake_ea_3'

run_policy = {
    'intertemporal': 1,
    'dps': 1
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


def nested_dict():
    return defaultdict(nested_dict)


nfe_settings = nested_dict()
nfe_settings['intertemporal']['single']['multi_obj']['deterministic'] = 120000
nfe_settings['intertemporal']['single']['many_obj']['deterministic'] = 300000
nfe_settings['dps']['single']['multi_obj']['deterministic'] = 120000
nfe_settings['dps']['single']['many_obj']['deterministic'] = 300000
nfe_settings['intertemporal']['multi']['multi_obj']['robust'] = 120000
nfe_settings['intertemporal']['multi']['many_obj']['robust'] = 300000
nfe_settings['dps']['multi']['multi_obj']['robust'] = 120000
nfe_settings['dps']['multi']['many_obj']['robust'] = 300000
nfe_settings['intertemporal']['moro']['multi_obj']['robust'] = 120000
nfe_settings['intertemporal']['moro']['many_obj']['robust'] = 300000
nfe_settings['dps']['moro']['multi_obj']['robust'] = 120000
nfe_settings['dps']['moro']['many_obj']['robust'] = 300000

model_settings = nested_dict()
model_settings['intertemporal']['multi_obj']['deterministic'] = (constrained_inter_lake_model, 'interMulti')
model_settings['intertemporal']['many_obj']['deterministic'] = (constrained_inter_lake_model, 'interMany')
model_settings['intertemporal']['multi_obj']['robust'] = (constrained_inter_robust_lake_model, 'interMultiRobust')
model_settings['intertemporal']['many_obj']['robust'] = (constrained_inter_robust_lake_model, 'interManyRobust')
model_settings['dps']['multi_obj']['deterministic'] = (constrained_dps_lake_model, 'dpsMulti')
model_settings['dps']['many_obj']['deterministic'] = (constrained_dps_lake_model, 'dpsMany')
model_settings['dps']['multi_obj']['robust'] = (constrained_dps_robust_lake_model, 'dpsMultiRobust')
model_settings['dps']['many_obj']['robust'] = (constrained_dps_robust_lake_model, 'dpsManyRobust')

num_objectives = {
    'multi_obj': lake_multi_obj,
    'many_obj': lake_many_obj,
}

if __name__ == '__main__':

    # ── Replication loop over random seeds ───────
    for seed in seeds:

        if activate_logging:
            ema_logging.log_to_stderr(ema_logging.INFO)

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

                            if key_3 == 'single' and key_5 == 'robust':
                                continue
                            if key_3 in ('multi', 'moro') and key_5 == 'deterministic':
                                continue

                            num_obj = num_objectives[key_4]
                            base_name = f'{key_1}_{key_2}_{key_3}_{num_obj}'
                            nfe = nfe_settings[key_1][key_3][key_4][key_5]

                            robust = (key_5 == 'robust')
                            many_obj = (key_4 == 'many_obj')
                            model_params = (constrained_many_objs_lake_params if many_obj
                                            else constrained_multi_objs_lake_params)
                            model_func, model_name = model_settings[key_1][key_4][key_5]
                            model = model_func(model_params, model_name)

                            name = f'{base_name}/seed{seed}'
                            print('--------------------------------------------------------------------')
                            print(f"This experiment is {name}")
                            print('--------------------------------------------------------------------')

                            start_time = time.time()
                            if key_3 == "moro":
                                method_params = constrained_moro_lake_params(
                                    name=name, nfe=nfe, algo=key_2,
                                    root_folder=root_folder, many_obj=many_obj,
                                    robust=robust, seed=seed)
                                constrained_moea_moro(model, method_params,
                                                      start_time, problem='lake')
                            else:
                                method_params = constrained_multi_lake_params(
                                    name=name, nfe=nfe, algo=key_2,
                                    root_folder=root_folder, many_obj=many_obj,
                                    robust=robust, seed=seed)
                                constrained_moea_multi(model, method_params,
                                                       start_time, problem='lake')
