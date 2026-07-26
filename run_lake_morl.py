import time
from collections import defaultdict

import numpy as np

from params_config import lake_multi_obj, lake_many_obj, seeds
from morl.morl_method_config import (
    multi_lake_morl_params,
    moro_lake_morl_params,
    morl_multi_lake,
    morl_moro_lake,
)

# ── Top-level output folder ───────────────────────────────────────────────────
root_folder = './lake_rl_3'

# ── Experiment toggles ────────────────────────────────────────────────────────

run_scoring = {
    'pareto': 1,
    'indicator': 0,
    'decomposition': 0,
}

run_scenario_method = {
    'single': 0,
    'multi': 1,
    'moro': 0
}

obj_uncertain = {
    'multi_obj': 0,
    'many_obj': 1
}

param_uncertain = {
    'deterministic': 1,
    'robust': 1,
}


# ── Timestep grid ─────────────────────────────────────────────────────────────

def _nested():
    return defaultdict(_nested)


timestep_settings = _nested()
timestep_settings['single']['multi_obj']['deterministic'] = 150000
timestep_settings['single']['many_obj']['deterministic'] = 300000
timestep_settings['multi']['multi_obj']['robust'] = 150000
timestep_settings['multi']['many_obj']['robust'] = 300000
timestep_settings['moro']['multi_obj']['robust'] = 300000
timestep_settings['moro']['many_obj']['robust'] = 600000

# ── PQL hyperparameters ───────────────────────────────────────────────────────

num_weight_divisions = {
    'multi_obj': 149,  # 150 weight vectors for 2 objectives
    'many_obj': 5,  # manageable grid for 6 objectives
}

# ── Objective metadata ────────────────────────────────────────────────────────
ref_points = {
    'multi_obj': np.full(lake_multi_obj, -5.0),
    'many_obj': np.full(lake_many_obj, -5.0),
}
num_objectives = {
    'multi_obj': lake_multi_obj,
    'many_obj': lake_many_obj,
}

# ── Main loop ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Replication loop over random seeds ───────────
    for seed in seeds:

        for key_1, val_1 in run_scoring.items():
            if not val_1:
                continue

            for key_3, val_3 in run_scenario_method.items():
                if not val_3:
                    continue

                for key_4, val_4 in obj_uncertain.items():
                    if not val_4:
                        continue

                    for key_5, val_5 in param_uncertain.items():
                        if not val_5:
                            continue

                        # Enforce valid (scenario_method, uncertainty) combinations —
                        # mirrors the same guards in run_tree_morl.py and run_lake_moea.py.
                        if key_3 == 'single' and key_5 == 'robust':
                            continue  # single only makes sense without param uncertainty
                        if key_3 in ('multi', 'moro') and key_5 == 'deterministic':
                            continue  # multi/moro require param uncertainty

                        timesteps = timestep_settings[key_3][key_4][key_5]
                        n_obj = num_objectives[key_4]
                        ref = ref_points[key_4]
                        nwd = num_weight_divisions[key_4]
                        robust = (key_5 == 'robust')
                        base_name = f'{key_1}_{key_3}_{n_obj}'
                        name = f'{base_name}/seed{seed}'
                        print('--------------------------------------------------------------------')
                        print(f'This experiment is {name}')
                        print('--------------------------------------------------------------------')

                        start_time = time.time()

                        if key_3 == 'moro':
                            params = moro_lake_morl_params(
                                name=name,
                                timesteps=timesteps,
                                scoring=key_1,
                                root_folder=root_folder,
                                many_obj=(key_4 == 'many_obj'),
                                robust=robust,
                                num_weight_divisions=nwd,
                                seed=seed,
                            )
                            result = morl_moro_lake(
                                params=params,
                                ref_point=ref,
                                n_obj=n_obj,
                                start_time=start_time,
                            )

                        else:
                            params = multi_lake_morl_params(
                                name=name,
                                timesteps=timesteps,
                                scoring=key_1,
                                root_folder=root_folder,
                                many_obj=(key_4 == 'many_obj'),
                                robust=robust,
                                num_weight_divisions=nwd,
                                seed=seed,
                            )
                            result = morl_multi_lake(
                                params=params,
                                ref_point=ref,
                                n_obj=n_obj,
                                start_time=start_time,
                            )

                        elapsed = int(time.time() - start_time)
                        if isinstance(result, list):
                            total_pcs = sum(result)
                        else:
                            total_pcs = result if result is not None else 0
                        print(
                            f'  Done in {time.strftime("%H:%M:%S", time.gmtime(elapsed))}. '
                            f'Total PCS size: {total_pcs}.'
                        )
