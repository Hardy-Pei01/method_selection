import time
from collections import defaultdict

import numpy as np

from params_config import tree_depth, tree_multi_obj, tree_many_obj, seeds
from morl.morl_method_config import (
    multi_tree_morl_params,
    moro_tree_morl_params,
    morl_multi,
    morl_moro,
)

# ── Top-level output folder ───────────────────────────────────────────────────
root_folder = f'./tree_rl_{tree_depth}'

# ── Experiment toggles ────────────────────────────────────────────────────────
# Set a value to 1 to include that dimension in the run grid.

# PQL action-evaluation method (analog of run_evo_method in run_tree_moea.py)
run_scoring = {
    'pareto': 1,
    'indicator': 1,
    'decomposition': 1,
}

run_scenario_method = {
    'single': 0,
    'multi': 0,
    'moro': 1,
}

obj_uncertain = {
    'multi_obj': 1,
    'many_obj': 1,
}

param_uncertain = {
    'deterministic': 1,
    'robust': 1
}


def _nested():
    return defaultdict(_nested)


timestep_settings = _nested()
timestep_settings['single']['multi_obj']['deterministic'] = 100000
timestep_settings['single']['many_obj']['deterministic'] = 100000
timestep_settings['multi']['multi_obj']['robust'] = 100000
timestep_settings['multi']['many_obj']['robust'] = 100000
timestep_settings['moro']['multi_obj']['robust'] = 100000
timestep_settings['moro']['many_obj']['robust'] = 100000

# ── PQL hyperparameters ───────────────────────────────────────────────────────
# Objective-count-dependent settings (no moea equivalent — PQL-specific).
num_weight_divisions = {
    'multi_obj': 149,  # C(149+2-1, 149) = 150 weight vectors for 2 objectives
    'many_obj': 5,  # manageable grid for 6 objectives
}

# ── Objective metadata ────────────────────────────────────────────────────────
ref_points = {
    'multi_obj': np.full(tree_multi_obj, -1.0),
    'many_obj': np.full(tree_many_obj,  -1.0),
}
csv_paths = {
    'multi_obj': f'./trees/depth{tree_depth}_dim{tree_multi_obj}.csv',
    'many_obj': f'./trees/depth{tree_depth}_dim{tree_many_obj}.csv',
}
num_objectives = {
    'multi_obj': tree_multi_obj,
    'many_obj': tree_many_obj,
}

if __name__ == '__main__':

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
                    # mirrors the same guards in run_tree_moea.py.
                    if key_3 == 'single' and key_5 == 'robust':
                        continue  # single only makes sense without param uncertainty
                    if key_3 in ('multi', 'moro') and key_5 == 'deterministic':
                        continue  # multi/moro require param uncertainty

                    timesteps = timestep_settings[key_3][key_4][key_5]
                    n_obj = num_objectives[key_4]
                    csv = csv_paths[key_4]
                    ref = ref_points[key_4]
                    nwd = num_weight_divisions[key_4]
                    base_name = f'{key_1}_{key_3}_{n_obj}'

                    # robust=0 for 'single' (non_param), 1 for 'multi'/'moro' (param)
                    robust = (key_5 == 'robust')

                    # ── Replication loop over random seeds ───────────
                    for seed in seeds:
                        name = f'{base_name}/seed{seed}'
                        print('--------------------------------------------------------------------')
                        print(f"This experiment is {name}, with depth={tree_depth}, num_obj={n_obj}")
                        print('--------------------------------------------------------------------')

                        start_time = time.time()

                        if key_3 == 'moro':
                            # Robust optimisation across all scenarios — mirrors moea_moro
                            params = moro_tree_morl_params(
                                name=name,
                                timesteps=timesteps,
                                scoring=key_1,
                                root_folder=root_folder,
                                many_obj=(key_4 == 'many_obj'),
                                robust=robust,
                                num_weight_divisions=nwd,
                                seed=seed,
                            )
                            result = morl_moro(
                                params=params,
                                ref_point=ref,
                                n_obj=n_obj,
                                csv_path=csv,
                                start_time=start_time,
                            )

                        else:
                            # Single scenario or separate-per-reference — mirrors moea_multi
                            params = multi_tree_morl_params(
                                name=name,
                                timesteps=timesteps,
                                scoring=key_1,
                                root_folder=root_folder,
                                many_obj=(key_4 == 'many_obj'),
                                robust=robust,
                                num_weight_divisions=nwd,
                                seed=seed,
                            )
                            result = morl_multi(
                                params=params,
                                ref_point=ref,
                                n_obj=n_obj,
                                csv_path=csv,
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
