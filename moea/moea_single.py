import os
import random
import time
import numpy as np
from ema_workbench import (MultiprocessingEvaluator, SequentialEvaluator)
from ema_workbench.em_framework.optimization import (EpsilonProgress)


def run_moea(model, params, file_end, reference, ref_num, start_time):
    archiveName = f'archives_{file_end}'
    convergenceName = f'convergences_{file_end}'

    if not os.path.exists(params.output_folder):
        os.makedirs(params.output_folder)

    if params.seed is not None:
        random.seed(params.seed)
        np.random.seed(params.seed)

    with MultiprocessingEvaluator(model, n_processes=32) as evaluator:
        arch, conv = evaluator.optimize(
            algorithm=params.algorithm,
            nfe=params.nfe,
            searchover='levers',
            reference=reference,
            epsilons=params.epsilons,
            convergence=[EpsilonProgress()],
            population_size=150
        )

    if ref_num is not None:
        elapsed = int(time.time() - start_time)
        conv['time'] = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        arch['reference_scenario'] = ref_num

    suffix = f'_{ref_num}' if ref_num is not None else ''
    arch.to_csv(f'{params.output_folder}/{archiveName}{suffix}.csv')
    conv.to_csv(f'{params.output_folder}/{convergenceName}{suffix}.csv')

    return arch, conv
