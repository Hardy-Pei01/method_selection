import morl.morl_single as single
import morl.morl_moro as moro
from fruit_tree import FruitTreeEnv
from two_lake import TwoLakeEnv
from constrained_two_lake import ConstrainedTwoLakeEnv
from params_config import default_tree_scenario, default_tree_scenario_robust, \
    slip_patterns_path, default_lake_scenario, tree_reference_scenarios, \
    lake_reference_scenarios, total_years, years_per_action, \
    constrained_lake_reference_scenarios


class base_morl_params:

    def __init__(self, name, timesteps, scoring, root_folder,
                 robust, seed=None):
        self.name = name
        self.timesteps = timesteps
        self.scoring = scoring
        self.output_folder = f'{root_folder}/{name}'
        self.robust = robust
        self.seed = seed


class base_tree_morl_params(base_morl_params):

    def __init__(self, name, timesteps, scoring, root_folder,
                 many_obj, robust, num_weight_divisions=5, seed=None):
        super().__init__(name, timesteps, scoring, root_folder,
                         robust, seed)
        self.many_obj = many_obj
        self.num_weight_divisions = num_weight_divisions


class multi_tree_morl_params(base_tree_morl_params):

    def __init__(self, name, timesteps, scoring, root_folder,
                 many_obj, robust, num_weight_divisions=5, seed=None):
        super().__init__(name, timesteps, scoring, root_folder,
                         many_obj, robust, num_weight_divisions, seed)

        self.references = tree_reference_scenarios


class moro_tree_morl_params(base_tree_morl_params):

    def __init__(self, name, timesteps, scoring, root_folder,
                 many_obj, robust, num_weight_divisions=5, seed=None):
        super().__init__(name, timesteps, scoring, root_folder,
                         many_obj, robust, num_weight_divisions, seed)


# ── Parameter classes — lake ──────────────────────────────────────────────────

class base_lake_morl_params(base_morl_params):
    def __init__(self, name, timesteps, scoring, root_folder,
                 many_obj, robust, num_weight_divisions=5, seed=None):
        super().__init__(name, timesteps, scoring, root_folder,
                         robust, seed)
        self.many_obj = many_obj
        self.num_weight_divisions = num_weight_divisions


class multi_lake_morl_params(base_lake_morl_params):
    def __init__(self, name, timesteps, scoring, root_folder,
                 many_obj, robust, num_weight_divisions=5, seed=None):
        super().__init__(name, timesteps, scoring, root_folder,
                         many_obj, robust, num_weight_divisions, seed)

        self.references = lake_reference_scenarios


class moro_lake_morl_params(base_lake_morl_params):
    def __init__(self, name, timesteps, scoring, root_folder,
                 many_obj, robust, num_weight_divisions=5, seed=None):
        super().__init__(name, timesteps, scoring, root_folder,
                         many_obj, robust, num_weight_divisions, seed)


# ── Helpers ───────────────────────────────────────────────────────────────────

def output_file_end(params):
    return f'{params.timesteps}'


def _build_lake_env(ref, n_obj):
    """Construct a TwoLakeEnv from a reference scenario dict."""
    return TwoLakeEnv(
        b1=ref.get('b1', 0.42), q1=ref.get('q1', 2.5),
        b2=ref.get('b2', 0.35), q2=ref.get('q2', 3.0),
        inflow_seed1=ref.get('inflow_seed1', 0),
        inflow_seed2=ref.get('inflow_seed2', 0),
        Pcrit1=ref.get('Pcrit1', None),
        Pcrit2=ref.get('Pcrit2', None),
        num_obj=n_obj,
        total_years=total_years,
        years_per_action=years_per_action,
    )


# ── Orchestrators ─────────────────────────────────────────────────────────────

def morl_multi(params, ref_point, n_obj, csv_path, start_time):
    depth = moro._get_depth(csv_path)
    file_end = output_file_end(params)

    if not params.robust:
        refs = [default_tree_scenario]
        label_refs = False
    else:
        refs = params.references + [default_tree_scenario_robust]
        label_refs = True

    archives = []
    for ref_num, ref in enumerate(refs):
        scenario_index = ref['scenario_index']
        print(f'  Reference scenario {ref_num} (scenario_index={scenario_index})')

        env = FruitTreeEnv(
            depth=depth, reward_dim=n_obj,
            csv_path=csv_path, observe=True,
            scenario_index=scenario_index,
            slip_patterns_path=slip_patterns_path,
        )

        archive_size = single.run_morl_single(
            env=env,
            scoring=params.scoring,
            timesteps=params.timesteps,
            ref_point=ref_point,
            num_weight_divisions=params.num_weight_divisions,
            output_folder=params.output_folder,
            file_end=file_end,
            ref_num=ref_num if label_refs else None,
            start_time=start_time,
            seed=params.seed,
        )
        archives.append(archive_size)

    return archives


def morl_moro(params, ref_point, n_obj, csv_path, start_time):
    file_end = output_file_end(params)
    return moro.run_moro(
        scoring=params.scoring,
        timesteps=params.timesteps,
        ref_point=ref_point,
        n_obj=n_obj,
        csv_path=csv_path,
        num_weight_divisions=params.num_weight_divisions,
        output_folder=params.output_folder,
        file_end=file_end,
        start_time=start_time,
        seed=params.seed,
    )


# ── Orchestrators — lake ──────────────────────────────────────────────────────

def morl_multi_lake(params, ref_point, n_obj, start_time):
    file_end = output_file_end(params)

    if not params.robust:
        refs = [default_lake_scenario]
        label_refs = False
    else:
        refs = params.references + [default_lake_scenario]
        label_refs = True

    archives = []
    for ref_num, ref in enumerate(refs):
        print(f'  Reference scenario {ref_num} '
              f"(b1={ref.get('b1', 'default')}, q1={ref.get('q1', 'default')})")

        env = _build_lake_env(ref, n_obj)

        archive_size = single.run_morl_single(
            env=env,
            scoring=params.scoring,
            timesteps=params.timesteps,
            ref_point=ref_point,
            num_weight_divisions=params.num_weight_divisions,
            output_folder=params.output_folder,
            file_end=file_end,
            ref_num=ref_num if label_refs else None,
            start_time=start_time,
            seed=params.seed,
        )
        archives.append(archive_size)

    return archives


def morl_moro_lake(params, ref_point, n_obj, start_time):
    file_end = output_file_end(params)
    return moro.run_moro_lake(
        scoring=params.scoring,
        timesteps=params.timesteps,
        ref_point=ref_point,
        n_obj=n_obj,
        num_weight_divisions=params.num_weight_divisions,
        output_folder=params.output_folder,
        file_end=file_end,
        start_time=start_time,
        seed=params.seed,
    )


# ================================================================
# Constrained two-lake method config — parallel infrastructure
# ================================================================

class constrained_multi_lake_morl_params(base_lake_morl_params):
    def __init__(self, name, timesteps, scoring, root_folder,
                 many_obj, robust, num_weight_divisions=5, seed=None):
        super().__init__(name, timesteps, scoring, root_folder,
                         many_obj, robust, num_weight_divisions, seed)
        self.references = constrained_lake_reference_scenarios


class constrained_moro_lake_morl_params(base_lake_morl_params):
    def __init__(self, name, timesteps, scoring, root_folder,
                 many_obj, robust, num_weight_divisions=5, seed=None):
        super().__init__(name, timesteps, scoring, root_folder,
                         many_obj, robust, num_weight_divisions, seed)


def _build_constrained_lake_env(ref, n_obj):
    """Constrained-env constructor with the same per-ref signature as
    _build_lake_env, used by constrained_morl_multi_lake."""
    return ConstrainedTwoLakeEnv(
        b1=ref.get('b1', 0.42), q1=ref.get('q1', 2.5),
        b2=ref.get('b2', 0.35), q2=ref.get('q2', 3.0),
        inflow_seed1=ref.get('inflow_seed1', 0),
        inflow_seed2=ref.get('inflow_seed2', 0),
        Pcrit1=ref.get('Pcrit1', None),
        Pcrit2=ref.get('Pcrit2', None),
        num_obj=n_obj,
        total_years=total_years,
        years_per_action=years_per_action,
    )


def constrained_morl_multi_lake(params, ref_point, n_obj, start_time):
    file_end = output_file_end(params)

    if not params.robust:
        refs = [default_lake_scenario]
        label_refs = False
    else:
        refs = params.references + [default_lake_scenario]
        label_refs = True

    archives = []
    for ref_num, ref in enumerate(refs):
        print(f'  Reference scenario {ref_num} '
              f"(b1={ref.get('b1', 'default')}, q1={ref.get('q1', 'default')})")

        env = _build_constrained_lake_env(ref, n_obj)

        archive_size = single.run_morl_single(
            env=env,
            scoring=params.scoring,
            timesteps=params.timesteps,
            ref_point=ref_point,
            num_weight_divisions=params.num_weight_divisions,
            output_folder=params.output_folder,
            file_end=file_end,
            ref_num=ref_num if label_refs else None,
            start_time=start_time,
            seed=params.seed,
        )
        archives.append(archive_size)

    return archives


def constrained_morl_moro_lake(params, ref_point, n_obj, start_time):
    file_end = output_file_end(params)
    return moro.run_moro_lake_constrained(
        scoring=params.scoring,
        timesteps=params.timesteps,
        ref_point=ref_point,
        n_obj=n_obj,
        num_weight_divisions=params.num_weight_divisions,
        output_folder=params.output_folder,
        file_end=file_end,
        start_time=start_time,
        seed=params.seed,
    )
