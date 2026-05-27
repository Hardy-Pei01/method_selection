from __future__ import annotations

import os
import re
import sys
import gzip
import pickle
import numpy as np
import pandas as pd
from types import SimpleNamespace


# ----------------------------------------------------------------------
# Module import setup — every adapter needs access to its env module.
# Drivers add the project root to sys.path; adapters import lazily.
# ----------------------------------------------------------------------
def _import_fruit_tree():
    from fruit_tree import FruitTreeEnv
    return FruitTreeEnv


def _import_two_lake():
    from two_lake import TwoLakeEnv
    return TwoLakeEnv


def _import_constrained_two_lake():
    from constrained_two_lake import ConstrainedTwoLakeEnv
    return ConstrainedTwoLakeEnv


def _import_pql():
    from morl.pql import PQL
    return PQL


# ======================================================================
# TREE adapter (FruitTreeEnv)
# ======================================================================
TREE_DEPTH = 9
TREE_CSV = {2: '../trees/depth9_dim2.csv',
            6: '../trees/depth9_dim6.csv'}


def _tree_build_env(scenario_handle, n_obj, observe):
    """Build a FruitTreeEnv. `scenario_handle` is a slip-pattern array
    (or None for build-only; the caller installs the pattern before
    rollout via `_tree_install_scenario`)."""
    FruitTreeEnv = _import_fruit_tree()
    env = FruitTreeEnv(depth=TREE_DEPTH, reward_dim=n_obj,
                       csv_path=TREE_CSV[n_obj], observe=bool(observe))
    return env


def _tree_install_scenario(env, slip_pattern):
    """In-place: swap the slip pattern. Reusing one env across scenarios
    avoids reloading the CSV / rebuilding the 2**depth tree array."""
    env._slip_pattern = slip_pattern
    return env


def _tree_load_default_scenarios():
    """Deterministic: one no-slip pattern."""
    n_internal = 2 ** TREE_DEPTH - 1
    return np.zeros((1, n_internal), dtype=bool)


def _tree_load_eval_scenarios(path):
    arr = np.load(path)
    n_internal = 2 ** TREE_DEPTH - 1
    assert arr.shape[1] == n_internal, \
        f'eval-slip width {arr.shape[1]} != 2**{TREE_DEPTH}-1'
    return arr  # shape (n_scen, n_internal)


def _tree_iter_scenarios(scenarios):
    """Yield scenarios one by one. For tree the container is a (N, M)
    numpy array; this yields each row."""
    for i in range(scenarios.shape[0]):
        yield scenarios[i]


def _tree_n_scenarios(scenarios):
    return int(scenarios.shape[0])


def _tree_get_box(setting, n_obj):
    """Tree uses the same box for both settings (det and robust)."""
    from params_config import tree_box_dim2, tree_box_dim6
    box = {2: tree_box_dim2, 6: tree_box_dim6}[n_obj]
    nadir = np.asarray(box['nadir'], dtype=float)
    ideal = np.asarray(box['ideal'], dtype=float)
    if not np.all(ideal > nadir):
        raise ValueError(f'Degenerate box for tree dim={n_obj}')
    return nadir, ideal


def _tree_moea_decision_cols(df, policy_kind):
    if policy_kind == 'intertemporal':
        cols = [c for c in df.columns if re.fullmatch(r'l\d+', c)]
    elif policy_kind == 'table':
        cols = [c for c in df.columns if re.fullmatch(r'n\d+', c)]
    else:
        raise ValueError(policy_kind)
    return sorted(cols, key=lambda c: int(re.search(r'\d+', c).group()))


def _tree_moea_decode_row(row, policy_kind, cols):
    """Returns the decision vector (int array)."""
    return row[cols].values.astype(int)


def _tree_moea_rollout(env, policy_kind, decisions, n_obj):
    """One rollout. Returns (total_reward (MAX-conv), feasible).
    Tree env is never infeasible — always True."""
    obs, _ = env.reset()
    total = np.zeros(n_obj, dtype=float)
    for step in range(TREE_DEPTH):
        if policy_kind == 'intertemporal':
            action = int(decisions[step])
        elif policy_kind == 'table':
            level, pos = obs
            node_id = int(2 ** level - 1) + pos
            action = int(decisions[node_id])
        else:
            raise ValueError(policy_kind)
        obs, reward, term, _, _ = env.step(action)
        total += reward
        if term:
            break
    return total, True


# --- Tree MOEA folder regex ---
_TREE_MOEA_RE_DET = re.compile(
    r'^(intertemporal|table)_(NSGAII|IBEA|MOEAD)_single_(\d+)_'
    r'(observable|non_observable)$')

_TREE_MOEA_RE_ROB = re.compile(
    r'^(intertemporal|table)_(NSGAII|IBEA|MOEAD)_(multi|moro)_(\d+)_'
    r'(observable|non_observable)$')


def _tree_moea_folder_regex(setting):
    return _TREE_MOEA_RE_DET if setting == 'deterministic' else _TREE_MOEA_RE_ROB


def _tree_moea_parse_match(m, setting):
    """Returns dict with algo, policy, scenm, n_obj, obs."""
    if setting == 'deterministic':
        policy, algo, n_obj_s, obs = m.groups()
        scenm = 'single'
    else:
        policy, algo, scenm, n_obj_s, obs = m.groups()
    return dict(algo=algo, policy=policy, scenm=scenm,
                n_obj=int(n_obj_s), obs=obs)


def _tree_moea_observe_from_folder(folder_name):
    """The folder name encodes observability for tree."""
    return 'non_observable' not in folder_name


# --- Tree MORL ---
def _tree_morl_load_agent(path, n_obj, observe=None):
    """Load a PQL agent saved against FruitTreeEnv."""
    PQL = _import_pql()
    FruitTreeEnv = _import_fruit_tree()
    with open(path, 'rb') as f:
        magic = f.read(2)
    opener = gzip.open if magic == b'\x1f\x8b' else open
    with opener(path, 'rb') as f:
        payload = pickle.load(f)
    cfg = payload['config']
    saved_robust = bool(cfg.get('robust', False))
    n_scenarios = payload.get('n_scenarios') if saved_robust else None

    env = FruitTreeEnv(
        depth=TREE_DEPTH, reward_dim=n_obj, csv_path=TREE_CSV[n_obj],
        observe=True, scenario_index=None, slip_patterns_path=None,
    )
    kwargs = dict(env=env, ref_point=np.asarray(cfg['ref_point']))
    if saved_robust and n_scenarios is not None:
        kwargs['robust'] = True
        kwargs['n_scenarios'] = n_scenarios
    agent = PQL(**kwargs)
    agent.load_q_table(path)
    return agent


def _tree_morl_action_meta(agent):
    """Tree MORL: scalar action (num_actions); state is encoded via
    np.ravel_multi_index(obs, env_shape)."""
    return {
        'env_shape': agent.env_shape,
        'num_actions': agent.num_actions,
    }


def _tree_morl_pick_action(cache, state_obs, target, action_meta):
    """L1 target match — same logic as the original tree_morl_reeval."""
    env_shape = action_meta['env_shape']
    state_flat = int(np.ravel_multi_index(state_obs, env_shape))
    per_action = cache.get(state_flat)
    if per_action is None:
        return 0, target, False

    best_action, best_dist, next_target = None, np.inf, target
    for a, (Q, Qsa) in per_action.items():
        dists = np.abs(Qsa - target).sum(axis=1)
        i = int(np.argmin(dists))
        if dists[i] < best_dist:
            best_dist = float(dists[i])
            best_action = a
            next_target = Q[i]
    if best_action is None:
        return 0, target, False
    return best_action, next_target, True


def _tree_morl_rollout(cache, env, target_vec, n_obj, action_meta):
    """One target-tracked rollout on FruitTreeEnv (slip already installed).
    Returns (total_reward, feasible=True)."""
    obs, _ = env.reset()
    target = np.array(target_vec, dtype=float)
    total = np.zeros(n_obj)
    for _ in range(TREE_DEPTH):
        action, next_target, _ = _tree_morl_pick_action(
            cache, obs, target, action_meta)
        obs, reward, terminal, _, _ = env.step(int(action))
        total += reward
        target = next_target
        if terminal:
            break
    return total, True


def _tree_morl_action_sequence(cache, env, target_vec, action_meta):
    """Diagnostic: action sequence under NO slip — counts unique
    target → action_seq mappings as a sanity check on MORL policies."""
    obs, _ = env.reset()
    env._slip_pattern = np.zeros(2 ** TREE_DEPTH - 1, dtype=bool)
    target = np.array(target_vec, dtype=float)
    actions = []
    for _ in range(TREE_DEPTH):
        action, next_target, _ = _tree_morl_pick_action(
            cache, obs, target, action_meta)
        actions.append(action)
        obs, _, terminal, _, _ = env.step(int(action))
        target = next_target
        if terminal:
            break
    return tuple(actions)


TREE = SimpleNamespace(
    name='tree',
    policy_kinds=('intertemporal', 'table'),
    has_observability=True,
    has_feasibility=False,
    morl_has_action_sequence_diag=True,
    morl_uses_subsample=False,
    # Env
    build_env=_tree_build_env,
    install_scenario=_tree_install_scenario,
    load_default_scenarios=_tree_load_default_scenarios,
    load_eval_scenarios=_tree_load_eval_scenarios,
    iter_scenarios=_tree_iter_scenarios,
    n_scenarios=_tree_n_scenarios,
    get_box=_tree_get_box,
    # MOEA
    moea_decision_cols=_tree_moea_decision_cols,
    moea_decode_row=_tree_moea_decode_row,
    moea_rollout=_tree_moea_rollout,
    moea_folder_regex=_tree_moea_folder_regex,
    moea_parse_match=_tree_moea_parse_match,
    moea_observe_from_folder=_tree_moea_observe_from_folder,
    # MORL
    morl_load_agent=_tree_morl_load_agent,
    morl_action_meta=_tree_morl_action_meta,
    morl_rollout=_tree_morl_rollout,
    morl_action_sequence=_tree_morl_action_sequence,
)


# ======================================================================
# LAKE adapter (TwoLakeEnv)
# ======================================================================
def _lake_scenario_dict_from_struct(s):
    return {
        'b1': float(s['b1']), 'q1': float(s['q1']),
        'b2': float(s['b2']), 'q2': float(s['q2']),
        'inflow_seed1': int(s['inflow_seed1']),
        'inflow_seed2': int(s['inflow_seed2']),
        'Pcrit1': float(s['Pcrit1']) if s['Pcrit1'] is not None else None,
        'Pcrit2': float(s['Pcrit2']) if s['Pcrit2'] is not None else None,
    }


def _lake_build_env(scenario, n_obj, observe=None):
    TwoLakeEnv = _import_two_lake()
    return TwoLakeEnv(
        b1=scenario['b1'], q1=scenario['q1'],
        b2=scenario['b2'], q2=scenario['q2'],
        inflow_seed1=scenario['inflow_seed1'],
        inflow_seed2=scenario['inflow_seed2'],
        Pcrit1=scenario.get('Pcrit1'),
        Pcrit2=scenario.get('Pcrit2'),
        num_obj=n_obj,
    )


def _lake_install_scenario(env, scenario):
    """Lake env state depends on construction args (Pcrit, b, q, seeds),
    so there's no in-place swap path — caller builds a fresh env per
    scenario. Returning env unchanged means callers should always
    `build_env(s, n_obj)` per scenario instead of relying on install."""
    return env  # callers ignore this for lake; they rebuild per scenario


def _lake_load_default_scenarios():
    from params_config import default_lake_scenario
    return [_lake_scenario_dict_from_struct(default_lake_scenario)]


def _lake_load_eval_scenarios(path):
    arr = np.load(path)
    return [_lake_scenario_dict_from_struct(s) for s in arr]


def _lake_iter_scenarios(scenarios):
    """Lake scenarios are a Python list; just yield them."""
    yield from scenarios


def _lake_n_scenarios(scenarios):
    return len(scenarios)


def _lake_get_box(setting, n_obj):
    if setting == 'deterministic':
        from params_config import (lake_box_deterministic_dim2,
                                   lake_box_deterministic_dim6)
        box = {2: lake_box_deterministic_dim2,
               6: lake_box_deterministic_dim6}[n_obj]
    else:
        from params_config import (lake_box_robust_dim2,
                                   lake_box_robust_dim6)
        box = {2: lake_box_robust_dim2,
               6: lake_box_robust_dim6}[n_obj]
    nadir = np.asarray(box['nadir'], dtype=float)
    ideal = np.asarray(box['ideal'], dtype=float)
    if not np.all(ideal > nadir):
        raise ValueError(f'Degenerate box for lake dim={n_obj}')
    return nadir, ideal


def _lake_moea_decision_cols(df, policy_kind):
    """For lake intertemporal, return ordered (u1_*, u2_*) cols.
    For lake dps, the column set is fixed names (c*, r*, w*); we still
    return them ordered for consistency, though the rollout looks them
    up by name."""
    if policy_kind == 'intertemporal':
        u1 = sorted(
            [c for c in df.columns if re.fullmatch(r'u1_\d+', c)],
            key=lambda c: int(re.search(r'\d+', c).group()))
        u2 = sorted(
            [c for c in df.columns if re.fullmatch(r'u2_\d+', c)],
            key=lambda c: int(re.search(r'\d+', c).group()))
        return u1 + u2
    elif policy_kind == 'dps':
        return ['c1_1', 'c2_1', 'r1_1', 'r2_1', 'w1_1',
                'c1_2', 'c2_2', 'r1_2', 'r2_2', 'w1_2']
    else:
        raise ValueError(policy_kind)


def _lake_moea_decode_row(row, policy_kind, cols):
    """For intertemporal: split cols into (u1_keys, u2_keys) and pull ints.
    For dps: 10 named floats. Returns a tuple of decoded primitives."""
    if policy_kind == 'intertemporal':
        n_each = len(cols) // 2
        u1_keys = cols[:n_each]
        u2_keys = cols[n_each:]
        u1 = [int(row[c]) for c in u1_keys]
        u2 = [int(row[c]) for c in u2_keys]
        return ('intertemporal', u1, u2)
    elif policy_kind == 'dps':
        params = tuple(float(row[c]) for c in cols)
        return ('dps',) + params
    else:
        raise ValueError(policy_kind)


def _lake_emission_from_rbf(xt, c1, c2, r1, r2, w1):
    rule = w1 * (abs(xt - c1) / r1) ** 3 + (1 - w1) * (abs(xt - c2) / r2) ** 3
    u = float(np.clip(rule, 0.0, 0.10))
    return int(round(u / 0.02))


def _lake_moea_rollout_inner(env, decoded, n_obj, track_feasibility):
    """Shared rollout for TwoLakeEnv and ConstrainedTwoLakeEnv. Returns
    (total_reward, feasible). For unconstrained envs the env.step()
    info dict has no 'feasible' key; .get('feasible', True) returns
    True automatically."""
    env.reset()
    total = np.zeros(n_obj, dtype=np.float64)
    feasible = True
    kind = decoded[0]
    if kind == 'intertemporal':
        _, u1, u2 = decoded
        for t in range(env.n_gym_steps):
            _, reward, _, _, info = env.step(
                np.array([u1[t], u2[t]], dtype=np.int64))
            total += np.asarray(reward, dtype=np.float64)
            if track_feasibility:
                feasible = bool(info.get('feasible', True))
    elif kind == 'dps':
        (_, c1_1, c2_1, r1_1, r2_1, w1_1,
            c1_2, c2_2, r1_2, r2_2, w1_2) = decoded
        for _ in range(env.n_gym_steps):
            X1, X2 = env.X1, env.X2
            u1 = _lake_emission_from_rbf(X1, c1_1, c2_1, r1_1, r2_1, w1_1)
            u2 = _lake_emission_from_rbf(X2, c1_2, c2_2, r1_2, r2_2, w1_2)
            _, reward, _, _, info = env.step(np.array([u1, u2], dtype=np.int64))
            total += np.asarray(reward, dtype=np.float64)
            if track_feasibility:
                feasible = bool(info.get('feasible', True))
    else:
        raise ValueError(kind)
    return total, feasible


def _lake_moea_rollout(env, policy_kind, decoded, n_obj):
    return _lake_moea_rollout_inner(env, decoded, n_obj,
                                    track_feasibility=False)


# --- Lake MOEA folder regex ---
_LAKE_MOEA_RE_DET = re.compile(
    r'^(intertemporal|dps)_(NSGAII|IBEA|MOEAD)_single_(\d+)$')

_LAKE_MOEA_RE_ROB = re.compile(
    r'^(intertemporal|dps)_(NSGAII|IBEA|MOEAD)_(multi|moro)_(\d+)$')


def _lake_moea_folder_regex(setting):
    return _LAKE_MOEA_RE_DET if setting == 'deterministic' else _LAKE_MOEA_RE_ROB


def _lake_moea_parse_match(m, setting):
    if setting == 'deterministic':
        policy, algo, n_obj_s = m.groups()
        scenm = 'single'
    else:
        policy, algo, scenm, n_obj_s = m.groups()
    return dict(algo=algo, policy=policy, scenm=scenm,
                n_obj=int(n_obj_s), obs=None)


def _lake_moea_observe_from_folder(folder_name):
    return None  # lake has no observability


# --- Lake MORL ---
def _lake_morl_load_agent(path, n_obj, observe=None):
    PQL = _import_pql()
    TwoLakeEnv = _import_two_lake()
    with open(path, 'rb') as f:
        magic = f.read(2)
    opener = gzip.open if magic == b'\x1f\x8b' else open
    with opener(path, 'rb') as f:
        payload = pickle.load(f)
    cfg = payload['config']
    saved_robust = bool(cfg.get('robust', False))
    n_scenarios = payload.get('n_scenarios') if saved_robust else None

    env = TwoLakeEnv(num_obj=n_obj)
    kwargs = dict(env=env, ref_point=np.asarray(cfg['ref_point']))
    if saved_robust and n_scenarios is not None:
        kwargs['robust'] = True
        kwargs['n_scenarios'] = n_scenarios
    agent = PQL(**kwargs)
    agent.load_q_table(path)
    return agent


def _lake_morl_action_meta(agent):
    return {
        'env_shape': tuple(int(x) for x in agent.env_shape),
        'env_shape_arr': np.array(tuple(int(x) for x in agent.env_shape)),
        'action_nvec': tuple(int(x) for x in agent.env.action_space.nvec),
    }


def _lake_morl_pick_action(cache, state_obs, target, action_meta):
    """L1 target match. Lake returns a flat action that must be
    unravelled to a multi-dim action via action_nvec downstream."""
    env_shape = action_meta['env_shape_arr']
    state_flat = int(np.ravel_multi_index(state_obs, env_shape))
    per_action = cache.get(state_flat)
    if per_action is None:
        return 0, target, False

    best_action, best_dist, next_target = None, np.inf, target
    for a, (Q, Qsa) in per_action.items():
        dists = np.abs(Qsa - target).sum(axis=1)
        i = int(np.argmin(dists))
        if dists[i] < best_dist:
            best_dist = float(dists[i])
            best_action = a
            next_target = Q[i]
    if best_action is None:
        return 0, target, False
    return best_action, next_target, True


def _lake_morl_rollout_inner(cache, env, target_vec, n_obj, action_meta,
                             track_feasibility):
    obs, _ = env.reset()
    target = np.array(target_vec, dtype=float)
    total = np.zeros(n_obj, dtype=np.float64)
    feasible = True
    action_nvec = action_meta['action_nvec']
    for _ in range(env.n_gym_steps):
        action_flat, next_target, _ = _lake_morl_pick_action(
            cache, obs, target, action_meta)
        action_nd = np.unravel_index(action_flat, action_nvec)
        obs, reward, terminated, truncated, info = env.step(
            np.array(action_nd, dtype=np.int64))
        total += np.asarray(reward, dtype=np.float64)
        target = next_target
        if track_feasibility:
            feasible = bool(info.get('feasible', True))
        if terminated or truncated:
            break
    return total, feasible


def _lake_morl_rollout(cache, env, target_vec, n_obj, action_meta):
    return _lake_morl_rollout_inner(cache, env, target_vec, n_obj,
                                    action_meta, track_feasibility=False)


LAKE = SimpleNamespace(
    name='lake',
    policy_kinds=('intertemporal', 'dps'),
    has_observability=False,
    has_feasibility=False,
    morl_has_action_sequence_diag=False,
    morl_uses_subsample=True,
    # Env
    build_env=_lake_build_env,
    install_scenario=_lake_install_scenario,
    load_default_scenarios=_lake_load_default_scenarios,
    load_eval_scenarios=_lake_load_eval_scenarios,
    iter_scenarios=_lake_iter_scenarios,
    n_scenarios=_lake_n_scenarios,
    get_box=_lake_get_box,
    # MOEA
    moea_decision_cols=_lake_moea_decision_cols,
    moea_decode_row=_lake_moea_decode_row,
    moea_rollout=_lake_moea_rollout,
    moea_folder_regex=_lake_moea_folder_regex,
    moea_parse_match=_lake_moea_parse_match,
    moea_observe_from_folder=_lake_moea_observe_from_folder,
    # MORL
    morl_load_agent=_lake_morl_load_agent,
    morl_action_meta=_lake_morl_action_meta,
    morl_rollout=_lake_morl_rollout,
    morl_action_sequence=None,
)


# ======================================================================
# CONSTRAINED_LAKE adapter (ConstrainedTwoLakeEnv)
# ======================================================================
def _clake_build_env(scenario, n_obj, observe=None):
    ConstrainedTwoLakeEnv = _import_constrained_two_lake()
    return ConstrainedTwoLakeEnv(
        b1=scenario['b1'], q1=scenario['q1'],
        b2=scenario['b2'], q2=scenario['q2'],
        inflow_seed1=scenario['inflow_seed1'],
        inflow_seed2=scenario['inflow_seed2'],
        Pcrit1=scenario.get('Pcrit1'),
        Pcrit2=scenario.get('Pcrit2'),
        num_obj=n_obj,
    )


def _clake_moea_rollout(env, policy_kind, decoded, n_obj):
    """Constrained variant: tracks feasibility."""
    return _lake_moea_rollout_inner(env, decoded, n_obj,
                                    track_feasibility=True)


def _clake_morl_load_agent(path, n_obj, observe=None):
    PQL = _import_pql()
    ConstrainedTwoLakeEnv = _import_constrained_two_lake()
    with open(path, 'rb') as f:
        magic = f.read(2)
    opener = gzip.open if magic == b'\x1f\x8b' else open
    with opener(path, 'rb') as f:
        payload = pickle.load(f)
    cfg = payload['config']
    saved_robust = bool(cfg.get('robust', False))
    n_scenarios = payload.get('n_scenarios') if saved_robust else None

    env = ConstrainedTwoLakeEnv(num_obj=n_obj)
    kwargs = dict(env=env, ref_point=np.asarray(cfg['ref_point']))
    if saved_robust and n_scenarios is not None:
        kwargs['robust'] = True
        kwargs['n_scenarios'] = n_scenarios
    agent = PQL(**kwargs)
    agent.load_q_table(path)
    return agent


def _clake_morl_rollout(cache, env, target_vec, n_obj, action_meta):
    return _lake_morl_rollout_inner(cache, env, target_vec, n_obj,
                                    action_meta, track_feasibility=True)


CONSTRAINED_LAKE = SimpleNamespace(
    name='constrained_lake',
    policy_kinds=('intertemporal', 'dps'),
    has_observability=False,
    has_feasibility=True,
    morl_has_action_sequence_diag=False,
    morl_uses_subsample=True,
    # Env
    build_env=_clake_build_env,
    install_scenario=_lake_install_scenario,  # same no-op semantics as lake
    load_default_scenarios=_lake_load_default_scenarios,  # same scenario format
    load_eval_scenarios=_lake_load_eval_scenarios,        # same scenario format
    iter_scenarios=_lake_iter_scenarios,
    n_scenarios=_lake_n_scenarios,
    get_box=_lake_get_box,  # uses lake_box_{det,robust}_dim* — same as lake
    # MOEA
    moea_decision_cols=_lake_moea_decision_cols,
    moea_decode_row=_lake_moea_decode_row,
    moea_rollout=_clake_moea_rollout,
    moea_folder_regex=_lake_moea_folder_regex,
    moea_parse_match=_lake_moea_parse_match,
    moea_observe_from_folder=_lake_moea_observe_from_folder,
    # MORL
    morl_load_agent=_clake_morl_load_agent,
    morl_action_meta=_lake_morl_action_meta,
    morl_rollout=_clake_morl_rollout,
    morl_action_sequence=None,
)


# ----------------------------------------------------------------------
# Lookup by name (used by drivers)
# ----------------------------------------------------------------------
ADAPTERS = {
    'tree': TREE,
    'lake': LAKE,
    'constrained_lake': CONSTRAINED_LAKE,
}


def get(name):
    """Lookup an adapter by name. Raises KeyError if not found."""
    if name not in ADAPTERS:
        raise KeyError(f'unknown env adapter {name!r}; '
                       f'available: {sorted(ADAPTERS)}')
    return ADAPTERS[name]
