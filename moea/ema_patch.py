"""Patch for an ema_workbench bug that produces wrong
(decisions, objectives) pairs when MOEAD's iterate() makes many small
evaluate_all() calls per generation.

Root cause: ema_workbench's process_levers() assigns policy names
'0', '1', '2', ... that reset to 0 on every evaluate_all call. Some
state inside perform_experiments / its callback machinery is keyed by
policy name, so old results leak into new calls under the same names.
MOEAD's per-subproblem evaluation pattern (~150 calls/generation,
each with 1-2 policies named '0', '1') triggers this. NSGAII and IBEA
make few calls per generation with up to 150 unique names per call,
so they don't trigger it.

Fix: replace process_levers with a version that uses globally unique
names ('p_0', 'p_1', 'p_2', ...) across all calls in a Python process.

This patch is harmless to NSGAII / IBEA (uniqueness is preserved) and
also covers MORO since process_robust internally delegates to
process_levers.

Importing this module applies the patch as a side-effect.
"""
import ema_workbench.em_framework.optimization as _opt_mod
import ema_workbench.em_framework.evaluators as _ev_mod
from ema_workbench.em_framework.optimization import _process
from ema_workbench.em_framework.points import Policy

_global_policy_counter = [0]


def _patched_process_levers(jobs):
    problem = jobs[0].solution.problem
    policies = []
    processed = _process(jobs, problem)
    for proc_job in processed:
        name = f'p_{_global_policy_counter[0]}'
        _global_policy_counter[0] += 1
        policy = Policy(name=name, **proc_job)
        policies.append(policy)
    scenarios = problem.reference
    return scenarios, policies


# Patch both module references — optimization.py defines it, evaluators.py
# imports it. Both need to point at the new version.
_opt_mod.process_levers = _patched_process_levers
_ev_mod.process_levers = _patched_process_levers
