"""Shared analysis helpers used by tree / lake / constrained_lake notebooks.

The notebooks themselves carry only:
  * paths to MORL/MOEA metrics CSVs
  * the env-specific vocabulary (condition style, label, order)
  * the env-specific folder regexes for convergence + runtime
  * one thin call per figure

Everything else — loading, summary printing, plotting (box plots), legend
construction, convergence aggregation, runtime collection — lives here.

Public API (alphabetical):

  Metrics loading & summary
    load_metrics(morl_csv, moea_csv, *, metrics, modes=None)
    print_metric_summary(df, *, metrics, n_objs, condition_order, modes=None)

  Figures
    figure_metrics(df, *, metrics, n_objs, ..., modes=None) -> None  (figure auto-displays in Jupyter)
    figure_counts(df, *, n_objs, ..., modes=None)            -> None  (figure auto-displays in Jupyter)
    figure_convergence(records, *, methods, conditions, n_objs, ...) -> None  (figure auto-displays in Jupyter)

  Convergence + runtime
    load_convergences(base, *, folder_re, condition_format, csv_prefix,
                      x_col, y_col, dropna_y=False) -> list of records
    collect_runtimes(base, *, folder_re, paradigm, condition_format,
                     csv_prefix, mode='per_file') -> list of dicts
    summarize_runtimes(runtime_df, *, title='') -> pd.DataFrame
"""

from __future__ import annotations

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ======================================================================
# Vocabulary (constants shared across all envs)
# ======================================================================
MORL_METHODS = ['pareto', 'indicator', 'decomposition']
MOEA_METHODS = ['NSGAII', 'IBEA', 'MOEAD']
ORDER = MORL_METHODS + MOEA_METHODS

METHOD_LABEL = {
    'pareto':        'PQL-Pareto',
    'indicator':     'PQL-Indicator',
    'decomposition': 'PQL-Decomp',
    'NSGAII':        'NSGA-II',
    'IBEA':          'IBEA',
    'MOEAD':         'MOEA/D',
}
PARADIGM_OF = {**{m: 'MORL' for m in MORL_METHODS},
               **{m: 'MOEA' for m in MOEA_METHODS}}
PARADIGM_COLOR = {'MORL': '#D85A30', 'MOEA': '#378ADD'}

ALGO_LINE_COLOR = {
    'NSGAII':        '#378ADD',
    'IBEA':          '#D85A30',
    'MOEAD':         '#1D9E75',
    'pareto':        '#D85A30',
    'indicator':     '#9E1D75',
    'decomposition': '#E58F37',
}

N_OBJS_DEFAULT = [2, 6]


# ======================================================================
# Loading metrics
# ======================================================================
def _required_columns(metrics, modes):
    """Build the column set we need from the metrics CSV."""
    cols = {'paradigm', 'method', 'condition', 'n_obj', 'seed'}
    if modes is None:
        for m in metrics:
            cols.add(m['col'])
        cols.add('n_solutions')
        cols.add('n_dom')
    else:
        for mode in modes:
            for m in metrics:
                cols.add(f"{m['base']}_{mode}")
            cols.add(f'n_solutions_{mode}')
            cols.add(f'n_dom_{mode}')
            cols.add(f'n_infeasible_{mode}')
    return cols


def load_metrics(morl_csv, moea_csv, *, metrics, modes=None):
    """Concat the MORL and MOEA metrics CSVs, keeping only required columns.

    For unconstrained data (modes=None): expects plain columns like
    `hv_ratio`, `spacing_norm`, `n_solutions`.
    For constrained data (modes=['strict'] etc.): expects suffixed columns
    like `hv_ratio_strict`, `n_solutions_tolerant`.

    Missing files cause a warning (that paradigm's rows just won't appear);
    missing columns raise.
    """
    need = _required_columns(metrics, modes)
    parts = []
    for path in (morl_csv, moea_csv):
        if not os.path.exists(path):
            print(f'  ! missing {path} — bars for that paradigm will be empty')
            continue
        d = pd.read_csv(path)
        missing = need - set(d.columns)
        if missing:
            raise SystemExit(f'{path}: missing columns {missing}')
        parts.append(d[list(need)])
    if not parts:
        raise SystemExit('no metrics CSV available — nothing to plot')
    return pd.concat(parts, ignore_index=True)


# ======================================================================
# Conditions / style lookup
# ======================================================================
def conditions_present(df, method, n_obj, condition_order):
    """Conditions actually in df for (method, n_obj), in CONDITION_ORDER plus
    unknowns appended at the end with a warning."""
    sub = df[(df.method == method) & (df.n_obj == n_obj)]
    present = sorted(sub.condition.unique())
    ordered = [c for c in condition_order if c in present]
    leftover = [c for c in present if c not in condition_order]
    if leftover:
        print(f'  ! {method} n_obj={n_obj}: unrecognised conditions {leftover}')
    return ordered + leftover


def _style_for(cond, condition_style):
    return condition_style.get(cond, dict(alpha=0.45, hatch='...'))


# ======================================================================
# Metric summary printing
# ======================================================================
def print_metric_summary(df, *, metrics, n_objs, condition_order,
                         modes=None):
    """Tabular summary: one row per (paradigm, method, condition, n_obj),
    with mean/std for each enabled (metric, mode) pair."""
    if modes is None:
        mode_cols = [(m['col'], '') for m in metrics]
        dom_cols = [('n_dom', '')]
    else:
        mode_cols = [(f"{m['base']}_{mode}", mode)
                     for m in metrics for mode in modes]
        dom_cols = [(f'n_dom_{mode}', mode) for mode in modes]
    header = (f"{'paradigm':<9}{'method':<14}{'condition':<28}"
              f"{'n_obj':>6}{'n':>5}")
    for col, mode in dom_cols:
        tag = 'n_dom' if mode == '' else f'n_dom·{mode[:3]}'
        header += f"  {tag:>11}{'±std':>9}"
    for col, mode in mode_cols:
        tag = col if mode == '' else col.replace(f'_{mode}', f'·{mode[:3]}')
        header += f"  {tag[:14]:>14}{'±std':>10}"
    print(header)
    for n_obj in n_objs:
        for m in ORDER:
            sub = df[(df.method == m) & (df.n_obj == n_obj)]
            if sub.empty:
                continue
            paradigm = PARADIGM_OF[m]
            for c in conditions_present(df, m, n_obj, condition_order):
                csub = sub[sub.condition == c]
                line = (f"{paradigm:<9}{METHOD_LABEL[m]:<14}{c:<28}"
                        f"{n_obj:>6}{len(csub):>5}")
                for col, _mode in dom_cols:
                    v = csub[col].values
                    v = v[~np.isnan(v)]
                    if len(v) == 0:
                        line += f"  {'nan':>11}{'nan':>9}"
                    else:
                        s = v.std(ddof=1) if len(v) > 1 else 0.0
                        line += f"  {v.mean():>11.1f}{s:>9.1f}"
                for col, _mode in mode_cols:
                    v = csub[col].values
                    v = v[~np.isnan(v)]
                    if len(v) == 0:
                        line += f"  {'nan':>14}{'nan':>10}"
                    else:
                        s = v.std(ddof=1) if len(v) > 1 else 0.0
                        line += f"  {v.mean():>14.5f}{s:>10.5f}"
                print(line)


# ======================================================================
# Bar+CI panel — used for both metrics and counts
# ======================================================================
def _ci95_half_width(vals):
    """Half-width of the t-distribution 95% CI for the mean.
    Returns 0 if fewer than 2 non-NaN observations."""
    n = len(vals)
    if n < 2:
        return 0.0
    sem = float(vals.std(ddof=1)) / np.sqrt(n)
    try:
        from scipy.stats import t
        t_crit = float(t.ppf(0.975, n - 1))
    except ImportError:
        # Coarse fallback for n <= 30; large-sample z thereafter.
        _T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
              6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
              11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
              16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
              21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
              26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}
        t_crit = _T.get(n - 1, 1.96)
    return t_crit * sem


def _plot_bar_panel(ax, df, n_obj, *, metric_col, ylabel, title,
                    show_xticklabels, condition_style, condition_order,
                    methods=None):
    """One panel of side-by-side bars: bar height = mean across seeds;
    black error bars span the t-distribution 95% CI of the mean.

    `methods` selects which method slots to draw (default: all of ORDER).
    Pass MOEA_METHODS or MORL_METHODS to show a single paradigm.

    Reading rule: error bars that do NOT overlap correspond (roughly) to
    means that differ significantly at p<0.05. Overlap doesn't necessarily
    mean non-significant — use a paired test for the formal call — but
    non-overlap is a strong visual signal.
    """
    methods = list(methods) if methods is not None else list(ORDER)
    SLOT_W = 0.9
    for i, method in enumerate(methods):
        conds = conditions_present(df, method, n_obj, condition_order)
        if not conds:
            continue
        color = PARADIGM_COLOR[PARADIGM_OF[method]]
        bw = SLOT_W / len(conds)
        for ci_idx, cond in enumerate(conds):
            vals = df[(df.method == method) &
                      (df.condition == cond) &
                      (df.n_obj == n_obj)][metric_col].values
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            bx = i + (ci_idx - (len(conds) - 1) / 2.0) * bw
            style = _style_for(cond, condition_style)
            mean = float(vals.mean())
            ax.bar(bx, mean, width=bw * 0.92, color=color,
                   alpha=style['alpha'], edgecolor='white', linewidth=0.6,
                   hatch=style['hatch'], zorder=2)
            half = _ci95_half_width(vals)
            if half > 0:
                ax.errorbar(bx, mean, yerr=half, fmt='none',
                            ecolor='0.15', elinewidth=0.9,
                            capsize=3.0, capthick=0.9, zorder=3)
    n_morl = sum(1 for m in methods if PARADIGM_OF[m] == 'MORL')
    if 0 < n_morl < len(methods):
        ax.axvline(n_morl - 0.5, color='0.6', linestyle=':',
                   linewidth=1.0)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(np.arange(len(methods)))
    if show_xticklabels:
        ax.set_xticklabels([METHOD_LABEL[m] for m in methods],
                           rotation=20, ha='right')figure_metrics_separate
    else:
        ax.set_xticklabels([])
    ax.grid(axis='y', linestyle=':', linewidth=0.7, alpha=0.7)
    ax.spines[['top', 'right']].set_visible(False)


# ======================================================================
# Legend
# ======================================================================
def legend_handles(df, *, condition_style, condition_label, condition_order):
    """One legend Patch per (paradigm, condition) actually plotted, plus
    a thin black bar showing what the error bars mean."""
    conditions_used = {p: set() for p in ('MORL', 'MOEA')}
    for _, r in df.iterrows():
        conditions_used[r.paradigm].add(r.condition)

    handles, seen = [], set()
    for cond in condition_order + sorted(
            c for cset in conditions_used.values()
            for c in cset if c not in condition_order):
        for paradigm in ('MORL', 'MOEA'):
            if cond not in conditions_used[paradigm]:
                continue
            if (paradigm, cond) in seen:
                continue
            seen.add((paradigm, cond))
            style = _style_for(cond, condition_style)
            handles.append(Patch(
                facecolor=PARADIGM_COLOR[paradigm],
                alpha=style['alpha'], hatch=style['hatch'],
                edgecolor='white',
                label=f'{paradigm} — {condition_label.get(cond, cond)}'))
    handles.append(Line2D([0], [0], color='0.15', linewidth=1.2,
                          marker='_', markersize=8, markeredgewidth=1.2,
                          label='95% CI of the mean'))
    return handles


# ======================================================================
# Figures: metrics (HV / spacing / etc.)
# ======================================================================
def figure_metrics(df, *, metrics, n_objs, condition_style, condition_label,
                   condition_order, modes=None, methods=None, suptitle='',
                   out_path=None):
    """Standard fig1.

    Layout: (len(metrics) × len(modes)) rows, len(n_objs) columns.
    - modes=None (unconstrained): one row per metric (2×n_objs typically)
    - modes=['strict'] / ['tolerant']: same shape, different columns
    - modes=['strict', 'tolerant']: 2×n_metrics rows (e.g. 4×2)

    `methods` restricts which methods (and hence paradigms) are drawn —
    e.g. methods=MOEA_METHODS for an MOEA-only figure. Default: all of ORDER.
    """
    if methods is not None:
        df = df[df.method.isin(methods)]
    if modes is None:
        plan = [(m['col'], m['ylabel'], m['better']) for m in metrics]
    else:
        plan = [(f"{m['base']}_{mode}",
                 f"{m['ylabel']} — {mode}", m['better'])
                for m in metrics for mode in modes]
    n_rows = len(plan)

    fig, axes = plt.subplots(n_rows, len(n_objs),
                             figsize=(13, 4.6 * n_rows), squeeze=False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12.5, fontweight='bold', y=1.00)

    for r, (col, ylabel_base, better) in enumerate(plan):
        ylabel = f'{ylabel_base}  ({better} is better)'
        is_bottom = (r == n_rows - 1)
        for c, n_obj in enumerate(n_objs):
            title = f'{n_obj}-objective' if r == 0 else ''
            _plot_bar_panel(axes[r][c], df, n_obj,
                            metric_col=col, ylabel=ylabel, title=title,
                            show_xticklabels=is_bottom,
                            condition_style=condition_style,
                            condition_order=condition_order,
                            methods=methods)

    handles = legend_handles(df, condition_style=condition_style,
                             condition_label=condition_label,
                             condition_order=condition_order)
    fig.legend(handles=handles, loc='lower center',
               ncol=min(len(handles), 4),
               bbox_to_anchor=(0.5, -0.04 / max(1, n_rows / 2)),
               frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 0.99])
    _save(fig, out_path)
    # Figure is auto-displayed by the matplotlib inline backend; returning
    # nothing prevents the cell-final-expression double-render in Jupyter.
    return None


def figure_metrics_separate(df, *, metrics, out_path=None, suptitle='', **kwargs):
    """Render each metric as its own figure (e.g. HV and spacing as two
    separate pictures) instead of stacked rows in one figure.

    Each metric is passed to figure_metrics on its own, so any modes
    (strict/tolerant) still stack within that metric's figure. If `out_path`
    is given, the metric column is inserted before the extension, e.g.
    out_path='fig1_det_tree.png' -> 'fig1_det_tree_hv_ratio.png' and
    'fig1_det_tree_spacing_norm.png'. All other arguments (n_objs,
    condition_style, condition_label, condition_order, modes) pass through."""
    for m in metrics:
        op = None
        if out_path:
            root, ext = os.path.splitext(out_path)
            op = f'{root}_{m["col"]}{ext}'
        st = f'{suptitle} — {m["ylabel"]}' if suptitle else ''
        figure_metrics(df, metrics=[m], suptitle=st, out_path=op, **kwargs)
    return None


# ======================================================================
# Figures: solution counts (n_solutions)
# ======================================================================
def figure_counts(df, *, n_objs, condition_style, condition_label,
                  condition_order, modes=None, methods=None, suptitle='',
                  out_path=None):
    """Standard fig2. With modes=None uses `n_solutions`; with modes uses
    `n_solutions_{mode}` and stacks one row per mode.

    `methods` restricts which methods (and hence paradigms) are drawn —
    e.g. methods=MOEA_METHODS for an MOEA-only figure. Default: all of ORDER.
    """
    if methods is not None:
        df = df[df.method.isin(methods)]
    if modes is None:
        plan = [('n_solutions', 'Number of solutions on front')]
    else:
        plan = [(f'n_solutions_{mode}',
                 f'Number of solutions on front ({mode})')
                for mode in modes]
    n_rows = len(plan)

    fig, axes = plt.subplots(n_rows, len(n_objs),
                             figsize=(13, 4.6 * n_rows), squeeze=False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12.5, fontweight='bold', y=1.00)

    for r, (col, ylabel) in enumerate(plan):
        is_bottom = (r == n_rows - 1)
        for c, n_obj in enumerate(n_objs):
            title = f'{n_obj}-objective' if r == 0 else ''
            _plot_bar_panel(axes[r][c], df, n_obj,
                            metric_col=col, ylabel=ylabel, title=title,
                            show_xticklabels=is_bottom,
                            condition_style=condition_style,
                            condition_order=condition_order,
                            methods=methods)

    handles = legend_handles(df, condition_style=condition_style,
                             condition_label=condition_label,
                             condition_order=condition_order)
    fig.legend(handles=handles, loc='lower center',
               ncol=min(len(handles), 4),
               bbox_to_anchor=(0.5, -0.04 / max(1, n_rows / 2)),
               frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 0.99])
    _save(fig, out_path)
    # Figure is auto-displayed by the matplotlib inline backend; returning
    # nothing prevents the cell-final-expression double-render in Jupyter.
    return None


# ======================================================================
# Convergence: loading + aggregation + plotting
# ======================================================================
def _file_ref(fname):
    """Trailing `_<int>.csv` -> ref; None if filename has no ref suffix."""
    n_trailing_nums = len(re.findall(r'_(\d+)(?=(?:_\d+)*\.csv$)', fname))
    if n_trailing_nums < 2:
        return None
    return int(re.search(r'_(\d+)\.csv$', fname).group(1))


def load_convergences(base, *, folder_re, condition_format,
                      csv_prefix, x_col, y_col, dropna_y=False):
    """Walk <base>/<config>/seed<N>/<csv_prefix>*.csv and yield one record
    per CSV. Each record: dict with condition, algo, n_obj, seed, ref, df
    (where df has canonical columns 'nfe' and 'epsilon_progress').

    `condition_format(groups)` -> (condition, method, n_obj)
        Tells the loader how to interpret regex groups for this env.
    `x_col`, `y_col` are the source column names (the loader renames them
        to canonical 'nfe' / 'epsilon_progress' so the plotter is generic).
    `dropna_y` drops rows where y is NaN (needed for MORL pcs_shift_eps —
        the first checkpoint has NaN because there's no prior PCS to diff).
    """
    records = []
    if not os.path.isdir(base):
        return records
    for d in sorted(os.listdir(base)):
        m = folder_re.match(d)
        if not m:
            continue
        condition, method, n_obj = condition_format(m.groups())
        config_dir = os.path.join(base, d)
        for seed_name in sorted(os.listdir(config_dir)):
            sd = os.path.join(config_dir, seed_name)
            if not os.path.isdir(sd):
                continue
            seed_m = re.search(r'seed(\d+)', seed_name)
            if not seed_m:
                continue
            seed = int(seed_m.group(1))
            for f in sorted(os.listdir(sd)):
                if not (f.startswith(csv_prefix) and f.endswith('.csv')):
                    continue
                df_c = pd.read_csv(os.path.join(sd, f))
                if x_col not in df_c.columns or y_col not in df_c.columns:
                    continue
                if dropna_y:
                    df_c = df_c.dropna(subset=[y_col])
                shaped = df_c[[x_col, y_col]].rename(
                    columns={x_col: 'nfe', y_col: 'epsilon_progress'})
                records.append({
                    'condition': condition, 'algo': method, 'n_obj': n_obj,
                    'seed': seed, 'ref': _file_ref(f), 'df': shaped,
                })
    return records


def aggregate_curves(records, condition, algo, n_obj, n_grid=200):
    """Mean ± std curves across seeds on a common nfe grid."""
    matching = [r for r in records
                if r['condition'] == condition and r['algo'] == algo
                and r['n_obj'] == n_obj]
    if not matching:
        return None, None, None
    if len(matching) == 1:
        df = matching[0]['df']
        return (df['nfe'].values,
                df['epsilon_progress'].values,
                np.zeros(len(df)))
    max_nfe = max(r['df']['nfe'].max() for r in matching)
    grid = np.linspace(0, max_nfe, n_grid)
    arr = np.array([np.interp(grid, r['df']['nfe'].values,
                              r['df']['epsilon_progress'].values)
                    for r in matching])
    return grid, arr.mean(axis=0), arr.std(axis=0)


def figure_convergence(records, *, methods, conditions, n_objs,
                       cond_labels=None, xlabel='NFE',
                       ylabel='ε-progress', suptitle='', out_path=None):
    """Convergence figure: conditions × n_obj grid of lines (mean ± shaded std
    across seeds). One line per method in `methods`."""
    cond_labels = cond_labels or {c: c for c in conditions}
    fig, axes = plt.subplots(len(conditions), len(n_objs),
                             figsize=(11, 3.2 * len(conditions)),
                             squeeze=False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12.5, fontweight='bold', y=1.00)

    for i, condition in enumerate(conditions):
        for j, n_obj in enumerate(n_objs):
            ax = axes[i][j]
            has_data = False
            panel_max_x = 0.0
            for method in methods:
                grid, mean, std = aggregate_curves(records, condition,
                                                   method, n_obj)
                if grid is None:
                    continue
                has_data = True
                panel_max_x = max(panel_max_x, float(grid.max()))
                color = ALGO_LINE_COLOR[method]
                ax.plot(grid, mean, color=color, linewidth=1.6,
                        label=METHOD_LABEL[method])
                if (std > 0).any():
                    ax.fill_between(grid, mean - std, mean + std,
                                    color=color, alpha=0.2, linewidth=0)
            ax.set_title(f'{cond_labels.get(condition, condition)} '
                         f'— {n_obj}-objective',
                         fontsize=10, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(linestyle=':', linewidth=0.7, alpha=0.7)
            ax.spines[['top', 'right']].set_visible(False)
            # Explicit per-panel x-range: 0 to the longest algorithm's max.
            # Each algorithm's curve naturally ends at its own training
            # budget, so shorter-running ones simply terminate earlier.
            if has_data:
                ax.set_xlim(0, panel_max_x)
            ax.set_ylim(bottom=0)
            if has_data:
                ax.legend(fontsize=8, loc='best', frameon=False)

    fig.tight_layout()
    _save(fig, out_path)
    # Figure is auto-displayed by the matplotlib inline backend; returning
    # nothing prevents the cell-final-expression double-render in Jupyter.
    return None


# ======================================================================
# Runtime collection
# ======================================================================
def hms_to_seconds(s):
    h, m, sec = str(s).split(':')
    return int(h) * 3600 + int(m) * 60 + int(sec)


def fmt_seconds(s):
    if pd.isna(s):
        return '--'
    s = int(s)
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f'{h:d}h{m:02d}m{sec:02d}s' if h else f'{m:d}m{sec:02d}s'


def collect_runtimes(base, *, folder_re, paradigm, condition_format,
                     csv_prefix, mode='per_file'):
    """Walk seeds, extract runtime per convergence file.

      mode='per_file': one row per (config, seed, ref) — typically what
                       you want for det runs and per-ref-file analysis.
      mode='per_seed': one row per (config, seed) — take MAX `time` across
                       all ref files. Use for robust runs whose ref files
                       log cumulative wall-clock from job start.
    """
    rows = []
    if not os.path.isdir(base):
        return rows
    for d in sorted(os.listdir(base)):
        m = folder_re.match(d)
        if not m:
            continue
        condition, method, n_obj = condition_format(m.groups())
        cfg_dir = os.path.join(base, d)
        for seed_name in sorted(os.listdir(cfg_dir)):
            sd = os.path.join(cfg_dir, seed_name)
            if not os.path.isdir(sd):
                continue
            seed_m = re.search(r'seed(\d+)', seed_name)
            if not seed_m:
                continue
            seed = int(seed_m.group(1))
            conv_files = sorted(f for f in os.listdir(sd)
                                if f.startswith(csv_prefix)
                                and f.endswith('.csv'))
            if mode == 'per_file':
                for f in conv_files:
                    df_c = pd.read_csv(os.path.join(sd, f))
                    if 'time' not in df_c.columns or len(df_c) == 0:
                        continue
                    rows.append({
                        'paradigm': paradigm, 'method': method,
                        'condition': condition, 'n_obj': n_obj,
                        'seed': seed, 'ref': _file_ref(f),
                        'runtime_seconds':
                            hms_to_seconds(df_c['time'].iloc[-1]),
                    })
            elif mode == 'per_seed':
                best = None
                for f in conv_files:
                    df_c = pd.read_csv(os.path.join(sd, f))
                    if 'time' not in df_c.columns or len(df_c) == 0:
                        continue
                    secs = hms_to_seconds(df_c['time'].iloc[-1])
                    if best is None or secs > best:
                        best = secs
                if best is not None:
                    rows.append({
                        'paradigm': paradigm, 'method': method,
                        'condition': condition, 'n_obj': n_obj,
                        'seed': seed, 'runtime_seconds': best,
                    })
            else:
                raise ValueError(f"mode must be 'per_file' or 'per_seed', "
                                 f"got {mode!r}")
    return rows


def summarize_runtimes(runtime_df, *, title='', metrics_df=None, modes=None,
                       count_bases=('n_infeasible', 'n_dom', 'n_solutions')):
    """Group by (paradigm, method, condition, n_obj) and print mean/std/total
    formatted as HMS. Returns the summary frame for further use.

    If `metrics_df` is given (the frame from load_metrics), the mean/std of
    each count column is merged in per cell and appended. Count column names
    are built from `count_bases` and `modes`, matching the suffix convention
    used elsewhere — the suffix is never hardcoded here. With modes=None the
    bare bases are used (`n_dom`, `n_solutions`; `n_infeasible` has no
    unconstrained equivalent and is skipped when absent). With modes=['strict']
    they become `n_infeasible_strict`, `n_dom_strict`, `n_solutions_strict`,
    and likewise for any other mode. Pass the same `modes` you gave
    load_metrics."""
    if title:
        print(f'\n=== {title} ===\n')
    if len(runtime_df) == 0:
        print('  (no runtime rows)')
        return runtime_df
    keys = ['paradigm', 'method', 'condition', 'n_obj']
    s = (runtime_df
         .groupby(keys)
         .runtime_seconds
         .agg(['count', 'mean', 'std'])
         .round(0).reset_index())
    s['mean_hms']  = s['mean'].apply(fmt_seconds)
    s['std_hms']   = s['std'].fillna(0).apply(fmt_seconds)
    s['total_hms'] = (s['mean'] * s['count']).apply(fmt_seconds)
    display = ['paradigm', 'method', 'condition', 'n_obj', 'count',
               'mean_hms', 'std_hms', 'total_hms']
    if metrics_df is not None:
        if modes is None:
            wanted = list(count_bases)
        else:
            wanted = [f'{b}_{mode}' for mode in modes for b in count_bases]
        present = [c for c in wanted if c in metrics_df.columns]
        if not present:
            print(f'  ! none of {wanted} found in metrics_df — counts skipped')
        else:
            g = metrics_df.groupby(keys)[present].agg(['mean', 'std'])
            # Flatten ('n_dom','mean') -> 'n_dom_mean', etc.
            g.columns = [f'{col}_{stat}' for col, stat in g.columns]
            g = g.reset_index()
            for c in present:
                g[f'{c}_mean'] = g[f'{c}_mean'].round(1)
                g[f'{c}_std'] = g[f'{c}_std'].fillna(0).round(1)
            s = s.merge(g, on=keys, how='left')
            for c in present:
                display += [f'{c}_mean', f'{c}_std']
    # Return the display-only frame (no raw-seconds duplicates). The notebook
    # renders this single table; we don't also print it as text, which would
    # produce a second, redundant table in Jupyter.
    return s[display].reset_index(drop=True)


# ======================================================================
# Convergence diagnostics (NFE for MOEA, steps for MORL)
# ======================================================================
def _diag_seeds(cfg_dir):
    return [d for d in sorted(os.listdir(cfg_dir))
            if os.path.isdir(os.path.join(cfg_dir, d))
            and re.search(r'seed\d+', d)]


def _moea_verdict(gain):
    if np.isnan(gain):
        return '?'
    if gain < 0.02:
        return 'converged'
    if gain < 0.05:
        return 'borderline'
    return 'NOT converged'


def _diag_moea(base, folder_re, fmt, prefix='convergences_'):
    """Per-cell NFE-convergence table from cumulative epsilon_progress.

    gain_last20pct (fraction of final epsilon_progress gained in the last 20%
    of the budget) is the key signal: ~0 => settled, large => still climbing.
    """
    rows = []
    if not os.path.isdir(base):
        print(f'  ! missing {base}')
        return pd.DataFrame()
    for d in sorted(os.listdir(base)):
        m = folder_re.match(d)
        if not m:
            continue
        cond, method, n_obj = fmt(m.groups())
        cfg = os.path.join(base, d)
        per_seed = {}
        for sd in _diag_seeds(cfg):
            for f in glob.glob(os.path.join(cfg, sd, f'{prefix}*.csv')):
                c = pd.read_csv(f)
                if 'epsilon_progress' not in c or 'nfe' not in c or len(c) < 2:
                    continue
                ep = c['epsilon_progress'].values.astype(float)
                nfe = c['nfe'].values.astype(float)
                final, budget = ep[-1], nfe[-1]
                if final <= 0:
                    nfe95 = nfe99 = 0.0
                    gain = 0.0
                else:
                    nfe95 = nfe[np.argmax(ep >= 0.95 * final)]
                    nfe99 = nfe[np.argmax(ep >= 0.99 * final)]
                    cut = nfe >= 0.8 * budget
                    gain = (final - ep[cut][0]) / final
                per_seed.setdefault(sd, []).append(
                    (budget, nfe95, nfe99, gain, final))
        agg = [np.mean(v, axis=0) for v in per_seed.values()]
        if not agg:
            continue
        a = np.array(agg)
        rows.append(dict(method=method, condition=cond, n_obj=n_obj,
                         seeds=len(agg),
                         budget=int(a[:, 0].mean()),
                         nfe95=int(a[:, 1].mean()),
                         nfe99=int(a[:, 2].mean()),
                         gain_last20pct=round(a[:, 3].mean(), 4),
                         final_eps=round(a[:, 4].mean(), 1),
                         verdict=_moea_verdict(a[:, 3].mean())))
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
            .sort_values(['condition', 'n_obj', 'method'])
            .reset_index(drop=True))


def _diag_morl(base, folder_re, fmt, prefix='convergence_', tau=1e-6,
               y_col='pcs_shift_eps'):
    """Per-cell step-convergence table from a shift metric -> 0.

    `y_col` selects the convergence column: 'pcs_shift_eps' (worst-case /
    max-aggregated, the default) or 'pcs_shift_eps_mean' (bulk-sensitive,
    available only after re-running training with the updated pql.py).
    conv_step is the first checkpoint from which every later value is <= tau
    (i.e. stays converged). A cell is 'converged' only if all seeds reach
    that; otherwise final_shift shows how far it stayed from zero.
    """
    rows = []
    if not os.path.isdir(base):
        print(f'  ! missing {base}')
        return pd.DataFrame()
    missing_col = False
    for d in sorted(os.listdir(base)):
        m = folder_re.match(d)
        if not m:
            continue
        cond, method, n_obj = fmt(m.groups())
        cfg = os.path.join(base, d)
        per = []
        for sd in _diag_seeds(cfg):
            fs = sorted(glob.glob(os.path.join(cfg, sd, f'{prefix}*.csv')))
            if not fs:
                continue
            c = pd.read_csv(fs[0])
            if 'timestep' not in c or y_col not in c:
                if y_col not in c:
                    missing_col = True
                continue
            cc = c.dropna(subset=[y_col])
            if len(cc) == 0:
                continue
            ts = cc['timestep'].values.astype(float)
            sh = cc[y_col].values.astype(float)
            budget = float(c['timestep'].values[-1])
            below = sh <= tau
            conv = np.nan
            for i in range(len(sh)):
                if below[i:].all():
                    conv = ts[i]
                    break
            per.append((budget, conv, sh[-1]))
        if not per:
            continue
        a = np.array(per, dtype=float)
        n_conv = int(np.sum(~np.isnan(a[:, 1])))
        rows.append(dict(method=method, condition=cond, n_obj=n_obj,
                         metric=y_col,
                         seeds=len(per),
                         budget=int(a[:, 0].mean()),
                         conv_step=(int(np.nanmean(a[:, 1])) if n_conv else None),
                         n_converged=n_conv,
                         final_shift=round(a[:, 2].mean(), 4),
                         verdict=('converged' if n_conv == len(per)
                                  else 'NOT converged')))
    if missing_col and not rows:
        print(f"  ! column {y_col!r} not in the convergence files "
              f"(re-run training with the updated pql.py to log it)")
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
            .sort_values(['condition', 'n_obj', 'method'])
            .reset_index(drop=True))


def diagnose_convergence(sections, *, title=''):
    """Run NFE/step convergence diagnostics for one benchmark, then summarise.

    `sections` is a list of dicts, one per (paradigm, setting) scan, e.g.:
        {'label': 'Robust MOEA', 'kind': 'moea',
         'base': f'{ROOT}/robust/moea',
         'folder_re': MOEA_ROB, 'fmt': f_moea_rob}      # 'prefix' optional
    `kind` is 'moea' (cumulative epsilon_progress vs nfe) or 'morl'
    (a shift metric vs timestep). For 'morl' sections, optional 'morl_y'
    selects the convergence column — 'pcs_shift_eps' (default, worst-case) or
    'pcs_shift_eps_mean' (bulk-sensitive, present only after re-running
    training with the updated pql.py). Each section's table is printed, then a
    combined summary flags the slowest cell and any non-converged cells.
    Returns {label: DataFrame}. Missing directories are skipped, not fatal,
    so the same call works on partial data."""
    if title:
        print(f'\n########## {title} ##########')
    out, moea_frames, morl_frames = {}, [], []
    for sec in sections:
        kind = sec['kind']
        prefix = sec.get('prefix',
                         'convergences_' if kind == 'moea' else 'convergence_')
        print(f'\n=== {sec["label"]} ===')
        if kind == 'moea':
            df = _diag_moea(sec['base'], sec['folder_re'], sec['fmt'], prefix)
            if len(df):
                moea_frames.append(df)
        elif kind == 'morl':
            df = _diag_morl(sec['base'], sec['folder_re'], sec['fmt'], prefix,
                            y_col=sec.get('morl_y', 'pcs_shift_eps'))
            if len(df):
                morl_frames.append(df)
        else:
            raise ValueError(f"section kind must be 'moea' or 'morl', "
                             f"got {kind!r}")
        print(df.to_string(index=False) if len(df) else '  (no data)')
        out[sec['label']] = df

    print('\n--- summary ---')
    if moea_frames:
        allm = pd.concat(moea_frames, ignore_index=True)
        w = allm.loc[allm['nfe99'].idxmax()]
        print(f"MOEA: largest nfe99 = {w['nfe99']} (budget {w['budget']}) "
              f"at {w['method']} / {w['condition']} / {w['n_obj']}-obj")
        bad = allm[allm['verdict'] != 'converged']
        if len(bad):
            print('  not fully converged:')
            print(bad[['method', 'condition', 'n_obj', 'budget', 'nfe99',
                       'gain_last20pct', 'verdict']].to_string(index=False))
        else:
            print('  all MOEA cells converged within budget.')
    if morl_frames:
        allr = pd.concat(morl_frames, ignore_index=True)
        conv = allr.dropna(subset=['conv_step'])
        if len(conv):
            w = conv.loc[conv['conv_step'].idxmax()]
            print(f"MORL: largest conv_step = {int(w['conv_step'])} "
                  f"(budget {w['budget']}) at {w['method']} / "
                  f"{w['condition']} / {w['n_obj']}-obj")
        bad = allr[allr['verdict'] != 'converged']
        if len(bad):
            print('  not converged within budget:')
            print(bad[['metric', 'method', 'condition', 'n_obj', 'budget',
                       'final_shift', 'verdict']].to_string(index=False))
        else:
            print('  all MORL cells converged within budget.')
    return out


# ======================================================================
# Internal: save helper
# ======================================================================
def _save(fig, out_path):
    if not out_path:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved {out_path}')