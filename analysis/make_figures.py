"""Build every report figure and its plotted data from frozen results.

Run from anywhere:  python analysis/make_figures.py
Needs matplotlib, numpy, pandas. Reads experiments/ and data/ only; writes
figures/<name>.png|.pdf and analysis/tables/<name>.csv (the exact numbers
plotted). No training, GPU or network access.

Aggregation rules (same as the frozen summary tables):
- per training seed (or ANALYZE repeat / MC run) compute quantiles over queries,
  then show the median across repeats with a min-max range;
- the exact-marginal and 10k uniform baselines are deterministic and repeated
  in every run directory; they are de-duplicated (taken once per scale);
- seed ranges are descriptive min-max, never confidence intervals.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments'
WL = ROOT / 'data/snapshot_resume_20261109_172915/workloads'
FIG = ROOT / 'figures'
TAB = ROOT / 'analysis/tables'
FIG.mkdir(exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

SCALES = ['1000000', '5000000', 'full']
SCALE_LABEL = {'1000000': '1M rows', '5000000': '5M rows', 'full': 'Full (12.5M rows)'}
SCALE_SHORT = {'1000000': '1M', '5000000': '5M', 'full': 'Full'}
ROWS = {'1000000': 1_000_000, '5000000': 5_000_000, 'full': 12_540_479}

# Fixed categorical order (validated reference palette, light mode). Colour
# follows the method everywhere; a ninth method uses neutral grey + its own marker.
METHODS = {
    'own_made':                    ('Own MADE',                 '#2a78d6', 'o'),
    'own_resmade':                 ('Own ResMADE',              '#eb6834', 's'),
    'official_resmade':            ('Official ResMADE',         '#1baf7a', 'D'),
    'exact_marginal_independence': ('Independence (exact marg.)', '#eda100', 'v'),
    'uniform_sample':              ('Uniform sample 10k rows',  '#e87ba4', '^'),
    'PG-default':                  ('PG default (stat 100)',    '#008300', 'P'),
    'PG-target1000':               ('PG stat 1000',             '#6250d6', 'X'),
    'PG-extstats':                 ('PG 1000 + ext. stats',     '#e34948', '*'),  # star drawn larger below
    'uniform_matched':             ('Uniform sample, storage-matched', '#8a8985', 'h'),
}
MODELS = ['own_made', 'own_resmade', 'official_resmade']
ALIASES = {'uniform_sample_10000': 'uniform_sample',
           'uniform_matched_own_resmade': 'uniform_matched',
           'matched_own_resmade': 'uniform_matched'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9.5,
    'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'lines.linewidth': 2, 'lines.markersize': 6,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'pdf.fonttype': 42,
})


def label(m):
    return METHODS[m][0]


def color(m):
    return METHODS[m][1]


def marker(m):
    return METHODS[m][2]


STAR_SCALE = 1.6  # '*' renders visibly smaller than other markers


def save(fig, name, table=None):
    fig.savefig(FIG / f'{name}.png', dpi=200, bbox_inches='tight')
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    if table is not None:
        table.to_csv(TAB / f'{name}.csv', index=False)
    print('wrote', name)


def method_legend(ax_or_fig, methods, **kw):
    handles = [Line2D([], [], color=color(m), marker=marker(m), linestyle='none',
                      markersize=6, label=label(m)) for m in methods]
    return ax_or_fig.legend(handles=handles, frameon=False, **kw)


# --------------------------------------------------------------------------
# Raw per-query loading (main workload)
# --------------------------------------------------------------------------
_PQ = {}


def per_query(scale, model, seed):
    key = (scale, model, seed)
    if key not in _PQ:
        _PQ[key] = pd.read_csv(EXP / f'main_evaluation_20261110/{scale}_{model}_seed{seed}/per_query.csv')
    return _PQ[key]


def model_queries(scale, model, seed):
    d = per_query(scale, model, seed)
    return d[d.method == model].set_index('query_id')


def baseline_queries(scale, method):
    d = per_query(scale, 'own_made', 0)  # baselines are identical in every run dir
    return d[d.method == method].set_index('query_id')


def workload_meta(scale):
    rows = []
    with open(WL / scale / 'main_native.jsonl', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            rows.append({'query_id': q['query_id'], 'predicate_count': len(q['predicates'])})
    return pd.DataFrame(rows).set_index('query_id')


def q(values, p):
    return float(np.quantile(np.asarray(values, dtype=float), p))


# --------------------------------------------------------------------------
# Fig 1 training curves
# --------------------------------------------------------------------------
def fig_training_curves():
    d = pd.read_csv(EXP / 'main_matrix_20261109/training_curves.csv')
    agg = (d.groupby(['scale', 'model', 'epoch']).mean_training_nll_nats
             .agg(['median', 'min', 'max']).reset_index())
    agg['scale'] = agg.scale.astype(str)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), sharey=True)
    for ax, s in zip(axes, SCALES):
        for m in MODELS:
            a = agg[(agg.scale == s) & (agg.model == m)]
            ax.fill_between(a.epoch, a['min'], a['max'], color=color(m), alpha=0.18, linewidth=0)
            ax.plot(a.epoch, a['median'], color=color(m), marker=marker(m), markersize=3.5,
                    markevery=3, label=label(m))
        ax.set_title(SCALE_LABEL[s])
        ax.set_xlabel('Epoch')
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_yscale('log')
        ax.set_ylim(13.8, 30)
        ax.set_yticks([14, 16, 18, 20, 25, 30])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylabel('Mean training NLL (nats/row)')
    method_legend(fig, MODELS, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.14))
    fig.text(0.5, -0.17, 'Line = median of 3 training seeds, band = min–max. ResMADE loss is computed '
             'with random column masking; MADE has none, so the two families are not on the same objective.',
             ha='center', fontsize=7, color=INK2)
    save(fig, 'fig01_training_curves', agg)


# --------------------------------------------------------------------------
# Fig 2 main accuracy: median / p95 / p99 for all estimators
# --------------------------------------------------------------------------
def main_accuracy_table():
    rows = []
    sr = pd.read_csv(EXP / 'main_evaluation_20261110/metrics_seed_ranges.csv')
    sr['scale'] = sr.scale.astype(str)
    for _, r in sr.iterrows():
        m = ALIASES.get(r.model, r.model)
        for k, p in [('median', 'p50'), ('p95', 'p95'), ('p99', 'p99')]:
            rows.append({'scale': r.scale, 'method': m, 'quantile': p,
                         'value': r[f'qerror_{k}_seed_median'], 'lo': r[f'qerror_{k}_seed_min'],
                         'hi': r[f'qerror_{k}_seed_max'], 'repeats': int(r.independent_training_seeds) or 1})
    pg = pd.read_csv(EXP / 'supplement_analysis_20261110/pg_metrics_configuration_ranges.csv')
    pg['scale'] = pg.scale.astype(str)
    for _, r in pg[pg.scope == 'main_native'].iterrows():
        for k, p in [('median', 'p50'), ('p95', 'p95'), ('p99', 'p99')]:
            rows.append({'scale': r.scale, 'method': r.method, 'quantile': p,
                         'value': r[f'qerror_{k}_repeat_median'], 'lo': r[f'qerror_{k}_repeat_min'],
                         'hi': r[f'qerror_{k}_repeat_max'], 'repeats': int(r.analyze_repetitions)})
    mc = pd.read_csv(EXP / 'supplement_analysis_20261110/metrics_configuration_ranges.csv')
    mc['scale'] = mc.scale.astype(str)
    for _, r in mc[(mc.experiment == 'matched_sampling_20261110') & (mc.model == 'uniform_matched_own_resmade')].iterrows():
        for k, p in [('median', 'p50'), ('p95', 'p95'), ('p99', 'p99')]:
            rows.append({'scale': r.scale, 'method': 'uniform_matched', 'quantile': p,
                         'value': r[f'qerror_{k}_run_median'], 'lo': r[f'qerror_{k}_run_min'],
                         'hi': r[f'qerror_{k}_run_max'], 'repeats': int(r.runs)})
    return pd.DataFrame(rows)


ORDER_ALL = ['own_made', 'own_resmade', 'official_resmade', 'uniform_matched', 'PG-extstats',
             'PG-target1000', 'PG-default', 'exact_marginal_independence', 'uniform_sample']


def fig_main_accuracy():
    t = main_accuracy_table()
    qmark = {'p50': 'o', 'p95': 's', 'p99': '^'}
    qoff = {'p50': -0.22, 'p95': 0.0, 'p99': 0.22}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), sharey=True)
    for ax, s in zip(axes, SCALES):
        for i, m in enumerate(ORDER_ALL):
            for p in ['p50', 'p95', 'p99']:
                r = t[(t.scale == s) & (t.method == m) & (t['quantile'] == p)]
                if r.empty:
                    continue
                r = r.iloc[0]
                x = i + qoff[p]
                ax.errorbar(x, r.value, yerr=[[r.value - r.lo], [r.hi - r.value]], fmt='none',
                            ecolor=color(m), elinewidth=1.2, capsize=2)
                ax.plot(x, r.value, marker=qmark[p], color=color(m), markersize=5.5,
                        markeredgecolor='white', markeredgewidth=0.8, linestyle='none')
        ax.set_yscale('log')
        ax.set_xticks(range(len(ORDER_ALL)))
        ax.set_xticklabels([label(m) for m in ORDER_ALL], rotation=55, ha='right', fontsize=7)
        ax.set_title(SCALE_LABEL[s])
        ax.axvline(2.5, color=GRID, linewidth=1)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('q-error (log scale)')
    handles = [Line2D([], [], color=INK2, marker=qmark[p], linestyle='none', label=n)
               for p, n in [('p50', 'median'), ('p95', '95th pct'), ('p99', '99th pct')]]
    fig.legend(handles=handles, frameon=False, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.text(0.5, -0.40, 'Main workload, 2,000 queries per scale. Learned models: median of 3 training seeds '
             '(bars = min–max). PG: 3 ANALYZE repeats. Storage-matched sample: 3 sample seeds, '
             'same bytes as own ResMADE parameters.', ha='center', fontsize=7, color=INK2)
    save(fig, 'fig02_main_accuracy', t)


# --------------------------------------------------------------------------
# Fig 3 survival curves P(q-error > x)
# --------------------------------------------------------------------------
def fig_survival():
    grid = np.logspace(0, 4, 160)
    out = []
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for ax, s in zip(axes, SCALES):
        series = []
        for m in MODELS:
            for seed in range(3):
                series.append((m, seed, model_queries(s, m, seed).qerror.values))
        for m in ['exact_marginal_independence', 'uniform_sample']:
            series.append((m, None, baseline_queries(s, m).qerror.values))
        for m, seed, v in series:
            surv = np.array([(v > x).mean() for x in grid])
            ax.step(grid, np.where(surv > 0, surv, np.nan), where='post', color=color(m),
                    linewidth=1.3 if seed is not None else 2,
                    alpha=0.85, label=label(m) if seed in (None, 0) else None)
            out += [{'scale': s, 'method': m, 'seed': seed, 'x': x, 'fraction_above': f}
                    for x, f in zip(grid[::8], surv[::8])]
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(4e-4, 1.05)
        ax.set_xlim(1, 1e4)
        ax.set_title(SCALE_LABEL[s])
        ax.set_xlabel('q-error threshold x')
    axes[0].set_ylabel('Fraction of queries with q-error > x')
    method_legend(fig, MODELS + ['exact_marginal_independence', 'uniform_sample'],
                  loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.15))
    fig.text(0.5, -0.19, 'Each learned model drawn once per training seed (3 thin lines); deterministic '
             'baselines drawn once. Curves end at the largest observed error (1/2000 = 5e-4).',
             ha='center', fontsize=7, color=INK2)
    save(fig, 'fig03_qerror_survival', pd.DataFrame(out))


# --------------------------------------------------------------------------
# Fig 4 paired own vs official ResMADE
# --------------------------------------------------------------------------
def fig_paired_resmade():
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9), sharey=True)
    bins = np.linspace(-0.25, 0.25, 51)
    for ax, s in zip(axes, SCALES):
        allr = []
        for seed in range(3):
            o = model_queries(s, 'own_resmade', seed).qerror
            f = model_queries(s, 'official_resmade', seed).qerror
            r = np.log10(o / f.loc[o.index])
            allr.append(r.values)
            rows.append({'scale': s, 'seed': seed, 'median_log10_ratio': float(np.median(r)),
                         'own_better_fraction': float((r < 0).mean()), 'tie_fraction': float((r == 0).mean()),
                         'p05_log10_ratio': q(r, .05), 'p95_log10_ratio': q(r, .95)})
        v = np.clip(np.concatenate(allr), bins[0], bins[-1])
        ax.hist(v, bins=bins, color=color('own_resmade'), edgecolor='white', linewidth=0.6)
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_title(SCALE_LABEL[s])
        ax.set_xlabel('log10( own / official q-error )')
        med = np.median(np.concatenate(allr))
        ax.text(0.97, 0.93, f'median {med:+.4f}\nown lower on {np.mean(np.concatenate(allr) < 0):.0%}',
                transform=ax.transAxes, ha='right', va='top', fontsize=7.5, color=INK2)
    axes[0].set_ylabel('Query-seed pairs (3 × 2,000)')
    fig.text(0.5, -0.1, 'Same query, same training seed, same sampler. Left of 0 = own ResMADE closer to '
             'the truth. Values beyond ±0.25 are clipped into the edge bins.', ha='center', fontsize=7, color=INK2)
    save(fig, 'fig04_paired_own_vs_official', pd.DataFrame(rows))


# --------------------------------------------------------------------------
# Fig 5 error by predicate count and selectivity (all five main-workload methods)
# --------------------------------------------------------------------------
SEL_BINS = [('<=1e-5', 0, 1e-5), ('(1e-5,1e-3]', 1e-5, 1e-3), ('(1e-3,1e-2]', 1e-3, 1e-2), ('>1e-2', 1e-2, 2)]


def grouped_rows():
    rows = []
    for s in SCALES:
        meta = workload_meta(s)
        def add(m, seed, d):
            d = d.join(meta)
            sel = d.truth / ROWS[s]
            for name, lo, hi in SEL_BINS:
                g = d[(sel > lo) & (sel <= hi)] if lo > 0 else d[sel <= hi]
                rows.append({'scale': s, 'method': m, 'seed': seed, 'grouping': 'selectivity', 'group': name,
                             'queries': len(g), 'p95': q(g.qerror, .95) if len(g) else np.nan})
            for k, g in d.groupby('predicate_count'):
                rows.append({'scale': s, 'method': m, 'seed': seed, 'grouping': 'predicate_count', 'group': str(k),
                             'queries': len(g), 'p95': q(g.qerror, .95)})
        for m in MODELS:
            for seed in range(3):
                add(m, seed, model_queries(s, m, seed))
        for m in ['exact_marginal_independence', 'uniform_sample']:
            add(m, None, baseline_queries(s, m))
    return pd.DataFrame(rows)


def fig_grouped():
    d = grouped_rows()
    agg = d.groupby(['scale', 'method', 'grouping', 'group']).agg(
        p95=('p95', 'median'), lo=('p95', 'min'), hi=('p95', 'max'), queries=('queries', 'first')).reset_index()
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharey='row')
    for row, (grouping, order, xlabel) in enumerate([
            ('predicate_count', [str(i) for i in range(5, 12)], 'Number of predicates'),
            ('selectivity', [b[0] for b in SEL_BINS], 'True selectivity (rows / table rows)')]):
        for ax, s in zip(axes[row], SCALES):
            for m in MODELS + ['exact_marginal_independence', 'uniform_sample']:
                a = agg[(agg.scale == s) & (agg.method == m) & (agg.grouping == grouping)].set_index('group').reindex(order)
                x = np.arange(len(order))
                ax.fill_between(x, a.lo, a.hi, color=color(m), alpha=0.18, linewidth=0)
                ax.plot(x, a.p95, color=color(m), marker=marker(m), markersize=4, label=label(m))
            n = agg[(agg.scale == s) & (agg.method == 'own_made') & (agg.grouping == grouping)].set_index('group').reindex(order).queries
            ax.set_xticks(np.arange(len(order)))
            ax.set_xticklabels([f'{g}\nn={int(k)}' for g, k in zip(order, n)], fontsize=6.5)
            ax.set_yscale('log')
            ax.set_xlabel(xlabel)
            if row == 0:
                ax.set_title(SCALE_LABEL[s])
        axes[row][0].set_ylabel('95th pct q-error (log)')
    method_legend(fig, MODELS + ['exact_marginal_independence', 'uniform_sample'],
                  loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.04))
    fig.text(0.5, -0.07, 'Main workload. Learned models: median of 3 seeds, band = min–max. n = queries in the group; '
             'groups with few queries have unstable tail estimates.', ha='center', fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, 'fig05_error_by_group', agg)


# --------------------------------------------------------------------------
# Fig 6 accuracy versus storage and estimation latency (full scale)
# --------------------------------------------------------------------------
def fig_cost_accuracy():
    a = pd.read_csv(EXP / 'followup_analysis_20261111/accuracy_cost_by_run.csv')
    a['scale'] = a.scale.astype(str)
    a = a[(a.scale == 'full')].copy()
    a['method'] = a.method.replace({'uniform_matched_own_resmade': 'uniform_matched'})
    a = a[a.method.isin(ORDER_ALL)]
    a['latency_ms'] = a.inference_median_ms.fillna(a.planning_median_ms)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, xcol, xlabel in [(axes[0], 'logical_synopsis_bytes', 'Synopsis size in bytes (log)'),
                             (axes[1], 'latency_ms', 'Median estimation time per query, ms (log)')]:
        for m in ORDER_ALL:
            g = a[a.method == m]
            ax.scatter(g[xcol], g.qerror_p95, color=color(m), marker=marker(m), s=46 * (STAR_SCALE**2 if marker(m) == '*' else 1),
                       edgecolor='white', linewidth=0.8, label=label(m), zorder=3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel('95th pct q-error, full table (log)')
    method_legend(fig, ORDER_ALL, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.13))
    fig.text(0.5, -0.22, 'One point per training seed / ANALYZE repeat / sample seed. Size: FP32 parameters (models), '
             'int32 sample rows, PG catalog rows (pg_column_size); shared dictionaries excluded.\nTime: GPU single-query '
             'inference (models), CPU (baselines), PG planner time (no execution) — different devices, not a speed-up claim.',
             ha='center', fontsize=6.8, color=INK2)
    save(fig, 'fig06_accuracy_vs_cost', a[['method', 'repeat', 'qerror_p95', 'logical_synopsis_bytes', 'latency_ms']])


# --------------------------------------------------------------------------
# Fig 7 robustness across workloads (full scale, p95)
# --------------------------------------------------------------------------
def workload_table():
    rows = []
    mt = main_accuracy_table()
    for _, r in mt[mt['quantile'] == 'p95'].iterrows():
        rows.append({'scale': r.scale, 'workload': 'main_native', 'method': r.method,
                     'p95': r.value, 'lo': r.lo, 'hi': r.hi})
    mc = pd.read_csv(EXP / 'supplement_analysis_20261110/metrics_configuration_ranges.csv')
    mc['scale'] = mc.scale.astype(str)
    keep = mc[mc.experiment.isin(['workload_eval_supplement_v2_20261110', 'workload_baselines_20261110',
                                  'stress_v2_baselines_20261110']) & (mc.scope != 'stress')]
    for _, r in keep.iterrows():
        m = ALIASES.get(r.model, r.model)
        if m not in METHODS:
            continue
        rows.append({'scale': r.scale, 'workload': r.scope, 'method': m,
                     'p95': r.qerror_p95_run_median, 'lo': r.qerror_p95_run_min, 'hi': r.qerror_p95_run_max})
    pg = pd.read_csv(EXP / 'supplement_analysis_20261110/pg_metrics_configuration_ranges.csv')
    pg['scale'] = pg.scale.astype(str)
    for _, r in pg[pg.scope != 'main_native'].iterrows():
        rows.append({'scale': r.scale, 'workload': r.scope, 'method': r.method,
                     'p95': r.qerror_p95_repeat_median, 'lo': r.qerror_p95_repeat_min, 'hi': r.qerror_p95_repeat_max})
    return pd.DataFrame(rows).drop_duplicates(['scale', 'workload', 'method'])


def fig_workloads():
    t = workload_table()
    wls = [('main_native', 'Main (2,000)'), ('scale_shared', 'Shared (1,000)'),
           ('predicate_sweep', 'Predicate sweep (1,100)'), ('stress_v2', 'Stress v2 (300)')]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    n = len(ORDER_ALL)
    for ax, s in zip(axes, SCALES):
        for j, m in enumerate(ORDER_ALL):
            for i, (w, _) in enumerate(wls):
                r = t[(t.scale == s) & (t.workload == w) & (t.method == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                x = i + (j - (n - 1) / 2) * 0.085
                ax.errorbar(x, r.p95, yerr=[[r.p95 - r.lo], [r.hi - r.p95]], fmt='none', ecolor=color(m), elinewidth=1)
                ax.plot(x, r.p95, marker=marker(m), color=color(m), markersize=5 * (STAR_SCALE if marker(m) == '*' else 1), markeredgecolor='white',
                        markeredgewidth=0.6, linestyle='none')
        ax.set_xticks(range(len(wls)))
        ax.set_xticklabels([w[1] for w in wls], rotation=20, ha='right', fontsize=7)
        ax.set_yscale('log')
        ax.set_title(SCALE_LABEL[s])
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('95th pct q-error (log)')
    method_legend(fig, ORDER_ALL, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.13))
    fig.text(0.5, -0.19, 'q-error with max(1,·) flooring over all queries, including zero-result stress queries. '
             'Shared and sweep use fixed raw-value predicates; stress v2 has 64 distinct semantics (repeats are not independent).',
             ha='center', fontsize=7, color=INK2)
    save(fig, 'fig07_workload_robustness', t)


# --------------------------------------------------------------------------
# Fig 8 sampling budget sweep (own ResMADE)
# --------------------------------------------------------------------------
def fig_budget():
    mc = pd.read_csv(EXP / 'supplement_analysis_20261110/metrics_configuration_ranges.csv')
    mc['scale'] = mc.scale.astype(str)
    b = mc[(mc.experiment == 'gpu_supplement_20261110') & (mc.width == 256)].sort_values('sampling_budget')
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharex=True)
    scol = {'1000000': '#2a78d6', '5000000': '#eb6834', 'full': '#1baf7a'}
    smark = {'1000000': 'o', '5000000': 's', 'full': 'D'}
    for ax, k, title in [(axes[0], 'p95', '95th pct q-error'), (axes[1], 'p99', '99th pct q-error'),
                         (axes[2], None, 'Median latency per query (ms)')]:
        for s in SCALES:
            g = b[b.scale == s]
            if k:
                y, lo, hi = g[f'qerror_{k}_run_median'], g[f'qerror_{k}_run_min'], g[f'qerror_{k}_run_max']
            else:
                y, lo, hi = g.latency_median_ms_run_median, g.latency_median_ms_run_min, g.latency_median_ms_run_max
            ax.fill_between(g.sampling_budget, lo, hi, color=scol[s], alpha=0.18, linewidth=0)
            ax.plot(g.sampling_budget, y, color=scol[s], marker=smark[s], markersize=4.5, label=SCALE_SHORT[s])
        ax.set_xscale('log')
        ax.set_xticks([100, 500, 2000, 8000])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(title)
        ax.set_xlabel('Progressive samples per query')
    axes[0].legend(frameon=False, title='Table')
    fig.text(0.5, -0.1, 'Own ResMADE, main workload. Line = median of 9 runs (3 training seeds × 3 MC seeds), '
             'band = min–max. Main tables use 2,000 samples.', ha='center', fontsize=7, color=INK2)
    save(fig, 'fig08_sampling_budget', b[['scale', 'sampling_budget', 'runs', 'qerror_p95_run_median', 'qerror_p95_run_min',
                                          'qerror_p95_run_max', 'qerror_p99_run_median', 'qerror_p99_run_min',
                                          'qerror_p99_run_max', 'latency_median_ms_run_median']])


# --------------------------------------------------------------------------
# Fig 9 model width (1M, own ResMADE)
# --------------------------------------------------------------------------
def fig_width():
    p = pd.read_csv(EXP / 'followup_analysis_20261111/nll_query_points.csv')
    p['scale'] = p.scale.astype(str)
    p = p[(p.scale == '1000000') & (p.model == 'own_resmade')]
    a = pd.read_csv(EXP / 'followup_analysis_20261111/accuracy_cost_by_run.csv')
    a['scale'] = a.scale.astype(str)
    a = a[(a.scale == '1000000') & (a.method == 'own_resmade')][['width', 'repeat', 'logical_synopsis_bytes', 'build_seconds']]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8))
    for ax, col, title in [(axes[0], 'qerror_p95', '95th pct q-error'), (axes[1], 'qerror_p99', '99th pct q-error'),
                           (axes[2], 'final_unmasked_nll', 'Final NLL (nats/row)')]:
        for w, g in p.groupby('width'):
            ax.scatter([w] * len(g), g[col], color=color('own_resmade'), marker='s', s=26, edgecolor='white', zorder=3)
        med = p.groupby('width')[col].median()
        ax.plot(med.index, med.values, color=color('own_resmade'), linewidth=1.5)
        ax.set_xscale('log', base=2)
        ax.set_xticks([128, 256, 512])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(title)
        ax.set_xlabel('Hidden width')
    mb = a.groupby('width').logical_synopsis_bytes.median() / 1e6
    tm = a.groupby('width').build_seconds.median()
    axes[3].bar([str(int(w)) for w in mb.index], mb.values, color=color('own_resmade'), width=0.55)
    for i, (v, t) in enumerate(zip(mb.values, tm.values)):
        axes[3].text(i, v, f'{v:.1f} MB\n{t:.0f} s train', ha='center', va='bottom', fontsize=6.5, color=INK2)
    axes[3].set_title('FP32 parameters (MB)')
    axes[3].set_xlabel('Hidden width')
    axes[3].set_ylim(0, mb.max() * 1.35)
    axes[3].grid(axis='x', visible=False)
    fig.text(0.5, -0.1, '1M table, own ResMADE, 3 training seeds per width (points); line = median. '
             'Lower NLL does not guarantee a lower tail q-error.', ha='center', fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, 'fig09_model_width', p)


# --------------------------------------------------------------------------
# Fig 10 joint-zero / near-positive workload
# --------------------------------------------------------------------------
def fig_joint_zero():
    j = pd.read_csv(EXP / 'joint_zero_model_eval_20261111/analysis/all_methods_configuration_ranges.csv')
    j['scale'] = j.scale.astype(str)
    j['method'] = j.method.replace(ALIASES)
    order = ['own_made', 'own_resmade', 'official_resmade', 'uniform_matched', 'PG-extstats',
             'PG-target1000', 'PG-default', 'exact_marginal_independence', 'uniform_sample']
    j = j[j.method.isin(order)]
    panels = [('positive_qerror_p95', 'Near-positive queries:\n95th pct q-error (log)', True),
              ('zero_estimate_p95', 'Joint-zero queries:\n95th pct estimated rows', False),
              ('positive_estimate_eq0', 'Near-positive queries\nestimated as 0 rows (%)', False),
              ('pair_strict_separation_fraction', 'Pairs ranked correctly\n(positive > zero), fraction', False)]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), sharey=True)
    smark = {'1000000': 'o', '5000000': 's', 'full': 'D'}
    y = {m: i for i, m in enumerate(order[::-1])}
    for ax, (col, title, logx) in zip(axes, panels):
        for m in order:
            for k, s in enumerate(SCALES):
                r = j[(j.method == m) & (j.scale == s)]
                if r.empty:
                    continue
                r = r.iloc[0]
                yy = y[m] + (k - 1) * 0.22
                ax.plot([r[f'{col}_run_min'], r[f'{col}_run_max']], [yy, yy], color=color(m), linewidth=1.2)
                ax.plot(r[f'{col}_run_median'], yy, marker=smark[s], color=color(m), markersize=4.5,
                        markeredgecolor='white', markeredgewidth=0.5, linestyle='none')
        if logx:
            ax.set_xscale('log')
            ax.set_xticks([2, 3, 5, 10])
            ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(title, fontsize=8.5)
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(list(y.values()))
    axes[0].set_yticklabels([label(m) for m in y], fontsize=7)
    handles = [Line2D([], [], color=INK2, marker=smark[s], linestyle='none', label=SCALE_SHORT[s]) for s in SCALES]
    fig.legend(handles=handles, frameon=False, loc='upper center', ncol=3, bbox_to_anchor=(0.6, 1.06), title=None)
    fig.text(0.5, -0.07, '100 pairs per scale: a rare positive query and the same query with one predicate changed so the '
             'joint result is 0 (every single predicate still matches rows). Models: 3 training × 3 MC seeds; PG: 3 ANALYZE '
             'repeats. Targeted workload, not a random SQL distribution.', ha='center', fontsize=6.8, color=INK2)
    save(fig, 'fig10_joint_zero', j)


# --------------------------------------------------------------------------
# Fig 11 scale trend on the same 958 always-positive queries
# --------------------------------------------------------------------------
def fig_scale_trend():
    s = pd.read_csv(EXP / 'supplement_analysis_20261110/scale_shared_common_positive.csv')
    s['scale'] = s.scale.astype(str)
    s['method'] = s.method.replace(ALIASES)
    s = s[s.method.isin(ORDER_ALL)]
    agg = s.groupby(['scale', 'method']).agg(p95=('qerror_p95', 'median'), lo=('qerror_p95', 'min'),
                                             hi=('qerror_p95', 'max'), p99=('qerror_p99', 'median'),
                                             runs=('qerror_p95', 'size')).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
    x = np.arange(3)
    for ax, col, title in [(axes[0], 'p95', '95th pct q-error'), (axes[1], 'p99', '99th pct q-error')]:
        for m in ORDER_ALL:
            g = agg[agg.method == m].set_index('scale').reindex(SCALES)
            if col == 'p95':
                ax.fill_between(x, g.lo, g.hi, color=color(m), alpha=0.15, linewidth=0)
            ax.plot(x, g[col], color=color(m), marker=marker(m), markersize=5, label=label(m))
        ax.set_xticks(x)
        ax.set_xticklabels([SCALE_SHORT[k] for k in SCALES])
        ax.set_yscale('log')
        ax.set_title(title)
        ax.set_xlabel('Table size (nested subsets)')
    method_legend(fig, ORDER_ALL, loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig.text(0.5, -0.07, 'The same 958 raw-value queries that are non-empty at every scale. '
             'Median across repeats; band (p95) = min–max.', ha='center', fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, 'fig11_scale_trend_common_queries', agg)


# --------------------------------------------------------------------------
# Appendix figures
# --------------------------------------------------------------------------
def figA1_training_cost():
    t = pd.read_csv(EXP / 'main_matrix_20261109/training_summary.csv')
    t['scale'] = t.scale.astype(str)
    agg = t.groupby(['scale', 'model']).agg(minutes=('elapsed_seconds', lambda v: np.median(v) / 60),
                                            gpu_mb=('peak_gpu_bytes', lambda v: np.median(v) / 2**20),
                                            ckpt_mb=('checkpoint_bytes', lambda v: np.median(v) / 1e6),
                                            nll=('final_nll', 'median')).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    w = 0.26
    for ax, col, title in [(axes[0], 'minutes', 'Training time, 20 epochs (min)'),
                           (axes[1], 'gpu_mb', 'Peak GPU memory (MiB)'),
                           (axes[2], 'ckpt_mb', 'Checkpoint file (MB)')]:
        for i, m in enumerate(MODELS):
            g = agg[agg.model == m].set_index('scale').reindex(SCALES)
            ax.bar(np.arange(3) + (i - 1) * (w + 0.02), g[col], width=w, color=color(m), label=label(m))
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels([SCALE_SHORT[s] for s in SCALES])
        ax.set_title(title)
        ax.grid(axis='x', visible=False)
    method_legend(fig, MODELS, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.text(0.5, -0.15, 'Median of 3 seeds; one RTX 3090, float32, batch 2048.', ha='center', fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, 'figA1_training_cost', agg)


def figA2_tail_groups():
    t = pd.read_csv(EXP / 'followup_analysis_20261111/tail_groups.csv')
    t['scale'] = t.scale.astype(str)
    t = t[t.model.isin(MODELS)]
    groupings = [('rare_equality_group', 'Rarest equality constant\n(full-table frequency)'),
                 ('positive_truth_le10', 'True count ≤ 10 rows'),
                 ('has_date_equality', 'Has date equality predicate'),
                 ('has_indicator_Y', 'Has a "Y" status indicator')]
    agg = t.groupby(['scale', 'model', 'grouping', 'group']).agg(
        p95=('qerror_p95', 'median'), lo=('qerror_p95', 'min'), hi=('qerror_p95', 'max'),
        n=('query_records', 'median')).reset_index()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    for ax, (gname, title) in zip(axes, groupings):
        a = agg[(agg.grouping == gname) & (agg.scale == 'full')]
        pref = ['<=1e-5', '<=1e-4', '<=1e-3', '>1e-3', 'False', 'True']
        groups = sorted(a.group.astype(str).unique(), key=lambda g: pref.index(g) if g in pref else 99)
        a = a.assign(group=a.group.astype(str))
        for i, m in enumerate(MODELS):
            g = a[a.model == m].set_index('group').reindex(groups)
            x = np.arange(len(groups)) + (i - 1) * 0.22
            ax.errorbar(x, g.p95, yerr=[g.p95 - g.lo, g.hi - g.p95], fmt='none', ecolor=color(m), elinewidth=1)
            ax.plot(x, g.p95, marker=marker(m), color=color(m), linestyle='none', markersize=5,
                    markeredgecolor='white', markeredgewidth=0.6)
        n = a[a.model == 'own_made'].set_index('group').reindex(groups).n
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels([f'{g}\nn={int(k)}' if k == k else str(g) for g, k in zip(groups, n)], fontsize=6.3)
        ax.set_title(title, fontsize=8.5)
        ax.set_yscale('log')
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('95th pct q-error (log)')
    method_legend(fig, MODELS, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.12))
    fig.text(0.5, -0.17, 'Full table, main workload; median of 3 seeds (bars = min–max). Thresholds fixed before analysis; '
             'small groups are descriptive only.', ha='center', fontsize=7, color=INK2)
    save(fig, 'figA2_tail_groups', agg)


def figA3_mc_vs_training():
    t = pd.read_csv(EXP / 'followup_analysis_20261111/training_vs_mc_spread.csv')
    t['scale'] = t.scale.astype(str)
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9), sharey=True)
    for ax, s in zip(axes, SCALES):
        g = t[t.scale == s].sort_values('sampling_budget')
        ax.plot(g.sampling_budget, g.p95_training_mean_sd, color='#2a78d6', marker='o', markersize=4,
                label='Between training seeds')
        ax.plot(g.sampling_budget, g.mean_within_training_p95_mc_sd, color='#eb6834', marker='s', markersize=4,
                label='Within a checkpoint (MC)')
        ax.set_xscale('log')
        ax.set_xticks([100, 500, 2000, 8000])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(SCALE_LABEL[s])
        ax.set_xlabel('Progressive samples per query')
    axes[0].set_ylabel('SD of 95th pct q-error')
    axes[0].legend(frameon=False)
    fig.text(0.5, -0.1, 'Own ResMADE, 3 training seeds × 3 MC seeds. Descriptive SDs from a 3×3 grid, not a formal '
             'variance decomposition.', ha='center', fontsize=7, color=INK2)
    save(fig, 'figA3_mc_vs_training_spread', t)


def figA4_latency():
    sr = pd.read_csv(EXP / 'main_evaluation_20261110/metrics_seed_ranges.csv')
    sr['scale'] = sr.scale.astype(str)
    pg = pd.read_csv(EXP / 'supplement_analysis_20261110/pg_metrics_configuration_ranges.csv')
    pg['scale'] = pg.scale.astype(str)
    rows = [{'scale': r.scale, 'method': r.model, 'ms': r.latency_median_ms_seed_median,
             'kind': 'GPU inference' if r.model in MODELS else 'CPU'} for _, r in sr.iterrows()]
    rows += [{'scale': r.scale, 'method': r.method, 'ms': r.mean_planning_ms_repeat_median, 'kind': 'PG planning (mean)'}
             for _, r in pg[pg.scope == 'main_native'].iterrows()]
    d = pd.DataFrame(rows)
    order = MODELS + ['exact_marginal_independence', 'uniform_sample', 'PG-default', 'PG-target1000', 'PG-extstats']
    fig, ax = plt.subplots(figsize=(8, 3.2))
    w = 0.26
    for k, s in enumerate(SCALES):
        g = d[d.scale == s].set_index('method').reindex(order)
        ax.bar(np.arange(len(order)) + (k - 1) * (w + 0.02), g.ms, width=w,
               color=[color(m) for m in order], alpha=[0.45, 0.7, 1.0][k], label=SCALE_SHORT[s])
    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([label(m) for m in order], rotation=35, ha='right', fontsize=7)
    ax.set_ylabel('Time per query, ms (log)')
    ax.grid(axis='x', visible=False)
    ax.legend(frameon=False, title='Table (light → dark)')
    fig.text(0.5, -0.26, 'Models: median single-query GPU latency incl. host-to-GPU mask transfer, shared unoptimised sampler '
             '(not the official optimised runtime).\nCPU baselines exclude building statistics. PG: planner time only. '
             'Different devices and boundaries: compare within a group, not across.', ha='center', fontsize=6.8, color=INK2)
    save(fig, 'figA4_latency', d)


def figA5_nll_vs_tail():
    p = pd.read_csv(EXP / 'followup_analysis_20261111/nll_query_points.csv')
    p['scale'] = p.scale.astype(str)
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    for ax, s in zip(axes, SCALES):
        g = p[(p.scale == s) & (p.width == 256)]
        for m in MODELS:
            h = g[g.model == m]
            ax.scatter(h.final_unmasked_nll, h.qerror_p95, color=color(m), marker=marker(m), s=34,
                       edgecolor='white', label=label(m))
        ax.set_title(SCALE_LABEL[s])
        ax.set_xlabel('Final unmasked NLL (nats/row)')
        ax.ticklabel_format(axis='x', useOffset=False)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    axes[0].set_ylabel('95th pct q-error')
    method_legend(fig, MODELS, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.12))
    fig.text(0.5, -0.17, 'One point per training seed, width 256. NLL is measured on the training table (no held-out split).',
             ha='center', fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, 'figA5_nll_vs_tail', p)


def figA6_pg_cost():
    pg = pd.read_csv(EXP / 'supplement_analysis_20261110/pg_metrics_configuration_ranges.csv')
    pg['scale'] = pg.scale.astype(str)
    g = pg[pg.scope == 'main_native']
    order = ['PG-default', 'PG-target1000', 'PG-extstats']
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    w = 0.26
    for ax, col, title in [(axes[0], 'analyze_seconds_repeat_median', 'ANALYZE time (s)'),
                           (axes[1], 'single_column_catalog_row_bytes_repeat_median', 'Single-column stats (bytes)'),
                           (axes[2], 'extended_catalog_row_bytes_repeat_median', 'Extended stats (bytes)')]:
        for i, m in enumerate(order):
            h = g[g.method == m].set_index('scale').reindex(SCALES)
            ax.bar(np.arange(3) + (i - 1) * (w + 0.02), h[col].fillna(0), width=w, color=color(m), label=label(m))
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels([SCALE_SHORT[s] for s in SCALES])
        ax.set_title(title)
        ax.grid(axis='x', visible=False)
    method_legend(fig, order, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.text(0.5, -0.15, 'PostgreSQL 17.6; median of 3 ANALYZE repeats. Bytes = pg_column_size of catalog rows, '
             'not physical pages.', ha='center', fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, 'figA6_pg_cost', g)


# --------------------------------------------------------------------------
# Improvement experiments (filled when CODEX_HANDOFF results exist)
# --------------------------------------------------------------------------
VARIANTS = [('baseline', 'Baseline ResMADE', '#eb6834', 's'), ('order', '+ MI column order', '#2a78d6', 'o'),
            ('steps', '+ equal 122,480 steps', '#1baf7a', 'D'), ('embed', '+ embedding inputs', '#eda100', 'v'),
            ('all', 'All three', '#e87ba4', '^')]


def fig_improvements():
    found = sorted(EXP.glob('improvement_*/analysis/metrics_seed_ranges.csv'))
    if not found:
        print('skip fig12_improvements: no improvement results yet')
        return
    imp = pd.read_csv(found[-1])
    imp['scale'] = imp.scale.astype(str)
    if 'workload' in imp:
        imp = imp[imp.workload == 'main_native']
    sr = pd.read_csv(EXP / 'main_evaluation_20261110/metrics_seed_ranges.csv')
    sr['scale'] = sr.scale.astype(str)
    base = sr[sr.model == 'own_resmade'].assign(variant='baseline')
    imp['variant'] = imp.variant.str.replace('V-', '', regex=False).str.lower()
    d = pd.concat([base, imp], ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=False)
    for ax, k, title in [(axes[0], 'median', 'Median q-error'), (axes[1], 'p95', '95th pct q-error'),
                         (axes[2], 'p99', '99th pct q-error')]:
        for i, (v, name, c, mk) in enumerate(VARIANTS):
            g = d[d.variant == v].set_index('scale').reindex(SCALES)
            x = np.arange(3) + (i - 2) * 0.14
            y = g[f'qerror_{k}_seed_median']
            ax.errorbar(x, y, yerr=[y - g[f'qerror_{k}_seed_min'], g[f'qerror_{k}_seed_max'] - y],
                        fmt='none', ecolor=c, elinewidth=1)
            ax.plot(x, y, marker=mk, color=c, linestyle='none', markersize=5.5, label=name,
                    markeredgecolor='white', markeredgewidth=0.6)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels([SCALE_SHORT[s] for s in SCALES])
        ax.set_title(title)
        ax.grid(axis='x', visible=False)
    axes[0].legend(frameon=False, fontsize=7)
    fig.text(0.5, -0.08, 'Own ResMADE variants on the main workload; median of 3 training seeds (bars = min–max).',
             ha='center', fontsize=7, color=INK2)
    save(fig, 'fig12_improvements', d)


if __name__ == '__main__':
    for f in [fig_training_curves, fig_main_accuracy, fig_survival, fig_paired_resmade, fig_grouped,
              fig_cost_accuracy, fig_workloads, fig_budget, fig_width, fig_joint_zero, fig_scale_trend,
              figA1_training_cost, figA2_tail_groups, figA3_mc_vs_training, figA4_latency, figA5_nll_vs_tail,
              figA6_pg_cost, fig_improvements]:
        f()
