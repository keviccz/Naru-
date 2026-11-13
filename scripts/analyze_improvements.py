"""Plan-driven, raw-CSV audit and plot-ready tables for fixed ResMADE ablations.

Quantiles are calculated inside each run before seed aggregation. MC ranges are
descriptive, never confidence intervals. No model or GPU import is needed.
"""
import argparse
from collections import Counter, defaultdict
import csv
import datetime as dt
import json
import math
from pathlib import Path
import statistics
import traceback


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        if keys:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    tmp.replace(path)


def quantile(values, p):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * p
    lo = math.floor(index)
    hi = math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def qerror(truth, estimate):
    return max(max(1.0, truth) / max(1.0, estimate),
               max(1.0, estimate) / max(1.0, truth))


def metrics(rows):
    positive = [r for r in rows if r['truth'] > 0]
    zero = [r for r in rows if r['truth'] == 0]
    result = {'queries': len(rows), 'positive_truth_queries': len(positive),
              'zero_truth_queries': len(zero), 'qerror_definition': 'floor1; positive-only metrics reported separately'}
    for label, subset in (('qerror', rows), ('qerror_positive', positive)):
        errors = [r['qerror'] for r in subset]
        for suffix, p in (('median', .5), ('p95', .95), ('p99', .99), ('max', 1)):
            result[label + '_' + suffix] = quantile(errors, p)
    result['latency_median_ms'] = quantile([r['latency_ms'] for r in rows], .5)
    result['latency_p95_ms'] = quantile([r['latency_ms'] for r in rows], .95)
    for threshold in (1, 10, 100):
        result['zero_estimate_gt' + str(threshold)] = sum(r['estimate'] > threshold for r in zero)
    result['positive_estimate_eq0'] = sum(r['estimate'] == 0 for r in positive)
    result['positive_estimate_lt1'] = sum(r['estimate'] < 1 for r in positive)
    result['zero_estimate_median'] = quantile([r['estimate'] for r in zero], .5)
    result['zero_estimate_p95'] = quantile([r['estimate'] for r in zero], .95)
    result['zero_estimate_max'] = quantile([r['estimate'] for r in zero], 1)
    return result


def path_mapper(plan, project):
    remote = plan['remote_project_root'].rstrip('/')
    def mapped(value):
        value = str(value)
        if value == remote:
            return project
        if value.startswith(remote + '/'):
            return project.joinpath(*value[len(remote) + 1:].split('/'))
        path = Path(value)
        if not path.is_absolute():
            raise ValueError('Path is outside the frozen project: ' + value)
        return path
    return mapped


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate_plan(plan):
    expected = {(v, s, seed) for v in ('order', 'steps', 'embed', 'all')
                for s in ('1000000', '5000000', 'full') for seed in range(3)}
    trains = plan['train_cells']
    require(len(trains) == 36 and {(t['variant'], t['scale'], t['seed']) for t in trains} == expected,
            'Training plan must contain the exact 36 fixed cells')
    require(len({t['id'] for t in trains}) == 36, 'Duplicate train ID')
    require({t['id'] for t in trains if t['reused_from_baseline']} ==
            {'steps_full_seed' + str(i) for i in range(3)}, 'Only steps/full may reuse B')
    cells = plan['eval_cells']
    require(len(cells) == 720 and len({c['id'] for c in cells}) == 720,
            'Evaluation plan must contain 720 unique logical cells')
    by_id = {c['id']: c for c in cells}
    expected_grid = {(w, 2000, 0) for w in
                     ('main_native', 'dev', 'scale_shared', 'predicate_sweep', 'stress_v2')}
    expected_grid |= {('joint_zero_nearzero', 2000, m) for m in range(3)}
    expected_grid |= {('budget', s, m) for s in (100, 500, 2000, 8000) for m in range(3)}
    for train in trains:
        group = [c for c in cells if c['train_id'] == train['id']]
        require(len(group) == 20 and {(c['workload'], c['samples'], c['mc_seed']) for c in group} == expected_grid,
                'Evaluation grid mismatch: ' + train['id'])
        for c in group:
            require((c['variant'], c['scale'], c['seed']) ==
                    (train['variant'], train['scale'], train['seed']), 'Evaluation/train identity mismatch')
            require(c['reused_from_baseline'] == train['reused_from_baseline'], 'Reuse identity mismatch')
            counts = {'main_native': 2000, 'dev': 200, 'scale_shared': 1000,
                      'predicate_sweep': 1100, 'stress_v2': 300,
                      'joint_zero_nearzero': 200, 'budget': 2000}
            require(c['expected_queries'] == counts[c['workload']], 'Fixed workload size mismatch')
            require(c['scope'] == ('main_native' if c['workload'] == 'budget' else c['workload']),
                    'Scope differs from frozen workload')
            expected_alias = c['workload'] == 'budget' and c['samples'] == 2000 and c['mc_seed'] == 0
            require(bool(c['alias_of']) == expected_alias, 'Only budget2000/MC0 must alias main')
            if expected_alias:
                source = by_id[c['alias_of']]
                require(source['train_id'] == train['id'] and source['workload'] == 'main_native',
                        'Invalid main/budget alias')
    require(len(plan['baseline_eval_cells']) == 8 and
            {(c['scale'], c['seed']) for c in plan['baseline_eval_cells']} ==
            {(s, i) for s in ('1000000', '5000000', 'full') for i in range(3)} - {('1000000', 0)},
            'Expected exactly the eight missing B dev cells')


def query_signature(q):
    return tuple(sorted((p['column_index'], tuple(sorted(p['allowed_codes']))) for p in q['predicates']))


def load_queries(path, scope, expected):
    queries = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    require(len(queries) == expected and len({q['query_id'] for q in queries}) == expected,
            'Frozen query count/ID mismatch: ' + str(path))
    require(all(q['workload'] == scope and q['truth'] >= 0 for q in queries), 'Query scope/truth mismatch')
    return {q['query_id']: q for q in queries}


def load_rows(directory, method, queries, check_summary=True):
    directory = Path(directory)
    require(read_json(directory / 'status.json')['status'] == 'COMPLETE', 'Incomplete evaluation: ' + str(directory))
    with (directory / 'per_query.csv').open(encoding='utf-8', newline='') as f:
        raw = [r for r in csv.DictReader(f) if r.get('method', method) == method]
    require(len(raw) == len(queries) and {r['query_id'] for r in raw} == set(queries),
            'Evaluation query set/method mismatch: ' + str(directory))
    rows = []
    for r in raw:
        q = queries[r['query_id']]
        result = {'query_id': r['query_id'], 'truth': float(r['truth']),
                  'estimate': float(r['estimate']), 'qerror': float(r['qerror']),
                  'latency_ms': float(r['latency_ms'])}
        require(all(math.isfinite(result[k]) and result[k] >= 0 for k in ('truth', 'estimate', 'qerror', 'latency_ms')),
                'Nonfinite/negative result: ' + str(directory))
        require(result['truth'] == q['truth'], 'Truth differs from frozen query')
        require(math.isclose(result['qerror'], qerror(result['truth'], result['estimate']), rel_tol=1e-8, abs_tol=1e-8),
                'Stored qerror differs from raw truth/estimate')
        result['qerror'] = qerror(result['truth'], result['estimate'])
        if r.get('workload'):
            require(r['workload'] == q['workload'], 'Stored workload mismatch')
        rows.append(result)
    if check_summary:
        summary = read_json(directory / 'summary.json')
        require(summary.get('status', 'COMPLETE') == 'COMPLETE', 'Summary not COMPLETE')
        stated = summary.get('methods', {}).get(method, summary)
        recomputed = metrics(rows)
        for key in ('qerror_median', 'qerror_p95', 'qerror_p99', 'qerror_max', 'latency_median_ms'):
            if key in stated:
                require(math.isclose(stated[key], recomputed[key], rel_tol=1e-8, abs_tol=1e-8),
                        'Summary/raw CSV mismatch for ' + key + ': ' + str(directory))
    return rows


def paired(rows, baseline, queries):
    base = {r['query_id']: r for r in baseline}
    groups = defaultdict(list)
    for row in rows:
        groups[query_signature(queries[row['query_id']])].append(row)
    ratios = []
    for group in groups.values():
        current = statistics.median(r['qerror'] for r in group)
        old = statistics.median(base[r['query_id']]['qerror'] for r in group)
        ratios.append(current / old)
    improved = sum(x < 1 - 1e-10 for x in ratios)
    tied = sum(abs(x - 1) <= 1e-10 for x in ratios)
    current = metrics(rows)
    old = metrics(baseline)
    return {'query_records': len(rows), 'unique_semantic_queries': len(groups),
            'qerror_definition': 'floor1 on all records; zero and positive diagnostics separate',
            'semantic_qerror_ratio_median': quantile(ratios, .5),
            'semantic_improved_fraction': improved / len(ratios),
            'semantic_tie_fraction': tied / len(ratios),
            'semantic_worse_fraction': (len(ratios) - improved - tied) / len(ratios),
            'baseline_qerror_p95': old['qerror_p95'],
            'qerror_p95_difference': current['qerror_p95'] - old['qerror_p95'],
            'qerror_p95_ratio': current['qerror_p95'] / old['qerror_p95'],
            'qerror_p99_difference': current['qerror_p99'] - old['qerror_p99'],
            'latency_median_ratio': (current['latency_median_ms'] / old['latency_median_ms'])
                                    if old['latency_median_ms'] else None}


def require_identical_rows(rows, reference, label):
    source = {r['query_id']: r for r in reference}
    require({r['query_id'] for r in rows} == set(source), 'Reuse query set mismatch: ' + label)
    for row in rows:
        require(all(row[k] == source[row['query_id']][k] for k in
                    ('truth', 'estimate', 'qerror', 'latency_ms')), 'Reuse changed inference values: ' + label)


def joint_metrics(rows, queries):
    pairs = defaultdict(dict)
    for row in rows:
        q = queries[row['query_id']]
        require(q['group'] in ('zero', 'near_positive'), 'Invalid joint group')
        require(q['truth'] == 0 if q['group'] == 'zero' else 1 <= q['truth'] <= 10, 'Invalid joint truth window')
        require(q['group'] not in pairs[q['pair_id']], 'Duplicate joint member')
        pairs[q['pair_id']][q['group']] = row
    require(len(pairs) == 100 and all(set(p) == {'zero', 'near_positive'} for p in pairs.values()),
            'Joint pairing must cover exactly 100 zero/positive pairs')
    details = [{'pair_id': pid, 'zero_estimate': p['zero']['estimate'],
                'positive_estimate': p['near_positive']['estimate'],
                'positive_truth': p['near_positive']['truth'],
                'strict_separation': int(p['near_positive']['estimate'] > p['zero']['estimate']),
                'tie': int(p['near_positive']['estimate'] == p['zero']['estimate'])}
               for pid, p in pairs.items()]
    strict = sum(r['strict_separation'] for r in details)
    ties = sum(r['tie'] for r in details)
    return {'pairs': 100, 'pair_strict_separation_fraction': strict / 100,
            'pair_tie_fraction': ties / 100, 'pair_half_tie_score': (strict + .5 * ties) / 100}, details


def aggregate(rows, group_keys, seed_key='training_seed'):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_keys)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        require(len({r[seed_key] for r in group}) == len(group), 'Repeated seed inside an aggregation group')
        result = dict(zip(group_keys, key))
        result.update(seed_count=len(group), seed_values=','.join(str(r[seed_key]) for r in group),
                      aggregation='per-run metric, then seed median/min/max; descriptive range, not CI')
        numeric = [k for k, value in group[0].items() if isinstance(value, (float, int)) and
                   not isinstance(value, bool) and k not in group_keys and k != seed_key]
        for metric in numeric:
            values = [r[metric] for r in group if r.get(metric) is not None]
            if values:
                result.update({metric + '_seed_median': statistics.median(values),
                               metric + '_seed_min': min(values), metric + '_seed_max': max(values)})
        output.append(result)
    return output


def run(args):
    root = Path(args.experiment_root).resolve()
    project = root.parent.parent
    out = root / 'analysis'
    out.mkdir(exist_ok=True)
    audit = {'status': 'RUNNING', 'started_at': dt.datetime.now(dt.timezone.utc).isoformat(),
             'partial_requested': args.partial, 'pending_train_ids': [], 'pending_eval_ids': [],
             'pending_baseline_eval_ids': [], 'matched_status': 'PENDING'}
    write_json(out / 'audit.json', audit)
    try:
        plan = read_json(root / 'plan.json')
        validate_plan(plan)
        mapped = path_mapper(plan, project)
        training, curves, metric_rows, pairs, worst, joint_pairs = [], [], [], [], [], []
        seen_csv, actual_inference, actual_estimates = set(), set(), 0
        actual_baseline_dev = 0
        reused_baseline_dev = 0
        eval_by_id = {c['id']: c for c in plan['eval_cells']}
        q_cache, b_cache = {}, {}
        shared_source = read_json(project / 'experiments/supplement_analysis_20261110/scale_shared_common_positive_query_ids.json')
        shared_ids = set(shared_source['query_ids'])
        require(len(shared_ids) == 958 and shared_source['unique_semantic_queries'] == 956, 'Frozen common-positive selection mismatch')
        for cell in plan['train_cells']:
            directory = mapped(cell['run_dir'])
            if not (directory / 'status.json').exists() or read_json(directory / 'status.json')['status'] != 'COMPLETE':
                audit['pending_train_ids'].append(cell['id'])
                continue
            cfg = read_json(directory / 'config.json')
            state = read_json(directory / 'status.json')
            result = read_json(directory / 'training_result.json')
            require((cfg['scale'], cfg['seed'], cfg['model']) == (cell['scale'], cell['seed'], 'own_resmade'), 'Training identity mismatch')
            require(result['training_steps'] == cell['total_steps'], 'Completed optimizer-step budget mismatch')
            require(cell['reused_from_baseline'] or cfg['variant'] == cell['variant'], 'Training variant mismatch')
            baseline_run = mapped(cell['baseline_run_dir'])
            baseline_cfg = read_json(baseline_run / 'config.json')
            baseline_result = read_json(baseline_run / 'training_result.json')
            baseline_state = read_json(baseline_run / 'status.json')
            meta = dict(train_id=cell['id'], variant=cell['variant'], scale=cell['scale'], training_seed=cell['seed'],
                        reused_from_baseline=cell['reused_from_baseline'], source_run_dir=str(directory))
            training.append(dict(meta, final_nll=result['final_full_table_nll_nats'],
                                 training_steps=result['training_steps'], real_epochs=result['training_epochs'],
                                 elapsed_seconds=state['elapsed_seconds'], parameter_count=cfg['parameter_count'],
                                 parameter_bytes=cfg['parameter_bytes'], checkpoint_bytes=result['checkpoint_bytes'],
                                 peak_gpu_bytes=result['peak_gpu_allocated_bytes'],
                                 baseline_final_nll=baseline_result['final_full_table_nll_nats'],
                                 nll_difference=result['final_full_table_nll_nats']-baseline_result['final_full_table_nll_nats'],
                                 parameter_bytes_ratio=cfg['parameter_bytes']/baseline_cfg['parameter_bytes'],
                                 elapsed_seconds_ratio=state['elapsed_seconds']/baseline_state['elapsed_seconds'],
                                 baseline_training_steps=baseline_result['training_steps'],
                                 embedding_columns=json.dumps(cfg.get('embedding_columns', [])),
                                 column_order=json.dumps(cfg.get('column_order', list(range(len(cfg['domain_sizes'])))))))
            records = [json.loads(line) for line in (directory / 'epoch_metrics.jsonl').read_text().splitlines() if line.strip()]
            require(len(records) == 20 and records[-1]['global_step'] == cell['total_steps'], 'Training curve coverage/last step mismatch')
            require(all(records[i]['global_step'] < records[i+1]['global_step'] for i in range(len(records)-1)), 'Nonmonotone training curve')
            for record in records:
                curves.append(dict(meta, reporting_point=record.get('virtual_epoch', record['epoch']),
                                   real_epoch_progress=record.get('real_epoch_progress', record['epoch']),
                                   window_kind=record.get('window_kind', 'real_epoch'), step=record['global_step'],
                                   training_nll=record['mean_training_nll_nats'], lr=record['last_lr'],
                                   rows_in_window=record['rows'], elapsed_seconds=record['elapsed_seconds'],
                                   rows_per_second=record['rows_per_second'],
                                   peak_gpu_bytes=record['peak_gpu_allocated_bytes']))
        for cell in plan['baseline_eval_cells']:
            directory = mapped(cell['output_dir'])
            if not (directory / 'status.json').exists() or read_json(directory / 'status.json')['status'] != 'COMPLETE':
                audit['pending_baseline_eval_ids'].append(cell['id'])
                continue
            qs = load_queries(mapped(cell['queries_path']), 'dev', 200)
            dev_rows = load_rows(directory, 'own_resmade', qs)
            dev_summary = read_json(directory / 'summary.json')
            if dev_summary.get('evaluation_performed', True):
                actual_inference.add(str(directory))
                actual_estimates += 200
                actual_baseline_dev += 1
            else:
                source = dev_summary.get('source_dir') or dev_summary.get('reuse_source_dir')
                require(bool(source), 'Reused B dev requires source_dir provenance')
                require_identical_rows(dev_rows, load_rows(mapped(source), 'own_resmade', qs), cell['id'])
                reused_baseline_dev += 1
        for cell in plan['eval_cells']:
            directory = mapped(cell['output_dir'])
            if not (directory / 'status.json').exists() or read_json(directory / 'status.json')['status'] != 'COMPLETE':
                audit['pending_eval_ids'].append(cell['id'])
                continue
            key = cell['queries_path']
            if key not in q_cache:
                q_cache[key] = load_queries(mapped(key), cell['scope'], cell['expected_queries'])
            queries = q_cache[key]
            method = 'own_resmade_' + cell['variant']
            rows = load_rows(directory, method, queries)
            seen_csv.add(str(directory / 'per_query.csv'))
            summary = read_json(directory / 'summary.json')
            is_performed = not cell['alias_of'] and not cell['reused_from_baseline']
            require(summary.get('evaluation_performed', True) == is_performed, 'Alias/reuse inference provenance mismatch')
            if cell['alias_of']:
                source = eval_by_id[cell['alias_of']]
                reference = load_rows(mapped(source['output_dir']), method, queries)
                require_identical_rows(rows, reference, cell['id'])
            elif cell['reused_from_baseline']:
                reference = load_rows(mapped(cell['reuse_source_dir']), 'own_resmade', queries)
                require_identical_rows(rows, reference, cell['id'])
            if is_performed:
                actual_inference.add(str(directory))
                actual_estimates += len(rows)
            meta = dict(eval_id=cell['id'], train_id=cell['train_id'], variant=cell['variant'], scale=cell['scale'],
                        training_seed=cell['seed'], workload=cell['workload'], sampling_budget=cell['samples'],
                        mc_seed=cell['mc_seed'], alias_of=cell['alias_of'] or '',
                        reused_from_baseline=cell['reused_from_baseline'], evaluation_performed=is_performed,
                        source_csv=str(directory / 'per_query.csv'))
            baseline_cell = eval_by_id[cell['alias_of']] if cell['alias_of'] else cell
            baseline_dir = mapped(baseline_cell['baseline_eval_dir'])
            meta['baseline_canonical_dir'] = str(baseline_dir)
            meta['baseline_canonical_eval_id'] = baseline_cell['id']
            bkey = (str(baseline_dir), key)
            if bkey not in b_cache:
                if not (baseline_dir / 'status.json').exists() or read_json(baseline_dir / 'status.json')['status'] != 'COMPLETE':
                    audit['pending_eval_ids'].append(cell['id'] + ':baseline')
                    continue
                b_cache[bkey] = load_rows(baseline_dir, 'own_resmade', queries)
            baseline = b_cache[bkey]
            subsets = [(meta, rows, baseline, queries)]
            if cell['workload'] == 'scale_shared':
                require(shared_ids <= set(queries), 'Common-positive IDs missing')
                selected_q = {k: v for k, v in queries.items() if k in shared_ids}
                require(all(q['truth'] > 0 for q in selected_q.values()) and
                        len({query_signature(q) for q in selected_q.values()}) == 956, 'Common-positive truth/semantic mismatch')
                subsets.append((dict(meta, workload='scale_shared_common_positive'),
                                [r for r in rows if r['query_id'] in shared_ids],
                                [r for r in baseline if r['query_id'] in shared_ids], selected_q))
            for submeta, subrows, subbase, subqs in subsets:
                result = dict(submeta, **metrics(subrows))
                if cell['workload'] == 'joint_zero_nearzero':
                    joint, details = joint_metrics(subrows, subqs)
                    result.update(joint)
                    joint_pairs.extend(dict(submeta, **r) for r in details)
                metric_rows.append(result)
                pair_result = paired(subrows, subbase, subqs)
                if any(r['truth'] == 0 for r in subrows) and any(r['truth'] > 0 for r in subrows):
                    positive_ids = {r['query_id'] for r in subrows if r['truth'] > 0}
                    positive_comparison = paired([r for r in subrows if r['query_id'] in positive_ids],
                                                 [r for r in subbase if r['query_id'] in positive_ids], subqs)
                    pair_result.update({'positive_only_' + k: v for k, v in positive_comparison.items()})
                pairs.append(dict(submeta, baseline_csv=str(baseline_dir / 'per_query.csv'), **pair_result))
            for r in sorted(rows, key=lambda r: r['qerror'], reverse=True)[:10]:
                worst.append(dict(meta, **r, predicates=json.dumps(queries[r['query_id']]['predicates'], separators=(',', ':'))))
        group_keys = ('variant', 'scale', 'workload', 'sampling_budget', 'mc_seed')
        write_csv(out / 'training_curves.csv', curves)
        write_csv(out / 'training_summary.csv', training)
        write_csv(out / 'training_seed_ranges.csv', aggregate(training, ('variant', 'scale')))
        write_csv(out / 'metrics_by_seed.csv', metric_rows)
        write_csv(out / 'metrics_seed_ranges.csv', aggregate(metric_rows, group_keys))
        write_csv(out / 'metrics_mc_ranges.csv', aggregate(metric_rows,
                  ('variant', 'scale', 'workload', 'sampling_budget', 'training_seed'), 'mc_seed'))
        write_csv(out / 'paired_vs_baseline.csv', pairs)
        write_csv(out / 'paired_seed_ranges.csv', aggregate(pairs, group_keys))
        write_csv(out / 'worst_queries.csv', worst)
        write_csv(out / 'joint_pairs.csv', joint_pairs)
        matched_root = root / 'matched'
        if (matched_root / 'status.json').exists():
            matched_status = read_json(matched_root / 'status.json')
            audit['matched_status'] = matched_status['status']
            if matched_status['status'] == 'COMPLETE':
                manifest = read_json(matched_root / 'manifest.json')
                draws = manifest['draws']
                require(len(draws) == 9 and len({(d['scale'], d['parameter_budget_bytes'], d['sample_seed']) for d in draws}) == 9,
                        'Matched sampling must contain nine unique draws')
                matched_metrics = []
                for draw in draws:
                    main = next(c for c in plan['eval_cells'] if c['scale'] == draw['scale'] and c['workload'] == 'main_native')
                    qs = load_queries(mapped(main['queries_path']), 'main_native', 2000)
                    target = matched_root / draw['id']
                    raw = load_rows(target, 'uniform_matched_improvement', qs)
                    stated = read_json(target / 'summary.json')
                    require(stated['sample_array_bytes'] <= draw['parameter_budget_bytes'] and
                            stated['sample_array_bytes'] == stated['sample_rows'] * stated['int32_row_bytes'], 'Matched storage mismatch')
                    matched_metrics.append(dict(draw, **metrics(raw)))
                write_csv(out / 'matched_metrics_by_seed.csv', matched_metrics)
                write_csv(out / 'matched_metrics_seed_ranges.csv', aggregate(matched_metrics,
                          ('scale', 'parameter_budget_bytes'), 'sample_seed'))
            else:
                write_csv(out / 'matched_metrics_by_seed.csv', [])
                write_csv(out / 'matched_metrics_seed_ranges.csv', [])
        else:
            write_csv(out / 'matched_metrics_by_seed.csv', [])
            write_csv(out / 'matched_metrics_seed_ranges.csv', [])
        audit.update(completed_train_cells=len(training), planned_train_cells=36,
                     planned_new_train_cells=33, reused_train_cells=3,
                     completed_logical_eval_cells=len({r['eval_id'] for r in metric_rows}), planned_logical_eval_cells=720,
                     unique_csv_audited=len(seen_csv), actual_new_inference_cells=len(actual_inference),
                     actual_new_estimates=actual_estimates,
                     actual_new_baseline_dev_cells=actual_baseline_dev,
                     reused_baseline_dev_cells=reused_baseline_dev,
                     expected_new_inference_cells=627+actual_baseline_dev,
                     expected_new_estimates=897600+200*actual_baseline_dev,
                     primary_metric='main_native qerror_p95 per training seed, then median/min/max',
                     semantic_pairing='median duplicate-record qerror inside semantic predicate sets before ratios',
                     common_positive='958 records / 956 semantic queries, filtered from original shared1000 results',
                     uncertainty='Training-seed and MC-seed ranges are descriptive, not confidence intervals')
        pending = any(audit[k] for k in ('pending_train_ids', 'pending_eval_ids', 'pending_baseline_eval_ids')) or audit['matched_status'] != 'COMPLETE'
        if not pending:
            require(len(training) == 36 and len({r['eval_id'] for r in metric_rows}) == 720,
                    'Full completion has incomplete cell sets')
            require(len(actual_inference) == 627+actual_baseline_dev and
                    actual_estimates == 897600+200*actual_baseline_dev, 'Actual-computation coverage mismatch')
        audit['results_status'] = 'NOT_RUN' if not metric_rows else ('PARTIAL' if pending else 'COMPLETE')
        audit['status'] = ('NOT_RUN' if not metric_rows else 'PASS_PARTIAL') if pending else 'PASS_COMPLETE'
        write_json(out / 'audit.json', audit)
        if pending and not args.partial:
            raise RuntimeError('Plan is incomplete; use --partial for explicitly incomplete tables')
        return 0
    except BaseException as error:
        audit.update(status='FAILED', error=repr(error), traceback=traceback.format_exc())
        write_json(out / 'audit.json', audit)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', required=True)
    parser.add_argument('--partial', action='store_true')
    raise SystemExit(run(parser.parse_args()))
