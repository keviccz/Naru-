"""CPU-only main2000 sampling at unique embedding FP32 parameter budgets.

V-embed and V-all share the same draw at identical scale/byte/seed budgets.
Common dictionaries are excluded on both sides. Serialization is not the budget.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
import traceback

from analyze_improvements import (load_queries, metrics, path_mapper, qerror,
                                  read_json, require, validate_plan, write_json)


def ready_configs(plan, mapped):
    cells = [t for t in plan['train_cells'] if t['variant'] in ('embed', 'all')]
    configs, waiting = [], []
    for cell in cells:
        directory = mapped(cell['run_dir'])
        if not (directory / 'status.json').exists() or not (directory / 'config.json').exists():
            waiting.append(cell['id'])
            continue
        state = read_json(directory / 'status.json')
        if state['status'] == 'FAILED':
            raise RuntimeError('Training failed: ' + cell['id'])
        if state['status'] != 'COMPLETE':
            waiting.append(cell['id'])
            continue
        cfg = read_json(directory / 'config.json')
        require((cfg['variant'], cfg['scale'], cfg['seed']) ==
                (cell['variant'], cell['scale'], cell['seed']), 'Training config identity mismatch')
        configs.append((cell, cfg, read_json(directory / 'training_result.json')))
    return configs, waiting


def run(args):
    root = Path(args.experiment_root).resolve()
    project = root.parent.parent
    plan = read_json(root / 'plan.json')
    validate_plan(plan)
    mapped = path_mapper(plan, project)
    out = root / 'matched'
    out.mkdir(exist_ok=False)
    state = {'status': 'WAITING', 'completed_draws': [], 'pending_training': [],
             'storage_contract': 'FP32 trainable parameter bytes versus materialized int32 sample rows; common per-scale dictionaries excluded on both sides; no index',
             'sample_seeds': [0, 1, 2], 'GPU_used': False}
    write_json(out / 'status.json', state)
    try:
        while True:
            configs, waiting = ready_configs(plan, mapped)
            state['pending_training'] = waiting
            write_json(out / 'status.json', state)
            if not waiting:
                break
            if not args.wait:
                raise RuntimeError('Complete embed/all training configs are required: ' + ','.join(waiting))
            time.sleep(min(args.poll_seconds, 60))
        os.environ['OMP_NUM_THREADS'] = str(args.threads)
        import numpy as np
        sys.path.insert(0, str(project))
        from src.baselines import UniformSample
        draws = []
        for scale in ('1000000', '5000000', 'full'):
            group = [(c, cfg, result) for c, cfg, result in configs if c['scale'] == scale]
            budgets = {cfg['parameter_bytes'] for _, cfg, _ in group}
            require(len(group) == 6 and len(budgets) == 1, 'Embed/all must share one fixed budget per scale')
            budget = budgets.pop()
            cfg = group[0][1]
            table = np.load(mapped(cfg['data_path']), mmap_mode='r', allow_pickle=False)
            require(table.ndim == 2 and table.dtype == np.dtype('int32') and len(table) == cfg['rows'],
                    'Frozen int32 sample storage contract mismatch')
            row_bytes = table.shape[1] * 4
            sample_rows = min(len(table), budget // row_bytes)
            require(sample_rows > 0, 'Budget cannot store even one row')
            sizes = cfg['original_domain_sizes']
            require(len(sizes) == table.shape[1], 'Original domains mismatch')
            query_cell = next(c for c in plan['eval_cells'] if c['scale'] == scale and c['workload'] == 'main_native')
            queries = load_queries(mapped(query_cell['queries_path']), 'main_native', 2000)
            refs = [{'train_id': c['id'], 'variant': c['variant'], 'training_seed': c['seed'],
                     'parameter_bytes': conf['parameter_bytes'], 'checkpoint_serialized_bytes': result['checkpoint_bytes']}
                    for c, conf, result in group]
            for sample_seed in range(3):
                name = scale + '_bytes' + str(budget) + '_sample_seed' + str(sample_seed)
                draw = {'id': name, 'scale': scale, 'parameter_budget_bytes': budget,
                        'sample_seed': sample_seed, 'sample_rows': sample_rows,
                        'int32_row_bytes': row_bytes, 'source_data_path': cfg['data_path'],
                        'shared_variants': ['embed', 'all'], 'reference_train_ids': [r['train_id'] for r in refs]}
                target = out / name
                target.mkdir()
                cellstate = {'status': 'RUNNING', 'queries_completed': 0, 'draw_id': name}
                write_json(target / 'status.json', cellstate)
                state.update(status='RUNNING', active_draw=name)
                write_json(out / 'status.json', state)
                sampler = UniformSample(table, sample_rows, sample_seed)
                require(sampler.table.nbytes == sample_rows * row_bytes and sampler.table.nbytes <= budget,
                        'Materialized sample exceeds logical parameter budget')
                rows = []
                with (target / 'per_query.csv').open('w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=['query_id', 'workload', 'truth', 'method', 'estimate', 'qerror', 'latency_ms'])
                    writer.writeheader()
                    for q in queries.values():
                        masks = [None] * table.shape[1]
                        for pred in q['predicates']:
                            i = pred['column_index']
                            require(masks[i] is None, 'Duplicate predicate column')
                            mask = np.zeros(sizes[i], dtype=bool)
                            mask[pred['allowed_codes']] = True
                            masks[i] = mask
                        start = time.perf_counter()
                        estimate = float(sampler.estimate(masks))
                        elapsed = 1000 * (time.perf_counter() - start)
                        row = {'query_id': q['query_id'], 'workload': q['workload'], 'truth': q['truth'],
                               'method': 'uniform_matched_improvement', 'estimate': estimate,
                               'qerror': qerror(q['truth'], estimate), 'latency_ms': elapsed}
                        rows.append(row)
                        writer.writerow(row)
                        if len(rows) % 100 == 0:
                            f.flush()
                            cellstate['queries_completed'] = len(rows)
                            write_json(target / 'status.json', cellstate)
                summary = dict(draw, status='COMPLETE', queries=len(rows),
                               reference_models=refs, sample_array_bytes=int(sampler.table.nbytes),
                               storage_contract=state['storage_contract'], sample_ids_saved=False,
                               sampling_policy='NumPy default_rng(sample_seed).choice(population, sample_rows, replace=False)',
                               latency_scope='host masks excluded; CPU exact scan of materialized sample included',
                               methods={'uniform_matched_improvement': metrics(rows)})
                write_json(target / 'summary.json', summary)
                cellstate.update(status='COMPLETE', queries_completed=len(rows), exit_code=0)
                write_json(target / 'status.json', cellstate)
                draws.append(draw)
                write_json(out / 'manifest.json', {'schema_version': 1, 'draws': draws,
                           'logical_variant_references': 2 * len(draws),
                           'deduplication_key': ['scale', 'parameter_budget_bytes', 'sample_seed'],
                           'expected_unique_draws': 9, 'storage_contract': state['storage_contract']})
                state['completed_draws'].append(name)
                write_json(out / 'status.json', state)
                del sampler
        require(len(draws) == 9, 'Expected exactly nine independent sample draws')
        state.update(status='COMPLETE', active_draw=None, exit_code=0)
        write_json(out / 'status.json', state)
        return 0
    except BaseException as error:
        state.update(status='FAILED', error=repr(error), exit_code=1)
        write_json(out / 'status.json', state)
        (out / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        if state.get('active_draw'):
            write_json(out / state['active_draw'] / 'status.json',
                       {'status': 'FAILED', 'error': repr(error), 'exit_code': 1})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', required=True)
    parser.add_argument('--wait', action='store_true', help='Wait for all 18 embed/all training receipts')
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    require(args.poll_seconds > 0 and args.threads > 0, 'Positive poll/thread counts required')
    raise SystemExit(run(args))
