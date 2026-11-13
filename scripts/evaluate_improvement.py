"""Evaluate one frozen own-ResMADE variant on original-index frozen queries.

Only the input columns and allowed-mask list are permuted. Category codes,
query order, truths and per-query RNG seeds remain the frozen contracts.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--queries', required=True)
    p.add_argument('--scope', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--samples', type=int, default=2000)
    p.add_argument('--mc-seed', type=int, default=0)
    p.add_argument('--method', help='Explicit own_resmade label for missing B dev inference')
    p.add_argument('--threads', type=int, default=4)
    return p.parse_args()


def main(a):
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    state = {'status': 'STARTING', 'queries_completed': 0, 'arguments': vars(a),
             'evaluation_performed': True}
    write_json(out / 'status.json', state)
    try:
        if a.samples < 1 or a.mc_seed < 0 or a.threads < 1:
            raise ValueError('positive sample/thread counts and nonnegative MC seed required')
        gpu = subprocess.check_output(
            ['nvidia-smi', '-i', '0', '--query-gpu=uuid,name', '--format=csv,noheader'],
            text=True).strip()
        uuid, name = map(str.strip, gpu.split(',', 1))
        if 'RTX 3090' not in name:
            raise RuntimeError('Physical GPU0 differs from the approved RTX 3090: ' + gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid
        import numpy as np
        import torch
        torch.set_num_threads(a.threads)
        project = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(project))
        from src.models import CategoricalMADE, progressive_sample
        from src.baselines import qerror
        run = Path(a.run).resolve()
        cfg = json.loads((run / 'config.json').read_text(encoding='utf-8'))
        training_state = json.loads((run / 'status.json').read_text(encoding='utf-8'))
        if training_state['status'] != 'COMPLETE' or cfg['model'] != 'own_resmade':
            raise ValueError('A complete own_resmade checkpoint is required')
        original_meta = json.loads(Path(cfg['domain_path']).read_text(encoding='utf-8'))
        original_sizes = [len(d) for d in original_meta['domains']]
        ncols = len(original_sizes)
        order = list(cfg.get('column_order', range(ncols)))
        if sorted(order) != list(range(ncols)):
            raise ValueError('column_order must be a bijection of original columns')
        inverse = [order.index(i) for i in range(ncols)]
        if cfg.get('inverse_order', inverse) != inverse:
            raise ValueError('inverse_order disagrees with column_order')
        if cfg.get('original_domain_sizes', original_sizes) != original_sizes:
            raise ValueError('Original domain sizes differ from the frozen dictionary')
        if list(cfg['domain_sizes']) != [original_sizes[i] for i in order]:
            raise ValueError('Model domain order differs from the frozen permutation')
        original_columns = original_meta['columns']
        if cfg.get('original_columns', original_columns) != original_columns:
            raise ValueError('Original column names differ from frozen dictionary')
        if list(cfg['columns']) != [original_columns[i] for i in order]:
            raise ValueError('Model column names differ from permutation')
        embedding_columns = cfg.get('embedding_columns')
        model = CategoricalMADE(
            cfg['domain_sizes'], hidden=cfg['width'], layers=cfg['layers'],
            residual=cfg['residual'], direct_io=cfg['direct_io'],
            column_masking=cfg['column_masking'], mask_seed=cfg.get('mask_seed', cfg['seed']),
            embedding_columns=embedding_columns, embed_size=cfg.get('embed_size', 32)).cuda().eval()
        checkpoint = torch.load(run / 'checkpoint.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        if parameter_bytes != cfg['parameter_bytes']:
            raise ValueError('Reconstructed trainable parameter budget differs from checkpoint config')
        table = np.load(cfg['data_path'], mmap_mode='r', allow_pickle=False)
        n = len(table)
        if table.shape != (cfg['rows'], ncols):
            raise ValueError('Data shape differs from training receipt')
        method = a.method or ('own_resmade_' + cfg['variant'])
        if a.method and a.method != 'own_resmade':
            raise ValueError('The method override is reserved for B own_resmade dev inference')
        queries = [json.loads(s) for s in Path(a.queries).read_text(encoding='utf-8').splitlines() if s.strip()]
        if not queries or any(q['workload'] != a.scope for q in queries):
            raise ValueError('Nonempty queries with the requested frozen scope required')
        if len({q['query_id'] for q in queries}) != len(queries):
            raise ValueError('Duplicate query IDs')

        def masks_for(q):
            natural = [None] * ncols
            for pred in q['predicates']:
                i = pred['column_index']
                if not 0 <= i < ncols or natural[i] is not None:
                    raise ValueError('A query must have at most one predicate per original column')
                if pred['column'] != original_columns[i]:
                    raise ValueError('Predicate name/index mismatch')
                codes = pred['allowed_codes']
                if len(codes) != len(set(codes)) or any(k < 0 or k >= original_sizes[i] for k in codes):
                    raise ValueError('Predicate codes outside original frozen domain')
                mask = np.zeros(original_sizes[i], dtype=bool)
                mask[codes] = True
                natural[i] = mask
            return [natural[i] for i in order]

        warmup = next((i for i, q in enumerate(queries)
                       if all(m is None or m.any() for m in masks_for(q))), None)
        if warmup is not None:
            progressive_sample(model, masks_for(queries[warmup]), samples=a.samples,
                               seed=0, wildcard_skipping=cfg['column_masking'])
        torch.cuda.synchronize()
        state.update(status='RUNNING', gpu_uuid=uuid, physical_gpu_index=0,
                     method=method, column_order=order, inverse_order=inverse,
                     embedding_columns=embedding_columns or [], parameter_bytes=parameter_bytes)
        write_json(out / 'status.json', state)
        records = []
        with (out / 'per_query.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['query_id', 'workload', 'truth', 'method',
                                                  'estimate', 'qerror', 'latency_ms'])
            writer.writeheader()
            for qi, q in enumerate(queries):
                masks = masks_for(q)
                torch.cuda.synchronize()
                start = time.perf_counter()
                estimate = n * progressive_sample(
                    model, masks, samples=a.samples, seed=100000 + qi + 1000000 * a.mc_seed,
                    wildcard_skipping=cfg['column_masking'])
                torch.cuda.synchronize()
                elapsed = 1000 * (time.perf_counter() - start)
                row = dict(query_id=q['query_id'], workload=q['workload'], truth=q['truth'],
                           method=method, estimate=estimate,
                           qerror=float(qerror(q['truth'], estimate)), latency_ms=elapsed)
                writer.writerow(row)
                records.append(row)
                state['queries_completed'] = qi + 1
                if (qi + 1) % 10 == 0:
                    f.flush()
                    write_json(out / 'status.json', state)
        e = np.asarray([r['qerror'] for r in records])
        times = np.asarray([r['latency_ms'] for r in records])
        positive = [r for r in records if r['truth'] > 0]
        zero = [r for r in records if r['truth'] == 0]
        met = dict(qerror_median=float(np.median(e)), qerror_p95=float(np.quantile(e, .95)),
                   qerror_p99=float(np.quantile(e, .99)), qerror_max=float(e.max()),
                   latency_median_ms=float(np.median(times)), positive_truth_queries=len(positive),
                   zero_truth_queries=len(zero), zero_estimate_gt1=sum(r['estimate'] > 1 for r in zero),
                   zero_estimate_gt10=sum(r['estimate'] > 10 for r in zero),
                   positive_estimate_eq0=sum(r['estimate'] == 0 for r in positive),
                   positive_estimate_lt1=sum(r['estimate'] < 1 for r in positive))
        if positive:
            pe = np.asarray([r['qerror'] for r in positive])
            met.update(qerror_positive_median=float(np.median(pe)),
                       qerror_positive_p95=float(np.quantile(pe, .95)),
                       qerror_positive_p99=float(np.quantile(pe, .99)),
                       qerror_positive_max=float(pe.max()))
        summary = dict(status='COMPLETE', queries=len(queries), method=method,
                       scope=a.scope, variant=cfg.get('variant', 'baseline'),
                       scale=cfg['scale'], training_seed=cfg['seed'], mc_seed=a.mc_seed,
                       sampling_budget=a.samples, column_order=order, inverse_order=inverse,
                       embedding_columns=embedding_columns or [], parameter_bytes=parameter_bytes,
                       warmup_query_index=warmup,
                       sampling_seed_policy='100000+original query index+1000000*mc_seed',
                       wildcard_skipping=cfg['column_masking'], evaluation_performed=True,
                       latency_scope='single warmed query; host numpy mask construction/permutation excluded; '
                                     'tensor conversion/transfer and synchronized GPU inference included',
                       methods={method: met})
        write_json(out / 'summary.json', summary)
        state.update(status='COMPLETE', queries_completed=len(queries), exit_code=0)
        write_json(out / 'status.json', state)
        print(json.dumps(summary, allow_nan=False))
        return 0
    except BaseException as exc:
        state.update(status='FAILED', error=repr(exc), exit_code=1)
        write_json(out / 'status.json', state)
        (out / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    raise SystemExit(main(parse_args()))
