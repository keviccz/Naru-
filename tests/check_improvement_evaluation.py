"""Compare new evaluator output with frozen B CSV; standard library only.

Queries must be the ORIGINAL full workload. The new CSV must contain its first
--count queries, unchanged IDs/order, so original-index RNG seeds stay identical.
This script does not run a model and does not compare latency values.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_improvements import qerror, require, write_json


def csv_path(value):
    path = Path(value)
    return path / 'per_query.csv' if path.is_dir() else path


def read_rows(path, method):
    with csv_path(path).open(encoding='utf-8', newline='') as f:
        return [r for r in csv.DictReader(f) if r.get('method', method) == method]


def main(args):
    receipt = {'status': 'RUNNING', 'model_executed_by_check': False,
               'new_csv': str(csv_path(args.new)), 'old_csv': str(csv_path(args.old)),
               'original_queries': args.queries, 'count': args.count, 'latency_compared': False}
    write_json(args.output, receipt)
    try:
        require(args.count > 0, 'Positive comparison count required')
        queries = [json.loads(s) for s in Path(args.queries).read_text(encoding='utf-8').splitlines() if s.strip()]
        require(len(queries) >= args.count, 'Original workload has too few queries')
        prefix = queries[:args.count]
        ids = [q['query_id'] for q in prefix]
        require(len(set(ids)) == args.count, 'Query IDs must be unique')
        current = read_rows(args.new, args.new_method)
        frozen = read_rows(args.old, args.old_method)
        require([r['query_id'] for r in current] == ids, 'New CSV must preserve original prefix IDs/order')
        frozen_by_id = {r['query_id']: r for r in frozen}
        require(len(frozen_by_id) == len(frozen) and set(ids) <= set(frozen_by_id), 'Frozen CSV IDs missing/duplicated')
        for query, row in zip(prefix, current):
            old = frozen_by_id[query['query_id']]
            for key in ('truth', 'estimate', 'qerror'):
                a, b = float(row[key]), float(old[key])
                require(math.isfinite(a) and a >= 0 and a == b,
                        'Exact mismatch for ' + query['query_id'] + '/' + key)
            require(float(row['truth']) == query['truth'], 'Truth differs from original workload')
            require(math.isclose(float(row['qerror']), qerror(query['truth'], float(row['estimate'])),
                                 rel_tol=1e-10, abs_tol=1e-10), 'Stored qerror mismatch')
        receipt.update(status='PASS', queries_compared=args.count,
                       exact_fields=['truth', 'estimate', 'qerror'], query_ids=ids)
        write_json(args.output, receipt)
        print(json.dumps(receipt, allow_nan=False))
        return 0
    except BaseException as error:
        receipt.update(status='FAILED', error=repr(error))
        write_json(args.output, receipt)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--new', required=True, help='New CSV file or evaluation directory')
    p.add_argument('--old', required=True, help='Frozen CSV file or evaluation directory')
    p.add_argument('--queries', required=True, help='Original full workload, not an arbitrarily selected subset')
    p.add_argument('--output', required=True)
    p.add_argument('--count', type=int, default=5)
    p.add_argument('--new-method', default='own_resmade')
    p.add_argument('--old-method', default='own_resmade')
    raise SystemExit(main(p.parse_args()))
