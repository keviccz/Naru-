"""Regression for the full ablation's permuted embedding coordinate contract."""
import argparse,importlib.util
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path)
source=parser.parse_args().source or Path(__file__).resolve().parents[1]/'scripts/run_improvement_queue.py'
spec=importlib.util.spec_from_file_location('queue_mapping_under_test',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
order=[1,4,0,5,7,6,3,2,9,8,10]
for original in ([],[6,7],[1,2,6,7]):
 positions=[i for i,col in enumerate(order) if col in original]
 cfg={'column_order':order,'embedding_columns':positions,'embedding_original_columns':[order[i] for i in positions]}
 module.validate_embedding_mapping(cfg,{'embedding_original_columns':original})
 if positions:
  bad=dict(cfg,embedding_columns=list(reversed(positions)))
  try:module.validate_embedding_mapping(bad,{'embedding_original_columns':original})
  except AssertionError:pass
  else:raise AssertionError('Invalid model positions accepted')
  bad=dict(cfg,embedding_original_columns=original)
  try:module.validate_embedding_mapping(bad,{'embedding_original_columns':original})
  except AssertionError:pass
  else:raise AssertionError('Unpermuted original-coordinate list accepted')
print('PASS: natural/empty and 1M/full embedding sets; wrong model coordinates rejected')
