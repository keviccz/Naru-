"""Equal logical synopsis storage: FP32 parameters vs int32 sampled rows.

Per-scale value dictionaries are common overhead and excluded on BOTH sides.
NPY/checkpoint serialization overhead separately reported, not mixed in budgets.
"""
import csv,json,os,sys,time,traceback
from pathlib import Path
os.environ['OMP_NUM_THREADS']='4'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));from src.baselines import UniformSample,qerror
out=ROOT/'experiments/matched_sampling_20261110';out.mkdir(exist_ok=False)
state={'status':'RUNNING','runs':[],'storage_contract':'logical FP32 trainable parameters vs materialized int32 sample rows; same per-scale dictionary excluded from both; no index','sample_seeds':[0,1,2]}
def save():(out/'status.json').write_text(json.dumps(state,indent=2))
save()
try:
 cells=json.loads((ROOT/'experiments/main_matrix_20261109/status.json').read_text())['runs']
 for scale in ['1000000','5000000','full']:
  table=np.load(ROOT/'data/snapshot_resume_20261109_172915/encoded'/f'{scale}.npy',mmap_mode='r');n=len(table)
  queries=[json.loads(x) for x in (ROOT/'data/snapshot_resume_20261109_172915/workloads'/scale/'main_native.jsonl').read_text().splitlines()]
  for model in ['own_made','own_resmade','official_resmade']:
   cfg=json.loads((Path(next(c['path'] for c in cells if c['scale']==scale and c['model']==model and c['seed']==0))/'config.json').read_text())
   budget=cfg['parameter_bytes'];sample_rows=min(n,budget//(table.shape[1]*4))
   for seed in [0,1,2]:
    name=f'{scale}_{model}_sample_seed{seed}';target=out/name;target.mkdir();cell={'name':name,'status':'RUNNING','rows':sample_rows,'parameter_budget_bytes':budget};state['runs'].append(cell);state['active']=name;save()
    sampler=UniformSample(table,sample_rows,seed);errors=[];times=[]
    with (target/'per_query.csv').open('w',newline='') as f:
     writer=csv.DictWriter(f,fieldnames=['query_id','truth','estimate','qerror','latency_ms']);writer.writeheader()
     for q in queries:
      masks=[None]*table.shape[1]
      for pred in q['predicates']:
       i=pred['column_index'];mask=np.zeros(cfg['domain_sizes'][i],dtype=bool);mask[pred['allowed_codes']]=True;masks[i]=mask
      start=time.perf_counter();est=sampler.estimate(masks);ms=1000*(time.perf_counter()-start);err=float(qerror(q['truth'],est));errors.append(err);times.append(ms)
      writer.writerow(dict(query_id=q['query_id'],truth=q['truth'],estimate=est,qerror=err,latency_ms=ms))
    summary={'status':'COMPLETE','queries':len(errors),'scale':scale,'reference_model':model,'sample_seed':seed,'parameter_budget_bytes':budget,'sample_rows':sample_rows,'sample_array_bytes':sampler.table.nbytes,'reference_checkpoint_bytes':cfg.get('parameter_bytes'),'shared_dictionary_policy':'excluded both sides','qerror_median':float(np.median(errors)),'qerror_p95':float(np.quantile(errors,.95)),'qerror_p99':float(np.quantile(errors,.99)),'qerror_max':max(errors),'latency_median_ms':float(np.median(times))}
    (target/'summary.json').write_text(json.dumps(summary,indent=2));cell.update(status='COMPLETE');save()
 state['status']='COMPLETE'
except BaseException as e:state.update(status='FAILED',error=repr(e));(out/'traceback.txt').write_text(traceback.format_exc());raise
finally:save()
