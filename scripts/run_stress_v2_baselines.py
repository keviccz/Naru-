"""CPU baselines for the frozen supplemental workloads; no model fitting."""
import csv,json,os,sys,time,traceback
from pathlib import Path
os.environ['OMP_NUM_THREADS']='2'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.baselines import UniformSample,qerror
out=ROOT/'experiments/stress_v2_baselines_20261110';out.mkdir(exist_ok=False)
state={'status':'WAITING_WORKLOAD','runs':[],'storage_contract':'logical FP32 parameter bytes vs int32 sample row bytes; shared dictionaries excluded both sides'}
def save():
 p=out/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(out/'status.json')
save()
try:
 while True:
  f=ROOT/'experiments/workload_supplement_v2_20261110/status.json'
  if f.exists():
   s=json.loads(f.read_text())['status']
   if s=='FAILED':raise RuntimeError('Workload generation failed')
   if s=='COMPLETE':break
  time.sleep(30)
 cells=json.loads((ROOT/'experiments/main_matrix_20261109/status.json').read_text())['runs']
 for scale in ['1000000','5000000','full']:
  table=np.load(ROOT/'data/snapshot_resume_20261109_172915/encoded'/f'{scale}.npy',mmap_mode='r');n=len(table)
  configs={m:json.loads((Path(next(c['path'] for c in cells if c['scale']==scale and c['model']==m and c['seed']==0))/'config.json').read_text()) for m in ['own_made','own_resmade','official_resmade']}
  hist=[np.bincount(table[:,i],minlength=d) for i,d in enumerate(configs['own_made']['domain_sizes'])]
  methods=[('exact_marginal_independence',None,None,0),('uniform_sample_10000',UniformSample(table,10000,20261109),20261109,10000*44)]
  for model,cfg in configs.items():
   for seed in [0,1,2]:methods.append(('matched_'+model,UniformSample(table,cfg['parameter_bytes']//44,seed),seed,cfg['parameter_bytes']))
  for scope in ['stress_v2']:
   queries=[json.loads(x) for x in (ROOT/'experiments/workload_supplement_v2_20261110'/scale/(scope+'.jsonl')).read_text().splitlines()]
   for method,sampler,seed,budget in methods:
    name=f'{scale}_{scope}_{method}_seed{seed}';target=out/name;target.mkdir();cell={'name':name,'scale':scale,'scope':scope,'method':method,'sample_seed':seed,'status':'RUNNING'};state['runs'].append(cell);state.update(status='RUNNING',active=name);save()
    errors=[];latencies=[]
    with (target/'per_query.csv').open('w',newline='') as f:
     writer=csv.DictWriter(f,fieldnames=['query_id','workload','truth','method','estimate','qerror','latency_ms']);writer.writeheader()
     for q in queries:
      masks=[None]*len(hist)
      for pred in q['predicates']:
       i=pred['column_index'];mask=np.zeros(len(hist[i]),dtype=bool);mask[pred['allowed_codes']]=True;masks[i]=mask
      start=time.perf_counter()
      if sampler is None:
       est=float(n)
       for i,mask in enumerate(masks):
        if mask is not None:est*=hist[i][mask].sum()/n
      else:est=sampler.estimate(masks)
      ms=1000*(time.perf_counter()-start);err=float(qerror(q['truth'],est));errors.append(err);latencies.append(ms)
      writer.writerow(dict(query_id=q['query_id'],workload=scope,truth=q['truth'],method=method,estimate=est,qerror=err,latency_ms=ms))
    summary={'status':'COMPLETE','scale':scale,'scope':scope,'method':method,'sample_seed':seed,'queries':len(queries),'parameter_budget_bytes':budget,'sample_rows':0 if sampler is None else len(sampler.table),'sample_array_bytes':0 if sampler is None else sampler.table.nbytes,'qerror_median':float(np.median(errors)),'qerror_p95':float(np.quantile(errors,.95)),'qerror_p99':float(np.quantile(errors,.99)),'qerror_max':max(errors),'latency_median_ms':float(np.median(latencies)),'timing_scope':'CPU estimate only; prebuilt histogram/sample; host mask construction excluded'}
    (target/'summary.json').write_text(json.dumps(summary,indent=2));cell['status']='COMPLETE';save()
 state['status']='COMPLETE'
except BaseException as e:
 state.update(status='FAILED',error=repr(e));(out/'traceback.txt').write_text(traceback.format_exc());raise
finally:save()
