"""Development evaluation, saved per-query results; no held-out tuning."""
import argparse,csv,json,os,sys,time,subprocess,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--queries',required=True);p.add_argument('--output',required=True);p.add_argument('--samples',type=int,default=2000);p.add_argument('--sample-rows',type=int,default=10000);a=p.parse_args()
uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
os.environ['CUDA_VISIBLE_DEVICES']=uuid
import numpy as np
import torch
torch.set_num_threads(4)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models import CategoricalMADE,progressive_sample
from src.baselines import UniformSample,qerror
run=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
status={'status':'RUNNING','queries_completed':0,'arguments':vars(a),'gpu_uuid':uuid}
def save():(out/'status.json').write_text(json.dumps(status,indent=2))
save()
try:
 cfg=json.loads((run/'config.json').read_text())
 if cfg['model'] not in ('own_made','own_resmade'):raise ValueError('Only independently implemented models supported by this evaluator')
 if json.loads((run/'status.json').read_text())['status']!='COMPLETE':raise ValueError('Training incomplete')
 model=CategoricalMADE(cfg['domain_sizes'],hidden=cfg['width'],layers=cfg['layers'],residual=cfg['residual'],direct_io=cfg['direct_io'],column_masking=cfg['column_masking']).cuda().eval()
 ck=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['state_dict'])
 table=np.load(cfg['data_path'],mmap_mode='r');n=len(table)
 hist=[np.bincount(table[:,i],minlength=d) for i,d in enumerate(cfg['domain_sizes'])]
 sampler=UniformSample(table,a.sample_rows,seed=20261109)
 queries=[json.loads(s) for s in Path(a.queries).read_text().splitlines() if s.strip()]
 def masks_for(q):
  masks=[None]*len(hist)
  for pred in q['predicates']:
   i=pred['column_index'];mask=np.zeros(len(hist[i]),dtype=bool);mask[pred['allowed_codes']]=True;masks[i]=mask
  return masks
 first=masks_for(queries[0]);progressive_sample(model,first,samples=a.samples,seed=0,wildcard_skipping=cfg['column_masking']);torch.cuda.synchronize()
 records=[]
 with (out/'per_query.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['query_id','workload','truth','method','estimate','qerror','latency_ms']);writer.writeheader()
  for qi,q in enumerate(queries):
   masks=masks_for(q);truth=q['truth'];torch.cuda.synchronize();start=time.perf_counter()
   est=n*progressive_sample(model,masks,samples=a.samples,seed=100000+qi,wildcard_skipping=cfg['column_masking']);torch.cuda.synchronize();ms=1000*(time.perf_counter()-start)
   values=[(cfg['model'],est,ms)]
   start=time.perf_counter();ind=float(n)
   for i,mask in enumerate(masks):
    if mask is not None:ind*=hist[i][mask].sum()/n
   values.append(('exact_marginal_independence',ind,1000*(time.perf_counter()-start)))
   start=time.perf_counter();sample=sampler.estimate(masks);values.append(('uniform_sample',sample,1000*(time.perf_counter()-start)))
   for method,estimate,elapsed in values:
    row=dict(query_id=q['query_id'],workload=q['workload'],truth=truth,method=method,estimate=estimate,qerror=float(qerror(truth,estimate)),latency_ms=elapsed);writer.writerow(row);records.append(row)
   status['queries_completed']=qi+1
   if (qi+1)%10==0:f.flush();save()
 summary={'scope':'development queries only; not final benchmark','queries':len(queries),'sampling_budget':a.samples,'uniform_sample_rows':len(sampler.table),'wildcard_skipping':cfg['column_masking'],'latency_scope':'single query; model includes host masks, transfer, synchronized GPU inference; baseline CPU lookup; all warmed; no PG comparison','methods':{}}
 for method in sorted({r['method'] for r in records}):
  rows=[r for r in records if r['method']==method];e=np.array([r['qerror'] for r in rows]);t=np.array([r['latency_ms'] for r in rows])
  summary['methods'][method]={'qerror_median':float(np.median(e)),'qerror_p95':float(np.quantile(e,.95)),'qerror_p99':float(np.quantile(e,.99)),'qerror_max':float(e.max()),'latency_median_ms':float(np.median(t))}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));status['status']='COMPLETE';save();print(json.dumps(summary))
except BaseException as e:
 status.update(status='FAILED',error=repr(e));save();(out/'traceback.txt').write_text(traceback.format_exc());raise
