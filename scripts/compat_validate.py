"""CPU-only independent upstream sampler and fixed-checkpoint version probe.

No vendor changes. Upstream receives numeric order-preserving domain surrogates;
NULL remains NaN. Main saved allowed_codes must equal upstream's numpy masks.
"""
import argparse,csv,itertools,json,os,platform,sys,time,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'vendor/naru'))
sys.path.insert(0,str(ROOT/'experiments/compat_supplement_20261110/upstream'))
import made,common,estimators
from src.models import progressive_sample

POLICY={'declared_before_execution':True,'device':'CPU only','samples':1024,'repeats':5,
 'mc_tolerance':'max(1e-6,0.05*max(abs(mean_upstream),abs(mean_shared)),6*sqrt(var_upstream/5+var_shared/5))',
 'exact_tolerance':'max(1e-6,0.05*abs(exact),6*sample_SE)',
 'version_logits_nll_max_abs':2e-5,'version_estimates':'same MC tolerance as independent sampler',
 'rng_policy':'Five fixed seeds per implementation; upstream global RNG versus shared local generator; no bitwise Monte Carlo requirement',
 'real_selection':'first 12 saved development queries, before inspecting results',
 'null_semantics':'ordinary comparisons to NaN yield no matches; no IS NULL extension in upstream API',
 'known_upstream_edge':'shortcircuit=True _sample_n requires >=2 constrained columns; report one/all-wildcard exceptions without patching vendor'}

class Adapter(torch.nn.Module):
 def __init__(self,raw,sizes):
  super().__init__();self.raw=raw;self.ncols=len(sizes);self.domain_sizes=sizes;self.column_masking=True
 def split_logits(self,x):return x.split(self.domain_sizes,dim=-1)
 def forward(self,x):
  enc=[]
  for i in range(self.ncols):
   v=self.raw.EncodeInput(x[:,i:i+1].clamp_min(0),natural_col=i)
   enc.append(torch.where(x[:,i:i+1]<0,self.raw.unk_embeddings[i].reshape(1,-1),v))
  return self.raw.forward_with_encoded_input(torch.cat(enc,dim=1))

def model(sizes,width,layers,seed):
 return made.MADE(len(sizes),[width]*layers,sum(sizes),input_bins=sizes,input_encoding='binary',output_encoding='one_hot',embed_size=32,seed=seed,residual_connections=True,do_direct_io_connections=True,column_masking=True,natural_ordering=False,fixed_ordering=np.arange(len(sizes))).eval()

def table_for(domains,names,nrows):
 cols=[]
 for name,domain in zip(names,domains):
  values=np.arange(len(domain),dtype=np.float64)
  for i,v in enumerate(domain):
   if v is None:values[i]=np.nan
  col=common.Column(name,distribution_size=len(domain));col.data=np.zeros(1);col.all_distinct_values=values;cols.append(col)
 table=common.Table('compat_domain_protocol',cols);table.cardinality=nrows
 return table

def query_from_preds(q,table):
 ops=[None]*len(table.columns);vals=[None]*len(ops);masks=[None]*len(ops)
 for p in q['predicates']:
  i=p['column_index'];ops[i]=p['operator'];vals[i]=table.columns[i].all_distinct_values[p['value_code']]
  masks[i]=estimators.OPS[ops[i]](table.columns[i].all_distinct_values,vals[i])
  if 'allowed_codes' in p:assert list(np.flatnonzero(masks[i]))==p['allowed_codes'],q['query_id']
 return ops,vals,masks

def exact_probability(raw,sizes,masks,skip):
 active=[i for i,m in enumerate(masks) if m is not None] if skip else list(range(len(sizes)))
 rows=list(itertools.product(*[range(sizes[i]) for i in active]));x=torch.zeros(len(rows),len(sizes),dtype=torch.long)
 if skip:x.fill_(-1)
 for j,i in enumerate(active):x[:,i]=torch.tensor([r[j] for r in rows])
 logits=Adapter(raw,sizes)(x);p=torch.ones(len(rows),dtype=torch.float64)
 for i,part in enumerate(logits.split(sizes,dim=-1)):
  if i not in active:continue
  codes=x[:,i];p*=part.softmax(1).gather(1,codes[:,None]).squeeze().double()
  if masks[i] is not None:p*=torch.as_tensor(masks[i])[codes]
 return p.sum().item()

def run_suite(label,raw,table,queries,out,rows=None,synthetic=False):
 sizes=[c.DistributionSize() for c in table.columns];adapter=Adapter(raw,sizes).eval()
 records=[];edges=[]
 # Separate objects: upstream constructor caches masked weights and detaches params.
 upstream_raw=model(sizes,32 if synthetic else 256,3 if synthetic else 5,71 if synthetic else 0)
 upstream_raw.load_state_dict(raw.state_dict())
 for skip in ([False,True] if synthetic else [True]):
  upstream=estimators.ProgressiveSampling(upstream_raw,table,POLICY['samples'],device=torch.device('cpu'),shortcircuit=skip)
  for q in queries:
   ops,vals,masks=query_from_preds(q,table);u=[];s=[]
   if skip and sum(o is not None for o in ops)<2:
    try:
     upstream.inp.zero_();upstream._sample_n(POLICY['samples'],np.arange(len(sizes)),table.columns,ops,vals)
     edges.append({'query_id':q['query_id'],'unexpected':'upstream accepted edge'})
    except Exception as e:edges.append({'query_id':q['query_id'],'upstream_exception':repr(e),'shared_selectivity':progressive_sample(adapter,masks,samples=POLICY['samples'],seed=700,wildcard_skipping=skip)})
    continue
   for r in range(POLICY['repeats']):
    seed=700+r;torch.manual_seed(seed);upstream.inp.zero_()
    started=time.perf_counter();uv=upstream._sample_n(POLICY['samples'],np.arange(len(sizes)),table.columns,ops,vals)
    um=time.perf_counter()-started
    started=time.perf_counter();sv=progressive_sample(adapter,masks,samples=POLICY['samples'],seed=seed,wildcard_skipping=skip);sm=time.perf_counter()-started
    u.append(uv);s.append(sv)
    records.append({'suite':label,'query_id':q['query_id'],'skip':skip,'repeat':r,'seed':seed,'upstream':uv,'shared':sv,'upstream_seconds':um,'shared_seconds':sm})
   se=float(np.sqrt(np.var(u,ddof=1)/len(u)+np.var(s,ddof=1)/len(s)))
   tol=max(1e-6,.05*max(abs(np.mean(u)),abs(np.mean(s))),6*se)
   check={'suite':label,'query_id':q['query_id'],'skip':skip,'mean_upstream':float(np.mean(u)),'mean_shared':float(np.mean(s)),'abs_diff':float(abs(np.mean(u)-np.mean(s))),'tolerance':tol,'pass':bool(abs(np.mean(u)-np.mean(s))<=tol),'predicate_count':sum(o is not None for o in ops)}
   if synthetic:
    exact=exact_probability(raw,sizes,masks,skip);check['exact_model_probability']=exact
    for name,values in [('upstream',u),('shared',s)]:
     exact_tol=max(1e-6,.05*abs(exact),6*float(np.std(values,ddof=1)/np.sqrt(len(values))))
     check[name+'_exact_pass']=bool(abs(np.mean(values)-exact)<=exact_tol)
   with (out/'checks.jsonl').open('a') as f:f.write(json.dumps(check)+'\n')
   print(json.dumps(check),flush=True)
 with (out/(label+'_repeats.csv')).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
 (out/(label+'_edges.json')).write_text(json.dumps(edges,indent=2))
 if rows is not None:
  x=torch.as_tensor(np.array(rows[:64]),dtype=torch.long)
  with torch.no_grad():
   logits=raw(x);nll=raw.nll(logits,x)
   masked=x.clone();masked[:,::2]=-1;masked_logits=adapter(masked)
  np.savez(out/'fixed_checkpoint_outputs.npz',logits=logits.numpy(),nll=nll.numpy(),masked_logits=masked_logits.numpy(),input=x.numpy())

def main():
 p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args()
 out=ROOT/'experiments/compat_supplement_20261110'/a.label;out.mkdir(parents=True,exist_ok=False)
 (out/'policy.json').write_text(json.dumps(POLICY,indent=2))
 env={'python':sys.version,'python_executable':sys.executable,'prefix':sys.prefix,'torch':torch.__version__,'torch_file':torch.__file__,'numpy':np.__version__,'platform':platform.platform(),'device':'cpu','threads':torch.get_num_threads(),'vendor_commit':'264cf4e9c96c9e34422f9eebc455a714aeef0b57','upstream_file':str(Path(estimators.__file__))}
 (out/'environment.json').write_text(json.dumps(env,indent=2));(out/'status.json').write_text('{"status":"RUNNING"}')
 try:
  torch.manual_seed(71);sizes=[3,4,3,2];raw=model(sizes,32,3,71);domains=[[None,1,2],list(range(4)),list(range(3)),list(range(2))];table=table_for(domains,['A','B','C','D'],72)
  specs=[('eq_range',[(0,'=',2),(1,'>=',1),(3,'=',1)]),('leading_wildcard',[(1,'<=',2),(3,'=',0)]),('middle_wildcard',[(0,'>=',1),(3,'=',1)]),('null_comparison',[(0,'=',0),(2,'=',1)]),('non_null_range',[(0,'<=',2),(2,'>=',1)]),('all_wildcards',[]),('one_filter',[(2,'=',1)])]
  qs=[{'query_id':n,'predicates':[{'column_index':i,'operator':o,'value_code':v} for i,o,v in ps]} for n,ps in specs]
  run_suite('synthetic',raw,table,qs,out,synthetic=True)
  run=ROOT/'experiments/main_matrix_20261109/1000000_official_resmade_seed0';cfg=json.loads((run/'config.json').read_text());d=json.loads(Path(cfg['domain_path']).read_text());raw=model(cfg['domain_sizes'],cfg['width'],cfg['layers'],cfg['seed']);raw.load_state_dict(torch.load(run/'checkpoint.pt',map_location='cpu')['state_dict'])
  queries=[json.loads(x) for x in (ROOT/'data/snapshot_resume_20261109_172915/workloads/1000000/dev.jsonl').read_text().splitlines()][:12]
  rows=np.load(cfg['data_path'],mmap_mode='r');table=table_for(d['domains'],d['columns'],len(rows));run_suite('real_1m',raw,table,queries,out,rows=rows)
  # A fresh model avoids inference-only cached weights/detached parameters.
  # This tests actual masked ResMADE forward/backward/Adam compatibility only.
  with torch.enable_grad():
   smoke=model(cfg['domain_sizes'],cfg['width'],cfg['layers'],cfg['seed']);smoke.load_state_dict(torch.load(run/'checkpoint.pt',map_location='cpu')['state_dict']);smoke.train()
   optim=torch.optim.Adam(smoke.parameters(),lr=1e-4);torch.manual_seed(20261110);losses=[]
   first=next(smoke.parameters()).detach().clone()
   for step in range(3):
    batch=torch.tensor(np.array(rows[step*128:(step+1)*128]),dtype=torch.long);optim.zero_grad();loss=smoke.nll(smoke(batch),batch).mean();loss.backward();optim.step();losses.append(float(loss.detach()))
   change=float((next(smoke.parameters()).detach()-first).abs().max());smoke_pass=bool(np.isfinite(losses).all() and change>0)
   (out/'training_smoke.json').write_text(json.dumps({'scope':'3 CPU Adam steps, batch128, actual official column masking enabled; compatibility only, not accuracy or convergence','losses':losses,'first_parameter_max_abs_change':change,'passed':smoke_pass},indent=2))
  checks=[json.loads(x) for x in (out/'checks.jsonl').read_text().splitlines()];passed=smoke_pass and all(x['pass'] and x.get('upstream_exact_pass',True) and x.get('shared_exact_pass',True) for x in checks)
  (out/'summary.json').write_text(json.dumps({'status':'COMPLETE','passed':passed,'checks':len(checks),'failed':[x for x in checks if not x['pass'] or not x.get('upstream_exact_pass',True) or not x.get('shared_exact_pass',True)],'scope':'independent upstream ProgressiveSampling CPU correctness, not upstream GPU latency, not full historical 1.13 training reproduction','checkpoint':str(run/'checkpoint.pt')},indent=2));(out/'status.json').write_text(json.dumps({'status':'COMPLETE','passed':passed}))
 except BaseException:
  (out/'status.json').write_text(json.dumps({'status':'FAILED','error':traceback.format_exc()}));raise
if __name__=='__main__':main()
