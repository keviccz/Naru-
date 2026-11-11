"""Portable CPU end-to-end acceptance. Tiny accuracy is not a DMV benchmark."""
import argparse,csv,datetime,importlib.util,itertools,json,math,operator,os,site,sqlite3,statistics,sys,time,traceback
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'package_source'))
import numpy as np
import torch
from src.models import CategoricalMADE,predicate_mask,progressive_sample
from src.baselines import UniformSample,independence,qerror
torch.set_num_threads(2);torch.set_num_interop_threads(1)
COLS=['region','vehicle','date','flag'];NULL='__TINY_NULL__'
def inside(path,prefix):
 try:Path(path).resolve().relative_to(prefix);return True
 except ValueError:return False
def dump(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def csvwrite(p,rows):
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def quantile(v,q):
 v=sorted(v);x=(len(v)-1)*q;i=int(x);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)
def summarize(out,name='summary.json'):
 rows=list(csv.DictReader((out/'per_query.csv').open()));groups={}
 for r in rows:
  t,e=float(r['truth']),float(r['estimate']);expected=max(max(1,t)/max(1,e),max(1,e)/max(1,t));assert math.isclose(expected,float(r['qerror']),rel_tol=1e-10)
  groups.setdefault(r['method'],[]).append(float(r['qerror']))
 result={'scope':'Tiny CPU end-to-end acceptance only; no DMV performance claim','raw_rows':len(rows),'methods':{k:dict(queries=len(v),qerror_median=statistics.median(v),qerror_p95=quantile(v,.95),qerror_max=max(v)) for k,v in groups.items()}}
 dump(out/name,result);return result
class Official(torch.nn.Module):
 def __init__(self,sizes):
  super().__init__();self.domain_sizes=tuple(sizes);self.ncols=len(sizes);self.column_masking=True
  spec=importlib.util.spec_from_file_location('tiny_official_made',P/'package_source/vendor/naru/made.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  self.raw=mod.MADE(self.ncols,[32]*3,sum(sizes),input_bins=sizes,input_encoding='binary',output_encoding='one_hot',seed=0,residual_connections=True,do_direct_io_connections=True,column_masking=True,natural_ordering=False,fixed_ordering=np.arange(self.ncols))
 def split_logits(self,x):return x.split(self.domain_sizes,dim=-1)
 def forward(self,x):
  if self.training:return self.raw(x)
  values=[]
  for i in range(self.ncols):
   enc=self.raw.EncodeInput(x[:,i:i+1].clamp_min(0),natural_col=i)
   values.append(torch.where(x[:,i:i+1]<0,self.raw.unk_embeddings[i].reshape(1,-1),enc))
  return self.raw.forward_with_encoded_input(torch.cat(values,dim=1))
 def nll(self,x):return self.raw.nll(self.forward(x),x)
def build(name,sizes):
 torch.manual_seed(0);np.random.seed(0)
 return Official(sizes) if name=='official_resmade' else CategoricalMADE(sizes,hidden=32,layers=3,residual=name!='own_made',direct_io=name!='own_made',column_masking=name!='own_made',mask_seed=0)
def main(out):
 out.mkdir(parents=True,exist_ok=False);status={'status':'RUNNING','phase':'environment','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};dump(out/'status.json',status)
 try:
  prefix=Path(sys.prefix).resolve();cfg=(prefix/'pyvenv.cfg').read_text();assert 'include-system-site-packages = false' in cfg.lower()
  assert sys.prefix!=sys.base_prefix and not site.ENABLE_USER_SITE
  assert torch.__version__=='2.4.1+cpu' and np.__version__=='1.24.4' and not torch.cuda.is_available()
  for mod in [torch,np]:assert inside(mod.__file__,prefix)
  assert not any('site-packages' in p and not inside(p,prefix) for p in sys.path)
  dump(out/'environment.json',{'python':sys.version,'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'sys_path':sys.path,'user_site_enabled':site.ENABLE_USER_SITE,'pyvenv_cfg':cfg,'torch':torch.__version__,'torch_file':torch.__file__,'numpy':np.__version__,'numpy_file':np.__file__,'cuda_available':torch.cuda.is_available(),'device':'CPU','LD_LIBRARY_PATH':os.environ.get('LD_LIBRARY_PATH',''),'threads':torch.get_num_threads()})
  checks=[]
  def check(name,condition,**extra):
   checks.append(dict(name=name,passed=bool(condition),**extra));assert condition,name
  raw=[[None if x==NULL else x for x in row.values()] for row in csv.DictReader((P/'data/tiny.csv').open(encoding='utf-8'))]
  check('raw_csv_rows',len(raw)==2048)
  domains=[sorted({row[i] for row in raw},key=lambda x:(x is not None,'' if x is None else x)) for i in range(4)];sizes=list(map(len,domains));codes=[{v:i for i,v in enumerate(d)} for d in domains]
  table=np.asarray([[codes[i][v] for i,v in enumerate(row)] for row in raw],dtype=np.int32);np.save(out/'encoded.npy',table);dump(out/'domains.json',{'columns':COLS,'domains':domains,'null_sentinel':NULL,'empty_string_is_value':True})
  check('null_distinct_from_empty',None in domains[3] and '' in domains[3] and codes[3][None]!=codes[3][''])
  check('encoded_raw_roundtrip',all([domains[i][table[j,i]] for i in range(4)]==row for j,row in enumerate(raw)))
  connection=sqlite3.connect(':memory:');connection.execute('CREATE TABLE tiny(region TEXT COLLATE BINARY,vehicle TEXT COLLATE BINARY,date TEXT COLLATE BINARY,flag TEXT COLLATE BINARY)');connection.executemany('INSERT INTO tiny VALUES(?,?,?,?)',raw)
  cases=[[],[(0,'=','East')],[(1,'=','bus')],[(2,'<=','2026-01-02')],[(3,'IS NULL',None)],[(3,'IS NOT NULL',None)],[(3,'=','')],[(3,'=',None)],[(0,'=','East'),(1,'=','bus')],[(2,'<','2026-01-01')],[(0,'=','West'),(3,'=','Y')],[(1,'=','van'),(2,'>=','2026-01-02'),(3,'=','N')],[(0,'=','East'),(1,'=','car'),(2,'=','2026-01-02')],[(3,'=','Y')]]
  workload=[];allmasks=[]
  for qi,preds in enumerate(cases):
   masks=[None]*4;clauses=[];params=[];saved=[]
   for i,op,value in preds:
    masks[i]=predicate_mask(domains[i],op,value).numpy();clauses.append('"'+COLS[i]+'" '+op+('' if op.startswith('IS ') else ' ?'))
    if not op.startswith('IS '):params.append(value)
    saved.append(dict(column=COLS[i],column_index=i,operator=op,value=value,allowed_codes=np.flatnonzero(masks[i]).tolist()))
   sql='SELECT COUNT(*) FROM tiny'+(' WHERE '+' AND '.join(clauses) if clauses else '');truth=connection.execute(sql,params).fetchone()[0]
   valid=np.ones(len(table),dtype=bool)
   for i,m in enumerate(masks):
    if m is not None:valid &= m[table[:,i]]
   check('sql_vs_encoded_'+str(qi),int(valid.sum())==truth)
   workload.append(dict(query_id=f'tiny_{qi:03d}',predicates=saved,truth=truth,sql=sql,sql_parameters=params));allmasks.append(masks)
  check('nonempty_marginal_joint_zero',workload[8]['truth']==0 and all(m is None or m.any() for m in allmasks[8]))
  with (out/'queries.jsonl').open('w',encoding='utf-8') as f:
   for q in workload:f.write(json.dumps(q)+'\n')
  x=torch.from_numpy(table.astype(np.int64));enumerated=torch.tensor(list(itertools.product(*[range(n) for n in sizes])),dtype=torch.long);curves=[];records=[];training=[]
  for name in ['own_made','own_resmade','official_resmade']:
   status.update(phase='training',model=name);dump(out/'status.json',status);model=build(name,sizes);model.eval()
   with torch.no_grad():initial=float(model.nll(x).mean())
   optimizer=torch.optim.Adam(model.parameters(),lr=.003);start=time.perf_counter()
   for epoch in range(1,81):
    model.train();order=torch.randperm(len(x));total=0.
    for pos in range(0,len(x),128):
     batch=x[order[pos:pos+128]];optimizer.zero_grad();loss=model.nll(batch).mean();check_finite=bool(torch.isfinite(loss));assert check_finite
     loss.backward();optimizer.step();total+=float(loss.detach())*len(batch)
    curves.append(dict(model=name,epoch=epoch,training_nll_nats=total/len(x),elapsed_seconds=time.perf_counter()-start))
   model.eval()
   with torch.no_grad():final=float(model.nll(x).mean());before=model(x[:64]).clone()
   check('training_improves_'+name,math.isfinite(final) and final<initial,initial_nll=initial,final_nll=final)
   checkpoint=out/(name+'.pt');torch.save({'state_dict':model.state_dict(),'domain_sizes':sizes,'model':name},checkpoint);loaded=build(name,sizes);loaded.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict']);loaded.eval()
   with torch.no_grad():difference=float((loaded(x[:64])-before).abs().max())
   check('checkpoint_reload_'+name,difference==0,max_abs_logit_difference=difference)
   model=loaded
   for i in range(4):
    changed=x[:32].clone()
    for j in range(i,4):changed[:,j]=(changed[:,j]+1)%sizes[j]
    with torch.no_grad():delta=float((model.split_logits(model(x[:32]))[i]-model.split_logits(model(changed))[i]).abs().max())
    check('future_mask_'+name+'_'+str(i),delta<1e-6,max_abs_difference=delta)
   if name!='official_resmade':
    invalid=x[:2].clone();invalid[:,3]=-1
    try:model.validate_codes(invalid);rejected=False
    except ValueError:rejected=True
    check('negative_code_rejected_'+name,rejected)
    if model.column_masking:model.validate_codes(invalid,allow_mask=True);check('mask_allowed_'+name,True)
   with torch.no_grad():
    logits=model.split_logits(model(enumerated));joint=torch.ones(len(enumerated),dtype=torch.float64)
    for i,piece in enumerate(logits):joint*=piece.softmax(1)[torch.arange(len(enumerated)),enumerated[:,i]].double()
   check('joint_probability_normalized_'+name,abs(float(joint.sum())-1)<2e-6,sum_probability=float(joint.sum()))
   for qi,(q,masks) in enumerate(zip(workload,allmasks)):
    good=np.ones(len(enumerated),dtype=bool)
    for i,m in enumerate(masks):
     if m is not None:good &= m[enumerated[:,i].numpy()]
    exact=float(joint[torch.from_numpy(good)].sum());rep=[progressive_sample(model,masks,samples=1024,seed=2000+k,wildcard_skipping=False) for k in range(10)];mean=statistics.mean(rep);se=statistics.stdev(rep)/math.sqrt(10);tol=max(.015,6*se)
    check('sampler_vs_model_enumeration_'+name+'_'+str(qi),abs(mean-exact)<=tol,exact_model_probability=exact,mc_mean=mean,mc_standard_error=se,tolerance=tol)
    est=len(table)*progressive_sample(model,masks,samples=1024,seed=100000+qi,wildcard_skipping=model.column_masking)
    records.append(dict(query_id=q['query_id'],truth=q['truth'],method=name,estimate=est,qerror=float(qerror(q['truth'],est))))
   training.append(dict(model=name,epochs=80,steps=1280,initial_unmasked_nll_nats=initial,final_unmasked_nll_nats=final,elapsed_seconds=time.perf_counter()-start,checkpoint_bytes=checkpoint.stat().st_size))
  sampler=UniformSample(table,256,20261111)
  for q,masks in zip(workload,allmasks):
   for name,estimate in [('exact_marginal_independence',independence(table,masks)),('uniform_sample256',sampler.estimate(masks))]:records.append(dict(query_id=q['query_id'],truth=q['truth'],method=name,estimate=estimate,qerror=float(qerror(q['truth'],estimate))))
  csvwrite(out/'per_query.csv',records);csvwrite(out/'training_curves.csv',curves);csvwrite(out/'training_summary.csv',training);dump(out/'checks.json',checks);summarize(out)
  status.update(status='COMPLETE',phase='done',checks=len(checks),training_cells=3,raw_rows=len(records),epochs_per_model=80,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());dump(out/'status.json',status);print(json.dumps(status))
 except BaseException as error:
  status.update(status='FAILED',error=repr(error));dump(out/'status.json',status);(out/'traceback.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--summarize-only',action='store_true');args=p.parse_args()
 if args.summarize_only:
  old=json.loads((args.output/'summary.json').read_text());new=summarize(args.output,'summary_rebuilt.json');assert old==new;print('SUMMARY_REBUILD_PASS')
 else:main(args.output)
