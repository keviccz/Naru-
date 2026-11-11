"""Independent raw-result reconstruction for supplementary experiment handoff."""
import argparse,csv,json,math,statistics,time,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args()
R=Path(__file__).resolve().parents[1];E=R/'experiments';O=E/'supplement_analysis_20261110';O.mkdir(exist_ok=True)
audit_path=R/'evidence/supplement_numerical_audit.json';table_counts={}
audit_path.write_text(json.dumps({'status':'RUNNING','started_unix':time.time()}),encoding='utf-8')
def fail(typ,value,tb):
 audit_path.write_text(json.dumps({'status':'FAILED','checked_unix':time.time(),'error_type':typ.__name__,'error':str(value),'derived_tables_valid':False}),encoding='utf-8');sys.__excepthook__(typ,value,tb)
sys.excepthook=fail
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def local(p):return R/str(p).split('DSC5003_Naru_20261109/',1)[1]
def quantile(v,q):
 v=sorted(v);x=(len(v)-1)*q;i=int(x);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)
def write(name,rows):
 table_counts[name]=len(rows)
 if not rows:
  (O/name).write_text('',encoding='utf-8');return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (O/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def metrics(rows):
 e=[float(r['qerror']) for r in rows];t=[float(r['latency_ms']) for r in rows];z=[r for r in rows if float(r['truth'])==0];positive=[float(r['qerror']) for r in rows if float(r['truth'])>0];ze=[float(r['estimate']) for r in z]
 return dict(queries=len(rows),qerror_median=statistics.median(e),qerror_p95=quantile(e,.95),qerror_p99=quantile(e,.99),qerror_max=max(e),qerror_positive_queries=len(positive),qerror_positive_median=statistics.median(positive) if positive else None,qerror_positive_p95=quantile(positive,.95) if positive else None,qerror_positive_p99=quantile(positive,.99) if positive else None,qerror_positive_max=max(positive) if positive else None,latency_median_ms=statistics.median(t),zero_truth_queries=len(z),zero_truth_estimate_gt0=sum(v>0 for v in ze),zero_truth_estimate_gt1=sum(v>1 for v in ze),zero_truth_estimate_mean=statistics.mean(ze) if ze else None,zero_truth_estimate_median=statistics.median(ze) if ze else None,zero_truth_estimate_p95=quantile(ze,.95) if ze else None,zero_truth_estimate_max=max(ze) if ze else None,positive_truth_estimate_zero=sum(float(r['truth'])>0 and float(r['estimate'])==0 for r in rows))
allstats=[];grouped=[];pending=[];roots={};rawcount=0;matched=[];cross=[];matched_draws={}
cache={}
def queries(scale,scope):
 key=scale,scope
 if key not in cache:
  path=(R/'data/snapshot_resume_20261109_172915/workloads'/scale/'main_native.jsonl') if scope=='main_native' else E/('workload_supplement_v2_20261110' if scope=='stress_v2' else 'workload_supplement_20261110')/scale/(scope+'.jsonl')
  cache[key]={q['query_id']:q for q in map(json.loads,path.read_text(encoding='utf-8').splitlines())}
 return cache[key]
specs=[('matched_sampling_20261110',27,'COMPLETE'),('gpu_supplement_20261110',114,'COMPLETE_BUDGET_SIZE'),('workload_eval_supplement_v2_20261110',81,'COMPLETE'),('workload_baselines_20261110',99,'COMPLETE'),('stress_v2_baselines_20261110',33,'COMPLETE')]
scales=['1000000','5000000','full'];models=['own_made','own_resmade','official_resmade'];seeds=[0,1,2]
expected_names={
 'matched_sampling_20261110':{f'{s}_{m}_sample_seed{k}' for s in scales for m in models for k in seeds},
 'gpu_supplement_20261110':{f'size_width{w}_seed{k}' for w in [128,512] for k in seeds}|{f'budget_{s}_seed{k}_s{b}_mc{mc}' for s in scales for k in seeds for b in [100,500,2000,8000] for mc in seeds},
 'workload_eval_supplement_v2_20261110':{f'{s}_{m}_seed{k}_{scope}' for s in scales for m in models for k in seeds for scope in ['scale_shared','predicate_sweep','stress_v2']}}
for name,scopes in [('workload_baselines_20261110',['scale_shared','predicate_sweep','stress']),('stress_v2_baselines_20261110',['stress_v2'])]:
 expected_names[name]={f'{s}_{scope}_{method}_seed{k}' for s in scales for scope in scopes for method,k in [('exact_marginal_independence',None),('uniform_sample_10000',20261109)]+[(f'matched_{m}',k) for m in models for k in seeds]}
for name,expected,done in specs:
 root=E/name
 if not (root/'status.json').exists():pending.append(name);continue
 state=load(root/'status.json');roots[name]={'status':state['status'],'expected_cells':expected,'audited_cells':0}
 names=[c['name'] for c in state.get('runs',[])];assert len(names)==len(set(names)) and set(names)<=expected_names[name]
 if state['status']==done:assert set(names)==expected_names[name]
 for cell in state.get('runs',[]):
  if cell['status']!='COMPLETE':continue
  folder=root/cell['name']
  if not all((folder/f).exists() for f in ['summary.json','per_query.csv']):pending.append(str(folder.relative_to(R)));continue
  s=load(folder/'summary.json');rows=list(csv.DictReader((folder/'per_query.csv').open(encoding='utf-8')))
  info={'experiment':name,'run':cell['name']}
  if name=='matched_sampling_20261110':
   scale=s['scale'];scope='main_native';info.update(model='uniform_matched_'+s['reference_model'],sample_seed=s['sample_seed'],reference_model=s['reference_model'],parameter_budget_bytes=s['parameter_budget_bytes'],sample_array_bytes=s['sample_array_bytes'],sample_rows=s['sample_rows'])
   assert s['sample_array_bytes']==s['sample_rows']*44<=s['parameter_budget_bytes']
   assert 0<=s['parameter_budget_bytes']-s['sample_array_bytes']<44
   # The historical raw summary's reference_checkpoint_bytes was mislabeled.
   main=load(E/'main_matrix_20261109/status.json')['runs'];c=next(c for c in main if c['scale']==scale and c['model']==s['reference_model'] and c['seed']==0)
   reference_cfg=load(local(c['path'])/'config.json');assert s['parameter_budget_bytes']==reference_cfg['parameter_bytes']
   assert cell['name']==f"{scale}_{s['reference_model']}_sample_seed{s['sample_seed']}" and s['sample_seed'] in seeds
   info['actual_reference_checkpoint_bytes']=load(local(c['path'])/'training_result.json')['checkpoint_bytes']
  elif name in ['workload_baselines_20261110','stress_v2_baselines_20261110']:
   scale=s['scale'];scope=s['scope'];info.update(model=s['method'],sample_seed=s['sample_seed'],parameter_budget_bytes=s['parameter_budget_bytes'],sample_array_bytes=s['sample_array_bytes'],sample_rows=s['sample_rows'])
   assert cell['name']==f"{scale}_{scope}_{s['method']}_seed{s['sample_seed']}"
   if s['method']=='exact_marginal_independence':assert s['sample_seed'] is None and s['sample_rows']==s['sample_array_bytes']==s['parameter_budget_bytes']==0
   elif s['method']=='uniform_sample_10000':assert s['sample_seed']==20261109 and s['sample_rows']==10000 and s['sample_array_bytes']==s['parameter_budget_bytes']==440000
   else:
    reference=s['method'].removeprefix('matched_');assert reference in models and s['sample_seed'] in seeds
    main=load(E/'main_matrix_20261109/status.json')['runs'];c=next(c for c in main if c['scale']==scale and c['model']==reference and c['seed']==0);cfg=load(local(c['path'])/'config.json')
    assert s['parameter_budget_bytes']==cfg['parameter_bytes'] and s['sample_array_bytes']==s['sample_rows']*44
    assert 0<=s['parameter_budget_bytes']-s['sample_array_bytes']<44
  else:
   st=load(folder/'status.json');args=st['arguments'];cfg=load(local(args['run'])/'config.json');scale=cfg['scale'];scope=args['scope'];info.update(model=cfg['model'],training_seed=cfg['seed'],width=cfg['width'],sampling_budget=args['samples'],mc_seed=args['mc_seed'])
   if name=='gpu_supplement_20261110':
    expected_name=f"size_width{cfg['width']}_seed{cfg['seed']}" if cell['name'].startswith('size_') else f"budget_{scale}_seed{cfg['seed']}_s{args['samples']}_mc{args['mc_seed']}"
    assert cell['name']==expected_name and cfg['model']=='own_resmade'
   else:assert cell['name']==f"{scale}_{cfg['model']}_seed{cfg['seed']}_{scope}" and args['samples']==2000 and args['mc_seed']==0
  info.update(scale=scale,scope=scope,canonical_workload=scope!='stress');qs=queries(scale,scope)
  if len(rows)!=len(qs):pending.append(str(folder.relative_to(R))+' incomplete CSV');continue
  seen=set()
  for row in rows:
   qid=row['query_id'];assert qid not in seen;seen.add(qid);truth=float(row['truth']);est=float(row['estimate']);err=float(row['qerror']);lat=float(row['latency_ms'])
   assert truth==qs[qid]['truth'] and truth>=0 and est>=0 and math.isfinite(est) and lat>=0 and math.isfinite(lat)
   ratio=max(max(1,truth)/max(1,est),max(1,est)/max(1,truth));assert math.isclose(ratio,err,rel_tol=1e-10)
   if 'method' in row and name!='matched_sampling_20261110':assert row['method']==info['model']
  m=metrics(rows);summary=s['methods'][info['model']] if 'methods' in s else s
  if name=='matched_sampling_20261110':
   key=scale,s['sample_rows'],s['sample_seed'];estimates={r['query_id']:float(r['estimate']) for r in rows}
   if key in matched_draws:assert matched_draws[key]==estimates
   else:matched_draws[key]=estimates
  for k in ['qerror_median','qerror_p95','qerror_p99','qerror_max','latency_median_ms']:assert math.isclose(m[k],summary[k],rel_tol=1e-9,abs_tol=1e-9),(folder,k)
  allstats.append(dict(info,**m));rawcount+=len(rows);roots[name]['audited_cells']+=1
  if scope!='main_native':
   buckets={}
   for row in rows:
    q=qs[row['query_id']];sel=q['truth']/({'1000000':1000000,'5000000':5000000,'full':12540479}[scale]);sb='zero' if not sel else '<=1e-5' if sel<=1e-5 else '<=1e-3' if sel<=1e-3 else '<=1e-2' if sel<=1e-2 else '>1e-2'
    for grouping,value in [('predicate_count',str(len(q['predicates']))),('selectivity',sb)]:buckets.setdefault((grouping,value),[]).append(row)
    if scope in ['stress','stress_v2']:buckets.setdefault(('stress_group',str(q.get('stress_group',q.get('group','unlabeled')))),[]).append(row)
   grouped.extend(dict(info,grouping=k[0],group=k[1],**metrics(v)) for k,v in buckets.items())
  if name=='gpu_supplement_20261110' and cell['name'].startswith('budget_') and info['sampling_budget']==2000 and info['mc_seed']==0:
   mainfolder=E/'main_evaluation_20261110'/f"{scale}_own_resmade_seed{info['training_seed']}";old={r['query_id']:float(r['estimate']) for r in csv.DictReader((mainfolder/'per_query.csv').open()) if r['method']=='own_resmade'}
   delta=max(abs(float(r['estimate'])-old[r['query_id']])/max(1,old[r['query_id']]) for r in rows)
   assert delta<1e-5,delta;cross.append(dict(scale=scale,seed=info['training_seed'],max_relative_estimate_difference=delta))
 if state['status']!=done or roots[name]['audited_cells']!=expected:pending.append(name+' not fully audited')
 if name=='gpu_supplement_20261110' and roots[name]['audited_cells']==114:assert {(r['scale'],r['seed']) for r in cross}=={(s,k) for s in scales for k in seeds} and len(cross)==9
size=E/'size_supplement_20261110';training=[];curves=[]
if (size/'status.json').exists():
 st=load(size/'status.json');roots[size.name]={'status':st['status'],'expected_cells':6,'audited_cells':0}
 for cell in st['runs']:
  f=size/cell['name']
  if cell['status']!='COMPLETE' or not all((f/x).exists() for x in ['status.json','config.json','training_result.json','epoch_metrics.jsonl']):continue
  cfg=load(f/'config.json');status=load(f/'status.json');res=load(f/'training_result.json');epochs=list(map(json.loads,(f/'epoch_metrics.jsonl').read_text().splitlines()))
  assert status['status']=='COMPLETE' and status['exit_code']==0 and len(epochs)==20 and [x['epoch'] for x in epochs]==list(range(1,21))
  assert cfg['scale']=='1000000' and cfg['model']=='own_resmade' and cfg['width'] in [128,512] and cfg['seed'] in [0,1,2]
  assert cell['name']==f"width{cfg['width']}_seed{cfg['seed']}" and cfg['layers']==5 and cfg['batch']==2048 and cfg['warmups']==8000 and cfg['epochs']==20
  assert res['training_epochs']==20 and res['training_steps']==9780 and res['rows']==1000000
  assert all(math.isfinite(e['mean_training_nll_nats']) for e in epochs)
  base=dict(scale=cfg['scale'],model=cfg['model'],width=cfg['width'],seed=cfg['seed']);training.append(dict(base,elapsed_seconds=status['elapsed_seconds'],parameter_bytes=cfg['parameter_bytes'],**res));curves.extend(dict(base,**e) for e in epochs);roots[size.name]['audited_cells']+=1
 if st['status']!='COMPLETE' or len(training)!=6:pending.append(size.name+' not fully audited')
 else:assert {(x['width'],x['seed']) for x in training}=={(w,k) for w in [128,512] for k in seeds}
else:pending.append(size.name)
config_groups={}
keys=['experiment','scale','scope','model','width','sampling_budget','reference_model','parameter_budget_bytes','sample_rows']
for row in allstats:config_groups.setdefault(tuple(row.get(k,'') for k in keys),[]).append(row)
ranges=[]
for key,rows in config_groups.items():
 row=dict(zip(keys,key));row.update(runs=len(rows),training_seed_count=len({r['training_seed'] for r in rows if 'training_seed' in r}),mc_seed_count=len({r['mc_seed'] for r in rows if 'mc_seed' in r}),sample_seed_count=len({r['sample_seed'] for r in rows if r.get('sample_seed') is not None}))
 for k in ['qerror_median','qerror_p95','qerror_p99','qerror_max','qerror_positive_median','qerror_positive_p95','qerror_positive_p99','qerror_positive_max','latency_median_ms']:
  v=[r[k] for r in rows];row.update({k+'_run_median':statistics.median(v),k+'_run_min':min(v),k+'_run_max':max(v)})
 ranges.append(row)
write('metrics_by_run.csv',allstats);write('metrics_configuration_ranges.csv',ranges);write('grouped_metrics.csv',grouped);write('size_training_summary.csv',training);write('size_training_curves.csv',curves);write('main_sampling_rechecks.csv',cross)
report={'status':'PASS_PARTIAL' if pending else 'PASS_NUMERICAL_SUPPLEMENTS','checked_unix':time.time(),'raw_csv_rows_checked':rawcount,'roots':roots,'pending':pending,'main_sampling_rechecks':len(cross),'matched_unique_draws':len(matched_draws),'matched_duplicate_policy':'Equal row budgets and sample seeds reproduce identical samples; reference model labels are not independent samples. Three sample seeds per unique budget.','zero_reporting':'qerror_* is all-query floor1 diagnostic; qerror_positive_* excludes truth zero; zero estimate counts and distribution are separate. stress is historical v1; stress_v2 canonical.','latency_correction':'Model timing excludes host numpy mask construction; includes host-mask to GPU conversion/transfer and synchronized inference. Raw older wording includes host masks was ambiguous.','ranges_policy':'Median and min-max of per-run query quantiles; descriptive grid, not a confidence interval or pooled-query quantile. Training and MC seed counts retained.','historical_field_correction':'matched raw reference_checkpoint_bytes is parameter budget; corrected analysis reports actual_reference_checkpoint_bytes from training receipt','scope':'numerical supplementary matrices only; PG and upstream compatibility have separate independent receipts'}
report['derived_table_rows']=table_counts
audit_path.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
if pending and not a.partial:raise SystemExit(2)
