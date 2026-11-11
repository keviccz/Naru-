"""Independent audit and plot-ready metrics for paired rare-positive/joint-zero tests."""
import csv,json,math,operator,statistics,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];E=R/'experiments';ROOT=E/'joint_zero_model_eval_20261111';OUT=ROOT/'analysis'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def quantile(v,q):
 v=sorted(v);x=(len(v)-1)*q;i=int(x);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)
def summarize(rows,queries):
 assert len(rows)==200 and len({r['query_id'] for r in rows})==200
 pairs={};zero=[];pos=[];error=[]
 for row in rows:
  q=queries[row['query_id']];truth=q['truth'];est=float(row['estimate']);assert truth==float(row['truth']) and est>=0 and math.isfinite(est)
  floor=max(max(1,truth)/max(1,est),max(1,est)/max(1,truth))
  field=next((k for k in ['qerror','q_error_floor1','qerror_floor1'] if k in row),None)
  if field is not None:assert math.isclose(floor,float(row[field]),rel_tol=1e-9)
  if 'qerror_positive' in row:
   if truth:assert math.isclose(floor,float(row['qerror_positive']),rel_tol=1e-9)
   else:assert row['qerror_positive']==''
  group='zero' if truth==0 else 'near_positive';pairs.setdefault(q['pair_id'],{})[group]=(q,est)
  if truth==0:zero.append(est)
  else:pos.append(est);error.append(floor)
 assert len(zero)==len(pos)==len(pairs)==100 and all(set(v)=={'zero','near_positive'} for v in pairs.values())
 pair_rows=[]
 for key,pair in pairs.items():
  z=pair['zero'][1];p=pair['near_positive'][1];truth=pair['near_positive'][0]['truth']
  pair_rows.append(dict(pair_id=key,zero_query_id=pair['zero'][0]['query_id'],positive_query_id=pair['near_positive'][0]['query_id'],positive_truth=truth,zero_estimate=z,positive_estimate=p,positive_greater=int(p>z),tie=int(p==z)))
 return dict(positive_queries=100,zero_queries=100,positive_qerror_median=statistics.median(error),positive_qerror_p95=quantile(error,.95),positive_qerror_p99=quantile(error,.99),positive_qerror_max=max(error),positive_estimate_eq0=sum(x==0 for x in pos),positive_estimate_lt1=sum(x<1 for x in pos),zero_estimate_median=statistics.median(zero),zero_estimate_mean=statistics.mean(zero),zero_estimate_p95=quantile(zero,.95),zero_estimate_max=max(zero),zero_estimate_gt1=sum(x>1 for x in zero),zero_estimate_gt10=sum(x>10 for x in zero),zero_estimate_gt100=sum(x>100 for x in zero),pair_strict_separation_fraction=sum(x['positive_greater'] for x in pair_rows)/100,pair_tie_fraction=sum(x['tie'] for x in pair_rows)/100,pair_half_tie_score=sum(x['positive_greater']+.5*x['tie'] for x in pair_rows)/100),pair_rows
def write(name,rows):
 with (OUT/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def main():
 OUT.mkdir(exist_ok=True);audit=R/'evidence/joint_model_audit.json';audit.write_text(json.dumps({'status':'RUNNING','started_unix':time.time()}))
 def fail(typ,value,tb):
  audit.write_text(json.dumps({'status':'FAILED','error':str(value),'error_type':typ.__name__}));sys.__excepthook__(typ,value,tb)
 sys.excepthook=fail
 state=read(ROOT/'status.json');assert state['status']=='COMPLETE' and len(state['runs'])==81
 expected={f'{s}_{m}_seed{k}_mc{mc}' for s in ['1000000','5000000','full'] for m in ['own_made','own_resmade','official_resmade'] for k in [0,1,2] for mc in [0,1,2]}
 names=[c['name'] for c in state['runs']];assert len(set(names))==81 and set(names)==expected
 queries={};coverage=[];all_metrics=[];all_pairs=[];raw_count=0;warmups=[]
 for scale in ['1000000','5000000','full']:
  path=E/'joint_zero_20261111/workloads'/scale/'joint_zero_nearzero.jsonl';qs=[json.loads(x) for x in path.read_text().splitlines()];assert len(qs)==200 and len({q['query_id'] for q in qs})==200
  dom=read(R/'data/snapshot_resume_20261109_172915/encoded'/(scale+'_domains.json'));domains=dom['domains'];pairs={};signatures=set()
  for q in qs:
   assert q['workload']=='joint_zero_nearzero' and (q['truth']==0 if q['group']=='zero' else 1<=q['truth']<=10)
   signature=tuple((p['column_index'],p['operator'],p['value']) for p in sorted(q['predicates'],key=lambda p:p['column_index']));assert signature not in signatures;signatures.add(signature)
   pairs.setdefault(q['pair_id'],{})[q['group']]=q
   for p in q['predicates']:
    assert p['column']==dom['columns'][p['column_index']] and p['value'] is not None
    fn={'=':operator.eq,'<':operator.lt,'<=':operator.le,'>':operator.gt,'>=':operator.ge}[p['operator']]
    actual=[i for i,v in enumerate(domains[p['column_index']]) if v is not None and fn(v,p['value'])];assert actual==p['allowed_codes'] and actual
  assert len(pairs)==100 and all(set(v)=={'zero','near_positive'} for v in pairs.values())
  for pair in pairs.values():
   z=sorted(pair['zero']['predicates'],key=lambda p:p['column_index']);p=sorted(pair['near_positive']['predicates'],key=lambda p:p['column_index'])
   assert [(x['column_index'],x['operator']) for x in z]==[(x['column_index'],x['operator']) for x in p]
   assert sum(x['value']!=y['value'] for x,y in zip(z,p))==1
  queries[scale]={q['query_id']:q for q in qs};coverage.append(dict(scale=scale,records=200,unique_semantic_queries=len(signatures),pairs=100,zero=100,near_positive=100))
 for c in state['runs']:
  f=ROOT/c['name'];status=read(f/'status.json');args=status['arguments'];cfg=read(R/args['run'].split('DSC5003_Naru_20261109/',1)[1]/'config.json');scale=cfg['scale']
  assert status['status']=='COMPLETE' and status['queries_completed']==200 and c['exit_code']==0 and c['status']=='COMPLETE'
  assert args['samples']==2000 and args['mc_seed'] in [0,1,2] and args['scope']=='joint_zero_nearzero' and c['name']==f"{scale}_{cfg['model']}_seed{cfg['seed']}_mc{args['mc_seed']}"
  rows=list(csv.DictReader((f/'per_query.csv').open()));assert {r['query_id'] for r in rows}==set(queries[scale]) and all(r['method']==cfg['model'] for r in rows)
  summary,pairs=summarize(rows,queries[scale]);meta=dict(run=c['name'],scale=scale,method=cfg['model'],training_seed=cfg['seed'],mc_seed=args['mc_seed'],sampling_budget=2000)
  latency=[float(r['latency_ms']) for r in rows];assert all(math.isfinite(x) and x>=0 for x in latency)
  all_metrics.append(dict(meta,latency_median_ms=statistics.median(latency),**summary));all_pairs.extend(dict(meta,**p) for p in pairs);raw_count+=len(rows)
  warmups.append(read(f/'summary.json')['warmup_query_index'])
 assert set(warmups)=={0} and raw_count==16200
 groups={}
 for row in all_metrics:groups.setdefault((row['scale'],row['method']),[]).append(row)
 ranges=[]
 for key,rows in groups.items():
  assert {(x['training_seed'],x['mc_seed']) for x in rows}=={(s,m) for s in [0,1,2] for m in [0,1,2]}
  row=dict(scale=key[0],method=key[1],training_seeds=3,mc_seeds=3,runs=9)
  for metric in ['positive_qerror_median','positive_qerror_p95','zero_estimate_median','zero_estimate_p95','zero_estimate_gt1','zero_estimate_gt10','positive_estimate_eq0','pair_strict_separation_fraction','pair_half_tie_score','latency_median_ms']:
   v=[x[metric] for x in rows];row.update({metric+'_run_median':statistics.median(v),metric+'_run_min':min(v),metric+'_run_max':max(v)})
  ranges.append(row)
 sanity=[]
 for scale,qs in queries.items():
  rows=[dict(query_id=qid,truth=q['truth'],estimate=0) for qid,q in qs.items()];m,_=summarize(rows,qs);sanity.append(dict(scale=scale,method='constant_zero_sanity',**m))
 write('metrics_by_run.csv',all_metrics);write('metrics_configuration_ranges.csv',ranges);write('paired_estimates.csv',all_pairs);write('coverage.csv',coverage);write('constant_zero_sanity.csv',sanity)
 report={'status':'PASS','model_runs':81,'raw_model_rows_checked':raw_count,'frozen_queries':600,'pairs':300,'paired_estimate_rows':len(all_pairs),'coverage':coverage,'warmup_indices':sorted(set(warmups)),'selection_scope':'Directed paired stress sample selected by exact truth and fixed seeds; not unbiased random SQL benchmark','uncertainty':'3 training seeds x3 MC seeds; ranges descriptive, no confidence interval claim','zero_reporting':'Raw estimated rows and >1/>10/>100 thresholds separate; positive-only floor1 qerror plus missed-near-positive counts; constant-zero sanity is not a competitive model'}
 audit.write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
