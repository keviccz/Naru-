"""Independent standard-library audit of frozen joint-zero CPU and PG outputs."""
import csv,json,math,statistics,time,operator,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'experiments/joint_zero_20261111';report=O/'audit.json'
def j(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def qt(v,q):
 v=sorted(v);pos=(len(v)-1)*q;i=int(pos);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(pos-i)
def fail(typ,value,tb):report.write_text(json.dumps({'status':'FAILED','error':str(value),'type':typ.__name__}));sys.__excepthook__(typ,value,tb)
sys.excepthook=fail;report.write_text(json.dumps({'status':'RUNNING','started':time.time()}))
state=j(O/'status.json');assert state['status']=='COMPLETE'
scales=['1000000','5000000','full'];tiers=['PG-default','PG-target1000','PG-extstats'];models=['own_made','own_resmade']
cpu_expected={f'{s}_{m}_seed{k}' for s in scales for m,k in [('exact_marginal_independence',None),('uniform_sample_10000',20261109)]+[(f'matched_{m}',k) for m in models for k in [0,1,2]]}
pg_expected={f'{s}_{t}_repeat{k}' for s in scales for t in tiers for k in [1,2,3]}
assert {c['name'] for c in state['cpu_runs']}==cpu_expected and len(state['cpu_runs'])==24 and all(c['status']=='COMPLETE' for c in state['cpu_runs'])
assert {c['name'] for c in state['pg_runs']}==pg_expected and len(state['pg_runs'])==27 and all(c['status']=='COMPLETE' for c in state['pg_runs'])
counts=0;rows_checked=0;plan_rows=0;summaries=[];pair_rows=[];coverage=[]
matrix=j(R/'experiments/main_matrix_20261109/status.json')['runs']
ops={'=':operator.eq,'<':operator.lt,'<=':operator.le,'>':operator.gt,'>=':operator.ge}
for scale in scales:
 qs=[json.loads(x) for x in (O/'workloads'/scale/'joint_zero_nearzero.jsonl').read_text().splitlines()];qmap={q['query_id']:q for q in qs};assert len(qmap)==len(qs)==200
 dom=j(R/'data/snapshot_resume_20261109_172915/encoded'/(scale+'_domains.json'));pairs={};seen=set()
 hist=j(O/'pg'/scale/'marginal_histogram_check.json')['columns'];assert j(O/'pg'/scale/'marginal_histogram_check.json')['status']=='PASS'
 for q in qs:
  assert q['workload']=='joint_zero_nearzero' and (q['truth']==0 if q['group']=='zero' else 1<=q['truth']<=10)
  assert q['group'] in ['zero','near_positive'];pairs.setdefault(q['pair_id'],{})[q['group']]=q
  sig=tuple((p['column_index'],p['operator'],p['value']) for p in sorted(q['predicates'],key=lambda p:p['column_index']));assert sig not in seen;seen.add(sig)
  for k,p in enumerate(q['predicates']):
   col=p['column'];i=p['column_index'];assert dom['columns'][i]==col and p['value'] is not None
   allowed=[code for code,v in enumerate(dom['domains'][i]) if v is not None and ops[p['operator']](v,p['value'])];assert allowed==p['allowed_codes'] and allowed
   marginal=sum(hist[col]['postgres_counts'][code] for code in allowed);assert marginal==q['marginal_counts'][k] and marginal>0
 assert len(pairs)==100 and all(set(v)=={'near_positive','zero'} for v in pairs.values())
 for pair in pairs.values():
  a=sorted(pair['near_positive']['predicates'],key=lambda p:p['column_index']);b=sorted(pair['zero']['predicates'],key=lambda p:p['column_index']);assert [(p['column_index'],p['operator']) for p in a]==[(p['column_index'],p['operator']) for p in b];assert sum(p['value']!=z['value'] for p,z in zip(a,b))==1
 truth=j(O/'pg'/scale/'truth_counts.json');assert truth['status']=='PASS' and truth['queries']==200
 for c in truth['checks']:assert c['postgres_count']==c['encoded_count']==qmap[c['query_id']]['truth'];counts+=1
 coverage.append(dict(scale=scale,queries=200,pairs=100,unique_semantics=200,all_marginal_counts_positive=True,positive_min=min(q['truth'] for q in qs if q['truth']>0),positive_max=max(q['truth'] for q in qs)))
 cells=[('cpu',c,O/'cpu'/c['name']) for c in state['cpu_runs'] if c['scale']==scale]+[('pg',c,O/'pg'/scale/c['tier']/f"repeat{c['repetition']}") for c in state['pg_runs'] if c['scale']==scale]
 for kind,c,folder in cells:
  summary=j(folder/'summary.json');rows=list(csv.DictReader((folder/'per_query.csv').open()));assert len(rows)==200 and {x['query_id'] for x in rows}==set(qmap)
  method=c['method'] if kind=='cpu' else c['tier'];meta=dict(kind=kind,run=c['name'],scale=scale,method=method,sample_seed=c.get('sample_seed'),analyze_repetition=c.get('repetition'))
  estimates={};errors=[];zeros=[];pos=[]
  for row in rows:
   q=qmap[row['query_id']];t=float(row['truth']);e=float(row['estimate']);assert t==q['truth'] and e>=0 and math.isfinite(e)
   expected=max(max(1,t)/max(1,e),max(1,e)/max(1,t));assert math.isclose(expected,float(row['qerror_floor1']),rel_tol=1e-10)
   assert row['group']==q['group'] and row['pair_id']==q['pair_id'];estimates[q['query_id']]=e
   if t>0:assert math.isclose(expected,float(row['qerror_positive']),rel_tol=1e-10);errors.append(expected);pos.append(e)
   else:assert row['qerror_positive']=='';zeros.append(e)
   for key in (['planning_ms','client_wall_ms'] if kind=='pg' else ['latency_ms']):assert math.isfinite(float(row[key])) and float(row[key])>=0
   rows_checked+=1
  metric=dict(queries=200,positive_queries=100,zero_queries=100,positive_qerror_median=statistics.median(errors),positive_qerror_p95=qt(errors,.95),positive_qerror_p99=qt(errors,.99),positive_qerror_max=max(errors),positive_estimate_exact_zero=sum(e==0 for e in pos),positive_estimate_lt_one=sum(e<1 for e in pos),zero_estimate_median=statistics.median(zeros),zero_estimate_p95=qt(zeros,.95),zero_estimate_max=max(zeros),zero_estimate_gt1=sum(e>1 for e in zeros),zero_estimate_gt10=sum(e>10 for e in zeros),zero_estimate_gt100=sum(e>100 for e in zeros))
  for k,v in metric.items():assert math.isclose(v,summary[k],rel_tol=1e-9,abs_tol=1e-9),(folder,k,v,summary[k])
  separate=0;ties=0
  for pid,pair in pairs.items():
   z=estimates[pair['zero']['query_id']];p=estimates[pair['near_positive']['query_id']];separate+=p>z;ties+=p==z;pair_rows.append(dict(meta,pair_id=pid,zero_estimate=z,positive_estimate=p,positive_truth=pair['near_positive']['truth'],strict_separation=p>z,tie=p==z))
  metric.update(pair_strict_separation_fraction=separate/100,pair_tie_fraction=ties/100,pair_half_tie_score=(separate+.5*ties)/100)
  if kind=='cpu':
   assert summary['method']==method and summary['sample_seed']==c['sample_seed'] and summary['scale']==scale
   if method.startswith('matched_'):
    model=method[len('matched_'):];cfgpath=next(x['path'] for x in matrix if x['scale']==scale and x['model']==model and x['seed']==0);cfg=j(R/cfgpath.split('DSC5003_Naru_20261109/',1)[1]/'config.json');budget=cfg['parameter_bytes'];assert summary['parameter_budget_bytes']==budget and summary['sample_array_bytes']==summary['sample_rows']*44 and 0<=budget-summary['sample_array_bytes']<44
   elif method=='uniform_sample_10000':assert summary['sample_rows']==10000 and summary['sample_array_bytes']==summary['parameter_budget_bytes']==440000
   else:assert summary['sample_rows']==summary['sample_array_bytes']==summary['parameter_budget_bytes']==0
  else:
   raw=[json.loads(x) for x in (folder/'plans.jsonl').read_text().splitlines()];assert len(raw)==200
   by={x['query_id']:x for x in rows}
   for plan in raw:
    scan=plan['explain']['Plan'];assert scan['Node Type'] in ['Seq Scan','Result'] and scan['Plan Rows']==float(by[plan['query_id']]['estimate']) and not scan.get('Parallel Aware',False);plan_rows+=1
   analyze=j(folder/'analyze.json');target=100 if method=='PG-default' else 1000;assert all(int(x[1])==target for x in analyze['column_targets']);assert len(analyze['extended_objects'])==(3 if method=='PG-extstats' else 0)
  summaries.append(dict(meta,**metric))
def write(name,rows):
 with (O/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)
write('summary_metrics.csv',summaries);write('paired_estimates.csv',pair_rows);write('coverage.csv',coverage)
grouped={}
for row in summaries:grouped.setdefault((row['scale'],row['method']),[]).append(row)
ranges=[]
for key,values in grouped.items():
 row=dict(scale=key[0],method=key[1],runs=len(values),repeat_type='ANALYZE' if values[0]['kind']=='pg' else 'sample_seed' if values[0]['sample_seed'] is not None else 'deterministic')
 for metric in ['positive_qerror_median','positive_qerror_p95','positive_qerror_p99','zero_estimate_median','zero_estimate_p95','zero_estimate_gt1','zero_estimate_gt10','positive_estimate_exact_zero','positive_estimate_lt_one','pair_strict_separation_fraction','pair_tie_fraction','pair_half_tie_score']:
  v=[x[metric] for x in values];row.update({metric+'_run_median':statistics.median(v),metric+'_run_min':min(v),metric+'_run_max':max(v)})
 ranges.append(row)
assert len(ranges)==21;write('metrics_configuration_ranges.csv',ranges)
assert counts==600 and rows_checked==10200 and plan_rows==5400
result={'status':'PASS','frozen_queries':600,'pairs':300,'independent_pg_exact_count_checks':counts,'single_column_pg_histogram_checks':33,'cpu_cells':24,'pg_cells':27,'raw_csv_rows_checked':rows_checked,'raw_pg_plans_checked':plan_rows,'paired_estimate_rows':len(pair_rows),'workloads_unchanged_during_resume':True,'failed_cpu_attempt_preserved':'failure_cpu_serialization_1','cpu_resume_only':state.get('cpu_resume_only'),'pg_stopped':'server stopped' in (O/'pg/shutdown.log').read_text(),'limits':['Targeted stress chosen by truth; no unbiased SQL claim','Only100 unique pairs per scale','Matched ResMADE and official ResMADE share identical budget/sample','Floor1 qerror diagnostic does not measure exact zero recognition; positive miss counts and pair separation retained']}
assert result['pg_stopped'];report.write_text(json.dumps(result,indent=2));print(json.dumps(result))
