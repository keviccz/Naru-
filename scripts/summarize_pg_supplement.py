"""Consistent repetition summaries plus independent main-thread spot audit."""
import csv,json,math,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'experiments/pg_supplement_20261110';O=R/'experiments/supplement_analysis_20261110';O.mkdir(exist_ok=True)
assert json.loads((P/'status.json').read_text())['status']=='COMPLETE'
validation=json.loads((R/'evidence/pg_validation.json').read_text());assert validation['status']=='PASS' and validation['queries']==118800 and validation['cells']==27
rows=list(csv.DictReader((P/'summary_metrics.csv').open(encoding='utf-8')));assert len(rows)==108
scales=['1000000','5000000','full'];tiers=['PG-default','PG-target1000','PG-extstats'];scopes=['main_native','scale_shared','predicate_sweep','stress_v2']
assert {(r['scale'],r['tier'],int(r['repetition']),r['scope']) for r in rows}=={(s,t,k,w) for s in scales for t in tiers for k in [1,2,3] for w in scopes}
groups={}
for row in rows:groups.setdefault((row['scale'],row['tier'],row['scope']),[]).append(row)
out=[]
for key,v in groups.items():
 r=dict(scale=key[0],method=key[1],scope=key[2],analyze_repetitions=3,queries=int(v[0]['queries']),positive_truth_queries=int(v[0]['nonzero_truth_queries']),zero_truth_queries=int(v[0]['zero_truth_queries']))
 for k in ['qerror_median','qerror_p95','qerror_p99','qerror_max','mean_planning_ms','mean_client_wall_ms','analyze_seconds','single_column_catalog_row_bytes','extended_catalog_row_bytes']:
  x=[float(z[k]) for z in v];r.update({k+'_repeat_median':statistics.median(x),k+'_repeat_min':min(x),k+'_repeat_max':max(x)})
 out.append(r)
with (O/'pg_metrics_configuration_ranges.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
checked=0
# The PG task audited every raw plan. The main thread independently checks
# one extended-statistics repetition at all scales, including repaired NULLs.
for scale in scales:
 for scope in ['main_native','stress_v2']:
  qfile=R/'data/snapshot_resume_20261109_172915/workloads'/scale/'main_native.jsonl' if scope=='main_native' else R/'experiments/workload_supplement_v2_20261110'/scale/'stress_v2.jsonl'
  qs={q['query_id']:q for q in map(json.loads,qfile.read_text().splitlines())};folder=P/scale/'PG-extstats/repeat1'
  metrics=list(csv.DictReader((folder/(scope+'_metrics.csv')).open()));plans=[json.loads(x) for x in (folder/(scope+'_plans.jsonl')).read_text().splitlines()];assert len(metrics)==len(plans)==len(qs)
  assert len({r['query_id'] for r in metrics})==len(qs)
  for row,raw in zip(metrics,plans):
   qid=row['query_id'];q=qs[qid];truth=float(row['truth']);estimate=float(row['estimate']);plan=raw['explain']['Plan'];assert raw['query_id']==qid and truth==q['truth'] and plan['Plan Rows']==estimate
   assert plan['Node Type'] in ['Seq Scan','Result'] and row['node']==plan['Node Type']
   expected=max(max(1,truth)/max(1,estimate),max(1,estimate)/max(1,truth));assert math.isclose(expected,float(row['q_error_floor1']),rel_tol=1e-10)
   if truth>0:assert math.isclose(expected,float(row['q_error']),rel_tol=1e-10)
   else:assert row['q_error']==''
   assert all(math.isfinite(float(row[k])) and float(row[k])>=0 for k in ['planning_ms','client_wall_ms']);checked+=1
report={'status':'PASS','delegated_full_plan_audit_rows':validation['queries'],'independent_main_thread_spot_rows':checked,'summary_scope_cells':len(rows),'configuration_summary_rows':len(out),'aggregation':'Median/min-max of three per-ANALYZE quantiles; ANALYZE repeats are not neural training seeds; true-zero queries excluded from qerror quantiles and reported separately','latency_boundary':'Server planning and client roundtrip, no SQL execution; not directly same timing boundary as synchronized GPU inference','storage_boundary':'Catalog logical row bytes; excludes heap data, shared catalog indexes/pages, WAL, executables; not physical synopsis attribution'}
(R/'evidence/pg_main_thread_review.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
