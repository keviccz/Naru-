"""Coverage and fixed-common-query summaries, using frozen truths only."""
import csv,json,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[1];E=R/'experiments';O=E/'supplement_analysis_20261110'
scales=['1000000','5000000','full'];coverage=[];queries={};shared={}
def signature(q):return json.dumps([(p['column'],p['operator'],p['value']) for p in sorted(q['predicates'],key=lambda p:p['column_index'])],ensure_ascii=False)
def write(name,rows):
 with (O/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def qt(v,q):
 v=sorted(v);x=(len(v)-1)*q;i=int(x);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)
def metrics(rows):
 v=[float(r.get('qerror',r.get('q_error'))) for r in rows]
 return dict(queries=len(v),qerror_median=statistics.median(v),qerror_p95=qt(v,.95),qerror_p99=qt(v,.99),qerror_max=max(v))
for scale in scales:
 for scope in ['scale_shared','predicate_sweep','stress_v2']:
  root='workload_supplement_v2_20261110' if scope=='stress_v2' else 'workload_supplement_20261110'
  qs=[json.loads(x) for x in (E/root/scale/(scope+'.jsonl')).read_text().splitlines()];queries[scale,scope]={q['query_id']:q for q in qs}
  zeros=[q for q in qs if q['truth']==0];empty=sum(any(not p['allowed_codes'] for p in q['predicates']) for q in zeros)
  coverage.append(dict(scale=scale,scope=scope,queries=len(qs),unique_semantic_queries=len(set(map(signature,qs))),positive_truth_queries=len(qs)-len(zeros),zero_truth_queries=len(zeros),zero_truth_with_empty_predicate_domain=empty,zero_truth_with_nonempty_predicate_domains=len(zeros)-empty))
 shared[scale]=queries[scale,'scale_shared']
assert all(set(shared[s])==set(shared['full']) for s in scales)
for qid in shared['full']:
 assert len({signature(shared[s][qid]) for s in scales})==1
 assert shared['1000000'][qid]['truth']<=shared['5000000'][qid]['truth']<=shared['full'][qid]['truth']
common={qid for qid in shared['full'] if all(shared[s][qid]['truth']>0 for s in scales)};assert len(common)==958
common_unique=len({signature(shared['full'][qid]) for qid in common});assert common_unique==956
(O/'scale_shared_common_positive_query_ids.json').write_text(json.dumps({'query_records':len(common),'unique_semantic_queries':common_unique,'selection':'Frozen truths positive at all three nested scales; independent of method estimates/errors','query_ids':sorted(common)},indent=2))
common_metrics=[];zero_records=[]
def process(rows,meta,scale,scope):
 if scope=='scale_shared':
  selected=[r for r in rows if r['query_id'] in common];assert len(selected)==958;common_metrics.append(dict(meta,scale=scale,scope='scale_shared_common_positive',**metrics(selected)))
 for r in rows:
  q=queries[scale,scope][r['query_id']]
  if q['truth']==0 and all(p['allowed_codes'] for p in q['predicates']):zero_records.append(dict(meta,scale=scale,query_id=q['query_id'],truth=0,estimate=float(r['estimate']),predicates=json.dumps(q['predicates'],ensure_ascii=False)))
for root in ['workload_eval_supplement_v2_20261110','workload_baselines_20261110']:
 st=json.loads((E/root/'status.json').read_text());assert st['status']=='COMPLETE'
 for c in st['runs']:
  folder=E/root/c['name'];summary=json.loads((folder/'summary.json').read_text())
  if root=='workload_eval_supplement_v2_20261110':
   args=json.loads((folder/'status.json').read_text())['arguments'];cfg=R/args['run'].split('DSC5003_Naru_20261109/',1)[1]/'config.json';cfg=json.loads(cfg.read_text());scope=args['scope'];scale=cfg['scale'];meta=dict(experiment=root,run=c['name'],method=cfg['model'],training_seed=cfg['seed'])
  else:scope=summary['scope'];scale=summary['scale'];meta=dict(experiment=root,run=c['name'],method=summary['method'],sample_seed=summary['sample_seed'])
  if scope not in ['scale_shared','predicate_sweep']:continue
  process(list(csv.DictReader((folder/'per_query.csv').open())),meta,scale,scope)
for scale in scales:
 for tier in ['PG-default','PG-target1000','PG-extstats']:
  for rep in [1,2,3]:
   for scope in ['scale_shared','predicate_sweep']:
    folder=E/'pg_supplement_20261110'/scale/tier/f'repeat{rep}';meta=dict(experiment='pg_supplement_20261110',run=f'{scale}_{tier}_repeat{rep}',method=tier,analyze_repetition=rep)
    process(list(csv.DictReader((folder/(scope+'_metrics.csv')).open())),meta,scale,scope)
warmups=[]
for f in (E/'workload_eval_supplement_v2_20261110').glob('*stress_v2/summary.json'):
 warmups.append(json.loads(f.read_text())['warmup_query_index'])
assert len(warmups)==27 and set(warmups)=={100}
write('workload_coverage.csv',coverage);write('scale_shared_common_positive.csv',common_metrics);write('nontrivial_zero_estimates.csv',zero_records)
report={'status':'PASS','fixed_common_positive_query_records':958,'fixed_common_positive_unique_semantics':956,'fixed_common_configuration_rows':len(common_metrics),'nonempty_domain_joint_zero_estimate_rows':len(zero_records),'stress_model_warmup_checks':27,'coverage':coverage,'interpretation':'stress_v2 zeros are all detected from empty allowed domains, so this is conformance coverage, not learned joint-zero generalization; nontrivial zeros occur in smaller-scale shared workloads. Query signatures sort conjunctions by column; predicate order does not create a new semantic query.'}
(R/'evidence/supplement_workload_analysis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
