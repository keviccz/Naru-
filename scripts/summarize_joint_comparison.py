"""Compare all frozen methods on the directed paired test; no fitting."""
import csv,json,math,statistics,time,sys
from pathlib import Path
from audit_joint_model_eval import summarize
R=Path(__file__).resolve().parents[1];E=R/'experiments';B=E/'joint_zero_20261111';M=E/'joint_zero_model_eval_20261111';O=M/'analysis'
audit_path=R/'evidence/joint_comparison_audit.json';audit_path.write_text(json.dumps({'status':'RUNNING','started_unix':time.time()}))
def fail(typ,value,tb):
 audit_path.write_text(json.dumps({'status':'FAILED','error':str(value),'error_type':typ.__name__,'derived_tables_valid':False}));sys.__excepthook__(typ,value,tb)
sys.excepthook=fail
def load(p):return json.loads(p.read_text())
def read(p):return list(csv.DictReader(p.open(encoding='utf-8')))
def write(name,rr):
 with (O/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rr for k in r)));w.writeheader();w.writerows(rr)
state=load(B/'status.json');assert state['status']=='COMPLETE' and load(R/'evidence/joint_model_audit.json')['status']=='PASS'
qs={s:{q['query_id']:q for q in map(json.loads,(B/'workloads'/s/'joint_zero_nearzero.jsonl').read_text().splitlines())} for s in ['1000000','5000000','full']}
allmetrics=read(O/'metrics_by_run.csv');pair_rows=read(O/'paired_estimates.csv');rowcount=0
expected_models={f'{s}_{m}_seed{k}_mc{mc}' for s in qs for m in ['own_made','own_resmade','official_resmade'] for k in [0,1,2] for mc in [0,1,2]}
assert len(allmetrics)==81 and {r['run'] for r in allmetrics}==expected_models
for r in allmetrics:
 assert r['run']==f"{r['scale']}_{r['method']}_seed{r['training_seed']}_mc{r['mc_seed']}" and int(r['positive_queries'])==int(r['zero_queries'])==100
assert len(pair_rows)==8100 and len({(r['run'],r['pair_id']) for r in pair_rows})==8100
for run in expected_models:assert sum(r['run']==run for r in pair_rows)==100
expected_cpu={f'{s}_{m}_seed{k}' for s in qs for m,k in [('exact_marginal_independence',None),('uniform_sample_10000',20261109)]+[(f'matched_{m}',k) for m in ['own_made','own_resmade'] for k in [0,1,2]]}
expected_pg={f'{s}_{t}_repeat{k}' for s in qs for t in ['PG-default','PG-target1000','PG-extstats'] for k in [1,2,3]}
assert len(state['cpu_runs'])==len(expected_cpu)==24 and {c['name'] for c in state['cpu_runs']}==expected_cpu
assert len(state['pg_runs'])==len(expected_pg)==27 and {c['name'] for c in state['pg_runs']}==expected_pg
main=load(E/'main_matrix_20261109/status.json')['runs']
for kind in ['cpu','pg']:
 for c in state[kind+'_runs']:
  assert c['status']=='COMPLETE'
  f=B/kind/c['name'] if kind=='cpu' else B/kind/c['scale']/c['tier']/f"repeat{c['repetition']}"
  s=load(f/'summary.json');scale=s['scale'];method=s['method'];rr=read(f/'per_query.csv')
  assert set(qs[scale])=={r['query_id'] for r in rr} and s['status']=='COMPLETE'
  metric,pairs=summarize(rr,qs[scale]);rowcount+=len(rr)
  mapping={'positive_estimate_exact_zero':'positive_estimate_eq0','positive_estimate_lt_one':'positive_estimate_lt1'}
  for key in ['positive_qerror_median','positive_qerror_p95','positive_qerror_p99','positive_qerror_max','zero_estimate_median','zero_estimate_p95','zero_estimate_max','zero_estimate_gt1','zero_estimate_gt10','zero_estimate_gt100','positive_estimate_exact_zero','positive_estimate_lt_one']:
   assert math.isclose(float(s[key]),metric[mapping.get(key,key)],rel_tol=1e-9,abs_tol=1e-9),(f,key)
  meta=dict(run=c['name'],scale=scale,method=method,source=kind)
  if kind=='cpu':
   seed=s['sample_seed'];assert c['name']==f'{scale}_{method}_seed{seed}'
   meta.update(sample_seed=seed,sample_rows=s['sample_rows'],sample_array_bytes=s['sample_array_bytes'],parameter_budget_bytes=s['parameter_budget_bytes'],latency_median_ms=statistics.median(float(r['latency_ms']) for r in rr))
   if method.startswith('matched_'):
    reference=method[len('matched_'):];cfg=load(R/next(c['path'] for c in main if c['scale']==scale and c['model']==reference and c['seed']==0).split('DSC5003_Naru_20261109/',1)[1]/'config.json');assert s['parameter_budget_bytes']==cfg['parameter_bytes'] and s['sample_array_bytes']==s['sample_rows']*44 and 0<=s['parameter_budget_bytes']-s['sample_array_bytes']<44
   elif method=='uniform_sample_10000':assert seed==20261109 and s['sample_rows']==10000 and s['sample_array_bytes']==440000
   else:assert seed is None and s['sample_rows']==s['sample_array_bytes']==0
  else:
   assert c['name']==f"{scale}_{method}_repeat{s['repetition']}";meta.update(analyze_repetition=s['repetition'],analyze_seconds=s['analyze_seconds'],planning_mean_ms=statistics.mean(float(r['planning_ms']) for r in rr),client_roundtrip_mean_ms=statistics.mean(float(r['client_wall_ms']) for r in rr));plans=[json.loads(x) for x in (f/'plans.jsonl').read_text().splitlines()];assert len(plans)==200
   for row,raw in zip(rr,plans):assert raw['query_id']==row['query_id'] and raw['explain']['Plan']['Plan Rows']==float(row['estimate']) and raw['explain']['Plan']['Node Type']==row['node']
  allmetrics.append(dict(meta,**metric));pair_rows.extend(dict(meta,**p) for p in pairs)
for r in allmetrics:
 if 'source' not in r:r['source']='model'
groups={}
for r in allmetrics:groups.setdefault((r['scale'],r['method']),[]).append(r)
aggregate=[]
for key,rr in groups.items():
 s=dict(scale=key[0],method=key[1],source=rr[0]['source'],runs=len(rr))
 for metric in ['positive_qerror_median','positive_qerror_p95','zero_estimate_median','zero_estimate_p95','zero_estimate_gt1','zero_estimate_gt10','positive_estimate_eq0','pair_strict_separation_fraction','pair_half_tie_score']:
  v=[float(r[metric]) for r in rr];s.update({metric+'_run_median':statistics.median(v),metric+'_run_min':min(v),metric+'_run_max':max(v)})
 aggregate.append(s)
assert len(allmetrics)==132 and len(pair_rows)==13200 and len(aggregate)==30 and rowcount==10200
assert set(groups)=={(s,m) for s in qs for m in ['own_made','own_resmade','official_resmade','exact_marginal_independence','uniform_sample_10000','matched_own_made','matched_own_resmade','PG-default','PG-target1000','PG-extstats']}
assert len({r['run'] for r in allmetrics})==132 and len({(r['run'],r['pair_id']) for r in pair_rows})==13200
modelaudit=load(R/'evidence/joint_model_audit.json');assert modelaudit['raw_model_rows_checked']==16200
write('all_methods_by_run.csv',allmetrics);write('all_methods_configuration_ranges.csv',aggregate);write('all_methods_paired_estimates.csv',pair_rows)
report={'status':'PASS','model_rows':modelaudit['raw_model_rows_checked'],'cpu_rows':24*200,'pg_plan_rows':27*200,'combined_raw_rows':modelaudit['raw_model_rows_checked']+rowcount,'method_run_rows':len(allmetrics),'paired_estimate_rows':len(pair_rows),'configuration_rows':len(aggregate),'selection':'600 directed stress records, 300 unique pairs; within-scale methods share fixed queries; across-scale query sets differ','aggregation':'Per-run metrics then median/min-max; model 3train x3MC, samples3draws or onefixed sample, PG3ANALYZE; not interchangeable replicates or confidence intervals','storage':'matched official sample aliases own_resmade because same FP32 budget; common dictionaries excluded','warning':'Zero-truth estimates have no ordinary ratio qerror; near-positive floor1 qerror can mask missed one-row queries, so report positive exact-zero counts and pair separation too'}
audit_path.write_text(json.dumps(report,indent=2));print(json.dumps(report))
lines=['# 联合零与近零正结果：全部方法分析交接','',
 '状态：COMPLETE，主线程重算PASS。600条冻结原值查询（每规模100对）全部通过实际PG COUNT；81项模型评估16200行、24CPU评估4800行、27PG评估5400计划，共26400原始记录。本地结果完整，原正式训练与主表保持不变。','',
 '选择只依据真实计数和固定种子，不读取模型误差。正例为1..10行，零例仅改单个原值谓词，列、操作符和谓词数保持配对一致；每条单列条件都有真实匹配，排除了空值域/NULL比较直接导致的零。每规模200条原值语义唯一。候选不足的规模用固定seed新行锚点补齐，来源和拒绝计数保存在兄弟joint_zero_20261111的manifest。','',
 '这是定向稀有结果压力分布，三个规模的查询集合不同，不能用该表独立推断数据规模效应。','',
 '## 结果','',
 '每个指标先按单run计算，再取重复的中位数。模型为3训练seed×3MC seed；匹配采样3个sample seed；PG3次ANALYZE；精确边际与10000行样本各1次。重复类型不同，不当作相同统计重复，也不标95%CI。','',
 '|规模|方法|正例q-error p95|零例估计p95行|正例估计恰为0/100|正例高于配对零例|',
 '|---|---|---:|---:|---:|---:|']
for scale in qs:
 for method in ['own_made','own_resmade','official_resmade','exact_marginal_independence','uniform_sample_10000','matched_own_resmade','PG-extstats']:
  x=next(r for r in aggregate if r['scale']==scale and r['method']==method)
  lines.append(f"|{scale}|{method}|{x['positive_qerror_p95_run_median']:.3f}|{x['zero_estimate_p95_run_median']:.3f}|{x['positive_estimate_eq0_run_median']:.0f}|{x['pair_strict_separation_fraction_run_median']:.0%}|")
lines+=['','## 解释与限制','',
 '- 三种模型能在本定向集合上较好区分稀有正例与配对零例，但输出不是精确0。full各模型的严格分离率均为99%，其中自写ResMADE九次运行均99%；不外推为任意SQL的零识别准确率。',
 '- full均匀10000行样本漏掉99/100正例，匹配ResMADE预算样本的三seed中位数为92/100。真正零例无样本命中不能单独证明优秀；只看零误报指标或floor1整体q-error会掩盖这种退化行为。',
 '- full自写ResMADE仍约22/100正例估计低于1行；floor1 q-error把truth=1且estimate<1都记为1，因此应同时报告原始估计、漏估计数和配对排名。没有把“未恰好输出0”解释成估计准确。',
 '- PG Seq Scan估计常下限为1，正/零配对容易同值。full多列统计严格分离60%，half-tie score为80%；两指标区别保留。',
 '- 精确边际的full配对分离率96%但正例p95为9，说明相对排序与绝对估计精度不同，不能只用分离率判断全面胜负。',
 '- 定向稀有查询中，官方ResMADE的full p95约2.975，自写约3.145，MADE约3.203；与原主表排序不同，不据此挑seed或改主默认配置。',
 '- 模型零结果估计仍有少量超过1行；这些是非空值域联合零，能够补充旧stress_v2只覆盖直接空域的限制。不存在充足证据证明联合零普遍优越。',
 '- 模型/采样逻辑存储比较均排除共同词典；official匹配样本复用own_resmade相同预算。PG目录行字节不是物理存储。GPU同步推断、CPU估计和PG规划/往返计时边界不同；未执行SQL，不声称端到端加速。','',
 '## 数据与复建','',
 '- `analysis/all_methods_by_run.csv`：132方法run；保留训练/MC/样本/ANALYZE身份、正例指标、零分布、pair分离。',
 '- `analysis/all_methods_configuration_ranges.csv`：30个规模/方法组合，逐run指标的median/min–max。',
 '- `analysis/all_methods_paired_estimates.csv`：13200行配对估计，可画同pair正例/零例散点；这些重复不增加300对独立查询数。',
 '- `analysis/metrics_by_run.csv`、`paired_estimates.csv`：模型81run原始重算；`constant_zero_sanity.csv`只作退化基线诊断，不算竞争模型。',
 '- 本根81个run的`per_query.csv`/status/summary/code为模型原始结果；CPU/PG原始结果在兄弟`joint_zero_20261111/`。',
 '- 正式epoch仍为660条；tiny的240条与本轮推断结果分开，不增加正式训练数。','',
 '从项目目录执行（均仅分析，不训练）：','',
 '```powershell','python Naru_Lab_20261109/scripts/audit_joint_model_eval.py','python Naru_Lab_20261109/scripts/summarize_joint_comparison.py','```','',
 '证据：项目evidence/joint_model_audit.json、joint_comparison_audit.json；CPU/PG分支audit.json及truth_counts.json。后续Agent据此作图和report，无需重跑实验。']
(M/'RESULTS_WITH_MODELS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
