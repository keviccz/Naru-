"""Paired rare-positive/joint-zero workloads, CPU baselines and private PG.

No model errors are read and no GPU is accessed. Frozen older outputs are read
only; only the existing private PostgreSQL runtime is restarted and restatted.
"""
import csv,ctypes,copy,gc,json,operator,os,sqlite3,subprocess,sys,time,traceback
from collections import Counter
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/joint_zero_20261111';SNAP=ROOT/'data/snapshot_resume_20261109_172915'
PG=ROOT/'experiments/pg_supplement_20261110';BIN=PG/'install/bin';CLUSTER=PG/'runtime/cluster';SOCKET=PG/'s'
OUT.mkdir(exist_ok=False);(OUT/'workloads').mkdir();(OUT/'cpu').mkdir();(OUT/'pg').mkdir()
SCALES=['1000000','5000000','full'];MODELS=['own_made','own_resmade'];SEED=2026111101
GROUPS=[['record_type','registration_class','body_type','fuel_type'],['state','county'],['scofflaw_indicator','suspension_indicator','revocation_indicator']]
CONFIG={'seed':SEED,'scales':SCALES,'pairs_per_scale':100,'positive_truth_window':[1,10],'workload':'joint_zero_nearzero','positive_generation':'Fixed-seed shuffled unique main-native rare positives, then fresh real-row all-column equality anchors pruned in fixed random order while retaining 1..10 truth and at least five predicates; date equality retained for efficient exact counting.','zero_generation':'Change exactly one original predicate value to another observed non-NULL in-domain value; preserve column/operator/count; all marginal counts strictly positive, conjunction exactly zero.','query_selection':'Only frozen data/truth, never model estimates or errors. Targeted paired stress, not unbiased SQL sample.','cpu_methods':'exact marginals; 10000-row sample seed20261109; parameter-matched MADE and ResMADE int32 rows seeds0/1/2; official ResMADE identical budget/sample alias, not extra independent sample','pg_tiers':{'PG-default':100,'PG-target1000':1000,'PG-extstats':1000},'pg_groups':GROUPS,'pg_repetitions':3,'zero_metrics':'raw estimates and >1/>10/>100 row thresholds, plus floor1 qerror diagnostic; ordinary qerror positive truth only','storage':'FP32 trainable parameter bytes versus int32 rows; common dictionary excluded from both. PG catalog logical row bytes exclude heap/index/page/WAL/executable overhead.'}
STATE={'status':'RUNNING','phase':'generate_workloads','started_server_unix':time.time(),'scales':{},'cpu_runs':[],'pg_runs':[]}
def dump(path,obj):
 p=Path(path);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False));tmp.replace(p)
def state(**kw):STATE.update(kw);dump(OUT/'status.json',STATE)
def write_csv(path,rows):
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
dump(OUT/'fixed_config.json',CONFIG);dump(OUT/'workloads/status.json',{'status':'RUNNING','phase':'generate'});state()
OPS={'=':operator.eq,'<':operator.lt,'<=':operator.le,'>':operator.gt,'>=':operator.ge}
def signature(q):return json.dumps([(p['column_index'],p['operator'],p['value']) for p in sorted(q['predicates'],key=lambda p:p['column_index'])],ensure_ascii=False)
def masks(q,sizes):
 m=[None]*len(sizes)
 for p in q['predicates']:
  x=np.zeros(sizes[p['column_index']],dtype=bool);x[p['allowed_codes']]=True;m[p['column_index']]=x
 return m
class CounterIndex:
 def __init__(self,table,domains):
  self.table=table;self.domains=domains;self.hist=[np.bincount(table[:,j],minlength=len(d)) for j,d in enumerate(domains)];self.index={};self.calls=0
 def count(self,q):
  self.calls+=1
  ranked=sorted(q['predicates'],key=lambda p:int(self.hist[p['column_index']][p['allowed_codes']].sum()))
  p=ranked[0];j=p['column_index'];codes=p['allowed_codes']
  if not codes:return 0
  if j not in self.index:
   order=np.argsort(self.table[:,j],kind='stable');edges=np.r_[0,np.cumsum(self.hist[j])];self.index[j]=order,edges
  order,edges=self.index[j];ids=np.concatenate([order[edges[k]:edges[k+1]] for k in codes]);ok=np.ones(len(ids),dtype=bool)
  for p in ranked:
   j=p['column_index'];allowed=np.zeros(len(self.domains[j]),dtype=bool);allowed[p['allowed_codes']]=True;ok &= allowed[self.table[ids,j]]
   if not ok.any():return 0
  return int(ok.sum())
def sqlite_predicate_check(qs,domains):
 db=sqlite3.connect(':memory:');seen=set();checked=0
 for j,d in enumerate(domains):db.execute(f'CREATE TABLE d{j}(code INTEGER,value TEXT COLLATE BINARY)');db.executemany(f'INSERT INTO d{j} VALUES (?,?)',enumerate(d))
 for q in qs:
  for p in q['predicates']:
   key=(p['column_index'],p['operator'],p['value'])
   if key in seen:continue
   seen.add(key);assert p['value'] is not None and p['operator'] in OPS
   got=[x[0] for x in db.execute(f"SELECT code FROM d{p['column_index']} WHERE value {p['operator']} ? ORDER BY code",(p['value'],))]
   assert got==p['allowed_codes'],key;checked+=1
 db.close();return checked
def make_workload(scale):
 started=time.monotonic();rng=np.random.default_rng(SEED+SCALES.index(scale));data=np.load(SNAP/'encoded'/f'{scale}.npy',mmap_mode='r');meta=json.loads((SNAP/'encoded'/f'{scale}_domains.json').read_text());cols=meta['columns'];domains=meta['domains'];counter=CounterIndex(data,domains)
 main=[json.loads(x) for x in (SNAP/'workloads'/scale/'main_native.jsonl').read_text().splitlines()];existing=[q for q in main if 1<=q['truth']<=10 and len(q['predicates'])>=2];rng.shuffle(existing)
 sources=Counter();rejections=Counter();positive_seen=set();zero_seen=set();pairs=[];candidates=0;fresh=0
 def validate_positive(q):
  return all(p['value'] is not None and p['allowed_codes'] and int(counter.hist[p['column_index']][p['allowed_codes']].sum())>0 for p in q['predicates'])
 def fresh_anchor():
  nonlocal fresh
  for _ in range(10000):
   fresh+=1;row=data[int(rng.integers(len(data)))];pred=[]
   if any(domains[j][int(v)] is None for j,v in enumerate(row)):rejections['fresh_anchor_NULL']+=1;continue
   for j,v in enumerate(row):pred.append({'column_index':j,'column':cols[j],'operator':'=','value':domains[j][int(v)],'value_code':int(v),'allowed_codes':[int(v)]})
   q={'predicates':pred,'source':'fresh_anchor','source_anchor_number':fresh};truth=counter.count(q)
   if not 1<=truth<=10:rejections['fresh_anchor_truth_outside_1_10']+=1;continue
   for j in rng.permutation([x for x in range(len(cols)) if x!=6]):
    if len(q['predicates'])<=5:break
    trial=dict(q,predicates=[p for p in q['predicates'] if p['column_index']!=int(j)]);count=counter.count(trial)
    if 1<=count<=10:q=trial;truth=count
   q['truth']=truth;return q
  raise RuntimeError('No fresh rare positive within fixed 10000 anchor cap')
 while len(pairs)<100:
  candidates+=1
  if candidates>10000:raise RuntimeError('10000 candidate cap before 100 pairs')
  if existing:
   base=copy.deepcopy(existing.pop());base['source']='main_native';base['source_query_id']=base.pop('query_id');assert counter.count(base)==base['truth']
  else:base=fresh_anchor()
  if not validate_positive(base):rejections['positive_invalid_margin']+=1;continue
  sig=signature(base)
  if sig in positive_seen:rejections['positive_duplicate']+=1;continue
  found=None;change=None
  for pi in rng.permutation(len(base['predicates'])):
   old=base['predicates'][int(pi)];j=old['column_index'];available=[k for k,v in enumerate(domains[j]) if v is not None and v!=old['value'] and counter.hist[j][k]>0]
   if not available:continue
   for code in rng.choice(available,size=min(8,len(available)),replace=False):
    code=int(code);value=domains[j][code];allowed=[k for k,v in enumerate(domains[j]) if v is not None and OPS[old['operator']](v,value)]
    if not allowed or allowed==old['allowed_codes']:rejections['replacement_empty_or_same_mask']+=1;continue
    trial=copy.deepcopy(base);new=dict(old,value=value,value_code=code,allowed_codes=allowed);trial['predicates'][int(pi)]=new
    if signature(trial) in zero_seen:rejections['zero_duplicate']+=1;continue
    if counter.count(trial)!=0:rejections['replacement_still_positive']+=1;continue
    found=trial;change={'column_index':j,'column':cols[j],'operator':old['operator'],'positive_value':old['value'],'zero_value':value};break
   if found is not None:break
  if found is None:rejections['no_single_replacement_zero']+=1;continue
  index=len(pairs);pid=f'{scale}_pair_{index:03d}';group=[]
  for typ,q,truth in [('near_positive',base,int(base['truth'])),('zero',found,0)]:
   result={k:v for k,v in q.items() if k not in ['workload','truth','selectivity','group','query_id']};result.update(query_id=f'{pid}_{typ}',workload='joint_zero_nearzero',pair_id=pid,group=typ,truth=truth,selectivity=truth/len(data),changed_predicate=change)
   result['marginal_counts']=[int(counter.hist[p['column_index']][p['allowed_codes']].sum()) for p in result['predicates']];assert min(result['marginal_counts'])>0;group.append(result)
  positive_seen.add(signature(group[0]));zero_seen.add(signature(group[1]));pairs.append(group);sources[base['source']]+=1
  if len(pairs)%10==0:state(phase='generate_workloads',active_scale=scale,generated_pairs=len(pairs))
 qs=[q for pair in pairs for q in pair];assert len(qs)==200 and len({signature(q) for q in qs})==200 and len({q['query_id'] for q in qs})==200
 for pos,zero in pairs:
  assert 1<=pos['truth']<=10 and zero['truth']==0 and len(pos['predicates'])==len(zero['predicates'])
  a=sorted(pos['predicates'],key=lambda p:p['column_index']);b=sorted(zero['predicates'],key=lambda p:p['column_index']);assert all((p['column_index'],p['operator'])==(z['column_index'],z['operator']) for p,z in zip(a,b));assert sum(p['value']!=z['value'] for p,z in zip(a,b))==1
 checked=sqlite_predicate_check(qs,domains);target=OUT/'workloads'/scale;target.mkdir()
 with (target/'joint_zero_nearzero.jsonl').open('w') as f:
  for q in qs:f.write(json.dumps(q,ensure_ascii=False)+'\n')
 receipt={'status':'GENERATED_AWAITING_PG_COUNTS','rows':len(data),'queries':200,'pairs':100,'unique_semantic_queries':200,'positive_truth_window':[1,10],'source_pairs':dict(sources),'existing_rare_positive_candidates':sum(1<=q['truth']<=10 for q in main),'fresh_anchors_considered':fresh,'candidate_positives_considered':candidates,'rejections':dict(rejections),'predicate_count_pairs':dict(Counter(str(len(pair[0]['predicates'])) for pair in pairs)),'positive_truth_distribution':dict(Counter(str(pair[0]['truth']) for pair in pairs)),'sqlite_predicate_checks':checked,'indexed_conjunction_count_calls':counter.calls,'seconds':time.monotonic()-started}
 dump(target/'manifest.json',receipt);STATE['scales'][scale]=receipt;state();return qs
def run(cmd):
 p=subprocess.run(cmd,capture_output=True,text=True)
 if p.returncode:raise RuntimeError(json.dumps({'command':cmd,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}))
 return p.stdout
def as_pg(args):return ['/usr/bin/setpriv','--reuid=763494','--regid=0','--clear-groups','--','/usr/bin/env','LD_PRELOAD='+str(PG/'identity_tools/usr/lib/libnss_wrapper.so'),'NSS_WRAPPER_PASSWD='+str(PG/'private_passwd'),'NSS_WRAPPER_GROUP='+str(PG/'private_group')]+args
class DB:
 def __init__(self):
  l=self.l=ctypes.CDLL(str(PG/'install/lib/libpq.so'))
  for name,args,ret in [('PQconnectdb',[ctypes.c_char_p],ctypes.c_void_p),('PQstatus',[ctypes.c_void_p],ctypes.c_int),('PQerrorMessage',[ctypes.c_void_p],ctypes.c_char_p),('PQexec',[ctypes.c_void_p,ctypes.c_char_p],ctypes.c_void_p),('PQresultStatus',[ctypes.c_void_p],ctypes.c_int),('PQntuples',[ctypes.c_void_p],ctypes.c_int),('PQnfields',[ctypes.c_void_p],ctypes.c_int),('PQgetvalue',[ctypes.c_void_p,ctypes.c_int,ctypes.c_int],ctypes.c_char_p),('PQclear',[ctypes.c_void_p],None),('PQfinish',[ctypes.c_void_p],None)]:fn=getattr(l,name);fn.argtypes=args;fn.restype=ret
  self.c=l.PQconnectdb(('host='+str(SOCKET)+' port=55439 user=naru_pg dbname=postgres').encode());assert l.PQstatus(self.c)==0,l.PQerrorMessage(self.c)
 def q(self,sql):
  l=self.l;r=l.PQexec(self.c,sql.encode())
  try:
   if l.PQresultStatus(r) not in (1,2):raise RuntimeError(l.PQerrorMessage(self.c).decode()+sql[:300])
   return [[l.PQgetvalue(r,i,j).decode() for j in range(l.PQnfields(r))] for i in range(l.PQntuples(r))]
  finally:l.PQclear(r)
 def close(self):self.l.PQfinish(self.c)
def lit(v,col):
 assert v is not None
 return "DATE '"+v[:10]+"'" if col=='reg_valid_date' else "'"+str(v).replace("'","''")+"'"
def where(q):return ' AND '.join('"'+p['column']+'" '+p['operator']+' '+lit(p['value'],p['column']) for p in q['predicates'])
def metric_summary(rows):
 positive=[r for r in rows if r['truth']>0];zero=[r for r in rows if r['truth']==0];e=np.array([r['qerror_positive'] for r in positive]);z=np.array([r['estimate'] for r in zero])
 return {'queries':len(rows),'positive_queries':len(positive),'zero_queries':len(z),'positive_qerror_median':float(np.median(e)),'positive_qerror_p95':float(np.quantile(e,.95)),'positive_qerror_p99':float(np.quantile(e,.99)),'positive_qerror_max':float(e.max()),'positive_estimate_exact_zero':sum(int(r['estimate']==0) for r in positive),'positive_estimate_lt_one':sum(int(r['estimate']<1) for r in positive),'zero_estimate_median':float(np.median(z)),'zero_estimate_p95':float(np.quantile(z,.95)),'zero_estimate_max':float(z.max()),'zero_estimate_gt1':int((z>1).sum()),'zero_estimate_gt10':int((z>10).sum()),'zero_estimate_gt100':int((z>100).sum())}
def error(truth,est):return max(max(1,truth)/max(1,est),max(1,est)/max(1,truth))
def pg_evaluate(workloads):
 state(phase='pg_start');target=OUT/'pg';os.chown(target,763494,0)
 probe=subprocess.run(as_pg([str(BIN/'pg_ctl'),'-D',str(CLUSTER),'status']),capture_output=True,text=True)
 assert probe.returncode!=0,'Existing private PG server already active; refuse to share mutable statistics'
 db=None;started=False
 try:
  (target/'start.log').write_text(run(as_pg([str(BIN/'pg_ctl'),'-D',str(CLUSTER),'-l',str(target/'server.log'),'-w','start'])));started=True;db=DB()
  db.q('SET max_parallel_workers_per_gather=0');db.q('SET jit=off');dump(target/'server_identity.json',db.q("SELECT version(),current_user,current_setting('listen_addresses'),current_setting('max_parallel_workers_per_gather')"))
  count_total=0
  for scale in SCALES:
   state(phase='pg_independent_truth',active_scale=scale);qs=workloads[scale];tab='dmv_'+scale;dst=target/scale;dst.mkdir();d=json.loads((SNAP/'encoded'/f'{scale}_domains.json').read_text());arr=np.load(SNAP/'encoded'/f'{scale}.npy',mmap_mode='r');assert int(db.q('SELECT COUNT(*) FROM '+tab)[0][0])==len(arr)
   histograms={}
   for j,col in enumerate(d['columns']):
    actual={json.loads(v[0])[0]:int(json.loads(v[0])[1]) for v in db.q('SELECT json_build_array("'+col+'",COUNT(*))::text FROM '+tab+' GROUP BY "'+col+'"')}
    dom=[v[:10] if col=='reg_valid_date' and v is not None else v for v in d['domains'][j]];hist=np.bincount(arr[:,j],minlength=len(dom));assert actual=={v:int(hist[k]) for k,v in enumerate(dom)},(scale,col)
    histograms[col]={'rows':int(hist.sum()),'domain_values':dom,'postgres_counts':[actual[v] for v in dom]}
   dump(dst/'marginal_histogram_check.json',{'status':'PASS','columns':histograms})
   checks=[]
   for qi,q in enumerate(qs):
    sql='SELECT COUNT(*) FROM '+tab+' WHERE '+where(q);begin=time.perf_counter();truth=int(db.q(sql)[0][0]);assert truth==q['truth'],(scale,q['query_id'],truth,q['truth'])
    checks.append({'query_id':q['query_id'],'pair_id':q['pair_id'],'group':q['group'],'postgres_count':truth,'encoded_count':q['truth'],'sql':sql,'seconds':time.perf_counter()-begin});count_total+=1
    if qi%20==19:state(phase='pg_independent_truth',active_scale=scale,verified_queries=count_total)
   dump(dst/'truth_counts.json',{'status':'PASS','queries':len(checks),'checks':checks});STATE['scales'][scale]['status']='COMPLETE';dump(OUT/'workloads'/scale/'manifest.json',STATE['scales'][scale])
  manifest={'status':'COMPLETE','config':CONFIG,'scales':STATE['scales'],'independent_pg_count_checks':count_total,'single_column_pg_histogram_checks':33,'workloads_immutable_after_this_point':True};assert count_total==600
  dump(OUT/'workloads/manifest.json',manifest);dump(OUT/'workloads/status.json',{'status':'COMPLETE','queries':600,'scales':SCALES,'independent_pg_count_checks':600});state(phase='workloads_frozen',verified_queries=600)
  for scale in SCALES:
   tab='dmv_'+scale;d=json.loads((SNAP/'encoded'/f'{scale}_domains.json').read_text());cols=d['columns']
   for tier,stats_target in CONFIG['pg_tiers'].items():
    for k in range(3):db.q('DROP STATISTICS IF EXISTS '+tab+'_ext'+str(k))
    for col in cols:db.q('ALTER TABLE '+tab+' ALTER COLUMN "'+col+'" SET STATISTICS '+str(stats_target))
    db.q('SET default_statistics_target='+str(stats_target))
    if tier=='PG-extstats':
     for k,group in enumerate(GROUPS):
      name=tab+'_ext'+str(k);db.q('CREATE STATISTICS '+name+' (dependencies,mcv,ndistinct) ON '+','.join('"'+c+'"' for c in group)+' FROM '+tab);db.q('ALTER STATISTICS '+name+' SET STATISTICS 1000')
    for rep in [1,2,3]:
     name=f'{scale}_{tier}_repeat{rep}';cell={'name':name,'scale':scale,'tier':tier,'repetition':rep,'status':'RUNNING'};STATE['pg_runs'].append(cell);state(phase='pg_analyze',active_pg=name);dst=target/scale/tier/f'repeat{rep}';dst.mkdir(parents=True)
     begin=time.perf_counter();db.q('ANALYZE '+tab);seconds=time.perf_counter()-begin
     single=int(db.q("SELECT COALESCE(SUM(pg_column_size(s)),0) FROM pg_statistic s WHERE starelid='"+tab+"'::regclass")[0][0]);ext=int(db.q("SELECT COALESCE(SUM(pg_column_size(d)),0) FROM pg_statistic_ext_data d JOIN pg_statistic_ext s ON s.oid=d.stxoid WHERE s.stxrelid='"+tab+"'::regclass")[0][0])
     dump(dst/'analyze.json',{'seconds':seconds,'single_column_catalog_row_bytes':single,'extended_catalog_row_bytes':ext,'column_targets':db.q("SELECT attname,attstattarget FROM pg_attribute WHERE attrelid='"+tab+"'::regclass AND attnum>0 ORDER BY attnum"),'extended_objects':db.q("SELECT pg_get_statisticsobjdef(oid),stxstattarget FROM pg_statistic_ext WHERE stxrelid='"+tab+"'::regclass")})
     dump(dst/'pg_stats.json',[json.loads(v[0]) for v in db.q("SELECT row_to_json(s) FROM pg_stats s WHERE tablename='"+tab+"' ORDER BY attname")]);dump(dst/'pg_stats_ext.json',[json.loads(v[0]) for v in db.q("SELECT row_to_json(s) FROM pg_stats_ext s WHERE tablename='"+tab+"' ORDER BY statistics_name")])
     rows=[];state(phase='pg_explain',active_pg=name)
     with (dst/'plans.jsonl').open('w') as f:
      for q in workloads[scale]:
       sql='SELECT * FROM '+tab+' WHERE '+where(q);begin=time.perf_counter();raw=json.loads(db.q('EXPLAIN (FORMAT JSON,SETTINGS TRUE,SUMMARY TRUE) '+sql)[0][0])[0];wall=(time.perf_counter()-begin)*1000;plan=raw['Plan'];assert plan['Node Type'] in ['Seq Scan','Result'] and not plan.get('Parallel Aware',False)
       if plan['Node Type']=='Seq Scan':assert plan['Relation Name']==tab
       est=plan['Plan Rows'];truth=q['truth'];qe=error(truth,est);rows.append(dict(query_id=q['query_id'],pair_id=q['pair_id'],group=q['group'],truth=truth,estimate=est,qerror_positive=qe if truth>0 else None,qerror_floor1=qe,planning_ms=raw['Planning Time'],client_wall_ms=wall,node=plan['Node Type']));f.write(json.dumps({'query_id':q['query_id'],'sql':sql,'explain':raw})+'\n')
     write_csv(dst/'per_query.csv',rows);summary=metric_summary(rows);summary.update(status='COMPLETE',scale=scale,method=tier,repetition=rep,analyze_seconds=seconds);dump(dst/'summary.json',summary);cell['status']='COMPLETE';state()
 finally:
  if db:db.close()
  if started:(target/'shutdown.log').write_text(run(as_pg([str(BIN/'pg_ctl'),'-D',str(CLUSTER),'-m','fast','-w','stop'])))
def cpu_evaluate(workloads):
 sys.path.insert(0,str(ROOT));from src.baselines import UniformSample
 matrix=json.loads((ROOT/'experiments/main_matrix_20261109/status.json').read_text())['runs']
 for scale in SCALES:
  data=np.load(SNAP/'encoded'/f'{scale}.npy',mmap_mode='r');d=json.loads((SNAP/'encoded'/f'{scale}_domains.json').read_text());sizes=list(map(len,d['domains']));hist=[np.bincount(data[:,j],minlength=v) for j,v in enumerate(sizes)];n=len(data)
  cfg={m:json.loads((Path(next(c['path'] for c in matrix if c['scale']==scale and c['model']==m and c['seed']==0))/'config.json').read_text()) for m in MODELS+['official_resmade']};assert cfg['own_resmade']['parameter_bytes']==cfg['official_resmade']['parameter_bytes']
  methods=[('exact_marginal_independence',None,None,0),('uniform_sample_10000',UniformSample(data,10000,20261109),20261109,440000)]
  for model in MODELS:
   for seed in [0,1,2]:methods.append(('matched_'+model,UniformSample(data,cfg[model]['parameter_bytes']//44,seed),seed,cfg[model]['parameter_bytes']))
  for method,sample,seed,budget in methods:
   name=f'{scale}_{method}_seed{seed}';dst=OUT/'cpu'/name;dst.mkdir();cell={'name':name,'scale':scale,'method':method,'sample_seed':seed,'status':'RUNNING'};STATE['cpu_runs'].append(cell);state(phase='cpu_baselines',active_cpu=name);rows=[]
   for q in workloads[scale]:
    allowed=masks(q,sizes);begin=time.perf_counter()
    if sample is None:
     est=float(n)
     for j,mask in enumerate(allowed):
      if mask is not None:est*=hist[j][mask].sum()/n
    else:est=sample.estimate(allowed)
    ms=(time.perf_counter()-begin)*1000;truth=q['truth'];qe=error(truth,est);rows.append(dict(query_id=q['query_id'],pair_id=q['pair_id'],group=q['group'],truth=truth,estimate=est,qerror_positive=qe if truth>0 else None,qerror_floor1=qe,latency_ms=ms))
   write_csv(dst/'per_query.csv',rows);summary=metric_summary(rows);summary.update(status='COMPLETE',scale=scale,method=method,sample_seed=seed,parameter_budget_bytes=budget,sample_rows=0 if sample is None else len(sample.table),sample_array_bytes=0 if sample is None else sample.table.nbytes,timing_scope='CPU estimate only; prebuilt numpy masks, samples and histograms excluded',official_sample_alias='matched_own_resmade' if method=='matched_own_resmade' else None);dump(dst/'summary.json',summary);cell['status']='COMPLETE';state()
try:
 workloads={scale:make_workload(scale) for scale in SCALES};dump(OUT/'workloads/status.json',{'status':'AWAITING_PG_VERIFICATION','queries':600});pg_evaluate(workloads);cpu_evaluate(workloads);state(status='COMPLETE',phase='done',ended_server_unix=time.time())
except BaseException as e:
 state(status='FAILED',error=repr(e),ended_server_unix=time.time());(OUT/'traceback.txt').write_text(traceback.format_exc());raise
