"""Isolated PostgreSQL baseline; server is CPU-only, Unix socket only."""
import os,json,time,subprocess,ctypes,csv,traceback
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'experiments/pg_supplement_20261110'; SNAP=ROOT/'data/snapshot_resume_20261109_172915'
BIN=R/'install/bin'; RT=R/'runtime'; SOCKET=R/'s'
STATE={'status':'RUNNING','started':time.time(),'version':'17.6','scope':'CPU-only; no TCP; no shared install'}
def state(**kw):
 STATE.update(kw);(R/'status.json').write_text(json.dumps(STATE,indent=2))
def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False))
def run(cmd):
 p=subprocess.run(cmd,capture_output=True,text=True)
 if p.returncode:raise RuntimeError(json.dumps({'cmd':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}))
 return p.stdout
def as_pg(args):
 return ['/usr/bin/setpriv','--reuid=763494','--regid=0','--clear-groups','--','/usr/bin/env','LD_PRELOAD='+str(R/'identity_tools/usr/lib/libnss_wrapper.so'),'NSS_WRAPPER_PASSWD='+str(R/'private_passwd'),'NSS_WRAPPER_GROUP='+str(R/'private_group')]+args
class DB:
 def __init__(self):
  l=self.l=ctypes.CDLL(str(R/'install/lib/libpq.so'))
  for name,args,ret in [('PQconnectdb',[ctypes.c_char_p],ctypes.c_void_p),('PQstatus',[ctypes.c_void_p],ctypes.c_int),('PQerrorMessage',[ctypes.c_void_p],ctypes.c_char_p),('PQexec',[ctypes.c_void_p,ctypes.c_char_p],ctypes.c_void_p),('PQresultStatus',[ctypes.c_void_p],ctypes.c_int),('PQntuples',[ctypes.c_void_p],ctypes.c_int),('PQnfields',[ctypes.c_void_p],ctypes.c_int),('PQgetvalue',[ctypes.c_void_p,ctypes.c_int,ctypes.c_int],ctypes.c_char_p),('PQclear',[ctypes.c_void_p],None),('PQfinish',[ctypes.c_void_p],None)]:
   fn=getattr(l,name);fn.argtypes=args;fn.restype=ret
  self.c=l.PQconnectdb(('host='+str(SOCKET)+' port=55439 user=naru_pg dbname=postgres').encode())
  if l.PQstatus(self.c)!=0:raise RuntimeError(l.PQerrorMessage(self.c))
 def q(self,sql):
  l=self.l;r=l.PQexec(self.c,sql.encode())
  try:
   if l.PQresultStatus(r) not in (1,2):raise RuntimeError(l.PQerrorMessage(self.c).decode()+sql[:300])
   return [[l.PQgetvalue(r,i,j).decode() for j in range(l.PQnfields(r))] for i in range(l.PQntuples(r))]
  finally:l.PQclear(r)
 def close(self):self.l.PQfinish(self.c)
def lit(v,col=None):
 if v is None:return 'NULL'
 if col=='reg_valid_date':return "DATE '"+v[:10]+"'"
 return "'"+str(v).replace("'","''")+"'"
def where(q):
 a=[]
 for p in q['predicates']:
  c=p['column'];op=p['operator'].upper();assert op in ('=','<','<=','>','>=','IS NULL','IS NOT NULL')
  a.append('"'+c+'" '+op+('' if op in ('IS NULL','IS NOT NULL') else ' '+lit(p['value'],c)))
 return ' AND '.join(a)
def readq(scale,kind):return [json.loads(x) for x in (SNAP/'workloads'/scale/(kind+'.jsonl')).read_text().splitlines()]
# These semantic groups are fixed before loading any main query file. No test metrics choose groups.
GROUPS=[['record_type','registration_class','body_type','fuel_type'],['state','county'],['scofflaw_indicator','suspension_indicator','revocation_indicator']]
CONFIG={'tiers':{'PG-default':100,'PG-target1000':1000,'PG-extstats':1000},'groups':GROUPS,'group_selection':'Fixed semantic groups from documented column meanings; never selected against main workload. Vehicle identity, geography, legal flags.','statistics_kinds':['dependencies','mcv','ndistinct'],'analyze_repetitions':3,'query_scopes':{'main_native':2000,'scale_shared':1000,'predicate_sweep':1100,'stress_v2':300},'zero_reporting':'q_error excludes true zero queries; q_error_floor1 is max(max(1,est)/max(1,truth),max(1,truth)/max(1,est)); explicit Result zero and planner row clamp1 retained as raw estimates','query_semantics':'Ordinary NULL comparison never matches; text COLLATE C; reg_valid_date DATE; main zero results excluded by frozen workload.','statistics_bytes':'Per-table sum(pg_column_size(pg_statistic row)) and sum(pg_column_size(pg_statistic_ext_data row)) separately. Logical catalog row bytes as returned by server, excludes catalog indexes, shared pages, WAL, executable, heap data. pg_table_size reported separately; not a physical storage attribution.'}
dump(R/'fixed_config.json',CONFIG)
state(phase='initdb')
db=None;started=False
try:
 RT.mkdir(exist_ok=True);SOCKET.mkdir(exist_ok=True)
 identity=R/'identity_tools';identity.mkdir(exist_ok=True)
 run(['/usr/bin/dpkg-deb','-x',str(R/'nss-wrapper.deb'),str(identity)])
 (R/'private_passwd').write_text('naru_pg:x:763494:0:Private project PostgreSQL:'+str(RT)+':/bin/false\n')
 (R/'private_group').write_text('root:x:0:naru_pg\n')
 for p in (RT,SOCKET):os.chown(p,763494,0);os.chmod(p,0o700)
 cluster=RT/'cluster'
 if not (cluster/'PG_VERSION').exists():
  out=run(as_pg([str(BIN/'initdb'),'-D',str(cluster),'--no-locale','--encoding=UTF8','--auth-local=trust','--auth-host=reject']));(R/'initdb.log').write_text(out)
 cfg="\nlisten_addresses = ''\nunix_socket_directories = '"+str(SOCKET)+"'\nunix_socket_permissions = 0700\nport = 55439\nshared_buffers = '256MB'\nmax_parallel_workers_per_gather = 0\nmax_parallel_workers = 0\njit = off\nautovacuum = off\n"
 with (cluster/'postgresql.conf').open('a') as f:f.write(cfg)
 (R/'server_config.txt').write_text(cfg)
 run(as_pg([str(BIN/'pg_ctl'),'-D',str(cluster),'-l',str(RT/'server.log'),'-w','start']));started=True
 db=DB();dump(R/'server_identity.json',db.q("SELECT version(),current_user,current_setting('listen_addresses'),(SELECT datcollate FROM pg_database WHERE datname=current_database()),current_setting('max_parallel_workers_per_gather')"))
 for scale in ['1000000','5000000','full']:
  state(phase='prepare_data',scale=scale)
  out=R/scale;out.mkdir(exist_ok=True)
  d=json.loads((SNAP/'encoded'/(scale+'_domains.json')).read_text());cols=d['columns'];domains=d['domains'];arr=np.load(SNAP/'encoded'/(scale+'.npy'),mmap_mode='r');n=len(arr)
  tab='dmv_'+scale
  date_col=cols.index('reg_valid_date');assert all(v is None or v.endswith('T00:00:00.000') for v in domains[date_col])
  if not db.q("SELECT to_regclass('"+tab+"')")[0][0]:
   db.q('CREATE TABLE '+tab+' ('+','.join('"'+c+'" '+('date' if c=='reg_valid_date' else 'text COLLATE "C"') for c in cols)+')')
   raw=out/'import.csv';t=time.perf_counter()
   # A distinct NULL sentinel is verified absent from all non-NULL domains. Empty strings remain quoted data.
   assert all('__PG_NARU_NULL_20261110__' not in x for x in domains)
   with raw.open('w',newline='') as f:
    for pos in range(0,n,100000):
     block={}
     for j,c in enumerate(cols):
      values=[('__PG_NARU_NULL_20261110__' if v is None else (v[:10] if c=='reg_valid_date' else v)) for v in domains[j]]
      block[c]=np.array(values,dtype=object)[arr[pos:pos+100000,j]]
     pd.DataFrame(block).to_csv(f,index=False,header=False,quoting=csv.QUOTE_MINIMAL)
   os.chmod(raw,0o644)
   db.q("COPY "+tab+" FROM '"+str(raw)+"' WITH (FORMAT CSV,NULL '__PG_NARU_NULL_20261110__')")
   dump(out/'import_receipt.json',{'rows':n,'seconds':time.perf_counter()-t,'source_npy':str(SNAP/'encoded'/(scale+'.npy')),'null_sentinel':'__PG_NARU_NULL_20261110__','empty_string_policy':'empty fields are non-NULL values because CSV NULL uses a custom sentinel','date_policy':'ISO dates normalized; source times all midnight'})
   raw.unlink() # own regenerable temporary COPY input only
  state(phase='independent_count_checks',scale=scale)
  actual=int(db.q('SELECT count(*) FROM '+tab)[0][0]);assert actual==n
  checks=[]
  for q in readq(scale,'dev')[:8]:
   t=time.perf_counter();count=int(db.q('SELECT count(*) FROM '+tab+' WHERE '+where(q))[0][0]);assert count==q['truth'],(scale,q['query_id'],count,q['truth'])
   checks.append({'query_id':q['query_id'],'postgres_count':count,'frozen_numpy_truth':q['truth'],'seconds':time.perf_counter()-t})
  for j,c in enumerate(cols):
   expected=int(np.sum(arr[:,j]==domains[j].index(None))) if None in domains[j] else 0
   count=int(db.q('SELECT count(*) FROM '+tab+' WHERE "'+c+'" IS NULL')[0][0]);assert count==expected
   checks.append({'null_column':c,'postgres_count':count,'encoded_count':expected})
  for c in ('fuel_type','color'):
   count=int(db.q('SELECT count(*) FROM '+tab+' WHERE "'+c+'" = NULL')[0][0]);assert count==0
   checks.append({'ordinary_null_comparison':c,'postgres_count':count,'expected':0})
  dump(out/'correctness.json',{'status':'PASS','rows':actual,'checks':checks})
  main=readq(scale,'main_native');assert len(main)==2000
  scopes={'main_native':main}
  for name,total in [('scale_shared',1000),('predicate_sweep',1100),('stress_v2',300)]:
   path=ROOT/('experiments/workload_supplement_v2_20261110' if name=='stress_v2' else 'experiments/workload_supplement_20261110')/scale/(name+'.jsonl')
   if name=='stress_v2':
    ready=ROOT/'experiments/workload_supplement_v2_20261110/status.json'
    while not ready.exists() or json.loads(ready.read_text()).get('status')!='COMPLETE':state(phase='waiting_stress_v2',scale=scale);time.sleep(5)
   scopes[name]=[json.loads(x) for x in path.read_text().splitlines()];assert len(scopes[name])==total
  null_cases={}
  for q in scopes['stress_v2']:
   if q.get('stress_group')=='null_predicate':
    p=q['predicates'][0];null_cases.setdefault((p['column'],p['operator']),q)
  assert set(null_cases)=={(c,op) for c in ('fuel_type','color') for op in ('IS NULL','IS NOT NULL')},set(null_cases)
  null_checks=[]
  for (c,op),q in sorted(null_cases.items()):
   sql='SELECT count(*) FROM '+tab+' WHERE '+where(q);count=int(db.q(sql)[0][0]);assert count==q['truth'],(scale,c,op,count,q['truth'])
   null_checks.append({'column':c,'operator':op,'query_id':q['query_id'],'sql':sql,'postgres_count':count,'frozen_numpy_truth':q['truth']})
  dump(out/'null_combo_correctness.json',{'status':'PASS','cases':4,'checks':null_checks})
  supplement_checks=[]
  for idx in [0,99,100,149,199,200,250,299]:
   q=scopes['stress_v2'][idx];sql='SELECT count(*) FROM '+tab+' WHERE '+where(q);count=int(db.q(sql)[0][0]);assert count==q['truth'],(scale,q['query_id'],count,q['truth'])
   supplement_checks.append({'query_id':q['query_id'],'stress_group':q.get('stress_group'),'sql':sql,'postgres_count':count,'frozen_numpy_truth':q['truth']})
  dump(out/'supplement_correctness.json',{'status':'PASS','checks':supplement_checks})
  for tier,target in CONFIG['tiers'].items():
   for k in range(3):db.q('DROP STATISTICS IF EXISTS '+tab+'_ext'+str(k))
   for c in cols:db.q('ALTER TABLE '+tab+' ALTER COLUMN "'+c+'" SET STATISTICS '+str(target))
   db.q('SET default_statistics_target = '+str(target))
   if tier=='PG-extstats':
    for k,group in enumerate(GROUPS):
     name=tab+'_ext'+str(k);db.q('CREATE STATISTICS '+name+' (dependencies,mcv,ndistinct) ON '+','.join('"'+c+'"' for c in group)+' FROM '+tab);db.q('ALTER STATISTICS '+name+' SET STATISTICS 1000')
   for rep in range(1,4):
    state(phase='analyze',scale=scale,tier=tier,repetition=rep)
    t=time.perf_counter();db.q('ANALYZE '+tab);analyze=time.perf_counter()-t
    dst=out/tier/('repeat'+str(rep));dst.mkdir(parents=True,exist_ok=True)
    stat=db.q("SELECT coalesce(sum(pg_column_size(s)),0) FROM pg_statistic s WHERE starelid='"+tab+"'::regclass")[0][0]
    ext=db.q("SELECT coalesce(sum(pg_column_size(d)),0) FROM pg_statistic_ext_data d JOIN pg_statistic_ext s ON s.oid=d.stxoid WHERE s.stxrelid='"+tab+"'::regclass")[0][0]
    dump(dst/'analyze.json',{'analyze_seconds':analyze,'single_column_catalog_row_bytes':int(stat),'extended_catalog_row_bytes':int(ext),'table_bytes':int(db.q("SELECT pg_table_size('"+tab+"')")[0][0]),'columns':db.q("SELECT attname,attstattarget FROM pg_attribute WHERE attrelid='"+tab+"'::regclass AND attnum>0 ORDER BY attnum"),'extended_definitions':db.q("SELECT pg_get_statisticsobjdef(oid),stxstattarget FROM pg_statistic_ext WHERE stxrelid='"+tab+"'::regclass"),'settings':db.q("SELECT name,setting FROM pg_settings WHERE name IN ('default_statistics_target','max_parallel_workers_per_gather','max_parallel_workers','jit','lc_collate','server_version','shared_buffers') ORDER BY name")})
    dump(dst/'pg_stats.json',[json.loads(x[0]) for x in db.q("SELECT row_to_json(s) FROM pg_stats s WHERE schemaname='public' AND tablename='"+tab+"' ORDER BY attname")])
    dump(dst/'pg_stats_ext.json',[json.loads(x[0]) for x in db.q("SELECT row_to_json(s) FROM pg_stats_ext s WHERE schemaname='public' AND tablename='"+tab+"' ORDER BY statistics_name")])
    state(phase='explain',scale=scale,tier=tier,repetition=rep)
    summaries={}
    for scope,queries in scopes.items():
     state(scope=scope)
     metrics=[]
     with (dst/(scope+'_plans.jsonl')).open('w') as f:
      for q in queries:
       sql='SELECT * FROM '+tab+' WHERE '+where(q)
       t=time.perf_counter();plan=json.loads(db.q('EXPLAIN (FORMAT JSON, SETTINGS TRUE, SUMMARY TRUE) '+sql)[0][0])[0];wall=(time.perf_counter()-t)*1000
       assert not plan['Plan'].get('Parallel Aware',False)
       assert plan['Plan']['Node Type'] in ('Seq Scan','Result'),plan['Plan']
       if plan['Plan']['Node Type']=='Seq Scan':assert plan['Plan']['Relation Name']==tab
       estimate=plan['Plan']['Plan Rows'];truth=q['truth']
       if scope=='main_native':assert truth>0
       qe=max(max(1,estimate)/max(1,truth),max(1,truth)/max(1,estimate))
       metrics.append({'scope':scope,'query_id':q['query_id'],'stress_group':q.get('stress_group',''),'predicate_count':len(q['predicates']),'estimate':estimate,'truth':truth,'q_error':qe if truth>0 else None,'q_error_floor1':qe,'planning_ms':plan.get('Planning Time'),'client_wall_ms':wall,'node':plan['Plan']['Node Type'],'plan_zero':estimate==0,'truth_zero':truth==0})
       f.write(json.dumps({'scope':scope,'query_id':q['query_id'],'sql':sql,'explain':plan},ensure_ascii=False)+'\n')
     with (dst/(scope+'_metrics.csv')).open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
     errors=np.array([m['q_error'] for m in metrics if m['truth']>0])
     summaries[scope]={'status':'COMPLETE','queries':len(metrics),'nonzero_truth_queries':len(errors),'qerror_nonzero_truth':dict(zip(['median','p90','p95','p99','max'],[float(x) for x in np.percentile(errors,[50,90,95,99,100])])) if len(errors) else {},'zero_truth_queries':sum(m['truth']==0 for m in metrics),'exact_zero_estimates':sum(m['estimate']==0 for m in metrics),'zero_truth_with_plan_one':sum(m['truth']==0 and m['estimate']==1 for m in metrics),'zero_truth_with_estimate_gt_one':sum(m['truth']==0 and m['estimate']>1 for m in metrics),'mean_planning_ms':float(np.mean([m['planning_ms'] for m in metrics])),'mean_client_wall_ms':float(np.mean([m['client_wall_ms'] for m in metrics]))}
    dump(dst/'summary.json',summaries)
 state(status='COMPLETE',phase='done',ended=time.time())
except Exception as e:
 state(status='FAILED',error=repr(e),traceback=traceback.format_exc(),ended=time.time());raise
finally:
 if db:db.close()
 if started:
  try:(R/'shutdown.log').write_text(run(as_pg([str(BIN/'pg_ctl'),'-D',str(RT/'cluster'),'-m','fast','-w','stop'])))
  except Exception as e:(R/'shutdown_error.txt').write_text(repr(e))
