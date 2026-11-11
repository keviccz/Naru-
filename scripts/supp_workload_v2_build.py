"""Repair NULL coverage in an isolated stress-v2 output; preserve all v1 files."""
import collections,itertools,json,sqlite3,time,traceback,datetime
from pathlib import Path
import numpy as np
from supp_workload_build import Compiler,Oracle,query
root=Path(__file__).resolve().parents[1]
source=root/'experiments/workload_supplement_20261110'
snapshot=root/'data/snapshot_resume_20261109_172915'
out=root/'experiments/workload_supplement_v2_20261110';out.mkdir(exist_ok=False)
state={'status':'RUNNING','source_v1':str(source),'snapshot':str(snapshot),'preserve_v1':True,
       'reason':'v1 column index j%2 and operator j%2 were coupled, covering fuel_type IS NULL and color IS NOT NULL only',
       'repair':'enumerate product of nullable columns and both NULL operators first, then cycle 25 times per pair',
       'counts_per_scale':300,'scales':{},'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def save():
    for name in ['status','manifest']:
        p=out/(name+'.tmp');p.write_text(json.dumps(state,indent=2),encoding='utf-8');p.replace(out/(name+'.json'))
save();start=time.time()
try:
    for scale in ['1000000','5000000','full']:
        spec=json.loads((snapshot/'encoded'/f'{scale}_domains.json').read_text());cols=spec['columns'];domains=spec['domains']
        table=np.load(snapshot/'encoded'/f'{scale}.npy',mmap_mode='r');compiler=Compiler(cols,domains);oracle=Oracle(table,domains)
        nullable=[i for i,d in enumerate(domains) if d[0] is None]
        assert {cols[i] for i in nullable}=={'fuel_type','color'}
        combinations=list(itertools.product(nullable,['IS NULL','IS NOT NULL']));assert len(combinations)==4
        db=sqlite3.connect(':memory:');db.execute('create table population(col integer,value text,frequency integer)')
        for i in nullable:
            codes,frequencies=np.unique(table[:,i],return_counts=True)
            db.executemany('insert into population values (?,?,?)',[(i,domains[i][int(code)],int(frequency)) for code,frequency in zip(codes,frequencies)])
        sql={}
        for i,op in combinations:
            sql[(i,op)]=db.execute('select coalesce(sum(frequency),0) from population where col=? and value '+op,(i,)).fetchone()[0]
            assert oracle.count([compiler.predicate(i,op,None)])==sql[(i,op)]
        old=[json.loads(line) for line in (source/scale/'stress.jsonl').read_text().splitlines()];assert len(old)==300
        dest=out/scale;dest.mkdir();coverage=collections.Counter();zero=0
        with (dest/'stress_v2.jsonl').open('x',encoding='utf-8') as f:
            for j,q in enumerate(old):
                raw=[{k:p[k] for k in ['column_index','operator','value']} for p in q['predicates']]
                extra={k:v for k,v in q.items() if k not in ['query_id','workload','predicates','truth','selectivity']}
                if q['stress_group']=='null_predicate':
                    i,op=combinations[(j-100)%len(combinations)];raw=[{'column_index':i,'operator':op,'value':None}];extra['nullable_domain']=True
                    coverage[(i,op)]+=1
                new=query(f'stress_v2_{j:05d}','stress_v2',raw,compiler,oracle,**extra)
                if q['stress_group']!='null_predicate':assert new['predicates']==q['predicates'] and new['truth']==q['truth']
                else:assert new['truth']==sql[(i,op)]
                zero+=int(new['truth']==0);f.write(json.dumps(new,ensure_ascii=False)+'\n')
        assert coverage==collections.Counter({pair:25 for pair in combinations})
        state['scales'][scale]={'status':'COMPLETE','rows':len(table),'queries':300,'zero_results':zero,
          'NULL_coverage':[{'column':cols[i],'operator':op,'queries':coverage[(i,op)],'truth':sql[(i,op)]} for i,op in combinations],
          'independent_SQL_check':'SQLite SUM(frequency) over raw-value frequencies independently obtained using np.unique; not Oracle bincount; 4 conditions verified'}
        save();print(scale,json.dumps(state['scales'][scale]),flush=True)
    state['status']='COMPLETE'
except BaseException as e:
    state.update(status='FAILED',error=repr(e));(out/'traceback.txt').write_text(traceback.format_exc());raise
finally:state.update(elapsed_seconds=time.time()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
