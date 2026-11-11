"""SQLite independently checks raw-value remapping, NULL and boundary semantics."""
import importlib.util,json,sqlite3,sys
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('supp',Path(__file__).resolve().parents[1]/'scripts/supp_workload_build.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
cols=['category','reg_valid_date'];domains=[[None,'','A','C'],[None,'2025-12-31T00:00:00.000','2026-01-02T00:00:00.000']]
table=np.array([[0,0],[1,1],[2,1],[3,2],[3,2]],dtype=np.int32);compiler=mod.Compiler(cols,domains);oracle=mod.Oracle(table,domains)
db=sqlite3.connect(':memory:');db.execute('create table t(category text, reg_valid_date text)');db.executemany('insert into t values (?,?)',[(domains[0][a],domains[1][b]) for a,b in table])
checks=0
for i,col in enumerate(cols):
    values=[None,'','A','B','C','Z'] if i==0 else [None,'2025-01-01T00:00:00.000','2025-12-31T00:00:00.000','2026-01-01T00:00:00.000','2026-01-02T00:00:00.000']
    for op in ['=','<','<=','>','>=','IS NULL','IS NOT NULL']:
        for value in values:
            pred=compiler.predicate(i,op,value)
            if op.startswith('IS '):expected=db.execute(f'select count(*) from t where {col} {op}').fetchone()[0]
            else:expected=db.execute(f'select count(*) from t where {col} {op} ?',(value,)).fetchone()[0]
            assert oracle.count([pred])==expected,(i,op,value,pred,expected);checks+=1
intersection_checks=0
for op1 in ['=','<','<=','>','>=']:
    for op2 in ['=','<','<=','>','>=']:
        predicates=[compiler.predicate(0,op1,'A'),compiler.predicate(1,op2,'2026-01-01T00:00:00.000')]
        expected=db.execute(f'select count(*) from t where category {op1} ? and reg_valid_date {op2} ?',('A','2026-01-01T00:00:00.000')).fetchone()[0]
        assert oracle.count(predicates)==expected,(op1,op2,expected);intersection_checks+=1
# A literal missing from a smaller domain must stay absent; ranges use insertion points.
small=mod.Compiler(cols,[[None,'A'],domains[1]])
assert small.predicate(0,'=','C')['allowed_codes']==[]
assert small.predicate(0,'<=','B')['allowed_codes']==[1]
assert small.predicate(0,'>=','B')['allowed_codes']==[]
assert small.predicate(0,'IS NULL',None)['allowed_codes']==[0]
no_null=mod.Compiler(['category'],[['','A']])
assert no_null.predicate(0,'IS NULL',None)['allowed_codes']==[]
assert no_null.predicate(0,'IS NOT NULL',None)['allowed_codes']==[0,1]
assert no_null.predicate(0,'=',None)['allowed_codes']==[]
assert no_null.predicate(0,'=','')['allowed_codes']==[0]
print(json.dumps({'status':'PASS','sqlite_raw_value_checks':checks,'intersection_checks':intersection_checks,'cross_domain_absent_and_range_checks':4,'nonnullable_empty_string_checks':4,'training_performed':False}))
