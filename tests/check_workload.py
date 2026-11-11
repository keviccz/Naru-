"""Independent SQLite checks for encoded exact workload truth; no model fits."""
import itertools,json,sqlite3,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from build_workload import allowed_codes,ExactOracle,generate,signature

domains=[[None]+list(range(12)),[None,'a','b'],[0,1]]
columns=['a','b','c']
table=np.array(list(itertools.product(range(13),range(3),range(2))),dtype=np.int32)
db=sqlite3.connect(':memory:');db.execute('create table t(a integer,b text,c integer)')
db.executemany('insert into t values (?,?,?)',[tuple(domains[i][int(v)] for i,v in enumerate(row)) for row in table])
oracle=ExactOracle(table,domains);checked=0
for i,d in enumerate(domains):
    for code in range(len(d)):
        for op in ('=','<=','>='):
            pred={'column_index':i,'operator':op,'value_code':code}
            expected=db.execute(f'select count(*) from t where {columns[i]} {op} ?',(d[code],)).fetchone()[0]
            assert oracle.count([pred])==expected
            checked+=1
dev,ds=generate(table,columns,domains,50,101,'dev',oracle,1,3)
main,ms=generate(table,columns,domains,50,102,'main_native',oracle,1,3,excluded={signature(q['predicates']) for q in dev})
assert not {signature(q['predicates']) for q in dev}&{signature(q['predicates']) for q in main}
for q in dev+main:
    terms=[];args=[]
    for pred in q['predicates']:
        terms.append(pred['column']+' '+pred['operator']+' ?');args.append(pred['value'])
        if len(domains[pred['column_index']])<10:assert pred['operator']=='='
        expected=np.flatnonzero(allowed_codes(domains[pred['column_index']],pred['operator'],pred['value_code'])).tolist()
        assert expected==pred['allowed_codes']
    truth=db.execute('select count(*) from t where '+' and '.join(terms),args).fetchone()[0]
    assert truth==q['truth'] and truth>0
again,_=generate(table,columns,domains,50,101,'dev',oracle,1,3)
assert again==dev
assert ds['zero_result_skipped']>0
print(json.dumps({'status':'PASS','single_predicate_sql_checks':checked,'generated_query_sql_checks':100,'dev':ds,'main':ms}))
