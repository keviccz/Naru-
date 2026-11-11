import json,sys,sqlite3
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from baselines import exact_count,independence,UniformSample,qerror

table=np.array([[0,0],[1,1],[1,1],[2,2],[2,2],[2,2]],dtype=np.int32)
# Code zero denotes NULL, excluded from every ordinary comparison.
db=sqlite3.connect(':memory:');db.execute('create table t(a integer,b integer)')
db.executemany('insert into t values (?,?)',[(None if a==0 else int(a),None if b==0 else int(b)) for a,b in table])
checks=[]
for threshold in [0,1,2,3]:
    mask=np.array([False,1<=threshold,2<=threshold])
    masks=[mask,mask]
    sql=db.execute('select count(*) from t where a<=? and b<=?',(threshold,threshold)).fetchone()[0]
    assert exact_count(table,masks)==sql
    assert UniformSample(table,6).estimate(masks)==sql
    checks.append({'threshold':threshold,'truth':sql})
mask=np.array([False,True,False]);masks=[mask,mask]
assert exact_count(table,masks)==2
assert abs(independence(table,masks)-2/3)<1e-12
assert exact_count(table,[np.array([True,False,False]),None])==db.execute('select count(*) from t where a is null').fetchone()[0]
assert qerror(2,2/3)==2 # clamped estimate, deliberate metric convention
assert qerror(0,0)==1
print(json.dumps({'status':'PASS','checks':checks,'correlated_true':2,'independence':independence(table,masks),'oracle':'independent SQLite comparisons'}))
