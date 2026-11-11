"""Preserve full unpatched upstream edge tracebacks separately from supported checks."""
import argparse,json,traceback
from pathlib import Path
import numpy as np
import torch
import compat_validate as c
p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args()
torch.manual_seed(71);sizes=[3,4,3,2];raw=c.model(sizes,32,3,71);table=c.table_for([[None,1,2],list(range(4)),list(range(3)),list(range(2))],['A','B','C','D'],72);up=c.estimators.ProgressiveSampling(raw,table,1024,device=torch.device('cpu'),shortcircuit=True)
results=[]
for name,preds in [('all_wildcards',[]),('one_filter',[{'column_index':2,'operator':'=','value_code':1}])]:
 q={'query_id':name,'predicates':preds};ops,vals,masks=c.query_from_preds(q,table)
 try:
  up.inp.zero_();value=up._sample_n(1024,np.arange(4),table.columns,ops,vals);result={'query_id':name,'unexpected_upstream_success':value}
 except Exception as e:
  result={'query_id':name,'upstream_exception':repr(e),'traceback':traceback.format_exc(),'shared_selectivity':c.progressive_sample(c.Adapter(raw,sizes),masks,samples=1024,seed=700,wildcard_skipping=True),'classification':'UPSTREAM_NATIVE_UNSUPPORTED; excluded from consistency pass count'}
 results.append(result)
out=c.ROOT/'experiments/compat_supplement_20261110'/a.label/'upstream_edge_tracebacks.json';out.write_text(json.dumps({'torch':torch.__version__,'native_unsupported_cases':len(results),'cases':results},indent=2));print(out)
