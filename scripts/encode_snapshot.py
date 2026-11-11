"""Two-pass, bounded-memory encoding; run on the authorized remote machine."""
import argparse,json,datetime,time
from pathlib import Path
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('snapshot');a=p.parse_args()
root=Path(a.snapshot);meta=json.loads((root/'manifest.json').read_text())
assert meta['status']=='COMPLETE'
cols=meta['columns'];sentinel=meta['null_sentinel'];n=meta['expected_rows']
out=root/'encoded';out.mkdir(exist_ok=False)
start=time.time();domains=[set() for _ in cols]
for chunk in pd.read_csv(root/'dmv.csv',dtype=str,keep_default_na=False,chunksize=200000):
    for i,c in enumerate(cols):domains[i].update(chunk[c].unique())
for i,c in enumerate(cols):
    null=sentinel in domains[i];domains[i].discard(sentinel)
    values=sorted(domains[i],key=pd.Timestamp if c=='reg_valid_date' else None)
    domains[i]=([None] if null else [])+values
maps=[{sentinel if v is None else v:j for j,v in enumerate(d)} for d in domains]
arr=np.lib.format.open_memmap(out/'full.npy',mode='w+',dtype=np.int32,shape=(n,len(cols)))
pos=0
for chunk in pd.read_csv(root/'dmv.csv',dtype=str,keep_default_na=False,chunksize=200000):
    for i,c in enumerate(cols):
        mapped=chunk[c].map(maps[i]);assert not mapped.isna().any()
        arr[pos:pos+len(chunk),i]=mapped.to_numpy(dtype=np.int32)
    pos+=len(chunk);print('encoded',pos,flush=True)
assert pos==n;arr.flush()
(out/'full_domains.json').write_text(json.dumps({'columns':cols,'domains':domains}),encoding='utf-8')
ids=np.random.default_rng(20261109).choice(n,size=min(5000000,n),replace=False)
np.save(out/'subset_row_ids.npy',ids)
for size in (1000000,5000000):
    if size>n:continue
    subset=np.array(arr[ids[:size]],copy=True)
    sd=[]
    for i in range(len(cols)):
        values,inverse=np.unique(subset[:,i],return_inverse=True)
        subset[:,i]=inverse;sd.append([domains[i][j] for j in values])
    np.save(out/f'{size}.npy',subset)
    (out/f'{size}_domains.json').write_text(json.dumps({'columns':cols,'domains':sd}),encoding='utf-8')
(out/'encoding_receipt.json').write_text(json.dumps({'status':'COMPLETE','rows':n,'columns':cols,'subset_seed':20261109,'elapsed_seconds':time.time()-start,'storage':'int32 NPY; dictionaries JSON','domain_policy':'per-scale domains; lexical categories, chronological dates; NULL first when present'},indent=2))
print('COMPLETE',time.time()-start,flush=True)
