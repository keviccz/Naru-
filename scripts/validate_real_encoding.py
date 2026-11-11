import csv,json,sqlite3
from pathlib import Path
import numpy as np
root=Path('/dongqishi01/Zhangdongjian/DSC5003_Naru_20261109');s=root/'data/snapshot_resume_20261109_172915'
spec=json.loads((s/'encoded/full_domains.json').read_text());d=spec['domains'];a=np.load(s/'encoded/full.npy',mmap_mode='r')
with (s/'dmv.csv').open(newline='') as f:
 reader=csv.reader(f);assert next(reader)==spec['columns']
 for i in range(10000):
  row=next(reader)
  for j,v in enumerate(row):assert d[j][a[i,j]]==(None if v=='__NARU_NULL__' else v)
ids=np.load(s/'encoded/subset_row_ids.npy',mmap_mode='r');rng=np.random.default_rng(16)
for scale in ['1000000','5000000']:
 b=np.load(s/'encoded'/f'{scale}.npy',mmap_mode='r');sd=json.loads((s/'encoded'/f'{scale}_domains.json').read_text())['domains']
 for i in rng.choice(len(b),1000,replace=False):
  for j in range(11):assert sd[j][b[i,j]]==d[j][a[ids[i],j]]
receipt={'status':'PASS','raw_roundtrip_cells':110000,'nested_subset_roundtrip_cells':22000,'formal_snapshot_rows':len(a),'scope':'raw prefix roundtrip and fixed random subset row mapping; not exhaustive predicate validation'}
(s/'encoded/real_validation.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
