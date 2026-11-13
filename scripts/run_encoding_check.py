import csv,json,subprocess
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];p=root/'data/synthetic_encoding_check';p.mkdir(parents=True,exist_ok=False)
cols=['state','reg_valid_date','flag'];rows=[['NA','2026-01-02T00:00:00.000',''],['__NARU_NULL__','__NARU_NULL__','Y'],['NY','2025-12-31T00:00:00.000','N']]*10
with (p/'dmv.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(cols);w.writerows(rows)
(p/'manifest.json').write_text(json.dumps({'status':'COMPLETE','source':'synthetic correctness fixture','columns':cols,'expected_rows':len(rows),'null_sentinel':'__NARU_NULL__'}))
r=subprocess.run(['/opt/conda/envs/R4.2/bin/python',str(root/'scripts/encode_snapshot.py'),str(p)],capture_output=True,text=True,timeout=60)
if r.returncode:raise RuntimeError(r.stderr)
a=np.load(p/'encoded/full.npy');d=json.loads((p/'encoded/full_domains.json').read_text())['domains']
for i,row in enumerate(rows):
 for j,v in enumerate(row):assert d[j][a[i,j]]==(None if v=='__NARU_NULL__' else v)
print(json.dumps({'status':'PASS','rows':len(rows),'columns':cols,'exact_roundtrip_cells':int(a.size),'null_empty_NA_distinct':True,'date_domain':d[1]}))
