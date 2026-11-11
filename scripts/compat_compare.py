"""Compare same checkpoint CPU outputs without claiming a historical training reproduction."""
import csv,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]/'experiments/compat_supplement_20261110'
base=np.load(root/'torch241/fixed_checkpoint_outputs.npz');old=np.load(root/'torch113/fixed_checkpoint_outputs.npz')
assert np.array_equal(base['input'],old['input']),'Fixed input rows differ'
checks={}
for k in ['logits','masked_logits','nll']:
 diff=np.abs(base[k]-old[k]);checks[k]={'max_abs':float(diff.max()),'mean_abs':float(diff.mean()),'tolerance':2e-5,'pass':bool(diff.max()<=2e-5)}
summaries={v:json.loads((root/v/'summary.json').read_text()) for v in ['torch241','torch113']}
data={}
for v in summaries:
 rows=[json.loads(x) for x in (root/v/'checks.jsonl').read_text().splitlines()]
 data[v]={(x['suite'],x['query_id'],x['skip']):x for x in rows}
mc=[]
for key,a in data['torch241'].items():
 b=data['torch113'][key]
 for method in ['upstream','shared']:
  # Repeat-level variances are retained in raw CSV; same fixed MC criterion.
  vals=[]
  for v in ['torch241','torch113']:
   with (root/v/(key[0]+'_repeats.csv')).open() as f:
    vals.append([float(r[method]) for r in csv.DictReader(f) if r['query_id']==key[1] and r['skip']==str(key[2])])
  tol=max(1e-6,.05*max(abs(np.mean(x)) for x in vals),6*float(np.sqrt(sum(np.var(x,ddof=1)/len(x) for x in vals))))
  diff=abs(np.mean(vals[0])-np.mean(vals[1]));mc.append({'suite':key[0],'query_id':key[1],'skip':key[2],'method':method,'abs_diff':float(diff),'tolerance':tol,'pass':bool(diff<=tol)})
result={'status':'COMPLETE','passed':all(x['pass'] for x in checks.values()) and all(x['pass'] for x in mc) and all(x['passed'] for x in summaries.values()),'deterministic':checks,'monte_carlo_comparisons':mc,'within_version':summaries,'scope':'Same official 1M seed0 checkpoint on CPU in PyTorch2.4.1 and1.13.1, including masked inputs and original upstream inference; no historical full-training equivalence claimed'}
(root/'version_comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps({'passed':result['passed'],'deterministic':checks,'mc_checks':len(mc),'mc_failures':sum(not x['pass'] for x in mc)}))
