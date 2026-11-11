"""Independent standard-library audit and plot-ready tables. No model execution."""
import csv,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
Q=ROOT/'experiments/main_evaluation_20261110'
def quantile(values,q):
    v=sorted(values);pos=(len(v)-1)*q;i=int(pos)
    return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(pos-i)
def write(name,rows):
    with (Q/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
state=json.loads((Q/'status.json').read_text());assert state['status']=='COMPLETE' and len(state['runs'])==27
stats=[];paired={};baseline={};count=0;worst=[];groups={}
for scale in ['1000000','5000000','full']:
    path=ROOT/'data/snapshot_resume_20261109_172915/workloads'/scale
    assert json.loads((path/'manifest.json').read_text())['status']=='COMPLETE'
    queries={q['query_id']:q for q in map(json.loads,(path/'main_native.jsonl').read_text(encoding='utf-8').splitlines())}
    assert len(queries)==2000
    for seed in [0,1,2]:
        for model in ['own_resmade','own_made','official_resmade']:
            folder=Q/f'{scale}_{model}_seed{seed}'
            assert json.loads((folder/'status.json').read_text())['status']=='COMPLETE'
            summary=json.loads((folder/'summary.json').read_text());rows=list(csv.DictReader((folder/'per_query.csv').open(encoding='utf-8')));assert len(rows)==6000
            per={};seen=set()
            for row in rows:
                method=row['method'];qid=row['query_id'];key=(method,qid);assert key not in seen;seen.add(key)
                truth=float(row['truth']);est=float(row['estimate']);err=float(row['qerror']);lat=float(row['latency_ms'])
                assert truth==queries[qid]['truth'] and truth>0 and est>=0 and math.isfinite(est) and lat>=0 and math.isfinite(lat)
                expected=max(max(1,truth)/max(1,est),max(1,est)/max(1,truth));assert math.isclose(expected,err,rel_tol=1e-10)
                per.setdefault(method,[]).append((qid,err,lat,est));count+=1
                if method!=model:
                    bkey=(scale,method,qid)
                    if bkey in baseline:assert math.isclose(baseline[bkey],est,rel_tol=1e-12)
                    else:baseline[bkey]=est
                else:
                    paired[(scale,model,seed,qid)]=err
                    q=queries[qid];sel=q['selectivity'];bucket='<=1e-5' if sel<=1e-5 else '<=1e-3' if sel<=1e-3 else '<=1e-2' if sel<=1e-2 else '>1e-2'
                    for grouping,value in [('predicate_count',str(len(q['predicates']))),('selectivity',bucket)]:groups.setdefault((scale,model,seed,grouping,value),[]).append(err)
            for method,values in per.items():
                e=[v[1] for v in values];t=[v[2] for v in values]
                result={'scale':scale,'model':method,'training_seed':seed,'source_model_run':model,'queries':len(e),'qerror_median':statistics.median(e),'qerror_p95':quantile(e,.95),'qerror_p99':quantile(e,.99),'qerror_max':max(e),'latency_median_ms':statistics.median(t)}
                for k in ['qerror_median','qerror_p95','qerror_p99','qerror_max','latency_median_ms']:assert math.isclose(result[k],summary['methods'][method][k],rel_tol=1e-9,abs_tol=1e-9),(folder,k)
                if method==model or (seed==0 and model=='own_resmade'):stats.append(result)
                if method==model:
                    for qid,err,lat,est in sorted(values,key=lambda x:x[1],reverse=True)[:10]:worst.append({'scale':scale,'model':model,'seed':seed,'query_id':qid,'qerror':err,'truth':queries[qid]['truth'],'estimate':est,'predicate_count':len(queries[qid]['predicates']),'predicates':json.dumps(queries[qid]['predicates'],ensure_ascii=False)})
aggregate=[]
for scale in ['1000000','5000000','full']:
    for model in ['own_made','own_resmade','official_resmade','exact_marginal_independence','uniform_sample']:
        rows=[v for v in stats if v['scale']==scale and v['model']==model];a={'scale':scale,'model':model,'independent_training_seeds':len(rows) if model.startswith(('own','official')) else 0}
        for key in ['qerror_median','qerror_p95','qerror_p99','qerror_max','latency_median_ms']:
            v=[x[key] for x in rows];a[key+'_seed_median']=statistics.median(v);a[key+'_seed_min']=min(v);a[key+'_seed_max']=max(v)
        aggregate.append(a)
pair=[]
for scale in ['1000000','5000000','full']:
    for seed in [0,1,2]:
        ratios=[paired[(scale,'own_resmade',seed,q)]/paired[(scale,'official_resmade',seed,q)] for q in queries] # same stable IDs per scale
        pair.append({'scale':scale,'seed':seed,'own_over_official_qerror_ratio_median':statistics.median(ratios),'own_better_fraction':sum(x<1 for x in ratios)/len(ratios),'ratio_p95':quantile(ratios,.95)})
group_rows=[dict(scale=k[0],model=k[1],seed=k[2],grouping=k[3],group=k[4],queries=len(v),qerror_median=statistics.median(v),qerror_p95=quantile(v,.95),qerror_max=max(v)) for k,v in groups.items()]
write('metrics_by_seed.csv',stats);write('metrics_seed_ranges.csv',aggregate);write('paired_resmade.csv',pair);write('worst_queries.csv',worst);write('grouped_metrics.csv',group_rows)
report={'status':'PASS','evaluation_cells':27,'raw_csv_rows_checked':count,'queries_per_cell':2000,'duplicate_baseline_estimates':'identical across repeated runs; deduplicated, not independent replicates','checks':['qerror from truth/estimate','truth joins saved workload','query-method keys unique','all summary quantiles independently recomputed','baseline repeated estimates identical'],'summaries':aggregate}
(ROOT/'evidence/main_evaluation_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','cells':27,'raw_rows':count,'plot_tables':5}))
