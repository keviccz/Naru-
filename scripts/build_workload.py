"""Frozen official-style row-anchored queries and independently computed truth.

Only dev and main_native are implemented. This is not a byte-for-byte upstream
generator: operators remain attached to their actual sampled columns, including
the all-column case (upstream eval_model.py has a special reordered return).
"""
import argparse
import datetime
import json
from pathlib import Path
import time
import numpy as np


def allowed_codes(domain, op, code):
    """Domains are ordered by the encoder; NULL never satisfies comparison."""
    if op not in ('=', '<=', '>='):
        raise ValueError('unsupported ordinary comparison')
    if not 0 <= code < len(domain):
        raise ValueError('threshold code outside domain')
    mask = np.zeros(len(domain), dtype=bool)
    if domain[code] is None:
        return mask
    if op == '=': mask[code] = True
    elif op == '<=': mask[:code+1] = True
    else: mask[code:] = True
    for i, value in enumerate(domain):
        if value is None: mask[i] = False
    return mask


def signature(predicates):
    return tuple(sorted((p['column_index'], p['operator'], p['value_code']) for p in predicates))


class ExactOracle:
    """Marginal frequencies order filters, but do not approximate truth."""
    def __init__(self, table, domains):
        if table.ndim != 2 or table.shape[1] != len(domains) or len(table) == 0:
            raise ValueError('nonempty 2D encoded table with one domain per column required')
        if not np.issubdtype(table.dtype, np.integer):
            raise TypeError('integer encoded table required')
        self.table, self.domains = table, domains
        self.hist = []
        for i, d in enumerate(domains):
            if not d or table[:, i].min() < 0 or table[:, i].max() >= len(d):
                raise ValueError('invalid encoded column')
            self.hist.append(np.bincount(table[:, i], minlength=len(d)))

    def count(self, predicates):
        filters=[]
        for p in predicates:
            i=p['column_index']
            mask=allowed_codes(self.domains[i],p['operator'],p['value_code'])
            matched=int(self.hist[i][mask].sum())
            if matched == 0:return 0
            filters.append((matched,i,mask))
        if not filters:return len(self.table)
        filters.sort(key=lambda x:x[0])
        _,i,mask=filters[0]
        ids=np.flatnonzero(mask[self.table[:,i]])
        for _,i,mask in filters[1:]:
            ids=ids[mask[self.table[ids,i]]]
            if not len(ids):return 0
        return int(len(ids))


def generate(table, columns, domains, count, seed, workload, oracle=None,
             min_predicates=5, max_predicates=11, excluded=None, on_accept=None):
    if count < 0 or not 1 <= min_predicates <= max_predicates <= len(columns):
        raise ValueError('invalid count or predicate range')
    rng=np.random.RandomState(seed)
    oracle=oracle or ExactOracle(table,domains)
    excluded=set() if excluded is None else set(excluded)
    queries=[];attempts=zero=overlap=0
    started=time.perf_counter()
    max_attempts=max(1000,count*1000)
    while len(queries)<count:
        attempts+=1
        if attempts>max_attempts:
            raise RuntimeError('attempt limit reached; insufficient valid queries')
        k=int(rng.randint(min_predicates,max_predicates+1))
        anchor=int(rng.randint(len(table)))
        indices=rng.choice(len(columns),size=k,replace=False)
        ops=rng.choice(['<=','>=','='],size=k)
        predicates=[]
        for i,op in zip(indices,ops):
            i=int(i);code=int(table[anchor,i])
            if len(domains[i])<10:op='='
            allowed=allowed_codes(domains[i],str(op),code)
            predicates.append({'column_index':i,'column':columns[i],
                               'operator':str(op),'value':domains[i][code],
                               'value_code':code,'allowed_codes':np.flatnonzero(allowed).tolist()})
        if signature(predicates) in excluded:
            overlap+=1;continue
        truth=oracle.count(predicates)
        if truth==0:
            zero+=1;continue
        queries.append({'query_id':f'{workload}_{len(queries):05d}',
                        'workload':workload,'anchor_row':anchor,
                        'predicates':predicates,'truth':truth,
                        'selectivity':truth/len(table)})
        if on_accept is not None:
            on_accept(queries[-1],{'retained':len(queries),'candidates':attempts,
                                  'zero_result_skipped':zero,'cross_split_duplicates_skipped':overlap})
    return queries,{'seed':seed,'requested':count,'retained':len(queries),
                    'candidates':attempts,'zero_result_skipped':zero,
                    'cross_split_duplicates_skipped':overlap,
                    'within_split_duplicates_retained':count-len({signature(q['predicates']) for q in queries}),
                    'elapsed_seconds':time.perf_counter()-started}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('snapshot');p.add_argument('--scale',required=True,choices=['1000000','5000000','full'])
    p.add_argument('--main-count',type=int,default=2000);p.add_argument('--dev-count',type=int,default=200)
    p.add_argument('--main-seed',type=int,default=20261110);p.add_argument('--dev-seed',type=int,default=20261111)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.main_seed==a.dev_seed:raise ValueError('dev/main seeds must differ')
    root=Path(a.snapshot);meta=json.loads((root/'manifest.json').read_text())
    if meta['status']!='COMPLETE':raise ValueError('snapshot not complete')
    table_path=root/'encoded'/f'{a.scale}.npy'
    domains_path=root/'encoded'/f'{a.scale}_domains.json'
    table=np.load(table_path,mmap_mode='r');spec=json.loads(domains_path.read_text())
    if spec['columns']!=meta['columns']:raise ValueError('snapshot/encoding column mismatch')
    expected=meta['expected_rows'] if a.scale=='full' else int(a.scale)
    if len(table)!=expected:raise ValueError('wrong scale cardinality')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    receipt={'status':'RUNNING','snapshot_id':root.name,'snapshot_rows_updated_at':meta['rowsUpdatedAt'],
             'source':meta['source'],'scale':a.scale,'rows':len(table),'table_path':str(table_path),
             'domains_path':str(domains_path),'columns':spec['columns'],
             'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'implemented':['dev','main_native'],'not_implemented':['scale_shared','predicate_sweep','stress'],
             'generator':'official-style row anchor; 5-11 columns; domain>=10 permits range; operators stay with sampled columns',
             'null_semantics':'ordinary comparisons with NULL are false; zero-result queries skipped',
             'duplicate_policy':'within-split duplicates retained as sampled; cross-split duplicates excluded',
             'exact_oracle':'full-table exact filter intersection; marginal frequencies only choose evaluation order'}
    def save(): (out/'manifest.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    save()
    try:
        oracle=ExactOracle(table,spec['domains'])
        def run_split(name,count,seed,excluded=None):
            with (out/f'{name}.jsonl').open('x',encoding='utf-8') as f:
                def accepted(q,stats):
                    f.write(json.dumps(q,ensure_ascii=False)+'\n')
                    if stats['retained']%50==0 or stats['retained']==count:
                        f.flush();receipt['active_workload']=name;receipt['progress']=stats;save()
                        print(json.dumps({'workload':name,**stats}),flush=True)
                return generate(table,spec['columns'],spec['domains'],count,seed,name,
                                oracle,excluded=excluded,on_accept=accepted)
        dev,ds=run_split('dev',a.dev_count,a.dev_seed)
        excluded={signature(q['predicates']) for q in dev}
        main,ms=run_split('main_native',a.main_count,a.main_seed,excluded)
        receipt.update(status='COMPLETE',workloads={'dev':ds,'main_native':ms})
    except Exception as e:
        receipt.update(status='FAILED',error=repr(e));raise
    finally:
        receipt['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
    print(json.dumps(receipt))

if __name__=='__main__':main()
