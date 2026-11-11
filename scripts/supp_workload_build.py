"""Supplementary workloads: original-value shared queries, predicate sweep, stress.

The generator is fixed before examining model errors. Exact counts use encoded
mask intersections. Ordinary NULL comparison is false; IS NULL is an extension.
"""
import argparse,datetime,json,time,traceback,bisect
from pathlib import Path
import numpy as np

SCALES=['1000000','5000000','full']
SEEDS={'shared':20261120,'sweep':20261121,'stress':20261122}

def comparable(v,column):
    if column=='reg_valid_date' and v is not None:
        return datetime.datetime.fromisoformat(v.replace('Z','+00:00'))
    return v

class Compiler:
    def __init__(self,columns,domains):
        self.columns=columns;self.domains=domains
        self.maps=[{v:j for j,v in enumerate(d)} for d in domains]
        self.keys=[];self.offset=[]
        for c,d in zip(columns,domains):
            off=int(bool(d) and d[0] is None);self.offset.append(off)
            keys=[comparable(x,c) for x in d[off:]]
            assert keys==sorted(keys),'domain not ordered'
            self.keys.append(keys)
    def predicate(self,i,op,value):
        d=self.domains[i];mask=np.zeros(len(d),dtype=bool);off=self.offset[i]
        if op=='IS NULL':
            if off:mask[0]=True
        elif op=='IS NOT NULL':mask[off:]=True
        elif value is not None:
            if op=='=':
                code=self.maps[i].get(value)
                if code is not None:mask[code]=True
            else:
                v=comparable(value,self.columns[i]);keys=self.keys[i]
                if op=='<=':mask[off:off+bisect.bisect_right(keys,v)]=True
                elif op=='>=':mask[off+bisect.bisect_left(keys,v):]=True
                elif op=='<':mask[off:off+bisect.bisect_left(keys,v)]=True
                elif op=='>':mask[off+bisect.bisect_right(keys,v):]=True
                else:raise ValueError(op)
        return {'column_index':i,'column':self.columns[i],'operator':op,'value':value,
                'value_code':self.maps[i].get(value),'allowed_codes':np.flatnonzero(mask).tolist()}
    def compile(self,raw):return [self.predicate(int(p['column_index']),p['operator'],p['value']) for p in raw]

class Oracle:
    def __init__(self,table,domains):
        self.table=table;self.domains=domains
        self.hist=[np.bincount(table[:,i],minlength=len(d)) for i,d in enumerate(domains)]
    def count(self,predicates):
        filters=[]
        for p in predicates:
            i=p['column_index'];mask=np.zeros(len(self.domains[i]),dtype=bool);mask[p['allowed_codes']]=True
            count=int(self.hist[i][mask].sum())
            if not count:return 0
            filters.append((count,i,mask))
        if not filters:return len(self.table)
        if len(filters)==1:return filters[0][0]
        filters.sort(key=lambda x:x[0]);_,i,m=filters[0];ids=np.flatnonzero(m[self.table[:,i]])
        for _,i,m in filters[1:]:
            ids=ids[m[self.table[ids,i]]]
            if not len(ids):return 0
        return int(len(ids))

def raw_anchor(table,compiler,rng,k):
    row=int(rng.randint(len(table)));indices=rng.choice(len(compiler.columns),k,replace=False)
    raw=[]
    for i in indices:
        i=int(i);op=str(rng.choice(['=','<=','>='])) if len(compiler.domains[i])>=10 else '='
        value=compiler.domains[i][int(table[row,i])]
        raw.append({'column_index':i,'operator':op,'value':value})
    return raw,row

def query(qid,workload,raw,compiler,oracle,**extra):
    preds=compiler.compile(raw);truth=oracle.count(preds)
    return dict(query_id=qid,workload=workload,predicates=preds,truth=truth,
                selectivity=truth/len(oracle.table),**extra)

def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();snapshot=Path(a.snapshot);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    meta=json.loads((snapshot/'manifest.json').read_text());assert meta['status']=='COMPLETE'
    status={'status':'RUNNING','snapshot':str(snapshot),'rowsUpdatedAt':meta['rowsUpdatedAt'],
            'seeds':SEEDS,'counts_per_scale':{'scale_shared':1000,'predicate_sweep':1100,'stress':300},
            'shared_policy':'1000 fixed raw queries anchored on full table; retained independent of result count; same values/operators/columns in all scales',
            'sweep_policy':'100 anchored candidates per predicate count 1..11; zero results retained and labeled',
            'stress_policy':'100 guaranteed zero (NULL ordinary comparisons or absent category), 100 IS NULL/NOT NULL, 100 boundary predicates',
            'date_semantics':'chronological ISO timestamps; categories lexical; NULL ordinary comparisons false',
            'main_workload_modified':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scales':{}}
    def save():
        temp=out/'manifest.tmp';temp.write_text(json.dumps(status,indent=2),encoding='utf-8');temp.replace(out/'manifest.json')
        temp=out/'status.tmp';temp.write_text(json.dumps(status,indent=2),encoding='utf-8');temp.replace(out/'status.json')
    save();start=time.time()
    try:
        full=np.load(snapshot/'encoded/full.npy',mmap_mode='r');spec=json.loads((snapshot/'encoded/full_domains.json').read_text());cols=spec['columns'];fc=Compiler(cols,spec['domains'])
        rng=np.random.RandomState(SEEDS['shared']);shared=[raw_anchor(full,fc,rng,int(rng.randint(5,12)))[0] for _ in range(1000)]
        with (out/'shared_raw.jsonl').open('x',encoding='utf-8') as f:
            for i,raw in enumerate(shared):f.write(json.dumps({'query_id':f'scale_shared_{i:05d}','predicates':raw},ensure_ascii=False)+'\n')
        for scale in SCALES:
            table=np.load(snapshot/'encoded'/f'{scale}.npy',mmap_mode='r');spec=json.loads((snapshot/'encoded'/f'{scale}_domains.json').read_text());comp=Compiler(cols,spec['domains']);oracle=Oracle(table,spec['domains']);sd=out/scale;sd.mkdir()
            sc={'status':'RUNNING','rows':len(table),'domain_sizes':[len(x) for x in comp.domains],'workloads':{}};status['scales'][scale]=sc;save()
            def write_set(name,items):
                counts={'rows':0,'zero_results':0,'groups':{}}
                with (sd/(name+'.jsonl')).open('x',encoding='utf-8') as f:
                    for i,(raw,extra) in enumerate(items):
                        q=query(f'{name}_{i:05d}',name,raw,comp,oracle,**extra)
                        if name=='stress' and extra['stress_group']=='guaranteed_zero':assert q['truth']==0
                        f.write(json.dumps(q,ensure_ascii=False)+'\n');counts['rows']+=1;counts['zero_results']+=int(q['truth']==0)
                        group=str(extra.get('predicate_count',extra.get('stress_group','all')));counts['groups'][group]=counts['groups'].get(group,0)+1
                        if (i+1)%50==0:f.flush();sc['active']=name;sc['progress']=counts;save();print(scale,name,i+1,flush=True)
                sc['workloads'][name]=counts;save()
            write_set('scale_shared',[(raw,{'shared_raw_id':f'scale_shared_{i:05d}'}) for i,raw in enumerate(shared)])
            rng=np.random.RandomState(SEEDS['sweep']);sweep=[]
            for k in range(1,12):
                for j in range(100):
                    raw,anchor=raw_anchor(table,comp,rng,k);sweep.append((raw,{'predicate_count':k,'anchor_row':anchor}))
            write_set('predicate_sweep',sweep)
            rng=np.random.RandomState(SEEDS['stress']);stress=[]
            for j in range(100):
                i=j%11
                if j%2==0:raw=[{'column_index':i,'operator':'=','value':None}]
                else:
                    choices=[k for k,c in enumerate(cols) if c!='reg_valid_date'];i=choices[j%len(choices)];v='__ABSENT_SUPPLEMENT_20261110__';assert v not in comp.maps[i];raw=[{'column_index':i,'operator':'=','value':v}]
                stress.append((raw,{'stress_group':'guaranteed_zero','construction':'ordinary_NULL_or_absent_literal'}))
            nullable=[i for i,d in enumerate(comp.domains) if d[0] is None]
            for j in range(100):
                i=(nullable or list(range(11)))[j%len(nullable or list(range(11)))];op='IS NULL' if j%2==0 else 'IS NOT NULL'
                stress.append(([{'column_index':i,'operator':op,'value':None}],{'stress_group':'null_predicate','nullable_domain':i in nullable}))
            for j in range(100):
                i=j%11;non_null=comp.domains[i][comp.offset[i]:];assert non_null
                op=['<','<=','>','>='][j%4];v=non_null[0] if j%2==0 else non_null[-1]
                stress.append(([{'column_index':i,'operator':op,'value':v}],{'stress_group':'boundary','boundary':'minimum' if j%2==0 else 'maximum'}))
            write_set('stress',stress);sc['status']='COMPLETE';save()
        status['status']='COMPLETE'
    except BaseException as e:
        status.update(status='FAILED',error=repr(e));(out/'traceback.txt').write_text(traceback.format_exc());raise
    finally:status.update(elapsed_seconds=time.time()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()

if __name__=='__main__':main()
