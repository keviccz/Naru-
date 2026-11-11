"""Cardinality baselines over a fixed encoded table and explicit predicate masks."""
import numpy as np

def exact_count(table, masks):
    valid=np.ones(len(table),dtype=bool)
    for i,m in enumerate(masks):
        if m is not None: valid &= np.asarray(m,dtype=bool)[table[:,i]]
    return int(valid.sum())

def independence(table, masks):
    if not len(table):return 0.0
    estimate=float(len(table))
    for i,m in enumerate(masks):
        if m is not None:estimate*=np.asarray(m,dtype=bool)[table[:,i]].mean()
    return estimate

class UniformSample:
    def __init__(self, table, rows, seed=0):
        if rows<=0:raise ValueError('sample size must be positive')
        self.population=len(table)
        ids=np.random.default_rng(seed).choice(len(table),min(rows,len(table)),replace=False)
        self.table=table[ids].copy()
    def estimate(self,masks):
        return 0.0 if not len(self.table) else self.population*exact_count(self.table,masks)/len(self.table)

def qerror(truth,estimate):
    a=np.maximum(1,np.asarray(truth,dtype=float));b=np.maximum(1,np.asarray(estimate,dtype=float))
    if np.any(~np.isfinite(b)) or np.any(np.asarray(estimate)<0):raise ValueError('invalid estimate')
    return np.maximum(a/b,b/a)
