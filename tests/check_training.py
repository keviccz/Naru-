"""GPU0 synthetic fit correctness experiment; not a DMV benchmark."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.models import CategoricalMADE,progressive_sample
torch.set_num_threads(4)
assert torch.cuda.is_available()
results=[]
for seed in (0,1,2):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    x=torch.tensor([[0,0],[1,1]],device='cuda').repeat(128,1)
    model=CategoricalMADE([2,2],hidden=32,layers=3,residual=True,direct_io=True).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=.01)
    initial=model.nll(x).mean().item();start=time.time()
    for step in range(300):
        opt.zero_grad(set_to_none=True);loss=model.nll(x).mean();loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
    torch.cuda.synchronize();model.eval();final=model.nll(x).mean().item()
    estimate=progressive_sample(model,[torch.tensor([True,False]),torch.tensor([True,False])],samples=2000,seed=seed)
    assert final<.72 and final<initial
    assert abs(estimate-.5)<.02
    results.append({'seed':seed,'steps':300,'initial_nll':initial,'final_nll':final,'true_selectivity':.5,'independence_selectivity':.25,'model_selectivity':estimate,'seconds':time.time()-start})
# Training-mode masking has finite backward gradients; no parameter update here.
m=CategoricalMADE([2,3,2],hidden=32,layers=3,residual=True,direct_io=True,column_masking=True).cuda().train()
x=torch.stack([torch.randint(n,(128,),device='cuda') for n in [2,3,2]],dim=1)
m.nll(x).mean().backward()
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.unknown)
print(json.dumps({'status':'PASS','kind':'synthetic_correctness_only','gpu':'physical GPU0','results':results,'masking_backward':'finite_nonzero_unknown_gradient'}))
