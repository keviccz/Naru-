import os,json,subprocess,time,traceback,shutil
from pathlib import Path
r=Path(__file__).resolve().parents[1];out=r/'experiments/gpu_supplement_20261110';out.mkdir(exist_ok=False)
state={'status':'WAITING_SIZE','runs':[],'budget_grid':{'scales':['1000000','5000000','full'],'training_seeds':[0,1,2],'mc_seeds':[0,1,2],'samples':[100,500,2000,8000],'model':'own_resmade'}}
def save():
 p=out/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(out/'status.json')
def idle_gpu():
 idle=0
 while idle<3:
  gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();uid,mem,util=map(str.strip,gpu.split(','));procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
  busy=any(x.split(',')[0].strip()==uid for x in procs.splitlines()) or int(mem)>1000 or int(util)>10;idle=0 if busy else idle+1
  state['gpu_check']=gpu;save()
  if idle<3:time.sleep(5)
def evaluate(name,run,queries,scope,samples=2000,mc=0):
 idle_gpu();target=out/name;cell={'name':name,'path':str(target),'status':'RUNNING'};state['runs'].append(cell);state.update(status='RUNNING',active=name);save()
 cmd=['/opt/conda/envs/R4.2/bin/python',str(r/'scripts/evaluate_supplement.py'),'--run',str(run),'--queries',str(queries),'--scope',scope,'--samples',str(samples),'--mc-seed',str(mc),'--output',str(target),'--skip-baselines']
 with (out/(name+'.log')).open('x') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'))
 cell.update(exit_code=p.returncode,status=json.loads((target/'status.json').read_text())['status']);save()
 if p.returncode:raise RuntimeError('Failed '+name)
save()
try:
 while True:
  p=r/'experiments/size_supplement_20261110/status.json'
  if p.exists():
   st=json.loads(p.read_text())['status']
   if st=='FAILED':raise RuntimeError('Size training failed')
   if st=='COMPLETE':break
  time.sleep(20)
 cells=json.loads((r/'experiments/main_matrix_20261109/status.json').read_text())['runs']
 for c in json.loads((r/'experiments/size_supplement_20261110/status.json').read_text())['runs']:
  evaluate('size_'+c['name'],Path(c['path']),r/'data/snapshot_resume_20261109_172915/workloads/1000000/main_native.jsonl','main_native')
 for scale in ['1000000','5000000','full']:
  for seed in [0,1,2]:
   run=next(Path(c['path']) for c in cells if c['model']=='own_resmade' and c['scale']==scale and c['seed']==seed)
   for samples in [100,500,2000,8000]:
    for mc in [0,1,2]:evaluate(f'budget_{scale}_seed{seed}_s{samples}_mc{mc}',run,r/'data/snapshot_resume_20261109_172915/workloads'/scale/'main_native.jsonl','main_native',samples,mc)
 state['status']='COMPLETE_BUDGET_SIZE'
except BaseException as e:state.update(status='FAILED',error=repr(e));(out/'traceback.txt').write_text(traceback.format_exc());raise
finally:save()
