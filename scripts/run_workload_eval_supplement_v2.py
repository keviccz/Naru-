import os,json,subprocess,time,traceback,shutil
from pathlib import Path
r=Path('/dongqishi01/Zhangdongjian/DSC5003_Naru_20261109');out=r/'experiments/workload_eval_supplement_v2_20261110';out.mkdir(exist_ok=False)
state={'status':'WAITING_PREREQUISITES','runs':[],'workloads':['scale_shared','predicate_sweep','stress_v2'],'checkpoint_cells':27,'samples':2000,'mc_seed':0}
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
  wait=r/'experiments/gpu_supplement_20261110/status.json'
  load=r/'experiments/workload_supplement_v2_20261110/status.json'
  if wait.exists() and load.exists():
   a=json.loads(wait.read_text())['status'];b=json.loads(load.read_text())['status']
   if a=='FAILED' or b=='FAILED':raise RuntimeError('Prerequisite failed')
   if a=='COMPLETE_BUDGET_SIZE' and b=='COMPLETE':break
  time.sleep(30)
 cells=json.loads((r/'experiments/main_matrix_20261109/status.json').read_text())['runs']
 for c in cells:
  for scope in ['scale_shared','predicate_sweep','stress_v2']:
   evaluate(f"{c['scale']}_{c['model']}_seed{c['seed']}_{scope}",Path(c['path']),r/('experiments/workload_supplement_v2_20261110' if scope=='stress_v2' else 'experiments/workload_supplement_20261110')/c['scale']/(scope+'.jsonl'),scope)
 state['status']='COMPLETE'
except BaseException as e:state.update(status='FAILED',error=repr(e));(out/'traceback.txt').write_text(traceback.format_exc());raise
finally:save()
