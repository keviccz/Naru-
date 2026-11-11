import os,json,subprocess,shutil,time,traceback
from pathlib import Path
r=Path('/dongqishi01/Zhangdongjian/DSC5003_Naru_20261109');out=r/'experiments/size_supplement_20261110';out.mkdir(exist_ok=False)
state={'status':'RUNNING','runs':[],'fixed_widths':[128,512],'reference_width':256,'seeds':[0,1,2]}
def save():(out/'status.json').write_text(json.dumps(state,indent=2))
save()
try:
 for width in [128,512]:
  for seed in [0,1,2]:
   idle=0
   while idle<3:
    gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();uid,mem,util=map(str.strip,gpu.split(','));procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    busy=any(x.split(',')[0].strip()==uid for x in procs.splitlines()) or int(mem)>1000 or int(util)>10;idle=0 if busy else idle+1
    if idle<3:time.sleep(5)
   name=f'width{width}_seed{seed}';target=out/name;cell={'name':name,'status':'RUNNING','path':str(target)};state['runs'].append(cell);state['active']=name;save()
   cmd=['/opt/conda/envs/R4.2/bin/python',str(r/'scripts/train_model.py'),'--encoded-dir',str(r/'data/snapshot_resume_20261109_172915/encoded'),'--scale','1000000','--model','own_resmade','--seed',str(seed),'--width',str(width),'--run-dir',str(target)]
   with (out/(name+'.log')).open('x') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'))
   cell['exit_code']=p.returncode;cell['status']=json.loads((target/'status.json').read_text())['status'];save()
   if p.returncode:raise RuntimeError('Failed '+name)
 state['status']='COMPLETE'
except BaseException as e:state.update(status='FAILED',error=repr(e));raise
finally:save()
