import os,json,subprocess,time,shutil,traceback
from pathlib import Path
r=Path('/dongqishi01/Zhangdongjian/DSC5003_Naru_20261109');q=r/'experiments/main_matrix_20261109';out=r/'experiments/main_evaluation_20261110';out.mkdir(exist_ok=False)
state={'status':'RUNNING','started':time.time(),'runs':[],'sampling_budget':2000,'scope':'main_native','runtime':'shared unoptimized sampler for architecture accuracy; not official optimized latency'}
def save():
 p=out/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(out/'status.json')
save()
try:
 assert json.loads((r/'experiments/official_dev_shared_sampler_retry1_20261110/status.json').read_text())['status']=='COMPLETE'
 code=out/'code';code.mkdir();(code/'scripts').mkdir();(code/'src').mkdir();(code/'vendor/naru').mkdir(parents=True)
 for f in ['scripts/evaluate_matrix_model.py','src/models.py','src/baselines.py','vendor/naru/made.py']:shutil.copy2(r/f,code/f)
 cells=json.loads((q/'status.json').read_text())['runs'];assert len(cells)==27
 for cell in cells:
  name=cell['scale']+'_'+cell['model']+'_seed'+str(cell['seed']);target=out/name
  record={'name':name,'path':str(target),'status':'PENDING'};state['runs'].append(record);state['active']=name;save()
  idle=0
  while idle<3:
   gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();uid,mem,util=[v.strip() for v in gpu.split(',')]
   procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
   busy=any(x.split(',')[0].strip()==uid for x in procs.splitlines()) or int(mem)>1000 or int(util)>10
   idle=0 if busy else idle+1;state.update(status='WAITING_GPU',gpu=gpu);save()
   if idle<3:time.sleep(5)
  record['status']='RUNNING';state['status']='RUNNING';save()
  cmd=['/opt/conda/envs/R4.2/bin/python',str(code/'scripts/evaluate_matrix_model.py'),'--run',cell['path'],'--queries',str(r/'data/snapshot_resume_20261109_172915/workloads'/cell['scale']/'main_native.jsonl'),'--output',str(target),'--scope','main_native']
  with (out/(name+'.log')).open('x') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'))
  record['exit_code']=p.returncode;record['status']=json.loads((target/'status.json').read_text())['status'];save()
  if p.returncode or record['status']!='COMPLETE':raise RuntimeError('Failed '+name)
 state['status']='COMPLETE'
except BaseException as e:state.update(status='FAILED',error=repr(e));(out/'traceback.txt').write_text(traceback.format_exc());raise
finally:state['updated']=time.time();save()
