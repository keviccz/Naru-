"""Frozen-checkpoint GPU0 evaluation for paired joint-zero/near-positive tests."""
import json,os,shutil,subprocess,time,traceback
from pathlib import Path
r=Path(__file__).resolve().parents[1];out=r/'experiments/joint_zero_model_eval_20261111';out.mkdir(exist_ok=False)
state={'status':'WAITING_WORKLOAD','runs':[],'scope':'joint_zero_nearzero','samples':2000,'mc_seeds':[0,1,2],'scales':['1000000','5000000','full'],'models':['own_made','own_resmade','official_resmade'],'training_seeds':[0,1,2],'expected_cells':81,'expected_queries_per_cell':200,'training_performed':False}
code=out/'code'
for rel in ['src/models.py','src/baselines.py','vendor/naru/made.py','scripts/evaluate_supplement.py']:
 p=code/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(r/rel,p)
def save():
 p=out/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(out/'status.json')
def idle_gpu():
 idle=0
 while idle<3:
  line=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();uid,mem,util=map(str.strip,line.split(','));apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
  busy=any(x.split(',')[0].strip()==uid for x in apps.splitlines()) or int(mem)>1000 or int(util)>10
  idle=0 if busy else idle+1;state['gpu_check']=line;save()
  if idle<3:time.sleep(2)
save()
try:
 ready=r/'experiments/joint_zero_20261111/workloads/status.json'
 while True:
  if ready.exists():
   s=json.loads(ready.read_text())['status']
   if s=='FAILED':raise RuntimeError('Workload generation failed')
   if s=='COMPLETE':break
  time.sleep(15)
 cells=json.loads((r/'experiments/main_matrix_20261109/status.json').read_text())['runs']
 assert len(cells)==27 and len({(c['scale'],c['model'],c['seed']) for c in cells})==27
 for c in cells:
  qpath=r/'experiments/joint_zero_20261111/workloads'/c['scale']/'joint_zero_nearzero.jsonl';qs=[json.loads(x) for x in qpath.read_text().splitlines()]
  assert len(qs)==200 and len({q['query_id'] for q in qs})==200 and sum(q['truth']==0 for q in qs)==100 and all(p['allowed_codes'] for q in qs for p in q['predicates'])
  for mc in [0,1,2]:
   idle_gpu();name=f"{c['scale']}_{c['model']}_seed{c['seed']}_mc{mc}";target=out/name;cell={'name':name,'scale':c['scale'],'model':c['model'],'training_seed':c['seed'],'mc_seed':mc,'path':str(target),'status':'RUNNING'};state['runs'].append(cell);state.update(status='RUNNING',active=name);save()
   cmd=['/opt/conda/envs/R4.2/bin/python',str(code/'scripts/evaluate_supplement.py'),'--run',c['path'],'--queries',str(qpath),'--scope','joint_zero_nearzero','--samples','2000','--mc-seed',str(mc),'--output',str(target),'--skip-baselines']
   with (out/(name+'.log')).open('x') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'))
   status=target/'status.json';child=json.loads(status.read_text()) if status.exists() else {'status':'FAILED','error':'Child status missing'}
   cell.update(status=child['status'],exit_code=p.returncode);save()
   if p.returncode or cell['status']!='COMPLETE':raise RuntimeError('Failed '+name+': '+str(child.get('error')))
 state['status']='COMPLETE'
except BaseException as exc:
 state.update(status='FAILED',error=repr(exc));(out/'traceback.txt').write_text(traceback.format_exc());raise
finally:save()
