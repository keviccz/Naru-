"""Frozen 27-cell training matrix; sequential GPU0, no tuning or automatic retry."""
import json,os,subprocess,shutil,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
snapshot=ROOT/'data/snapshot_resume_20261109_172915'
queue=ROOT/'experiments/main_matrix_20261109';queue.mkdir(exist_ok=False)
state={'status':'PREPARING','started':time.time(),'runs':[],'gpu':0,'evaluation_status':'NOT_RUN','policy':'fixed 20 epochs; seeds 0/1/2; no retries or best-seed selection'}
def save():
 t=queue/'status.tmp';t.write_text(json.dumps(state,indent=2));t.replace(queue/'status.json')
save()
try:
 for scale in ['1000000','5000000','full']:
  assert json.loads((snapshot/'workloads'/scale/'manifest.json').read_text())['status']=='COMPLETE'
 assert json.loads((snapshot/'encoded/real_validation.json').read_text())['status']=='PASS'
 code=queue/'code'
 for f in ['scripts/train_model.py','src/models.py','vendor/naru/made.py']:
  dst=code/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dst)
 for scale in ['1000000','5000000','full']:
  for seed in [0,1,2]:
   for model in ['own_resmade','own_made','official_resmade']:
    reuse=scale=='1000000' and seed==0 and model=='own_resmade'
    out=ROOT/'experiments/pilot_1m_own_resmade_seed0' if reuse else queue/f'{scale}_{model}_seed{seed}'
    cell={'scale':scale,'model':model,'seed':seed,'path':str(out),'status':'PENDING','reused_pilot':reuse};state['runs'].append(cell)
    if reuse:
     c=json.loads((out/'config.json').read_text());st=json.loads((out/'status.json').read_text())
     for key,value in [('epochs',20),('width',256),('layers',5),('batch',2048),('warmups',8000),('seed',0),('model',model)]:assert c[key]==value,(key,c[key])
     assert st['status']=='COMPLETE'
     cell.update(status='COMPLETE',result=st['result']);save();continue
    check=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    used,util=map(int,check.split(','))
    if used>1000 or util>10:raise RuntimeError('GPU0 busy; preserving queue without interfering: '+check)
    cmd=['/opt/conda/envs/R4.2/bin/python',str(code/'scripts/train_model.py'),'--encoded-dir',str(snapshot/'encoded'),'--scale',scale,'--model',model,'--seed',str(seed),'--epochs','20','--width','256','--layers','5','--batch','2048','--warmups','8000','--run-dir',str(out)]
    cell.update(status='RUNNING',command=cmd,started=time.time());state.update(status='RUNNING',active=str(out));save()
    with (queue/(out.name+'.log')).open('x') as log:
     p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4'))
    cell.update(exit_code=p.returncode,ended=time.time());st=json.loads((out/'status.json').read_text());cell['status']=st['status'];save()
    if p.returncode or st['status']!='COMPLETE':raise RuntimeError('Matrix cell failed: '+str(out))
    cell['result']=st['result'];save()
 state.update(status='COMPLETE',completed_cells=len(state['runs']))
except BaseException as e:
 state.update(status='FAILED',error=repr(e));(queue/'traceback.txt').write_text(traceback.format_exc());raise
finally:state['updated']=time.time();save()
