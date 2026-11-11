import json,math
from pathlib import Path
root=Path(__file__).resolve().parents[1]
queue=root/'experiments/main_matrix_20261109';state=json.loads((queue/'status.json').read_text())
expected={(scale,model,seed) for scale in ['1000000','5000000','full'] for model in ['own_resmade','own_made','official_resmade'] for seed in [0,1,2]}
seen=set();rows=[]
for cell in state['runs']:
 if cell['status']!='COMPLETE':continue
 key=(cell['scale'],cell['model'],cell['seed']);assert key in expected and key not in seen;seen.add(key)
 path=root/cell['path'].split('/DSC5003_Naru_20261109/',1)[1]
 cfg=json.loads((path/'config.json').read_text());status=json.loads((path/'status.json').read_text());res=json.loads((path/'training_result.json').read_text());epochs=[json.loads(x) for x in (path/'epoch_metrics.jsonl').read_text().splitlines()]
 assert status['status']=='COMPLETE' and status['exit_code']==0
 assert len(epochs)==20 and [e['epoch'] for e in epochs]==list(range(1,21))
 for k,v in [('model',key[1]),('seed',key[2]),('epochs',20),('batch',2048),('width',256),('layers',5),('warmups',8000),('physical_gpu_index',0)]:assert cfg[k]==v,(key,k,cfg[k])
 assert res['rows']==(12540479 if key[0]=='full' else int(key[0]))
 assert res['training_epochs']==20 and math.isfinite(res['final_full_table_nll_nats'])
 assert all(math.isfinite(e['mean_training_nll_nats']) and e['rows']==res['rows'] for e in epochs)
 rows.append({'scale':key[0],'model':key[1],'seed':key[2],'final_nll':res['final_full_table_nll_nats'],'elapsed_seconds':status['elapsed_seconds'],'peak_gpu_bytes':res['peak_gpu_allocated_bytes'],'checkpoint_bytes':res['checkpoint_bytes']})
report={'status':'PASS_COMPLETE' if seen==expected and state['status']=='COMPLETE' else 'PASS_PARTIAL','completed':len(seen),'expected':27,'missing':sorted(expected-seen),'runs':rows,'scope':'training records only; held-out evaluation not verified'}
(root/'evidence/main_matrix_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({'status':report['status'],'completed':len(seen),'expected':27}))
