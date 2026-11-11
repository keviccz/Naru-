import os,subprocess,json
os.environ['CUDA_VISIBLE_DEVICES']='0'
p=subprocess.run(['/opt/conda/envs/R4.2/bin/python','/dongqishi01/Zhangdongjian/DSC5003_Naru_20261109/tests/check_official_alignment.py'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),capture_output=True,text=True,timeout=90)
print(json.dumps({'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}))
if p.returncode:raise RuntimeError('baseline checks failed')


