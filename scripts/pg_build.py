import json,subprocess,time,tarfile,os
from pathlib import Path
r=(Path(__file__).resolve().parents[1]/'experiments/pg_supplement_20261110');r.mkdir(exist_ok=True)
history=r/'build_attempts';history.mkdir(exist_ok=True)
for fn in ['build_status.json','configure.log','make.log','install.log']:
 p=r/fn
 if p.exists():__import__('shutil').copy2(p,history/(str(int(time.time()))+'_'+fn))
s={'status':'BUILDING','started':time.time(),'version':'17.6','prefix':str(r/'install'),'source':'https://ftp.postgresql.org/pub/source/v17.6/postgresql-17.6.tar.bz2'}
def save():(r/'build_status.json').write_text(json.dumps(s,indent=2))
save()
try:
 if not (r/'postgresql-17.6').exists():
  with tarfile.open(r/'postgresql-17.6.tar.bz2') as t:
   for m in t.getmembers():
    if not (r/m.name).resolve().is_relative_to(r.resolve()): raise RuntimeError('unsafe archive')
   t.extractall(r)
 src=r/'postgresql-17.6'
 tools=r/'build_tools';tools.mkdir(exist_ok=True)
 for pkg in ['bison','flex']:subprocess.run(['/usr/bin/dpkg-deb','-x',str(r/(pkg+'.deb')),str(tools)],check=True)
 env={'PATH':str(tools/'usr/bin')+':/usr/bin:/bin','MAKELEVEL':'0','BISON_PKGDATADIR':str(tools/'usr/share/bison'),'HOME':os.environ.get('HOME','/root'),'LANG':'C','LC_ALL':'C'}
 for name,cmd in [('configure',['./configure','--prefix='+str(r/'install'),'--without-readline','--without-zlib','--without-icu','--without-lz4','--without-zstd']),('make',['make','-j4']),('install',['make','install'])]:
  s['phase']=name;save()
  with (r/(name+'.log')).open('w') as f:subprocess.run(cmd,cwd=src,stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
 s['status']='COMPLETE'
except Exception as e:s.update(status='FAILED',error=repr(e));raise
finally:s['ended']=time.time();save()
