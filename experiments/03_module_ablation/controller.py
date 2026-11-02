from pathlib import Path
import os,sys,subprocess,json,time,traceback,hashlib,zipfile
O=Path(__file__).resolve().parent;B=O.parents[1];env=os.environ.copy();env['PYTHONUTF8']='1';env['MPLCONFIGDIR']=str(B/'work/mpl-r15');start=time.time();status={'status':'running','started_unix':start,'serial_cases':['S1','M2','L3'],'python':sys.executable}
def save():(O/'controller-status.json').write_text(json.dumps(status,indent=2),encoding='utf8')
save()
try:
 for case in ['S1','M2','L3']:
  status['current_case']=case;save()
  with (O/f'{case}-formal.log').open('w',encoding='utf8') as f:p=subprocess.run([sys.executable,str(O/'run_ablation.py'),'--case',case],env=env,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
  if p.returncode:raise RuntimeError(f'{case} exit {p.returncode}; see {case}-formal.log')
 status['status']='summarizing';save();subprocess.run([sys.executable,str(O/'summarize_ablation.py')],env=env,check=True)
 # Save full fingerprints and a compressed archive only after all validations succeed.
 hashes={str(p.relative_to(O)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in O.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name not in ['file-checksums.json','controller-status.json']}
 (O/'file-checksums.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
 dest=B/'outputs/R1-5模块消融完整记录.zip'
 with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for p in O.rglob('*'):
   if p.is_file() and '__pycache__' not in str(p):z.write(p,str(p.relative_to(O)))
 import shutil
 shutil.copyfile(O/'结果说明.md',B/'outputs/R1-5模块消融结果说明.md')
 status['status']='complete';status['seconds_total']=time.time()-start;status['archive']=str(dest);save()
except BaseException:
 status['status']='failed';status['error']=traceback.format_exc();save();raise
