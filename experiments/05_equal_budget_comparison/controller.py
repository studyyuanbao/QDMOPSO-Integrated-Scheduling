from pathlib import Path
import subprocess,sys,time,json,hashlib,datetime,os,traceback
T=Path(__file__).resolve().parent;S=T/'controller-status.json'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(x):
 p=S.with_suffix('.tmp');p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');p.replace(S)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
start=time.perf_counter();jobs=json.loads((T/'jobs.json').read_text(encoding='utf-8'));completed=[];manifest=json.loads((T/'runtime-manifest.json').read_text(encoding='utf-8'));save({'status':'starting','pid':os.getpid(),'total':630,'completed':0,'utc':utc()})
try:
 for job in jobs:
  for name,d in manifest.items():assert digest(T/name)==d,'Frozen runtime changed: '+name
  folder=T/'runs'/job['id'];summary=folder/'summary.json'
  if summary.exists():
   row=json.loads(summary.read_text(encoding='utf-8'));assert row['status']=='PASS' and row['nfe']==5000;completed.append(job['id']);continue
  assert not folder.exists(),'Incomplete attempt exists; inspect before retry: '+job['id']
  if (T/'STOP_REQUESTED').exists():save({'status':'stopped_by_request','completed':len(completed),'next_job':job['id'],'utc':utc()});sys.exit(0)
  save({'status':'running','pid':os.getpid(),'current':job,'completed':len(completed),'total':630,'elapsed_seconds':time.perf_counter()-start,'utc':utc()})
  with (T/'worker-console.log').open('a',encoding='utf-8') as f:
   f.write('\n'+utc()+' '+job['id']+' START\n');f.flush();p=subprocess.Popen([sys.executable,str(T/'run_job.py'),'--job',str(job['order'])],stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),cwd=T)
   save({'status':'running','pid':os.getpid(),'worker_pid':p.pid,'current':job,'completed':len(completed),'total':630,'elapsed_seconds':time.perf_counter()-start,'utc':utc()})
   try:code=p.wait(timeout=3660)
   except subprocess.TimeoutExpired:p.kill();p.wait();raise RuntimeError('Worker hard-timeout: '+job['id'])
  assert code==0,'Worker error: '+job['id'];row=json.loads(summary.read_text(encoding='utf-8'));assert row['status']=='PASS' and row['nfe']==5000
  completed.append(job['id']);print(utc(),len(completed),'/630',job['id'],'seconds',round(row['search_seconds_including_checks_and_recording'],2),flush=True)
  save({'status':'running','pid':os.getpid(),'completed':len(completed),'total':630,'last_completed':job['id'],'elapsed_seconds':time.perf_counter()-start,'utc':utc()})
 save({'status':'summarizing','pid':os.getpid(),'completed':630,'total':630,'elapsed_seconds':time.perf_counter()-start,'utc':utc()})
 subprocess.run([sys.executable,str(T/'summarize_progress.py')],check=True,cwd=T)
 save({'status':'complete','completed':630,'total':630,'elapsed_seconds':time.perf_counter()-start,'utc':utc(),'formal_six_instance_comparison_started':True})
 print('FORMAL RUNS COMPLETE; OFFLINE METRICS AND STATISTICS PENDING',flush=True)
except BaseException:
 (T/'controller-failure.txt').write_text(traceback.format_exc(),encoding='utf-8');save({'status':'failed','pid':os.getpid(),'completed':len(completed),'total':630,'elapsed_seconds':time.perf_counter()-start,'utc':utc(),'error':traceback.format_exc()});raise
