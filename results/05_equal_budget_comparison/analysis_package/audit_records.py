from pathlib import Path
import json,gzip,hashlib,time,collections
T=Path(__file__).resolve().parent;A=T/'analysis';A.mkdir(exist_ok=True);jobs=json.loads((T/'jobs.json').read_text(encoding='utf-8'));start=time.time();rows=[]
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,d in json.loads((T/'code-hashes.json').read_text(encoding='utf-8')).items():assert h(T/'code'/name)==d
for n,j in enumerate(jobs,1):
 f=T/'runs'/j['id'];s=json.loads((f/'summary.json').read_text(encoding='utf-8'));assert s['status']=='PASS' and s['nfe']==5000
 actual=json.loads((f/'job.json').read_text(encoding='utf-8'));assert actual==j
 if j['previous_job']:
  assert s['q_input_sha256']==h(T/'runs'/j['previous_job']/'q-final.pkl')
 if j['algorithm']=='QDMOPSO':assert s['initial_q_loading_verified']
 sources=collections.Counter();ids=[];nfe=0
 with gzip.open(f/'evaluations.jsonl.gz','rt',encoding='utf-8') as stream:
  for line in stream:
   r=json.loads(line);nfe+=1;assert r['nfe']==nfe;sources[r['source']]+=1;ids.append(r['schedule_id']);assert r['cost_components']['max_error']<1e-7
 assert nfe==5000 and dict(sources)==s['evaluations_by_source']
 # Fully consume schedules to verify gzip CRC and line count; individual JSON checked during search.
 count=0
 with gzip.open(f/'schedules.jsonl.gz','rt',encoding='utf-8') as stream:
  for line in stream:
   count+=1
   if count in (1,5000):
    r=json.loads(line);assert r['nfe']==count and r['schedule_id']==ids[count-1]
 assert count==5000
 for streamname in ['states','decisions','updates','archive-events']:
  with gzip.open(f/(streamname+'.jsonl.gz'),'rt',encoding='utf-8') as stream:
   for line in stream:json.loads(line)
 rows.append({'job':j['id'],'nfe':nfe,'schedule_lines':count,'status':'PASS'})
 (A/'audit-progress.json').write_text(json.dumps({'completed':n,'total':630,'elapsed_seconds':time.time()-start,'last_job':j['id']}),encoding='utf-8')
 if n%30==0:print(n,'/630 audit',flush=True)
result={'status':'PASS','jobs':630,'evaluations':3150000,'schedule_lines':3150000,'scope':'all evaluation JSON, consecutive NFE, cost checks, source counts; all schedule gzip CRC and line counts, first/last schedule identity; all learning/archive JSON; Q lineage; frozen code hashes. Does not rerun decoding or independently reparse every full schedule.','seconds':time.time()-start,'rows':rows}
(A/'record-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print('AUDIT COMPLETE',flush=True)
