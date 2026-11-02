from pathlib import Path
import json,csv
T=Path(__file__).resolve().parent
rows=[]
for j in json.loads((T/'jobs.json').read_text(encoding='utf-8')):
 p=T/'runs'/j['id']/'summary.json'
 if p.exists():
  r=json.loads(p.read_text(encoding='utf-8'));rows.append({k:r.get(k) for k in ['job_id','case','algorithm','mode','episode','seed','nfe','status','total_seconds_including_finalization','output_hv_normalized','output_spacing_raw','reward_sum','q_output_states']})
with (T/'run-summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
 if rows:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(T/'completion.json').write_text(json.dumps({'complete':len(rows)==630 and all(r['status']=='PASS' for r in rows),'runs':len(rows),'offline_igd_statistics_pending':True},indent=2),encoding='utf-8')
