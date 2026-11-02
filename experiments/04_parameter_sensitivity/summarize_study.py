from pathlib import Path
import json,csv,gzip,hashlib,zipfile,statistics,sys
T=Path(__file__).resolve().parent;B=T.parents[1];O=B/'outputs';jobs=json.loads((T/'jobs.json').read_text(encoding='utf-8'));protocol=json.loads((T/'protocol.json').read_text(encoding='utf-8'));rows=[]
for job in jobs:
 f=T/'runs'/job['id'];r=json.loads((f/'summary.json').read_text(encoding='utf-8'));assert r['status']=='PASS' and r['nfe']==5000;assert r['configuration_id']==job['configuration_id'] and r['seed']==job['seed']
 if job['previous_job']:
  prev=T/'runs'/job['previous_job']/'q-final.pkl';assert hashlib.sha256(prev.read_bytes()).hexdigest()==r['q_input_sha256']
 if job['algorithm']=='QDMOPSO':assert r['initial_q_loading_verified']
 rows.append(r)
assert len(rows)==285 and sum(r['mode']=='train' for r in rows)==180
keys=['job_id','order','configuration_id','algorithm','mode','episode','seed','nfe','search_seconds_including_checks_and_recording','total_seconds_including_finalization','output_hv_normalized','output_spacing_raw','output_spacing_valid','q_input_states','q_output_states','reward_sum','max_cost_check_error','q_input_sha256','q_output_sha256']
with (T/'run-summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r.get(k) for k in keys} for r in rows)
configs=json.loads((T/'calibration-candidates.json').read_text(encoding='utf-8'))['configurations'];scores=[]
for c in configs:
 rr=[r for r in rows if r['configuration_id']==c['configuration_id'] and r['mode']=='test'];assert len(rr)==3
 vv=[r['output_hv_normalized'] for r in rr];scores.append({'algorithm':c['algorithm'],'configuration_id':c['configuration_id'],'parameters':c['parameters'],'original_setting':c['baseline'],'mean_hv':statistics.mean(vv),'sd_hv_conditional_tests':statistics.stdev(vv),'test_seeds':[r['seed'] for r in rr],'mean_test_seconds':statistics.mean(r['search_seconds_including_checks_and_recording'] for r in rr)})
selected=[]
for name in ['QDMOPSO','MOPSO','MOWOA','MOGWO','MOACO']:
 ss=[r for r in scores if r['algorithm']==name];best=max(r['mean_hv'] for r in ss);picked=next(r for r in ss if best-r['mean_hv']<=1e-12);selected.append(picked)
(T/'candidate-results.json').write_text(json.dumps(scores,ensure_ascii=False,indent=2),encoding='utf-8');(T/'recommended-configurations.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf-8')
with (T/'candidate-results.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(scores[0]));w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in scores)
obs=[];states=[];actions=[]
for r in rows:
 if r['algorithm']!='QDMOPSO':continue
 f=T/'runs'/r['job_id'];g=json.loads((f/'generations.json').read_text(encoding='utf-8'))
 with gzip.open(f/'updates.jsonl.gz','rt',encoding='utf-8') as z:updates={u['nfe']:u for u in (json.loads(x) for x in z)}
 for v in g:
  u=updates.get(v['nfe']);q=[a for row in u['q_table'] for a in row['values']] if u else []
  obs.append({'job_id':r['job_id'],'configuration_id':r['configuration_id'],'mode':r['mode'],'episode':r['episode'],'seed':r['seed'],'generation':v['generation'],'nfe':v['nfe'],'state':json.dumps(v['state']),'action':v['action'],'reward':v['reward'],'reward_components':json.dumps(v['reward_components']),'hv_internal':v['hv_internal'],'hv_analysis_normalized':v['hv_analysis_normalized'],'sp':v['sp'],'sp_valid':v['sp_valid'],'q_states':v['q_states'],'q_mean':statistics.mean(q) if q else None,'q_max':max(q) if q else None,'td_error':u['td_error'] if u else None,'epsilon':v['epsilon'],'seconds':v['seconds'],'igd':None,'igd_reason':v['igd_reason']})
with (T/'learning-observations-for-plotting.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(obs[0]));w.writeheader();w.writerows(obs)
text='# M1有限参数校准结果\n\n285次运行全部完成：180训练、27次Q测试和78次基线测试；每次5000评价，共1425000评价。每个Q配置只有一条独立训练序列，3次测试属于同表条件重复，不视作3次独立训练。推荐配置按预先固定平均归一化输出HV和并列规则选择。原设置和全部候选保留，无更换种子或选择性延长训练。\n\n|算法|推荐编号|参数|平均HV|\n|---|---|---|---:|\n'
for x in selected:text+=f"|{x['algorithm']}|{x['configuration_id']}|{json.dumps(x['parameters'],ensure_ascii=False)}|{x['mean_hv']:.9f}|\n"
text+='\n本结果仅用于M1有限校准及所考察范围的参数响应，不证明原参数最优、跨规模鲁棒或充分收敛。配置使用的业务模型与代码版本见protocol.json和代码快照。训练/测试时间、费用分项、全部排程与Q轨迹已保存。正式六实例比较尚未启动；需先报告推荐值和原设置差异再确认。\n'
p=O/'M1有限参数校准结果.md';p.write_text(text,encoding='utf-8-sig');files=[f for f in T.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.name not in ['checksums.json','controller.log','controller-error.log']]
(T/'checksums.json').write_text(json.dumps([{'file':str(f.relative_to(T)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in files],ensure_ascii=False,indent=2),encoding='utf-8')
with zipfile.ZipFile(O/'M1有限参数校准完整记录.zip','w',zipfile.ZIP_DEFLATED) as z:
 z.write(p,p.name)
 for f in files+[T/'checksums.json']:z.write(f,str(f.relative_to(T)))
(T/'record-validation.json').write_text(json.dumps({'status':'PASS','runs':285,'evaluations':1425000,'q_checkpoint_chains_checked':True,'loading_verified':True,'selection_rule_fixed':True},indent=2),encoding='utf-8')
print('SUMMARY AND ARCHIVE COMPLETE',flush=True)
