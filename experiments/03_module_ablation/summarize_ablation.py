from pathlib import Path
import csv,json,gzip,hashlib,statistics,platform,sys
import numpy as np
O=Path(__file__).resolve().parent
rows=[];generations=[];freq=[]
def front(a):
 a=np.unique(np.array(a,dtype=float),axis=0);a=a[np.lexsort((a[:,1],a[:,0]))];out=[];best=float('inf')
 for x,y in a:
  if y<best:out.append([float(x),float(y)]);best=y
 return out
def writecsv(p,data):
 keys=list(dict.fromkeys(k for r in data for k in r))
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for r in data:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})
for case in ['S1','M2','L3']:
 for p in (O/case).glob('*/sequence-*/*/summary.json'):
  r=json.loads(p.read_text(encoding='utf8'))
  if r['mode'] not in ['train','test']:continue
  assert r['nfe']==5000 and sum(r['module_evaluations'].values())==5000
  assert hashlib.sha256((p.parent/'q-final.pkl').read_bytes()).hexdigest()==r['q_output_sha256']
  rows.append(r)
  with (p.parent/'generations.csv').open(encoding='utf-8-sig',newline='') as f:
   for g in csv.DictReader(f):generations.append({'case':case,'variant':r['variant'],'sequence':r['sequence'],'episode':r['episode'],'mode':r['mode'],'seed':r['seed'],**g})
  with gzip.open(p.parent/'decisions.jsonl.gz','rt',encoding='utf8') as f:
   for line in f:
    d=json.loads(line);freq.append({'case':case,'variant':r['variant'],'sequence':r['sequence'],'episode':r['episode'],'mode':r['mode'],'seed':r['seed'],'state':d['state'],'action':d['action'],'generation':d['generation']})
assert len(rows)==1476 and sum(r['mode']=='train' for r in rows)==1260
for case in ['S1','M2','L3']:
 tests=[r for r in rows if r['case']==case and r['mode']=='test'];ref=np.array(json.loads((O/case/'analysis-reference.json').read_text()))
 rf=front([p for r in tests for p in r['output_front']]);(O/case/'empirical-reference-front.json').write_text(json.dumps({'source':'Union of final output fronts from all72 predeclared test runs; empirical not true Pareto front','normalization_divisors':ref.tolist(),'points':rf}),encoding='utf8');rr=np.array(rf)/ref
 for r in tests:
  aa=np.unique(np.round(np.array(r['output_front']),10),axis=0)
  r['igd_empirical_normalized']=float(np.min(np.linalg.norm(rr[:,None,:]-aa[None,:,:]/ref,axis=2),axis=1).mean()) if len(aa) else None
  if len(aa)>=3:
   dist=np.linalg.norm(aa[:,None,:]-aa[None,:,:],axis=2);np.fill_diagonal(dist,np.inf);r['sp_raw']=float(np.std(np.min(dist,axis=1),ddof=1));r['sp_valid']=True
  else:r['sp_raw']=None;r['sp_valid']=False
writecsv(O/'run-summary.csv',rows);writecsv(O/'observations-for-plotting.csv',generations);writecsv(O/'state-action-observations.csv',freq)
summaries=[]
for case in ['S1','M2','L3']:
 for variant in json.loads((O/'protocol.json').read_text())['variants']:
  rs=[r for r in rows if r['case']==case and r['variant']==variant and r['mode']=='test'];assert len(rs)==9
  hv=[r['output_hv'] for r in rs];summaries.append({'case':case,'variant':variant,'tests':9,'training_sequences':0 if variant=='random_action' else 3,'hv_mean':statistics.mean(hv),'hv_std':statistics.stdev(hv),'hv_median':statistics.median(hv),'sequence_mean_hv':[statistics.mean(r['output_hv'] for r in rs if r['sequence']==seq) for seq in [1,2,3]],'igd_mean':statistics.mean(r['igd_empirical_normalized'] for r in rs),'instrumented_test_seconds_mean':statistics.mean(r['seconds'] for r in rs),'training_seconds_sum':sum(r['seconds'] for r in rows if r['case']==case and r['variant']==variant and r['mode']=='train')})
writecsv(O/'comparison-summary.csv',summaries)
(O/'completion.json').write_text(json.dumps({'status':'complete','runs':len(rows),'training_runs':1260,'test_runs':216,'total_nfe':sum(r['nfe'] for r in rows),'validation':'Exact NFE and module-count sum per run; Q file hashes; nine tests per cell; formal/pilot separation','statistical_scope':'Descriptive statistics; 3 learned training clusters, not9 independent trainings. No claim of significance or complete R1-5 resolution.'},indent=2),encoding='utf8')
lines=['# R1-5模块消融补充结果','', '新协议：固定S1/M2/L3，8组，学习组独立匹配训练，5000次评价预算，预算比例进度。结果不可直接替换旧100代图表。', '', '|案例|组别|平均HV|HV标准差|平均经验IGD|测试耗时均值（含记录）|','|---|---|---:|---:|---:|---:|']
for r in summaries:lines.append(f"|{r['case']}|{r['variant']}|{r['hv_mean']:.6f}|{r['hv_std']:.6f}|{r['igd_mean']:.6f}|{r['instrumented_test_seconds_mean']:.2f}秒|")
lines+=['','每组9次测试分属3条训练序列，不能把它们当成9次独立训练。随机组无训练。经验IGD参照来自本次所有测试输出档案的合并非支配集，不是真实前沿。时间含日志和快照开销，不能称纯算法计算时间。','', '包含逐代奖励及分项、Q历史、状态动作、评价来源、全部目标点和排程、Q检查点、输入与配置。敏感性和与四算法的公平比较不由本批次替代。']
(O/'结果说明.md').write_text('\n'.join(lines),encoding='utf8');print('SUMMARY COMPLETE',len(rows),flush=True)
