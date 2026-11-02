from pathlib import Path
import json,csv,math,itertools,collections
import numpy as np
from scipy.stats import friedmanchisquare,rankdata
T=Path(__file__).resolve().parent;O=T.parents[1]/'outputs';A=T/'analysis';A.mkdir(exist_ok=True)
jobs=json.loads((T/'jobs.json').read_text(encoding='utf-8'));refs=json.loads((T/'analysis-references.json').read_text(encoding='utf-8'));cases=list(refs);algs=['QDMOPSO','MOPSO','MOWOA','MOGWO','MOACO']
def dump(n,x):(A/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvsave(n,rr):
 with (A/n).open('w',encoding='utf-8-sig',newline='') as f:
  if rr:
   w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def nd(x):
 x=np.asarray(sorted(set(map(tuple,x))),dtype=float);out=[];best=float('inf')
 for p in x:
  if p[1]<best:out.append(p);best=p[1]
 return np.asarray(out)
def hv(x,ref):
 y=nd(x);best=ref[1];area=0
 for a,b in y:
  if b<best:area+=(ref[0]-a)*(best-b);best=b
 return float(area/(ref[0]*ref[1]))
allrows=[(j,json.loads((T/'runs'/j['id']/'summary.json').read_text(encoding='utf-8'))) for j in jobs];tests=[(j,s) for j,s in allrows if j['mode']=='test'];assert len(tests)==270
fronts={}
for case in cases:
 raw=nd([p for j,s in tests if j['case']==case for p in s['output_front']]);fronts[case]=raw
 dump('empirical-reference-'+case+'.json',{'case':case,'source':'union of final outputs of all 45 formal tests only, deduplicated and nondominated; no training/M1/pilot/old data','raw_front':raw.tolist(),'normalization_divisor':refs[case]['analysis_hv_reference']})
metrics=[]
for j,s in tests:
 ref=np.asarray(refs[j['case']]['analysis_hv_reference']);points=nd(s['output_front']);assert np.isfinite(points).all() and (points>=0).all() and (points<=ref).all();v=hv(points,ref);assert abs(v-s['output_hv_normalized'])<1e-10
 x=points/ref;target=fronts[j['case']]/ref;igd=float(np.linalg.norm(target[:,None,:]-x[None,:,:],axis=2).min(axis=1).mean());distinct=np.unique(np.round(points,10),axis=0);sp=None
 if len(distinct)>=3:
  ds=np.linalg.norm(distinct[:,None,:]-distinct[None,:,:],axis=2);np.fill_diagonal(ds,np.inf);sp=float(np.std(ds.min(axis=1),ddof=1))
 metrics.append({'case':j['case'],'algorithm':j['algorithm'],'seed':j['seed'],'training_group':j['training_group'],'job_id':j['id'],'hv_normalized':v,'igd_empirical_normalized':igd,'spacing_raw_euclidean':sp,'spacing_valid':sp is not None,'distinct_front_points':len(distinct),'seconds_instrumented':s['total_seconds_including_finalization'],'nfe':s['nfe']})
csvsave('formal-test-metrics.csv',metrics)
aggregate=[]
for case in cases:
 for alg in algs:
  rr=[r for r in metrics if r['case']==case and r['algorithm']==alg];r={'case':case,'algorithm':alg,'tests':len(rr),'Q_independent_training_sequences':3 if alg=='QDMOPSO' else None}
  for key in ['hv_normalized','igd_empirical_normalized','spacing_raw_euclidean','seconds_instrumented']:
   v=[x[key] for x in rr if x[key] is not None];r[key+'_mean']=float(np.mean(v)) if v else None;r[key+'_sd_conditional_runs']=float(np.std(v,ddof=1)) if len(v)>1 else None;r[key+'_valid_n']=len(v)
  r['training_seconds_total']=sum(s['total_seconds_including_finalization'] for j,s in allrows if j['case']==case and j['algorithm']==alg and j['mode']=='train');aggregate.append(r)
csvsave('case-algorithm-summary.csv',aggregate)
paired=[]
for case in cases:
 for alg in algs[1:]:
  for g in [1,2,3]:
   a=[r for r in metrics if r['case']==case and r['algorithm']=='QDMOPSO' and r['training_group']==g];b={r['seed']:r for r in metrics if r['case']==case and r['algorithm']==alg};paired.append({'case':case,'baseline':alg,'training_group':g,'delta_hv_Q_minus_baseline':float(np.mean([r['hv_normalized']-b[r['seed']]['hv_normalized'] for r in a])),'delta_igd_baseline_minus_Q':float(np.mean([b[r['seed']]['igd_empirical_normalized']-r['igd_empirical_normalized'] for r in a]))})
csvsave('paired-training-group-effects.csv',paired)
def exact_signed_rank(d):
 d=np.asarray(d);d=d[np.abs(d)>1e-12]
 if not len(d):return 1.
 ranks=rankdata(np.abs(d));obs=abs(float(np.dot(np.sign(d),ranks)));vals=[abs(float(np.dot(signs,ranks))) for signs in itertools.product([-1,1],repeat=len(d))];return sum(v>=obs-1e-12 for v in vals)/len(vals)
stats={}
for metric,direction in [('hv_normalized_mean',-1),('igd_empirical_normalized_mean',1)]:
 mat=np.array([[next(r[metric] for r in aggregate if r['case']==case and r['algorithm']==alg) for alg in algs] for case in cases]);ranks=np.array([rankdata(direction*v) for v in mat]);f=friedmanchisquare(*mat.T);comp=[]
 for k,alg in enumerate(algs[1:],1):
  d=(mat[:,0]-mat[:,k])*(-direction);p=exact_signed_rank(d);comp.append({'baseline':alg,'exact_two_sided_p':p,'per_case_favorable_differences':d.tolist(),'wins':int((d>1e-12).sum()),'ties':int((np.abs(d)<=1e-12).sum()),'losses':int((d< -1e-12).sum()),'mean_rank_advantage':float(np.mean(ranks[:,k]-ranks[:,0]))})
 order=sorted(range(4),key=lambda k:comp[k]['exact_two_sided_p']);running=0
 for i,k in enumerate(order):running=max(running,min(1.,(4-i)*comp[k]['exact_two_sided_p']));comp[k]['holm_adjusted_p']=running
 stats[metric]={'instances':cases,'M1_excluded':True,'n_blocks':6,'Friedman_chi_square':float(f.statistic),'Friedman_asymptotic_p':float(f.pvalue),'Kendall_W':float(f.statistic/(6*4)),'mean_ranks':dict(zip(algs,ranks.mean(axis=0).tolist())),'Q_vs_baselines':comp,'scope':'Six fixed instance blocks; Friedman asymptotic p exploratory with only six blocks. Exact paired signed-rank via all sign assignments (ties midrank; zeros removed); Holm across four comparisons separately per metric. HV primary, empirical IGD secondary. Posthoc values retained transparently regardless of omnibus; not nine independent Q trainings.'}
dump('statistics.json',stats);dump('metric-protocol.json',{'SP':'Euclidean nearest-neighbor spacing in raw objective units; round to 10 decimals for validity/deduplication; at least 3 points; not relabeled Spread or Manhattan SP','HV':'fixed input-derived analysis reference and coordinate scaling; no internal-learning HV mixed','IGD':'mean Euclidean distance from same-instance empirical reference to output in fixed normalized coordinates; in-sample pooled empirical reference, not true Pareto front','aggregation':'each Q training group 3 tests; group means equally weighted, 3 groups per case. Baseline seeds same group partition. Descriptive run SD conditional on trained tables, not training-replicate uncertainty.'})
learning=[]
for j,s in allrows:
 if j['algorithm']!='QDMOPSO':continue
 p=T/'runs'/j['id']/'generations.csv'
 if p.exists():
  for r in csv.DictReader(p.open(encoding='utf-8-sig')):learning.append(dict(job_id=j['id'],case=j['case'],mode=j['mode'],training_group=j['training_group'],episode=j['episode'],seed=j['seed'],**r))
csvsave('learning-observations-for-plotting.csv',learning)
report='# 六实例正式比较 初步离线分析\n\n全部630次原始运行结束，本报告仅汇总正式270测试；记录流审计状态另见record-audit.json，未完成审计前不称最终定稿。未新增运行。\n\n|实例|Q HV|MOPSO HV|MOWOA HV|MOGWO HV|MOACO HV|\n|---|---:|---:|---:|---:|---:|\n'
for case in cases:report+='|'+case+'|'+ '|'.join(f"{next(r['hv_normalized_mean'] for r in aggregate if r['case']==case and r['algorithm']==alg):.6f}" for alg in algs)+'|\n'
report+='\n所有值为平均归一化HV，越大越好。完整均值、标准差、SP有效数、时间和训练成本见case-algorithm-summary.csv。原目标点、固定参考、经验IGD前沿和逐次值完整保留。\n\n'
for key,st in stats.items():
 report+=f"## {key}\n\nFriedman统计量{st['Friedman_chi_square']:.4f}，渐近p={st['Friedman_asymptotic_p']:.6g}，Kendall W={st['Kendall_W']:.4f}。仅六个固定实例，渐近检验作为有限探索性结果；不把相关实例变成随机总体证据。\n\n"
 for c in st['Q_vs_baselines']:report+=f"- Q对{c['baseline']}：胜/平/负={c['wins']}/{c['ties']}/{c['losses']}；精确双侧p={c['exact_two_sided_p']:.6g}，Holm后p={c['holm_adjusted_p']:.6g}。\n"
report+='\n后验比较在每指标四项内校正，HV为主指标，经验IGD为次指标；不能据不显著认定等效。三点不足SP留空，不填0。经验IGD使用本批合并输出前沿，有池化内样本局限。学习内部HV与这里分析HV分开。时间包含记录检查，不能直接称纯搜索速度。排名如有变化，后续按既定四因素说明，不筛实例或种子。\n'
(O/'六实例正式比较初步分析.md').write_text(report,encoding='utf-8');dump('analysis-status.json',{'metrics_complete':True,'formal_tests':270,'training_runs_excluded_from_performance':360,'learning_rows':len(learning),'record_audit_pending':not (A/'record-audit.json').exists()});print(report)
