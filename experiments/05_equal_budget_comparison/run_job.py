from pathlib import Path
import os,sys,json,hashlib,random,importlib,inspect,time,contextlib,traceback,gzip,csv,shutil,platform,pickle
from collections import Counter,defaultdict
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
T=Path(__file__).resolve().parent;B=T.parents[1];C=T/'code';R=T/'runs';R.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR']=str(T/'mpl-cache');os.environ['MPLBACKEND']='Agg';sys.path.insert(0,str(C));sys.dont_write_bytecode=True
os.chdir(R)
import decoder,objective,product_semantics
class BudgetStop(BaseException):pass
class TimeStop(BaseException):pass

def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
 if isinstance(x,(np.integer,)):return int(x)
 if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
 if isinstance(x,np.bool_):return bool(x)
 return x

def save(p,x):p.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def check(schedule,c,fit):
 assert len(schedule)==c.num_tasks;rows={r['task']:r for r in schedule};assert set(rows)==set(range(c.num_tasks))
 types=product_semantics.product_types(c);used=set();holding=0.;machines=defaultdict(list);direct=0.
 for t,r in rows.items():
  res=r['resource'];assert res in c.num_suppliers[t];assert r['product_type']==types[c.task_to_product[t]]
  assert abs(r['end']-r['start']-c.machine_task_map[res,t]['time'])<1e-8;direct+=c.machine_task_map[res,t]['cost']
  assert all(r['start']>=rows[p]['end']-1e-8 for p in c.precedence.get(t,[]))
  if c.task_stage[t]=='采购':assert abs(r['start'])<1e-8
  else:
   machines[res].append(r);units=r['consumed_units'];assert Counter(u['part'] for u in units)==Counter(c.assembly_requirements.get(t,[]))
   for u in units:
    k=u['purchase_task'];assert k not in used;used.add(k);pr=rows[k];assert c.task_stage[k]=='采购';assert c.task_to_part[k]==u['part'];assert pr['product_type']==r['product_type'];assert r['start']>=pr['end']-1e-8;assert abs(u['arrival']-pr['end'])<1e-8;holding+=r['start']-pr['end']
 assert used=={t for t in rows if c.task_stage[t]=='采购'}
 idle=0.
 for mm,rr in machines.items():
  rr=sorted(rr,key=lambda r:(r['start'],r['task']));end=0.;prev=None
  for r in rr:
   setup=c.product_change_penalty if prev is not None and prev!=r['product_type'] else 0.
   assert r['start']>=end+setup-1e-8;idle+=r['start']-end;end=r['end'];prev=r['product_type']
  assert abs(sum(r['start']-(rr[i-1]['end'] if i else 0) for i,r in enumerate(rr))-(end-sum(r['end']-r['start'] for r in rr)))<1e-8
 expected=[max(r['end'] for r in schedule),direct+.5*holding+.2*idle];err=max(abs(fit[i]-expected[i]) for i in range(2));assert err<1e-7
 return {'direct':direct,'holding':.5*holding,'nonprocessing':.2*idle,'max_error':err}

manifest={'file_hashes':json.loads((T/'code-hashes.json').read_text(encoding='utf-8'))}
for name,digest in manifest['file_hashes'].items():assert h(C/name)==digest,name

from utils import calculate_hypervolume
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--job',type=int,required=True);args=parser.parse_args()
job=json.loads((T/'jobs.json').read_text(encoding='utf-8'))[args.job-1];assert job['order']==args.job
args.case=job['id']
reference=json.loads((T/'analysis-references.json').read_text(encoding='utf-8'))[job['case']]['analysis_hv_reference']
JOBS=[(job['case'],job['algorithm'],5000)]
q_source_lines=(C/'pso_rl_new1.py').read_text(encoding='utf-8').splitlines()
ENTRIES={'QDMOPSO':('pso_rl_new1','run_mopso','update_archive'),'MOPSO':('mopso_basic','run_mopso','update_archive_simplified'),'MOWOA':('mowoa','run_mowoa','update_archive_simplified'),'MOGWO':('mogwo','run_mogwo','update_archive'),'MOACO':('moaco_2','run_moaco','update_archive_simplified')}
summaries=[]
for case,label,budget in JOBS:
 folder=R/job['id']
 if folder.exists():raise RuntimeError('Do not overwrite an attempt: '+str(folder))
 folder.mkdir();src=T/'instances'/case/'config.py'
 assert h(src)==json.loads((T/'inputs.json').read_text(encoding='utf-8'))[case]['config_sha256']
 if case.startswith('G'):assert h(src.parent/'input.json')==json.loads((T/'inputs.json').read_text(encoding='utf-8'))[case]['input_sha256']
 random.seed(42);np.random.seed(42);spec=importlib.util.spec_from_file_location('input_'+case,src);cm=importlib.util.module_from_spec(spec);spec.loader.exec_module(cm);c=cm.Config()
 c.num_particles=c.num_ants=50;c.num_iterations=10000;c.archive_size=400;c.ca_size=c.da_size=200;c.variant='full';c.mopso_w=.4;c.mopso_c1=c.mopso_c2=2.;c.mopso_vmax=.5;c.q_learning_alpha=.25;c.q_discount_gamma=.98;c.q_training_epsilon=.5
 for key,value in job['parameters'].items():setattr(c,key,value)
 save(folder/'config.json',vars(c));save(folder/'job.json',job);shutil.copy2(src,folder/'config-source.py');save(folder/'import-paths.json',{'decoder':decoder.__file__,'objective':objective.__file__,'product_semantics':product_semantics.__file__})
 checkpoint=T/'runs'/job['previous_job']/'q-final.pkl' if job['previous_job'] else None
 if checkpoint:assert checkpoint.exists(),str(checkpoint)
 expected_initial=pickle.loads(checkpoint.read_bytes()) if checkpoint else {}
 input_q_hash=h(checkpoint) if checkpoint else None
 save(folder/'q-input.json',{'path':str(checkpoint) if checkpoint else None,'sha256':input_q_hash,'states':len(expected_initial),'mode':job['mode']})
 if case.startswith('G'):shutil.copy2(src.parent/'input.json',folder/'input.json')
 random.seed(job['seed']);np.random.seed(job['seed']);save(folder/'rng-initial.json',{'python':random.getstate(),'numpy':np.random.get_state()})
 mod,entry,archive_name=ENTRIES[label];m=importlib.import_module(mod);assert m.decode is decoder.decode and m.evaluate is objective.evaluate
 for attr,val in [('global_min_time',float('inf')),('global_max_time',-float('inf')),('global_min_cost',float('inf')),('global_max_cost',-float('inf'))]:setattr(objective,attr,val)
 evals=[];pending=[];gens=[];agents=[];choices=[];decode_stats=Counter();counts=Counter();state={'archives':([],[]) if label=='QDMOPSO' else [],'last_archive_nfe':0};streams={n:gzip.open(folder/(n+'.jsonl.gz'),'wt',encoding='utf-8',compresslevel=2) for n in ['evaluations','schedules','states','decisions','updates','archive-events']}
 start=time.perf_counter();overhead={'emit':0.,'independent_checks':0.};base_eval=m.evaluate;base_dec=m.decode;base_archive=getattr(m,archive_name)
 def deadline():
  if time.perf_counter()-start>=3600:raise TimeStop()
 def emit(name,row):
  t0=time.perf_counter();streams[name].write(json.dumps(clean(row),ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n');overhead['emit']+=time.perf_counter()-t0
 def evaluate(schedule,config,*a,**kw):
  deadline()
  if len(evals)>=budget:raise BudgetStop()
  caller=inspect.currentframe().f_back;fun=caller.f_code.co_name;line=caller.f_lineno
  if fun=='local_search_elites':source='local_search'
  elif label=='QDMOPSO':
   src_line=q_source_lines[line-1]
   source='reconstruction' if 'child.fitness' in src_line else ('kick' if 'p.fitness' in src_line else ('initialization' if len(choices)==0 else 'swarm'))
  else:source='initialization' if len(evals)<50 else 'swarm'
  value=base_eval(schedule,config,*a,**kw);assert np.isfinite(value).all();assert all(0<=value[i]<=reference[i] for i in range(2)),value;t0=time.perf_counter();parts=check(schedule,config,value);overhead['independent_checks']+=time.perf_counter()-t0
  sid=hashlib.sha256(json.dumps(clean(schedule),sort_keys=True,ensure_ascii=False).encode()).hexdigest();row={'nfe':len(evals)+1,'source':source,'caller':{'function':fun,'line':line},'fitness':value,'cost_components':parts,'schedule_id':sid,'seconds':time.perf_counter()-start};evals.append(row);counts[source]+=1;emit('evaluations',row);emit('schedules',{'nfe':len(evals),'schedule_id':sid,'schedule':schedule})
  pending.append(SimpleNamespace(fitness=list(value),schedule=deepcopy(schedule),position=[],crowding_distance=0.))
  if len(evals)%500==0:
   save(T/'current-progress.json',{'job_id':job['id'],'order':job['order'],'total_jobs':630,'nfe':len(evals),'budget':5000,'seconds':time.perf_counter()-start,'status':'running'})
   for stream in streams.values():stream.flush()
  return value
 def decode(p,config):
  deadline()
  if len(evals)>=budget:raise BudgetStop()
  ans=base_dec(p,config);decode_stats['calls']+=1
  if not ans:decode_stats['empty']+=1;raise AssertionError('Empty decoded schedule')
  return ans
 def archive(*a,**kw):
  deadline();ans=base_archive(*a,**kw);state['archives']=deepcopy(ans);state['last_archive_nfe']=len(evals);pending.clear()
  flat=ans[0]+ans[1] if label=='QDMOPSO' else ans;emit('archive-events',{'nfe':len(evals),'points':[p.fitness for p in flat]});deadline();return ans
 import particle as particle_module
 original_initializer=particle_module.generate_feasible_particle
 def bounded_initializer(*a,**kw):
  deadline();value=original_initializer(*a,**kw);deadline();return value
 particle_module.generate_feasible_particle=bounded_initializer
 m.evaluate=evaluate;m.decode=decode;setattr(m,archive_name,archive);c.budget_progress=lambda:max(0.,min(1.,(len(evals)-50)/(budget-50)))
 def record(v,a,skipped):
  gens.append(clean({'generation':int(v['t'])+1,'nfe':len(evals),'state':v['state'],'action':v['action'],'reward':None if skipped else v.get('reward'),'reward_components':None if skipped else a.last_reward_components,'skipped_update':skipped,'hv_internal':v.get('current_hv'),'sp':v.get('sp_now'),'q_states':len(a.q_table),'epsilon':a.epsilon,'seconds':time.perf_counter()-start}))
 base_record=record
 def record_with_analysis(v,a,skipped):
  base_record(v,a,skipped)
  pts=[p.fitness for p in v.get('archive_union',[]) if np.isfinite(p.fitness).all()]
  gens[-1]['hv_analysis_normalized']=calculate_hypervolume(pts,reference)/(reference[0]*reference[1]) if pts else 0.
  gens[-1]['sp_valid']=v.get('sp_now') is not None and bool(np.isfinite(v.get('sp_now')))
  gens[-1]['igd']=None
  gens[-1]['igd_reason']='computed offline after all formal tests; not available during search'
 c.record_generation=record_with_analysis
 if label=='QDMOPSO':
  Base=m.QLearningAgent
  class Agent(Base):
   def __init__(self,*a,**kw):super().__init__(*a,**kw);agents.append(self)
   def get_state_indicators(self,hv,sp,progress):
    before=getattr(self,'best_spacing',None);s=super().get_state_indicators(hv,sp,progress);emit('states',{'nfe':len(evals),'hv':hv,'sp':sp,'progress':progress,'previous_best_sp':before,'state':s});return s
   def choose_action(self,s):
    if not choices:
     assert dict(self.q_table)==expected_initial,'Q loading mismatch'
     assert self.epsilon==(0. if job['mode']=='test' else .5),'epsilon mode mismatch'
    a=super().choose_action(s);choices.append(a);emit('decisions',{'nfe':len(evals),'state':s,'action':a,'epsilon':self.epsilon,'q':self.q_table[s]});return a
   def update(self,s,a,r,ns):
    before=list(self.q_table[s]);idx=self.actions.index(a);td=r+self.gamma*max(self.q_table.get(ns,[0.]*7))-before[idx];super().update(s,a,r,ns);assert abs(self.q_table[s][idx]-before[idx]-self.alpha*td)<1e-8;emit('updates',{'alpha':self.alpha,'gamma':self.gamma,'td_error':td,'nfe':len(evals),'state':s,'action':a,'reward':r,'reward_components':self.last_reward_components,'next_state':ns,'q_before':before,'q_after':self.q_table[s],'q_table':[{'state':k,'values':v} for k,v in self.q_table.items()]})
  m.QLearningAgent=Agent
 status='FAIL';error=None;stop_seconds=None;pending_count=0;archive_finish_seconds=0.;output_n=0
 try:
  with (folder/'run.log').open('w',encoding='utf-8') as f,contextlib.redirect_stdout(f):
   try:
    if label=='QDMOPSO':getattr(m,entry)(c,total_iterations=10000,save_q_table_path=None,load_q_table_path=str(checkpoint) if checkpoint else None,exploit_only=job['mode']=='test')
    else:getattr(m,entry)(c)
    raise AssertionError('Iteration safety limit before evaluation budget')
   except BudgetStop:status='PASS';assert len(evals)==budget
   except TimeStop:status='TIMEOUT'
  stop_seconds=time.perf_counter()-start
  if status=='PASS':
   pending_count=len(pending);old=state['archives'];t0=time.perf_counter()
   if label=='QDMOPSO':ans=base_archive(pending,old[0],old[1],c);flat=ans[0]+ans[1]
   else:ans=base_archive(pending,old,c);flat=ans
   archive_finish_seconds=time.perf_counter()-t0;output_n=len(flat);assert flat
   save(folder/'final-output-archive.json',[{'fitness':p.fitness,'schedule':p.schedule} for p in flat])
  else:
   old=state['archives'];flat=old[0]+old[1] if label=='QDMOPSO' else old;save(folder/'timeout-partial-candidates.json',[{'fitness':p.fitness,'schedule':p.schedule} for p in pending]);save(folder/'timeout-last-archive.json',[{'fitness':p.fitness,'schedule':p.schedule} for p in flat])
 except BaseException:
  error=traceback.format_exc();status='FAIL';(folder/'failure.txt').write_text(error,encoding='utf-8')
 finally:
  particle_module.generate_feasible_particle=original_initializer
  m.evaluate=base_eval;m.decode=base_dec;setattr(m,archive_name,base_archive)
  if label=='QDMOPSO':m.QLearningAgent=Base
  for f in streams.values():f.close()
 save(folder/'generations.json',gens)
 if gens:
  with (folder/'generations.csv').open('w',encoding='utf-8-sig',newline='') as f:
   wr=csv.DictWriter(f,fieldnames=list(gens[0]));wr.writeheader();wr.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in gens)
 if agents:
  q={tuple(k):list(v) for k,v in agents[0].q_table.items()};save(folder/'q-final.json',[{'state':k,'values':v} for k,v in q.items()]);(folder/'q-final.pkl').write_bytes(pickle.dumps(q))
 if checkpoint:assert h(checkpoint)==input_q_hash,'Input checkpoint mutated'
 save(folder/'rng-final.json',{'python':random.getstate(),'numpy':np.random.get_state()})
 result={'case':case,'algorithm':label,'status':status,'target_nfe':budget,'nfe':len(evals),'search_seconds_including_checks_and_recording':stop_seconds,'total_seconds_including_finalization':time.perf_counter()-start,'archive_finalization_seconds':archive_finish_seconds,'pending_evaluated_candidates_incorporated':pending_count,'output_archive_size':output_n,'decode':dict(decode_stats),'evaluations_by_source':dict(counts),'complete_q_generations':len(gens),'q_decisions':len(choices),'max_cost_check_error':max((r['cost_components']['max_error'] for r in evals),default=None),'measured_partial_overhead_seconds':overhead,'time_scope':'instrumented; only complete PASS runs yield complete-budget timing; timeout is lower bound','source_manifest_sha256':h(T/'code-hashes.json'),'config_source_sha256':h(src)}
 if status=='PASS':
  points=sorted(set(tuple(p.fitness) for p in flat));nd=[];best=float('inf')
  for x,y in points:
   if y<best:nd.append([x,y]);best=y
  hv=calculate_hypervolume(nd,reference)/(reference[0]*reference[1]);sp=None
  if len(nd)>=3:
   ar=np.asarray(nd);ds=np.linalg.norm(ar[:,None,:]-ar[None,:,:],axis=2);np.fill_diagonal(ds,np.inf);sp=float(np.std(ds.min(axis=1),ddof=1))
  result.update({'configuration_id':job['configuration_id'],'job_id':job['id'],'order':job['order'],'mode':job['mode'],'episode':job['episode'],'seed':job['seed'],'parameters':job['parameters'],'output_hv_normalized':hv,'output_spacing_raw':sp,'output_spacing_valid':sp is not None,'output_front':nd,'analysis_reference':reference,'igd':None,'igd_reason':'computed offline after all formal tests; training and pilots excluded','q_input_sha256':input_q_hash,'q_input_states':len(expected_initial),'q_output_sha256':h(folder/'q-final.pkl') if agents else None,'q_output_states':len(agents[0].q_table) if agents else None,'reward_sum':sum(x['reward'] or 0 for x in gens),'initial_q_loading_verified':bool(agents),'time_budget':'NFE stopped, no equal-walltime claim'})
 save(folder/'summary.json',result);summaries.append(result);print(job['id'],status,len(evals),round(result['total_seconds_including_finalization'],2),flush=True)
 if error:raise RuntimeError(case+' '+label+' failed; see failure.txt')
assert status=='PASS',status
print('JOB COMPLETE',flush=True)
