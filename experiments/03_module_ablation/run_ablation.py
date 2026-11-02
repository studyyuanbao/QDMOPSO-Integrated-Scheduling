from pathlib import Path
import os,sys,json,csv,gzip,hashlib,shutil,random,time,contextlib,pickle,traceback,math,inspect,argparse
from collections import Counter
from copy import deepcopy
import numpy as np
B=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;C=O/'code'
sys.path.insert(0,str(C));os.chdir(O)
import matplotlib
matplotlib.use('Agg')
import pso_rl_new1 as algo
from rl_agent_new import QLearningAgent as BaseAgent
from utils import calculate_hypervolume
class BudgetReached(BaseException):pass
source_lines={}
for i,line in enumerate((C/'pso_rl_new1.py').read_text(encoding='utf8').splitlines(),1):
 if '= evaluate(' in line:
  source_lines[i]='initialization' if i<400 else ('kick' if 'p.fitness' in line else ('reconstruction' if 'child.fitness' in line else 'swarm'))
base_archive=algo.update_archive
def archive(new,ca,da,config):
 if config.variant!='single_archive':return base_archive(new,ca,da,config)
 pool=algo.nondominated(new+ca+da);unique={tuple(np.round(p.fitness,8)):p for p in pool};pool=list(unique.values())
 if len(pool)>400:
  f=np.array([p.fitness for p in pool]);d=np.zeros(len(pool))
  for j in range(2):
   order=np.argsort(f[:,j],kind='stable');d[order[0]]=d[order[-1]]=np.inf;span=np.ptp(f[:,j])
   if span>0:d[order[1:-1]]+=(f[order[2:],j]-f[order[:-2],j])/span
  pool=[pool[i] for i in np.argsort(-d,kind='stable')[:400]]
 return deepcopy(pool),[]
algo.update_archive=archive
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
 if isinstance(x,(np.integer,)):return int(x)
 if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
 if isinstance(x,np.bool_):return bool(x)
 return x

def write_json(p,x):p.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def hashfile(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def front(vals):
 a=np.unique(np.asarray(vals,dtype=float),axis=0);a=a[np.lexsort((a[:,1],a[:,0]))];f=[];best=float('inf')
 for x,y in a:
  if y<best:f.append([float(x),float(y)]);best=y
 return f

def run(label,seq,episode,mode,seed,checkpoint,config_class,reference,variant):
 folder=O/label/variant/f'sequence-{seq}'/(f'train-{episode:02d}' if mode=='train' else f'{mode}-{seed}');folder.mkdir(parents=True,exist_ok=True)
 c=config_class();c.num_particles=50;c.variant=variant
 terminal={};event_counts=Counter();last_record_nfe=0;budget=5000
 c.budget_progress=lambda: max(0.,min(1.,(len(fitness)-50)/(budget-50)))
 random.seed(seed);np.random.seed(seed)
 write_json(folder/'rng-initial.json',{'python':random.getstate(),'numpy':np.random.get_state()})
 initial=pickle.loads(checkpoint.read_bytes()) if checkpoint else {};initialkeys=set(initial)
 checkpoints_input_hash=hashfile(checkpoint) if checkpoint else None
 generations=[];decisions=[];update_count=0;state_count=0;fitness=[];schedule_seen=set();agents=[];schedule_particles={};visits=Counter();state_visits=Counter();start=time.perf_counter()
 streams={name:gzip.open(folder/(name+'.jsonl.gz'),'wt',encoding='utf8',compresslevel=2) for name in ['states','decisions','updates','q-history','evaluations','schedules','reward-inputs']}
 trace_seconds=0.
 def emit(name,x):
  nonlocal trace_seconds
  t0=time.perf_counter();streams[name].write(json.dumps(clean(x),ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n');trace_seconds+=time.perf_counter()-t0
 class Agent(BaseAgent):
  def __init__(self,*a,**kw):super().__init__(*a,**kw);agents.append(self)
  def get_state_indicators(self,hv,sp,progress):
   nonlocal state_count
   old={'best_sp':getattr(self,'best_spacing',None),'previous_valid_sp':self.prev_spacing,'hv_short':self.hv_ewma,'hv_long':getattr(self,'hv_baseline_ewma',None)}
   state=super().get_state_indicators(hv,sp,progress)
   if variant=='simple_state':state=(state[2],)
   state_count+=1
   emit('states',{'call':state_count,'hv':hv,'sp':sp,'sp_valid':sp is not None and bool(np.isfinite(sp)),'progress':progress,'history_before':old,'best_sp_after':self.best_spacing,'sp_ratio':getattr(self,'last_sp_ratio',None),'state':state})
   return state
  def choose_action(self,state):
   clone=np.random.RandomState();clone.set_state(np.random.get_state());explored=bool(clone.rand()<self.epsilon)
   q=list(self.q_table.get(state,[0.]*len(self.actions)));existed=state in self.q_table
   action=int(np.random.choice(self.actions)) if variant=='random_action' else super().choose_action(state);state_visits[state]+=1;visits[int(action)]+=1
   row={'generation':len(decisions)+1,'state':state,'action':int(action),'epsilon':self.epsilon,'choice_rule':'uniform_random' if variant=='random_action' else 'epsilon_greedy','explored':True if variant=='random_action' else explored,'all_q_equal':len(set(q))==1,'state_in_initial_q':state in initialkeys,'state_in_current_q_before_choice':existed,'q_values':q,'visits':state_visits[state]}
   decisions.append(row);emit('decisions',row);return action
  def calculate_composite_reward(self,*args,**kwargs):
   import inspect
   inputs=dict(inspect.signature(BaseAgent.calculate_composite_reward).bind(self,*args,**kwargs).arguments);inputs.pop('self',None)
   result=super().calculate_composite_reward(*args,**kwargs)
   emit('reward-inputs',{'generation':len(decisions),'inputs':inputs,'reward':result,'components':self.last_reward_components.copy()});return result
  def update(self,state,action,reward,next_state):
   nonlocal update_count
   if variant=='random_action':return
   before=list(self.q_table.get(state,[0.]*len(self.actions)));target=reward+self.gamma*max(self.q_table.get(next_state,[0.]*len(self.actions)));idx=self.actions.index(action)
   super().update(state,action,reward,next_state);after=list(self.q_table[state]);update_count+=1
   assert abs(after[idx]-before[idx]-self.alpha*(target-before[idx]))<1e-8
   emit('updates',{'generation':len(decisions),'state':state,'action':int(action),'reward':reward,'components':getattr(self,'last_reward_components',{}).copy(),'next_state':next_state,'q_before':before,'q_after':after,'q_target':target,'td_error':target-before[idx],'bootstrap_used':True,'last_generation':False})
   emit('q-history',{'generation':len(decisions),'table':[{'state':k,'values':v.copy()} for k,v in self.q_table.items()]})
 def record(v,a,skipped):
  nonlocal last_record_nfe
  last_record_nfe=len(fitness)
  schedule_particles.clear()
  pts=algo.safe_extract_fitness_tuples(v.get('archive_union',[]));valid=sorted({tuple(round(float(z),10) for z in p) for p in pts if np.isfinite(p).all()});sp=algo.safe_calculate_spacing(v.get('archive_union',[]));hv=v.get('current_hv')
  analysis_hv=calculate_hypervolume(pts,reference)/(reference[0]*reference[1]) if pts else 0.
  row={'generation':int(v['t'])+1,'nfe':len(fitness),'seconds':time.perf_counter()-start,'state':v['state'],'action':int(v['action']),'reward':None if skipped else v.get('reward'),'reward_components':None if skipped else a.last_reward_components.copy(),'update_skipped':skipped,'hv_internal':hv,'hv_internal_reference':v.get('ref_point'),'hv_analysis_normalized':analysis_hv,'hv_analysis_reference':reference,'sp':sp,'sp_valid':bool(np.isfinite(sp)),'sp_reason':'' if np.isfinite(sp) else 'fewer_than_3_unique_finite_points','igd':None,'igd_reason':'no_matched_reference','front':valid,'archive_sizes':{'CA':len(v.get('convergence_archive',[])),'DA':len(v.get('diversity_archive',[]))},'unique_points':len(valid),'outside_analysis_reference':sum(p[0]>reference[0] or p[1]>reference[1] for p in pts),'module_evaluations':dict(event_counts),'best_evaluated_time':min(p[0] for p in fitness),'best_evaluated_cost':min(p[1] for p in fitness),'q_states':len(a.q_table),'epsilon':a.epsilon,'cumulative_reward':a.cumulative_reward}
  generations.append(row)
  if len(fitness)>=budget:
   terminal.update(v.copy())
   raise BudgetReached()
 c.record_generation=record
 base_evaluate=algo.evaluate
 def evaluate(schedule,config,*a,**kw):
  frame=sys._getframe(1);parent=frame
  while parent and parent.f_code.co_name!='run_mopso':parent=parent.f_back
  if len(fitness)>=budget:
   if parent:terminal.update(parent.f_locals.copy())
   raise BudgetReached()
  name=frame.f_code.co_name
  if name=='local_search_elites':source='local_search'
  elif parent and 'particle' in parent.f_locals and parent.f_locals.get('t',-1)<0:source='initialization'
  elif parent:
   line=parent.f_lineno
   source=source_lines.get(line,'unknown')
  else:source='unknown'
  event_counts[source]+=1
  result=base_evaluate(schedule,config,*a,**kw);fit=[float(result[0]),float(result[1])];fitness.append(fit)
  particle_copy=deepcopy(frame.f_locals.get('temp_particle') if source=='local_search' else (parent.f_locals.get('child') if source=='reconstruction' else (parent.f_locals.get('p') if source=='kick' else parent.f_locals.get('particle'))))
  assert particle_copy is not None,source
  particle_copy.fitness=result
  schedule_particles[len(fitness)]=(len(fitness),particle_copy)
  payload=json.dumps(clean(schedule),ensure_ascii=False,allow_nan=False,separators=(',',':'));sid=hashlib.sha256(payload.encode()).hexdigest()
  if sid not in schedule_seen:schedule_seen.add(sid);streams['schedules'].write('{"schedule_id":"'+sid+'","schedule":'+payload+'}\n')
  emit('evaluations',{'nfe':len(fitness),'source':source,'generation_in_progress':len(decisions),'fitness':fit,'schedule_id':sid,'seconds':time.perf_counter()-start});return result
 algo.QLearningAgent=Agent;algo.evaluate=evaluate
 try:
  try:
   with (folder/'run.log').open('w',encoding='utf8') as log,contextlib.redirect_stdout(log):result=algo.run_mopso(c,total_iterations=10000,load_q_table_path=str(checkpoint) if checkpoint else None,exploit_only=mode!='train',save_q_table_path=None)
   raise RuntimeError('Iteration safety limit reached before NFE budget')
  except BudgetReached:
   assert len(fitness)==budget and terminal
   # Finish output archive bookkeeping for every evaluated candidate in the incomplete generation.
   pending=[]
   for pair in list(schedule_particles.values()):
    if pair[0]>last_record_nfe:pending.append(pair[1])
   ca,da=algo.update_archive(pending,terminal['convergence_archive'],terminal['diversity_archive'],c)
   output_front=algo.safe_extract_fitness_tuples(algo.nondominated(ca+da))
   write_json(folder/'final-output-front.json',output_front)
   emit('states',{'event':'budget_stop','nfe':len(fitness),'generation_in_progress':len(decisions),'last_complete_generation':len(generations),'partial_generation_q_update':False,'pending_evaluated_candidates_incorporated':len(pending),'output_front':output_front})
 finally:
  algo.evaluate=base_evaluate
  for stream in streams.values():stream.close()
 assert len(fitness)==budget
 assert update_count==(0 if variant=='random_action' else max(0,len(generations)-1))
 assert event_counts['unknown']==0
 if variant=='no_local_search':assert event_counts['local_search']==0
 if variant=='no_reconstruction':assert event_counts['reconstruction']==0
 if variant=='no_kick':assert event_counts['kick']==0
 assert all(d['epsilon']==(.5 if mode in ('train','pilot') else 0) for d in decisions)
 assert all(g['outside_analysis_reference']==0 for g in generations)
 if checkpoint:assert hashfile(checkpoint)==checkpoints_input_hash
 a=agents[-1];saved={tuple(int(x) for x in k):[float(z) for z in v] for k,v in a.q_table.items()};cp=folder/'q-final.pkl';cp.write_bytes(pickle.dumps(saved));write_json(folder/'q-final.json',[{'state':k,'values':v} for k,v in saved.items()])
 with (folder/'generations.csv').open('w',encoding='utf-8-sig',newline='') as file:
  writer=csv.DictWriter(file,fieldnames=list(generations[0]));writer.writeheader()
  for row in generations:writer.writerow({k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(list,tuple,dict)) else clean(v) for k,v in row.items()})
 write_json(folder/'generations.json',generations);write_json(folder/'rng-final.json',{'python':random.getstate(),'numpy':np.random.get_state()})
 f=front(fitness);rawhv=calculate_hypervolume(f,reference);summary={'case':label,'variant':variant,'module_evaluations':dict(event_counts),'output_front':output_front,'final_archive_sizes':{'CA':len(ca),'DA':len(da)},'output_hv':calculate_hypervolume(output_front,reference)/(reference[0]*reference[1]),'partial_generation_decisions':len(decisions)-len(generations),'sequence':seq,'episode':episode,'mode':mode,'seed':seed,'nfe':len(fitness),'seconds':time.perf_counter()-start,'trace_emit_seconds':trace_seconds,'runtime_scope':'Instrumented wall time including logging and candidate snapshots; trace_emit_seconds measures only JSONL emit calls, not total overhead','updates':update_count,'decisions':len(decisions),'state_calls':state_count,'initial_q_states':len(initial),'final_q_states':len(saved),'unique_decision_states':len(state_visits),'action_counts':dict(visits),'component_counts':[dict(Counter(str(r['state'][i]) for r in decisions)) for i in range(1 if variant=='simple_state' else 5)],'unseen_initial_decisions':sum(not r['state_in_initial_q'] for r in decisions),'equal_q_decisions':sum(r['all_q_equal'] for r in decisions),'analysis_hv':rawhv/(reference[0]*reference[1]),'front_of_all_evaluations':f,'reward_sum':sum(g['reward'] or 0. for g in generations),'sp_invalid_generations':sum(not g['sp_valid'] for g in generations),'reward_clip_generations':sum(g['reward'] is not None and abs(g['reward'])>=20 for g in generations),'q_input_sha256':checkpoints_input_hash,'q_output_sha256':hashfile(cp),'status':'completed'}
 write_json(folder/'summary.json',summary);return cp,summary

VARIANTS=['full','random_action','single_archive','no_local_search','no_reconstruction','no_kick','simple_state','no_sp_reward']
def config_for(label):
 env={};exec(compile((O/label/'config.py').read_text(encoding='utf8'),label,'exec'),env);return env['Config']
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['S1','M2','L3'],default='S1');parser.add_argument('--pilot',action='store_true');args=parser.parse_args();label=args.case;Config=config_for(label)
 ref=json.loads((O/label/'analysis-reference.json').read_text(encoding='utf8'))
 (O/label).mkdir(exist_ok=True)
 rows=[]
 if args.pilot:
  for v in VARIANTS:
   cp,row=run(label,0,1,'pilot',53991,None,Config,ref,v);rows.append(row);print(v,row['nfe'],row['module_evaluations'],flush=True)
  write_json(O/f'{label}-pilot-summary.json',rows);print('PILOT PASSED',flush=True);return
 empty=O/label/'empty.pkl';empty.write_bytes(pickle.dumps({}))
 for v in VARIANTS:
  for seq in range(1,4):
   checkpoint=None
   if v!='random_action':
    for ep in range(1,21):
     folder=O/label/v/f'sequence-{seq}'/f'train-{ep:02d}'
     if (folder/'summary.json').exists():checkpoint=folder/'q-final.pkl';row=json.loads((folder/'summary.json').read_text(encoding='utf8'))
     else:checkpoint,row=run(label,seq,ep,'train',51000+(seq-1)*100+ep,checkpoint,Config,ref,v)
     rows.append(row);print(label,v,seq,ep,'train',round(row['seconds'],2),flush=True)
   for k in range(3):
    seed=53001+(seq-1)*3+k;folder=O/label/v/f'sequence-{seq}'/f'test-{seed}'
    if (folder/'summary.json').exists():row=json.loads((folder/'summary.json').read_text(encoding='utf8'))
    else:_,row=run(label,seq,None,'test',seed,checkpoint or empty,Config,ref,v)
    rows.append(row);print(label,v,seq,seed,'test',round(row['output_hv'],6),flush=True)
   write_json(O/f'{label}-progress.json',{'runs':len(rows),'last':rows[-1]})
 write_json(O/f'{label}-completed.json',{'status':'complete','runs':len(rows),'summaries':rows})
if __name__=='__main__':
 try:main()
 except BaseException:
  (O/('failure-'+str(os.getpid())+'.txt')).write_text(traceback.format_exc(),encoding='utf8');raise
