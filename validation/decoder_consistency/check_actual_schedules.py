from pathlib import Path
from collections import Counter,defaultdict
import json
from validate_full_model import build,independent_check,GRB
B=Path(__file__).resolve().parents[1];d=json.loads((B/'work/main-schedules.json').read_text(encoding='utf8'))
def indexed(value):return {int(k):v for k,v in (value.items() if isinstance(value,dict) else enumerate(value))}
stage=indexed(d['task_stage']);types=indexed(d['types'])
products=indexed(d['task_to_product']);parts=indexed(d['task_to_part'])
info={(r['resource'],r['task']):r for r in d['resources']}
eligible={int(k):v for k,v in d['num_suppliers'].items()}
proc=[t for t in stage if stage[t]=='采购'];ops=[t for t in stage if stage[t]!='采购'];index={t:k for k,t in enumerate(proc)}
sharing=lambda t:(types[products[t]],parts[t])
purchases=[{'class':sharing(t),'options':[(info[r,t]['time'],info[r,t]['cost']) for r in eligible[t]]} for t in proc]
ot={t:types[products[t]] for t in ops};times={t:{r:info[r,t]['time'] for r in eligible[t]} for t in ops}
direct={(t,r):info[r,t]['cost'] for t in ops for r in eligible[t]}
demand={}
for t in ops:
 for part,n in Counter(d['assembly_requirements'].get(str(t),[])).items():demand[t,(ot[t],part)]=n
arcs=[(p,t) for t in ops for p in d['precedence'].get(str(t),[]) if p in ops]
reports=[]
for i,snapshot in enumerate(d['schedules']):
 schedule=snapshot['schedule'];rows={r['task']:r for r in schedule}
 fixed={t:(rows[t]['resource'],rows[t]['start']) for t in ops}
 model,v=build(ot,times,purchases,demand,arcs,d['setup'],fixed=fixed,direct=direct)
 # Fix supplier CHOICES, not only lead times: equal lead times may have different costs.
 for t in proc:
  model.addConstr(v['A'][index[t]]==rows[t]['end'])
 # z variables are recovered through an explicit model handle, never guessed by variable name.
 for (k,s),z in v['z'].items():model.addConstr(z==int(eligible[proc[k]][s]==rows[proc[k]]['resource']))
 consumption={(t,index[u['purchase_task']]) for t in ops for u in rows[t]['consumed_units']}
 for pair,u in v['u'].items():model.addConstr(u==int(pair in consumption))
 model.setObjective(v['cost']);model.optimize()
 assert model.Status==GRB.OPTIMAL,(i,model.Status)
 independent_check(v)
 alg={'procurement':sum(info[rows[t]['resource'],t]['cost'] for t in proc),
      'processing':sum(info[rows[t]['resource'],t]['cost'] for t in ops),
      'holding':.5*sum(rows[t]['start']-rows[u['purchase_task']]['end'] for t in ops for u in rows[t]['consumed_units'])}
 machines=defaultdict(list)
 for t in ops:machines[rows[t]['resource']].append(rows[t])
 alg['nonprocessing']=.2*sum(max(r['end'] for r in rr)-sum(r['end']-r['start'] for r in rr) for rr in machines.values())
 mip={'procurement':sum(purchases[k]['options'][s][1]*z.X for (k,s),z in v['z'].items()),
      'processing':sum(direct.get((t,m),0)*x.X for (t,m),x in v['x'].items()),
      'holding':.5*sum(h.X for h in v['h'].values()),'nonprocessing':.2*sum(a.X for a in v['idle'].values())}
 errors={k:abs(alg[k]-mip[k]) for k in alg}
 assert max(errors.values())<1e-5
 assert abs(v['cmax'].X-snapshot['fitness'][0])<1e-5
 assert abs(sum(alg.values())-snapshot['fitness'][1])<1e-5
 reports.append({'schedule':i+1,'status':'PASS','makespan':v['cmax'].X,'cost':sum(alg.values()),'cost_components':alg,'max_component_difference':max(errors.values())})
 model.dispose()
out={'scope':'Actual main.py algorithm path with revised decoder and objective; 1 instance each of P1,P2,P3; 10 particles,12 iterations,seed42. Not historical reproduction or comparative experiment.',
 'unique_schedules':len(reports),'all_passed':True,'results':reports}
(B/'work/actual-schedule-validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'unique_schedules':len(reports),'all_passed':True,'max_component_difference':max(r['max_component_difference'] for r in reports)},indent=2))
