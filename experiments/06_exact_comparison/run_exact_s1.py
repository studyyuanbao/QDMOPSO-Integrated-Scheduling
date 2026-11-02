from pathlib import Path
from collections import Counter
import json,time,hashlib,shutil
from validate_full_model import build,independent_check,GRB
B=Path(__file__).resolve().parents[1];out=B/'work/exact-s1';out.mkdir(exist_ok=True)
d=json.loads((B/'work/record-bundle/S1-input.json').read_text(encoding='utf8'))
proc=[t for t,s in enumerate(d['task_stage']) if s=='采购'];ops=[t for t,s in enumerate(d['task_stage']) if s!='采购']
info={(r['task_id'],r['resource_id']):r for r in d['resource_records']};eligible={int(k):v for k,v in d['eligible_resources'].items()}
types={t:d['task_to_product'][t] for t in ops}
purchases=[{'class':(d['task_to_product'][t],d['task_to_part'][str(t)]),'options':[(info[t,r]['actual_duration'],info[t,r]['direct_cost']) for r in eligible[t]]} for t in proc]
times={t:{r:info[t,r]['actual_duration'] for r in eligible[t]} for t in ops}
direct={(t,r):info[t,r]['direct_cost'] for t in ops for r in eligible[t]}
demand={(t,(types[t],part)):n for t in ops for part,n in Counter(d['assembly_requirements'].get(str(t),[])).items()}
arcs=[(p,t) for t in ops for p in d['precedence'].get(str(t),[]) if p in ops]
records=[]
for label in ['makespan','cost']:
 model,v=build(types,times,purchases,demand,arcs,2,direct=direct)
 model.Params.TimeLimit=300;model.Params.Threads=2;model.Params.Seed=42
 model.Params.LogFile=str(out/f'{label}.log')
 model.setObjective(v['cmax'] if label=='makespan' else v['cost'],GRB.MINIMIZE)
 model.write(str(out/f'{label}.lp'));model.optimize()
 row={'label':label,'status':model.Status,'runtime':model.Runtime,'solutions':model.SolCount,'bound':model.ObjBound,'time_limit':300}
 if model.SolCount:
  independent_check(v);model.write(str(out/f'{label}.sol'))
  row.update({'objective':model.ObjVal,'gap':model.MIPGap,'makespan':v['cmax'].X,'cost':v['cost'].getValue()})
 records.append(row);(out/'endpoints.json').write_text(json.dumps(records,indent=2),encoding='utf8');print(json.dumps(row),flush=True)
 model.dispose()
shutil.copy2(B/'work/record-bundle/S1-input.json',out/'S1-input.json')
shutil.copy2(Path(__file__),out/'run_exact_s1.py')
shutil.make_archive(str(B/'outputs/S1精确方法端点求解记录'),'zip',out)
