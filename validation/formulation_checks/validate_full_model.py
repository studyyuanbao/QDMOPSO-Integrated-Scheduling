"""Small synthetic tests of revised manuscript equations; NOT paper reproduction."""
import sys, json, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent/'solver-deps'))
import gurobipy as gp
from gurobipy import GRB

def build(types, times, purchases=(), demand=None, arcs=(), setup=2, fixed=None, broken=False, direct=None):
    # Operation IDs encode (i,j) in these tiny instances; types is tau_i.
    demand = demand or {}
    O=list(types); M=sorted({m for a in times.values() for m in a})
    K=list(range(len(purchases)))
    B=max([v[0] for p in purchases for v in p['options']]+[0])+sum(max(a.values()) for a in times.values())+len(O)*setup
    L=2*B+setup
    model=gp.Model(); model.Params.OutputFlag=0; model.Params.TimeLimit=20
    x={(a,m):model.addVar(vtype=GRB.BINARY,ub=int(m in times[a])) for a in O for m in M}
    T={a:model.addVar(ub=B) for a in O}; C={a:model.addVar(ub=B) for a in O}
    z={(k,s):model.addVar(vtype=GRB.BINARY) for k in K for s in range(len(purchases[k]['options']))}
    A={k:model.addVar(ub=B) for k in K}; h={k:model.addVar(ub=B) for k in K}
    u={(a,k):model.addVar(vtype=GRB.BINARY) for a in O for k in K}
    y={(a,b,m):model.addVar(vtype=GRB.BINARY) for a in O for b in O if a!=b for m in M}
    e={(a,m):model.addVar(ub=B) for a in O for m in M}
    E={m:model.addVar(ub=B) for m in M}; idle={m:model.addVar() for m in M}
    cmax=model.addVar(ub=B)
    for a in O:
        model.addConstr(gp.quicksum(x[a,m] for m in M)==1)
        model.addConstr(C[a]==T[a]+gp.quicksum(times[a].get(m,0)*x[a,m] for m in M))
    for k in K:
        model.addConstr(gp.quicksum(z[k,s] for s in range(len(purchases[k]['options'])))==1)
        model.addConstr(A[k]==gp.quicksum(v[0]*z[k,s] for s,v in enumerate(purchases[k]['options'])))
        model.addConstr(gp.quicksum(u[a,k] for a in O)==1)
        for a in O:
            model.addConstr(T[a]>=A[k]-L*(1-u[a,k]))
            model.addConstr(h[k]>=T[a]-A[k]-L*(1-u[a,k]))
            model.addConstr(h[k]<=T[a]-A[k]+L*(1-u[a,k]))
    for a in O:
        for r in set(p['class'] for p in purchases)|set(r for aa,r in demand):
            model.addConstr(gp.quicksum(u[a,k] for k in K if purchases[k]['class']==r)==demand.get((a,r),0))
    for a,b in arcs: model.addConstr(T[b]>=C[a])
    if broken:
        for a in O: model.addConstr(T[a]>=C[a],name='original_self_precedence_'+str(a))
    for a,b in itertools.combinations(O,2):
        for m in M:
            delta=int(types[a]!=types[b])
            model.addConstr(T[b]>=T[a]+times[a].get(m,0)*x[a,m]+setup*delta-L*(1-y[a,b,m]))
            model.addConstr(T[a]>=T[b]+times[b].get(m,0)*x[b,m]+setup*delta-L*(1-y[b,a,m]))
            model.addConstr(y[a,b,m]+y[b,a,m]>=x[a,m]+x[b,m]-1)
            model.addConstr(y[a,b,m]+y[b,a,m]<=1)
            for aa,bb in [(a,b),(b,a)]:
                model.addConstr(y[aa,bb,m]<=x[a,m]);model.addConstr(y[aa,bb,m]<=x[b,m])
    for m in M:
        for a in O:
            model.addConstr(e[a,m]<=B*x[a,m]); model.addConstr(e[a,m]<=C[a])
            model.addConstr(e[a,m]>=C[a]-B*(1-x[a,m]))
        model.addGenConstrMax(E[m],[e[a,m] for a in O],constant=0)
        model.addConstr(idle[m]==E[m]-gp.quicksum(times[a].get(m,0)*x[a,m] for a in O))
    model.addGenConstrMax(cmax,list(E.values()),constant=0)
    direct=direct or {}
    # q_ks is substituted by z_ks using Eqs. (17)-(18); no quantity assumption is changed.
    cost=gp.quicksum(v[1]*z[k,s] for k in K for s,v in enumerate(purchases[k]['options']))+gp.quicksum(direct.get((a,m),0)*x[a,m] for a in O for m in M)+0.5*gp.quicksum(h.values())+0.2*gp.quicksum(idle.values())
    if fixed:
        for a,(m,start) in fixed.items():model.addConstr(x[a,m]==1);model.addConstr(T[a]==start)
    return model,dict(T=T,C=C,x=x,A=A,h=h,u=u,E=E,idle=idle,cmax=cmax,cost=cost,M=M,O=O,K=K,types=types,times=times,purchases=purchases,demand=demand,arcs=arcs,setup=setup)

def independent_check(v):
    eps=1e-5
    rows=sorted([(v['T'][a].X,a,next(m for m in v['M'] if v['x'][a,m].X>.5),v['C'][a].X) for a in v['O']])
    last={}; prev={}; nonproc=0
    for start,a,m,end in rows:
        assert abs(end-start-v['times'][a][m])<eps
        assert start+eps>=last.get(m,0)+(v['setup'] if m in prev and prev[m]!=v['types'][a] else 0)
        nonproc+=start-last.get(m,0);last[m]=end;prev[m]=v['types'][a]
    for a,b in v['arcs']: assert v['T'][b].X+eps>=v['C'][a].X
    for k in v['K']:
        consumers=[a for a in v['O'] if v['u'][a,k].X>.5];assert len(consumers)==1
        a=consumers[0];assert abs(v['h'][k].X-(v['T'][a].X-v['A'][k].X))<eps
        assert v['h'][k].X>=-eps
    for a in v['O']:
        for r in set(p['class'] for p in v['purchases']):
            assert sum(v['u'][a,k].X>.5 for k in v['K'] if v['purchases'][k]['class']==r)==v['demand'].get((a,r),0)
    assert abs(sum(a.X for a in v['idle'].values())-nonproc)<eps
    assert abs(v['cmax'].X-max(last.values(),default=0))<eps
    for m in v['M']:assert abs(v['E'][m].X-last.get(m,0))<eps
    return rows

def main():
    results=[]
    fixtures=[
      ('one_operation',{0:'A'},{0:{0:3}},[],{},[],None,3,0),
      ('precedence_chain',{0:'A',1:'A'},{0:{0:3},1:{1:4}},[],{},[(0,1)],None,7,.6),
      ('different_type_setup',{0:'A',1:'B'},{0:{0:3},1:{0:4}},[],{},[],None,9,.4),
      ('same_type_no_setup',{0:'A',1:'A'},{0:{0:3},1:{0:4}},[],{},[],None,7,0),
      ('fork_join',{0:'A',1:'A',2:'A'},{0:{0:2},1:{1:3},2:{2:1}},[],{},[(0,2),(1,2)],None,4,.6),
      ('purchase_wait_unused_machine',{0:'A'},{0:{0:3,1:4}},[{'class':('A','p'),'options':[(5,10)]}],{(0,('A','p')):1},[],{0:(0,5)},8,11),
      ('pooled_units_independent_suppliers',{0:'A',1:'A'},{0:{0:2},1:{0:2}},[{'class':('A','p'),'options':[(1,10),(4,1)]},{'class':('A','p'),'options':[(1,10),(4,1)]}],{(0,('A','p')):1,(1,('A','p')):1},[],None,5,3.8),
    ]
    for name,types,times,purchases,demand,arcs,fixed,expected_time,expected_cost in fixtures:
        for objective,expected in [('time',expected_time),('cost',expected_cost)]:
            model,v=build(types,times,purchases,demand,arcs,fixed=fixed)
            model.setObjective(v['cmax'] if objective=='time' else v['cost']);model.optimize()
            assert model.Status==GRB.OPTIMAL,(name,model.Status)
            independent_check(v)
            assert abs(model.ObjVal-expected)<1e-5,(name,objective,model.ObjVal,expected)
            results.append(dict(case=name,objective=objective,value=model.ObjVal,expected=expected,status='OPTIMAL'))
    # Fast expensive vs slow cheap machine: include nonzero assembly direct costs.
    for objective,expected in [('time',2),('cost',1)]:
        model,v=build({0:'A'},{0:{0:2,1:5}},direct={(0,0):10,(0,1):1})
        model.setObjective(v['cmax'] if objective=='time' else v['cost']);model.optimize()
        assert model.Status==GRB.OPTIMAL;independent_check(v)
        assert abs(model.ObjVal-expected)<1e-5
        results.append(dict(case='assembly_machine_cost_tradeoff',objective=objective,value=model.ObjVal,expected=expected,status='OPTIMAL'))
    model,v=build({0:'A'},{0:{0:3}},broken=True)
    model.optimize();assert model.Status==GRB.INFEASIBLE
    model.computeIIS()
    results.append(dict(case='original_self_pair',status='INFEASIBLE',iis=[c.ConstrName for c in model.getConstrs() if c.IISConstr]))
    # Positive durations, mutually reversed precedence are infeasible even without self-pairs.
    model,v=build({0:'A',1:'A'},{0:{0:1},1:{1:1}},arcs=[(0,1),(1,0)])
    model.optimize();assert model.Status==GRB.INFEASIBLE
    results.append(dict(case='all_distinct_pairs_precedence',status='INFEASIBLE'))
    # Same local part name across product types cannot cover a missing B unit.
    model,v=build({0:'B'},{0:{0:1}},[{'class':('A','p'),'options':[(1,0)]}],{(0,('B','p')):1})
    model.optimize();assert model.Status==GRB.INFEASIBLE
    results.append(dict(case='cross_type_stock_rejected',status='INFEASIBLE'))
    # With a self-pair, the original Eq. (12) requires 1 <= 2*y <= 1
    # for an assigned operation. No binary y can satisfy that condition.
    model=gp.Model();model.Params.OutputFlag=0
    ys=model.addVar(vtype=GRB.BINARY)
    model.addConstr(2*ys>=1);model.addConstr(2*ys<=1)
    model.optimize();assert model.Status==GRB.INFEASIBLE
    results.append(dict(case='machine_self_pair_eq12',status='INFEASIBLE'))
    # Fix actual decoder schedules into ALL revised model constraints.
    from types import SimpleNamespace
    import random
    sys.path.insert(0,str(Path(__file__).parent/'type_revision'))
    from decoder import decode
    from objective import evaluate
    config=SimpleNamespace(num_tasks=8,product_type_of_instance={0:'A',1:'A',2:'B',3:'B'},
        task_to_product={i:i%4 for i in range(8)},task_to_part={i:'p' for i in range(4)},
        task_stage={i:('采购' if i<4 else '部装') for i in range(8)},
        num_suppliers={i:([0,1] if i<4 else [10,11]) for i in range(8)},
        precedence={i:[] for i in range(8)},assembly_requirements={i:['p'] for i in range(4,8)},
        product_change_penalty=2,machine_task_map={})
    for i in range(4):
        config.machine_task_map[0,i]={'time':1,'cost':10}
        config.machine_task_map[1,i]={'time':4,'cost':1}
    for i in range(4,8):
        config.machine_task_map[10,i]={'time':2,'cost':0}
        config.machine_task_map[11,i]={'time':3,'cost':0}
    random.seed(42)
    for iteration in range(20):
        particle=SimpleNamespace(position=[random.choice(config.num_suppliers[i]) for i in range(8)]+[random.random() for i in range(8)])
        schedule=decode(particle,config); assert len(schedule)==8
        expected_t,expected_c=evaluate(schedule,config,update_global=False)
        types={i:config.product_type_of_instance[i%4] for i in range(4,8)}
        purchases=[{'class':(config.product_type_of_instance[i],'p'),'options':[(1,10),(4,1)]} for i in range(4)]
        demand={(i,(types[i],'p')):1 for i in types}
        fixed={r['task']:(r['resource'],r['start']) for r in schedule if r['task']>=4}
        model,v=build(types,{i:{10:2,11:3} for i in types},purchases,demand,fixed=fixed)
        for k in range(4):model.addConstr(v['A'][k]==config.machine_task_map[particle.position[k],k]['time'])
        consumed={(r['task'],unit['purchase_task']) for r in schedule if r['task']>=4 for unit in r['consumed_units']}
        for pair,var in v['u'].items():model.addConstr(var==int(pair in consumed))
        model.setObjective(v['cost']);model.optimize();assert model.Status==GRB.OPTIMAL
        independent_check(v)
        assert abs(v['cmax'].X-expected_t)<1e-5
        assert abs(model.ObjVal-expected_c)<1e-5
    results.append(dict(case='decoder_fixed_schedule_full_model',schedules=20,status='PASS',checks='all constraints; exact makespan; total cost; unit consumption; typed changeovers'))
    output=dict(scope='Synthetic structural validation only; not reproduction of manuscript experiments',gurobi_version=gp.gurobi.version(),results=results)
    Path(__file__).with_name('full-model-validation.json').write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(output,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
