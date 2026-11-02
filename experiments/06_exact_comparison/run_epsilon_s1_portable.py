from pathlib import Path
import json,time,shutil
exec(compile(Path(__file__).with_name('run_exact_s1_portable.py').read_text(encoding='utf8').split('records=[]')[0],'exact-setup','exec'))
endpoints=json.loads((out/'endpoints.json').read_text());low=endpoints[1]['cost'];high=endpoints[0]['cost']
records=[]
# Endpoints are refined as well; epsilon sampling remains finite, not full-front enumeration.
cases=[('endpoint_time',None),('endpoint_cost',low)]+[(f'epsilon_{i:02d}',low+(high-low)*i/9) for i in range(10)]
for label,epsilon in cases:
 model,v=build(types,times,purchases,demand,arcs,2,direct=direct);model.Params.Threads=2;model.Params.Seed=42
 model.Params.LogFile=str(out/f'{label}.log');model.Params.TimeLimit=60
 if epsilon is not None:model.addConstr(v['cost']<=epsilon+1e-7,name='epsilon_cost')
 model.setObjective(v['cmax'],GRB.MINIMIZE);model.write(str(out/f'{label}.lp'));model.optimize()
 row={'label':label,'epsilon':epsilon,'phase1_status':model.Status,'phase1_bound':model.ObjBound,'phase1_runtime':model.Runtime}
 if model.SolCount:
  t=v['cmax'].X;row.update({'phase1_time':t,'phase1_cost':v['cost'].getValue(),'phase1_gap':model.MIPGap})
  model.addConstr(v['cmax']<=t+1e-7,name='lex_time');model.setObjective(v['cost'],GRB.MINIMIZE);model.Params.TimeLimit=60;model.optimize()
  row.update({'phase2_status':model.Status,'phase2_runtime':model.Runtime,'phase2_bound':model.ObjBound})
  if model.SolCount:
   independent_check(v);model.write(str(out/f'{label}.sol'));row.update({'time':v['cmax'].X,'cost':v['cost'].getValue(),'phase2_gap':model.MIPGap})
 records.append(row);(out/'epsilon-results.json').write_text(json.dumps(records,indent=2),encoding='utf8');print(json.dumps(row),flush=True);model.dispose()
shutil.copy2(Path(__file__),out/'run_epsilon_s1_portable.py')
(out/'说明.md').write_text('S1修订模型的端点与有限ε限制求解。输入为含扰动固定S1，与现有A组相同。端点结果四舍五入匹配历史两个数值，但不构成历史实验整体复现。ε上界取初次最短时间端点的可行成本，该端点当时尚未作成本二级优化；允许冗余区间，所有点保留。成本与时间限制含1e-7数值松弛。每个ε先最小时间，再在所得时间限制下最小成本；每阶段60秒。端点原始求解时限300秒。状态2表示Gurobi报告OPTIMAL，误差容限下最优，不代表有限采样枚举了连续完整前沿。LP、SOL、日志及JSON包含阈值、目标、界、间隙与耗时。尚未完成与启发式结果的统一对照，也尚未对AE术语来源作最终核对。时间上界论证须在正文明确：固定采购、物料分配和机器总序后，可将开始时间压缩为约束图的最早时刻，不增加持有或机器非加工成本；无环有效路径至多包含全部装配加工和切换及最大到货时长，故在本模型假设下采用构造B。若引入释放窗、负成本或其他上界约束须重新论证。',encoding='utf8')
shutil.make_archive(str(out.parent/'S1-exact-recheck'),'zip',out)
