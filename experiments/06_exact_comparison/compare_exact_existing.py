from pathlib import Path
import json,csv,statistics,shutil
B=Path(__file__).resolve().parents[1];P=B/'work/exact-s1';E=json.loads((P/'epsilon-results.json').read_text())
ends=json.loads((P/'endpoints.json').read_text());best_t=ends[0]['objective'];best_c=ends[1]['objective'];rows=[];comparisons=[]
for seed in range(44001,44006):
 with (B/f'work/reward-ablation/A-{seed}/train-evaluations.csv').open(encoding='utf-8-sig') as f:values=[(float(r['time']),float(r['cost'])) for r in csv.DictReader(f)]
 t=min(x for x,y in values);c=min(y for x,y in values)
 rows.append({'seed':seed,'nfe':len(values),'best_time':t,'time_gap_percent':100*(t-best_t)/best_t,'best_cost':c,'cost_gap_percent':100*(c-best_c)/best_c})
 for e in E:
  if e.get('epsilon') is None:continue
  feasible=[x for x,y in values if y<=e['epsilon']+1e-5]
  comparisons.append({'seed':seed,'label':e['label'],'cost_limit':e['epsilon'],'exact_time':e.get('time'),'algorithm_best_time_within_cost':min(feasible) if feasible else None,'feasible_evaluations':len(feasible)})
for filename,data in [('algorithm-endpoint-comparison.csv',rows),('algorithm-epsilon-comparison.csv',comparisons)]:
 with (P/filename).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
table='\n'.join(f"|{r['seed']}|{r['nfe']}|{r['best_time']:.6f}|{r['time_gap_percent']:.3f}%|{r['best_cost']:.6f}|{r['cost_gap_percent']:.3f}%|" for r in rows)
text=f'''# S1精确方法与现有算法结果对照（正文及回复草稿）

固定含扰动S1，数据种子42，21任务；A组为50粒子、100代、在线学习，奖励权重HV=.8、SP=.2及切换惩罚.01。复用现有5次全部目标评价结果，没有新增启发式运行。这里是搜索期间找到的最好值，不是末代有限档案，两个目标最小值通常来自不同排程。

Gurobi单目标最优：完工时间{best_t:.8f}，总成本{best_c:.8f}。它们构成理想点坐标，不代表存在一个同时达到两者的解。

|种子|实际NFE|找到的最短工期|相对最优差距|找到的最低成本|相对最优差距|
|---|---:|---:|---:|---:|---:|
{table}

相对差距=(算法最好值−已证明单目标最优值)/单目标最优值。另附algorithm-epsilon-comparison.csv，对每个成本上限从已有全部评价中筛选满足上限的最好工期；没有满足者留空，不填0或随意惩罚值。数值可行比较使用1e-5容差，精确求解约束采用1e-7松弛及求解器自身容差。

## 正文可用描述

We compared the revised formulation with the heuristic on the fixed S1 instance. Gurobi certified a minimum makespan of {best_t:.5f} and a minimum total cost of {best_c:.5f}, obtained from separate single-objective optimizations. A finite epsilon-constraint study minimized makespan under ten prescribed cost limits, followed by secondary cost minimization. The solver reported optimal solutions at all sampled limits. These sampled solutions do not constitute an enumeration of the complete continuous Pareto frontier. The heuristic comparison uses all solutions evaluated in five predefined runs, and reports both endpoint gaps and the best makespan found within each sampled cost limit.

## AE回复草稿

中文：感谢副编辑要求补充精确方法比较。我们已在与启发式一致的固定S1输入上，使用Gurobi求解修订后的双目标模型。除分别求解两个单目标端点外，我们在10个预定成本限制下最小化完工时间，并进行二级成本优化。各采样水平均获求解器最优状态。我们将这些结果与5个预定种子的算法运行中实际找到的解比较，报告端点差距及相同成本限制下的最好工期，并保存阈值、最优界、求解状态、耗时与排程。我们明确区分有限水平的最优解与完整Pareto前沿。

English: Thank you for requesting a comparison with an exact method. We solved the revised bi-objective formulation in Gurobi using the same fixed S1 input as the heuristic. In addition to the two single-objective endpoints, makespan was minimized under ten prescribed cost limits, followed by secondary cost minimization. The solver reported optimality at all sampled limits. These results were compared with solutions actually found in five predefined heuristic runs, reporting endpoint gaps and the best makespan found under the same cost limits. Thresholds, bounds, solver status, runtime and schedules are retained. We explicitly distinguish optimal solutions at finitely sampled limits from the complete Pareto frontier.

## 提交边界（不可删除后当成已全部完成）

- 本方法当前实现为有限epsilon限制加二级优化。AE使用restricted-epsilon这一术语的具体定义/引用尚未最终核实；在核实前不可仅改名宣称完全满足特定方法要求。
- 10个水平用初次最短工期求解所得成本作为上端，因此较宽，若干水平重复；保留全部记录，不隐藏。不能用这些稀疏点充当真实IGD参考前沿。
- 时间界压缩论证与正文符号仍需写入最终稿；比较适用固定采购、单调持有及非加工费用和现有工艺/机器约束。
- 启发式来自100代在线学习诊断，不是预训练后测试。精确比较不是严格相同CPU或NFE预算排名。
- 本文件为表格与回复草稿，尚未写入Word、未补页码，尚未公开仓库。不能说这些编辑工作已经完成。
'''
(B/'outputs/S1精确比较表与AE回复草稿.md').write_text(text,encoding='utf8')
shutil.copy2(B/'outputs/S1精确比较表与AE回复草稿.md',P/'比较说明.md')
shutil.make_archive(str(B/'outputs/S1精确方法端点求解记录'),'zip',P)
print(table)
