import random
import numpy as np
from typing import List, Tuple

from particle import Particle
from decoder import decode
from objective import evaluate
from utils import update_archive, calculate_hypervolume, calculate_spacing, calculate_igd
from pso import sbx_crossover, polynomial_mutation


def run_spea2(config):
    """
    SPEA2 最小可行实现，复用现有编码/解码/评估与遗传操作，
    使用双档案作为外部精英库，返回与 pso.run_mopso 同结构数据。
    """
    archive_size_history: List[int] = []
    all_fitness: List[List[Tuple[float, float]]] = []
    global_min_time_list: List[float] = []
    global_min_cost_list: List[float] = []
    hv_history: List[float] = []
    igd_values: List[float] = []
    final_sp = 0.0

    # 初始化种群
    population = [Particle(config, feasible_init=True) for _ in range(config.num_particles)]
    for ind in population:
        ind.schedule = decode(ind, config)
        if ind.schedule:
            ind.fitness = evaluate(ind.schedule, config)
            ind.update_pbest()

    valid = [p for p in population if p.schedule]
    convergence_archive, diversity_archive = update_archive(valid, [], [], config)
    archive_size_history.append(len(convergence_archive) + len(diversity_archive))

    ref_point = getattr(config, 'hv_ref_point', (120, 600))
    if diversity_archive:
        pareto0 = [p.fitness for p in diversity_archive if all(v is not None and v != float('inf') for v in p.fitness)]
        hv_history.append(calculate_hypervolume(pareto0, ref_point) if pareto0 else 0.0)
    else:
        hv_history.append(0.0)

    global_best_time = min([p.fitness[0] for p in valid], default=float('inf'))
    global_best_cost = min([p.fitness[1] for p in valid], default=float('inf'))
    global_min_time_list.append(global_best_time)
    global_min_cost_list.append(global_best_cost)

    try:
        reference_set = np.load('reference_set.npy')
    except Exception:
        reference_set = None

    for _ in range(config.num_iterations):
        # 选择（SPEA2 会用强度与密度，这里简化为按目标和排序+保留多样性）
        pool = sorted(list(convergence_archive) + list(diversity_archive), key=lambda p: p.fitness[0] + p.fitness[1])
        if not pool:
            pool = valid
        parents = pool[: max(2, len(pool) // 2)]

        # 交叉+变异
        offspring: List[Particle] = []
        random.shuffle(parents)
        usable = len(parents) if len(parents) % 2 == 0 else len(parents) - 1
        for i in range(0, usable, 2):
            c1_pos, c2_pos = sbx_crossover(parents[i].position, parents[i + 1].position, config)
            c1_pos = polynomial_mutation(c1_pos, config)
            c2_pos = polynomial_mutation(c2_pos, config)
            c1 = Particle(config)
            c1.position = c1_pos
            c1.resource_assignment = c1_pos[:config.num_tasks]
            c1.priority_values = c1_pos[config.num_tasks:]
            c2 = Particle(config)
            c2.position = c2_pos
            c2.resource_assignment = c2_pos[:config.num_tasks]
            c2.priority_values = c2_pos[config.num_tasks:]
            offspring.extend([c1, c2])

        # 解码评估
        new_valid: List[Particle] = []
        for child in offspring:
            child.schedule = decode(child, config)
            if child.schedule:
                child.fitness = evaluate(child.schedule, config)
                new_valid.append(child)

        # 更新外部档案（双档案）
        if new_valid:
            convergence_archive, diversity_archive = update_archive(new_valid, convergence_archive, diversity_archive, config)

        # 记录
        all_fitness.append([p.fitness for p in new_valid])
        archive_size_history.append(len(convergence_archive) + len(diversity_archive))
        if diversity_archive:
            pareto = [p.fitness for p in diversity_archive if all(v is not None and v != float('inf') for v in p.fitness)]
            hv_now = calculate_hypervolume(pareto, ref_point) if pareto else 0.0
        else:
            hv_now = 0.0
        hv_history.append(hv_now)
        # 转换IGD和Spacing的输入格式
        diversity_fitness = [p.fitness for p in diversity_archive if 
                           all(val is not None and val != float('inf') for val in p.fitness)]
        reference_fitness = [tuple(r) for r in reference_set] if reference_set is not None else None
        
        final_sp = calculate_spacing(diversity_fitness) if diversity_fitness else 0.0
        if reference_fitness is not None:
            igd_values.append(calculate_igd(diversity_fitness, reference_fitness))
        else:
            igd_values.append(float('nan'))

        if new_valid:
            global_best_time = min(global_best_time, min(p.fitness[0] for p in new_valid))
            global_best_cost = min(global_best_cost, min(p.fitness[1] for p in new_valid))
        global_min_time_list.append(global_best_time)
        global_min_cost_list.append(global_best_cost)

    return (
        convergence_archive,
        diversity_archive,
        all_fitness,
        global_min_time_list,
        global_min_cost_list,
        archive_size_history,
        {},
        hv_history,
        final_sp,
        igd_values,
    )





