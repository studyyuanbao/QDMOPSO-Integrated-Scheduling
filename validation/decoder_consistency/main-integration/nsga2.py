import random
import numpy as np
from typing import List, Tuple
from particle import Particle
from decoder import decode
from objective import evaluate
from utils import update_archive, calculate_hypervolume, calculate_spacing, calculate_igd


def dominates_simple(a, b):
    """简单支配关系判断"""
    return a[0] <= b[0] and a[1] <= b[1] and (a[0] < b[0] or a[1] < b[1])


def non_dominated_sort(population):
    """非支配排序"""
    fronts = []
    remaining = population[:]

    while remaining:
        current_front = []
        for p1 in remaining:
            dominated = False
            for p2 in remaining:
                if p1 != p2 and dominates_simple(p2.fitness, p1.fitness):
                    dominated = True
                    break
            if not dominated:
                current_front.append(p1)
        fronts.append(current_front)
        remaining = [p for p in remaining if p not in current_front]
    return fronts


def simple_crossover(parent1, parent2):
    """最基础均匀交叉"""
    child1 = np.copy(parent1.position)
    child2 = np.copy(parent2.position)
    for i in range(len(parent1.position)):
        if random.random() < 0.5:
            child1[i], child2[i] = parent2.position[i], parent1.position[i]
    return child1, child2


def simple_mutation(position, config, mutation_prob=0.1):
    """最基础变异"""
    mutated = np.copy(position)
    for i in range(len(mutated)):
        if random.random() < mutation_prob:
            if i < config.num_tasks:
                mutated[i] = np.random.randint(0, 2)
            else:
                delta = (random.random() - 0.5) * 0.1
                mutated[i] += delta
                mutated[i] = np.clip(mutated[i], 0, 1)
    return mutated


def calculate_crowding_distance(front):
    """简化拥挤度，只保留边界点为inf"""
    if not front:
        return
    front[0].crowding_distance = float('inf')
    front[-1].crowding_distance = float('inf')
    for p in front[1:-1]:
        p.crowding_distance = 0.0


def tournament_selection(population, tournament_size=2):
    """最简单锦标赛选择"""
    tournament = random.sample(population, min(tournament_size, len(population)))
    return min(tournament, key=lambda p: sum(p.fitness))


def run_nsga2(config):
    """最基础 NSGA-II 实现"""
    global pareto_front
    all_fitness: List[List[Tuple[float, float]]] = []
    global_min_time_list: List[float] = []
    global_min_cost_list: List[float] = []
    hv_history: List[float] = []
    igd_values: List[float] = []
    archive_size_history: List[int] = []
    final_sp = 0.0

    # 初始化种群
    population = [Particle(config, feasible_init=True) for _ in range(config.num_particles)]
    for ind in population:
        ind.schedule = decode(ind, config)
        if ind.schedule:
            ind.fitness = evaluate(ind.schedule, config)

    valid = [p for p in population if p.schedule]
    global_best_time = min([p.fitness[0] for p in valid], default=float('inf'))
    global_best_cost = min([p.fitness[1] for p in valid], default=float('inf'))
    global_min_time_list.append(global_best_time)
    global_min_cost_list.append(global_best_cost)

    # 加载参考集
    try:
        reference_set = np.load('reference_set.npy')
    except Exception:
        reference_set = None

    ref_point = getattr(config, 'hv_ref_point', (120, 600))
    print("开始 NSGA-Ⅱ 优化...")
    for gen in range(config.num_iterations):
        # 非支配排序
        fronts = non_dominated_sort(population)
        for front in fronts:
            calculate_crowding_distance(front)

        # 子代生成
        offspring = []
        while len(offspring) < config.num_particles:
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child1_pos, child2_pos = simple_crossover(parent1, parent2)
            # 对交叉结果进行变异
            child1_pos = simple_mutation(child1_pos, config)
            child2_pos = simple_mutation(child2_pos, config)

            # 创建子代粒子
            child1 = Particle(config)
            child1.position = child1_pos
            child1.resource_assignment = child1_pos[:config.num_tasks]
            child1.priority_values = child1_pos[config.num_tasks:]
            child1.schedule = decode(child1, config)
            if child1.schedule:
                child1.fitness = evaluate(child1.schedule, config)

            child2 = Particle(config)
            child2.position = child2_pos
            child2.resource_assignment = child2_pos[:config.num_tasks]
            child2.priority_values = child2_pos[config.num_tasks:]
            child2.schedule = decode(child2, config)
            if child2.schedule:
                child2.fitness = evaluate(child2.schedule, config)

            offspring.extend([child1, child2])

        # 合并父代 + 子代
        combined = population + offspring
        fronts = non_dominated_sort(combined)
        # 下一代填充
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= config.num_particles:
                new_population.extend(front)
            else:
                remaining = config.num_particles - len(new_population)
                new_population.extend(front[:remaining])
                break
        population = new_population

        valid_pop = [p for p in population if p.schedule]
        if valid_pop:
            all_fitness.append([p.fitness for p in valid_pop])
            global_best_time = min(global_best_time, min(p.fitness[0] for p in valid_pop))
            global_best_cost = min(global_best_cost, min(p.fitness[1] for p in valid_pop))
        global_min_time_list.append(global_best_time)
        global_min_cost_list.append(global_best_cost)

        # 记录Pareto前沿大小
        pareto_front = fronts[0] if fronts else []
        archive_size_history.append(len(pareto_front))

        # HV指标
        if pareto_front:
            pareto_fitness = [p.fitness for p in pareto_front if
                              all(v is not None and v != float('inf') for v in p.fitness)]
            hv_now = calculate_hypervolume(pareto_fitness, ref_point) if pareto_fitness else 0.0
        else:
            hv_now = 0.0
        hv_history.append(hv_now)

        # IGD指标
        if reference_set is not None and pareto_front:
            pareto_fitness = [p.fitness for p in pareto_front if
                              all(v is not None and v != float('inf') for v in p.fitness)]
            reference_fitness = [tuple(r) for r in reference_set]
            igd_values.append(calculate_igd(pareto_fitness, reference_fitness))
        else:
            igd_values.append(float('nan'))

        # Spacing
        if pareto_front:
            pareto_fitness = [p.fitness for p in pareto_front if
                              all(v is not None and v != float('inf') for v in p.fitness)]
            final_sp = calculate_spacing(pareto_fitness) if pareto_fitness else 0.0

        if gen % 50 == 0 or gen == config.num_iterations - 1:
            print(
                f"Gen {gen:3d}: Pop={len(population)} | Pareto={len(pareto_front)} | HV={hv_now:.2f} | Best Time={global_best_time:.1f} | Best Cost={global_best_cost:.1f}")

    return (
        [],  # convergence_archive
        pareto_front,  # diversity_archive
        all_fitness,
        global_min_time_list,
        global_min_cost_list,
        archive_size_history,
        {},  # q_table
        hv_history,
        final_sp,
        igd_values,
    )
