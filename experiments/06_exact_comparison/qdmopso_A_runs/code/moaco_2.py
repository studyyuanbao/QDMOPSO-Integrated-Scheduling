import random
import numpy as np
from typing import List, Tuple
from particle import Particle
from decoder import decode
from objective import evaluate
from utils import dominates, calculate_hypervolume, calculate_spacing, calculate_igd


class Ant:
    """蚂蚁类，用于构建解"""

    def __init__(self, config):
        self.config = config
        self.position = [0.0] * (2 * config.num_tasks)
        self.resource_assignment = [0] * config.num_tasks
        self.priority_values = [0.0] * config.num_tasks
        self.schedule = []
        self.fitness = [float('inf'), float('inf')]
        self._decoded = False  # 标记是否已解码

    def construct_solution_fast(self, pheromone_matrix, heuristic_matrix, alpha=1.0, beta=2.0):
        """快速构建解 - 减少不必要的计算"""
        # 重用现有的数组来减少内存分配
        for i in range(self.config.num_tasks):
            available_resources = self.config.num_suppliers[i]
            if not available_resources:
                self.resource_assignment[i] = 0
                continue

            # 快速概率计算
            if len(available_resources) == 1:
                # 只有一个选择时直接使用
                self.resource_assignment[i] = available_resources[0]
            else:
                # 预计算概率
                max_prob = 0
                best_resource = available_resources[0]

                for resource in available_resources:
                    pheromone = pheromone_matrix[i].get(resource, 1.0)
                    heuristic = heuristic_matrix[i].get(resource, 1.0)
                    prob = (pheromone ** alpha) * (heuristic ** beta)

                    if prob > max_prob or (prob == max_prob and random.random() < 0.5):
                        max_prob = prob
                        best_resource = resource

                self.resource_assignment[i] = best_resource

            self.position[i] = self.resource_assignment[i]

        # 快速生成优先级
        for i in range(self.config.num_tasks):
            self.priority_values[i] = random.random()
            self.position[self.config.num_tasks + i] = self.priority_values[i]

        self._decoded = False

    def evaluate_solution(self):
        """评估解（惰性评估）"""
        if not self._decoded:
            self.schedule = decode(self, self.config)
            if self.schedule:
                self.fitness = evaluate(self.schedule, self.config)
            else:
                self.fitness = [float('inf'), float('inf')]
            self._decoded = True
        return self.fitness


def initialize_pheromone_matrix_fast(config, initial_value=1.0):
    """快速初始化信息素矩阵"""
    pheromone_matrix = {}
    for task_id in range(config.num_tasks):
        resources = config.num_suppliers[task_id]
        pheromone_matrix[task_id] = {resource: initial_value for resource in resources}
    return pheromone_matrix


def initialize_heuristic_matrix_fast(config):
    """快速初始化启发式信息矩阵"""
    heuristic_matrix = {}
    machine_task_map = config.machine_task_map

    for task_id in range(config.num_tasks):
        resources = config.num_suppliers[task_id]
        heuristic_matrix[task_id] = {}

        for resource in resources:
            key = (resource, task_id)
            if key in machine_task_map:
                cost = machine_task_map[key].get('cost', 1.0)
                time = machine_task_map[key].get('time', 1.0)
            else:
                cost = time = 1.0

            heuristic_matrix[task_id][resource] = 1.0 / (cost + time + 1e-6)

    return heuristic_matrix


def update_pheromone_fast(pheromone_matrix, archive, rho=0.1, Q=100.0):
    """快速信息素更新"""
    # 批量蒸发
    for task_id in pheromone_matrix:
        for resource in pheromone_matrix[task_id]:
            new_value = pheromone_matrix[task_id][resource] * (1 - rho)
            pheromone_matrix[task_id][resource] = max(new_value, 0.1)

    # 批量增强 - 限制增强的解数量
    if archive:
        # 只使用前几个解来增强，减少计算量
        max_solutions = min(5, len(archive))
        for ant in archive[:max_solutions]:
            if ant.fitness[0] != float('inf'):
                delta_pheromone = Q / (sum(ant.fitness) + 1e-6)

                for task_id, resource in enumerate(ant.resource_assignment):
                    if (task_id in pheromone_matrix and
                            resource in pheromone_matrix[task_id]):
                        pheromone_matrix[task_id][resource] += delta_pheromone


def update_archive_fast(new_ants, archive, config, max_size=50):
    """快速档案更新 - 简化非支配排序"""
    # 合并解
    all_solutions = new_ants + archive

    # 快速过滤有效解
    valid_solutions = []
    for ant in all_solutions:
        if (ant.fitness[0] != float('inf') and
                ant.fitness[1] != float('inf') and
                not np.isnan(ant.fitness[0]) and
                not np.isnan(ant.fitness[1])):
            valid_solutions.append(ant)

    if not valid_solutions:
        return archive

    # 如果解数量不多，直接返回
    if len(valid_solutions) <= max_size:
        return valid_solutions

    # 简化非支配排序 - 只进行一轮
    non_dominated = []
    for i, p in enumerate(valid_solutions):
        dominated = False
        for j, q in enumerate(valid_solutions):
            if i != j and dominates(q.fitness, p.fitness):
                dominated = True
                break
        if not dominated:
            non_dominated.append(p)

    # 如果非支配解还是太多，按拥挤距离截断
    if len(non_dominated) > max_size:
        # 简化的拥挤距离计算
        if len(non_dominated) >= 2:
            # 按时间排序
            sorted_by_time = sorted(non_dominated, key=lambda x: x.fitness[0])
            time_range = sorted_by_time[-1].fitness[0] - sorted_by_time[0].fitness[0]

            # 按成本排序
            sorted_by_cost = sorted(non_dominated, key=lambda x: x.fitness[1])
            cost_range = sorted_by_cost[-1].fitness[1] - sorted_by_cost[0].fitness[1]

            # 计算简化的拥挤距离
            crowding_scores = []
            for ant in non_dominated:
                score = 0
                if time_range > 0:
                    # 找到相邻解
                    time_idx = sorted_by_time.index(ant)
                    if time_idx > 0 and time_idx < len(sorted_by_time) - 1:
                        score += (sorted_by_time[time_idx + 1].fitness[0] -
                                  sorted_by_time[time_idx - 1].fitness[0]) / time_range

                if cost_range > 0:
                    cost_idx = sorted_by_cost.index(ant)
                    if cost_idx > 0 and cost_idx < len(sorted_by_cost) - 1:
                        score += (sorted_by_cost[cost_idx + 1].fitness[1] -
                                  sorted_by_cost[cost_idx - 1].fitness[1]) / cost_range

                crowding_scores.append(score)

            # 按拥挤距离排序并截断
            sorted_indices = np.argsort(crowding_scores)[::-1]  # 降序
            non_dominated = [non_dominated[i] for i in sorted_indices[:max_size]]
        else:
            non_dominated = non_dominated[:max_size]

    return non_dominated


def safe_extract_fitness_tuples_fast(ants):
    """快速提取适应度元组"""
    return [(ant.fitness[0], ant.fitness[1]) for ant in ants
            if (hasattr(ant, 'fitness') and
                ant.fitness[0] != float('inf') and
                ant.fitness[1] != float('inf') and
                not np.isnan(ant.fitness[0]) and
                not np.isnan(ant.fitness[1]))]


def run_moaco_fast(config):
    """运行优化的多目标人工蚁群算法 - 速度优化版"""
    print("开始快速MOACO优化...")

    # 优化参数
    num_ants = min(config.num_particles, 30)  # 限制蚂蚁数量以提高速度
    alpha = 1.0
    beta = 2.0
    rho = 0.1
    Q = 100.0

    # 预初始化
    print("初始化信息素和启发式矩阵...")
    pheromone_matrix = initialize_pheromone_matrix_fast(config)
    heuristic_matrix = initialize_heuristic_matrix_fast(config)

    # 记录变量
    archive_size_history = []
    all_fitness = []
    global_min_time_list = []
    global_min_cost_list = []
    hv_history = []
    igd_values = []

    global_best_time = float('inf')
    global_best_cost = float('inf')

    # 初始化档案
    archive = []
    ants_pool = [Ant(config) for _ in range(num_ants)]  # 预创建蚂蚁池

    # 加载参考集
    try:
        reference_set = np.load('reference_set.npy')
    except Exception:
        reference_set = None

    # ===== 参考点设置 =====
    ref_point = getattr(config, 'hv_ref_point', None)
    if ref_point is None:
        # 快速初始化参考点
        print("估算参考点...")
        for ant in ants_pool:
            ant.construct_solution_fast(pheromone_matrix, heuristic_matrix, alpha, beta)
            ant.evaluate_solution()

        fitness_init = safe_extract_fitness_tuples_fast(ants_pool)
        if fitness_init:
            max_time = max(t for t, c in fitness_init)
            max_cost = max(c for t, c in fitness_init)
            ref_point = (max_time * 1.5 + 1.0, max_cost * 1.5 + 1.0)
        else:
            ref_point = (20000.0, 20000.0)
        config.hv_ref_point = ref_point
    else:
        ref_point = tuple(ref_point)

    print(f"使用的 HV 参考点 (ref_point) = {ref_point}")
    print(f"蚂蚁数量: {num_ants}, 迭代次数: {config.num_iterations}")

    # 主循环 - 优化版
    for iteration in range(config.num_iterations):
        # 1. 重用蚂蚁构建解
        new_ants = []
        for ant in ants_pool:
            ant.construct_solution_fast(pheromone_matrix, heuristic_matrix, alpha, beta)
            ant.evaluate_solution()  # 惰性评估
            new_ants.append(ant)

        # 2. 快速更新档案
        archive = update_archive_fast(new_ants, archive, config)

        # 3. 每5代更新一次信息素以减少计算量
        if iteration % 5 == 0 and archive:
            update_pheromone_fast(pheromone_matrix, archive, rho, Q)

        # --- 记录指标 (减少记录频率) ---
        if iteration % 2 == 0:  # 每2代记录一次
            valid_ants = [ant for ant in new_ants if ant.fitness[0] != float('inf')]
            if valid_ants:
                current_fitness = [ant.fitness for ant in valid_ants]
                all_fitness.append(current_fitness)

                current_best_time = min(ant.fitness[0] for ant in valid_ants)
                current_best_cost = min(ant.fitness[1] for ant in valid_ants)

                global_best_time = min(global_best_time, current_best_time)
                global_best_cost = min(global_best_cost, current_best_cost)

        global_min_time_list.append(global_best_time)
        global_min_cost_list.append(global_best_cost)
        archive_size_history.append(len(archive))

        # 计算指标 - 每10代计算一次以减少开销
        if iteration % 10 == 0 or iteration == config.num_iterations - 1:
            pareto_fitness = [ant.fitness for ant in archive
                              if ant.fitness[0] != float('inf') and ant.fitness[1] != float('inf')]

            if pareto_fitness:
                hv = calculate_hypervolume(pareto_fitness, ref_point)
                igd = calculate_igd(pareto_fitness, reference_set) if reference_set is not None else float('nan')
            else:
                hv = 0.0
                igd = float('nan')

            hv_history.append(hv)
            igd_values.append(igd)
        else:
            # 保持上一次的值
            if hv_history:
                hv_history.append(hv_history[-1])
                igd_values.append(igd_values[-1])
            else:
                hv_history.append(0.0)
                igd_values.append(float('nan'))

        # 输出进度 - 减少输出频率
        if iteration % 100 == 0 or iteration == config.num_iterations - 1:
            current_hv = hv_history[-1] if hv_history else 0.0
            current_igd = igd_values[-1] if igd_values else float('nan')
            print(f"Iter {iteration:3d}: Archive={len(archive):2d} | "
                  f"HV={current_hv:.2f} | IGD={current_igd:.4f} | "
                  f"Best Time={global_best_time:.1f} | Best Cost={global_best_cost:.1f}")

    # 计算最终Spacing
    if archive:
        pareto_fitness = [ant.fitness for ant in archive
                          if ant.fitness[0] != float('inf') and ant.fitness[1] != float('inf')]
        final_sp = calculate_spacing(pareto_fitness) if pareto_fitness else 0.0
    else:
        final_sp = 0.0

    print(f"优化完成！最终档案大小: {len(archive)}")

    # 返回结果
    return (
        archive,  # convergence_archive
        archive,  # diversity_archive
        all_fitness,
        global_min_time_list,
        global_min_cost_list,
        archive_size_history,
        {},  # q_table (空字典占位)
        hv_history,
        final_sp,
        igd_values,
    )


# 保持原函数名兼容性
run_moaco = run_moaco_fast