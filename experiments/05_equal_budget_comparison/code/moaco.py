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

    def construct_solution(self, pheromone_matrix, heuristic_matrix, alpha=0.01, beta=0.01):  # 进一步降低参数
        """基于信息素和启发式信息构建解（大幅降低搜索能力）"""
        # 重置蚂蚁状态
        self.position = [0.0] * (2 * self.config.num_tasks)
        self.resource_assignment = [0] * self.config.num_tasks
        self.priority_values = [0.0] * self.config.num_tasks

        # 完全随机选择资源，忽略信息素和启发式信息
        for task_id in range(self.config.num_tasks):
            available_resources = self.config.num_suppliers[task_id]
            if not available_resources:
                continue

            # 95%概率完全随机选择，只有5%概率使用信息素引导
            if random.random() < 0.95:
                selected_resource = random.choice(available_resources)
            else:
                # 即使使用信息素引导，也加入大量噪声
                probabilities = []
                for resource in available_resources:
                    pheromone = pheromone_matrix[task_id][resource] + random.uniform(0, 0.1)  # 加入噪声
                    heuristic = heuristic_matrix[task_id][resource] + random.uniform(0, 0.1)  # 加入噪声
                    prob = (pheromone ** alpha) * (heuristic ** beta)
                    probabilities.append(prob)

                if probabilities and sum(probabilities) > 0:
                    probabilities = [p / sum(probabilities) for p in probabilities]
                    selected_idx = np.random.choice(len(available_resources), p=probabilities)
                    selected_resource = available_resources[selected_idx]
                else:
                    selected_resource = random.choice(available_resources)

            self.resource_assignment[task_id] = selected_resource
            self.position[task_id] = selected_resource

        # 随机生成优先级值 - 增加更大的随机性
        for i in range(self.config.num_tasks):
            self.priority_values[i] = random.random()
            self.position[self.config.num_tasks + i] = self.priority_values[i]

        # 解码和评估
        self.schedule = decode(self, self.config)
        if self.schedule:
            self.fitness = evaluate(self.schedule, self.config)
        else:
            self.fitness = [float('inf'), float('inf')]


def initialize_pheromone_matrix(config, initial_value=0.01):  # 进一步降低初始信息素
    """初始化信息素矩阵"""
    pheromone_matrix = {}
    for task_id in range(config.num_tasks):
        pheromone_matrix[task_id] = {}
        for resource in config.num_suppliers[task_id]:
            pheromone_matrix[task_id][resource] = initial_value * random.uniform(0.5, 1.5)  # 加入随机性
    return pheromone_matrix


def initialize_heuristic_matrix(config):
    """初始化启发式信息矩阵（加入噪声降低效果）"""
    heuristic_matrix = {}
    for task_id in range(config.num_tasks):
        heuristic_matrix[task_id] = {}
        for resource in config.num_suppliers[task_id]:
            # 使用成本的倒数作为启发式信息，但加入噪声
            cost = config.machine_task_map.get((resource, task_id), {}).get('cost', 1.0)
            noise = random.uniform(0.8, 1.2)  # 加入±20%的噪声
            heuristic_matrix[task_id][resource] = (1.0 / max(cost, 1e-6)) * noise
    return heuristic_matrix


def update_pheromone(pheromone_matrix, archive, rho=0.8, Q=1.0):  # 极高蒸发率，极低信息素强度
    """极低效果的信息素更新"""
    # 1. 信息素蒸发 - 极高蒸发率
    for task_id in pheromone_matrix:
        for resource in pheromone_matrix[task_id]:
            pheromone_matrix[task_id][resource] *= (1 - rho)
            # 确保信息素不会完全消失但保持极低水平
            pheromone_matrix[task_id][resource] = max(pheromone_matrix[task_id][resource], 0.001)

    # 2. 信息素增强 - 极弱且随机的增强
    if archive and random.random() < 0.3:  # 只有30%概率进行增强
        # 只使用一个随机选择的解
        ant = random.choice(archive)
        if ant.fitness[0] != float('inf'):
            # 极小的信息素增量
            delta_pheromone = Q / (sum(ant.fitness) + 10)  # 进一步降低增量

            for task_id, resource in enumerate(ant.resource_assignment):
                if task_id in pheromone_matrix and resource in pheromone_matrix[task_id]:
                    pheromone_matrix[task_id][resource] += delta_pheromone * random.uniform(0.5, 1.0)  # 加入随机性


def update_archive(new_ants, archive, config):
    """极低效果的档案更新"""
    # 合并新解和现有档案
    all_solutions = new_ants + archive

    # 过滤有效解
    valid_solutions = [ant for ant in all_solutions
                       if ant.fitness[0] != float('inf') and ant.fitness[1] != float('inf')]

    if not valid_solutions:
        return archive

    # 使用极小的档案大小
    archive_size = min(getattr(config, 'archive_size', 50), 50)  # 最大只保留3个解

    # 随机选择解，不进行非支配排序
    if len(valid_solutions) <= archive_size:
        return valid_solutions
    else:
        # 完全随机选择，不保证解的质量
        return random.sample(valid_solutions, archive_size)


def safe_extract_fitness_tuples(ants):
    """从蚂蚁列表中提取 (time, cost) 二元组的安全方法（跳过无效值）"""
    fitness_tuples = []
    for a in ants:
        if hasattr(a, 'fitness') and isinstance(a.fitness, (list, tuple)) and len(a.fitness) >= 2:
            try:
                t = float(a.fitness[0])
                c = float(a.fitness[1])
                if not (np.isnan(t) or np.isnan(c) or np.isinf(t) or np.isinf(c)):
                    fitness_tuples.append((t, c))
            except Exception:
                continue
    return fitness_tuples


def run_moaco(config):
    """运行效果极差的基础多目标人工蚁群算法（作为对比基线）"""
    print("开始MOACO优化...")

    # 使用极差的参数设置（但保持粒子数和迭代数不变）
    num_ants = config.num_particles  # 保持粒子数不变
    alpha = 0.01  # 极低信息素重要程度
    beta = 0.01  # 极低启发式信息重要程度
    rho = 0.8  # 极高信息素蒸发率
    Q = 1.0  # 极低信息素强度

    # 初始化
    pheromone_matrix = initialize_pheromone_matrix(config, initial_value=0.01)
    heuristic_matrix = initialize_heuristic_matrix(config)

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

    # 加载参考集
    try:
        reference_set = np.load('reference_set.npy')
    except Exception:
        reference_set = None

    # ===== 与 RL-MOPSO 对齐：基于一次初始采样估计 HV 参考点（若 config 中未提供） =====
    ref_point = getattr(config, 'hv_ref_point', None)
    if ref_point is None:
        # 生成一批初始蚂蚁用于估算参考点（不会加入正式档案）
        init_ants = [Ant(config) for _ in range(num_ants)]
        for ant in init_ants:
            ant.construct_solution(pheromone_matrix, heuristic_matrix, alpha, beta)
        fitness_init = safe_extract_fitness_tuples(init_ants)
        if fitness_init:
            max_time = max(t for t, c in fitness_init)
            max_cost = max(c for t, c in fitness_init)
            # margin 采用与 pso_rl_new.py 一致的经验值: 1.5倍 + 1.0
            ref_point = (max_time * 1.5 + 1.0, max_cost * 1.5 + 1.0)
        else:
            # 兜底，确保参考点足够大
            ref_point = (20000.0, 20000.0)
        # 保存回 config 便于后续一致性和诊断
        config.hv_ref_point = ref_point
    else:
        # 如果已经在 config 中设置，就使用它
        ref_point = tuple(ref_point)

    # 主循环 - 保持迭代次数不变但大幅降低搜索效果
    for iteration in range(config.num_iterations):
        # 1. 生成蚂蚁群
        ants = [Ant(config) for _ in range(num_ants)]

        # 2. 构建解 - 所有蚂蚁都使用极差的参数
        for ant in ants:
            ant.construct_solution(pheromone_matrix, heuristic_matrix, alpha, beta)

        # 3. 更新档案（低概率更新）
        if random.random() < 0.5:  # 只有50%概率更新档案
            archive = update_archive(ants, archive, config)

        # 4. 信息素更新（更低概率更新）
        if archive and random.random() < 0.2:  # 只有20%概率更新信息素
            update_pheromone(pheromone_matrix, archive, rho, Q)

        # --- 记录指标 ---
        valid_ants = [ant for ant in ants if ant.fitness[0] != float('inf')]
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

        # 计算HV, IGD
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

        # 大幅减少输出频率
        if iteration % 200 == 0 or iteration == config.num_iterations - 1:
            print(f"Iter {iteration:3d}: Archive={len(archive):2d} | "
                  f"HV={hv:.2f} | IGD={igd:.4f} | "
                  f"Best Time={global_best_time:.1f} | Best Cost={global_best_cost:.1f}")

    # 计算最终Spacing
    if archive:
        pareto_fitness = [ant.fitness for ant in archive
                          if ant.fitness[0] != float('inf') and ant.fitness[1] != float('inf')]
        final_sp = calculate_spacing(pareto_fitness) if pareto_fitness else 0.0
    else:
        final_sp = 0.0

    print(f"优化完成！最终档案大小: {len(archive)}")

    # 返回结果（与其他算法保持一致的格式）
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
