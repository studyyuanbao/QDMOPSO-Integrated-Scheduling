import random
import numpy as np
from typing import List, Tuple
from particle import Particle
from decoder import decode
from objective import evaluate
from utils import dominates, calculate_hypervolume, calculate_spacing, calculate_igd


class Wolf:
    """灰狼个体类"""

    def __init__(self, config):
        self.config = config
        self.position = [0.0] * (2 * config.num_tasks)
        self.resource_assignment = [0] * config.num_tasks
        self.priority_values = [0.0] * config.num_tasks
        self.schedule = []
        self.fitness = (float('inf'), float('inf'))  # 使用元组而不是列表

    def initialize(self):
        """初始化灰狼位置"""
        # 资源分配部分
        for i in range(self.config.num_tasks):
            available = self.config.num_suppliers[i]
            if available:
                self.resource_assignment[i] = random.choice(available)
                self.position[i] = self.resource_assignment[i]

        # 优先级部分
        for i in range(self.config.num_tasks):
            self.priority_values[i] = random.random()
            self.position[self.config.num_tasks + i] = self.priority_values[i]

        # 评估
        self.evaluate()

    def evaluate(self):
        """评估灰狼的适应度"""
        self.schedule = decode(self, self.config)
        if self.schedule:
            fitness_result = evaluate(self.schedule, self.config)
            if fitness_result is not None and len(fitness_result) >= 2:
                self.fitness = (float(fitness_result[0]), float(fitness_result[1]))
            else:
                self.fitness = (float('inf'), float('inf'))
        else:
            self.fitness = (float('inf'), float('inf'))
        return self.fitness


def safe_extract_fitness_tuples(wolves):
    """从灰狼列表中安全地提取适应度元组"""
    fitness_tuples = []
    for w in wolves:
        if (hasattr(w, 'fitness') and
                isinstance(w.fitness, (list, tuple)) and
                len(w.fitness) >= 2):
            try:
                time_val = float(w.fitness[0])
                cost_val = float(w.fitness[1])
                if not (np.isnan(time_val) or np.isnan(cost_val) or
                        np.isinf(time_val) or np.isinf(cost_val)):
                    fitness_tuples.append((time_val, cost_val))
            except (ValueError, TypeError):
                continue
    return fitness_tuples


def fast_non_dominated_sort(solutions):
    """快速非支配排序"""
    fronts = [[]]
    for p in solutions:
        p.domination_count = 0
        p.dominated_solutions = []
        for q in solutions:
            if p is q:
                continue
            if dominates(p.fitness, q.fitness):
                p.dominated_solutions.append(q)
            elif dominates(q.fitness, p.fitness):
                p.domination_count += 1
        if p.domination_count == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in p.dominated_solutions:
                q.domination_count -= 1
                if q.domination_count == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]


def calculate_crowding_distance(front):
    """计算拥挤距离"""
    if not front:
        return []

    num_objectives = len(front[0].fitness)
    crowding_distances = [0.0] * len(front)

    for obj_idx in range(num_objectives):
        # 按当前目标函数排序
        sorted_front = sorted(front, key=lambda x: x.fitness[obj_idx])

        # 边界解的拥挤距离设为无穷大
        if len(sorted_front) > 0:
            crowding_distances[0] = float('inf')
            crowding_distances[-1] = float('inf')

        # 计算目标函数范围
        if len(sorted_front) >= 2:
            min_obj = sorted_front[0].fitness[obj_idx]
            max_obj = sorted_front[-1].fitness[obj_idx]
            obj_range = max_obj - min_obj

            if obj_range > 0:
                for i in range(1, len(front) - 1):
                    prev_obj = sorted_front[i - 1].fitness[obj_idx]
                    next_obj = sorted_front[i + 1].fitness[obj_idx]
                    crowding_distances[i] += (next_obj - prev_obj) / obj_range

    return crowding_distances


def update_archive(wolves, archive, config):
    """更新档案 - 使用非支配排序和拥挤距离"""
    # 合并候选解
    all_solutions = wolves + archive

    # 过滤有效解
    valid_solutions = [w for w in all_solutions
                       if w.fitness[0] != float('inf') and w.fitness[1] != float('inf')]

    if not valid_solutions:
        return archive

    # 非支配排序
    fronts = fast_non_dominated_sort(valid_solutions)

    # 构建新档案
    new_archive = []
    archive_size = getattr(config, 'archive_size', 50)

    for front in fronts:
        if len(new_archive) + len(front) <= archive_size:
            new_archive.extend(front)
        else:
            # 对最后一层前端按拥挤距离排序
            crowding_distances = calculate_crowding_distance(front)
            sorted_front = sorted(zip(front, crowding_distances), key=lambda x: x[1], reverse=True)
            remaining_slots = archive_size - len(new_archive)
            new_archive.extend([item[0] for item in sorted_front[:remaining_slots]])
            break

    return new_archive


def select_alpha_beta_delta(archive):
    """从档案中选择alpha, beta, delta狼"""
    if not archive:
        return None, None, None

    # 计算拥挤距离
    crowding_distances = calculate_crowding_distance(archive)

    # 选择拥挤距离最大的三个解作为alpha, beta, delta
    if len(archive) >= 3:
        sorted_indices = np.argsort(crowding_distances)[::-1]  # 降序排列
        alpha = archive[sorted_indices[0]]
        beta = archive[sorted_indices[1]]
        delta = archive[sorted_indices[2]]
    elif len(archive) == 2:
        alpha = archive[0]
        beta = archive[1]
        delta = archive[0]  # 重复alpha
    elif len(archive) == 1:
        alpha = archive[0]
        beta = archive[0]  # 重复alpha
        delta = archive[0]  # 重复alpha
    else:
        return None, None, None

    return alpha, beta, delta


def update_position(wolf, alpha, beta, delta, a, config):
    """更新灰狼位置"""
    new_position = wolf.position.copy()

    for i in range(len(wolf.position)):
        if alpha and beta and delta:
            # 计算三个领导狼的影响
            r1, r2 = random.random(), random.random()
            A1 = 2 * a * r1 - a
            C1 = 2 * r2

            D_alpha = abs(C1 * alpha.position[i] - wolf.position[i])
            X1 = alpha.position[i] - A1 * D_alpha

            r1, r2 = random.random(), random.random()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2

            D_beta = abs(C2 * beta.position[i] - wolf.position[i])
            X2 = beta.position[i] - A2 * D_beta

            r1, r2 = random.random(), random.random()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2

            D_delta = abs(C3 * delta.position[i] - wolf.position[i])
            X3 = delta.position[i] - A3 * D_delta

            # 新位置是三个领导狼位置的平均
            new_position[i] = (X1 + X2 + X3) / 3.0
        else:
            # 如果没有领导狼，进行随机搜索
            if i < config.num_tasks:
                available = config.num_suppliers[i]
                if available:
                    new_position[i] = random.choice(available)
            else:
                new_position[i] = random.random()

    # 约束处理
    for i in range(len(new_position)):
        if i < config.num_tasks:
            # 资源分配部分：离散值
            available = config.num_suppliers[i]
            if available:
                # 确保选择有效的资源
                if new_position[i] not in available:
                    # 找到最接近的有效资源
                    closest_resource = min(available, key=lambda x: abs(x - new_position[i]))
                    new_position[i] = closest_resource
        else:
            # 优先级部分：连续值在[0,1]范围内
            new_position[i] = np.clip(new_position[i], 0, 1)

    return new_position


def run_mogwo(config):
    """运行多目标灰狼优化算法"""
    print("开始MOGWO优化...")

    # 参数设置
    num_wolves = config.num_particles
    max_iterations = config.num_iterations

    # 记录变量
    archive_size_history = []
    all_fitness = []
    global_min_time_list = []
    global_min_cost_list = []
    hv_history = []
    igd_values = []

    global_best_time = float('inf')
    global_best_cost = float('inf')

    # 初始化灰狼群
    wolves = [Wolf(config) for _ in range(num_wolves)]
    for wolf in wolves:
        wolf.initialize()
        if wolf.fitness[0] != float('inf'):
            global_best_time = min(global_best_time, wolf.fitness[0])
            global_best_cost = min(global_best_cost, wolf.fitness[1])

    # 初始化档案
    archive = update_archive(wolves, [], config)
    archive_size_history.append(len(archive))

    # ===== 参考点设置 =====
    initial_ref = getattr(config, 'hv_ref_point', None)
    if initial_ref is not None:
        ref_point = tuple(initial_ref)
    else:
        valid_init = [w for w in wolves if w.fitness[0] != float('inf') and w.fitness[1] != float('inf')]
        fitness_init = safe_extract_fitness_tuples(valid_init)
        if fitness_init:
            max_time = max(t for t, c in fitness_init)
            max_cost = max(c for t, c in fitness_init)
            ref_point = (max_time * 1.5 + 1.0, max_cost * 1.5 + 1.0)
        else:
            ref_point = (20000.0, 20000.0)
        config.hv_ref_point = ref_point

    print(f"使用的 HV 参考点 (ref_point) = {ref_point}")

    # 加载参考集
    try:
        reference_set = np.load('reference_set.npy')
    except Exception:
        reference_set = None
        print("警告：无法加载参考集，IGD计算将受影响")

    # 初始指标计算
    if archive:
        init_pareto = [w.fitness for w in archive if w.fitness[0] != float('inf')]
        if init_pareto:
            current_hv = calculate_hypervolume(init_pareto, ref_point)
        else:
            current_hv = 0.0
    else:
        current_hv = 0.0
    hv_history.append(current_hv)

    # 初始IGD计算
    diversity_fitness = [w.fitness for w in archive if w.fitness[0] != float('inf')]
    reference_fitness = [tuple(r) for r in reference_set] if reference_set is not None else None
    if reference_fitness is not None and diversity_fitness:
        igd_values.append(calculate_igd(diversity_fitness, reference_fitness))
    else:
        igd_values.append(float('nan'))

    # 主循环
    for iteration in range(max_iterations):
        current_gen_fitness = []

        # 参数a线性递减
        a = 2 - 2 * (iteration / max_iterations)

        # 选择领导狼
        alpha, beta, delta = select_alpha_beta_delta(archive)

        # 更新每只灰狼的位置
        for wolf in wolves:
            # 保存旧位置和适应度（使用list()而不是copy()）
            old_position = list(wolf.position)  # 转换为列表以便修改
            old_fitness = wolf.fitness  # 元组不需要copy

            # 更新位置
            new_position = update_position(wolf, alpha, beta, delta, a, config)
            wolf.position = new_position

            # 更新资源分配和优先级
            wolf.resource_assignment = wolf.position[:config.num_tasks]
            wolf.priority_values = wolf.position[config.num_tasks:]

            # 评估新位置
            wolf.evaluate()

            # 记录当前代的适应度
            if wolf.fitness[0] != float('inf'):
                current_gen_fitness.append(wolf.fitness)

                # 更新全局最优
                global_best_time = min(global_best_time, wolf.fitness[0])
                global_best_cost = min(global_best_cost, wolf.fitness[1])

        # 更新档案
        archive = update_archive(wolves, archive, config)

        # 记录指标
        all_fitness.append(current_gen_fitness)
        global_min_time_list.append(global_best_time)
        global_min_cost_list.append(global_best_cost)
        archive_size_history.append(len(archive))

        # 计算HV
        current_pareto = [w.fitness for w in archive if w.fitness[0] != float('inf')]
        if current_pareto:
            current_hv = calculate_hypervolume(current_pareto, ref_point)
        else:
            current_hv = 0.0
        hv_history.append(current_hv)

        # 计算IGD
        diversity_fitness = [w.fitness for w in archive if w.fitness[0] != float('inf')]
        reference_fitness = [tuple(r) for r in reference_set] if reference_set is not None else None

        if reference_fitness is not None and diversity_fitness:
            igd_values.append(calculate_igd(diversity_fitness, reference_fitness))
        else:
            igd_values.append(float('nan'))

        # 输出进度
        if iteration % 50 == 0 or iteration == max_iterations - 1:
            print(f"Iter {iteration:3d}: Archive Size={len(archive):2d} | "
                  f"HV={current_hv:.2f} | Best Time={global_best_time:.1f} | "
                  f"Best Cost={global_best_cost:.1f}")

    # 计算最终Spacing
    final_pareto = [w.fitness for w in archive if w.fitness[0] != float('inf')]
    final_sp = calculate_spacing(final_pareto) if final_pareto else 0.0

    print(f"优化完成！最终档案大小: {len(archive)}")

    # 返回结果（与其他算法保持一致的格式）
    return (
        archive,  # convergence_archive
        archive,  # diversity_archive
        all_fitness,
        global_min_time_list,
        global_min_cost_list,
        archive_size_history,
        {},  # 空Q表占位
        hv_history,
        final_sp,
        igd_values
    )