import random
import numpy as np
from particle import Particle
from decoder import decode
from objective import evaluate
from pso_rl_new import safe_extract_fitness_tuples
from utils import dominates, calculate_spacing, calculate_hypervolume, calculate_igd


def update_archive_standard(particles, archive, config):
    """
    标准MOPSO档案更新：
    1. 合并当前粒子群和档案
    2. 快速非支配排序
    3. 使用拥挤距离维护多样性
    """
    # 过滤有效粒子
    valid_particles = [
        p for p in particles
        if p.schedule and p.fitness is not None and not any(np.isnan(p.fitness)) and all(
            val is not None and val != float('inf') for val in p.fitness)
    ]

    # 合并候选解
    all_candidates = valid_particles + archive

    if not all_candidates:
        return []

    # 快速非支配排序
    fronts = fast_non_dominated_sort(all_candidates)

    # 构建新档案
    new_archive = []
    for front in fronts:
        if len(new_archive) + len(front) <= config.archive_size:
            new_archive.extend(front)
        else:
            # 对最后一层前端按拥挤距离排序
            crowding_distances = calculate_crowding_distance(front)
            sorted_front = sorted(zip(front, crowding_distances), key=lambda x: x[1], reverse=True)
            remaining_slots = config.archive_size - len(new_archive)
            new_archive.extend([item[0] for item in sorted_front[:remaining_slots]])
            break

    return new_archive


def fast_non_dominated_sort(candidates):
    """快速非支配排序"""
    fronts = [[]]
    for p in candidates:
        p.domination_count = 0
        p.dominated_solutions = []
        for q in candidates:
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

    return fronts[:-1]  # 去掉最后一个空前端


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
        crowding_distances[0] = float('inf')
        crowding_distances[-1] = float('inf')

        # 计算目标函数范围
        min_obj = sorted_front[0].fitness[obj_idx]
        max_obj = sorted_front[-1].fitness[obj_idx]
        obj_range = max_obj - min_obj

        if obj_range > 0:
            for i in range(1, len(front) - 1):
                prev_obj = sorted_front[i - 1].fitness[obj_idx]
                next_obj = sorted_front[i + 1].fitness[obj_idx]
                crowding_distances[i] += (next_obj - prev_obj) / obj_range

    return crowding_distances


def select_leader(archive):
    """基于拥挤距离选择leader"""
    if not archive:
        return None

    crowding_distances = calculate_crowding_distance(archive)

    # 优先选择拥挤距离大的解（稀疏区域）
    max_crowding_idx = np.argmax(crowding_distances)
    return archive[max_crowding_idx]


def run_mopso(config):
    """
    标准MOPSO算法 - 作为对比实验的基线
    """
    print("开始标准MOPSO优化...")

    # 初始化
    swarm = [Particle(config, feasible_init=True) for _ in range(config.num_particles)]
    archive = []

    # 记录变量
    archive_size_history = []
    all_fitness = []
    global_min_time_list = []
    global_min_cost_list = []
    hv_history = []
    igd_values = []

    global_best_time = float('inf')
    global_best_cost = float('inf')

    # 初始化粒子
    for particle in swarm:
        particle.schedule = decode(particle, config)
        if particle.schedule:
            particle.fitness = evaluate(particle.schedule, config)
            if particle.fitness is not None and not any(np.isnan(particle.fitness)):
                particle.update_pbest()
                if not np.isnan(particle.fitness[0]):
                    global_best_time = min(global_best_time, particle.fitness[0])
                if not np.isnan(particle.fitness[1]):
                    global_best_cost = min(global_best_cost, particle.fitness[1])

    # 初始档案更新
    valid_particles = [p for p in swarm if p.schedule and p.fitness is not None and not any(np.isnan(p.fitness))]
    if valid_particles:
        archive = update_archive_standard(valid_particles, [], config)

    archive_size_history.append(len(archive))

    # 参考点设置
    initial_ref = getattr(config, 'hv_ref_point', None)
    if initial_ref is not None:
        ref_point = tuple(initial_ref)
    else:
        valid_init = [p for p in swarm if
                      hasattr(p, 'fitness') and p.fitness is not None and not any(np.isnan(p.fitness))]
        fitness_init = safe_extract_fitness_tuples(valid_init)
        if fitness_init:
            max_time = max(t for t, c in fitness_init)
            max_cost = max(c for t, c in fitness_init)
            ref_point = (max_time * 1.5 + 1.0, max_cost * 1.5 + 1.0)
        else:
            ref_point = (20000.0, 20000.0)
        config.hv_ref_point = ref_point

    print(f"使用的 HV 参考点 (ref_point) = {ref_point}")

    # 初始HV
    if archive:
        init_pareto = [p.fitness for p in archive if
                       all(val is not None and val != float('inf') for val in p.fitness)]
        if init_pareto:
            current_hv = calculate_hypervolume(init_pareto, ref_point)
        else:
            current_hv = 0.0
    else:
        current_hv = 0.0
    hv_history.append(current_hv)

    # 加载参考集
    try:
        reference_set = np.load('reference_set.npy')
    except Exception:
        reference_set = None
        print("警告：无法加载参考集，IGD计算将受影响")

    # 主循环
    for t in range(config.num_iterations):
        current_gen_fitness = []

        # 标准MOPSO参数设置
        w_max, w_min = 0.9, 0.4
        w = w_max - (w_max - w_min) * t / config.num_iterations  # 线性递减
        c1, c2 = 2.0, 2.0

        # Leader选择
        leader = select_leader(archive) if archive else None
        if leader is None:
            # 如果没有档案，从当前群中选择pbest最好的
            valid_swarm = [p for p in swarm if hasattr(p, 'pbest_fitness') and p.pbest_fitness is not None]
            if valid_swarm:
                leader = min(valid_swarm, key=lambda x: x.pbest_fitness[0] + x.pbest_fitness[1])
            else:
                continue

        # 更新粒子
        for particle in swarm:
            # PSO速度位置更新（保持您原有的逻辑）
            position = particle.position
            velocity = particle.velocity
            pbest_position = particle.pbest_position
            leader_position = leader.position

            for i in range(len(position)):
                r1, r2 = random.random(), random.random()
                cognitive = c1 * r1 * (pbest_position[i] - position[i])
                social = c2 * r2 * (leader_position[i] - position[i])
                velocity[i] = w * velocity[i] + cognitive + social

                V_MAX = 3.0
                if abs(velocity[i]) > V_MAX:
                    velocity[i] = np.sign(velocity[i]) * V_MAX

                if i < config.num_tasks:
                    available = config.num_suppliers[i]
                    if available:
                        change_prob = 1 / (1 + np.exp(-velocity[i]))
                        if random.random() < change_prob:
                            position[i] = random.choice(available)
                else:
                    position[i] = np.clip(position[i] + velocity[i], 0, 1)

            particle.position = position
            particle.velocity = velocity
            particle.resource_assignment = position[:config.num_tasks]
            particle.priority_values = position[config.num_tasks:]

            # 解码和评估
            particle.schedule = decode(particle, config)
            if particle.schedule:
                particle.fitness = evaluate(particle.schedule, config)

                if particle.fitness is not None and not any(np.isnan(particle.fitness)):
                    current_gen_fitness.append(particle.fitness)
                    if dominates(particle.fitness, particle.pbest_fitness):
                        particle.update_pbest()
                    if not np.isnan(particle.fitness[0]):
                        global_best_time = min(global_best_time, particle.fitness[0])
                    if not np.isnan(particle.fitness[1]):
                        global_best_cost = min(global_best_cost, particle.fitness[1])

        # 更新档案
        valid_particles = [p for p in swarm if p.schedule and p.fitness is not None and not any(np.isnan(p.fitness))]
        archive = update_archive_standard(valid_particles, archive, config)

        # 记录指标
        all_fitness.append(current_gen_fitness)
        global_min_time_list.append(global_best_time)
        global_min_cost_list.append(global_best_cost)
        archive_size_history.append(len(archive))

        # 计算指标
        current_pareto = [p.fitness for p in archive if
                          all(val is not None and val != float('inf') for val in p.fitness)]
        if current_pareto:
            current_hv = calculate_hypervolume(current_pareto, ref_point)
        else:
            current_hv = 0.0
        hv_history.append(current_hv)

        diversity_fitness = [p.fitness for p in archive if
                             all(val is not None and val != float('inf') for val in p.fitness)]
        reference_fitness = [tuple(r) for r in reference_set] if reference_set is not None else None
        if reference_fitness is not None and diversity_fitness:
            igd_values.append(calculate_igd(diversity_fitness, reference_fitness))
        else:
            igd_values.append(float('nan'))

        # 输出进度
        if t % 50 == 0 or t == config.num_iterations - 1:
            print(f"Iter {t:3d}: Archive Size={len(archive):2d} | "
                  f"HV={current_hv:.2f} | Best Time={global_best_time:.1f} | "
                  f"Best Cost={global_best_cost:.1f}")

    # 最终计算
    final_pareto = [p.fitness for p in archive if
                    all(val is not None and val != float('inf') for val in p.fitness)]
    final_sp = calculate_spacing(final_pareto) if final_pareto else 0.0

    print(f"优化完成！最终档案大小: {len(archive)}")

    return (
        archive, archive,
        all_fitness,
        global_min_time_list,
        global_min_cost_list,
        archive_size_history,
        {},
        hv_history,
        final_sp,
        igd_values
    )