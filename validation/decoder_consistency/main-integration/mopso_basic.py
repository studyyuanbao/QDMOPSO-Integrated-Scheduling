import random

import numpy as np
from particle import Particle
from decoder import decode
from objective import evaluate
from pso_rl_new import safe_extract_fitness_tuples
from utils import dominates, calculate_spacing, calculate_hypervolume, calculate_igd


def update_archive_simplified(particles, archive, config):
    """
    最简化档案更新：
    1. 合并当前粒子群和现有档案中的所有解。
    2. 找出所有非支配解。
    3. 如果档案超出最大容量，则随机移除超出的部分。
    """
    # 过滤掉解码或评估失败的粒子
    # 确保粒子有 schedule, fitness 且 fitness 中的值不是 None 也不是 NaN
    valid_particles = [
        p for p in particles
        if p.schedule and p.fitness is not None and not any(np.isnan(p.fitness)) and all(
            val is not None and val != float('inf') for val in p.fitness)
    ]

    # 合并所有候选解
    all_candidates = valid_particles + archive

    # 找出非支配解
    non_dominated = []
    for cand in all_candidates:
        # 检查cand是否被all_candidates中的任一其他解支配
        is_dominated = False
        for other in all_candidates:
            if other is not cand and other.fitness is not None and dominates(other.fitness, cand.fitness):
                is_dominated = True
                break
        if not is_dominated:
            non_dominated.append(cand)

    # 多样性维护：如果档案太大，简单随机选择
    if len(non_dominated) > config.archive_size:
        non_dominated = random.sample(non_dominated, config.archive_size)

    return non_dominated


def run_mopso(config):
    """
    最基本的标准MOPSO算法（已对齐 HV 参考点初始化）
    - 使用标准PSO更新公式
    - 随机选择档案中的leader
    - 最简化的非支配解档案管理 (统一一个档案)
    - 保留所有历史记录和指标计算以兼容返回格式
    """
    print("开始MOPSO优化...")

    # 初始化
    swarm = [Particle(config, feasible_init=True) for _ in range(config.num_particles)]

    # 统一的非支配解档案
    archive = []

    # 记录变量（为了兼容返回格式，必须保留）
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
            # 只有当fitness有效时才更新pbest
            if particle.fitness is not None and not any(np.isnan(particle.fitness)):
                particle.update_pbest()
                if not np.isnan(particle.fitness[0]):
                    global_best_time = min(global_best_time, particle.fitness[0])
                if not np.isnan(particle.fitness[1]):
                    global_best_cost = min(global_best_cost, particle.fitness[1])

    # 初始档案更新
    valid_particles = [p for p in swarm if p.schedule and p.fitness is not None and not any(np.isnan(p.fitness))]
    if valid_particles:
        archive = update_archive_simplified(valid_particles, [], config)

    archive_size_history.append(len(archive))

    # ===== 与 pso_rl_new 对齐：基于初始有效粒子估计 HV 参考点（若 config 中未提供） =====
    initial_ref = getattr(config, 'hv_ref_point', None)
    if initial_ref is not None:
        ref_point = tuple(initial_ref)
    else:
        # 使用初始有效粒子的 fitness 来估计参考点并加 margin
        valid_init = [p for p in swarm if hasattr(p, 'fitness') and p.fitness is not None and not any(np.isnan(p.fitness))]
        fitness_init = safe_extract_fitness_tuples(valid_init)
        if fitness_init:
            max_time = max(t for t, c in fitness_init)
            max_cost = max(c for t, c in fitness_init)
            # margin 可调：沿用 pso_rl_new 的 1.5 倍 + 1.0
            ref_point = (max_time * 1.5 + 1.0, max_cost * 1.5 + 1.0)
        else:
            # 最后兜底（确保比任何合理值都大）
            ref_point = (20000.0, 20000.0)
        # 保存回 config 便于诊断/一致性
        config.hv_ref_point = ref_point
    # ===== 结束 ref_point 对齐部分 =====

    print(f"使用的 HV 参考点 (ref_point) = {ref_point}")

    # 初始HV（必须保留）
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

    # 加载参考集（必须保留）
    try:
        reference_set = np.load('reference_set.npy')
    except Exception:
        reference_set = None
        print("警告：无法加载参考集，IGD计算将受影响")

    # 主循环
    for t in range(config.num_iterations):
        current_gen_fitness = []

        # 标准PSO参数
        w = 1.0  # 惯性权重 (可保持为常数)
        c1 = 2.0  # 认知参数
        c2 = 2.0  # 社会参数

        # 简单的leader选择：从档案中随机选择
        candidates = archive if archive else [p for p in swarm if p.schedule]

        if candidates:
            leader = random.choice(candidates)
        else:
            # 如果没有候选leader，跳过这次迭代
            continue

        # 更新粒子群
        for particle in swarm:
            position = particle.position
            velocity = particle.velocity
            pbest_position = particle.pbest_position
            leader_position = leader.position

            # 标准PSO速度更新
            for i in range(len(position)):
                r1, r2 = random.random(), random.random()
                cognitive = c1 * r1 * (pbest_position[i] - position[i])
                social = c2 * r2 * (leader_position[i] - position[i])
                velocity[i] = w * velocity[i] + cognitive + social

                # 速度限制
                V_MAX = 3.0
                if abs(velocity[i]) > V_MAX:
                    velocity[i] = np.sign(velocity[i]) * V_MAX

                # 位置更新
                if i < config.num_tasks:
                    # 资源分配部分：基于速度概率选择资源
                    available = config.num_suppliers[i]
                    if available:
                        # 简单的概率转换
                        change_prob = 1 / (1 + np.exp(-velocity[i]))
                        if random.random() < change_prob:
                            position[i] = random.choice(available)
                else:
                    # 优先级部分：标准连续更新
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

                    # 更新个体最优
                    if dominates(particle.fitness, particle.pbest_fitness):
                        particle.update_pbest()

                    # 更新全局最优
                    if not np.isnan(particle.fitness[0]):
                        global_best_time = min(global_best_time, particle.fitness[0])
                    if not np.isnan(particle.fitness[1]):
                        global_best_cost = min(global_best_cost, particle.fitness[1])

        # 更新档案
        valid_particles = [p for p in swarm if p.schedule and p.fitness is not None and not any(np.isnan(p.fitness))]
        archive = update_archive_simplified(valid_particles, archive, config)

        # 记录指标（必须保留）
        all_fitness.append(current_gen_fitness)
        global_min_time_list.append(global_best_time)
        global_min_cost_list.append(global_best_cost)
        archive_size_history.append(len(archive))

        # 计算HV（使用已估算的 ref_point）
        current_pareto = [p.fitness for p in archive if
                          all(val is not None and val != float('inf') for val in p.fitness)]
        if current_pareto:
            current_hv = calculate_hypervolume(current_pareto, ref_point)
        else:
            current_hv = 0.0
        hv_history.append(current_hv)

        # 计算IGD（必须保留）
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

    # 计算最终Spacing（必须保留）
    final_pareto = [p.fitness for p in archive if
                    all(val is not None and val != float('inf') for val in p.fitness)]
    final_sp = calculate_spacing(final_pareto) if final_pareto else 0.0

    print(f"优化完成！最终档案大小: {len(archive)}")

    # 返回结果（兼容原有的函数接口，将统一的 archive 同时作为 convergence_archive 和 diversity_archive 返回）
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