import copy
import random
import numpy as np
from particle import Particle
from decoder import decode
from objective import evaluate
from pso_rl_new import safe_extract_fitness_tuples
from utils import dominates, calculate_spacing, calculate_hypervolume, calculate_igd, calculate_spread
from mowoa import update_archive_simplified  # 沿用档案更新函数


# ----------------------------------------------------------------------
# 注意：update_archive_simplified 沿用 MOPSO/MOWOA 中的定义，这里不再重复粘贴。
# ----------------------------------------------------------------------
def run_moabc(config):
    """
    最基础的 MOABC 算法 v2.1：
    1. 搜索公式简化为随机扰动 (不变)。
    2. 离散编码采用最暴力的直接映射 (进一步简化为纯截断取整)。
    3. 保留雇佣蜂、观察蜂和侦察蜂三个阶段 (结构不变)。
    """
    print("开始MOABC优化 (v2.1 - 最基础版本...")

    # --- 修复方案：定义一个粒子复制辅助函数 (保持不变) ---
    def copy_particle_simple(original_particle):
        new_particle = Particle(config, feasible_init=False)
        new_particle.position = copy.deepcopy(original_particle.position)
        new_particle.velocity = copy.deepcopy(original_particle.velocity)
        new_particle.pbest_position = copy.deepcopy(original_particle.pbest_position)
        new_particle.pbest_fitness = copy.deepcopy(original_particle.pbest_fitness)
        new_particle.schedule = original_particle.schedule
        new_particle.fitness = original_particle.fitness
        new_particle.resource_assignment = original_particle.resource_assignment
        new_particle.priority_values = original_particle.priority_values
        return new_particle

    # -----------------------------------------------

    # --- 初始化部分 (保持不变) ---
    swarm = [Particle(config, feasible_init=True) for _ in range(config.num_particles)]
    archive = []

    limit = getattr(config, 'abc_limit', config.num_particles // 2)
    failure_counts = np.zeros(config.num_particles)

    archive_size_history = []
    all_fitness = []
    global_min_time_list = []
    global_min_cost_list = []
    hv_history = []
    igd_values = []
    spread_values = []

    global_best_time = float('inf')
    global_best_cost = float('inf')

    for idx, particle in enumerate(swarm):
        particle.schedule = decode(particle, config)
        if particle.schedule:
            particle.fitness = evaluate(particle.schedule, config)
            if particle.fitness is not None and not any(np.isnan(particle.fitness)):
                particle.update_pbest()
                if not np.isnan(particle.fitness[0]):
                    global_best_time = min(global_best_time, particle.fitness[0])
                if not np.isnan(particle.fitness[1]):
                    global_best_cost = min(global_best_cost, particle.fitness[1])
            else:
                failure_counts[idx] = limit + 1

    valid_particles = [p for p in swarm if p.schedule and p.fitness is not None and not any(np.isnan(p.fitness))]
    if valid_particles:
        archive = update_archive_simplified(valid_particles, [], config)

    archive_size_history.append(len(archive))

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

    if archive:
        init_pareto = [p.fitness for p in archive if
                       all(val is not None and val != float('inf') for val in p.fitness)]
        current_hv = calculate_hypervolume(init_pareto, ref_point) if init_pareto else 0.0
    else:
        current_hv = 0.0
    hv_history.append(current_hv)

    try:
        reference_set = np.load('reference_set.npy')
    except Exception:
        reference_set = None
        print("警告：无法加载参考集，IGD/Spread 计算将受影响")

    diversity_fitness_init = [p.fitness for p in archive if
                              all(val is not None and val != float('inf') for val in p.fitness)]
    reference_fitness_init = [tuple(r) for r in reference_set] if reference_set is not None else None

    igd_init = float('nan')
    spread_init = float('nan')
    if reference_fitness_init is not None and diversity_fitness_init:
        try:
            igd_init = calculate_igd(diversity_fitness_init, reference_fitness_init)
            spread_init = calculate_spread(diversity_fitness_init, reference_fitness_init)
        except Exception:
            pass

    igd_values.append(igd_init)
    spread_values.append(spread_init)

    # --- 初始化部分结束 ---

    # 主循环
    for t in range(config.num_iterations):
        current_gen_fitness = []

        # 1. 雇佣蜂阶段 (Employed Bee Phase)
        for idx, particle in enumerate(swarm):
            candidate = copy_particle_simple(particle)
            current_position = np.array(particle.position)

            # 随机扰动向量 (保持不变)
            D = len(current_position)
            random_step = np.random.uniform(-0.5, 0.5, D)

            new_position = current_position + random_step

            # --- 核心简化：最暴力的直接映射离散化处理 (简化 np.round -> int()) ---
            for i in range(D):
                new_pos_i = new_position[i]

                if i < config.num_tasks:
                    # 资源分配部分 (离散编码)
                    available = config.num_suppliers[i]
                    if available:
                        num_available = len(available)
                        # 移除 np.round，使用 int() 纯截断取整
                        mapped_index_float = np.fmod(abs(new_pos_i), num_available)
                        final_index = int(mapped_index_float)
                        final_index = min(final_index, num_available - 1)
                        candidate.position[i] = available[final_index]

                else:
                    # 优先级部分 (连续编码): 保持不变
                    candidate.position[i] = np.clip(new_pos_i, 0, 1)

            candidate.resource_assignment = candidate.position[:config.num_tasks]
            candidate.priority_values = candidate.position[config.num_tasks:]

            # 评估新候选解并进行贪婪选择 (保持不变)
            candidate.schedule = decode(candidate, config)
            if candidate.schedule:
                candidate.fitness = evaluate(candidate.schedule, config)

                if candidate.fitness is not None and not any(np.isnan(candidate.fitness)):
                    if not np.isnan(candidate.fitness[0]):
                        global_best_time = min(global_best_time, candidate.fitness[0])
                    if not np.isnan(candidate.fitness[1]):
                        global_best_cost = min(global_best_cost, candidate.fitness[1])

                    should_replace = dominates(candidate.fitness, particle.fitness)

                    if should_replace:
                        swarm[idx] = candidate
                        failure_counts[idx] = 0
                    else:
                        failure_counts[idx] += 1
            else:
                failure_counts[idx] += 1

        # 2. 观察蜂阶段 (Onlooker Bee Phase) (逻辑保持不变)
        if archive:
            onlooker_indices = random.sample(range(config.num_particles), config.num_particles // 2)

            for idx in onlooker_indices:
                particle = swarm[idx]
                candidate = copy_particle_simple(particle)
                current_position = np.array(particle.position)

                # 随机扰动 (与雇佣蜂搜索公式相同)
                D = len(current_position)
                random_step = np.random.uniform(-0.5, 0.5, D)
                new_position = current_position + random_step

                # 混合编码处理 (简化 int() 截断)
                for i in range(D):
                    new_pos_i = new_position[i]
                    if i < config.num_tasks:
                        available = config.num_suppliers[i]
                        if available:
                            num_available = len(available)
                            # 移除 np.round，使用 int() 纯截断取整
                            mapped_index_float = np.fmod(abs(new_pos_i), num_available)
                            final_index = int(mapped_index_float)
                            final_index = min(final_index, num_available - 1)
                            candidate.position[i] = available[final_index]
                    else:
                        candidate.position[i] = np.clip(new_pos_i, 0, 1)

                candidate.resource_assignment = candidate.position[:config.num_tasks]
                candidate.priority_values = candidate.position[config.num_tasks:]

                # 评估并进行贪婪选择 (保持不变)
                candidate.schedule = decode(candidate, config)
                if candidate.schedule:
                    candidate.fitness = evaluate(candidate.schedule, config)
                    if candidate.fitness is not None and not any(np.isnan(candidate.fitness)):
                        if dominates(candidate.fitness, particle.fitness):
                            swarm[idx] = candidate
                            failure_counts[idx] = 0

        # 3. 侦察蜂阶段 (Scout Bee Phase) (逻辑保持不变)
        scout_indices = np.where(failure_counts >= limit)[0]

        for idx in scout_indices:
            # 随机生成新的食物源 (新的粒子)
            new_scout = Particle(config, feasible_init=True)
            new_scout.schedule = decode(new_scout, config)
            if new_scout.schedule:
                new_scout.fitness = evaluate(new_scout.schedule, config)
                if new_scout.fitness is not None and not any(np.isnan(new_scout.fitness)):
                    new_scout.update_pbest()
                    swarm[idx] = new_scout
                    failure_counts[idx] = 0

                    if not np.isnan(new_scout.fitness[0]):
                        global_best_time = min(global_best_time, new_scout.fitness[0])
                    if not np.isnan(new_scout.fitness[1]):
                        global_best_cost = min(global_best_cost, new_scout.fitness[1])

        # 4. 更新档案和指标 (保持不变)
        valid_particles = [p for p in swarm if p.schedule and p.fitness is not None and not any(np.isnan(p.fitness))]
        archive = update_archive_simplified(valid_particles, archive, config)

        current_gen_fitness = [p.fitness for p in valid_particles]
        all_fitness.append(current_gen_fitness)
        global_min_time_list.append(global_best_time)
        global_min_cost_list.append(global_best_cost)
        archive_size_history.append(len(archive))

        current_pareto = [p.fitness for p in archive if
                          all(val is not None and val != float('inf') for val in p.fitness)]
        current_hv = calculate_hypervolume(current_pareto, ref_point) if current_pareto else 0.0
        hv_history.append(current_hv)

        diversity_fitness = [p.fitness for p in archive if
                             all(val is not None and val != float('inf') for val in p.fitness)]
        reference_fitness = [tuple(r) for r in reference_set] if reference_set is not None else None

        igd_now = float('nan')
        spread_now = float('nan')

        if reference_fitness is not None and diversity_fitness:
            igd_now = calculate_igd(diversity_fitness, reference_fitness)
            try:
                spread_now = calculate_spread(diversity_fitness, reference_fitness)
            except Exception:
                pass

        igd_values.append(igd_now)
        spread_values.append(spread_now)

        if t % 50 == 0 or t == config.num_iterations - 1:
            print(f"Iter {t:3d}: Archive Size={len(archive):2d} | "
                  f"HV={current_hv:.2f} | Δ={spread_now:.3f} | Best Time={global_best_time:.1f} | "
                  f"Best Cost={global_best_cost:.1f}")

    # 计算最终Spacing (保持不变)
    final_pareto = [p.fitness for p in archive if
                    all(val is not None and val != float('inf') for val in p.fitness)]
    final_sp = calculate_spacing(final_pareto) if final_pareto else 0.0

    print(f"优化完成！最终档案大小: {len(archive)}")

    # 返回结果 (保持不变)
    return (
        archive,  # convergence_archive (1)
        archive,  # diversity_archive (2)
        all_fitness,  # (3)
        global_min_time_list,  # (4)
        global_min_cost_list,  # (5)
        archive_size_history,  # (6)
        {},  # 空Q表占位 (7)
        hv_history,  # (8)
        final_sp,  # (9)
        igd_values,  # (10)
        spread_values  # <-- Spread 历史记录 (11)
    )