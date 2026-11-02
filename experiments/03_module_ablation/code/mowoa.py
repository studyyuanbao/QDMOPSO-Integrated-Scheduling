import random
import numpy as np
from particle import Particle
from decoder import decode
from objective import evaluate
from pso_rl_new import safe_extract_fitness_tuples
# 【修改点 1：导入 calculate_spread】
from utils import dominates, calculate_spacing, calculate_hypervolume, calculate_igd, calculate_spread


# 沿用MOPSO的档案更新函数，确保一致性
def update_archive_simplified(particles, archive, config):
    """
    最简化档案更新：
    1. 合并当前粒子群和现有档案中的所有解。
    2. 找出所有非支配解。
    3. 如果档案超出最大容量，则随机移除超出的部分。
    （此函数与您的MOPSO版本完全相同）
    """
    # 过滤掉解码或评估失败的粒子
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


def run_mowoa(config):
    """
    最基础的 MOWOA 算法 v3.0：采用最暴力的直接映射进行离散化，
    去除所有概率、阈值和差值计算，使算法逻辑达到最简化。
    """
    print("开始MOWOA优化 (v3.0 - 直接映射离散化)...")

    # --- 初始化部分 (保持不变) ---
    swarm = [Particle(config, feasible_init=True) for _ in range(config.num_particles)]
    archive = []

    archive_size_history = []
    all_fitness = []
    global_min_time_list = []
    global_min_cost_list = []
    hv_history = []
    igd_values = []
    # 【修改点 2：初始化 spread_values 列表】
    spread_values = []

    global_best_time = float('inf')
    global_best_cost = float('inf')

    # ... (初始化粒子、初始档案更新、HV参考点对齐、初始HV、加载参考集 逻辑完全不变)

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

    # **【修改点 3：计算并记录初始 IGD 和 Spread 值】**
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
    spread_values.append(spread_init)  # <-- 记录初始 Spread 值

    # --- 初始化部分结束 ---

    # 主循环
    for t in range(config.num_iterations):
        current_gen_fitness = []

        # WOA 参数更新 (保持不变)
        a = 2 - t * (2 / config.num_iterations)
        b = 1.0

        # Leader 选择 (保持不变)
        candidates = archive if archive else [p for p in swarm if p.schedule]
        if not candidates:
            archive_size_history.append(len(archive))
            hv_history.append(current_hv)
            igd_values.append(float('nan'))
            spread_values.append(float('nan'))  # <-- 确保 spread 列表长度一致
            continue

        leader = random.choice(candidates)
        leader_position = np.array(leader.position)

        # 更新粒子群 (鲸鱼群)
        for particle in swarm:
            position = np.array(particle.position)

            # WOA 核心随机参数 (保持不变)
            r1, r2 = random.random(), random.random()
            A = 2 * a * r1 - a
            C = 2 * r2
            l = random.uniform(-1, 1)
            p = random.random()

            new_position = np.zeros_like(position)

            # --- MOWOA 位置更新 (核心逻辑保持不变) ---
            if p < 0.5:
                if abs(A) < 1:
                    D = np.abs(C * leader_position - position)
                    new_position = leader_position - A * D
                else:
                    rand_whale = random.choice(swarm)
                    rand_position = np.array(rand_whale.position)
                    D = np.abs(C * rand_position - position)
                    new_position = rand_position - A * D
            else:
                D_prime = np.abs(leader_position - position)
                spiral_factor = D_prime * np.exp(b * l) * np.cos(2 * np.pi * l)
                new_position = leader_position + spiral_factor

            # --- 核心修改：最暴力的直接映射离散化处理 (保持不变) ---
            for i in range(len(position)):
                old_pos_i = position[i]
                new_pos_i = new_position[i]

                if i < config.num_tasks:
                    # 资源分配部分 (离散编码):
                    available = config.num_suppliers[i]
                    if available:
                        # **最暴力/最基础的映射：将连续值四舍五入并取模，映射到可用资源列表的索引上**
                        num_available = len(available)

                        # 1. 将新位置值 new_pos_i 映射到 [0, num_available) 范围内
                        mapped_index_float = np.fmod(abs(new_pos_i), num_available)

                        # 2. 四舍五入取整，得到最终的索引
                        final_index = int(np.round(mapped_index_float))

                        # 3. 赋值
                        # 确保索引在有效范围内 [0, num_available-1]
                        final_index = min(final_index, num_available - 1)
                        position[i] = available[final_index]

                else:
                    # 优先级部分 (连续编码): 保持不变
                    position[i] = np.clip(new_pos_i, 0, 1)

                # 计算等效速度 (保持不变)
                particle.velocity[i] = position[i] - old_pos_i

            # ... (粒子属性更新、解码、评估、pbest 更新、全局最优更新 逻辑完全不变)
            particle.position = list(position)
            particle.resource_assignment = particle.position[:config.num_tasks]
            particle.priority_values = particle.position[config.num_tasks:]

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

        # --- 档案更新、指标记录、输出 ---
        valid_particles = [p for p in swarm if p.schedule and p.fitness is not None and not any(np.isnan(p.fitness))]
        archive = update_archive_simplified(valid_particles, archive, config)

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
        spread_now = float('nan')  # <-- 初始化 Spread 变量

        if reference_fitness is not None and diversity_fitness:
            igd_now = calculate_igd(diversity_fitness, reference_fitness)
            try:
                # **新增 Spread 计算**
                spread_now = calculate_spread(diversity_fitness, reference_fitness)
            except Exception:
                pass

        igd_values.append(igd_now)
        spread_values.append(spread_now)  # 【修改点 4：记录迭代 Spread 值】

        if t % 50 == 0 or t == config.num_iterations - 1:
            # **【修改点 5：更新输出，加入 Spread (Δ)】**
            print(f"Iter {t:3d}: Archive Size={len(archive):2d} | "
                  f"HV={current_hv:.2f} | Δ={spread_now:.3f} | Best Time={global_best_time:.1f} | "
                  f"Best Cost={global_best_cost:.1f}")

    # 计算最终Spacing (保持不变)
    final_pareto = [p.fitness for p in archive if
                    all(val is not None and val != float('inf') for val in p.fitness)]
    final_sp = calculate_spacing(final_pareto) if final_pareto else 0.0

    print(f"优化完成！最终档案大小: {len(archive)}")

    # 【修改点 6：返回结果，新增 spread_values 作为第11个元素】
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
        spread_values  # <-- 新增 Spread 历史记录 (11)
    )