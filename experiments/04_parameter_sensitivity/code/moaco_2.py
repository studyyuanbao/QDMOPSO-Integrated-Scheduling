import random
import numpy as np

from decoder import decode
from mopso_basic import update_archive_simplified
from objective import evaluate
from particle import Particle
from pso_rl_new import safe_extract_fitness_tuples
# 【修改点 1：导入 calculate_spread】
from utils import calculate_hypervolume, dominates, calculate_igd, calculate_spacing, calculate_spread


# 假设 Particle, decode, evaluate, dominates, calculate_hypervolume, etc. 仍然可用
# 假设 config 中有 num_ants, num_iterations, rho (蒸发率), alpha (信息素重要性), beta (启发式重要性)

# ----------------------------------------------------------------------
# 沿用 update_archive_simplified 函数（为简洁不再重复粘贴）
# ----------------------------------------------------------------------

def run_moaco(config):
    """
    最基本的标准 MOACO 算法 (已修复 Config 属性缺失问题)
    - 蚁群基于信息素 (Tau) 和启发式信息 (Eta) 进行路径选择。
    - 信息素仅由档案中的非支配解更新。
    - 使用统一的非支配解档案管理。
    """
    print("开始MOACO优化...")

    # --- 初始化参数和结构 ---

    # 修复：为缺失的 config 属性提供健壮的默认值
    try:
        # 优先使用配置中的 num_ants，否则使用 num_particles
        num_ants = getattr(config, 'num_ants', config.num_particles)
        rho = getattr(config, 'rho', 0.5)  # 信息素蒸发率
        alpha = getattr(config, 'alpha', 1.0)  # 信息素重要性
        beta = getattr(config, 'beta', 2.0)  # 启发式信息重要性
    except AttributeError as e:
        # 如果 config.num_particles 也没有，则使用一个硬编码的默认值
        num_ants = 20
        rho, alpha, beta = 0.5, 1.0, 2.0
        print(f"警告: MOACO 缺乏关键配置参数。使用默认值。错误: {e}")

    # 修复：确定决策变量总维度 D
    try:
        # 资源分配维度
        num_tasks = config.num_tasks
        # 优先级维度：如果 config 中没有 num_priorities，则默认等于 num_tasks (常见于调度问题)
        num_priorities = getattr(config, 'num_priorities', num_tasks)
        D = num_tasks + num_priorities
    except AttributeError as e:
        print(f"致命错误: Config 缺少 num_tasks 属性。无法确定维度 D。错误: {e}")
        return ([], [], [], [], [], [], {}, [], 0.0, [], [])  # 返回空结果以避免崩溃

    # MOACO 核心结构：信息素向量 Tau
    initial_tau = 1.0 / (num_ants * D) if (num_ants * D) > 0 else 1.0
    pheromone_tau_vec = np.full(D, initial_tau)

    # 蚁群（Agents）
    # 确保使用修复后的 num_ants
    swarm = [Particle(config, feasible_init=True) for _ in range(num_ants)]
    archive = []

    # 记录变量（与前两个算法保持一致）
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

    # 初始化粒子 (Agents) - 解码、评估
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

    # 初始档案更新 (与前两个算法保持一致)
    valid_particles = [p for p in swarm if p.schedule and p.fitness is not None and not any(np.isnan(p.fitness))]
    if valid_particles:
        archive = update_archive_simplified(valid_particles, [], config)

    archive_size_history.append(len(archive))

    # ===== HV 参考点对齐 =====
    # (HV 参考点对齐逻辑保持不变，确保 ref_point 定义正确)
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
        current_hv = calculate_hypervolume(init_pareto, ref_point) if init_pareto else 0.0
    else:
        current_hv = 0.0
    hv_history.append(current_hv)

    # 加载参考集
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

        # 1. 信息素蒸发
        pheromone_tau_vec = (1 - rho) * pheromone_tau_vec

        # 2. 路径构建与评估 (使用修复后的 num_tasks 变量)

        for ant_idx in range(num_ants):
            ant_particle = swarm[ant_idx]
            # 注意：这里需要确保 ant_particle.position 的长度为 D
            new_position = np.zeros(D)  # 使用 D 维度

            # 启发式信息 Eta (简化：所有决策变量的启发式信息权重相同)
            eta_vec = np.ones(D)

            # 路径选择（构建决策变量）
            for i in range(D):
                tau = pheromone_tau_vec[i]
                eta = eta_vec[i]

                # 组合概率的简化映射
                selection_pressure = (alpha * tau + beta * eta) / (alpha + beta)

                # 资源分配 (前 num_tasks 维)
                if i < num_tasks:  # 使用修复后的 num_tasks
                    available = config.num_suppliers[i]
                    if available:
                        if random.random() < selection_pressure * 0.8:
                            # 倾向于利用信息素：随机选择
                            new_pos_val = random.choice(available)
                        else:
                            # 随机探索
                            new_pos_val = random.choice(available)

                        new_position[i] = new_pos_val
                    else:
                        new_position[i] = ant_particle.position[i]

                        # 优先级 (后 num_priorities 维)
                else:
                    if random.random() < selection_pressure:
                        # 局部开发
                        perturbation = random.uniform(-0.1, 0.1)
                        new_position[i] = np.clip(ant_particle.position[i] + perturbation, 0, 1)
                    else:
                        # 全局探索
                        new_position[i] = random.random()

            # 更新粒子状态
            ant_particle.position = list(new_position)
            # 使用 num_tasks 进行切片
            ant_particle.resource_assignment = new_position[:num_tasks]
            ant_particle.priority_values = new_position[num_tasks:]

            # 评估
            ant_particle.schedule = decode(ant_particle, config)
            if ant_particle.schedule:
                ant_particle.fitness = evaluate(ant_particle.schedule, config)

                if ant_particle.fitness is not None and not any(np.isnan(ant_particle.fitness)):
                    current_gen_fitness.append(ant_particle.fitness)

                    if dominates(ant_particle.fitness, ant_particle.pbest_fitness):
                        ant_particle.update_pbest()

                    if not np.isnan(ant_particle.fitness[0]):
                        global_best_time = min(global_best_time, ant_particle.fitness[0])
                    if not np.isnan(ant_particle.fitness[1]):
                        global_best_cost = min(global_best_cost, ant_particle.fitness[1])

        # 3. 更新档案
        valid_particles = [p for p in swarm if p.schedule and p.fitness is not None and not any(np.isnan(p.fitness))]
        archive = update_archive_simplified(valid_particles, archive, config)

        # 4. 信息素更新 (MOACO 核心)
        delta_tau_contribution = 1.0 / (len(archive) if len(archive) > 0 else 1.0)

        for p in archive:
            if p.fitness is not None and not any(np.isnan(p.fitness)):
                for i in range(D):
                    pheromone_tau_vec[i] += delta_tau_contribution

        # 5. 记录指标和输出 (保持一致)
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
                  f"HV={current_hv:.2f} | IGD={igd_now:.3f} | Δ={spread_now:.3f} | Best Time={global_best_time:.1f} | "
                  f"Best Cost={global_best_cost:.1f}")

    # 计算最终Spacing
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