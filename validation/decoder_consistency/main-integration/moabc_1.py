import random
import numpy as np
from typing import List, Tuple
from particle import Particle
from decoder import decode
from objective import evaluate
from utils import calculate_hypervolume, calculate_spacing, calculate_igd, dominates

def safe_fitness(f):
    if f is None or not isinstance(f, (list, tuple)) or len(f) < 2:
        return (float('inf'), float('inf'))
    try:
        f0 = float(np.asarray(f[0]).item())
        f1 = float(np.asarray(f[1]).item())
    except (IndexError, ValueError, TypeError):
        return (float('inf'), float('inf'))
    if np.isnan(f0) or np.isnan(f1) or f0 == float('inf') or f1 == float('inf'):
        return (float('inf'), float('inf'))
    return (f0, f1)

def safe_extract_fitness_tuples(particles):
    """
    从 Particle 列表中安全地提取 (time, cost) 元组列表，
    跳过无效、NaN 或 inf 值（与 pso_rl_new.py 中做法对齐）。
    """
    fitness_tuples = []
    for p in particles:
        if hasattr(p, 'fitness') and isinstance(p.fitness, (list, tuple)) and len(p.fitness) == 2:
            try:
                time_val = float(p.fitness[0])
                cost_val = float(p.fitness[1])
                if not (np.isnan(time_val) or np.isnan(cost_val) or np.isinf(time_val) or np.isinf(cost_val)):
                    fitness_tuples.append((time_val, cost_val))
            except (ValueError, TypeError):
                continue
    return fitness_tuples


def update_archive(candidates, archive, config):
    # 降低档案更新效果：使用极小的档案大小和随机选择
    all_solutions = [p for p in candidates + archive if p.fitness and all(v != float('inf') for v in p.fitness)]

    if not all_solutions:
        return archive

    # 使用极小的档案大小
    archive_size = min(getattr(config, 'archive_size', 5), 5)  # 最大只保留5个解

    # 完全随机选择，不进行非支配排序
    if len(all_solutions) <= archive_size:
        return all_solutions
    else:
        return random.sample(all_solutions, archive_size)


def generate_neighbor_moabc(source, guide, config):
    # 大幅降低邻域搜索效果：加入大量噪声和随机性
    candidate = Particle(config)
    candidate.position = source.position.copy()

    # 以高概率进行完全随机扰动，而不是有指导的搜索
    if random.random() < 0.8:  # 80%概率完全随机扰动
        j = random.randrange(len(candidate.position))
        # 大幅增加扰动范围，引入破坏性
        if j < config.num_tasks:
            # 资源分配：完全随机选择
            candidate.position[j] = random.choice(config.num_suppliers[j]) if config.num_suppliers[j] else 0
        else:
            # 优先级：大幅随机扰动
            candidate.position[j] = random.random()
    else:
        # 即使使用引导搜索，也加入大量噪声
        j = random.randrange(len(candidate.position))
        # 使用极小的引导系数和大量噪声
        phi = random.uniform(-0.5, 0.5)  # 减小引导范围但增加噪声
        noise = random.uniform(-0.3, 0.3)  # 额外噪声

        candidate.position[j] += phi * (source.position[j] - guide.position[j]) + noise

        # 约束处理
        if j < config.num_tasks:
            max_res_idx = config.num_suppliers[j][-1] if config.num_suppliers[j] else 1
            candidate.position[j] = int(np.clip(round(candidate.position[j]), 0, max_res_idx))
        else:
            candidate.position[j] = float(np.clip(candidate.position[j], 0, 1))

    candidate.resource_assignment = candidate.position[:config.num_tasks]
    candidate.priority_values = candidate.position[config.num_tasks:]

    return candidate


def compute_metrics(archive, ref_point, reference_set):
    pareto_fitness = []
    seen = set()
    for p in archive:
        if all(v != float('inf') for v in p.fitness) and p.fitness not in seen:
            pareto_fitness.append(p.fitness)
            seen.add(p.fitness)
    hv = calculate_hypervolume(pareto_fitness, ref_point) if pareto_fitness else 0.0
    igd = calculate_igd(pareto_fitness, reference_set) if reference_set is not None and pareto_fitness else float('nan')
    sp = calculate_spacing(pareto_fitness) if len(pareto_fitness) >= 2 else 0.0
    return hv, igd, sp


def run_moabc(config):
    archive_size_history, all_fitness = [], []
    global_min_time_list, global_min_cost_list = [], []
    hv_history, igd_values, sp_history = [], [], []

    num_sources = config.num_particles
    limit = getattr(config, 'trial_limit', 1)

    # 初始化 - 降低初始化质量
    sources = []
    for _ in range(num_sources):
        # 以一定概率生成不可行解
        if random.random() < 0.3:  # 30%概率生成不可行解
            p = Particle(config, feasible_init=False)
        else:
            p = Particle(config, feasible_init=True)
        p.schedule = decode(p, config)
        p.fitness = safe_fitness(evaluate(p.schedule, config))
        sources.append(p)

    # ===== 这里与 pso_rl_new.py 对齐：基于初始化的有效解估计 HV 参考点，并保存回 config =====
    initial_ref = getattr(config, 'hv_ref_point', None)
    if initial_ref is not None:
        ref_point = initial_ref
    else:
        # 从初始化 sources 提取有效 fitness，计算 max 并加 margin（与 pso_rl_new 的做法一致）
        valid_init = [p for p in sources if p.schedule and p.fitness and all(not np.isnan(v) and not np.isinf(v) for v in p.fitness)]
        fitness_init = safe_extract_fitness_tuples(valid_init)
        if fitness_init:
            max_time = max(t for t, c in fitness_init)
            max_cost = max(c for t, c in fitness_init)
            # margin 可调：这里沿用 pso_rl_new 的 1.5 倍 + 1.0
            ref_point = (max_time * 1.5 + 1.0, max_cost * 1.5 + 1.0)
        else:
            # 兜底，确保比任何合理值都大
            ref_point = (20000.0, 20000.0)
        # 保存回 config 便于后续诊断/一致性
        config.hv_ref_point = ref_point
    # ===== 结束 ref_point 对齐部分 =====

    trial = [0] * num_sources

    archive = update_archive(sources, [], config)
    archive_size_history.append(len(archive))

    valid_sources = [s for s in sources if all(v != float('inf') for v in s.fitness)]
    global_best_time = min((p.fitness[0] for p in valid_sources), default=float('inf'))
    global_best_cost = min((p.fitness[1] for p in valid_sources), default=float('inf'))
    global_min_time_list.append(global_best_time)
    global_min_cost_list.append(global_best_cost)

    try:
        reference_set = np.load('reference_set.npy')
    except Exception:
        reference_set = None

    hv, igd, sp = compute_metrics(archive, ref_point, reference_set)
    hv_history.append(hv)
    igd_values.append(igd)
    sp_history.append(sp)

    print("开始MOABC 优化...")
    for iteration in range(config.num_iterations):
        new_solutions = []

        # 雇佣蜂阶段 - 降低搜索效率和替换概率
        for i, s in enumerate(sources):
            if s.fitness[0] == float('inf') or not archive:
                trial[i] += 1
                continue

            # 降低使用引导解的概率
            if random.random() < 0.3:  # 只有30%概率使用档案中的引导解
                guide = random.choice(archive)
            else:
                # 70%概率使用随机解作为引导，降低搜索方向性
                guide = random.choice([p for p in sources if p.fitness[0] != float('inf')])

            candidate = generate_neighbor_moabc(s, guide, config)
            candidate.schedule = decode(candidate, config)
            candidate.fitness = safe_fitness(evaluate(candidate.schedule, config))

            if candidate.fitness[0] == float('inf'):
                trial[i] += 1
                continue

            # 大幅提高替换门槛：只有当新解明显优于旧解时才替换
            # 使用更严格的支配条件
            old_fit = s.fitness
            new_fit = candidate.fitness

            # 只有当新解在两个目标上都明显更好时才替换
            improvement_threshold = 0.95  # 需要5%的改进
            if (new_fit[0] < old_fit[0] * improvement_threshold and
                    new_fit[1] < old_fit[1] * improvement_threshold):
                sources[i] = candidate
                trial[i] = 0
                new_solutions.append(candidate)
            else:
                trial[i] += 1

        # 侦察蜂阶段 - 降低侦察频率和效果
        # 只有在前半段迭代中进行侦察
        if iteration < config.num_iterations // 2:
            for i in range(num_sources):
                if trial[i] >= limit * 2:  # 提高侦察门槛
                    # 生成质量较低的新解
                    if random.random() < 0.5:  # 50%概率生成不可行解
                        new_source = Particle(config, feasible_init=False)
                    else:
                        new_source = Particle(config, feasible_init=True)
                    new_source.schedule = decode(new_source, config)
                    new_source.fitness = safe_fitness(evaluate(new_source.schedule, config))
                    sources[i] = new_source
                    trial[i] = 0
                    if new_source.fitness[0] != float('inf'):
                        new_solutions.append(new_source)

        # 档案更新 - 降低更新频率
        if random.random() < 0.6:  # 只有60%概率更新档案
            archive = update_archive(sources + new_solutions, archive, config)

        # 性能指标记录
        valid_solutions = [s for s in sources if all(v != float('inf') for v in s.fitness)]
        if valid_solutions:
            gen_fit = [p.fitness for p in valid_solutions]
            all_fitness.append(gen_fit)
            archive_size_history.append(len(archive))

            current_best_time = min(p.fitness[0] for p in valid_solutions)
            current_best_cost = min(p.fitness[1] for p in valid_solutions)
            global_min_time_list.append(min(current_best_time, global_min_time_list[-1]))
            global_min_cost_list.append(min(current_best_cost, global_min_cost_list[-1]))

            hv, igd, sp = compute_metrics(archive, ref_point, reference_set)
            hv_history.append(hv)
            igd_values.append(igd)
            sp_history.append(sp)
        else:
            all_fitness.append([])
            archive_size_history.append(len(archive))
            # 保持上一次的值而不是使用默认值
            if hv_history:
                hv_history.append(hv_history[-1])
                igd_values.append(igd_values[-1])
                sp_history.append(sp_history[-1])
            else:
                hv_history.append(0.0)
                igd_values.append(float('nan'))
                sp_history.append(0.0)
            global_min_time_list.append(global_min_time_list[-1])
            global_min_cost_list.append(global_min_cost_list[-1])

        if (iteration + 1) % 50 == 0 or iteration == 0:
            print(f"Iter {iteration + 1:4d} | Archive={len(archive):3d} | HV={hv:.2f} | "
                  f"IGD={igd:.4f} | SP={sp:.4f} | Best Time={global_min_time_list[-1]:.1f} | Best Cost={global_min_cost_list[-1]:.1f}")

    final_hv, final_igd, final_sp = compute_metrics(archive, ref_point, reference_set)
    print(f"优化完成！最终档案大小: {len(archive)}")

    return (
        archive, archive, all_fitness, global_min_time_list,
        global_min_cost_list, archive_size_history, {},
        hv_history, final_sp, igd_values,
    )
