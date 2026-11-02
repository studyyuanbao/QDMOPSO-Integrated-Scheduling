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
    """
    标准档案更新：使用非支配排序和拥挤距离
    """
    # 过滤有效解
    all_solutions = [p for p in candidates + archive
                     if p.fitness and all(v != float('inf') for v in p.fitness)]

    if not all_solutions:
        return archive

    # 非支配排序
    fronts = fast_non_dominated_sort(all_solutions)

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


def select_guide_from_archive(archive):
    """
    安全地从档案中选择引导解
    """
    if not archive:
        return None

    # 计算拥挤距离
    crowding_distances = calculate_crowding_distance(archive)

    # 检查是否有有效的拥挤距离
    valid_distances = [d for d in crowding_distances if not np.isnan(d) and not np.isinf(d)]

    if not valid_distances or sum(valid_distances) == 0:
        # 如果所有拥挤距离都无效，随机选择
        return random.choice(archive)

    # 构建概率分布（确保没有NaN）
    probabilities = []
    for d in crowding_distances:
        if np.isnan(d) or np.isinf(d) or d < 0:
            probabilities.append(0.0)
        else:
            probabilities.append(d)

    # 归一化概率
    total = sum(probabilities)
    if total == 0:
        # 如果总和为0，使用均匀分布
        return random.choice(archive)

    probabilities = [p / total for p in probabilities]

    # 确保概率有效
    if any(np.isnan(p) for p in probabilities) or sum(probabilities) == 0:
        return random.choice(archive)

    try:
        guide_idx = np.random.choice(len(archive), p=probabilities)
        return archive[guide_idx]
    except (ValueError, RuntimeError):
        # 如果概率选择失败，随机选择
        return random.choice(archive)


def generate_neighbor_moabc(source, guide, config):
    """
    标准的邻域搜索：使用合理的扰动策略
    """
    candidate = Particle(config)
    candidate.position = source.position.copy()

    # 选择要修改的维度
    j = random.randrange(len(candidate.position))

    # 合理的扰动范围
    phi = random.uniform(-1.0, 1.0)

    if j < config.num_tasks:
        # 资源分配：基于引导解的改进
        if random.random() < 0.7:  # 70%概率使用引导信息
            # 在源解和引导解之间进行插值
            candidate.position[j] = source.position[j] + phi * (source.position[j] - guide.position[j])
            # 约束处理
            max_res_idx = config.num_suppliers[j][-1] if config.num_suppliers[j] else 1
            candidate.position[j] = int(np.clip(round(candidate.position[j]), 0, max_res_idx))
        else:
            # 30%概率随机探索
            candidate.position[j] = random.choice(config.num_suppliers[j]) if config.num_suppliers[j] else 0
    else:
        # 优先级：连续值扰动
        candidate.position[j] = source.position[j] + phi * (source.position[j] - guide.position[j])
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
    """
    标准的多目标人工蜂群算法
    """
    archive_size_history, all_fitness = [], []
    global_min_time_list, global_min_cost_list = [], []
    hv_history, igd_values, sp_history = [], [], []

    num_sources = config.num_particles
    limit = getattr(config, 'trial_limit', 10)  # 合理的尝试限制

    # 初始化 - 生成可行解
    sources = []
    for _ in range(num_sources):
        p = Particle(config, feasible_init=True)  # 总是生成可行解
        p.schedule = decode(p, config)
        p.fitness = safe_fitness(evaluate(p.schedule, config))
        sources.append(p)

    # ===== 参考点设置（与pso_rl_new对齐） =====
    initial_ref = getattr(config, 'hv_ref_point', None)
    if initial_ref is not None:
        ref_point = initial_ref
    else:
        valid_init = [p for p in sources if
                      p.schedule and p.fitness and all(not np.isnan(v) and not np.isinf(v) for v in p.fitness)]
        fitness_init = safe_extract_fitness_tuples(valid_init)
        if fitness_init:
            max_time = max(t for t, c in fitness_init)
            max_cost = max(c for t, c in fitness_init)
            ref_point = (max_time * 1.5 + 1.0, max_cost * 1.5 + 1.0)
        else:
            ref_point = (20000.0, 20000.0)
        config.hv_ref_point = ref_point

    print(f"使用的 HV 参考点 (ref_point) = {ref_point}")

    trial = [0] * num_sources

    # 初始档案更新
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

    print("开始标准MOABC优化...")
    for iteration in range(config.num_iterations):
        new_solutions = []

        # 雇佣蜂阶段 - 标准搜索策略
        for i, s in enumerate(sources):
            if s.fitness[0] == float('inf'):
                trial[i] += 1
                continue

            # 从档案中选择引导解
            if archive:
                guide = select_guide_from_archive(archive)
                if guide is None:
                    # 如果引导解选择失败，从有效源中随机选择
                    valid_sources = [p for p in sources if p.fitness[0] != float('inf')]
                    if not valid_sources:
                        continue
                    guide = random.choice(valid_sources)
            else:
                # 如果没有档案，从源中随机选择
                valid_sources = [p for p in sources if p.fitness[0] != float('inf')]
                if not valid_sources:
                    continue
                guide = random.choice(valid_sources)

            candidate = generate_neighbor_moabc(s, guide, config)
            candidate.schedule = decode(candidate, config)
            candidate.fitness = safe_fitness(evaluate(candidate.schedule, config))

            if candidate.fitness[0] == float('inf'):
                trial[i] += 1
                continue

            # 标准替换策略：新解不支配旧解时才替换
            old_fit = s.fitness
            new_fit = candidate.fitness

            if dominates(new_fit, old_fit) or (not dominates(old_fit, new_fit) and random.random() < 0.5):
                sources[i] = candidate
                trial[i] = 0
                new_solutions.append(candidate)
            else:
                trial[i] += 1

        # 侦察蜂阶段 - 标准策略
        for i in range(num_sources):
            if trial[i] >= limit:
                # 生成新解（保持可行性）
                new_source = Particle(config, feasible_init=True)
                new_source.schedule = decode(new_source, config)
                new_source.fitness = safe_fitness(evaluate(new_source.schedule, config))
                sources[i] = new_source
                trial[i] = 0
                if new_source.fitness[0] != float('inf'):
                    new_solutions.append(new_source)

        # 档案更新 - 每次迭代都更新
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
            # 保持上一次的值
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