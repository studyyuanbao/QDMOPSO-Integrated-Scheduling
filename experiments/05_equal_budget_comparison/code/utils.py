import math
from typing import List, Tuple

import numpy as np
from copy import deepcopy

def dominates(a, b):
    """标准Pareto支配判断：a 是否支配 b"""
    # 确保a和b都是列表或元组
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
        raise TypeError("适应度值必须是列表或元组")
    if len(a) != len(b):
        raise ValueError("适应度值的维度必须相同")

    # 如果a的每个维度都小于等于b，并且至少有一个维度严格小于b，则a支配b
    return all(a[i] <= b[i] for i in range(len(a))) and any(a[i] < b[i] for i in range(len(a)))


def nondominated(particles):
    """
    从粒子列表中筛选出非支配解
    :param particles: 粒子列表，每个粒子应该有fitness属性
    :return: 非支配解列表
    """
    if not particles:
        return []
    
    # 过滤出有效的粒子（有fitness且不为None或inf）
    valid_particles = []
    for p in particles:
        if hasattr(p, 'fitness') and p.fitness is not None:
            try:
                # 检查fitness是否有效
                if isinstance(p.fitness, (list, tuple)) and len(p.fitness) >= 2:
                    f1, f2 = float(p.fitness[0]), float(p.fitness[1])
                    if not (np.isnan(f1) or np.isnan(f2) or f1 == float('inf') or f2 == float('inf')):
                        valid_particles.append(p)
            except (ValueError, TypeError):
                continue
    
    if not valid_particles:
        return []
    
    # 筛选非支配解
    non_dominated = []
    for i, p1 in enumerate(valid_particles):
        is_dominated = False
        for j, p2 in enumerate(valid_particles):
            if i != j:
                # 检查p2是否支配p1
                if dominates(p2.fitness, p1.fitness):
                    is_dominated = True
                    break
        if not is_dominated:
            non_dominated.append(p1)
    
    return non_dominated

def epsilon_dominates(a, b, epsilon):
    """
    ε-支配判断：a 是否 ε-支配 b
    a ε-支配 b，当且仅当对于所有目标 i，a[i] <= b[i] + epsilon 且至少存在一个目标 k，使得 a[k] < b[k] + epsilon
    """
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
        raise TypeError("适应度值必须是列表或元组")
    if len(a) != len(b):
        raise ValueError("适应度值的维度必须相同")

    condition1 = True  # all(a[i] <= b[i] + epsilon for i in range(len(a)))
    condition2 = False  # any(a[i] < b[i] + epsilon for i in range(len(a)))

    for i in range(len(a)):
        if a[i] > b[i] + epsilon:
            condition1 = False
            break
        if a[i] < b[i] + epsilon:
            condition2 = True

    return condition1 and condition2


# 去重（浮点容差）
def remove_duplicates_particles(particles, decimals=8):
    unique = []
    seen = set()
    for p in particles:
        # 保护：确保有 fitness 且可迭代、长度 >= 2
        if not hasattr(p, 'fitness') or p.fitness is None:
            continue
        try:
            fitness_iter = tuple(float(x) for x in p.fitness)
        except Exception:
            continue
        # 进一步过滤 NaN/inf
        if any(np.isnan(v) or np.isinf(v) for v in fitness_iter):
            continue
        key = tuple(round(float(x), decimals) for x in fitness_iter)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def update_archive(new_particles, convergence_archive, diversity_archive, config):
    """
    基于双档案机制的存档更新（性能优先）：
    - CA：用 ε-支配筛选 + Iε 指标排序，强调收敛
    - DA：用标准 Pareto 非支配 + SDE 多样性排序，强调分布
    两个档案独立从各自的合并池更新，以提升稳定性与性能。
    """
    # 1) 合并与去重（容差）
    combined_ca = remove_duplicates_particles(new_particles + convergence_archive, decimals=8)
    combined_da = remove_duplicates_particles(new_particles + diversity_archive, decimals=8)

    # 2) 更新 CA：ε-支配非支配筛选
    temp_ca = []
    for p_candidate in combined_ca:
        is_dominated = False
        for p_existing in combined_ca:
            if p_candidate != p_existing and epsilon_dominates(
                    p_existing.fitness, p_candidate.fitness, getattr(config, 'epsilon', 0.0)):
                is_dominated = True
                break
        if not is_dominated:
            temp_ca.append(p_candidate)

    # Iε 指标排序（收敛优先），使用稳定排序确保边界解优先
    # 先计算所有fitness值，避免排序过程中依赖变化
    fitness_scores = []
    for p in temp_ca:
        fitness_val = calculate_fitness(p, temp_ca)
        fitness_scores.append((fitness_val, p))

    # 使用 reverse=True 更稳健（适配正/负评分）
    fitness_scores.sort(key=lambda x: x[0], reverse=True)
    temp_ca = [p for _, p in fitness_scores]
    
    ca_size = max(0, getattr(config, 'ca_size', 100))
    convergence_archive = temp_ca[:ca_size] if len(temp_ca) > ca_size else temp_ca

    # 3) 更新 DA：标准 Pareto 非支配筛选 + SDE 排序
    temp_da = []
    for p_candidate in combined_da:
        is_dominated = False
        for p_existing in combined_da:
            if p_candidate != p_existing and dominates(p_existing.fitness, p_candidate.fitness):
                is_dominated = True
                break
        if not is_dominated:
            temp_da.append(p_candidate)
    temp_da = sort_by_diversity(temp_da, config)
    da_size = max(0, getattr(config, 'da_size', 100))
    diversity_archive = temp_da[:da_size] if len(temp_da) > da_size else temp_da

    return deepcopy(convergence_archive), deepcopy(diversity_archive)



def calculate_fitness(particle, archive):
    """
    计算解的Iε适应度值(用于收敛档案)
    参考论文公式(5)
    修复：防止math.exp溢出
    """
    alpha = 0.1  # 缩放因子
    fitness = 0.0
    for p in archive:
        if p != particle:
            # 计算Iε指标
            ie = max(
                particle.fitness[i] - p.fitness[i]
                for i in range(len(particle.fitness))
            )
            # 防止math.exp溢出：限制指数范围
            exp_arg = ie / alpha
            if exp_arg > 700:  # math.exp(700)接近float最大值
                exp_val = float('inf')
            elif exp_arg < -700:  # math.exp(-700)接近0
                exp_val = 0.0
            else:
                exp_val = math.exp(exp_arg)
            fitness -= exp_val
    return fitness


def sort_by_diversity(archive, config):
    """
    基于Shift-based Density Estimation的多样性排序
    参考论文第3.1节【基于转移的密度估计策略】
    修复：返回按SDE得分从高到低排序的稳定列表
    """
    if len(archive) <= 1:
        return archive

    # 1. 保留边界解
    sorted_by_time = sorted(archive, key=lambda p: p.fitness[0])
    sorted_by_cost = sorted(archive, key=lambda p: p.fitness[1])

    # 使用列表而不是set，保持顺序
    selected = []
    boundary_particles = [sorted_by_time[0], sorted_by_time[-1],
                         sorted_by_cost[0], sorted_by_cost[-1]]
    
    # 去重并保持顺序
    for p in boundary_particles:
        if p not in selected:
            selected.append(p)

    remaining = [p for p in archive if p not in selected]

    # 2. 迭代选择最不相似解
    while remaining:
        # 计算每个剩余解与已选解的最小SDE距离
        sde_distances = []
        for p in remaining:
            min_sde = float('inf')
            for s in selected:
                # 计算shifted距离
                shifted_fitness = [
                    max(p.fitness[0], s.fitness[0]) if p.fitness[0] < s.fitness[0] else p.fitness[0],
                    max(p.fitness[1], s.fitness[1]) if p.fitness[1] < s.fitness[1] else p.fitness[1]
                ]
                dist = math.sqrt(
                    (shifted_fitness[0] - p.fitness[0]) ** 2 +
                    (shifted_fitness[1] - p.fitness[1]) ** 2
                )
                min_sde = min(min_sde, dist)
            sde_distances.append((min_sde, p))

        # 选择最大SDE距离的解
        best_dist, best_p = max(sde_distances, key=lambda x: x[0])
        selected.append(best_p)
        remaining.remove(best_p)

    # 3. 按SDE得分从高到低排序，确保稳定顺序
    sde_scores = []
    for p in selected:
        min_sde = float('inf')
        for s in selected:
            if p != s:
                # 计算shifted距离
                shifted_fitness = [
                    max(p.fitness[0], s.fitness[0]) if p.fitness[0] < s.fitness[0] else p.fitness[0],
                    max(p.fitness[1], s.fitness[1]) if p.fitness[1] < s.fitness[1] else p.fitness[1]
                ]
                dist = math.sqrt(
                    (shifted_fitness[0] - p.fitness[0]) ** 2 +
                    (shifted_fitness[1] - p.fitness[1]) ** 2
                )
                min_sde = min(min_sde, dist)
        sde_scores.append((min_sde, p))
    
    # 按SDE得分从高到低排序，确保边界解优先
    sde_scores.sort(key=lambda x: -x[0])
    return [p for _, p in sde_scores]

def archive_improved(prev_archive, new_archive):
    """检查新解是否改进了档案"""
    if not prev_archive:
        return True
    for sol in new_archive:
        is_nondominated = True
        for a in prev_archive:
            if dominates(a.fitness, sol.fitness):
                is_nondominated = False
                break
        if is_nondominated:
            return True
    return False


def calculate_igd(fitness_tuples, reference_fitness_tuples):
    """
    计算 IGD (Inverted Generational Distance)。
    定义：对每个参考点，找到解集中最近的解的距离，然后对所有参考点取平均。
    """

    # --- 1. 处理 solution set (fitness_tuples) ---
    # 假设 fitness_tuples 仍是 Python 列表/元组
    if not fitness_tuples:
        sol_arr = np.empty((0, 2))  # 保持二维结构
    else:
        sol_arr = np.array(fitness_tuples, dtype=float)

    # --- 2. 处理 reference set (reference_fitness_tuples) ---
    if reference_fitness_tuples is None:
        # 如果 reference_set 加载失败或明确为 None
        ref_arr = np.empty((0, 2))
    elif isinstance(reference_fitness_tuples, np.ndarray):
        # 如果是 NumPy 数组，避免布尔判断，直接检查 size
        if reference_fitness_tuples.size > 0:
            ref_arr = reference_fitness_tuples.astype(float)
        else:
            ref_arr = np.empty((0, 2))
    else:
        # 如果是 Python 列表/元组
        if reference_fitness_tuples:
            ref_arr = np.array(reference_fitness_tuples, dtype=float)
        else:
            ref_arr = np.empty((0, 2))

    # --- 3. 检查空集 ---
    # 使用 .size 来安全判断 NumPy 数组是否为空
    if sol_arr.size == 0 or ref_arr.size == 0:
        return float("inf")

    # 确保数组是二维的 (N, M)
    if sol_arr.ndim == 1:
        sol_arr = sol_arr.reshape(1, -1)
    if ref_arr.ndim == 1:
        ref_arr = ref_arr.reshape(1, -1)

    # 维度检查
    if sol_arr.shape[1] != ref_arr.shape[1]:
        raise ValueError("解集与参考集维度不一致，无法计算 IGD")

    # 逐个参考点，计算到解集的最近距离
    min_dists = []
    for r in ref_arr:
        # 注意：这里假设 r 和 sol_arr 都是 (1, M) 和 (N, M)
        dists = np.linalg.norm(sol_arr - r, axis=1)
        min_dists.append(np.min(dists))

    return float(np.mean(min_dists))


def calculate_hypervolume(points, ref_point):
    """
    2D hypervolume for minimization problems.
    points: iterable of (x,y) or Nx2 array
    ref_point: (ref_x, ref_y) — 参考点（必须比所有想要计入的点更差）
    返回：超体积（float）
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return 0.0

    # 只保留在参考点“左下”区域内的点（即 x <= ref_x 且 y <= ref_y）
    mask = (points[:, 0] <= ref_point[0]) & (points[:, 1] <= ref_point[1])
    points = points[mask]
    if points.size == 0:
        return 0.0

    # 去重（可选，但有助于稳定性）
    points = np.unique(points, axis=0)

    # 筛出非支配点（Pareto 前沿）。这里为“没有其他点支配它”的点被保留。
    nondominated = []
    n = len(points)
    for i in range(n):
        p = points[i]
        dominated_by_any = False
        for j in range(n):
            if i == j:
                continue
            q = points[j]
            # q 支配 p 当且仅当 q <= p（逐项）且至少有一项严格小（最小化问题）
            if np.all(q <= p) and np.any(q < p):
                dominated_by_any = True
                break
        if not dominated_by_any:
            nondominated.append(tuple(p))

    if not nondominated:
        return 0.0

    front = np.array(nondominated)
    # 按第一目标升序（x 小到大）
    front = front[np.argsort(front[:, 0])]

    # 计算超体积：从右向左累加矩形面积
    hv = 0.0
    last_x = ref_point[0]
    ref_y = ref_point[1]

    # 为安全起见，如果最差 y 比 ref_y 大，则将其裁剪到 ref_y（不过前面已筛掉 y > ref_y 的点）
    for x, y in front[::-1]:
        width = last_x - x
        height = ref_y - y
        if width > 0 and height > 0:
            hv += width * height
        last_x = x

    return float(hv)


def calculate_spacing(fitness_tuples, debug=True):
    """
    修复版的Spacing计算
    """
    if not fitness_tuples or len(fitness_tuples) < 2:
        if debug:
            print(f"⚠️ Spacing: 解数量不足")
        return 0.0

    try:
        # 去重处理（保持原样）
        unique_fitness = []
        seen = set()
        for ft in fitness_tuples:
            key = (round(ft[0], 10), round(ft[1], 10))
            if key not in seen:
                seen.add(key)
                unique_fitness.append(ft)

        if len(unique_fitness) < 2:
            return 0.0

        arr = np.array(unique_fitness, dtype=float)
        n_solutions = len(arr)

        # 优化：使用向量化计算距离矩阵
        # 计算每对解之间的欧氏距离
        dist_matrix = np.linalg.norm(arr[:, np.newaxis, :] - arr[np.newaxis, :, :], axis=2)

        # 将对角线设为很大的值（避免选择自身）
        np.fill_diagonal(dist_matrix, np.inf)

        # 找到每个解到其他解的最小距离
        min_dists = np.min(dist_matrix, axis=1)

        # 计算Spacing指标
        d_mean = np.mean(min_dists)
        sp = np.sqrt(np.sum((min_dists - d_mean) ** 2) / (n_solutions - 1))  # 注意这里是n-1

        return float(sp)

    except Exception as e:
        if debug:
            print(f"❌ Spacing计算错误: {e}")
        return 0.0


def calculate_spread(pareto_front: List[Tuple[float, float]], reference_set: List[Tuple[float, float]]) -> float:
    """
    计算简化的 Spread 指标 (Δ)，衡量解集的分布范围。
    该指标衡量档案解集与真实PF边界的距离（越小越好）。

    Args:
        pareto_front: 当前非支配解集的适应度元组列表。
        reference_set: 真实或估计的 Pareto 最优集适应度元组列表。

    Returns:
        float: Spread 值 (越小越好)。
    """
    if not pareto_front or not reference_set:
        return 0.0

    # 1. 找到参考集（真实PF）的边界点
    ref_min_t = min(t for t, c in reference_set)
    ref_max_t = max(t for t, c in reference_set)
    ref_min_c = min(c for t, c in reference_set)
    ref_max_c = max(c for t, c in reference_set)

    # 2. 找到当前档案的边界点
    arc_min_t = min(t for t, c in pareto_front)
    arc_max_t = max(t for t, c in pareto_front)
    arc_min_c = min(c for t, c in pareto_front)
    arc_max_c = max(c for t, c in pareto_front)

    # 3. 计算极值点与真实边界的距离
    D_t = abs(arc_min_t - ref_min_t) + abs(arc_max_t - ref_max_t)
    D_c = abs(arc_min_c - ref_min_c) + abs(arc_max_c - ref_max_c)

    # 最终 Spread 指标简化为边界覆盖程度
    return D_t + D_c