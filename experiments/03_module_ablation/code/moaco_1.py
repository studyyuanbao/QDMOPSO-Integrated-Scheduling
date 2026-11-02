import random
import numpy as np
from typing import List, Tuple, Any
from particle import Particle  # 保持接口兼容
from decoder import decode
from objective import evaluate
from utils import dominates, calculate_hypervolume, calculate_spacing, calculate_igd

# ------------------ 工具函数 ------------------
def solution_to_point(sol: Any):
    """
    从多种可能的解表示中提取 (time, cost)。
    支持：
      - tuple/list (t,c)
      - 对象 .fitness (list/tuple)
      - dict with 'fitness' key
    返回 (t, c) 或 None（表示该解无效）
    """
    if sol is None:
        return None
    # tuple/list 直接表示
    if isinstance(sol, (tuple, list)) and len(sol) >= 2:
        try:
            t = float(sol[0]); c = float(sol[1])
            if np.isfinite(t) and np.isfinite(c):
                return (t, c)
        except Exception:
            return None
    # 对象属性 .fitness
    try:
        f = getattr(sol, 'fitness', None)
        if f is not None and isinstance(f, (tuple, list)) and len(f) >= 2:
            t = float(f[0]); c = float(f[1])
            if np.isfinite(t) and np.isfinite(c):
                return (t, c)
    except Exception:
        pass
    # dict['fitness']
    try:
        if isinstance(sol, dict) and 'fitness' in sol:
            f = sol['fitness']
            if isinstance(f, (tuple, list)) and len(f) >= 2:
                t = float(f[0]); c = float(f[1])
                if np.isfinite(t) and np.isfinite(c):
                    return (t, c)
    except Exception:
        pass
    return None

# ------------------ 蚂蚁与矩阵初始化 ------------------
class Ant:
    """蚂蚁类，用于构建解（资源分配 + 优先级）"""
    def __init__(self, config):
        self.config = config
        self.position = [0.0] * (2 * config.num_tasks)
        self.resource_assignment = [None] * config.num_tasks
        self.priority_values = [0.0] * config.num_tasks
        self.schedule = None
        self.fitness = [float('inf'), float('inf')]

    def construct_solution(self, pheromone_matrix, heuristic_matrix, alpha=1.0, beta=2.0):
        # 资源分配
        for task_id in range(self.config.num_tasks):
            available_resources = self.config.num_suppliers[task_id]
            if not available_resources:
                self.resource_assignment[task_id] = None
                self.position[task_id] = -1
                continue

            weights = []
            for resource in available_resources:
                pher = pheromone_matrix.get(task_id, {}).get(resource, 1.0)
                heur = heuristic_matrix.get(task_id, {}).get(resource, 1.0)
                try:
                    val = (max(pher, 1e-8) ** alpha) * (max(heur, 1e-8) ** beta)
                except Exception:
                    val = 1.0
                weights.append(val)

            total = sum(weights)
            if total > 0 and len(weights) == len(available_resources):
                probs = [w / total for w in weights]
                idx = np.random.choice(len(available_resources), p=probs)
                selected = available_resources[idx]
            else:
                selected = random.choice(available_resources)

            self.resource_assignment[task_id] = selected
            self.position[task_id] = selected

        # 优先级
        for i in range(self.config.num_tasks):
            pv = random.random()
            self.priority_values[i] = pv
            self.position[self.config.num_tasks + i] = pv

        # 解码与评估
        self.schedule = decode(self, self.config)
        if self.schedule:
            try:
                fit = evaluate(self.schedule, self.config)
                if fit is None or len(fit) < 2:
                    self.fitness = [float('inf'), float('inf')]
                else:
                    self.fitness = [float(fit[0]), float(fit[1])]
            except Exception:
                self.fitness = [float('inf'), float('inf')]
        else:
            self.fitness = [float('inf'), float('inf')]


def initialize_pheromone_matrix(config, initial_value=1.0):
    pm = {}
    for task_id in range(config.num_tasks):
        pm[task_id] = {}
        for resource in config.num_suppliers[task_id]:
            pm[task_id][resource] = float(initial_value)
    return pm


def initialize_heuristic_matrix(config):
    hm = {}
    for task_id in range(config.num_tasks):
        hm[task_id] = {}
        for resource in config.num_suppliers[task_id]:
            info = config.machine_task_map.get((resource, task_id), {})
            cost = float(info.get('cost', 1.0))
            time = float(info.get('time', 1.0))
            heur = 1.0 / (cost + time + 1e-6)
            hm[task_id][resource] = heur
    return hm

# ------------------ 非支配排序与拥挤距离 ------------------
def fast_non_dominated_sort(solutions: List[Any]):
    fronts = []
    S = {}
    n = {}
    rank = {}

    for p in solutions:
        S[p] = []
        n[p] = 0

    front0 = []
    for p in solutions:
        for q in solutions:
            if p is q:
                continue
            if dominates(p.fitness, q.fitness):
                S[p].append(q)
            elif dominates(q.fitness, p.fitness):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            front0.append(p)
    fronts.append(front0)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S.get(p, []):
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
        else:
            break

    return fronts


def calculate_crowding_distance(front: List[Any]):
    m = len(front)
    if m == 0:
        return []
    distances = [0.0] * m
    if m == 1:
        return [float('inf')]
    num_obj = len(front[0].fitness)
    for obj_idx in range(num_obj):
        sorted_idx = sorted(range(m), key=lambda i: front[i].fitness[obj_idx])
        distances[sorted_idx[0]] = float('inf')
        distances[sorted_idx[-1]] = float('inf')
        min_obj = front[sorted_idx[0]].fitness[obj_idx]
        max_obj = front[sorted_idx[-1]].fitness[obj_idx]
        denom = max_obj - min_obj
        if denom <= 0:
            continue
        for j in range(1, m - 1):
            prev_val = front[sorted_idx[j - 1]].fitness[obj_idx]
            next_val = front[sorted_idx[j + 1]].fitness[obj_idx]
            distances[sorted_idx[j]] += (next_val - prev_val) / denom
    return distances

# ------------------ 信息素更新与档案管理 ------------------
def update_pheromone(pheromone_matrix, archive, rho=0.1, Q=100.0, min_pher=0.01, max_pher=1e6):
    for task_id in pheromone_matrix:
        for resource in pheromone_matrix[task_id]:
            pheromone_matrix[task_id][resource] *= (1.0 - rho)
            if pheromone_matrix[task_id][resource] < min_pher:
                pheromone_matrix[task_id][resource] = min_pher

    if not archive:
        return
    for ant in archive:
        pt = solution_to_point(ant)
        if pt is None:
            continue
        denom = (ant.fitness[0] + ant.fitness[1]) + 1e-6
        delta = Q / denom
        for task_id, resource in enumerate(ant.resource_assignment):
            if resource is None:
                continue
            if task_id in pheromone_matrix and resource in pheromone_matrix[task_id]:
                pheromone_matrix[task_id][resource] += delta
                if pheromone_matrix[task_id][resource] > max_pher:
                    pheromone_matrix[task_id][resource] = max_pher


def update_archive(new_ants: List[Ant], archive: List[Ant], config):
    all_solutions = list(new_ants) + list(archive)
    valid = []
    for s in all_solutions:
        pt = solution_to_point(s)
        if pt is not None:
            valid.append(s)
    if not valid:
        return archive

    unique = []
    seen = set()
    for s in valid:
        pt = solution_to_point(s)
        key = (round(pt[0], 8), round(pt[1], 8))
        if key not in seen:
            seen.add(key)
            unique.append(s)
    valid = unique

    fronts = fast_non_dominated_sort(valid)
    new_archive = []
    archive_size = int(getattr(config, 'archive_size', 50))
    for front in fronts:
        if len(new_archive) + len(front) <= archive_size:
            new_archive.extend(front)
        else:
            crowding = calculate_crowding_distance(front)
            paired = list(zip(front, crowding))
            paired_sorted = sorted(paired, key=lambda x: x[1], reverse=True)
            slots = archive_size - len(new_archive)
            chosen = [p[0] for p in paired_sorted[:max(0, slots)]]
            new_archive.extend(chosen)
            break
    return new_archive


def safe_extract_fitness_tuples(ants: List[Any]):
    out = []
    for a in ants:
        pt = solution_to_point(a)
        if pt is not None:
            out.append(pt)
    return out

# ------------------ 主函数：run_moaco ------------------
def run_moaco(config):
    print("开始 MOACO 优化...")

    num_ants = int(getattr(config, 'num_particles', 30))
    alpha = float(getattr(config, 'moaco_alpha', 1.0))
    beta = float(getattr(config, 'moaco_beta', 2.0))
    rho = float(getattr(config, 'moaco_rho', 0.1))
    Q = float(getattr(config, 'moaco_Q', 50.0))

    pheromone_matrix = initialize_pheromone_matrix(config, initial_value=1.0)
    heuristic_matrix = initialize_heuristic_matrix(config)

    archive_size_history = []
    all_fitness = []
    global_min_time_list = []
    global_min_cost_list = []
    hv_history = []
    igd_values = []

    global_best_time = float('inf')
    global_best_cost = float('inf')

    archive: List[Ant] = []

    try:
        reference_set = np.load('reference_set.npy')
    except Exception:
        reference_set = None

    ref_point = getattr(config, 'hv_ref_point', None)
    if ref_point is None:
        init_ants = [Ant(config) for _ in range(max(4, num_ants // 2))]
        for a in init_ants:
            a.construct_solution(pheromone_matrix, heuristic_matrix, alpha=alpha, beta=beta)
        fitness_init = safe_extract_fitness_tuples(init_ants)
        if fitness_init:
            max_time = max(t for t, _ in fitness_init)
            max_cost = max(c for _, c in fitness_init)
            ref_point = (max_time * 1.5 + 1.0, max_cost * 1.5 + 1.0)
        else:
            ref_point = (20000.0, 20000.0)
        config.hv_ref_point = ref_point
    else:
        ref_point = tuple(ref_point)

    print(f"使用的 HV 参考点 (ref_point) = {ref_point}")

    num_iterations = int(getattr(config, 'num_iterations', 100))
    for it in range(num_iterations):
        ants = [Ant(config) for _ in range(num_ants)]
        for ant in ants:
            ant.construct_solution(pheromone_matrix, heuristic_matrix, alpha=alpha, beta=beta)

        archive = update_archive(ants, archive, config)

        update_pheromone(pheromone_matrix, archive, rho=rho, Q=Q)

        valid_ants = [a for a in ants if solution_to_point(a) is not None]
        if valid_ants:
            current_fitness = [a.fitness for a in valid_ants]
            all_fitness.append(current_fitness)
            current_best_time = min(a.fitness[0] for a in valid_ants)
            current_best_cost = min(a.fitness[1] for a in valid_ants)
            global_best_time = min(global_best_time, current_best_time)
            global_best_cost = min(global_best_cost, current_best_cost)

        global_min_time_list.append(global_best_time)
        global_min_cost_list.append(global_best_cost)
        archive_size_history.append(len(archive))

        pareto_fitness = [solution_to_point(a) for a in archive if solution_to_point(a) is not None]
        if pareto_fitness:
            try:
                hv = calculate_hypervolume(pareto_fitness, ref_point)
            except Exception:
                hv = 0.0
            if reference_set is not None and len(reference_set) > 0:
                try:
                    igd = calculate_igd(pareto_fitness, reference_set)
                except Exception:
                    igd = float('nan')
            else:
                igd = float('nan')
        else:
            hv = 0.0
            igd = float('nan')

        hv_history.append(hv)
        igd_values.append(igd)

        if it % 50 == 0 or it == num_iterations - 1:
            print(f"Iter {it:3d}: Archive={len(archive):2d} | HV={hv:.2f} | IGD={igd if not np.isnan(igd) else float('nan'):.4f} | BestTime={global_best_time:.2f} | BestCost={global_best_cost:.2f}")

    pareto_fitness_final = [solution_to_point(a) for a in archive if solution_to_point(a) is not None]
    final_sp = calculate_spacing(pareto_fitness_final) if pareto_fitness_final else 0.0

    print(f"MOACO 完成！最终档案大小: {len(archive)}")

    return (
        archive,
        archive,
        all_fitness,
        global_min_time_list,
        global_min_cost_list,
        archive_size_history,
        {},
        hv_history,
        final_sp,
        igd_values
    )
