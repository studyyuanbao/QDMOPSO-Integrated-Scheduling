import math
import random
from typing import List, Tuple
import numpy as np
from particle import Particle
from decoder import decode
from objective import evaluate
from structure_learning import structure_learning_recombine
from utils import update_archive, dominates, select_leader_by_roulette, calculate_spacing, calculate_hypervolume, \
    calculate_igd


# def calculate_spacing(archive):
#     """Calculate spacing (SP) indicator for the Pareto front."""
#     if len(archive) < 2:
#         return 0.0
#     distances = []
#     for i in range(1, len(archive)):
#         distances.append(np.linalg.norm(np.array(archive[i].fitness) - np.array(archive[i-1].fitness)))
#     return np.mean(distances)
# def calculate_hypervolume(pareto_front: List[Tuple[float, float]], reference_point: Tuple[float, float]) -> float:
#     """
#     计算给定Pareto前沿的超体积（Hypervolume, HV）。
#     要求：
#     1. 参考点必须支配所有Pareto前沿点（即 reference_point >= 所有点的各维度值）。
#     2. Pareto前沿中不能包含无限值（inf）或NaN。
#     """
#     if not pareto_front:
#         return 0.0
#
#     # 过滤无效点（inf/nan）并去重，保持顺序
#     pareto_front = [p for i, p in enumerate(pareto_front) if all(not math.isinf(val) and not math.isnan(val) for val in p)]
#     pareto_front = list(dict.fromkeys(pareto_front))  # 保持顺序去重
#
#     if not pareto_front:
#         return 0.0
#
#     # 检查参考点是否支配所有点（避免参考点错误）
#     ref_time, ref_cost = reference_point
#     if any(time >= ref_time or cost >= ref_cost for time, cost in pareto_front):
#         raise ValueError("参考点必须严格支配所有Pareto前沿点！")
#
#     # 按时间升序排序（时间越小，越先支配）
#     sorted_front = sorted(pareto_front, key=lambda x: x[0])
#     volume = 0.0
#     ref_cost_current = ref_cost  # 动态更新的“屋顶”高度
#
#     # 计算每个点的超体积
#     for time, cost in sorted_front:
#         if time < ref_time and cost < ref_cost_current:
#             width = ref_time - time
#             height = ref_cost_current - cost
#             volume += width * height
#             ref_cost_current = cost  # 更新屋顶高度
#
#     return volume

def safe_random_resource(task_id, config):
    """为粒子位置中的资源分配部分生成一个安全的随机资源"""
    if task_id < len(config.num_suppliers):
        available_resources = config.num_suppliers[task_id]
        if available_resources:
            return random.choice(available_resources)
        else:
            raise ValueError(f"任务 {task_id} 没有可用的资源定义")
    else:
        raise ValueError(f"任务 {task_id} 超出num_suppliers定义范围")


def local_search_elites(convergence_archive, diversity_archive, config, num_elites_to_search=5):
    """对双档案中的精英粒子进行局部搜索"""
    new_particles = []
    # 合并两个档案并选择精英
    combined_elites = sorted(
        convergence_archive + diversity_archive,
        key=lambda p: p.fitness[0] + p.fitness[1]
    )[:num_elites_to_search]

    for elite in combined_elites:
        for _ in range(2):  # 每个精英粒子生成2个新解
            new_pos = elite.position[:]

            # 资源分配扰动
            for i in range(config.num_tasks):
                if random.random() < 0.2:  # 20%概率扰动资源分配
                    new_pos[i] = safe_random_resource(i, config)

            # 优先级扰动
            for i in range(config.num_tasks, 2 * config.num_tasks):
                if random.random() < 0.2:  # 20%概率扰动优先级
                    new_pos[i] = max(0, min(1, new_pos[i] + random.uniform(-0.1, 0.1)))

            # 创建新粒子
            temp_particle = Particle(config)
            temp_particle.position = new_pos
            temp_particle.resource_assignment = new_pos[:config.num_tasks]
            temp_particle.priority_values = new_pos[config.num_tasks:]

            # 解码和评估
            temp_particle.schedule = decode(temp_particle, config)
            if temp_particle.schedule:
                temp_particle.fitness = evaluate(temp_particle.schedule, config)
                new_particles.append(temp_particle)

    return new_particles


def sbx_crossover(parent1, parent2, config, eta_c=20):
    child1 = np.zeros_like(parent1, dtype=float)
    child2 = np.zeros_like(parent2, dtype=float)

    for i in range(len(parent1)):
        if random.random() <= getattr(config, 'crossover_prob', 0.9):
            if i < config.num_tasks:
                x1 = min(parent1[i], parent2[i])
                x2 = max(parent1[i], parent2[i])
                if x1 == x2:
                    child1[i] = child2[i] = x1
                    continue
                # protection from division by zero
                denom = (x2 - x1) if abs(x2 - x1) > 1e-12 else 1e-12
                beta = 1.0 + (2.0 * (x1 - getattr(config, 'resource_min', 0)) / denom)
                beta = max(beta, 1e-9)
                alpha = 2.0 - beta ** -(eta_c + 1)
                u = random.random()
                if u <= 1.0 / alpha:
                    base = u * alpha
                else:
                    base = 1.0 / (2.0 - u * alpha)
                base = max(base, 1e-9)
                beta_q = base ** (1.0 / (eta_c + 1))
                c1 = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))
                c2 = 0.5 * ((x1 + x2) + beta_q * (x2 - x1))
                child1[i] = int(np.clip(round(c1), getattr(config, 'resource_min', 0), getattr(config, 'resource_max', 1)))
                child2[i] = int(np.clip(round(c2), getattr(config, 'resource_min', 0), getattr(config, 'resource_max', 1)))
            else:
                x1, x2 = parent1[i], parent2[i]
                if abs(x1 - x2) > 1e-14:
                    if x1 > x2:
                        x1, x2 = x2, x1
                    denom = (x2 - x1) if abs(x2 - x1) > 1e-14 else 1e-14
                    beta = 1.0 + (2.0 * (x1) / denom)
                    beta = max(beta, 1e-9)
                    alpha = 2.0 - beta ** -(eta_c + 1)
                    u = random.random()
                    if u <= 1.0 / alpha:
                        base = u * alpha
                    else:
                        base = 1.0 / (2.0 - u * alpha)
                    base = max(base, 1e-9)
                    beta_q = base ** (1.0 / (eta_c + 1))
                    c1 = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))
                    c2 = 0.5 * ((x1 + x2) + beta_q * (x2 - x1))
                else:
                    c1 = x1
                    c2 = x2
                child1[i] = np.clip(c1, 0, 1)
                child2[i] = np.clip(c2, 0, 1)
        else:
            child1[i] = parent1[i]
            child2[i] = parent2[i]

    return child1, child2


def polynomial_mutation(individual, config, eta_m=20):
    mutated = np.copy(individual)
    for i in range(len(individual)):
        if random.random() < getattr(config, 'mutation_prob', 0.1):
            if i < config.num_tasks:
                x = individual[i]
                r = random.random()
                # small integer perturbation - keep it simple and robust
                if r < 0.5:
                    x = x - 1
                else:
                    x = x + 1
                mutated[i] = int(np.clip(round(x), getattr(config, 'resource_min', 0), getattr(config, 'resource_max', 1)))
            else:
                x = individual[i]
                r = random.random()
                delta = (random.random() - 0.5) * 0.2
                x = x + delta
                mutated[i] = np.clip(x, 0, 1)
    return mutated


def genetic_operation(parents, config):
    offspring = []
    if len(parents) < 2:
        return offspring
    random.shuffle(parents)
    num_parents = len(parents) if len(parents) % 2 == 0 else len(parents) - 1
    for i in range(0, num_parents, 2):
        child1_pos, child2_pos = sbx_crossover(parents[i].position, parents[i + 1].position, config)
        child1 = Particle(config)
        child1.position = child1_pos
        child2 = Particle(config)
        child2.position = child2_pos
        offspring.extend([child1, child2])
    for child in offspring:
        child.position = polynomial_mutation(child.position, config)
        child.resource_assignment = child.position[:config.num_tasks]
        child.priority_values = child.position[config.num_tasks:]
    return offspring


def run_mopso(config):
    """主MOPSO算法，采用双档案策略"""
    # 初始化粒子群和双档案
    archive_size_history = []
    all_fitness = []
    global_min_time_list = []
    global_min_cost_list = []

    global_best_time = float('inf')
    global_best_cost = float('inf')
    no_improve_count = 0
    hv_history = []  # 每一代的超体积记录
    final_sp = []
    igd_values = []
    current_hv =[]

    # 初始化种群
    swarm = [Particle(config, feasible_init=True) for _ in range(config.num_particles)]
    convergence_archive, diversity_archive = [], []

    # 初始化粒子
    for particle in swarm:
        particle.schedule = decode(particle, config)
        if particle.schedule:
            particle.fitness = evaluate(particle.schedule, config)
            particle.update_pbest()
            if not np.isnan(particle.fitness[0]):
                global_best_time = min(global_best_time, particle.fitness[0])
            if not np.isnan(particle.fitness[1]):
                global_best_cost = min(global_best_cost, particle.fitness[1])

    valid_particles = [p for p in swarm if p.schedule]
    if valid_particles:
        convergence_archive, diversity_archive = update_archive(valid_particles, [], [], config)
    # 记录初始档案大小
    archive_size_history.append(len(convergence_archive) + len(diversity_archive))

    # 初始HV
    if diversity_archive:
        init_pareto = [p.fitness for p in diversity_archive if
                       all(val is not None and val != float('inf') for val in p.fitness)]
        if init_pareto:
            current_hv = calculate_hypervolume(init_pareto, (120, 600))
    hv_history.append(current_hv)

    print("开始 MOPSO 优化...")
    for t in range(config.num_iterations):
        current_gen_fitness = []
        valid_particles = []
        progress = t / config.num_iterations

        # 动态权重（保持原有的动态权重计算）
        dyn_weight_time = 0.5 + 0.6 * (1 - math.exp(-8 * progress))
        dyn_weight_cost = 1.5 - 0.6 * (1 - math.exp(-8 * progress))

        # 使用固定参数（与RL版本中的action=0对应）
        w, c1, c2 = 0.7, 1.0, 1.5

        # 选择 leader (保持原有的leader选择逻辑)
        candidates = list(convergence_archive) + list(diversity_archive)

        # leader softmax selection (scores computed from normalized time/cost)
        eps = 1e-9
        candidate_times = np.array([p.fitness[0] for p in candidates])
        candidate_costs = np.array([p.fitness[1] for p in candidates])

        time_mean = np.mean(candidate_times)
        time_std = np.std(candidate_times)
        cost_mean = np.mean(candidate_costs)
        cost_std = np.std(candidate_costs)

        if time_std < 1e-9:
            norm_times = np.zeros_like(candidate_times)
        else:
            norm_times = (candidate_times - time_mean) / time_std
        norm_times = (norm_times + 3) / 6

        if cost_std < 1e-9:
            norm_costs = np.zeros_like(candidate_costs)
        else:
            norm_costs = (candidate_costs - cost_mean) / cost_std
        norm_costs = (norm_costs + 3) / 6

        leader_weight_time = 0.6
        leader_weight_cost = 0.4
        combined_values = leader_weight_time * norm_times + leader_weight_cost * norm_costs
        scores = 1.0 / (combined_values + eps)
        scaled_scores = scores - np.max(scores)
        exp_scores = np.exp(scaled_scores)
        probs = exp_scores / np.sum(exp_scores)
        leader = np.random.choice(candidates, p=probs)
        # 更新粒子群
        V_MAX = 3.0
        for particle in swarm:
            position = particle.position
            velocity = particle.velocity
            pbest_position = particle.pbest_position
            leader_position = leader.position

            for i in range(len(position)):
                r1, r2 = random.random(), random.random()
                cognitive = c1 * r1 * (pbest_position[i] - position[i])
                social = c2 * r2 * (leader_position[i] - position[i])
                velocity[i] = w * velocity[i] + cognitive + social

                if abs(velocity[i]) > V_MAX:
                    velocity[i] = np.sign(velocity[i]) * (
                            V_MAX + np.log1p(abs(velocity[i]) - V_MAX)
                    )

                if i < config.num_tasks:
                    available = config.num_suppliers[i]
                    if not available:
                        continue

                    def velocity_to_prob(v, k=1.5):
                        return 1 / (1 + np.exp(-k * v))

                    change_prob = velocity_to_prob(velocity[i])

                    if random.random() < change_prob:
                        current_resource = position[i]

                        if velocity[i] > 0:
                            if progress < 0.4:
                                time_weight, cost_weight = 0.5, 0.5
                            else:
                                if global_best_time < float('inf') and global_best_cost < float('inf'):
                                    time_ratio = global_best_time / (global_best_time + global_best_cost)
                                    cost_ratio = global_best_cost / (global_best_time + global_best_cost)
                                    time_weight, cost_weight = cost_ratio, time_ratio
                                else:
                                    time_weight, cost_weight = 0.5, 0.5

                                best_score = float('inf')
                                best_resource = current_resource
                                for res in available:
                                    if res != current_resource:
                                        task_info = config.machine_task_map.get((res, i), {})
                                        time_val = task_info.get('time', float('inf'))
                                        cost_val = task_info.get('cost', float('inf'))
                                        score = time_weight * time_val + cost_weight * cost_val
                                        if score < best_score:
                                            best_score = score
                                            best_resource = res
                                position[i] = best_resource

                        else:
                            other_resources = [r for r in available if r != current_resource]
                            if other_resources:
                                position[i] = random.choice(other_resources)

                else:
                    V_MAX_cont = max(0.5, 2.0 * (1 - progress))
                    velocity[i] = np.clip(velocity[i], -V_MAX_cont, V_MAX_cont)
                    position[i] = np.clip(position[i] + velocity[i], 0, 1)

            particle.position = position
            particle.velocity = velocity

            particle.resource_assignment = position[:config.num_tasks]
            particle.priority_values = position[config.num_tasks:]
            particle.schedule = decode(particle, config)

            if particle.schedule:
                particle.fitness = evaluate(particle.schedule, config)

                if dominates(particle.fitness, particle.pbest_fitness):
                    particle.update_pbest()
                else:
                    current_weighted = dyn_weight_time * particle.fitness[0] + dyn_weight_cost * particle.fitness[1]
                    pbest_weighted = dyn_weight_time * particle.pbest_fitness[0] + dyn_weight_cost * particle.pbest_fitness[1]
                    if current_weighted < pbest_weighted:
                        particle.update_pbest()

                current_gen_fitness.append(particle.fitness)

                if not np.isnan(particle.fitness[0]):
                    global_best_time = min(global_best_time, particle.fitness[0])
                if not np.isnan(particle.fitness[1]):
                    global_best_cost = min(global_best_cost, particle.fitness[1])

        # 更新档案
        valid_particles = [p for p in swarm if p.schedule]
        convergence_archive, diversity_archive = update_archive(valid_particles, convergence_archive,
                                                                diversity_archive, config)

        # 记录指标
        all_fitness.append(current_gen_fitness)
        global_min_time_list.append(global_best_time)
        global_min_cost_list.append(global_best_cost)

        # 在此处统一记录档案大小（每代一次）
        archive_size_history.append(len(convergence_archive) + len(diversity_archive))

        # 遗传操作（降低频率）
        if t % max(1, config.genetic_interval * 2) == 0 and t > 0 and (convergence_archive or diversity_archive):
            genetic_parents = sorted(convergence_archive + diversity_archive,
                                     key=lambda p: p.fitness[0] + p.fitness[1])[:config.num_genetic_parents]
            genetic_offspring = genetic_operation(genetic_parents, config)
            valid_offspring = []
            for child in genetic_offspring:
                child.schedule = decode(child, config)
                if child.schedule:
                    child.fitness = evaluate(child.schedule, config)
                    valid_offspring.append(child)
            if valid_offspring:
                convergence_archive, diversity_archive = update_archive(valid_offspring, convergence_archive,
                                                                        diversity_archive, config)

        # 局部搜索（降低频率）
        # if t % 40 == 0 and t > 0 and (convergence_archive or diversity_archive):
        #     new_particles = local_search_elites(convergence_archive, diversity_archive, config)
        #     if new_particles:
        #         convergence_archive, diversity_archive = update_archive(new_particles, convergence_archive,
        #                                                                 diversity_archive, config)

        current_pareto = [p.fitness for p in diversity_archive if
                          all(val is not None and val != float('inf') for val in p.fitness)]
        if current_pareto:
            current_hv = calculate_hypervolume(current_pareto, (120, 600))

        # hv_max = max(hv_history) if hv_history else current_hv
        hv_history.append(current_hv)
        
        # 转换IGD和Spacing的输入格式
        diversity_fitness = [p.fitness for p in diversity_archive if 
                           all(val is not None and val != float('inf') for val in p.fitness)]
        reference_fitness = [tuple(r) for r in reference_set] if reference_set is not None else None
        
        final_sp = calculate_spacing(diversity_fitness) if diversity_fitness else 0.0
        
        # reference set for IGD
        try:
            reference_set = np.load('reference_set.npy')
        except Exception:
            reference_set = None
            print("警告：无法加载参考集，IGD计算将受影响")
        
        if reference_fitness is not None:
            igd_values.append(calculate_igd(diversity_fitness, reference_fitness))
        else:
            igd_values.append(float('nan'))

        # 日志输出（降低频率）
        if t % 50 == 0 or t == config.num_iterations - 1:
            print(
                f"Iter {t:3d}: CA={len(convergence_archive):2d} DA={len(diversity_archive):2d} | "
                f"HV={current_hv:.2f} | Best Time={global_best_time:.1f} | "
                f"Best Cost={global_best_cost:.1f}")

    # 返回结果
    return (
        convergence_archive,
        diversity_archive,
        all_fitness,
        global_min_time_list,
        global_min_cost_list,
        archive_size_history,
        {},                # RL 中是 Q 表，这里用空字典占位
        hv_history,        # 原始版本中也计算超体积或者返回0列表
        final_sp,
        igd_values
    )
