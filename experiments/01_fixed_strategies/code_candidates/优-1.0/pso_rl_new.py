import math
import random
from typing import List, Optional
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from particle import Particle
from decoder import decode
from objective import evaluate
from rl_agent_new import QLearningAgent
from utils import update_archive,  dominates, calculate_spacing, calculate_hypervolume, \
    calculate_igd, archive_improved, nondominated


def safe_extract_fitness_tuples(particles):
    """
    安全地从 Particle 列表中提取适应度元组列表
    确保所有 fitness 值都是有效的数值类型
    
    Args:
        particles: Particle 对象列表
        
    Returns:
        list: 有效的适应度元组列表 [(time, cost), ...]
    """
    fitness_tuples = []
    for p in particles:
        if hasattr(p, 'fitness') and isinstance(p.fitness, (list, tuple)) and len(p.fitness) == 2:
            try:
                # 确保 fitness 值是数值类型
                time_val = float(p.fitness[0])
                cost_val = float(p.fitness[1])
                # 检查是否为有效数值（非 NaN 和无穷大）
                if not (np.isnan(time_val) or np.isnan(cost_val) or 
                       np.isinf(time_val) or np.isinf(cost_val)):
                    fitness_tuples.append((time_val, cost_val))
            except (ValueError, TypeError):
                # 跳过无效的 fitness 值
                continue
    return fitness_tuples


def safe_calculate_hypervolume(particles, ref_point):
    """
    安全地计算超体积指标
    
    Args:
        particles: Particle 对象列表
        ref_point: 参考点
        
    Returns:
        float: 超体积值
    """
    fitness_tuples = safe_extract_fitness_tuples(particles)
    return calculate_hypervolume(fitness_tuples, ref_point) if fitness_tuples else 0.0


def safe_calculate_igd(particles, reference_set):
    """
    安全地计算 IGD 指标
    
    Args:
        particles: Particle 对象列表
        reference_set: 参考集
        
    Returns:
        float: IGD 值
    """
    if reference_set is None or len(reference_set) == 0:
        return float('nan')
    fitness_tuples = safe_extract_fitness_tuples(particles)
    if not fitness_tuples:
        return float('nan')
    return calculate_igd(fitness_tuples, reference_set)


def safe_calculate_spacing(particles):
    """
    安全地计算间距指标
    
    Args:
        particles: Particle 对象列表
        
    Returns:
        float: 间距值
    """
    if not particles:
        return 0.0
    fitness_tuples = safe_extract_fitness_tuples(particles)
    if not fitness_tuples:
        return 0.0
    return calculate_spacing(fitness_tuples)


def get_parameters_by_action(action, progress):
    """更温和的参数策略"""
    if action == 0:
        return 0.7, 1.0, 1.5, 0.0, False, 'normal'
    elif action == 1:
        w_max, w_min = 0.9, 0.4
        w = w_max - (w_max - w_min) * progress
        return w, 1.5, 1.5, 0.0, False, 'normal'
    elif action == 2:
        if progress < 0.5:
            return 0.9, 1.8, 0.8, 0.0, False, 'normal'  # 更温和
        else:
            return 0.5, 0.8, 1.8, 0.0, False, 'normal'
    elif action == 3:
        w = 0.8 - 0.2 * progress  # 更温和的探索
        return w, 1.2, 1.2, 0.08, False, 'low'  # 降低扰动概率
    elif action == 4:
        w = 0.6 - 0.2 * progress
        return max(0.3, w), 1.6, 1.6, 0.0, True, 'normal'
    elif action == 5:
        w = 0.7
        return w, 1.4, 1.6, 0.03, False, 'high'  # 降低扰动
    elif action == 6:
        w = 0.8
        return w, 1.0, 1.0, 0.15, False, 'low'   # 大幅降低扰动
    else:
        return 0.7, 1.0, 1.5, 0.0, False, 'normal'


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


# local_search_elites, sbx_crossover, polynomial_mutation, genetic_operation remain mostly unchanged
# (kept for readability). If you want I can further simplify them, but they were OK semantically.
# --- retained functions (no behavioural change) ---

def local_search_elites(convergence_archive, diversity_archive, config, num_elites_to_search=5):
    new_particles = []
    combined_elites = sorted(
        convergence_archive + diversity_archive,
        key=lambda p: p.fitness[0] + p.fitness[1]
    )[:num_elites_to_search]

    for elite in combined_elites:
        for _ in range(2):
            new_pos = elite.position[:]
            for i in range(config.num_tasks):
                if random.random() < 0.2:
                    new_pos[i] = safe_random_resource(i, config)
            for i in range(config.num_tasks, 2 * config.num_tasks):
                if random.random() < 0.2:
                    new_pos[i] = max(0, min(1, new_pos[i] + random.uniform(-0.1, 0.1)))
            temp_particle = Particle(config)
            temp_particle.position = new_pos
            temp_particle.resource_assignment = new_pos[:config.num_tasks]
            temp_particle.priority_values = new_pos[config.num_tasks:]
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
                child1[i] = int(
                    np.clip(round(c1), getattr(config, 'resource_min', 0), getattr(config, 'resource_max', 1)))
                child2[i] = int(
                    np.clip(round(c2), getattr(config, 'resource_min', 0), getattr(config, 'resource_max', 1)))
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
                mutated[i] = int(
                    np.clip(round(x), getattr(config, 'resource_min', 0), getattr(config, 'resource_max', 1)))
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


def select_leader_softmax(convergence_archive, diversity_archive, temperature=1.0):
    candidates = convergence_archive + diversity_archive
    if not candidates:
        return None
    scores = np.array([1.0 / (p.fitness[0] + p.fitness[1] + 1e-6) for p in candidates])
    exp_scores = np.exp(scores / temperature)
    probs = exp_scores / np.sum(exp_scores)
    return np.random.choice(candidates, p=probs)


def run_mopso(config, total_iterations=100000, save_q_table_path: Optional[str] = "2.0-q_table1.pkl",
              load_q_table_path: Optional[str] = None, exploit_only: bool = False, fixed_action: Optional[int] = None):
    # 存储所有结果
    # 先不设置ref_point，等初始化swarm后再计算

    # 1) 奖励相关数据
    cumulative_rewards_history = []
    avg_rewards_history = []

    # 2) Q-learning 相关数据
    q_mean_history = []
    q_max_history = []
    q_min_history = []
    q_std_history = []
    epsilon_history = []
    state_count_history = []

    # 3) 性能指标
    best_fitness_history = []
    all_generational_best_history = []

    # 一次性初始化所有组件
    num_tasks = config.num_tasks
    num_particles = config.num_particles

    # 初始化种群
    swarm = [Particle(config, feasible_init=True) for _ in range(num_particles)]
    convergence_archive, diversity_archive = [], []

    # 初始化agent
    # agent = QLearningAgent(
    #     actions=[0, 1, 2, 3, 4, 5, 6],  # 动作集
    #     alpha=0.2,
    #     gamma=0.8,
    #     epsilon=0.9,
    #     epsilon_min=0.01,
    #     epsilon_decay=0.99,
    #     total_iterations=total_iterations,
    # )
    agent = QLearningAgent(
        actions=[0, 1, 2, 3, 4, 5, 6],
        alpha=0.25,  # 略微提高学习率
        gamma=0.98,  # 强烈建议提高，强调长期奖励
        epsilon=0.5,  # 降低初始探索率，允许更早开始利用
        epsilon_min=0.01,
        epsilon_decay=0.99,
        total_iterations=total_iterations,
    )
    # 如果提供了预训练Q表，则加载
    if load_q_table_path and os.path.exists(load_q_table_path):
        try:
            with open(load_q_table_path, "rb") as f:
                loaded_q = pickle.load(f)
            agent.q_table.clear()
            agent.q_table.update(loaded_q)
            if exploit_only:
                agent.epsilon = 0.0
                agent.epsilon_min = 0.0
                agent.epsilon_decay = 1.0
            print(f"已加载预训练 Q 表：{load_q_table_path}，exploit_only={exploit_only}")
        except Exception as e:
            print(f"加载 Q 表失败（{load_q_table_path}）：{e}")

    # reference set for IGD
    try:
        reference_set = np.load('reference_set.npy')
    except Exception:
        reference_set = None
        print("警告：无法加载参考集，IGD计算将受影响")

    # 性能指标记录
    archive_size_history, all_fitness = [], []
    global_min_time_list, global_min_cost_list = [], []
    hv_history_abs, igd_values = [], []  # 重命名为hv_history_abs表示绝对值
    final_sp = None

    prev_hv, current_hv = 0.0, 0.0
    global_best_time, global_best_cost = float('inf'), float('inf')
    rewards_history = []
    q_values_per_iteration = []

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

    # 在初始化 swarm、档案之后计算参考点
    initial_ref = getattr(config, 'hv_ref_point', None)
    if initial_ref is not None:
        ref_point = initial_ref
    else:
        # 使用初始有效粒子的 fitness 来估计参考点并加 margin
        valid_init = [p for p in swarm if hasattr(p,'fitness') and p.fitness is not None]
        fitness_init = safe_extract_fitness_tuples(valid_init)
        if fitness_init:
            max_time = max(t for t,c in fitness_init)
            max_cost = max(c for t,c in fitness_init)
            ref_point = (max_time * 1.5 + 1.0, max_cost * 1.5 + 1.0)  # margin 可调
        else:
            # 最后兜底（确保比任何合理值都大）
            ref_point = (20000.0, 20000.0)
    # 建议把 ref_point 保存回 config 以便诊断
    config.hv_ref_point = ref_point
    print(f"使用的 HV 参考点 (ref_point) = {ref_point}")
    
    # 记录初始档案大小
    archive_size_history.append(len(convergence_archive) + len(diversity_archive))

    # 初始HV（仅用于建立基线，不做下降比较；使用整个非支配集）
    initial_archive_union = nondominated(list(convergence_archive) + list(diversity_archive))
    if initial_archive_union:
        current_hv = safe_calculate_hypervolume(initial_archive_union, ref_point)
    else:
        current_hv = 0.0
    hv_history_abs.append(current_hv)

    print("开始 MOPSO-RL 优化...")

    for t in range(total_iterations):
        current_gen_fitness = []
        progress = t / max(1, total_iterations)
        # 在训练循环中
        current_iteration = t
        train_progress = current_iteration / total_iterations
        # 计算下一个状态的进度（处理边界情况）
        if current_iteration + 1 < total_iterations:
            next_progress = (current_iteration + 1) / total_iterations
        else:
            next_progress = 1.0  # 最后一次迭代

        # 动态权重（保持原有的动态权重计算）
        dyn_weight_cost = 0.8 + 0.4 * (1 - math.exp(-6 * progress))  # 从0.8→1.2
        dyn_weight_time = 1.2 - 0.4 * (1 - math.exp(-6 * progress))  # 从1.2→0.8
        # dyn_weight_time = 0.8 + 0.4 * (1 - math.exp(-6 * progress))  # 从0.8→1.2
        # dyn_weight_cost = 1.2 - 0.4 * (1 - math.exp(-6 * progress))  # 从1.2→0.8

        # 使用新状态（基于指标趋势）
        archive_union = nondominated(list(convergence_archive) + list(diversity_archive))
        # 使用整个非支配集计算 HV 状态量，直接传原始HV，由agent内部缩放
        hv_for_state = safe_calculate_hypervolume(archive_union, ref_point)
        sp_for_state = safe_calculate_spacing(archive_union)
        
        state = agent.get_state_indicators(hv_for_state, sp_for_state, train_progress)
        # 支持固定动作模式，用于对比实验
        if fixed_action is None:
            action = agent.choose_action(state)
        else:
            action = fixed_action

        # 根据动作获取PSO参数
        w, c1, c2, diversity_kick_prob, local_search_boost, genetic_freq = get_parameters_by_action(action, progress)

        # 选择 leader (保持原有的leader选择逻辑)
        candidates = archive_union

        if not candidates:
            candidates = [p for p in swarm if p.schedule]
        if not candidates:
            # 使用指标型状态，避免空档案时报错
            hv_tmp = 0.0
            sp_tmp = 0.0
            # 直接传原始HV
            next_state = agent.get_state_indicators(hv_tmp, sp_tmp, progress)
            agent.update(state, action, 0.0, next_state)
            archive_size_history.append(len(convergence_archive) + len(diversity_archive))
            hv_history_abs.append(current_hv)
            all_fitness.append([])
            continue

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

        leader_weight_time = 0.5
        leader_weight_cost = 0.5
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

                if i < num_tasks:
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

            particle.resource_assignment = position[:num_tasks]
            particle.priority_values = position[num_tasks:]
            particle.schedule = decode(particle, config)
            if particle.schedule:
                particle.fitness = evaluate(particle.schedule, config)
                current_gen_fitness.append(particle.fitness)

                if dominates(particle.fitness, particle.pbest_fitness):
                    particle.update_pbest()
                else:
                    current_weighted = dyn_weight_time * particle.fitness[0] + dyn_weight_cost * particle.fitness[1]
                    pbest_weighted = dyn_weight_time * particle.pbest_fitness[0] + dyn_weight_cost * \
                                     particle.pbest_fitness[1]
                    if current_weighted < pbest_weighted:
                        particle.update_pbest()

                if not np.isnan(particle.fitness[0]):
                    global_best_time = min(global_best_time, particle.fitness[0])
                if not np.isnan(particle.fitness[1]):
                    global_best_cost = min(global_best_cost, particle.fitness[1])

        # 更新档案
        valid_particles = [p for p in swarm if p.schedule]
        prev_archive_union = nondominated(list(convergence_archive) + list(diversity_archive))
        
        # 使用HV贡献度优先的档案更新策略
        convergence_archive, diversity_archive = update_archive(
            valid_particles, convergence_archive, diversity_archive, config)
        
        new_archive_union = nondominated(list(convergence_archive) + list(diversity_archive))
        
        # 检查档案是否改进了
        archive_improvement = archive_improved(prev_archive_union, new_archive_union)

        # 多样性干预：随机扰动部分粒子
        if diversity_kick_prob > 0 and random.random() < diversity_kick_prob:
            kick_ratio = 0.2
            num_kick = max(1, int(len(swarm) * kick_ratio))
            kick_indices = np.random.choice(len(swarm), size=num_kick, replace=False)
            for idx in kick_indices:
                p = swarm[idx]
                # 对资源分配部分随机打散
                for i in range(num_tasks):
                    if random.random() < 0.3:
                        available = config.num_suppliers[i]
                        if available:
                            p.position[i] = random.choice(available)
                # 对连续部分加入扰动
                for i in range(num_tasks, len(p.position)):
                    if random.random() < 0.5:
                        p.position[i] = np.clip(p.position[i] + np.random.uniform(-0.3, 0.3), 0, 1)
                p.resource_assignment = p.position[:num_tasks]
                p.priority_values = p.position[num_tasks:]
                p.schedule = decode(p, config)
                if p.schedule:
                    p.fitness = evaluate(p.schedule, config)

        # 遗传操作（频率可调）
        if genetic_freq == 'high':
            genetic_interval = max(1, config.genetic_interval)
        elif genetic_freq == 'low':
            genetic_interval = max(1, config.genetic_interval * 3)
        else:
            genetic_interval = max(1, config.genetic_interval * 2)
        if t % genetic_interval == 0 and t > 0 and (convergence_archive or diversity_archive):
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
        # 局部搜索（可增强）
        ls_interval = 40 if not local_search_boost else 20
        if t % ls_interval == 0 and t > 0 and (convergence_archive or diversity_archive):
            new_particles = local_search_elites(convergence_archive, diversity_archive, config,
                                                num_elites_to_search=8 if local_search_boost else 5)
            if new_particles:
                convergence_archive, diversity_archive = update_archive(new_particles, convergence_archive,
                                                                        diversity_archive, config)

        # 计算HV并保留旧的 prev_hv（修复：不要提前覆盖 prev_hv）；使用整个非支配集
        archive_union = nondominated(list(convergence_archive) + list(diversity_archive))
        current_hv = safe_calculate_hypervolume( archive_union, ref_point)
        # 最终检查档案是否改进了
        archive_improvement = archive_improved(prev_archive_union, new_archive_union)

        # 在第一次有有效档案后再初始化prev_hv
        if t == 0 and archive_union:
            prev_hv = current_hv
            continue  # 跳过第一代的奖励计算

        # 保存旧 prev_hv 用于奖励计算
        prev_hv_old = prev_hv  # <--- 关键：保留上一代的值

        # 组合奖励（HV/IGD；Spacing仅记录）
        hv_max = max(hv_history_abs) if hv_history_abs else current_hv
        sp_now = safe_calculate_spacing(archive_union)
        igd_now = safe_calculate_igd(archive_union, reference_set)
        igd_prev = igd_values[-1] if igd_values else np.nan
        sp_prev = final_sp if final_sp is not None else 0.0

        if t == 0:
            reward = 0.0
        else:
            # 使用原始HV进行奖励计算 + Spacing
            hv_prev_raw = getattr(agent, 'prev_hv_raw', prev_hv_old)
            reward = agent.calculate_composite_reward(hv_prev_raw, current_hv,
                                                      sp_prev, sp_now,
                                                      action,
                                                      w_hv=0.8, w_sp=0.2)

        rewards_history.append(reward)
        agent.cumulative_reward += reward

        # 之后才更新 prev_hv（将上一代替为当前）
        prev_hv = current_hv
        # 同步更新 agent 的上一代原始HV
        agent.prev_hv_raw = current_hv

        # 记录当前迭代的 Q 值（保存被访问状态的 Q 列表）；使用整个非支配集的 HV
        archive_union = nondominated(list(convergence_archive) + list(diversity_archive))
        hv_next = safe_calculate_hypervolume(archive_union, ref_point)
        sp_next = safe_calculate_spacing(archive_union)
        
        # 直接传入原始HV，由agent内部进行对数缩放
        next_state = agent.get_state_indicators(hv_next, sp_next, next_progress)
        q_values = agent.q_table.get(next_state, [0.0] * len(agent.actions))
        q_values_per_iteration.append(q_values)

        # 更新 RL (仅在非固定动作模式下)
        if fixed_action is None:
            agent.update(state, action, reward, next_state)

        # 记录指标
        hv_history_abs.append(current_hv)
        final_sp = sp_now
        if reference_set is not None:
            igd_values.append(igd_now)
        else:
            igd_values.append(float('nan'))

        all_fitness.append(current_gen_fitness)
        global_min_time_list.append(global_best_time)
        global_min_cost_list.append(global_best_cost)

        # 在此处统一记录档案大小（每代一次）
        archive_size_history.append(len(convergence_archive) + len(diversity_archive))

        all_generational_best_history.append((t, global_best_time, global_best_cost))

        # 日志输出（降低频率）
        if t % 50 == 0 or t == total_iterations - 1:
            if fixed_action is not None:
                # 固定动作模式：简洁输出
                action_names = [
                    "0:固定参数", "1:线性递减w", "2:分段策略", "3:强探索+扰动",
                    "4:强利用+局部搜索", "5:遗传高频", "6:强打散"
                ]
                action_name = action_names[fixed_action] if fixed_action < len(action_names) else f"动作{fixed_action}"
                print(
                    f"Iter {t:3d}: Action={action_name} | CA={len(convergence_archive):2d} DA={len(diversity_archive):2d} | "
                    f"HV={current_hv:.2f}| IGD={igd_now:.2f} | SP={sp_now:.3f} | Best Time={global_best_time:.1f} | "
                    f"Best Cost={global_best_cost:.1f} | Archive_Improved={archive_improvement} ")
            else:
                # RL模式：完整Q-learning信息
                q_vals = agent.q_table.get(next_state, None)
                if q_vals:
                    max_q = max(q_vals)
                    best_action_idx = np.argmax(q_vals)
                    action_names = [
                        "固定参数", "线性递减w", "分段策略", "强探索+扰动",
                        "强利用+局部搜索", "遗传高频", "强打散"
                    ]
                    best_action_name = action_names[best_action_idx] if best_action_idx < len(action_names) else f"动作{best_action_idx}"
                    q_info = f"MaxQ={max_q:.3f}({best_action_name})"
                else:
                    q_info = "MaxQ=N/A"

                print(
                    f"Iter {t:3d}: States={len(agent.q_table)} | CA={len(convergence_archive):2d} DA={len(diversity_archive):2d} | "
                    f"HV={current_hv:.2f}| IGD={igd_now:.2f} | SP={sp_now:.3f} | ε={agent.epsilon:.2f} | Best Time={global_best_time:.1f} | "
                    f"Best Cost={global_best_cost:.1f} | CumulReward={agent.cumulative_reward:.3f} | Archive_Improved={archive_improvement}  | {q_info}")

        # 每100次迭代衰减epsilon (仅在RL模式)
        if t % 100 == 0 and t > 0 and fixed_action is None:
            agent.decay_epsilon()
            print(f"ε衰减到: {agent.epsilon:.3f}")

        # 每1000次迭代记录统计信息
        if t % 1000 == 0 and t > 0:
            # Q 值统计
            if q_values_per_iteration:
                flattened = [v for qs in q_values_per_iteration[-1000:] for v in qs]  # 最近1000次迭代
            else:
                flattened = agent.flatten_qtable_values()

            if flattened:
                q_mean_history.append(np.mean(flattened))
                q_max_history.append(np.max(flattened))
                q_min_history.append(np.min(flattened))
                q_std_history.append(np.std(flattened))
            else:
                q_mean_history.append(0.0)
                q_max_history.append(0.0)
                q_min_history.append(0.0)
                q_std_history.append(0.0)

            epsilon_history.append(agent.epsilon)
            state_count_history.append(len(agent.q_table))

            # 记录奖励统计
            recent_rewards = rewards_history[-1000:] if len(rewards_history) >= 1000 else rewards_history
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            avg_rewards_history.append(avg_reward)
            cumulative_rewards_history.append(agent.cumulative_reward)

            # 记录最优适应度
            best_fitness_history.append((global_best_time, global_best_cost))

            print(f"Iter {t}: 累计奖励: {agent.cumulative_reward:.3f}, 平均奖励: {avg_reward:.3f}")
            print(
                f"Q统计: 均值={q_mean_history[-1]:.3f}, 最大={q_max_history[-1]:.3f}, 最小={q_min_history[-1]:.3f},标准差={q_std_history[-1]:.3f}")
            print(f"最优适应度: Time={global_best_time:.1f}, Cost={global_best_cost:.1f}")
    # 绘图部分：确保 x 轴长度一致
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    # plt.subplots_adjust(hspace=0.3, wspace=0.25)
    #
    # def smooth_data(y: List, box_pts: int = 5):
    #     """
    #     使用 'valid' 模式进行卷积，只返回有效平滑数据，避免边界突变。
    #
    #     Args:
    #         y (List[float]): 待平滑的数据列表。
    #         box_pts (int): 平滑窗口大小，数值越大曲线越平滑。
    #
    #     Returns:
    #         np.ndarray: 平滑后的数据。
    #     """
    #     if len(y) < box_pts:
    #         return np.array(y)
    #     box = np.ones(box_pts) / box_pts
    #     return np.convolve(y, box, mode='valid')
    #
    # # 设置平滑窗口大小
    # box_pts = 5
    # half_window = box_pts // 2
    # checkpoints_range = range(1000, len(cumulative_rewards_history) * 1000 + 1, 1000)
    #
    # # 1. 累计奖励
    # if len(cumulative_rewards_history) >= box_pts:
    #     smoothed_rewards = smooth_data(cumulative_rewards_history, box_pts)
    #     # 计算平滑曲线的正确 x 轴范围，以保证与原始数据对齐
    #     smoothed_x_range = range(1000 + half_window * 1000, len(smoothed_rewards) * 1000 + 1000 + half_window * 1000, 1000)
    #     ax1.plot(smoothed_x_range, smoothed_rewards, '-', linewidth=2.5, label='平滑累计奖励')
    #     ax1.plot(checkpoints_range, cumulative_rewards_history, 'o', markersize=4, alpha=0.5, label='原始')
    # else:
    #     ax1.plot(checkpoints_range, cumulative_rewards_history, 'o-', linewidth=2, markersize=6)
    # ax1.set_title('累计奖励趋势', fontsize=14)
    # ax1.set_ylabel('累计奖励', fontsize=12)
    # ax1.set_xlabel('迭代次数', fontsize=12)
    # ax1.grid(True, linestyle='--', alpha=0.6)
    # ax1.legend()
    #
    # # 2. 平均奖励
    # if len(avg_rewards_history) >= box_pts:
    #     smoothed_avg_rewards = smooth_data(avg_rewards_history, box_pts)
    #     smoothed_x_range = range(1000 + half_window * 1000, len(smoothed_avg_rewards) * 1000 + 1000 + half_window * 1000, 1000)
    #     ax2.plot(smoothed_x_range, smoothed_avg_rewards, '-', linewidth=2.5, label='平滑平均奖励')
    #     ax2.plot(checkpoints_range, avg_rewards_history, 'o', markersize=4, alpha=0.5, label='原始')
    # else:
    #     ax2.plot(checkpoints_range, avg_rewards_history, 'o-', linewidth=2, markersize=6)
    # ax2.set_title('平均奖励趋势', fontsize=14)
    # ax2.set_ylabel('平均奖励', fontsize=12)
    # ax2.set_xlabel('迭代次数', fontsize=12)
    # ax2.grid(True, linestyle='--', alpha=0.6)
    # ax2.legend()
    #
    # # 3. Q 值稳定性
    # if q_mean_history and len(q_mean_history) >= box_pts:
    #     smoothed_q_mean = smooth_data(q_mean_history, box_pts)
    #     smoothed_q_max = smooth_data(q_max_history, box_pts)
    #     smoothed_q_min = smooth_data(q_min_history, box_pts)
    #     smoothed_x_range = range(1000 + half_window * 1000, len(smoothed_q_mean) * 1000 + 1000 + half_window * 1000, 1000)
    #     ax3.plot(smoothed_x_range, smoothed_q_mean, '-', linewidth=2.5, label='平均Q值(平滑)')
    #     ax3.plot(smoothed_x_range, smoothed_q_max, '-', linewidth=2.5, label='最大Q值(平滑)')
    #     ax3.plot(smoothed_x_range, smoothed_q_min, '-', linewidth=2.5, label='最小Q值(平滑)')
    #     ax3.plot(checkpoints_range, q_mean_history, 'o', markersize=3, alpha=0.3)
    #     ax3.plot(checkpoints_range, q_max_history, '^', markersize=3, alpha=0.3)
    #     ax3.plot(checkpoints_range, q_min_history, 'v', markersize=3, alpha=0.3)
    # else:
    #     if q_mean_history:
    #         ax3.plot(checkpoints_range, q_mean_history, 's-', label='平均Q值')
    #         ax3.plot(checkpoints_range, q_max_history, '^-', label='最大Q值')
    #         ax3.plot(checkpoints_range, q_min_history, 'v-', label='最小Q值')
    #
    # ax3.set_title('Q值稳定性分析', fontsize=14)
    # ax3.set_ylabel('Q值', fontsize=12)
    # ax3.set_xlabel('迭代次数', fontsize=12)
    # ax3.legend()
    # ax3.grid(True, linestyle='--', alpha=0.6)
    #
    # # 4. 最优适应度趋势
    # if best_fitness_history:
    #     best_times = [f[0] for f in best_fitness_history]
    #     best_costs = [f[1] for f in best_fitness_history]
    #     ax4.plot(checkpoints_range, best_times, 'o-', label='最优时间', linewidth=2)
    #     ax4_twin = ax4.twinx()
    #     ax4_twin.plot(checkpoints_range, best_costs, 's-', label='最优成本', linewidth=2, color='orange')
    #     ax4.set_ylabel('最优时间', fontsize=12)
    #     ax4_twin.set_ylabel('最优成本', fontsize=12)
    #     ax4.set_xlabel('迭代次数', fontsize=12)
    #     ax4.set_title('最优适应度趋势', fontsize=14)
    #     ax4.grid(True, linestyle='--', alpha=0.6)
    #     ax4.legend(loc='upper left')
    #     ax4_twin.legend(loc='upper right')
    #
    # plt.tight_layout()
    # plt.savefig('convergence_diagnosis-2.0-q_table1.png', dpi=300)
    # plt.show()
    # 训练完成后保存 Q 表（以普通 dict 形式，避免 defaultdict 的 lambda 无法序列化）
    # try:
    #     if save_q_table_path:
    #         serializable_q = {k: list(v) for k, v in agent.q_table.items()}
    #         with open(save_q_table_path, "wb") as f:
    #             pickle.dump(serializable_q, f)
    #         print(f"训练完成，模型已保存到 {save_q_table_path}")
    # except Exception as e:
    #     print(f"保存 Q 表失败：{e}")

    # 返回结果
    result = {
        'convergence_archive': convergence_archive,
        'diversity_archive': diversity_archive,
        'all_fitness': all_fitness,
        'global_min_time_list': global_min_time_list,
        'global_min_cost_list': global_min_cost_list,
        'archive_size_history': archive_size_history,
        'q_table': agent.q_table,
        'hv_history': hv_history_abs,
        'final_sp': final_sp,
        'igd_values': igd_values,
        'rewards_history': rewards_history,
        'q_values_per_iteration': q_values_per_iteration,
        'best_fitness_history': best_fitness_history,
        'cumulative_rewards_history': cumulative_rewards_history,
        'avg_rewards_history': avg_rewards_history,
        'q_mean_history': q_mean_history,
        'q_max_history': q_max_history,
        'q_min_history': q_min_history,
        'q_std_history': q_std_history,
        'epsilon_history': epsilon_history,
        'state_count_history': state_count_history
    }

    return result, best_fitness_history, all_generational_best_history

