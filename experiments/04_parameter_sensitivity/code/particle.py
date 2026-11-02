import random
from utils import dominates
from feasible_init import generate_feasible_particle


class Particle:
    def __init__(self, config, feasible_init=False, init_progress=0.0):
        self.schedule = []
        self.num_tasks = config.num_tasks
        self.crowding_distance = 0.0

        if feasible_init:
            self.position = generate_feasible_particle(config, progress=init_progress)
            self.resource_assignment = self.position[:self.num_tasks]
            self.priority_values = self.position[self.num_tasks:]
        else:
            self.resource_assignment = [
                self._select_resource_with_cost(i, config, init_progress)
                for i in range(self.num_tasks)
            ]
            self.priority_values = [random.random() for _ in range(self.num_tasks)]
            self.position = self.resource_assignment + self.priority_values

        self.velocity = [0.0] * (2 * self.num_tasks)
        self.fitness = [float('inf'), float('inf')]
        self.pbest_position = self.position[:]
        self.pbest_fitness = self.fitness[:]

    def _select_resource_with_cost(self, task_id, config, progress):
        """平衡时间和成本的资源选择"""
        if task_id not in config.num_suppliers:
            raise ValueError(f"任务 {task_id} 未定义可用资源")

        available_resources = config.num_suppliers[task_id]

        # 如果只有1个可选资源，直接返回
        if len(available_resources) == 1:
            return available_resources[0]

        # 修改：综合考虑时间和成本
        scored_resources = []
        for res in available_resources:
            task_data = config.machine_task_map.get((res, task_id), {})
            time_val = task_data.get('time', float('inf'))
            cost_val = task_data.get('cost', float('inf'))

            # 动态权重：前期偏时间，后期偏成本
            # progress=0时time_weight=0.7, progress=1时time_weight=0.3
            time_weight = 0.7 - 0.4 * progress
            cost_weight = 1.0 - time_weight

            # 由于不知道全局范围，使用相对评分
            # 找到当前任务所有资源的范围来归一化
            all_times = [config.machine_task_map.get((r, task_id), {}).get('time', float('inf'))
                         for r in available_resources]
            all_costs = [config.machine_task_map.get((r, task_id), {}).get('cost', float('inf'))
                         for r in available_resources]

            # 移除无穷大值
            valid_times = [t for t in all_times if t < float('inf')]
            valid_costs = [c for c in all_costs if c < float('inf')]

            if valid_times and valid_costs:
                min_time, max_time = min(valid_times), max(valid_times)
                min_cost, max_cost = min(valid_costs), max(valid_costs)

                # 归一化到[0,1]范围（越小越好）
                if max_time > min_time:
                    norm_time = (time_val - min_time) / (max_time - min_time)
                else:
                    norm_time = 0

                if max_cost > min_cost:
                    norm_cost = (cost_val - min_cost) / (max_cost - min_cost)
                else:
                    norm_cost = 0

                score = time_weight * norm_time + cost_weight * norm_cost
            else:
                # 备用方案：直接使用原始值
                score = time_weight * time_val + cost_weight * cost_val

            scored_resources.append((score, res, time_val, cost_val))

        scored_resources.sort(key=lambda x: x[0])

        # 动态选择策略
        if random.random() < 0.8:  # 80%选择最优
            best_res = scored_resources[0][1]
            # 可选：记录选择信息用于调试
            # if random.random() < 0.01:  # 1%概率输出调试信息
            #     print(
            #         f"任务{task_id}: 选择资源{best_res} (时间:{scored_resources[0][2]:.1f}, 成本:{scored_resources[0][3]:.1f})")
            return best_res
        else:  # 20%探索其他选项
            return random.choice([r[1] for r in scored_resources[1:]]) if len(scored_resources) > 1 else \
            scored_resources[0][1]

    def update_pbest(self):
        if dominates(self.fitness, self.pbest_fitness):
            self.pbest_position = self.position[:]
            self.pbest_fitness = self.fitness[:]