import random
import math


def generate_feasible_particle(config, progress=0.0):
    """
    生成可行粒子，支持动态概率选择低成本资源
    Args:
        config: 配置参数对象
        progress: 迭代进度（0.0~1.0），0表示初始，1表示最后一代
    """
    num_tasks = config.num_tasks
    precedence = config.precedence

    scheduled = []
    scheduled_set = set()
    unscheduled = set(range(num_tasks))
    ready_tasks = {i for i in range(num_tasks) if config.precedence.get(i, []) == []}

    priority_values = [0.0] * num_tasks
    resource_assignment = [0] * num_tasks

    while unscheduled:
        if not ready_tasks:
            raise Exception("构造失败：可能是precedence中存在环")

        task = random.choice(list(ready_tasks))

        # 动态成本启发式资源选择
        if task < len(config.num_suppliers):
            available_resources = config.num_suppliers[task]
            if not available_resources:
                raise ValueError(f"任务 {task} 没有可用的资源定义")

            # 按成本升序排序资源
            available_resources.sort(
                key=lambda res: config.machine_task_map.get((res, task), {}).get('cost', float('inf')))

            # 动态概率计算（使用S型曲线平滑过渡）
            def get_selection_prob():
                min_prob = 0.3
                max_prob = 0.9
                # 原为 8，这里加大陡峭程度
                return min_prob + (max_prob - min_prob) / (1 + math.exp(-16 * (progress - 0.2)))

            # 根据动态概率选择资源
            if random.random() < get_selection_prob():
                resource = available_resources[0]  # 选择最低成本资源
            else:
                # 从非最低成本的资源中随机选择（保持多样性）
                resource = random.choice(
                    available_resources[1:] if len(available_resources) > 1 else available_resources)
        else:
            raise ValueError(f"任务 {task} 超出num_suppliers定义范围")

        priority = random.random()

        resource_assignment[task] = resource
        priority_values[task] = priority
        scheduled.append(task)
        scheduled_set.add(task)
        unscheduled.remove(task)
        ready_tasks.remove(task)

        for t in unscheduled:
            if t not in ready_tasks and all(pred in scheduled_set for pred in precedence.get(t, [])):
                ready_tasks.add(t)

    return resource_assignment + priority_values