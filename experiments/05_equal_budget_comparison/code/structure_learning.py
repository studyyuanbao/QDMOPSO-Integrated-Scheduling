import random


def structure_learning_recombine(pbest_position, config, inherit_ratio=0.5):
    """
    从pbest继承部分任务结构，其余部分重新构造，形成新个体（更偏向低成本资源）。
    """
    num_tasks = config.num_tasks
    inherit_len = int(num_tasks * inherit_ratio)

    # 拷贝前 inherit_len 个任务的结构
    new_resource = pbest_position[:inherit_len] + [0] * (num_tasks - inherit_len)
    new_priority = pbest_position[num_tasks:num_tasks + inherit_len] + [0.0] * (num_tasks - inherit_len)

    # 已调度任务集合
    fixed_tasks = set(range(inherit_len))

    # 构造剩余部分
    scheduled = list(fixed_tasks)
    unscheduled = set(range(num_tasks)) - fixed_tasks
    ready_tasks = {t for t in unscheduled if all(p in scheduled for p in config.precedence.get(t, []))}

    while unscheduled:
        if not ready_tasks:
            raise Exception("结构学习失败：可能前置冲突")

        task = random.choice(list(ready_tasks))
        task_stage = config.task_stage[task]

        # 根据任务阶段选择资源（优先低成本）
        if task_stage == '采购':
            if task in config.task_suppliers and config.task_suppliers[task]:
                res_list = config.task_suppliers[task]
                res_list.sort(key=lambda r: config.machine_task_map.get((r, task), {}).get('cost', float('inf')))
                res = res_list[0]  # 选择最低成本资源
            else:
                print(f"警告：结构学习中，采购任务 {task} 未定义供应商或列表为空。")
                res = 0
        else:
            if task_stage in config.machine_resources and config.machine_resources[task_stage]:
                res_list = config.machine_resources[task_stage]
                res_list.sort(key=lambda r: config.machine_task_map.get((r, task), {}).get('cost', float('inf')))
                res = res_list[0]  # 选择最低成本资源
            else:
                print(f"错误：结构学习中，任务 {task} 阶段 '{task_stage}' 未定义资源或为空。")
                raise ValueError(f"结构学习无法为任务 {task} 的阶段 {task_stage} 找到可用资源。")

        prio = random.random()

        new_resource[task] = res
        new_priority[task] = prio

        scheduled.append(task)
        unscheduled.remove(task)
        ready_tasks.remove(task)

        # 更新 ready_tasks
        for t in unscheduled:
            if all(pred in scheduled for pred in config.precedence.get(t, [])):
                ready_tasks.add(t)

    return new_resource + new_priority