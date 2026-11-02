from collections import defaultdict

# 全局变量记录目标函数的范围
global_min_time = float('inf')
global_max_time = -float('inf')
global_min_cost = float('inf')
global_max_cost = -float('inf')

def evaluate(schedule, config, update_global=True):
    global global_min_time, global_max_time, global_min_cost, global_max_cost

    if not schedule:
        return float('inf'), float('inf')

    makespan = max(task['end'] for task in schedule)
    total_cost = 0
    inventory_cost = 0
    idle_time_penalty = 0

    machine_last_end = defaultdict(int)

    for task in schedule:
        resource = task['resource']
        task_id = task['task']
        start = task['start']
        end = task['end']
        stage = task['stage']

        cost = config.machine_task_map.get((resource, task_id), {}).get('cost')
        if cost is not None:
            total_cost += cost

        idle = max(0, start - machine_last_end[resource])
        idle_time_penalty += idle
        machine_last_end[resource] = end

        if stage == '采购':
            part = config.task_to_part.get(task_id)
            if part:
                next_use = min(
                    (t['start'] for t in schedule if
                     part in config.assembly_requirements.get(t['task'], []) and t['start'] > end),
                    default=end + 10
                )
                inventory_cost += next_use - end

    total_cost += 0.5 * inventory_cost + 0.2 * idle_time_penalty

    # 更新全局最大最小值
    if update_global:
        global_min_time = min(global_min_time, makespan)
        global_max_time = max(global_max_time, makespan)
        global_min_cost = min(global_min_cost, total_cost)
        global_max_cost = max(global_max_cost, total_cost)

    return makespan, total_cost
