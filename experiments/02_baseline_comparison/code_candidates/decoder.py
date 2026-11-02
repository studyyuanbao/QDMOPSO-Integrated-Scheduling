import random
from collections import defaultdict, deque
from typing import Deque, DefaultDict, Dict, List, Mapping, Sequence, Tuple, Optional

def decode(particle, config):
    """解码粒子位置为可行的调度方案（修复版：避免 ready_queue 重复 / 阻塞处理优化）"""
    num_tasks: int = config.num_tasks

    # ========== 1) 资源分配验证 ==========
    resource_assignment: List[int] = []
    for task_id, assigned_res in enumerate(particle.position[:num_tasks]):
        if task_id not in config.num_suppliers:
            print(f"解码错误：任务 {task_id} 未定义可用资源")
            return []

        available_res: Sequence[int] = config.num_suppliers[task_id]
        if not available_res:
            print(f"解码错误：任务 {task_id} 无可用资源")
            return []

        resource_assignment.append(assigned_res if assigned_res in available_res else random.choice(list(available_res)))

    priority_values: Sequence[float] = particle.position[num_tasks:]
    if len(priority_values) < num_tasks:
        print("解码错误：优先级向量长度不足")
        return []

    # ========== 2) 初始化图结构 ==========
    indegree: List[int] = [0] * num_tasks
    successors: DefaultDict[int, List[int]] = defaultdict(list)

    for succ, preds in config.precedence.items():
        for pred in preds:
            indegree[succ] += 1
            successors[pred].append(succ)

    # 就绪队列（按优先级降序），并维护一个 in_queue 集合用于去重
    initial_ready = sorted(
        [i for i in range(num_tasks) if indegree[i] == 0],
        key=lambda x: priority_values[x],
        reverse=True
    )
    ready_queue: Deque[int] = deque(initial_ready)
    in_queue: set[int] = set(initial_ready)

    # ========== 3) 调度状态 ==========
    schedule: List[Dict] = []
    scheduled_tasks: set[int] = set()
    task_end_time: List[float] = [0.0] * num_tasks
    machine_available: DefaultDict[int, float] = defaultdict(float)
    last_product: Dict[int, Optional[int]] = {}
    part_inventory: DefaultDict[str, List[float]] = defaultdict(list)
    blocked_due_to_parts: set[int] = set()
    blocked_retry_count: DefaultDict[int, int] = defaultdict(int)
    MAX_BLOCK_RETRY = 5

    task_waiting_for_parts: DefaultDict[int, List[str]] = defaultdict(list)
    part_consumers: DefaultDict[str, List[int]] = defaultdict(list)
    for t in range(num_tasks):
        req = getattr(config, 'assembly_requirements', {}).get(t, [])
        for part in req:
            part_consumers[part].append(t)

    part_to_purchase_tasks: DefaultDict[str, List[int]] = defaultdict(list)
    for t, part_name in getattr(config, 'task_to_part', {}).items():
        if config.task_stage[t] == '采购':
            part_to_purchase_tasks[part_name].append(t)

    iteration_count = 0
    max_iterations = num_tasks * 100  # 放宽一点

    # ========== 4) 主循环 ==========
    while (ready_queue or blocked_due_to_parts) and iteration_count < max_iterations:
        iteration_count += 1

        # 当就绪队列为空但有阻塞任务时，推动相关采购任务（把满足前置的采购加入队列）
        if not ready_queue and blocked_due_to_parts:
            # 推动缺失部件的采购任务
            for bt in list(blocked_due_to_parts):
                req_parts = getattr(config, 'assembly_requirements', {}).get(bt, [])
                for p in req_parts:
                    # 估计当前时间为已知机器可用时间的最大值
                    current_time_estimate = max(list(machine_available.values()) + [0.0])
                    part_available = any(t_end <= current_time_estimate for t_end in part_inventory.get(p, []))
                    if not part_available:
                        for pur_t in part_to_purchase_tasks.get(p, []):
                            if pur_t in scheduled_tasks or pur_t in in_queue:
                                continue
                            # 只要采购任务的前置已完成即可入队
                            if all(pred in scheduled_tasks for pred in config.precedence.get(pur_t, [])):
                                ready_queue.appendleft(pur_t)
                                in_queue.add(pur_t)

            # 如果仍为空，选取阻塞次数最少的任务做一次重试（把它放回队列以避免饥饿）
            if not ready_queue and blocked_due_to_parts:
                task_id = min(blocked_due_to_parts, key=lambda x: blocked_retry_count[x])
                if task_id not in scheduled_tasks and task_id not in in_queue:
                    ready_queue.appendleft(task_id)
                    in_queue.add(task_id)
                    blocked_due_to_parts.discard(task_id)

        if not ready_queue:
            break

        task_id: int = ready_queue.popleft()
        in_queue.discard(task_id)

        # 如果已经被调度（可能因重复入队）就跳过
        if task_id in scheduled_tasks:
            continue

        res: int = resource_assignment[task_id]
        stage: str = config.task_stage[task_id]
        product: int = config.task_to_product[task_id]

        task_data: Optional[Dict[str, float]] = config.machine_task_map.get((res, task_id))
        if not task_data:
            print(f"解码错误：任务 {task_id} 在资源 {res} 上未定义加工参数")
            return []

        # 前置完成时间
        pred_end_times = [task_end_time[p] for p in config.precedence.get(task_id, [])]
        ready_time: float = max(pred_end_times) if pred_end_times else 0.0

        # 换型惩罚
        setup_time: float = 0.0
        prev_prod = last_product.get(res)
        if stage != '采购' and prev_prod is not None and prev_prod != product:
            setup_time = float(config.product_change_penalty)

        start_time: float = max(ready_time, machine_available[res] + setup_time)
        if pred_end_times and start_time < max(pred_end_times):
            start_time = max(pred_end_times)

        end_time: float = start_time + float(task_data['time'])

        if stage == '采购':
            # 找到在 ready_queue（快照）中同供应商且未调度的采购任务并批处理
            rq_snapshot = list(ready_queue)
            same_supplier_tasks = []
            for t in rq_snapshot:
                if t in scheduled_tasks:
                    continue
                if config.task_stage[t] == '采购' and resource_assignment[t] == res:
                    if t not in same_supplier_tasks:
                        same_supplier_tasks.append(t)

            batch_tasks = [task_id] + [t for t in same_supplier_tasks if t != task_id]

            # 批处理所有任务（注意去重）
            for parallel_task in batch_tasks:
                if parallel_task in scheduled_tasks:
                    continue
                parallel_data = config.machine_task_map.get((res, parallel_task))
                if not parallel_data:
                    print(f"解码错误：并行任务 {parallel_task} 参数缺失")
                    return []

                parallel_end = start_time + float(parallel_data['time'])
                task_end_time[parallel_task] = parallel_end

                produced_part = config.task_to_part.get(parallel_task)
                if produced_part:
                    part_inventory[produced_part].append(parallel_end)
                    part_inventory[produced_part].sort()

                schedule.append({
                    'task': parallel_task,
                    'resource': res,
                    'product': config.task_to_product[parallel_task],
                    'start': start_time,
                    'end': parallel_end,
                    'stage': '采购'
                })
                scheduled_tasks.add(parallel_task)

                # 更新后继并入队（去重）
                for succ in successors[parallel_task]:
                    indegree[succ] -= 1
                    if indegree[succ] == 0 and succ not in scheduled_tasks and succ not in in_queue:
                        ready_queue.append(succ)
                        in_queue.add(succ)

                # 如果该 parallel_task 还在 ready_queue（可能有多次出现），全部移除并更新 in_queue
                while parallel_task in ready_queue:
                    try:
                        ready_queue.remove(parallel_task)
                    except ValueError:
                        break
                if parallel_task in in_queue:
                    in_queue.discard(parallel_task)

                if parallel_task in blocked_due_to_parts:
                    blocked_due_to_parts.discard(parallel_task)

            # 更新资源可用为批次中最晚结束时间
            machine_available[res] = max(machine_available[res],
                                         max((task_end_time[t] for t in batch_tasks), default=machine_available[res]))

            # 采购完成后，检查相关阻塞装配任务是否可被唤醒
            for produced_part in list(part_inventory.keys()):
                # 检查每个消费者
                for consumer in part_consumers.get(produced_part, []):
                    if consumer in scheduled_tasks or consumer in in_queue:
                        continue
                    # 判断该消费者的所有部件是否已经到货（至少有一个到货记录且时间不晚于当前可用时间）
                    reqs = config.assembly_requirements.get(consumer, [])
                    ok = True
                    for rp in reqs:
                        lst = part_inventory.get(rp, [])
                        if not lst:
                            ok = False
                            break
                    if ok and all(pred in scheduled_tasks for pred in config.precedence.get(consumer, [])):
                        ready_queue.append(consumer)
                        in_queue.add(consumer)
                        if consumer in blocked_due_to_parts:
                            blocked_due_to_parts.discard(consumer)

        else:
            # 装配/其它任务：检查部件需求
            required_parts = config.assembly_requirements.get(task_id, [])
            if required_parts:
                base_time = max(ready_time, machine_available[res])
                part_ready_times = []
                missing = False
                for p in required_parts:
                    lst = part_inventory.get(p, [])
                    if not lst:
                        # 标记为阻塞（不立刻入队）
                        blocked_due_to_parts.add(task_id)
                        blocked_retry_count[task_id] += 1
                        task_waiting_for_parts[task_id] = required_parts
                        missing = True
                        break
                    else:
                        # 取最早到货时间
                        part_ready_times.append(min(lst))

                if missing:
                    # 跳过这次调度，等待采购把部件补上（不要把任务再次 append 到 ready_queue）
                    continue

                parts_available_time = max(part_ready_times) if part_ready_times else base_time
                if parts_available_time > base_time:
                    start_time = parts_available_time
                    end_time = start_time + float(task_data["time"])

            # 正常调度（装配/加工）
            task_end_time[task_id] = end_time
            machine_available[res] = end_time
            last_product[res] = product

            schedule.append({
                'task': task_id,
                'resource': res,
                'product': product,
                'start': start_time,
                'end': end_time,
                'stage': stage
            })
            scheduled_tasks.add(task_id)

            # 消费部件：移除最早且 <= start_time 的时间戳
            if required_parts:
                for p in required_parts:
                    lst = part_inventory[p]
                    removed = False
                    for idx, t_end in enumerate(lst):
                        if t_end <= start_time:
                            del lst[idx]
                            removed = True
                            break
                    # 若没有可消费项，说明逻辑出错（应该不会出现）
                    if not removed:
                        # 作为保护，不报错，继续
                        pass

                if task_id in blocked_due_to_parts:
                    blocked_due_to_parts.discard(task_id)

            # 更新后继并入队（去重）
            for succ in successors[task_id]:
                indegree[succ] -= 1
                if indegree[succ] == 0 and succ not in scheduled_tasks and succ not in in_queue:
                    ready_queue.append(succ)
                    in_queue.add(succ)

        # 每次循环结束后重新排序 ready_queue（采购优先、阻塞任务降低优先级）
        if ready_queue:
            ready_list = list(ready_queue)
            ready_list.sort(key=lambda x: (
                2 if config.task_stage[x] == '采购' else 0,
                priority_values[x] - (1e6 if x in blocked_due_to_parts else 0)
            ), reverse=True)
            ready_queue.clear()
            ready_queue.extend(ready_list)
            in_queue = set(ready_list)  # 更新 in_queue 使之和队列一致

    # ========== 5) 完整性与时序校验 ==========
    if len(schedule) != num_tasks:
        unscheduled = set(range(num_tasks)) - {t['task'] for t in schedule}
        print(f"解码错误：未调度任务 {unscheduled}")
        print(f"调试信息: ready_queue={list(ready_queue)}, blocked_due_to_parts={blocked_due_to_parts}")
        print(f"迭代次数: {iteration_count}/{max_iterations}")
        return []

    for task in schedule:
        for pred in config.precedence.get(task['task'], []):
            if task['start'] < task_end_time[pred]:
                print(f"时序错误：任务{task['task']}(开始于{task['start']}) 早于前置任务{pred}(结束于{task_end_time[pred]})")
                return []

    return sorted(schedule, key=lambda x: x['start'])
