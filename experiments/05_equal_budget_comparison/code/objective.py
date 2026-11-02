"""Compute holding costs from actual unit consumption, with zero-wait allowed.

Idle cost intentionally retains the original initial-wait/setup convention;
its reconciliation with manuscript equations (15)-(16) is a separate step.
"""
from collections import Counter, defaultdict
from product_semantics import product_types

global_min_time = float('inf')
global_max_time = -float('inf')
global_min_cost = float('inf')
global_max_cost = -float('inf')


def evaluate(schedule, config, update_global=True):
    global global_min_time, global_max_time, global_min_cost, global_max_cost
    if not schedule:
        return float('inf'), float('inf')
    types = product_types(config)
    purchases = {r['task']:r for r in schedule if r['stage'] == '采购'}
    used = set()
    holding = 0.0
    direct_cost = sum(float(config.machine_task_map[r['resource'], r['task']]['cost'])
                      for r in schedule)
    for op in schedule:
        if op.get('product_type') != types[config.task_to_product[op['task']]]:
            return float('inf'), float('inf')
        if op['stage'] == '采购':
            if abs(op['start']) > 1e-8:
                return float('inf'), float('inf')
            continue
        records = op.get('consumed_units')
        if records is None:
            return float('inf'), float('inf')  # Do not silently score old schedules.
        if Counter(v['part'] for v in records) != Counter(config.assembly_requirements.get(op['task'], [])):
            return float('inf'), float('inf')
        for item in records:
            k = item['purchase_task']
            if k not in purchases or k in used:
                return float('inf'), float('inf')
            if config.task_to_part[k] != item['part']:
                return float('inf'), float('inf')
            if (types[config.task_to_product[k]] != types[config.task_to_product[op['task']]]
                    or item.get('product_type') != types[config.task_to_product[op['task']]]):
                return float('inf'), float('inf')
            arrival = purchases[k]['end']
            if abs(arrival-item['arrival']) > 1e-8 or op['start'] < arrival-1e-8:
                return float('inf'), float('inf')
            used.add(k)
            holding += max(0.0, op['start']-arrival)
    if used != set(purchases):
        return float('inf'), float('inf')
    machines = defaultdict(list)
    for r in schedule:
        if r['stage'] != '采购':
            machines[r['resource']].append(r)
    idle = 0.0
    for records in machines.values():
        last = 0.0
        previous_type = None
        for r in sorted(records, key=lambda r:(r['start'],r['task'])):
            current_type = types[config.task_to_product[r['task']]]
            setup = (float(config.product_change_penalty)
                     if previous_type is not None and previous_type != current_type else 0.0)
            if r['start'] < last + setup - 1e-8:
                return float('inf'), float('inf')
            idle += max(0.0, r['start']-last)
            last = r['end']
            previous_type = current_type
    makespan = max(r['end'] for r in schedule)
    cost = direct_cost + 0.5*holding + 0.2*idle
    if update_global:
        global_min_time=min(global_min_time,makespan)
        global_max_time=max(global_max_time,makespan)
        global_min_cost=min(global_min_cost,cost)
        global_max_cost=max(global_max_cost,cost)
    return makespan,cost
