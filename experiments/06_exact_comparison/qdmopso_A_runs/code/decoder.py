"""Procurement revision: independent unit sourcing at t=0 and pooled FIFO use."""
from collections import Counter, defaultdict
import random
from product_semantics import product_types, stock_key


def decode(particle, config):
    n = config.num_tasks
    types = product_types(config)
    if len(particle.position) < 2 * n:
        return []
    resources = []
    for t, requested in enumerate(particle.position[:n]):
        eligible = config.num_suppliers.get(t, [])
        if not eligible:
            return []
        resources.append(requested if requested in eligible else random.choice(eligible))
    priorities = particle.position[n:2*n]
    schedule, completed, finish = [], set(), {}
    inventory = defaultdict(list)
    procurement = [t for t in range(n) if config.task_stage[t] == '采购']
    assembly = set(range(n)) - set(procurement)
    produced = Counter(stock_key(types[config.task_to_product[t]], config.task_to_part[t])
                       for t in procurement)
    required = Counter()
    for t in assembly:
        required.update(stock_key(types[config.task_to_product[t]], part)
                        for part in config.assembly_requirements.get(t, []))
    if produced != required:
        return []  # Exact demand matching; do not invent stock or leave surplus.

    # Each task purchases one physical unit. All orders start at zero, even
    # when several tasks use the same supplier. There is no supplier capacity.
    for t in procurement:
        if config.precedence.get(t, []):
            return []  # A procurement predecessor contradicts the t=0 policy.
        r = resources[t]
        info = config.machine_task_map.get((r, t))
        if info is None or float(info['time']) < 0:
            return []
        end = float(info['time'])
        part = config.task_to_part[t]
        product_type = types[config.task_to_product[t]]
        inventory[stock_key(product_type, part)].append((end, t))
        finish[t] = end
        completed.add(t)
        schedule.append(dict(task=t, resource=r, product=config.task_to_product[t],
                             product_type=product_type,
                             stage='采购', start=0.0, end=end, part=part))
    for stock in inventory.values():
        stock.sort()

    available = defaultdict(float)
    last_product_type = {}
    while assembly:
        ready = [t for t in assembly
                 if all(p in completed for p in config.precedence.get(t, []))]
        if not ready:
            return []
        t = max(ready, key=lambda t: (priorities[t], -t))
        r = resources[t]
        info = config.machine_task_map.get((r, t))
        if info is None or float(info['time']) < 0:
            return []
        product_type = types[config.task_to_product[t]]
        demands = Counter(config.assembly_requirements.get(t, []))
        consumed = []
        for part, quantity in demands.items():
            scope = stock_key(product_type, part)
            if len(inventory[scope]) < quantity:
                return []
            consumed.extend((part, arrival, k)
                            for arrival, k in inventory[scope][:quantity])
        material_time = max((v[1] for v in consumed), default=0.0)
        pred_time = max((finish[p] for p in config.precedence.get(t, [])), default=0.0)
        product = config.task_to_product[t]
        setup = (float(config.product_change_penalty)
                 if r in last_product_type and last_product_type[r] != product_type else 0.0)
        start = max(pred_time, available[r] + setup, material_time)
        end = start + float(info['time'])
        records = [dict(part=part, product_type=product_type, purchase_task=k, arrival=arrival,
                        use=start, holding_time=start-arrival)
                   for part, arrival, k in consumed]
        for part, quantity in demands.items():
            del inventory[stock_key(product_type, part)][:quantity]
        schedule.append(dict(task=t, resource=r, product=product,
                             product_type=product_type,
                             stage=config.task_stage[t], start=start, end=end,
                             consumed_units=records, setup_time=setup))
        available[r], last_product_type[r], finish[t] = end, product_type, end
        assembly.remove(t)
        completed.add(t)
    if any(inventory.values()):
        return []
    return sorted(schedule, key=lambda v: (v['start'], v['task']))
