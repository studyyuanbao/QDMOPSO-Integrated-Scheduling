"""Product-type metadata and the scope of interchangeable component units.

Explicit config.product_type_of_instance mappings are preferred. The fallback
recognizes only the three exact stage templates supplied with this manuscript.
No thresholds on instance IDs, lot sizes or task counts are used.
"""
from collections import defaultdict
import math

KNOWN_STAGE_TEMPLATES = {
    ('采购','采购','采购','部装','总装'): 'P1',
    ('采购','采购','采购','采购','部装','部装','总装'): 'P2',
    ('采购','采购','采购','采购','采购','部装','部装','部装','总装'): 'P3',
}


def product_types(config):
    grouped = defaultdict(list)
    for t in range(config.num_tasks):
        grouped[config.task_to_product[t]].append(t)
    supplied = getattr(config, 'product_type_of_instance', None)
    if supplied is not None:
        if not isinstance(supplied, dict) or any(i not in supplied for i in grouped):
            raise ValueError('product_type_of_instance must map every product instance to its type.')
        if any(supplied[i] is None for i in grouped):
            raise ValueError('Product types cannot be None.')
        return {i: supplied[i] for i in grouped}
    result = {}
    for i, tasks in grouped.items():
        stages = tuple(config.task_stage[t] for t in sorted(tasks))
        if stages not in KNOWN_STAGE_TEMPLATES:
            raise ValueError('Unknown product template; provide config.product_type_of_instance explicitly.')
        result[i] = KNOWN_STAGE_TEMPLATES[stages]
    return result


def stock_key(product_type, part):
    """Equal names alone never make cross-product-type stock interchangeable."""
    return product_type, part


def setup_duration(task, previous_end):
    """Read an actual setup duration for chart/export; never infer it from idle."""
    if task['stage'] == '采购':
        return 0.0
    if 'product_type' not in task or 'setup_time' not in task:
        raise ValueError('Regenerate the schedule with the revised decoder before plotting.')
    duration = float(task['setup_time'])
    if not math.isfinite(duration) or duration < 0:
        raise ValueError('Invalid setup duration.')
    if previous_end < 0:
        if duration > 1e-8:
            raise ValueError('First operation has no preceding-product changeover.')
    elif float(task['start']) - previous_end < duration - 1e-8:
        raise ValueError('Schedule gap is shorter than the required setup.')
    return duration
