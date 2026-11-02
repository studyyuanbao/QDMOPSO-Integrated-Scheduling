from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False  # 显示负号
from matplotlib import rcParams
from init3 import Config
from pso_rl_new import run_mopso
from typing import List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import numpy as np
import pandas as pd


# 设置字体
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

def validate_schedule(schedule, config):
    """调度合法性验证函数"""
    task_start = {t['task']: t['start'] for t in schedule}
    task_end = {t['task']: t['end'] for t in schedule}
    errors = []

    for t in schedule:
        tid = t['task']
        preds = config.precedence.get(tid, [])
        for p in preds:
            if task_start[tid] < task_end[p]:
                errors.append(f"任务{tid} 在其前置任务 {p} 结束前开始：{task_start[tid]} < {task_end[p]}")

    if errors:
        print("甘特图验证失败：")
        for e in errors:
            print("  ", e)
    else:
        print("所有前置任务顺序验证通过")

def plot_pareto(archive):
    """Pareto前沿图"""
    times = [p.fitness[0] for p in archive]
    costs = [p.fitness[1] for p in archive]

    plt.figure(figsize=(8, 6))
    plt.scatter(times, costs, c='blue', label='Pareto Front')
    plt.xlabel("完工时间 (Makespan)")
    plt.ylabel("成本 (Total Cost)")
    plt.title("Pareto 前沿")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_hypervolume_trend(hv_history: List[float]):
    """绘制Hypervolume随迭代次数变化曲线"""
    # 检查hv_history是否为列表且元素为数字
    if not isinstance(hv_history, list):
        print(f"错误：hv_history 不是列表，而是 {type(hv_history)}")
        return
    if not all(isinstance(x, (int, float)) for x in hv_history):
        print("错误：hv_history 中包含非数值元素")
        return

    plt.figure(figsize=(8, 6))
    # 显式指定x和y
    plt.plot(range(len(hv_history)), hv_history, color='green', linewidth=2, marker='o', markersize=3)
    plt.xlabel('迭代次数')
    plt.ylabel('Hypervolume')
    plt.title('Hypervolume 变化趋势')
    plt.grid(True)
    plt.show()


# --- 【新增函数】生成固定的全局资源编号映射 ---
def generate_global_resource_map(config):
    """
    根据 Config 对象中的全局任务信息，生成固定的、全局资源编号映射。
    """
    # 阶段顺序和前缀定义，确保编号和排序一致
    stages_order_cn = ["采购", "部装", "总装"]
    stage_to_prefix = {"采购": "P", "部装": "S", "总装": "F"}

    # 1. 收集所有 unique resources 及其 stage
    resource_to_stages = {}
    # 遍历所有 (资源, 任务) 组合，从任务获取阶段信息
    for (res_id, task_id) in config.machine_task_map.keys():
        stage = config.task_stage[task_id]
        resource_to_stages.setdefault(res_id, set()).add(stage)

    # 2. 确定资源的主阶段 (按阶段顺序优先级，例如：总装 > 部装 > 采购)
    # 这里的逻辑是为了防止一个资源被误认为是多个阶段的，取其在顺序中更靠后的阶段。
    resource_stage_map = {}
    stage_priority = {s: i for i, s in enumerate(stages_order_cn)}
    for res_id, stages in resource_to_stages.items():
        # 选择在 stages_order_cn 中索引最大的阶段作为主阶段 (总装=2 > 部装=1 > 采购=0)
        primary_stage = max(stages, key=lambda s: stage_priority.get(s, -1))
        resource_stage_map[res_id] = primary_stage

    # 3. 按阶段分组并全局编号
    staged_resources = {}
    for r_id, stage in resource_stage_map.items():
        staged_resources.setdefault(stage, []).append(r_id)

    global_resource_map = {}
    for stage in stages_order_cn:
        if stage not in staged_resources: continue

        # 按原始 ID 排序，确保编号固定
        resources_in_stage = sorted(staged_resources[stage])
        prefix = stage_to_prefix.get(stage, 'R')

        for i, original_id in enumerate(resources_in_stage, start=1):
            global_resource_map[original_id] = {
                'new_id': i,
                'prefix': prefix,
                'stage': stage
            }
    return global_resource_map


# def draw_gantt(schedule, title="Optimal Schedule Gantt Chart", global_resource_map=None):
#     """
#     绘制甘特图，具备所有优化功能。
#     - 🚨 新增：支持传入 global_resource_map 以使用固定的、全局资源编号。
#     - 🚨 新增：切换时间条纹增加外框轮廓。
#     """
#     if not schedule:
#         print("schedule is empty")
#         return
#
#     # --- 参数定义 ---
#     FIXED_SETUP_TIME = 2.0  # 产品换型时间惩罚值
#
#     # --- 阶段定义、缩写、前缀和顺序映射 ---
#     stages_order_cn = ["采购", "部装", "总装"]
#     stages_order_index = {stage: i for i, stage in enumerate(stages_order_cn)}
#     stage_to_prefix = {"采购": "P", "部装": "S", "总装": "F"}
#     stage_prefix_to_cn = {'P': '采购', 'S': '部装', 'F': '总装'}
#     stage_to_short_en = {"采购": "Prc", "部装": "Sub", "总装": "Fin"}
#
#     # --- 字体设置 ---
#     from matplotlib import font_manager
#     available_fonts = {f.name for f in font_manager.fontManager.ttflist}
#     preferred_font = "Times New Roman" if "Times New Roman" in available_fonts else 'sans-serif'
#     plt.rcParams['axes.unicode_minus'] = False
#
#     # --- 1) 资源编号映射 ---
#     if global_resource_map is not None and isinstance(global_resource_map, dict):
#         # 使用用户提供的全局映射 (固定编号)
#         resource_new_id_map = global_resource_map
#     else:
#         # 未提供全局映射，回退到基于当前解的原始ID (R{original_id})
#         resource_new_id_map = {}
#         for t in schedule:
#             original_resource_id = t.get('resource', 0)
#             stage = t.get('stage', '总装')
#             resource_new_id_map[original_resource_id] = {'new_id': original_resource_id, 'prefix': 'R', 'stage': stage}
#
#     # --- 2) 计算产品内部序号 prod_seq_map ---
#     prod_tasks = {}
#     for t in schedule:
#         try:
#             p = int(t.get('product', 0))
#         except:
#             p = 0
#         prod_tasks.setdefault(p, []).append(t)
#
#     prod_seq_map = {}
#     for p, tasks in prod_tasks.items():
#         tasks_sorted = sorted(tasks, key=lambda x: (
#             stages_order_cn.index(x.get('stage', '总装')) if x.get('stage', '总装') in stages_order_cn else 99,
#             float(x.get('start', 0) or 0)
#         ))
#         for idx, t in enumerate(tasks_sorted, start=1):
#             prod_seq_map[id(t)] = idx
#
#     # --- 3) 为资源分配子行（AssignedRow）和记录切换时间 ---
#     resource_tasks = {}
#     for t in schedule:
#         original_resource_id = t.get('resource', 0)
#
#         # 使用固定映射信息
#         new_info = resource_new_id_map.get(original_resource_id)
#         if new_info is None:
#             # 如果不在映射中，使用原始ID作为新ID和默认前缀
#             new_info = {'new_id': original_resource_id, 'prefix': 'R', 'stage': t.get('stage', '总装')}
#
#         new_resource_key = (new_info['prefix'], new_info['new_id'])
#         resource_tasks.setdefault(new_resource_key, []).append(t)
#         t['new_resource_key'] = new_resource_key
#
#     task_row_assignment = {}
#
#     def resource_key_sort_key(new_resource_key):
#         prefix, new_id = new_resource_key
#         stage_cn = stage_prefix_to_cn.get(prefix, '总装')
#         order_index = stages_order_index.get(stage_cn, 99)
#         return (order_index, new_id)
#
#     sorted_new_resource_keys = sorted(resource_tasks.keys(), key=resource_key_sort_key)
#
#     last_end_time = {}
#
#     for new_resource_key in sorted_new_resource_keys:
#         tasks_on_resource = resource_tasks[new_resource_key]
#         tasks_sorted = sorted(tasks_on_resource, key=lambda x: float(x.get('start', 0) or 0))
#         rows_end_time = []
#
#         for task in tasks_sorted:
#             st = float(task.get('start', 0) or 0)
#             ed = float(task.get('end', 0) or st + 1)
#             assigned_row = -1
#
#             for idx in range(len(rows_end_time)):
#                 if rows_end_time[idx] <= st:
#                     assigned_row = idx
#                     last_end_time[id(task)] = rows_end_time[idx]
#                     rows_end_time[idx] = ed
#                     break
#             if assigned_row == -1:
#                 assigned_row = len(rows_end_time)
#                 last_end_time[id(task)] = -1
#                 rows_end_time.append(ed)
#             task_row_assignment[id(task)] = assigned_row
#
#     # --- 4) 构建 Y 位置映射和标签 ---
#     y_ticks = []
#     y_labels = []
#     current_y = 0.0
#     resource_row_counts = {}
#     resource_base = {}
#
#     for new_resource_key in sorted_new_resource_keys:
#         prefix, new_id = new_resource_key
#         tasks_on_resource = resource_tasks.get(new_resource_key, [])
#         rows_used = max((task_row_assignment.get(id(t), 0) for t in tasks_on_resource), default=0) + 1
#         resource_row_counts[new_resource_key] = rows_used
#         resource_base[new_resource_key] = current_y
#         for r in range(rows_used):
#             y_center = current_y + r
#             y_ticks.append(y_center)
#             # 使用固定编号作为 Y 轴标签
#             if r == 0:
#                 y_labels.append(f"{prefix}{new_id}")
#             else:
#                 y_labels.append(f"{prefix}{new_id}-{r + 1}")
#         current_y += rows_used + 0.5
#     y_limit = current_y
#
#     # --- 5) 绘制图表 ---
#     bar_height = 0.8
#     fig_height = max(8, len(y_ticks) * 0.5)
#     fig, ax = plt.subplots(figsize=(14, fig_height))
#     fig.patch.set_facecolor('white')
#     ax.set_facecolor('white')
#     ax.grid(False)
#
#     # 产品颜色
#     product_colors = {}
#     try:
#         products = set(int(t.get('product', 0)) for t in schedule if 'product' in t)
#         num_products = max(products) + 1 if products else 1
#         color_scheme = [(0.121, 0.466, 0.705), (1.0, 0.498, 0.054), (0.172, 0.627, 0.172), (0.839, 0.152, 0.156),
#                         (0.580, 0.403, 0.741), (0.682, 0.780, 0.909), (0.549, 0.337, 0.294), (0.890, 0.466, 0.760),
#                         (0.498, 0.498, 0.498), (0.737, 0.741, 0.133)]
#         for p in range(num_products):
#             product_colors[p] = color_scheme[p % len(color_scheme)]
#     except:
#         num_products = 0
#         pass
#
#     # --- 5a) 绘制切换时间条 (Setup Bars) ---
#     setup_color = '#C7C7C7'
#     setup_hatch = '///'
#     SETUP_EDGE_COLOR = (0.2, 0.2, 0.2)
#     SETUP_LINE_WIDTH = 1.2
#
#     for t in schedule:
#         new_resource_key = t.get('new_resource_key')
#         base = resource_base.get(new_resource_key, 0.0)
#         assigned = task_row_assignment.get(id(t), 0)
#         y_center = base + assigned
#
#         st = float(t.get('start', 0) or 0)
#         last_ed = last_end_time.get(id(t), -1.0)
#
#         if last_ed >= 0.0 and st > last_ed:
#             idle_width = st - last_ed
#             setup_width = min(idle_width, FIXED_SETUP_TIME)
#             bottom_y = y_center - bar_height / 2.0
#
#             # 绘制切换时间条，加上外框（已修正）
#             ax.barh(bottom_y, setup_width, left=last_ed, height=bar_height,
#                     color=setup_color,
#                     edgecolor=SETUP_EDGE_COLOR,
#                     linewidth=SETUP_LINE_WIDTH,
#                     hatch=setup_hatch, align='edge', zorder=2, clip_on=True)
#
#     # --- 5b) 绘制任务条 (Task Bars) 和标签 ---
#     for t in schedule:
#         new_resource_key = t.get('new_resource_key')
#         base = resource_base.get(new_resource_key, 0.0)
#         assigned = task_row_assignment.get(id(t), 0)
#         y_center = base + assigned
#
#         st = float(t.get('start', 0) or 0)
#         ed = float(t.get('end', 0) or st + 1)
#         width = max(ed - st, 0.1)
#         p = int(t.get('product', 0))
#         s_cn = t.get('stage', '总装')
#
#         base_color = product_colors.get(p, (0.7, 0.7, 0.7))
#         bottom_y = y_center - bar_height / 2.0
#
#         ax.barh(bottom_y, width, left=st, height=bar_height, color=base_color,
#                 edgecolor=(0.2, 0.2, 0.2), linewidth=1.2, align='edge', zorder=4, clip_on=True)
#
#         seq = prod_seq_map.get(id(t), '?')
#         s_short = stage_to_short_en.get(s_cn, s_cn[:3])
#         label = f"P{p + 1}-{s_short}-{seq}"
#         center_x = st + width * 0.5
#
#         brightness = 0.299 * base_color[0] + 0.587 * base_color[1] + 0.114 * base_color[2]
#         text_color = 'white' if brightness < 0.6 else 'black'
#
#         ax.text(center_x, y_center, label, ha='center', va='center',
#                 fontsize=10, color=text_color, fontname=preferred_font,
#                 fontweight='bold', zorder=6, clip_on=True)
#
#     # 资源分隔线
#     cum_y = 0.0
#     for new_resource_key in sorted_new_resource_keys:
#         rows_used = resource_row_counts.get(new_resource_key, 0)
#         if cum_y > 0:
#             sep_y = cum_y - 0.25
#             ax.axhline(y=sep_y, color='gray', linestyle='-', linewidth=0.5, alpha=0.6)
#         cum_y += rows_used + 0.5
#
#     # 坐标轴设置
#     ax.set_yticks(y_ticks)
#     ax.set_yticklabels(y_labels, fontsize=10, fontname=preferred_font)
#
#     ax.invert_yaxis()
#     ax.set_xlabel("Time (Unit)", fontsize=12, fontname=preferred_font)
#     ax.set_ylabel("Resource (Stage / Sub-line)", fontsize=12, fontname=preferred_font)
#     ax.set_title(title, fontsize=16, pad=16, fontweight='bold', fontname=preferred_font)
#
#     # 图例 (Legend)
#     prod_handles = [mpatches.Patch(facecolor=product_colors[p], edgecolor='black',
#                                    label=f"Product {p + 1}") for p in range(num_products)]
#
#     # 切换时间图例也带有外框
#     setup_handle = mpatches.Patch(facecolor=setup_color,
#                                   edgecolor=SETUP_EDGE_COLOR,
#                                   hatch=setup_hatch,
#                                   label="Setup Time (Switching)")
#     prod_handles.append(setup_handle)
#
#     legend = ax.legend(handles=prod_handles, title="Legend", loc='lower right',
#                        fontsize=10, title_fontsize=11, prop={'family': preferred_font})
#     for text in legend.get_texts():
#         text.set_fontname(preferred_font)
#     legend.get_title().set_fontname(preferred_font)
#
#     # X/Y 轴范围
#     max_end = max((float(t.get('end', 0) or 0) for t in schedule), default=0)
#     if max_end <= 0:
#         max_end = max((float(t.get('start', 0) or 0) + 1 for t in schedule), default=1)
#     x_margin_right = max_end * 0.05
#     ax.set_xlim(0, max_end + x_margin_right)
#     ax.set_ylim(-0.5, y_limit + 0.5)
#
#     plt.tight_layout()
#     plt.show()
def draw_gantt(schedule, title="Optimal Schedule Gantt Chart", global_resource_map=None, save_path=None):
    """
    绘制甘特图，具备所有优化功能。
    - 🚨 新增：支持将甘特图数据（包括任务、切换时间和空闲时间）导出为 CSV。
    """
    if not schedule:
        print("schedule is empty")
        return

    # --- 参数定义 ---
    FIXED_SETUP_TIME = 2.0  # 产品换型时间惩罚值

    # --- 阶段定义、缩写、前缀和顺序映射 ---
    stages_order_cn = ["采购", "部装", "总装"]
    stages_order_index = {stage: i for i, stage in enumerate(stages_order_cn)}
    stage_to_prefix = {"采购": "P", "部装": "S", "总装": "F"}
    stage_prefix_to_cn = {'P': '采购', 'S': '部装', 'F': '总装'}
    stage_to_short_en = {"采购": "Prc", "部装": "Sub", "总装": "Fin"}

    # --- 字体设置 ---
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    if "SimHei" in available_fonts:
        preferred_font = "SimHei"
    elif "Microsoft YaHei" in available_fonts:
        preferred_font = "Microsoft YaHei"
    elif "Times New Roman" in available_fonts:
        preferred_font = "Times New Roman"
    else:
        preferred_font = 'sans-serif'

    plt.rcParams['axes.unicode_minus'] = False

    # --- 1) 资源编号映射 ---
    if global_resource_map is not None and isinstance(global_resource_map, dict):
        resource_new_id_map = global_resource_map
    else:
        resource_new_id_map = {}
        for t in schedule:
            original_resource_id = t.get('resource', 0)
            stage = t.get('stage', '总装')
            resource_new_id_map[original_resource_id] = {'new_id': original_resource_id, 'prefix': 'R', 'stage': stage}

    # --- 2) 计算产品内部序号 prod_seq_map ---
    prod_tasks = {}
    for t in schedule:
        try:
            p = int(t.get('product', 0))
        except:
            p = 0
        prod_tasks.setdefault(p, []).append(t)

    prod_seq_map = {}
    for p, tasks in prod_tasks.items():
        tasks_sorted = sorted(tasks, key=lambda x: (
            stages_order_cn.index(x.get('stage', '总装')) if x.get('stage', '总装') in stages_order_cn else 99,
            float(x.get('start', 0) or 0)
        ))
        for idx, t in enumerate(tasks_sorted, start=1):
            prod_seq_map[id(t)] = idx

    # --- 3) 为资源分配子行（AssignedRow）和记录切换时间 ---
    resource_tasks = {}
    for t in schedule:
        original_resource_id = t.get('resource', 0)

        new_info = resource_new_id_map.get(original_resource_id)
        if new_info is None:
            new_info = {'new_id': original_resource_id, 'prefix': 'R', 'stage': t.get('stage', '总装')}

        new_resource_key = (new_info['prefix'], new_info['new_id'])
        resource_tasks.setdefault(new_resource_key, []).append(t)
        t['new_resource_key'] = new_resource_key

    task_row_assignment = {}

    def resource_key_sort_key(new_resource_key):
        prefix, new_id = new_resource_key
        stage_cn = stage_prefix_to_cn.get(prefix, '总装')
        order_index = stages_order_index.get(stage_cn, 99)
        return (order_index, new_id)

    sorted_new_resource_keys = sorted(resource_tasks.keys(), key=resource_key_sort_key)

    last_end_time = {}

    for new_resource_key in sorted_new_resource_keys:
        tasks_on_resource = resource_tasks[new_resource_key]
        tasks_sorted = sorted(tasks_on_resource, key=lambda x: float(x.get('start', 0) or 0))
        rows_end_time = []

        for task in tasks_sorted:
            st = float(task.get('start', 0) or 0)
            ed = float(task.get('end', 0) or st + 1)
            assigned_row = -1

            for idx in range(len(rows_end_time)):
                if rows_end_time[idx] <= st:
                    assigned_row = idx
                    last_end_time[id(task)] = rows_end_time[idx]
                    rows_end_time[idx] = ed
                    break
            if assigned_row == -1:
                assigned_row = len(rows_end_time)
                last_end_time[id(task)] = -1
                rows_end_time.append(ed)
            task_row_assignment[id(task)] = assigned_row

    # --- 4) 构建 Y 位置映射和标签 ---
    y_ticks = []
    y_labels = []
    current_y = 0.0
    resource_row_counts = {}
    resource_base = {}

    for new_resource_key in sorted_new_resource_keys:
        prefix, new_id = new_resource_key
        tasks_on_resource = resource_tasks.get(new_resource_key, [])
        rows_used = max((task_row_assignment.get(id(t), 0) for t in tasks_on_resource), default=0) + 1
        resource_row_counts[new_resource_key] = rows_used
        resource_base[new_resource_key] = current_y
        for r in range(rows_used):
            y_center = current_y + r
            y_ticks.append(y_center)
            if r == 0:
                y_labels.append(f"{prefix}{new_id}")
            else:
                y_labels.append(f"{prefix}{new_id}-{r + 1}")
        current_y += rows_used + 0.5
    y_limit = current_y

    # ----------------------------------------------------
    # --- 6) 🚨 核心功能：数据收集与CSV保存 ---
    # ----------------------------------------------------
    if save_path:
        data_records = []

        for t in schedule:
            new_resource_key = t.get('new_resource_key')
            prefix, new_id = new_resource_key
            base = resource_base.get(new_resource_key, 0.0)
            assigned = task_row_assignment.get(id(t), 0)
            y_center = base + assigned  # Y轴中心坐标

            st = float(t.get('start', 0) or 0)
            ed = float(t.get('end', 0) or st + 1)
            last_ed = last_end_time.get(id(t), -1.0)

            # --- 6A. 记录任务数据 (Type: Task) ---
            data_records.append({
                'Type': 'Task',
                'Task_ID': t.get('task'),
                'Product_ID': int(t.get('product', 0)) + 1,
                'Stage': t.get('stage'),
                'Resource_Original_ID': t.get('resource'),
                'Resource_Key': f"{prefix}{new_id}",
                'Subline': assigned + 1,
                'Y_Position_Center': y_center,
                'Start_Time': st,
                'End_Time': ed,
                'Duration': ed - st,
                'is_Setup': 0
            })

            # --- 6B. 记录切换/空闲时间数据 (Type: Setup/Idle) ---
            if last_ed >= 0.0 and st > last_ed:
                idle_width = st - last_ed
                setup_width = min(idle_width, FIXED_SETUP_TIME)

                # 1. 切换时间 (Setup Time) Block
                if setup_width > 0:
                    data_records.append({
                        'Type': 'Setup',
                        'Task_ID': f"SETUP_{t.get('task')}",
                        'Product_ID': int(t.get('product', 0)) + 1,
                        'Stage': t.get('stage'),
                        'Resource_Original_ID': t.get('resource'),
                        'Resource_Key': f"{prefix}{new_id}",
                        'Subline': assigned + 1,
                        'Y_Position_Center': y_center,
                        'Start_Time': last_ed,
                        'End_Time': last_ed + setup_width,
                        'Duration': setup_width,
                        'is_Setup': 1
                    })

                # 2. 额外等待时间 (Idle Time) Block
                idle_only_width = idle_width - setup_width
                if idle_only_width > 0:
                    data_records.append({
                        'Type': 'Idle',
                        'Task_ID': f"IDLE_{t.get('task')}",
                        'Product_ID': int(t.get('product', 0)) + 1,
                        'Stage': t.get('stage'),
                        'Resource_Original_ID': t.get('resource'),
                        'Resource_Key': f"{prefix}{new_id}",
                        'Subline': assigned + 1,
                        'Y_Position_Center': y_center,
                        'Start_Time': last_ed + setup_width,
                        'End_Time': st,
                        'Duration': idle_only_width,
                        'is_Setup': 0
                    })

        # 保存到CSV
        df = pd.DataFrame(data_records)
        df = df.sort_values(by=['Y_Position_Center', 'Start_Time']).reset_index(drop=True)

        try:
            # 使用 utf-8-sig 编码兼容 Excel 和 MATLAB 的中文显示
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"\n--- 成功将调度数据保存到: {save_path} ---")
        except Exception as e:
            print(f"\n--- 警告: 保存CSV文件时出错: {e} ---")

    # ----------------------------------------------------
    # --- 7) 绘制图表 (与之前版本一致) ---
    # ----------------------------------------------------
    bar_height = 0.8
    fig_height = max(8, len(y_ticks) * 0.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    # ... (绘图代码保持不变) ...

    # 产品颜色
    product_colors = {}
    try:
        products = set(int(t.get('product', 0)) for t in schedule if 'product' in t)
        num_products = max(products) + 1 if products else 1
        color_scheme = [(0.121, 0.466, 0.705), (1.0, 0.498, 0.054), (0.172, 0.627, 0.172), (0.839, 0.152, 0.156),
                        (0.580, 0.403, 0.741), (0.682, 0.780, 0.909), (0.549, 0.337, 0.294), (0.890, 0.466, 0.760),
                        (0.498, 0.498, 0.498), (0.737, 0.741, 0.133)]
        for p in range(num_products):
            product_colors[p] = color_scheme[p % len(color_scheme)]
    except:
        num_products = 0
        pass

    # --- 7a) 绘制切换时间条 (Setup Bars) ---
    setup_color = '#C7C7C7'
    setup_hatch = '///'
    SETUP_EDGE_COLOR = (0.2, 0.2, 0.2)
    SETUP_LINE_WIDTH = 1.2

    for t in schedule:
        new_resource_key = t.get('new_resource_key')
        base = resource_base.get(new_resource_key, 0.0)
        assigned = task_row_assignment.get(id(t), 0)
        y_center = base + assigned

        st = float(t.get('start', 0) or 0)
        last_ed = last_end_time.get(id(t), -1.0)

        if last_ed >= 0.0 and st > last_ed:
            idle_width = st - last_ed
            setup_width = min(idle_width, FIXED_SETUP_TIME)
            bottom_y = y_center - bar_height / 2.0

            # 绘制切换时间条
            ax.barh(bottom_y, setup_width, left=last_ed, height=bar_height,
                    color=setup_color,
                    edgecolor=SETUP_EDGE_COLOR,
                    linewidth=SETUP_LINE_WIDTH,
                    hatch=setup_hatch, align='edge', zorder=2, clip_on=True)

    # --- 7b) 绘制任务条 (Task Bars) 和标签 ---
    for t in schedule:
        new_resource_key = t.get('new_resource_key')
        base = resource_base.get(new_resource_key, 0.0)
        assigned = task_row_assignment.get(id(t), 0)
        y_center = base + assigned

        st = float(t.get('start', 0) or 0)
        ed = float(t.get('end', 0) or st + 1)
        width = max(ed - st, 0.1)
        p = int(t.get('product', 0))
        s_cn = t.get('stage', '总装')

        base_color = product_colors.get(p, (0.7, 0.7, 0.7))
        bottom_y = y_center - bar_height / 2.0

        ax.barh(bottom_y, width, left=st, height=bar_height, color=base_color,
                edgecolor=(0.2, 0.2, 0.2), linewidth=1.2, align='edge', zorder=4, clip_on=True)

        seq = prod_seq_map.get(id(t), '?')
        s_short = stage_to_short_en.get(s_cn, s_cn[:3])
        label = f"P{p + 1}-{s_short}-{seq}"
        center_x = st + width * 0.5

        brightness = 0.299 * base_color[0] + 0.587 * base_color[1] + 0.114 * base_color[2]
        text_color = 'white' if brightness < 0.6 else 'black'

        ax.text(center_x, y_center, label, ha='center', va='center',
                fontsize=10, color=text_color, fontname=preferred_font,
                fontweight='bold', zorder=6, clip_on=True)

    # 资源分隔线 (保持不变)
    cum_y = 0.0
    for new_resource_key in sorted_new_resource_keys:
        rows_used = resource_row_counts.get(new_resource_key, 0)
        if cum_y > 0:
            sep_y = cum_y - 0.25
            ax.axhline(y=sep_y, color='gray', linestyle='-', linewidth=0.5, alpha=0.6)
        cum_y += rows_used + 0.5

    # 坐标轴设置
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=10, fontname=preferred_font)

    ax.invert_yaxis()
    ax.set_xlabel("Time (Unit)", fontsize=12, fontname=preferred_font)
    ax.set_ylabel("Resource (Stage / Sub-line)", fontsize=12, fontname=preferred_font)
    ax.set_title(title, fontsize=16, pad=16, fontweight='bold', fontname=preferred_font)

    # 图例 (Legend)
    prod_handles = [mpatches.Patch(facecolor=product_colors[p], edgecolor='black',
                                   label=f"Product {p + 1}") for p in range(num_products)]

    setup_handle = mpatches.Patch(facecolor=setup_color,
                                  edgecolor=SETUP_EDGE_COLOR,
                                  hatch=setup_hatch,
                                  label="Setup Time (Switching)")
    prod_handles.append(setup_handle)

    legend = ax.legend(handles=prod_handles, title="Legend", loc='lower right',
                       fontsize=10, title_fontsize=11, prop={'family': preferred_font})
    for text in legend.get_texts():
        text.set_fontname(preferred_font)
    legend.get_title().set_fontname(preferred_font)

    # X/Y 轴范围
    max_end = max((float(t.get('end', 0) or 0) for t in schedule), default=0)
    if max_end <= 0:
        max_end = max((float(t.get('start', 0) or 0) + 1 for t in schedule), default=1)
    x_margin_right = max_end * 0.05
    ax.set_xlim(0, max_end + x_margin_right)
    ax.set_ylim(-0.5, y_limit + 0.5)

    plt.tight_layout()
    plt.show()
# def draw_gantt(schedule, config, title="最优调度方案甘特图", pattern_style="diagonal"):
#     """
#     顶刊学术风格甘特图（优化排版）
#     - 条形高度、行间距、阶段分隔优化
#     - 条内显示“产品号-工序号”，如 1-1
#     - 产品颜色+阶段纹理区分
#     - 条形文字描边增强可读性
#     - 阶段分隔线明确，图例整齐
#     """
#     import re
#     import matplotlib.pyplot as plt
#     import matplotlib.patches as mpatches
#     import matplotlib.patheffects as path_effects
#     import numpy as np
#     import colorsys
#
#     if not schedule:
#         print("schedule 为空")
#         return
#
#     # ---- 阶段顺序与纹理 ----
#     stages_order = ["采购", "部装", "总装"]
#     stage_hatches = {
#         "采购": "////",
#         "部装": {"diagonal": "//", "dots": "..", "cross": "xx"}.get(pattern_style, ".."),
#         "总装": None
#     }
#
#     # ---- 字体和条高 ----
#     fontsize_label = 10
#     fontsize_yticks = 9
#     fontsize_title = 14
#     bar_height = 0.65
#     row_spacing = 0.9
#     stage_spacing = 0.5
#
#     # ---- 颜色工具函数 ----
#     def clamp(x, a=0.0, b=1.0):
#         return max(a, min(b, x))
#
#     def rgb_to_hls(rgb):
#         return colorsys.rgb_to_hls(*rgb)
#
#     def hls_to_rgb(hls):
#         return colorsys.hls_to_rgb(*hls)
#
#     def adjust_luminance(rgb, factor):
#         h, l, s = rgb_to_hls(rgb)
#         l = clamp(l * factor, 0.0, 1.0)
#         return hls_to_rgb((h, l, s))
#
#     def darken(rgb, factor=0.5):
#         return adjust_luminance(rgb, factor)
#
#     # ---- 排序规则 ----
#     stage_index = {s: i for i, s in enumerate(stages_order)}
#     def extract_task_number(task_field):
#         if task_field is None:
#             return None
#         try:
#             return int(task_field)
#         except Exception:
#             pass
#         m = re.search(r'(\d+)', str(task_field))
#         if m:
#             return int(m.group(1))
#         return None
#     def general_sort_key(x):
#         s = x.get('stage', '')
#         sidx = stage_index.get(s, len(stages_order))
#         return (sidx, x.get('start', 0), extract_task_number(x.get('task', None)) or 0)
#
#     # ---- 产品颜色 ----
#     try:
#         num_products = getattr(config, "num_products", None)
#         if num_products is None:
#             num_products = max(int(t.get('product', 0)) for t in schedule) + 1
#     except Exception:
#         num_products = max(
#             (int(t.get('product', 0)) if str(t.get('product', '')).isdigit() else 0) for t in schedule) + 1
#
#     cmap = plt.get_cmap("tab10")
#     product_base_colors = {p: tuple(cmap(p % 10)[:3]) for p in range(num_products)}
#     product_edge_colors = {p: darken(product_base_colors[p], 0.5) for p in product_base_colors}
#
#     # ---- 生成产品工序序号 ----
#     prod_tasks = {}
#     for t in schedule:
#         try:
#             p = int(t.get('product', 0))
#         except Exception:
#             p = 0
#         prod_tasks.setdefault(p, []).append(t)
#
#     prod_seq_map = {}
#     for p, tasks in prod_tasks.items():
#         tasks_sorted = sorted(tasks, key=general_sort_key)
#         for idx, t in enumerate(tasks_sorted, start=1):
#             key = t.get('task')
#             if key is None:
#                 key = f"{t.get('start',0)}_{t.get('end',0)}_{t.get('resource',-1)}"
#             prod_seq_map[(int(p), str(key))] = idx
#
#     # ---- 按(stage,resource)分行避免重叠 ----
#     sr_tasks = {}
#     for t in schedule:
#         s = t.get('stage', '')
#         if s not in stages_order:
#             s = "总装"
#         try:
#             res_label = int(t.get('resource', None))
#         except Exception:
#             res_label = str(t.get('resource', None))
#         sr_tasks.setdefault((s, res_label), []).append(t)
#
#     for k in sr_tasks:
#         sr_tasks[k].sort(key=lambda x: (x.get('start',0), extract_task_number(x.get('task',None)) or 0))
#
#     y_rows = {}
#     for (s,r), tasks in sr_tasks.items():
#         rows_end, rows_tasks = [], []
#         for t in tasks:
#             st = float(t.get('start',0) or 0)
#             ed = float(t.get('end',0) or st+1)
#             placed = False
#             for i, last_end in enumerate(rows_end):
#                 if st >= last_end:
#                     rows_end[i] = ed
#                     rows_tasks[i].append(t)
#                     placed = True
#                     break
#             if not placed:
#                 rows_end.append(ed)
#                 rows_tasks.append([t])
#         y_rows[(s,r)] = rows_tasks
#
#     # ---- 分配 y 位置 ----
#     y_positions, y_ticks, y_labels = {}, [], []
#     y_base = 0.0
#     stage_bottoms = {}
#     for s in stages_order:
#         res_list = sorted([r for (ss,r) in y_rows.keys() if ss==s], key=lambda x: (isinstance(x,str), x))
#         stage_start = y_base
#         for r in res_list:
#             rows_for_r = y_rows.get((s,r),[])
#             for row_idx, row in enumerate(rows_for_r):
#                 for t in row:
#                     y_positions[id(t)] = y_base
#                 y_ticks.append(y_base)
#                 if len(rows_for_r)>1:
#                     y_labels.append(f"{s}-R{r}-{row_idx+1}")
#                 else:
#                     y_labels.append(f"{s}-R{r}")
#                 y_base += row_spacing
#         y_base += stage_spacing
#         stage_bottoms[s] = y_base
#
#     total_rows = len(y_ticks)
#     height = max(6, total_rows*0.4)
#     fig, ax = plt.subplots(figsize=(14, height))
#     fig.patch.set_facecolor('white')
#     ax.set_facecolor('white')
#     ax.grid(True, axis='x', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)
#
#     # ---- 绘制条形 ----
#     for t in schedule:
#         s = t.get('stage','')
#         if s not in stages_order:
#             s = "总装"
#         y = y_positions.get(id(t))
#         if y is None: continue
#         st = float(t.get('start',0) or 0)
#         ed = float(t.get('end',0) or st+1)
#         width = max(ed-st,0.1)
#         try:
#             p = int(t.get('product',0))
#         except Exception:
#             p=0
#         base_color = product_base_colors.get(p,(0.75,0.75,0.75))
#         edge_color = product_edge_colors.get(p,(0.25,0.25,0.25))
#         hatch_style = stage_hatches.get(s,None)
#         ax.barh(y, width, left=st, height=bar_height, color=base_color,
#                 edgecolor=edge_color, linewidth=1.0, hatch=hatch_style,
#                 zorder=4, clip_on=True)
#
#         # 工序编号标签：显示 产品号-工序号
#         key = t.get('task')
#         if key is None:
#             res_label = t.get('resource',-1)
#             key = f"{st}_{ed}_{res_label}"
#         seq = prod_seq_map.get((p,str(key)))
#         label = f"{p+1}-{seq}" if seq is not None else f"{p+1}-?"
#         min_label_width = 1.0
#         center_x = st + width*0.5
#         if width >= min_label_width:
#             txt = ax.text(center_x, y, label, ha='center', va='center',
#                           fontsize=fontsize_label, fontweight='semibold', color='white', zorder=8)
#             txt.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='black'),
#                                   path_effects.Normal()])
#         else:
#             txt = ax.text(ed+0.05, y, label, ha='left', va='center',
#                           fontsize=fontsize_label*0.9, color='black', zorder=8)
#
#     # ---- 阶段分隔线 ----
#     for s in stages_order[:-1]:
#         ax.axhline(stage_bottoms[s], color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
#
#     # ---- 坐标轴 ----
#     ax.set_yticks(y_ticks)
#     ax.set_yticklabels(y_labels, fontsize=fontsize_yticks)
#     ax.invert_yaxis()
#     ax.set_xlabel("时间", fontsize=11, labelpad=8)
#     ax.set_ylabel("资源", fontsize=11, labelpad=8)
#     ax.set_title(title, fontsize=fontsize_title, pad=16, fontweight='bold')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_color('gray')
#     ax.spines['bottom'].set_color('gray')
#     ax.tick_params(axis='x', which='major', labelsize=9)
#     ax.tick_params(axis='y', which='major', labelsize=fontsize_yticks)
#
#     # ---- 图例 ----
#     prod_handles = [mpatches.Patch(facecolor=product_base_colors[p],
#                                    edgecolor=product_edge_colors[p],
#                                    label=f"产品{p+1}") for p in range(num_products)]
#     stage_handles = [
#         mpatches.Patch(facecolor='white', edgecolor='black', hatch=stage_hatches['采购'], label='采购'),
#         mpatches.Patch(facecolor='white', edgecolor='black', hatch=stage_hatches['部装'], label='部装'),
#         mpatches.Patch(facecolor='white', edgecolor='black', hatch=None, label='总装')
#     ]
#     first_legend = ax.legend(handles=prod_handles, title="产品",
#                              bbox_to_anchor=(1.02,1), loc='upper left',
#                              frameon=True, fancybox=True, fontsize=9, title_fontsize=10)
#     ax.add_artist(first_legend)
#     ax.legend(handles=stage_handles, title="阶段",
#               bbox_to_anchor=(1.02,0.5), loc='upper left',
#               frameon=True, fancybox=True, fontsize=9, title_fontsize=10)
#
#     # ---- X轴范围 ----
#     min_start = min((float(t.get('start',0) or 0) for t in schedule), default=0)
#     max_end = max((float(t.get('end',0) or 0) for t in schedule), default=0)
#     if max_end <= 0:
#         max_end = max((float(t.get('start',0) or 0)+1 for t in schedule), default=1)
#     x_margin_right = max_end*0.02
#     ax.set_xlim(0, max_end+x_margin_right)
#
#     plt.tight_layout()
#     plt.subplots_adjust(right=0.85)
#     plt.show()



# def draw_gantt(schedule, config, title="最优调度方案甘特图"):
#     """
#     显示上保证：每个产品内部按阶段顺序（采购->部装->总装）。
#     采购阶段内部按显式序号（seq/order/...）优先；若无则从 task 提取数字。
#     即使采购任务同时开始（start 相同），也会按该序号/任务号从上到下排列。
#     """
#     import re
#     import matplotlib.pyplot as plt
#     import matplotlib.patches as mpatches
#
#     if not schedule:
#         print("schedule 为空")
#         return
#
#     # 可识别的显式序号字段
#     order_fields = ['seq', 'order', 'proc_seq', 'proc_order', 'order_in_stage']
#
#     # 基本设定
#     num_products = getattr(config, "num_products", max(int(t['product']) for t in schedule) + 1)
#     cmap = plt.get_cmap("tab20")
#     product_colors = {p: cmap(p % 20) for p in range(num_products)}
#     stages_order = ["采购", "部装", "总装"]
#     stage_index = {s: i for i, s in enumerate(stages_order)}
#
#     # 按 product 分组
#     prod_tasks = {p: [] for p in range(num_products)}
#     for t in schedule:
#         p = int(t.get('product', 0))
#         prod_tasks.setdefault(p, []).append(t)
#
#     # 尝试从 task 字段提取整数编号（"T62"->62）
#     def extract_task_number(task_field):
#         if task_field is None:
#             return None
#         try:
#             return int(task_field)
#         except Exception:
#             pass
#         m = re.search(r'(\d+)', str(task_field))
#         if m:
#             return int(m.group(1))
#         return None
#
#     def get_order_value(task):
#         for f in order_fields:
#             if f in task and task[f] is not None:
#                 try:
#                     return float(task[f])
#                 except Exception:
#                     try:
#                         return float(str(task[f]).strip())
#                     except Exception:
#                         return None
#         return None
#
#     # 排序规则：确保采购阶段按显式序号/任务号排序（优先级），否则按 start
#     prod_tasks_sorted = {}
#     for p, tasks in prod_tasks.items():
#         def sort_key(x):
#             stg = x.get('stage', '')
#             sidx = stage_index.get(stg, len(stages_order))
#             if stg == "采购":
#                 ordv = get_order_value(x)
#                 if ordv is not None:
#                     # 用显式序号排序；作为次键保留 start，最后用 task 编号防止完全相同键导致不稳定顺序
#                     tnum = extract_task_number(x.get('task', None)) or 0
#                     return (sidx, 0, ordv, x.get('start', 0), tnum)
#                 tnum = extract_task_number(x.get('task', None))
#                 if tnum is not None:
#                     # 用 task 编号排序（当作采购内部顺序）
#                     return (sidx, 1, tnum, x.get('start', 0))
#                 # 最后退回到 start 排序
#                 return (sidx, 2, x.get('start', 0), extract_task_number(x.get('task', None)) or 0)
#             else:
#                 # 其他阶段按 start 排序
#                 return (sidx, x.get('start', 0), extract_task_number(x.get('task', None)) or 0)
#         prod_tasks_sorted[p] = sorted(tasks, key=sort_key)
#
#     # 绘图准备
#     total_rows = sum(len(prod_tasks_sorted[p]) for p in range(num_products)) + (num_products - 1)
#     height = max(6, total_rows * 0.45)
#     fig, ax = plt.subplots(figsize=(14, height))
#
#     yticks = []
#     ytick_labels = []
#     y_base = 0
#
#     # 绘制每个产品（每个任务一行，产品间空一行）
#     for p in range(num_products):
#         tasks = prod_tasks_sorted.get(p, [])
#         if not tasks:
#             ax.axhline(y=y_base, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
#             yticks.append(y_base)
#             ytick_labels.append(f"产品{p + 1} — 空")
#             y_base += 1
#             y_base += 1
#             continue
#
#         prev_stage = None
#         for t in tasks:
#             st, ed = t.get('start', 0), t.get('end', 0)
#             s = t.get('stage', '')
#             y = y_base
#             color = product_colors.get(p, (0.7, 0.7, 0.7))
#
#             ax.axhline(y=y, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
#             ax.barh(y, ed - st, left=st, height=0.8,
#                     color=color, edgecolor='black', linewidth=0.6, zorder=2)
#
#             label = f"T{t.get('task','?')} (M{int(t.get('resource', -1))})"
#             ax.text((st + ed) / 2, y, label, ha='center', va='center', color='black',
#                     fontsize=8, clip_on=True, zorder=3)
#
#             yticks.append(y)
#             if s != prev_stage:
#                 ytick_labels.append(f"产品{p + 1} — {s}")
#                 prev_stage = s
#             else:
#                 ytick_labels.append("")
#
#             y_base += 1
#
#         y_base += 1  # 产品间空行
#
#     # 坐标轴与样式
#     ax.set_yticks(yticks)
#     ax.set_yticklabels(ytick_labels)
#     ax.invert_yaxis()
#     ax.set_xlabel("时间")
#     ax.set_title(title, fontsize=13, pad=12)
#     ax.xaxis.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray')
#     ax.set_axisbelow(True)
#     for spine in ["top", "right"]:
#         ax.spines[spine].set_visible(False)
#
#     # 图例
#     legend_handles = [mpatches.Patch(color=product_colors[p], label=f"产品{p + 1}") for p in range(num_products)]
#     ax.legend(handles=legend_handles, title="产品", bbox_to_anchor=(1.02, 1),
#               loc='upper left', borderaxespad=0)
#
#     # X 轴范围
#     max_end = max((t.get('end', 0) for t in schedule), default=0)
#     ax.set_xlim(0, max_end + max(1, 0.05 * max_end))
#
#     plt.tight_layout()
#     plt.show()



def plot_fitness_progress(all_fitness, global_min_time, global_min_cost):
    iterations = list(range(len(all_fitness)))

    # 确保数据是 float 类型
    avg_time = [sum(float(f[0]) for f in gen) / len(gen) for gen in all_fitness]
    avg_cost = [sum(float(f[1]) for f in gen) / len(gen) for gen in all_fitness]
    min_time = [min(float(f[0]) for f in gen) for gen in all_fitness]
    min_cost = [min(float(f[1]) for f in gen) for gen in all_fitness]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("迭代次数")
    ax1.set_ylabel("完工时间", color='tab:blue')
    ax1.plot(iterations, global_min_time, label="全局最优完工时间", color='tab:blue', linestyle='-')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("总成本", color='tab:red')
    ax2.plot(iterations, global_min_cost, label="全局最优总成本", color='tab:red', linestyle='-')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.suptitle("适应度演化趋势（双目标）")
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    plt.grid(True)
    plt.show()

def plot_q_table(q_table, actions):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    data = []
    for state, values in q_table.items():
        row = {"state": state}
        for a, v in zip(actions, values):
            row[a] = v
        data.append(row)

    df = pd.DataFrame(data).set_index("state")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("强化学习 Q 表（leader 策略）")
    plt.ylabel("状态（离散编码）")
    plt.xlabel("动作")
    plt.tight_layout()
    plt.show()


def plot_archive_size(archive_size_history):
    plt.figure()
    plt.plot(archive_size_history, label="档案大小", color='purple')
    plt.xlabel("迭代次数")
    plt.ylabel("解的个数")
    plt.title("档案大小变化趋势")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 在 main.py 中修改以下部分

if __name__ == "__main__":
    config = Config()
    print(config.machine_task_map[(0, 0)])

    print("开始运行 MOPSO 多目标调度优化 (RL增强版)...\n")

    # 现在 run_mopso 返回三个对象
    result, best_fitness_history, all_generational_best_history = run_mopso(config, total_iterations=50)

    # 直接从result字典中获取结果
    ca = result['convergence_archive']
    da = result['diversity_archive']
    all_fitness = result['all_fitness']
    global_min_time = result['global_min_time_list']
    global_min_cost = result['global_min_cost_list']
    archive_size_history = result['archive_size_history']
    q_table = result['q_table']
    hv_history = result['hv_history']  # 这里应该是列表
    final_sp = result['final_sp']
    igd_values = result['igd_values']
    rewards_history = result['rewards_history']
    q_values_per_iteration = result['q_values_per_iteration']

    pareto_front = da
    print("\n优化完成，非支配解个数:", len(pareto_front))
    # if not best_fitness_history:
    #     print("警告：best_fitness_history 为空，跳过累计最优趋势图绘制。")
    # else:
    #     # 获取历史最优值
    #     best_times = [t for (t, c) in best_fitness_history]
    #     best_costs = [c for (t, c) in best_fitness_history]
    #
    #     # 计算历史前缀最优（累计最小值）
    #     prefix_best_times = np.minimum.accumulate(best_times)
    #     prefix_best_costs = np.minimum.accumulate(best_costs)
    #
    #     # 创建图表
    #     plt.figure(figsize=(10, 6))
    #
    #     # 使用双Y轴展示时间和成本
    #     ax1 = plt.gca()
    #     ax2 = ax1.twinx()
    #
    #     # 绘制累计最优时间（带平滑处理）
    #     if len(prefix_best_times) > 5:
    #         # 使用移动平均平滑曲线
    #         window_size = max(1, len(prefix_best_times) // 20)
    #         smoothed_times = np.convolve(prefix_best_times, np.ones(window_size) / window_size, mode='valid')
    #         ax1.plot(range(window_size, len(prefix_best_times) + 1), smoothed_times,
    #                  'o-', color="#1f77b4", linewidth=2, markersize=6,
    #                  label="累计最优时间(平滑)")
    #     else:
    #         ax1.plot(range(1, len(prefix_best_times) + 1), prefix_best_times,
    #                  'o-', color="#1f77b4", linewidth=2, markersize=6,
    #                  label="累计最优时间")
    #
    #     # 绘制累计最优成本（带平滑处理）
    #     if len(prefix_best_costs) > 5:
    #         window_size = max(1, len(prefix_best_costs) // 20)
    #         smoothed_costs = np.convolve(prefix_best_costs, np.ones(window_size) / window_size, mode='valid')
    #         ax2.plot(range(window_size, len(prefix_best_costs) + 1), smoothed_costs,
    #                  's-', color="#ff7f0e", linewidth=2, markersize=6,
    #                  label="累计最优成本(平滑)")
    #     else:
    #         ax2.plot(range(1, len(prefix_best_costs) + 1), prefix_best_costs,
    #                  's-', color="#ff7f0e", linewidth=2, markersize=6,
    #                  label="累计最优成本")
    #
    #     # 标记全局最优点
    #     min_time_idx = int(np.argmin(prefix_best_times))
    #     min_cost_idx = int(np.argmin(prefix_best_costs))
    #
    #     ax1.plot(min_time_idx + 1, prefix_best_times[min_time_idx],
    #              '*', color='darkblue', markersize=12,
    #              label=f"全局最优时间 ({prefix_best_times[min_time_idx]:.1f})")
    #
    #     ax2.plot(min_cost_idx + 1, prefix_best_costs[min_cost_idx],
    #              'D', color='darkorange', markersize=10,
    #              label=f"全局最优成本 ({prefix_best_costs[min_cost_idx]:.1f})")
    #
    #     # 添加改进区域阴影
    #     for i in range(1, len(prefix_best_times)):
    #         if prefix_best_times[i] < prefix_best_times[i - 1]:
    #             ax1.axvspan(i, i + 1, color="lightblue", alpha=0.3)
    #         if prefix_best_costs[i] < prefix_best_costs[i - 1]:
    #             ax2.axvspan(i, i + 1, color="moccasin", alpha=0.3)
    #
    #     # 添加参考线
    #     if len(prefix_best_times) > 1:
    #         avg_time = float(np.mean(prefix_best_times))
    #         ax1.axhline(avg_time, color='blue', linestyle='--', alpha=0.7,
    #                     label=f"平均时间 ({avg_time:.1f})")
    #
    #         avg_cost = float(np.mean(prefix_best_costs))
    #         ax2.axhline(avg_cost, color='orange', linestyle='--', alpha=0.7,
    #                     label=f"平均成本 ({avg_cost:.1f})")
    #
    #     # 设置坐标轴标签
    #     ax1.set_xlabel("检查点 (每1000次迭代)", fontsize=12, fontweight='bold')
    #     ax1.set_ylabel("最优时间", fontsize=12, fontweight='bold', color="#1f77b4")
    #     ax2.set_ylabel("最优成本", fontsize=12, fontweight='bold', color="#ff7f0e")
    #
    #     # 设置标题
    #     plt.title("累计最优适应度趋势 (时间 vs 成本)", fontsize=14, fontweight='bold', pad=20)
    #
    #     # 设置网格和样式
    #     ax1.grid(True, linestyle='--', alpha=0.6)
    #     ax1.set_axisbelow(True)
    #
    #     # 设置颜色
    #     ax1.tick_params(axis='y', labelcolor="#1f77b4")
    #     ax2.tick_params(axis='y', labelcolor="#ff7f0e")
    #
    #     # 合并图例（仅当存在带标签的artist时显示）
    #     lines1, labels1 = ax1.get_legend_handles_labels()
    #     lines2, labels2 = ax2.get_legend_handles_labels()
    #     if (lines1 or lines2) and (labels1 or labels2):
    #         ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
    #                    bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True,
    #                    fancybox=True, shadow=True)
    #
    #     # 添加数据统计框
    #     stats_text = (
    #         f"时间优化: {prefix_best_times[0]:.1f} → {prefix_best_times[-1]:.1f} ({((prefix_best_times[0] - prefix_best_times[-1]) / max(1e-9, prefix_best_times[0]) * 100):.1f}%)\n"
    #         f"成本优化: {prefix_best_costs[0]:.1f} → {prefix_best_costs[-1]:.1f} ({((prefix_best_costs[0] - prefix_best_costs[-1]) / max(1e-9, prefix_best_costs[0]) * 100):.1f}%)"
    #     )
    #     plt.figtext(0.5, 0.01, stats_text, ha="center", fontsize=10,
    #                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    #
    #     # 调整布局
    #     plt.tight_layout()
    #     plt.subplots_adjust(bottom=0.25)


        # 保存并显示
        # plt.savefig("prefix_best_fitness_enhanced-2.0-q_table1.png", dpi=300, bbox_inches='tight')
        # plt.show()

    # === HV曲线 ===
    # if hv_history:
    #     print("hv_history的类型:", type(hv_history))
    #     if isinstance(hv_history, list) and len(hv_history) > 0:
    #         print("hv_history的前几个元素:", hv_history[:5])
    #     else:
    #         print("hv_history的内容:", hv_history)
    #     # plot_hypervolume_trend(hv_history)
    # else:
    #     print("警告: 未能记录Hypervolume历史。")

    try:
        if pareto_front:
            best_particle = min(pareto_front, key=lambda p: p.fitness[0])
            validate_schedule(best_particle.schedule, config)
            ms, cost = best_particle.fitness
            print(f"最优解之一 (时间最短): 完工时间={ms:.2f}, 总成本={cost:.2f}")

            plot_pareto(pareto_front)

            title = f"Optimal Schedule Gantt Chart (time={ms:.1f},cost={cost:.1f})"

            # 1. 生成全局资源映射
            fixed_resource_map = generate_global_resource_map(config)

            # 2. 调用更新后的 draw_gantt，传入全局映射，并保存CSV文件
            csv_file_path = '11.17best_schedule_for_matlab.csv'  # 你可以自定义文件名
            draw_gantt(
                best_particle.schedule,
                title=title,
                global_resource_map=fixed_resource_map,
                save_path=csv_file_path
            )
        else:
            print("警告：未找到有效解")
    except Exception as e:
        print(f"后处理或绘图时发生错误: {str(e)}")


