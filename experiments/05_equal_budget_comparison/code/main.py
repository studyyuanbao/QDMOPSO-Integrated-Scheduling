import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'
rcParams['axes.unicode_minus'] = False  # 显示负号
from matplotlib import rcParams
from init import Config
from pso_rl_new import run_mopso
import matplotlib.pyplot as plt
from typing import List

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


def draw_gantt(schedule, config, title="最优调度方案甘特图"):
    """
    显示上保证：每个产品内部按阶段顺序（采购->部装->总装）。
    采购阶段内部按显式序号（seq/order/...）优先；若无则从 task 提取数字。
    即使采购任务同时开始（start 相同），也会按该序号/任务号从上到下排列。
    """
    import re
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if not schedule:
        print("schedule 为空")
        return

    # 可识别的显式序号字段
    order_fields = ['seq', 'order', 'proc_seq', 'proc_order', 'order_in_stage']

    # 基本设定
    num_products = getattr(config, "num_products", max(int(t['product']) for t in schedule) + 1)
    cmap = plt.get_cmap("tab20")
    product_colors = {p: cmap(p % 20) for p in range(num_products)}
    stages_order = ["采购", "部装", "总装"]
    stage_index = {s: i for i, s in enumerate(stages_order)}

    # 按 product 分组
    prod_tasks = {p: [] for p in range(num_products)}
    for t in schedule:
        p = int(t.get('product', 0))
        prod_tasks.setdefault(p, []).append(t)

    # 尝试从 task 字段提取整数编号（"T62"->62）
    def extract_task_number(task_field):
        if task_field is None:
            return None
        try:
            return int(task_field)
        except Exception:
            pass
        m = re.search(r'(\d+)', str(task_field))
        if m:
            return int(m.group(1))
        return None

    def get_order_value(task):
        for f in order_fields:
            if f in task and task[f] is not None:
                try:
                    return float(task[f])
                except Exception:
                    try:
                        return float(str(task[f]).strip())
                    except Exception:
                        return None
        return None

    # 排序规则：确保采购阶段按显式序号/任务号排序（优先级），否则按 start
    prod_tasks_sorted = {}
    for p, tasks in prod_tasks.items():
        def sort_key(x):
            stg = x.get('stage', '')
            sidx = stage_index.get(stg, len(stages_order))
            if stg == "采购":
                ordv = get_order_value(x)
                if ordv is not None:
                    # 用显式序号排序；作为次键保留 start，最后用 task 编号防止完全相同键导致不稳定顺序
                    tnum = extract_task_number(x.get('task', None)) or 0
                    return (sidx, 0, ordv, x.get('start', 0), tnum)
                tnum = extract_task_number(x.get('task', None))
                if tnum is not None:
                    # 用 task 编号排序（当作采购内部顺序）
                    return (sidx, 1, tnum, x.get('start', 0))
                # 最后退回到 start 排序
                return (sidx, 2, x.get('start', 0), extract_task_number(x.get('task', None)) or 0)
            else:
                # 其他阶段按 start 排序
                return (sidx, x.get('start', 0), extract_task_number(x.get('task', None)) or 0)
        prod_tasks_sorted[p] = sorted(tasks, key=sort_key)

    # 绘图准备
    total_rows = sum(len(prod_tasks_sorted[p]) for p in range(num_products)) + (num_products - 1)
    height = max(6, total_rows * 0.45)
    fig, ax = plt.subplots(figsize=(14, height))

    yticks = []
    ytick_labels = []
    y_base = 0

    # 绘制每个产品（每个任务一行，产品间空一行）
    for p in range(num_products):
        tasks = prod_tasks_sorted.get(p, [])
        if not tasks:
            ax.axhline(y=y_base, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
            yticks.append(y_base)
            ytick_labels.append(f"产品{p + 1} — 空")
            y_base += 1
            y_base += 1
            continue

        prev_stage = None
        for t in tasks:
            st, ed = t.get('start', 0), t.get('end', 0)
            s = t.get('stage', '')
            y = y_base
            color = product_colors.get(p, (0.7, 0.7, 0.7))

            ax.axhline(y=y, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
            ax.barh(y, ed - st, left=st, height=0.8,
                    color=color, edgecolor='black', linewidth=0.6, zorder=2)

            label = f"T{t.get('task','?')} (M{int(t.get('resource', -1))})"
            ax.text((st + ed) / 2, y, label, ha='center', va='center', color='black',
                    fontsize=8, clip_on=True, zorder=3)

            yticks.append(y)
            if s != prev_stage:
                ytick_labels.append(f"产品{p + 1} — {s}")
                prev_stage = s
            else:
                ytick_labels.append("")

            y_base += 1

        y_base += 1  # 产品间空行

    # 坐标轴与样式
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    ax.invert_yaxis()
    ax.set_xlabel("时间")
    ax.set_title(title, fontsize=13, pad=12)
    ax.xaxis.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # 图例
    legend_handles = [mpatches.Patch(color=product_colors[p], label=f"产品{p + 1}") for p in range(num_products)]
    ax.legend(handles=legend_handles, title="产品", bbox_to_anchor=(1.02, 1),
              loc='upper left', borderaxespad=0)

    # X 轴范围
    max_end = max((t.get('end', 0) for t in schedule), default=0)
    ax.set_xlim(0, max_end + max(1, 0.05 * max_end))

    plt.tight_layout()
    plt.show()



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
    result, best_fitness_history, all_generational_best_history = run_mopso(config, total_iterations=200000)

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
    if not best_fitness_history:
        print("警告：best_fitness_history 为空，跳过累计最优趋势图绘制。")
    else:
        # 获取历史最优值
        best_times = [t for (t, c) in best_fitness_history]
        best_costs = [c for (t, c) in best_fitness_history]

        # 计算历史前缀最优（累计最小值）
        prefix_best_times = np.minimum.accumulate(best_times)
        prefix_best_costs = np.minimum.accumulate(best_costs)

        # 创建图表
        plt.figure(figsize=(10, 6))

        # 使用双Y轴展示时间和成本
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        # 绘制累计最优时间（带平滑处理）
        if len(prefix_best_times) > 5:
            # 使用移动平均平滑曲线
            window_size = max(1, len(prefix_best_times) // 20)
            smoothed_times = np.convolve(prefix_best_times, np.ones(window_size) / window_size, mode='valid')
            ax1.plot(range(window_size, len(prefix_best_times) + 1), smoothed_times,
                     'o-', color="#1f77b4", linewidth=2, markersize=6,
                     label="累计最优时间(平滑)")
        else:
            ax1.plot(range(1, len(prefix_best_times) + 1), prefix_best_times,
                     'o-', color="#1f77b4", linewidth=2, markersize=6,
                     label="累计最优时间")

        # 绘制累计最优成本（带平滑处理）
        if len(prefix_best_costs) > 5:
            window_size = max(1, len(prefix_best_costs) // 20)
            smoothed_costs = np.convolve(prefix_best_costs, np.ones(window_size) / window_size, mode='valid')
            ax2.plot(range(window_size, len(prefix_best_costs) + 1), smoothed_costs,
                     's-', color="#ff7f0e", linewidth=2, markersize=6,
                     label="累计最优成本(平滑)")
        else:
            ax2.plot(range(1, len(prefix_best_costs) + 1), prefix_best_costs,
                     's-', color="#ff7f0e", linewidth=2, markersize=6,
                     label="累计最优成本")

        # 标记全局最优点
        min_time_idx = int(np.argmin(prefix_best_times))
        min_cost_idx = int(np.argmin(prefix_best_costs))

        ax1.plot(min_time_idx + 1, prefix_best_times[min_time_idx],
                 '*', color='darkblue', markersize=12,
                 label=f"全局最优时间 ({prefix_best_times[min_time_idx]:.1f})")

        ax2.plot(min_cost_idx + 1, prefix_best_costs[min_cost_idx],
                 'D', color='darkorange', markersize=10,
                 label=f"全局最优成本 ({prefix_best_costs[min_cost_idx]:.1f})")

        # 添加改进区域阴影
        for i in range(1, len(prefix_best_times)):
            if prefix_best_times[i] < prefix_best_times[i - 1]:
                ax1.axvspan(i, i + 1, color="lightblue", alpha=0.3)
            if prefix_best_costs[i] < prefix_best_costs[i - 1]:
                ax2.axvspan(i, i + 1, color="moccasin", alpha=0.3)

        # 添加参考线
        if len(prefix_best_times) > 1:
            avg_time = float(np.mean(prefix_best_times))
            ax1.axhline(avg_time, color='blue', linestyle='--', alpha=0.7,
                        label=f"平均时间 ({avg_time:.1f})")

            avg_cost = float(np.mean(prefix_best_costs))
            ax2.axhline(avg_cost, color='orange', linestyle='--', alpha=0.7,
                        label=f"平均成本 ({avg_cost:.1f})")

        # 设置坐标轴标签
        ax1.set_xlabel("检查点 (每1000次迭代)", fontsize=12, fontweight='bold')
        ax1.set_ylabel("最优时间", fontsize=12, fontweight='bold', color="#1f77b4")
        ax2.set_ylabel("最优成本", fontsize=12, fontweight='bold', color="#ff7f0e")

        # 设置标题
        plt.title("累计最优适应度趋势 (时间 vs 成本)", fontsize=14, fontweight='bold', pad=20)

        # 设置网格和样式
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_axisbelow(True)

        # 设置颜色
        ax1.tick_params(axis='y', labelcolor="#1f77b4")
        ax2.tick_params(axis='y', labelcolor="#ff7f0e")

        # 合并图例（仅当存在带标签的artist时显示）
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if (lines1 or lines2) and (labels1 or labels2):
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
                       bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True,
                       fancybox=True, shadow=True)

        # 添加数据统计框
        stats_text = (
            f"时间优化: {prefix_best_times[0]:.1f} → {prefix_best_times[-1]:.1f} ({((prefix_best_times[0] - prefix_best_times[-1]) / max(1e-9, prefix_best_times[0]) * 100):.1f}%)\n"
            f"成本优化: {prefix_best_costs[0]:.1f} → {prefix_best_costs[-1]:.1f} ({((prefix_best_costs[0] - prefix_best_costs[-1]) / max(1e-9, prefix_best_costs[0]) * 100):.1f}%)"
        )
        plt.figtext(0.5, 0.01, stats_text, ha="center", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))

        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)

        # 保存并显示
        plt.savefig("prefix_best_fitness_enhanced-2.0-q_table1.png", dpi=300, bbox_inches='tight')
        plt.show()

    # === HV曲线 ===
    if hv_history:
        print("hv_history的类型:", type(hv_history))
        if isinstance(hv_history, list) and len(hv_history) > 0:
            print("hv_history的前几个元素:", hv_history[:5])
        else:
            print("hv_history的内容:", hv_history)
        plot_hypervolume_trend(hv_history)
    else:
        print("警告: 未能记录Hypervolume历史。")

    try:
        if pareto_front:
            best_particle = min(pareto_front, key=lambda p: p.fitness[0])
            validate_schedule(best_particle.schedule, config)
            ms, cost = best_particle.fitness
            print(f"最优解之一 (时间最短): 完工时间={ms:.2f}, 总成本={cost:.2f}")

            plot_pareto(pareto_front)

            title = f"最优调度方案之一（时间={ms:.1f}，成本={cost:.1f}）"
            draw_gantt(best_particle.schedule, config, title=title)
        else:
            print("警告：未找到有效解")
    except Exception as e:
        print(f"后处理或绘图时发生错误: {str(e)}")

