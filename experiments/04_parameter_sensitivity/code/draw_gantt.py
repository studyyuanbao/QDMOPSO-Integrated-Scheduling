import copy
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import os
from init import Config
from pso_rl_new import run_mopso as run_rl_mopso
from mopso_basic import run_mopso as run_plain_mopso
from utils import calculate_hypervolume, calculate_spread, calculate_spacing
import csv
import math
import pandas as pd
from collections import defaultdict
import pathlib
import pickle
import matplotlib.patches as mpatches
from matplotlib import font_manager
from typing import List, Tuple, Any

# 可选算法 (保留原代码逻辑)
try:
    from moabc import run_moabc
except Exception:
    run_moabc = None
try:
    from mopso_basic import run_mopso
except Exception:
    run_mopso = None
try:
    from moaco_2 import run_moaco
except Exception:
    run_moaco = None
try:
    from mogwo import run_mogwo
except Exception:
    run_mogwo = None
try:
    from mowoa import run_mowoa
except Exception:
    run_mowoa = None
# 中文字体设置
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False


# ------------------ 工具函数 (保留原代码逻辑) ------------------
def solution_to_point(sol):
    """从一个解对象或元组中提取 (time, cost)，返回 ((t, c), sol) 或 (None, None)。"""
    if sol is None:
        return None, None

    # 🚨 确保能从 Solution 对象中提取 schedule
    if hasattr(sol, 'schedule') and hasattr(sol, 'fitness'):
        f = sol.fitness
        if isinstance(f, (list, tuple)) and len(f) >= 2:
            try:
                t = float(f[0])
                c = float(f[1])
                if np.isfinite(t) and np.isfinite(c):
                    return (t, c), sol  # 返回点和原始解对象
            except:
                pass

    # 原有的提取 fitness 逻辑
    if hasattr(sol, 'fitness'):
        f = sol.fitness
        if isinstance(f, (list, tuple)) and len(f) >= 2:
            try:
                t = float(f[0])
                c = float(f[1])
                if np.isfinite(t) and np.isfinite(c):
                    return (t, c), sol  # 返回点和原始解对象
            except:
                pass
    return None, None


def nondominated_filter(points):
    """
    输入: list/array of (t,c) 假定目标均为越小越好
    输出: numpy array 的非支配点子集
    """
    # ... (保持原代码不变) ...
    # --- 修正空值检查 ---
    # 检查输入是否为空（同时兼容列表和 NumPy 数组）
    if isinstance(points, np.ndarray):
        # 对于 NumPy 数组，检查 size 是否为 0
        if points.size == 0:
            return np.empty((0, 2))
        pts = points
    elif isinstance(points, (list, tuple)):
        # 对于列表/元组，使用常规的 not 检查
        if not points:
            return np.empty((0, 2))
        pts = np.array(points)
    else:
        # 处理不可识别的类型
        return np.empty((0, 2))
    # --- 修正结束 ---

    n = len(pts)
    is_nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_nd[i]:
            continue
        for j in range(n):
            if j == i:
                continue
            # 如果 j 支配 i
            if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                is_nd[i] = False
                break
    return pts[is_nd]


# ------------------ 新增工具函数：保留 Solution 对象的非支配筛选 ------------------
def nondominated_filter_with_solutions(solutions_with_points: List[Tuple[Tuple[float, float], Any]]):
    """
    输入: list of ((t,c), solution_object)
    输出: list of non-dominated solution_object
    """
    if not solutions_with_points:
        return []

    # 提取所有点
    points = np.array([pt for pt, sol in solutions_with_points])
    n = len(points)
    is_nd = np.ones(n, dtype=bool)

    # 执行非支配筛选
    for i in range(n):
        if not is_nd[i]:
            continue
        for j in range(n):
            if j == i:
                continue
            # 如果 j 支配 i
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                is_nd[i] = False
                break

    # 返回非支配的 solution 对象
    nd_solutions = [solutions_with_points[i][1] for i in range(n) if is_nd[i]]
    return nd_solutions


# ------------------ 辅助函数：查找最优解 ------------------
def find_best_solutions(archive: List, config: Config):
    """
    从 archive (solution objects) 中找到 Makespan (O1) 和 Cost (O2) 的最优解。
    返回 (best_time_solution, best_cost_solution)
    """
    best_time = float('inf')
    best_cost = float('inf')
    best_time_sol = None
    best_cost_sol = None

    for sol in archive:
        pt, sol_obj = solution_to_point(sol)
        if pt is None:
            continue
        t, c = pt

        # 确保 time 和 cost 都是有限数
        if not np.isfinite(t) or not np.isfinite(c):
            continue

        if t < best_time:
            best_time = t
            best_time_sol = sol_obj

        if c < best_cost:
            best_cost = c
            best_cost_sol = sol_obj

    # 如果两个最优解是同一个，则返回不同的引用（如果可能的话）
    # 在这个场景下，通常它们已经是最好的两个解对象了
    return best_time_sol, best_cost_sol


# ------------------ 其他工具函数 (保持原代码不变) ------------------
def summarize_mean_std(hv_list, igd_list, sp_list, avg_time_list, avg_cost_list):
    # ... (保持原代码不变) ...
    def ms(arr):
        arr_np = np.array(arr, dtype=float)
        return float(np.nanmean(arr_np)), float(np.nanstd(arr_np))

    return {
        'HV': ms(hv_list),
        'IGD': ms(igd_list),
        'SP': ms(sp_list),
        'AvgTime': ms(avg_time_list),
        'AvgCost': ms(avg_cost_list)
    }
# ... (run_algorithm_once, compute_metrics_for_da, build_reference_set, build_unified_ref_point, mean_ci, save_per_run_and_plot, save_and_plot_pareto_fronts 保持原代码不变) ...
def run_algorithm_once(name, alg_fn, config, q_table_path=None, fixed_action=None, allow_learning=False, seed=None):
    """
    返回统一格式： (combined_archive_list, hv_history, final_sp, igd_values)
    """
    # 固定随机性
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # deepcopy config 以避免跨 run 污染
    cfg = copy.deepcopy(config)

    # RL 风格（pso_rl_new）通常有额外参数
    if name.startswith('RL'):
        try:
            result = alg_fn(
                cfg,
                total_iterations=getattr(cfg, 'num_iterations', 100),
                load_q_table_path=q_table_path,
                save_q_table_path=None,
                exploit_only=not allow_learning,
                fixed_action=fixed_action
            )
        except TypeError:
            result = alg_fn(cfg)

        # RL-MOPSO 返回的是元组 (result_dict, best_fitness_history, all_generational_best_history)
        if isinstance(result, tuple) and len(result) >= 1:
            # 第一个元素是结果字典
            result_dict = result[0]
            if isinstance(result_dict, dict):
                ca = result_dict.get('convergence_archive', []) or []
                da = result_dict.get('diversity_archive', []) or []
                hv_history = result_dict.get('hv_history', []) or []
                final_sp = result_dict.get('final_sp', 0.0)
                igd_values = result_dict.get('igd_values', []) or []

                # print(f"DEBUG {name}: convergence_archive size = {len(ca)}")
                # print(f"DEBUG {name}: diversity_archive size = {len(da)}")

                combined = list(da) + list(ca)
                return combined, hv_history, final_sp, igd_values
            else:
                print(f"DEBUG {name}: First element of tuple is not a dict: {type(result_dict)}")
                return [], [], 0.0, []
        elif isinstance(result, dict):
            # 如果直接返回字典（兼容旧版本）
            ca = result.get('convergence_archive', []) or []
            da = result.get('diversity_archive', []) or []
            hv_history = result.get('hv_history', []) or []
            final_sp = result.get('final_sp', 0.0)
            igd_values = result.get('igd_values', []) or []

            combined = list(da) + list(ca)
            return combined, hv_history, final_sp, igd_values
        else:
            print(f"DEBUG {name}: Unexpected result type: {type(result)}")
            return [], [], 0.0, []
    else:
        # 其他算法
        result = alg_fn(cfg)
        if isinstance(result, (list, tuple)):
            if len(result) >= 10:
                ca = result[0] if len(result) > 0 else []
                da = result[1] if len(result) > 1 else []
                hv_history = result[7] if len(result) > 7 else []
                final_sp = result[8] if len(result) > 8 else 0.0
                igd_values = result[9] if len(result) > 9 else []
            else:
                ca = result[0] if len(result) > 0 else []
                da = result[0] if len(result) > 0 else []
                hv_history = result[1] if len(result) > 1 else []
                final_sp = result[2] if len(result) > 2 else 0.0
                igd_values = result[3] if len(result) > 3 else []
            combined = list(da) + list(ca)
            return combined, hv_history, final_sp, igd_values
        elif isinstance(result, dict):
            ca = result.get('convergence_archive', []) or []
            da = result.get('diversity_archive', []) or []
            hv_history = result.get('hv_history', []) or []
            final_sp = result.get('final_sp', 0.0)
            igd_values = result.get('igd_values', []) or []
            combined = list(da) + list(ca)
            return combined, hv_history, final_sp, igd_values
        else:
            return [], [], 0.0, []


# ------------------ 计算指标 ------------------
def compute_metrics_for_da(da, ref_point, reference_set_array):
    valid = []
    total_time = 0.0
    total_cost = 0.0

    # print(f"DEBUG: Processing {len(da)} solutions")

    for i, p in enumerate(da):
        # 🚨 关键修改：正确解包 solution_to_point 的返回值。
        # pt_and_sol 接收 (t, c) 元组，_ 接收 solution 对象
        pt_and_sol, _ = solution_to_point(p)

        # 检查是否返回了有效的目标点元组 (t, c)
        if pt_and_sol is None:
            # print(f"DEBUG: Solution {i} is invalid: {p}")
            continue

        # 确保 pt_and_sol 是长度为 2 的序列
        if not isinstance(pt_and_sol, (list, tuple)) or len(pt_and_sol) != 2:
            continue

        # 明确提取浮点数时间 t 和成本 c
        try:
            t = float(pt_and_sol[0])
            c = float(pt_and_sol[1])
        except (ValueError, TypeError):
            continue

        valid.append((t, c))
        total_time += t  # 现在 total_time (float) += t (float) 是正确的
        total_cost += c
        # ...

    # print(f"DEBUG: Found {len(valid)} valid solutions from {len(da)} total solutions")

    n = len(valid)
    if n == 0:
        # print("DEBUG: No valid solutions found!")
        # 保持 nan 行为，确保 IGD/SP 也是 0.0 或 nan
        return 0.0, np.nan, 0.0, np.nan, np.nan

        # 计算HV
    try:
        hv = calculate_hypervolume(valid, ref_point)
        # print(f"DEBUG: HV calculated = {hv}")
    except Exception as e:
        # print(f"DEBUG: HV calculation failed: {e}")
        hv = 0.0

    # 计算IGD
    if reference_set_array is not None and len(reference_set_array) > 0:
        try:
            da_array = np.array(valid)
            dists = []
            for ref_i in reference_set_array:
                # 确保 ref_i 是 NumPy 数组
                ref_i = np.asarray(ref_i)
                if ref_i.size != 2:
                    continue
                # 计算到所有解的最短距离
                mins = np.min(np.linalg.norm(da_array - ref_i.reshape(1, 2), axis=1))
                dists.append(mins)
            igd = float(np.mean(dists)) if dists else np.nan
            # print(f"DEBUG: IGD calculated = {igd}")
        except Exception as e:
            # print(f"DEBUG: IGD calculation failed: {e}")
            igd = np.nan
    else:
        igd = np.nan
        # print("DEBUG: No reference set for IGD calculation")

    # 计算Spacing
    try:
        sp = calculate_spacing(valid)
        # print(f"DEBUG: Spacing calculated = {sp}")
    except Exception as e:
        # print(f"DEBUG: Spacing calculation failed: {e}")
        sp = 0.0

    avg_time = total_time / n
    avg_cost = total_cost / n
    # print(f"DEBUG: AvgTime = {avg_time}, AvgCost = {avg_cost}")

    return hv, igd, sp, avg_time, avg_cost

# ------------------ 构建统一参考集 ------------------
# 假设这个函数位于您代码中发生错误的区域（例如 draw_gantt.py 的 line 350 附近）
def build_reference_set(all_archives):
    """
    从所有 Archive 中提取所有目标点，并进行严格的格式检查。
    """
    all_points = []

    for da in all_archives:
        if not da:
            continue

        for sol in da:
            # 🚨 关键修改：使用新的签名接收目标点 (t, c)
            pt_and_sol, _ = solution_to_point(sol)

            # 严格检查 pt_and_sol 是否是有效的长度为 2 的目标点
            if (pt_and_sol is not None and
                    isinstance(pt_and_sol, (list, tuple)) and
                    len(pt_and_sol) == 2):

                try:
                    # 最终安全检查：确保它们是有限的数字
                    t = float(pt_and_sol[0])
                    c = float(pt_and_sol[1])
                    if np.isfinite(t) and np.isfinite(c):
                        all_points.append((t, c))
                except (ValueError, TypeError):
                    # 忽略目标值无法转换为浮点数的解
                    continue

    if not all_points:
        return np.empty((0, 2))

    pts = np.array(all_points)  # 错误在这里解决

    # 返回非支配筛选后的点
    return nondominated_filter(pts)

# ------------------ 构建统一 HV 参考点 ------------------
def build_unified_ref_point(all_archives, margin=1.5):
    """
    遍历所有 Archive，找到所有目标值的最大值（用于设定 HV 参考点）。
    """
    max_time = None
    max_cost = None

    for da in all_archives:
        if not da:
            continue

        for sol in da:
            # 🚨 关键修改：正确解包，只取目标点 pt_and_sol
            pt_and_sol, _ = solution_to_point(sol)

            if (pt_and_sol is not None and
                    isinstance(pt_and_sol, (list, tuple)) and
                    len(pt_and_sol) == 2):

                try:
                    # 将目标值明确转换为浮点数并提取
                    t = float(pt_and_sol[0])
                    c = float(pt_and_sol[1])

                    # 检查是否为有限数
                    if not (np.isfinite(t) and np.isfinite(c)):
                        continue

                    # 比较并更新最大时间
                    if max_time is None:
                        max_time = t
                    else:
                        max_time = max(max_time, t)

                    # 比较并更新最大成本
                    if max_cost is None:
                        max_cost = c
                    else:
                        max_cost = max(max_cost, c)

                except (ValueError, TypeError):
                    continue

    # 如果没有找到任何有效解，返回默认值
    if max_time is None or max_cost is None:
        return np.array([1.0, 1.0])  # 默认安全值

    # 计算并返回加了 margin 的参考点
    ref_point = np.array([max_time * margin, max_cost * margin])
    return ref_point

# ------------------ 均值±95%CI ------------------
try:
    from scipy.stats import t
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def mean_ci(values, confidence=0.95):
    vals = np.array(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, np.nan
    m = vals.mean()
    s = vals.std(ddof=1) if n > 1 else 0.0
    se = s / math.sqrt(n) if n > 0 else 0.0
    if n > 1 and _HAS_SCIPY:
        tcrit = t.ppf(1 - (1 - confidence) / 2, n - 1)
        margin = tcrit * se
    elif n == 1:
        margin = 0.0
    else:
        margin = 1.96 * se
    return m, m - margin, m + margin

# ------------------ 保存指标与绘图（保留原有） ------------------
def save_per_run_and_plot(collected, reference_set, ref_point, out_dir="results_plots"):
    os.makedirs(out_dir, exist_ok=True)
    records = []
    for method, runs in collected.items():
        for ridx, da in enumerate(runs):
            hv, igd, sp, avg_time, avg_cost = compute_metrics_for_da(da, ref_point, reference_set)
            records.append({"method": method, "run_id": ridx, "hv": hv, "igd": igd, "sp": sp, "avg_time": avg_time, "avg_cost": avg_cost})
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, "per_run_metrics.csv"), index=False)

    # 箱线图和均值±CI（对空数据做保护）
    methods = sorted(df['method'].unique())
    for metric in ["hv", "igd", "sp"]:
        plt.figure(figsize=(max(6, 1.2 * len(methods)), 5))
        data = [df[df['method'] == m][metric].dropna().values for m in methods]
        safe_data = [d if len(d) > 0 else np.array([np.nan]) for d in data]
        plt.boxplot(safe_data, labels=methods, showmeans=True)
        plt.ylabel(metric)
        plt.title(f"{metric.upper()} 箱线图")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{metric}_boxplot.png"))
        plt.close()

        # 均值±95%CI
        means, lowers, uppers, ns = [], [], [], []
        for m in methods:
            vals = df[df['method'] == m][metric].values
            mval, lo, up = mean_ci(vals)
            means.append(mval); lowers.append(lo); uppers.append(up); ns.append(int(np.sum(~np.isnan(vals))))
        means = np.array(means); lowers = np.array(lowers); uppers = np.array(uppers)
        err_low = means - lowers; err_up = uppers - means

        plt.figure(figsize=(max(6, 1.2 * len(methods)), 5))
        x = np.arange(len(methods))
        plt.errorbar(x, np.nan_to_num(means), yerr=[np.nan_to_num(err_low), np.nan_to_num(err_up)], fmt='o', capsize=5)
        plt.xticks(x, methods, rotation=45, ha='right'); plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{metric}_mean_ci.png"))
        plt.close()

    return df

# ------------------ 把合并非支配前沿保存为 CSV 并绘图 ------------------
def save_and_plot_pareto_fronts(per_method_pareto, out_dir="pareto_fronts", ref_point=None, scatter_all=False):
    """
    per_method_pareto: dict method -> ndarray of non-dominated points (Nx2)
    scatter_all: 若 True 同时绘制每个算法的所有点（会很乱），通常设 False，只画合并非支配前沿
    """
    outdir = pathlib.Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 保存每个方法的 pareto 到 CSV，并同时生成合并 CSV
    all_rows = []
    for method, pts in per_method_pareto.items():
        if pts is None or len(pts) == 0:
            continue
        # 保存单独 CSV
        csv_path = outdir / f"{method}_pareto.csv"
        with open(csv_path, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'cost'])
            for t, c in pts:
                writer.writerow([t, c])
        for t, c in pts:
            all_rows.append([method, float(t), float(c)])

    # 保存全部合并的 csv
    all_csv = outdir / "all_methods_pareto.csv"
    with open(all_csv, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'time', 'cost'])
        writer.writerows(all_rows)

    # 画图：每个算法画一条线（合并非支配前沿）
    plt.figure(figsize=(8,6))
    ax = plt.gca()
    cmap = plt.get_cmap("tab10")
    methods = sorted(per_method_pareto.keys())
    for i, method in enumerate(methods):
        pts = per_method_pareto[method]
        if pts is None or len(pts) == 0:
            continue
        # 按 time 排序，画线
        pts_sorted = pts[np.argsort(pts[:,0])]
        ax.plot(pts_sorted[:,0], pts_sorted[:,1], label=method, color=cmap(i%10), linewidth=2)
        ax.scatter(pts_sorted[:,0], pts_sorted[:,1], s=18, color=cmap(i%10), alpha=0.9)

    # 可选：画参考点
    if ref_point is not None:
        ax.scatter([ref_point[0]], [ref_point[1]], marker='x', s=80, color='k', label='HV ref point')

    ax.set_xlabel("Objective 1 (time) — minimize")
    ax.set_ylabel("Objective 2 (cost) — minimize")
    ax.set_title("Pareto front (per method, merged non-dominated across runs)")
    ax.grid(True, linestyle='--', alpha=0.3)
    # 若你希望纵轴较小的值在上面可启用下面一行；否则注释掉
    # plt.gca().invert_yaxis()
    ax.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plot_path = outdir / "pareto_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print("已保存合并 pareto CSV 到:", all_csv)
    print("已保存 pareto 对比图到:", plot_path)
    return all_csv, plot_path

# ----------------------------------------------------
# --- 【整合】您提供的甘特图函数 (draw_gantt, generate_global_resource_map) ---
# ----------------------------------------------------
def generate_global_resource_map(config):
    # ... (函数实现保持不变) ...
    stages_order_cn = ["采购", "部装", "总装"]
    stage_to_prefix = {"采购": "P", "部装": "S", "总装": "F"}
    resource_to_stages = {}
    for (res_id, task_id) in config.machine_task_map.keys():
        stage = config.task_stage[task_id]
        resource_to_stages.setdefault(res_id, set()).add(stage)
    resource_stage_map = {}
    stage_priority = {s: i for i, s in enumerate(stages_order_cn)}
    for res_id, stages in resource_to_stages.items():
        primary_stage = max(stages, key=lambda s: stage_priority.get(s, -1))
        resource_stage_map[res_id] = primary_stage
    staged_resources = {}
    for r_id, stage in resource_stage_map.items():
        staged_resources.setdefault(stage, []).append(r_id)
    global_resource_map = {}
    for stage in stages_order_cn:
        if stage not in staged_resources: continue
        resources_in_stage = sorted(staged_resources[stage])
        prefix = stage_to_prefix.get(stage, 'R')
        for i, original_id in enumerate(resources_in_stage, start=1):
            global_resource_map[original_id] = {
                'new_id': i,
                'prefix': prefix,
                'stage': stage
            }
    return global_resource_map


def draw_gantt(schedule, title="Optimal Schedule Gantt Chart", global_resource_map=None, save_path=None):
    # ... (函数实现保持不变) ...
    if not schedule:
        print("schedule is empty")
        return

    # --- 参数定义 ---
    FIXED_SETUP_TIME = 2.0
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
    # --- 6) 核心功能：数据收集与CSV保存 ---
    # ----------------------------------------------------
    if save_path:
        data_records = []
        for t in schedule:
            new_resource_key = t.get('new_resource_key')
            prefix, new_id = new_resource_key
            base = resource_base.get(new_resource_key, 0.0)
            assigned = task_row_assignment.get(id(t), 0)
            y_center = base + assigned
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
                if setup_width > 0:
                    data_records.append({
                        'Type': 'Setup', 'Task_ID': f"SETUP_{t.get('task')}",
                        'Product_ID': int(t.get('product', 0)) + 1,
                        'Stage': t.get('stage'), 'Resource_Original_ID': t.get('resource'),
                        'Resource_Key': f"{prefix}{new_id}",
                        'Subline': assigned + 1, 'Y_Position_Center': y_center, 'Start_Time': last_ed,
                        'End_Time': last_ed + setup_width,
                        'Duration': setup_width, 'is_Setup': 1
                    })
                idle_only_width = idle_width - setup_width
                if idle_only_width > 0:
                    data_records.append({
                        'Type': 'Idle', 'Task_ID': f"IDLE_{t.get('task')}", 'Product_ID': int(t.get('product', 0)) + 1,
                        'Stage': t.get('stage'), 'Resource_Original_ID': t.get('resource'),
                        'Resource_Key': f"{prefix}{new_id}",
                        'Subline': assigned + 1, 'Y_Position_Center': y_center, 'Start_Time': last_ed + setup_width,
                        'End_Time': st,
                        'Duration': idle_only_width, 'is_Setup': 0
                    })
        df = pd.DataFrame(data_records)
        df = df.sort_values(by=['Y_Position_Center', 'Start_Time']).reset_index(drop=True)
        try:
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"\n--- 成功将调度数据保存到: {save_path} ---")
        except Exception as e:
            print(f"\n--- 警告: 保存CSV文件时出错: {e} ---")

    # ----------------------------------------------------
    # --- 7) 绘制图表 (保持不变) ---
    # ----------------------------------------------------
    bar_height = 0.8
    fig_height = max(8, len(y_ticks) * 0.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))
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
            ax.barh(bottom_y, setup_width, left=last_ed, height=bar_height, color=setup_color,
                    edgecolor=SETUP_EDGE_COLOR, linewidth=SETUP_LINE_WIDTH, hatch=setup_hatch, align='edge', zorder=2,
                    clip_on=True)

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
        ax.barh(bottom_y, width, left=st, height=bar_height, color=base_color, edgecolor=(0.2, 0.2, 0.2), linewidth=1.2,
                align='edge', zorder=4, clip_on=True)
        seq = prod_seq_map.get(id(t), '?')
        s_short = stage_to_short_en.get(s_cn, s_cn[:3])
        label = f"P{p + 1}-{s_short}-{seq}"
        center_x = st + width * 0.5
        brightness = 0.299 * base_color[0] + 0.587 * base_color[1] + 0.114 * base_color[2]
        text_color = 'white' if brightness < 0.6 else 'black'
        ax.text(center_x, y_center, label, ha='center', va='center', fontsize=10, color=text_color,
                fontname=preferred_font, fontweight='bold', zorder=6, clip_on=True)

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
    prod_handles = [mpatches.Patch(facecolor=product_colors[p], edgecolor='black', label=f"Product {p + 1}") for p in
                    range(num_products)]
    setup_handle = mpatches.Patch(facecolor=setup_color, edgecolor=SETUP_EDGE_COLOR, hatch=setup_hatch,
                                  label="Setup Time (Switching)")
    prod_handles.append(setup_handle)
    legend = ax.legend(handles=prod_handles, title="Legend", loc='lower right', fontsize=10, title_fontsize=11,
                       prop={'family': preferred_font})
    for text in legend.get_texts(): text.set_fontname(preferred_font)
    legend.get_title().set_fontname(preferred_font)

    # X/Y 轴范围
    max_end = max((float(t.get('end', 0) or 0) for t in schedule), default=0)
    if max_end <= 0: max_end = max((float(t.get('start', 0) or 0) + 1 for t in schedule), default=1)
    x_margin_right = max_end * 0.05
    ax.set_xlim(0, max_end + x_margin_right)
    ax.set_ylim(-0.5, y_limit + 0.5)

    plt.tight_layout()
    plt.show()


# ------------------ 主程序 (修改运行RL-MOPSO的部分) ------------------
if __name__ == '__main__':
    base_config = Config()
    base_config.num_iterations = 100
    base_config.num_particles = 50

    num_runs = 20
    q_table_path = "1.0-q_table4.pkl"

    # 构建算法字典（只包含已加载的函数）
    algorithms = {}
    if run_rl_mopso is not None:
        algorithms['RL-MOPSO'] = run_rl_mopso
    # ... (其他算法保持不变) ...
    # if run_moabc:
    #     algorithms['MOABC'] = run_moabc
    if run_moaco:
        algorithms['MOACO'] = run_moaco
    if run_plain_mopso:
        algorithms['MOPSO'] = run_plain_mopso
    if run_mogwo:
        algorithms['MOGWO'] = run_mogwo
    if run_mowoa:
        algorithms['MOWOA'] = run_mowoa

    if not algorithms:
        raise SystemExit("未发现可运行算法")

    collected = {name: [] for name in algorithms.keys()}
    base_seed = 32
    seeds = [base_seed + i for i in range(num_runs)]
    print(f"使用种子序列: {seeds} (确保结果可重现)")

    results_dir = pathlib.Path("results_archives")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 🚨 新增：用于收集 RL-MOPSO 所有解的列表
    rl_mopso_all_solutions_with_points = []

    # 预计算全局资源映射 (如果需要固定编号)
    global_res_map = generate_global_resource_map(base_config)

    for name, fn in algorithms.items():
        print(f">>> 正在运行 {name}...")
        for run_id, seed in enumerate(seeds):
            print(f"Running {name} run {run_id + 1}/{num_runs} (seed={seed})")
            fixed_action = int(name.split('-')[-1]) if name.startswith('RL-fixed-') else None
            da, hv_hist, final_sp, igd_vals = run_algorithm_once(name, fn, base_config, q_table_path=q_table_path,
                                                                 fixed_action=fixed_action, allow_learning=False,
                                                                 seed=seed)
            collected[name].append(da)

            # 🚨 收集 RL-MOPSO 的所有解及其对应的目标点
            if name == 'RL-MOPSO':
                for sol in da:
                    pt_and_sol, sol_obj = solution_to_point(sol)
                    if pt_and_sol is not None:
                        rl_mopso_all_solutions_with_points.append((pt_and_sol, sol_obj))

            # 保存单次运行结果以便复现：保存 archive + metadata (保持原代码不变)
            save_obj = {
                "method": name,
                "run_id": run_id,
                "seed": seed,
                "archive": da,
                "hv_history": hv_hist,
                "final_sp": final_sp,
                "igd_values": igd_vals,
                "config": copy.deepcopy(base_config)
            }
            with open(results_dir / f"{name}_run{run_id}.pkl", "wb") as fh:
                pickle.dump(save_obj, fh)

    # ====================================================================
    # 🚨 【核心修改】：提取 RL-MOPSO 合并 Pareto 前沿的最优解并绘制/保存
    # ====================================================================
    print("\n--- 正在处理 RL-MOPSO (合并 Pareto 前沿) 的最优甘特图 ---")

    # 1. 筛选 RL-MOPSO 的合并非支配解集 (Solution Objects)
    merged_rl_mopso_archive = nondominated_filter_with_solutions(rl_mopso_all_solutions_with_points)

    if not merged_rl_mopso_archive:
        print("警告: RL-MOPSO 合并 Archive 为空，无法提取最优解。")
    else:
        # 2. 从合并的 Archive 中找到最优解
        best_time_sol, best_cost_sol = find_best_solutions(merged_rl_mopso_archive, base_config)

        gantt_plots_dir = pathlib.Path("gantt_outputs_merged_front")
        gantt_plots_dir.mkdir(parents=True, exist_ok=True)

        # 3. 时间最优解 (Makespan)
        if best_time_sol and hasattr(best_time_sol, 'schedule') and best_time_sol.schedule:
            T_opt = best_time_sol.fitness[0]
            C_val = best_time_sol.fitness[1]
            title = f"RL-MOPSO (Merged Front, Time Optimal) - Time={T_opt:.2f}, Cost={C_val:.2f}"
            csv_path = gantt_plots_dir / "RL_MOPSO_MergedFront_Time_Optimal_Data2.csv"

            # 绘图和保存 CSV
            draw_gantt(best_time_sol.schedule, title=title,
                       global_resource_map=global_res_map,
                       save_path=csv_path)
        else:
            print("警告: 无法找到或提取时间最优解的调度信息。")

        # 4. 成本最优解 (Cost)
        if best_cost_sol and hasattr(best_cost_sol, 'schedule') and best_cost_sol.schedule:
            T_val = best_cost_sol.fitness[0]
            C_opt = best_cost_sol.fitness[1]
            title = f"RL-MOPSO (Merged Front, Cost Optimal) - Time={T_val:.2f}, Cost={C_opt:.2f}"
            csv_path = gantt_plots_dir / "RL_MOPSO_MergedFront_Cost_Optimal_Data2.csv"

            # 绘图和保存 CSV
            draw_gantt(best_cost_sol.schedule, title=title,
                       global_resource_map=global_res_map,
                       save_path=csv_path)
        else:
            print("警告: 无法找到或提取成本最优解的调度信息。")

    # ====================================================================
    # --- 原有算法比较逻辑 (保持不变) ---
    # ====================================================================

    # 汇总所有的 DA 用于生成统一 reference set 与 ref point
    all_da = [da for runs in collected.values() for da in runs if da]
    reference_set = build_reference_set(all_da)
    np.save('reference_set.npy', reference_set)
    print(f"\n统一参考集生成完成，包含 {len(reference_set)} 个解")
    ref_point = build_unified_ref_point(all_da, margin=1.5)
    print(f"统一HV参考点: {ref_point}（对所有算法/运行使用相同参考点）")

    # ========== 对每个算法合并其多次运行并做非支配筛选 (仅点) ==========
    per_method_pareto = {}
    for name, runs in collected.items():
        merged_pts = []
        for da in runs:
            if not da:
                continue
            for sol in da:
                pt_and_sol, sol_obj = solution_to_point(sol)
                if pt_and_sol is not None:
                    merged_pts.append(pt_and_sol)
        # 这里使用原始的 nondominated_filter，只保留点用于绘图
        nd = nondominated_filter(merged_pts)
        per_method_pareto[name] = nd

    # 保存并画图
    save_and_plot_pareto_fronts(per_method_pareto, out_dir="pareto_fronts-gantt2", ref_point=ref_point,
                                scatter_all=False)

    # ========== 原有指标统计（保留） ==========
    metrics = {}
    csv_rows = []
    for name, runs in collected.items():
        hv_list, igd_list, sp_list, avg_time_list, avg_cost_list = [], [], [], [], []
        for da in runs:
            hv, igd, sp, avg_time, avg_cost = compute_metrics_for_da(da, ref_point, reference_set)
            hv_list.append(hv)
            igd_list.append(igd)
            sp_list.append(sp)
            avg_time_list.append(avg_time)
            avg_cost_list.append(avg_cost)
        metrics[name] = summarize_mean_std(hv_list, igd_list, sp_list, avg_time_list, avg_cost_list)
        hv_m, hv_s = metrics[name]['HV']
        igd_m, igd_s = metrics[name]['IGD']
        sp_m, sp_s = metrics[name]['SP']
        avg_time_m, avg_time_s = metrics[name]['AvgTime']
        avg_cost_m, avg_cost_s = metrics[name]['AvgCost']
        print(
            f"{name}: HV={hv_m:.4f}±{hv_s:.4f}, IGD={igd_m:.4f}±{igd_s:.4f}, SP={sp_m:.4f}±{sp_s:.4f}, AvgTime={avg_time_m:.1f}±{avg_time_s:.1f}, AvgCost={avg_cost_m:.1f}±{avg_cost_s:.1f}")
        csv_rows.append([name, hv_m, hv_s, igd_m, igd_s, sp_m, sp_s, avg_time_m, avg_time_s, avg_cost_m, avg_cost_s])

    # 导出 CSV（统计表）
    headers = ['Method', 'HV_mean', 'HV_std', 'IGD_mean', 'IGD_std', 'SP_mean', 'SP_std', 'AvgTime_mean', 'AvgTime_std',
               'AvgCost_mean', 'AvgCost_std']
    with open('comparison-1.0-4-gantt2.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_rows)

    # 导出 per-run 指标并绘图
    df = save_per_run_and_plot(collected, reference_set, ref_point, out_dir="results_plots_gantt2")

    print("全部完成。合并的非支配解已保存，统计结果也保存！")