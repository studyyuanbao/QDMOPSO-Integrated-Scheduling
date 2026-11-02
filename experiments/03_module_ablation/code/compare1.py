import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import os
from init import Config
from mopso_basic import run_mopso as run_plain_mopso
from pso_rl_new import run_mopso as run_rl_mopso
from utils import calculate_hypervolume, calculate_spacing
import csv

# 可选算法
try:
    from nsga2 import run_nsga2
except Exception:
    run_nsga2 = None
try:
    from moabc import run_moabc
except Exception:
    run_moabc = None
try:
    from spea2 import run_spea2
except Exception:
    run_spea2 = None

# 设置支持中文的字体，确保只设置一次
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False


def summarize_mean_std(hv_list, igd_list, sp_list, avg_time_list, avg_cost_list):
    """返回均值±标准差，便于论文汇报。"""

    def ms(arr):
        if not arr:
            return (0.0, 0.0)
        # 确保使用 numpy 的 nanmean 和 nanstd 来处理可能出现的 NaN
        arr_np = np.array(arr, dtype=float)
        return (float(np.nanmean(arr_np)), float(np.nanstd(arr_np)))

    return {
        'HV': ms(hv_list),
        'IGD': ms(igd_list),
        'SP': ms(sp_list),
        'AvgTime': ms(avg_time_list),
        'AvgCost': ms(avg_cost_list)
    }


def run_algorithm_once(name, alg_fn, config, q_table_path=None, fixed_action=None, allow_learning=False, seed=None):
    """统一运行一次算法，返回 (da, hv_history, final_sp, igd_values)。"""
    # 设置随机种子确保结果可重现
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    if name == 'RL-MOPSO' or (name.startswith('RL-fixed')):
        result, _, _ = run_rl_mopso(
            config,
            total_iterations=getattr(config, 'num_iterations', 100),
            load_q_table_path=q_table_path,
            save_q_table_path=None,
            exploit_only=not allow_learning,  # allow_learning=True时exploit_only=False
            fixed_action=fixed_action
        )
        return result.get('convergence_archive', []) + result.get('diversity_archive', []), result.get('hv_history',
                                                                                                       []), result.get(
            'final_sp', 0.0), result.get(
            'igd_values', [])
    else:
        # run_result 结构: ca, da, ..., hv_history, final_sp, igd_values (假设至少有 10 个返回值)
        run_result = alg_fn(config)
        ca, da, _, _, _, _, _, hv_history, final_sp, igd_values = run_result[:10]
        return (da + ca), hv_history, final_sp, igd_values


def build_reference_set(collected_da_lists):
    """从所有档案中提取全局非支配前沿作为参考集"""
    all_points = []
    for da in collected_da_lists:
        if not da:
            continue
        for p in da:
            # 确保 p.fitness 是 (Time, Cost) 的数值元组
            if hasattr(p, 'fitness') and isinstance(p.fitness, (list, tuple)) and len(p.fitness) == 2:
                try:
                    fitness_tuple = (float(p.fitness[0]), float(p.fitness[1]))
                    if not (np.isnan(fitness_tuple[0]) or np.isnan(fitness_tuple[1])):
                        all_points.append(fitness_tuple)
                except (ValueError, TypeError):
                    continue
    if not all_points:
        return np.empty((0, 2))

    # 筛选非支配解
    pts = np.array(all_points, dtype=float)
    keep = []
    for i in range(len(pts)):
        dominated = False
        for j in range(len(pts)):
            if i == j:
                continue
            # 检查 j 是否支配 i
            if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return pts[keep]


def build_unified_ref_point(collected_da_lists, margin: float = 1.1):
    """基于所有算法/多次运行的解，统一生成一个HV参考点。

    规则：取全体可行解在 (Time, Cost) 两维上的最大值，并乘以 margin（>1）留安全边界。
    若没有可行解，则返回一个保守的大参考点。
    """
    max_time = None
    max_cost = None
    for da in collected_da_lists:
        if not da:
            continue
        for p in da:
            if hasattr(p, 'fitness') and isinstance(p.fitness, (list, tuple)) and len(p.fitness) == 2:
                try:
                    t = float(p.fitness[0])
                    c = float(p.fitness[1])
                    if not (np.isnan(t) or np.isnan(c) or t == float('inf') or c == float('inf')):
                        max_time = t if max_time is None else max(max_time, t)
                        max_cost = c if max_cost is None else max(max_cost, c)
                except (ValueError, TypeError):
                    continue
    if max_time is None or max_cost is None:
        return (20000.0, 20000.0)
    return (max_time * margin + 1.0, max_cost * margin + 1.0)


def compute_metrics_for_da(da, ref_point, reference_set_array):
    """计算 HV, IGD, SP, AvgTime 和 AvgCost 指标"""

    valid_fitness_tuples = []
    total_time_sum = 0.0  # Time 目标
    total_cost_sum = 0.0  # Cost 目标

    for p in da:
        if hasattr(p, 'fitness') and isinstance(p.fitness, (list, tuple)) and len(p.fitness) == 2:
            try:
                # 假设 fitness[0] 是 Time (完工时间), fitness[1] 是 Cost (总成本)
                f1_time = float(p.fitness[0])
                f2_cost = float(p.fitness[1])

                # 过滤掉无效值
                if not (np.isnan(f1_time) or np.isnan(f2_cost) or f1_time == float('inf') or f2_cost == float('inf')):
                    fitness_tuple = (f1_time, f2_cost)
                    valid_fitness_tuples.append(fitness_tuple)
                    total_time_sum += f1_time
                    total_cost_sum += f2_cost
            except (ValueError, TypeError):
                continue

    num_valid_solutions = len(valid_fitness_tuples)

    if num_valid_solutions == 0:
        return 0.0, np.nan, 0.0, np.nan, np.nan  # HV, IGD, SP, AvgTime, AvgCost

    # --- 1. HV 计算 ---
    hv = calculate_hypervolume(valid_fitness_tuples, ref_point)

    # --- 2. IGD 计算 ---
    if len(reference_set_array) > 0:
        da_array = np.array(valid_fitness_tuples)
        igd_distances = [np.min(np.linalg.norm(da_array - np.array(ref_point_i), axis=1)) for ref_point_i in
                         reference_set_array]
        igd = np.mean(igd_distances)
    else:
        igd = np.nan

    # --- 3. SP 计算 ---
    sp = calculate_spacing(valid_fitness_tuples)

    # --- 4 & 5. AvgTime 和 AvgCost 计算 ---
    avg_time = total_time_sum / num_valid_solutions
    avg_cost = total_cost_sum / num_valid_solutions

    return hv, igd, sp, avg_time, avg_cost


def export_csv(summary_rows, out_path='comparison_summary_with_avg_objectives.csv'):
    # CSV 头部保持不变
    headers = ['Method', 'HV_mean', 'HV_std', 'IGD_mean', 'IGD_std', 'SP_mean', 'SP_std',
               'AvgTime_mean', 'AvgTime_std', 'AvgCost_mean', 'AvgCost_std']
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in summary_rows:
            writer.writerow(row)
    print(f"\n结果已保存到 {out_path}")


def debug_metrics_consistency(collected, ref_point, reference_set):
    """验证两种计算方式是否一致"""
    print("\n=== 调试指标一致性 ===")

    # 选择一个算法进行验证
    test_method = list(collected.keys())[0]
    test_da = collected[test_method][0]  # 第一次运行

    # 方式1：直接计算
    hv1, igd1, sp1, avg_time1, avg_cost1 = compute_metrics_for_da(
        test_da, ref_point, reference_set
    )

    print(f"直接计算 - HV: {hv1:.4f}, IGD: {igd1:.4f}, SP: {sp1:.4f}")
    print(f"参考点: {ref_point}, 参考集大小: {len(reference_set)}")
    print(f"输入解集大小: {len(test_da)}")

    # 统计有效解数量
    valid_count = 0
    for p in test_da:
        if hasattr(p, 'fitness') and isinstance(p.fitness, (list, tuple)) and len(p.fitness) == 2:
            try:
                t, c = float(p.fitness[0]), float(p.fitness[1])
                if not (np.isnan(t) or np.isnan(c)):
                    valid_count += 1
            except:
                continue
    print(f"有效解数量: {valid_count}")

    return hv1, igd1, sp1


def check_data_integrity(collected):
    """检查收集的数据是否完整"""
    print("\n=== 数据完整性检查 ===")

    for name in collected.keys():
        runs = collected[name]
        print(f"{name}: {len(runs)} 次运行")

        # 检查每次运行的解数量
        for i, da in enumerate(runs):
            valid_solutions = 0
            for p in da:
                if hasattr(p, 'fitness') and isinstance(p.fitness, (list, tuple)) and len(p.fitness) == 2:
                    try:
                        t, c = float(p.fitness[0]), float(p.fitness[1])
                        if not (np.isnan(t) or np.isnan(c)):
                            valid_solutions += 1
                    except:
                        continue
            if i < 3:  # 只显示前3次运行避免输出过多
                print(f"  运行 {i + 1}: {valid_solutions} 个有效解")
        if len(runs) > 3:
            print(f"  ... 其余 {len(runs) - 3} 次运行省略")


# -*- coding: utf-8 -*-
"""
compare_mopso_versions_with_plots.py
说明：
在原 compare_mopso_versions.py 的基础上，增加：
- 每次运行的 hv / igd / sp / avg_time / avg_cost 导出到 CSV
- 画出箱线图和均值 ±95% 置信区间图
- 结果保存在 results_plots/ 文件夹下
"""

import math
import pandas as pd

# ------------------ 计算均值与置信区间 ------------------
try:
    from scipy.stats import t

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def mean_ci(values, confidence=0.95):
    """计算均值及置信区间 (mean, lower, upper)，忽略 NaN"""
    vals = np.array(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    m = vals.mean()
    s = vals.std(ddof=1) if n > 1 else 0.0
    se = s / math.sqrt(n) if n > 0 else 0.0
    if n > 1 and _HAS_SCIPY:
        df = n - 1
        tcrit = t.ppf(1 - (1 - confidence) / 2, df)
        margin = tcrit * se
    elif n == 1:
        margin = 0.0
    else:
        z = 1.96
        margin = z * se
    return (m, m - margin, m + margin)


# ------------------ 核心导出 + 绘图函数 ------------------
def save_per_run_and_plot(collected, ref_point, reference_set, out_dir="results_plots"):
    """
    保存每次运行指标 + 画箱线图 + 均值±95%CI 图
    参数：
        collected: dict，键=算法名，值=list[da]（每次运行的档案）
        ref_point: tuple，用于 HV 计算
        reference_set: numpy array，用于 IGD 计算
    """
    os.makedirs(out_dir, exist_ok=True)

    # 获取 compute_metrics_for_da 函数
    compute_fn = globals().get("compute_metrics_for_da")
    if compute_fn is None:
        raise RuntimeError("未找到 compute_metrics_for_da，请确认主脚本中已定义该函数")

    print(f"[INFO] 开始计算指标，参考点: {ref_point}, 参考集大小: {len(reference_set)}")

    # ---- 1) 逐 run 计算指标 ----
    records = []
    for method, runs in collected.items():
        print(f"[INFO] 处理算法: {method}, {len(runs)} 次运行")
        for ridx, da in enumerate(runs):
            try:
                # 修复参数顺序：应该是 (da, ref_point, reference_set)
                hv, igd, sp, avg_time, avg_cost = compute_fn(da, ref_point, reference_set)
            except Exception as e:
                print(f"[ERROR] 计算 {method} 第 {ridx + 1} 次运行指标出错: {e}")
                hv = igd = sp = avg_time = avg_cost = np.nan

            def tofloat(x):
                try:
                    return float(x)
                except Exception:
                    return np.nan

            records.append({
                "method": method,
                "run_id": ridx,
                "hv": tofloat(hv),
                "igd": tofloat(igd),
                "sp": tofloat(sp),
                "avg_time": tofloat(avg_time),
                "avg_cost": tofloat(avg_cost),
            })

    df = pd.DataFrame.from_records(records)
    csv_path = os.path.join(out_dir, "per_run_metrics_long.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] 已保存每次运行指标到: {csv_path} （共 {len(df)} 行）")

    # ---- 2) 箱线图 ----
    for metric in ["hv", "igd", "sp"]:
        plt.figure(figsize=(max(6, 1.2 * len(df['method'].unique())), 5))
        methods = sorted(df['method'].unique())
        data = [df[df['method'] == m][metric].dropna().values for m in methods]
        if all(len(d) == 0 for d in data):
            print(f"[WARN] 指标 {metric} 无数据，跳过箱线图绘制。")
            continue
        plt.boxplot(data, tick_labels=methods, showmeans=True)
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} 箱线图（每次运行）")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"{metric}_boxplot.png")
        plt.savefig(out_png)
        plt.close()
        print(f"[INFO] 已保存 {metric} 箱线图到: {out_png}")

    # ---- 3) 均值 ±95% CI 图 ----
    for metric in ["hv", "igd", "sp"]:
        methods = sorted(df['method'].unique())
        means, lowers, uppers, ns = [], [], [], []
        for m in methods:
            vals = df[df['method'] == m][metric].values
            mval, lo, up = mean_ci(vals, confidence=0.95)
            means.append(mval)
            lowers.append(lo)
            uppers.append(up)
            ns.append(int(np.sum(~np.isnan(vals))))

        means = np.array(means, dtype=float)
        lowers = np.array(lowers, dtype=float)
        uppers = np.array(uppers, dtype=float)
        err_low = means - lowers
        err_up = uppers - means

        plt.figure(figsize=(max(6, 1.2 * len(methods)), 5))
        x = np.arange(len(methods))
        plt.errorbar(x, means, yerr=[err_low, err_up], fmt='o', capsize=5)
        plt.xticks(x, methods, rotation=45, ha='right')
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} 平均值 ±95% CI (n 如图)")
        ylim = plt.ylim()
        y_text = ylim[0] - 0.05 * (ylim[1] - ylim[0])
        for i, n in enumerate(ns):
            plt.text(i, y_text, f"n={n}", ha='center', va='top', fontsize=8)
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"{metric}_mean_ci.png")
        plt.savefig(out_png)
        plt.close()
        print(f"[INFO] 已保存 {metric} 均值±95%CI 图到: {out_png}")

    # ---- 4) 汇总表 ----
    summary_rows = []
    for metric in ["hv", "igd", "sp"]:
        methods = sorted(df['method'].unique())
        for m in methods:
            vals = df[df['method'] == m][metric].values
            mean_, lo, up = mean_ci(vals, confidence=0.95)
            summary_rows.append({
                "metric": metric,
                "method": m,
                "mean": mean_,
                "ci_lower": lo,
                "ci_upper": up,
                "n": int(np.sum(~np.isnan(vals)))
            })
    summary_df = pd.DataFrame.from_records(summary_rows)
    summary_csv = os.path.join(out_dir, "summary_mean_ci.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[INFO] 已保存均值±95%CI 汇总表到: {summary_csv}")

    return df, summary_df


# ------------------ 自动执行入口 ------------------
def try_auto_save_after_run(out_dir="results_plots"):
    """若全局变量存在 collected, reference_set, ref_point，则自动保存图和表"""
    collected = globals().get("collected", None)
    reference_set = globals().get("reference_set", None)
    ref_point = globals().get("ref_point", None)
    if collected is None or reference_set is None or ref_point is None:
        print("[INFO] 未找到 collected/reference_set/ref_point，若需导出请手动调用：")
        print("    from compare_mopso_versions_with_plots import save_per_run_and_plot")
        print("    save_per_run_and_plot(collected, ref_point, reference_set)")
        return None
    try:
        return save_per_run_and_plot(collected, ref_point, reference_set, out_dir=out_dir)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    config = Config()
    config.num_iterations = 100
    config.num_particles = 50
    num_runs = 20  # 设定运行次数

    q_table_path = "1.0-q_table4.pkl"

    # 算法注册表：MOPSO、RL自适应、RL固定动作（0..6），以及可选经典算法
    algorithms = {
        # 'MOPSO': run_plain_mopso,
        'RL-MOPSO': run_rl_mopso
    }
    for a in range(7):
        algorithms[f'RL-fixed-{a}'] = run_rl_mopso
    # algorithms[f'RL-fixed-{6}'] = run_rl_mopso
    # if run_nsga2:
    # algorithms['NSGA-II'] = run_nsga2
    # if run_moabc:
    #     algorithms['MOABC'] = run_moabc
    # if run_spea2:
    # algorithms['SPEA2'] = run_spea2

    # 过滤掉未实现的算法
    algorithms = {k: v for k, v in algorithms.items() if v is not None}

    if not algorithms:
        print("警告：未发现可运行的算法函数，请检查导入或注释。")
        exit()

    print("\n===== 运行各算法并收集前沿 (Run: {}) =====".format(num_runs))
    collected = {name: [] for name in algorithms.keys()}

    # 预定义种子，确保所有算法在相同条件下比较
    base_seed = 42
    seeds = [base_seed + i for i in range(num_runs)]
    print(f"使用种子序列: {seeds} (确保结果可重现)")

    for name, fn in algorithms.items():
        print(f">>> 正在运行 {name}...")
        for run_id in range(num_runs):
            print(f"Running {name} run {run_id + 1}/{num_runs}")
            try:
                fixed_action = int(name.split('-')[-1]) if name.startswith('RL-fixed-') else None
                da, hv_hist, _, _ = run_algorithm_once(name, fn, config, q_table_path=q_table_path,
                                                       fixed_action=fixed_action, allow_learning=False,
                                                       seed=seeds[run_id])
                collected[name].append(da)
            except Exception as e:
                # 忽略因算法未实现导致的错误
                if 'not callable' in str(e):
                    continue
                print(f"  {name} Run {run_id + 1} 出错: {e}")

    # 统一参考集
    all_da = []
    for runs in collected.values():
        for da in runs:
            if da:
                all_da.append(da)
    reference_set = build_reference_set(all_da)
    np.save('reference_set.npy', reference_set)
    print(f"\n统一参考集生成完成，包含 {len(reference_set)} 个解")

    # 统一参考点（用于HV）：基于所有收集到的解动态生成，并覆盖 config.hv_ref_point
    ref_point = build_unified_ref_point(all_da, margin=1.1)
    print(f"统一HV参考点: {ref_point}（对所有算法/运行使用相同参考点）")

    # 数据完整性检查
    check_data_integrity(collected)

    # 调试指标一致性
    debug_metrics_consistency(collected, ref_point, reference_set)

    # 基于统一参考集计算指标
    metrics = {}
    print("\n===== 计算指标（统一参考集） =====")

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

        # 提取均值和标准差
        hv_m, hv_s = metrics[name]['HV']
        igd_m, igd_s = metrics[name]['IGD']
        sp_m, sp_s = metrics[name]['SP']
        avg_time_m, avg_time_s = metrics[name]['AvgTime']
        avg_cost_m, avg_cost_s = metrics[name]['AvgCost']

        # 打印你需要的简洁单行输出 (保留你新增的 AvgTime 和 AvgCost)
        output_line = (
            f"{name}: HV={hv_m:.4f}±{hv_s:.4f}, IGD={igd_m:.4f}±{igd_s:.4f}, SP={sp_m:.4f}±{sp_s:.4f}, "
            f"AvgTime={avg_time_m:.1f}±{avg_time_s:.1f}, AvgCost={avg_cost_m:.1f}±{avg_cost_s:.1f}"
        )
        print(output_line)

        # CSV 原始数值数据
        csv_rows.append([name, hv_m, hv_s, igd_m, igd_s, sp_m, sp_s, avg_time_m, avg_time_s, avg_cost_m, avg_cost_s])

    # 导出CSV
    export_csv(csv_rows, out_path='comparison-1.0-4-c1.csv')

    # 自动保存绘图结果
    print("\n===== 生成绘图结果 =====")
    try_auto_save_after_run("results_plots_c1")