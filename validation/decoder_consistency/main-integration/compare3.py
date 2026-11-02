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
import math
import pandas as pd

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

# 中文字体设置
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False


# ------------------ 计算均值/标准差 ------------------
def summarize_mean_std(hv_list, igd_list, sp_list, avg_time_list, avg_cost_list):
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


# ------------------ 单次运行算法 ------------------
def run_algorithm_once(name, alg_fn, config, q_table_path=None, fixed_action=None, allow_learning=False, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    if name.startswith('RL'):
        result, _, _ = run_rl_mopso(
            config,
            total_iterations=getattr(config, 'num_iterations', 100),
            load_q_table_path=q_table_path,
            save_q_table_path=None,
            exploit_only=not allow_learning,
            fixed_action=fixed_action
        )
        return result.get('convergence_archive', []) + result.get('diversity_archive', []), \
            result.get('hv_history', []), result.get('final_sp', 0.0), result.get('igd_values', [])
    else:
        run_result = alg_fn(config)
        ca, da, *_, hv_history, final_sp, igd_values = run_result[:10]
        return da + ca, hv_history, final_sp, igd_values


# ------------------ 构建统一参考集 ------------------
def build_reference_set(collected_da_lists):
    all_points = []
    for da in collected_da_lists:
        if not da:
            continue
        for p in da:
            if hasattr(p, 'fitness') and isinstance(p.fitness, (list, tuple)) and len(p.fitness) == 2:
                try:
                    t, c = float(p.fitness[0]), float(p.fitness[1])
                    if not (np.isnan(t) or np.isnan(c)):
                        all_points.append((t, c))
                except:
                    continue
    if not all_points:
        return np.empty((0, 2))
    pts = np.array(all_points)
    keep = []
    for i in range(len(pts)):
        dominated = False
        for j in range(len(pts)):
            if i == j: continue
            if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                dominated = True
                break
        if not dominated: keep.append(i)
    return pts[keep]


# ------------------ 构建统一 HV 参考点 ------------------
def build_unified_ref_point(collected_da_lists, margin=1.1):
    max_time = max_cost = None
    for da in collected_da_lists:
        if not da: continue
        for p in da:
            if hasattr(p, 'fitness') and len(p.fitness) == 2:
                try:
                    t, c = float(p.fitness[0]), float(p.fitness[1])
                    if not (np.isnan(t) or np.isnan(c)):
                        max_time = t if max_time is None else max(max_time, t)
                        max_cost = c if max_cost is None else max(max_cost, c)
                except:
                    continue
    if max_time is None or max_cost is None:
        return (20000.0, 20000.0)
    return (max_time * margin + 1.0, max_cost * margin + 1.0)


# ------------------ 计算指标 ------------------
def compute_metrics_for_da(da, ref_point, reference_set_array):
    valid = []
    total_time = total_cost = 0.0
    for p in da:
        if hasattr(p, 'fitness') and len(p.fitness) == 2:
            try:
                t, c = float(p.fitness[0]), float(p.fitness[1])
                if not (np.isnan(t) or np.isnan(c)):
                    valid.append((t, c))
                    total_time += t
                    total_cost += c
            except:
                continue
    n = len(valid)
    if n == 0: return 0.0, np.nan, 0.0, np.nan, np.nan
    hv = calculate_hypervolume(valid, ref_point)
    if len(reference_set_array) > 0:
        da_array = np.array(valid)
        igd_distances = [np.min(np.linalg.norm(da_array - np.array(ref_i), axis=1)) for ref_i in reference_set_array]
        igd = np.mean(igd_distances)
    else:
        igd = np.nan
    sp = calculate_spacing(valid)
    return hv, igd, sp, total_time / n, total_cost / n


# ------------------ 均值±95%CI ------------------
try:
    from scipy.stats import t

    _HAS_SCIPY = True
except:
    _HAS_SCIPY = False


def mean_ci(values, confidence=0.95):
    vals = np.array(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n == 0: return np.nan, np.nan, np.nan
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


# ------------------ 保存指标与绘图 ------------------
def save_per_run_and_plot(collected, reference_set, ref_point, out_dir="results_plots"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 创建输出目录: {os.path.abspath(out_dir)}")

    records = []
    for method, runs in collected.items():
        for ridx, da in enumerate(runs):
            hv, igd, sp, avg_time, avg_cost = compute_metrics_for_da(da, ref_point, reference_set)
            records.append({"method": method, "run_id": ridx, "hv": hv, "igd": igd, "sp": sp, "avg_time": avg_time,
                            "avg_cost": avg_cost})

    # 保存详细指标CSV
    csv_path = os.path.join(out_dir, "per_run_metrics.csv")
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"💾 详细运行指标已保存至: {os.path.abspath(csv_path)}")

    # 箱线图和均值±CI
    plot_files = []
    for metric in ["hv", "igd", "sp"]:
        # 箱线图
        plt.figure(figsize=(max(6, 1.2 * len(df['method'].unique())), 5))
        methods = sorted(df['method'].unique())
        data = [df[df['method'] == m][metric].dropna().values for m in methods]

        # 修复：将 labels 改为 tick_labels
        box_plot = plt.boxplot(data, tick_labels=methods, showmeans=True)
        plt.ylabel(metric)
        plt.title(f"{metric.upper()} 箱线图")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        boxplot_path = os.path.join(out_dir, f"{metric}_boxplot.png")
        plt.savefig(boxplot_path)
        plt.close()
        plot_files.append(boxplot_path)

        # 均值±95%CI
        means, lowers, uppers, ns = [], [], [], []
        for m in methods:
            vals = df[df['method'] == m][metric].values
            mval, lo, up = mean_ci(vals)
            means.append(mval)
            lowers.append(lo)
            uppers.append(up)
            ns.append(int(np.sum(~np.isnan(vals))))
        means = np.array(means)
        lowers = np.array(lowers)
        uppers = np.array(uppers)
        err_low = means - lowers
        err_up = uppers - means
        plt.figure(figsize=(max(6, 1.2 * len(methods)), 5))
        x = np.arange(len(methods))
        plt.errorbar(x, means, yerr=[err_low, err_up], fmt='o', capsize=5)
        plt.xticks(x, methods, rotation=45, ha='right')
        plt.ylabel(metric)
        plt.tight_layout()
        mean_ci_path = os.path.join(out_dir, f"{metric}_mean_ci.png")
        plt.savefig(mean_ci_path)
        plt.close()
        plot_files.append(mean_ci_path)

    print(f"🖼️  图表文件已保存至 {out_dir} 目录:")
    for plot_file in plot_files:
        print(f"   - {os.path.basename(plot_file)}")

    return df


# ------------------ 主程序 ------------------
if __name__ == '__main__':
    config = Config()
    config.num_iterations = 100
    config.num_particles = 50
    num_runs = 20
    q_table_path = "1.0-q_table4.pkl"

    algorithms = {'RL-MOPSO': run_rl_mopso}
    for a in range(7): algorithms[f'RL-fixed-{a}'] = run_rl_mopso
    algorithms = {k: v for k, v in algorithms.items() if v is not None}
    if not algorithms: exit("未发现可运行算法")

    collected = {name: [] for name in algorithms.keys()}
    base_seed = 42
    seeds = [base_seed + i for i in range(num_runs)]
    print(f"🌱 使用种子序列: {seeds} (确保结果可重现)")

    for name, fn in algorithms.items():
        print(f"\n🚀 正在运行 {name}...")
        for run_id, seed in enumerate(seeds):
            print(f"  运行进度: {name} 第 {run_id + 1}/{num_runs} 次")
            fixed_action = int(name.split('-')[-1]) if name.startswith('RL-fixed-') else None
            da, hv_hist, _, _ = run_algorithm_once(name, fn, config, q_table_path=q_table_path,
                                                   fixed_action=fixed_action, allow_learning=False, seed=seed)
            collected[name].append(da)

    print(f"\n📊 正在构建参考集...")
    all_da = [da for runs in collected.values() for da in runs if da]
    reference_set = build_reference_set(all_da)

    # 保存参考集
    ref_set_path = 'reference_set.npy'
    np.save(ref_set_path, reference_set)
    print(f"💾 参考集已保存至: {os.path.abspath(ref_set_path)}")
    print(f"   参考集包含 {len(reference_set)} 个非支配解")

    ref_point = build_unified_ref_point(all_da, margin=1.1)
    print(f"📐 HV参考点: {ref_point}")

    print(f"\n📈 正在计算性能指标...")
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
        print(f"📊 {name}:")
        print(f"   HV={hv_m:.4f}±{hv_s:.4f}")
        print(f"   IGD={igd_m:.4f}±{igd_s:.4f}")
        print(f"   SP={sp_m:.4f}±{sp_s:.4f}")
        print(f"   平均时间={avg_time_m:.1f}±{avg_time_s:.1f}")
        print(f"   平均成本={avg_cost_m:.1f}±{avg_cost_s:.1f}")
        csv_rows.append([name, hv_m, hv_s, igd_m, igd_s, sp_m, sp_s, avg_time_m, avg_time_s, avg_cost_m, avg_cost_s])

    # 导出 CSV
    csv_filename = 'comparison-1.0-4-c3.csv'
    headers = ['Method', 'HV_mean', 'HV_std', 'IGD_mean', 'IGD_std', 'SP_mean', 'SP_std', 'AvgTime_mean', 'AvgTime_std',
               'AvgCost_mean', 'AvgCost_std']
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_rows)
    print(f"\n💾 汇总统计已保存至: {os.path.abspath(csv_filename)}")

    # 导出绘图
    print(f"\n🎨 正在生成图表...")
    plot_dir = "results_plots_c3"
    save_per_run_and_plot(collected, reference_set, ref_point, out_dir=plot_dir)

    print(f"\n✅ 所有数据保存完成!")
    print(f"📂 主要输出文件:")
    print(f"   - {os.path.abspath(csv_filename)}")
    print(f"   - {os.path.abspath(ref_set_path)}")
    print(f"   - {os.path.abspath(plot_dir)}/ 目录下的所有图表文件")