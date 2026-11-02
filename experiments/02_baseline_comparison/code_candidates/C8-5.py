import copy
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import os
from init1 import Config
from pso_rl_new1 import run_mopso as run_rl_mopso
from mopso_basic import run_mopso as run_plain_mopso
from utils import calculate_hypervolume, calculate_spread, calculate_spacing, calculate_igd  # 确保 calculate_spread 已导入
import csv
import math
import pandas as pd
from collections import defaultdict
import pathlib
import pickle

# **健壮性增强：固定 PYTHONHASHSEED**
# 确保跨进程/重启时，哈希函数的随机性被消除，提高结果的可重现性。
os.environ['PYTHONHASHSEED'] = '0'

# 可选算法
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


# ------------------ 工具函数 ------------------
def solution_to_point(sol):
    """从一个解对象或元组中提取 (time, cost)。无效返回 None。"""
    if sol is None:
        return None
    if isinstance(sol, (tuple, list)) and len(sol) >= 2:
        try:
            t = float(sol[0])
            c = float(sol[1])
            if np.isfinite(t) and np.isfinite(c):
                return (t, c)
        except:
            return None
    if hasattr(sol, 'fitness'):
        f = sol.fitness
        if isinstance(f, (list, tuple)) and len(f) >= 2:
            try:
                t = float(f[0])
                c = float(f[1])
                if np.isfinite(t) and np.isfinite(c):
                    return (t, c)
            except:
                return None
    return None


def nondominated_filter(points):
    """
    输入: list/array of (t,c) 假定目标均为越小越好
    输出: numpy array 的非支配点子集（保留所有非支配点，顺序不重要）
    """
    if not points:
        return np.empty((0, 2))
    pts = np.array(points)
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


# ------------------ 计算均值/标准差 ------------------
# **修改点 2：在 summarize_mean_std 中加入 spread_list**
def summarize_mean_std(hv_list, igd_list, sp_list, spread_list, avg_time_list, avg_cost_list):
    def ms(arr):
        arr_np = np.array(arr, dtype=float)
        return float(np.nanmean(arr_np)), float(np.nanstd(arr_np))

    return {
        'HV': ms(hv_list),
        'IGD': ms(igd_list),
        'SP': ms(sp_list),
        'Spread': ms(spread_list),  # <-- 新增
        'AvgTime': ms(avg_time_list),
        'AvgCost': ms(avg_cost_list)
    }


# ------------------ 单次运行算法（更稳健） ------------------
# **修改点 3：调整 run_algorithm_once 以兼容 RL-MOPSO 的 11 个返回值**
def run_algorithm_once(name, alg_fn, config, q_table_path=None, fixed_action=None, allow_learning=False, seed=None):
    """
    返回统一格式： (combined_archive_list, hv_history, final_sp, igd_values, spread_values)
    注意：非 RL 算法的 spread_values 为 []
    """
    # 固定随机性
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # deepcopy config 以避免跨 run 污染
    cfg = copy.deepcopy(config)

    # 默认值
    combined, hv_history, final_sp, igd_values, spread_values = [], [], 0.0, [], []

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

        # RL-MOPSO (pso_rl_new1.py) 返回 11 个元素的元组
        if isinstance(result, tuple) and len(result) >= 11:
            # 结构: (ca, da, ..., hv_history_abs[7], final_sp_calc[8], igd_values[9], spread_values[10])
            ca = result[0] or []
            da = result[1] or []
            hv_history = result[7]
            final_sp = result[8]
            igd_values = result[9]
            spread_values = result[10]  # <-- 捕获 Spread 历史记录
            combined = list(da) + list(ca)
            return combined, hv_history, final_sp, igd_values, spread_values

        elif isinstance(result, tuple) and len(result) >= 1:
            # 兼容旧版 RL-MOPSO（返回字典或旧元组）
            result_dict = result[0] if len(result) >= 1 else {}
            if isinstance(result_dict, dict):
                ca = result_dict.get('convergence_archive', []) or []
                da = result_dict.get('diversity_archive', []) or []
                hv_history = result_dict.get('hv_history', []) or []
                final_sp = result_dict.get('final_sp', 0.0)
                igd_values = result_dict.get('igd_values', []) or []
                # 旧版字典中没有 spread_values
                spread_values = []
                combined = list(da) + list(ca)
                return combined, hv_history, final_sp, igd_values, spread_values

    else:
        # 其他算法
        result = alg_fn(cfg)
        if isinstance(result, (list, tuple)):
            # 兼容 mopso_basic 的 10 个返回值
            if len(result) >= 10:
                ca = result[0] if len(result) > 0 else []
                da = result[1] if len(result) > 1 else []
                hv_history = result[7] if len(result) > 7 else []
                final_sp = result[8] if len(result) > 8 else 0.0
                igd_values = result[9] if len(result) > 9 else []
                spread_values = []  # 非 RL 算法不返回历史记录
            # 兼容其他可能返回 4 个指标的算法 (archive, hv_hist, final_sp, igd_vals)
            elif len(result) >= 4:
                ca = result[0] if len(result) > 0 else []
                da = result[0] if len(result) > 0 else []
                hv_history = result[1] if len(result) > 1 else []
                final_sp = result[2] if len(result) > 2 else 0.0
                igd_values = result[3] if len(result) > 3 else []
                spread_values = []
            else:
                pass  # 使用默认的 []

            combined = list(da) + list(ca)
            return combined, hv_history, final_sp, igd_values, spread_values

    # 如果所有分支都没有返回，使用默认值
    print(
        f"DEBUG {name}: Unexpected result type or length: {type(result)} / len={len(result) if isinstance(result, tuple) else 'N/A'}")
    return [], [], 0.0, [], []


# ------------------ 计算指标 ------------------
# **修改点 1：在 compute_metrics_for_da 中计算并返回 spread**
def compute_metrics_for_da(da, ref_point, reference_set_array):
    valid = []
    total_time = 0.0
    total_cost = 0.0

    for i, p in enumerate(da):
        pt = solution_to_point(p)
        if pt is None:
            continue
        t, c = pt
        valid.append((t, c))
        total_time += t
        total_cost += c

    n = len(valid)
    if n == 0:
        # 新增一个 NaN 用于 Spread
        return 0.0, np.nan, 0.0, np.nan, np.nan, np.nan

        # 计算HV
    try:
        hv = calculate_hypervolume(valid, ref_point)
    except Exception as e:
        hv = 0.0

    # 计算IGD
    if reference_set_array is not None and len(reference_set_array) > 0:
        try:
            da_array = np.array(valid)
            dists = []
            for ref_i in reference_set_array:
                ref_i = np.asarray(ref_i)
                if ref_i.size != 2:
                    continue
                # 兼容 NumPy 1.20+ 和旧版本
                if da_array.size > 0:
                    mins = np.min(np.linalg.norm(da_array - ref_i.reshape(1, 2), axis=1))
                    dists.append(mins)
            igd = float(np.mean(dists)) if dists else np.nan
        except Exception as e:
            igd = np.nan
    else:
        igd = np.nan

    # 计算Spacing
    try:
        sp = calculate_spacing(valid)
    except Exception as e:
        sp = 0.0

    # 计算 Spread (Delta) <-- 新增 Spread 计算
    spread = float('nan')
    if reference_set_array is not None and len(reference_set_array) > 0:
        try:
            reference_fitness = [tuple(r) for r in reference_set_array]
            spread = calculate_spread(valid, reference_fitness)
        except Exception as e:
            spread = float('nan')

    avg_time = total_time / n
    avg_cost = total_cost / n

    return hv, igd, sp, avg_time, avg_cost, spread  # <-- 返回新增的 spread


# ------------------ 构建统一参考集 ------------------
def build_reference_set(collected_da_lists):
    all_points = []
    for da in collected_da_lists:
        if not da:
            continue
        for p in da:
            pt = solution_to_point(p)
            if pt is not None:
                all_points.append(pt)
    if not all_points:
        return np.empty((0, 2))
    pts = np.array(all_points)
    # 非支配过滤
    keep = []
    for i in range(len(pts)):
        dominated = False
        for j in range(len(pts)):
            if i == j: continue
            if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return pts[keep]


# ------------------ 构建统一 HV 参考点 ------------------
def build_unified_ref_point(collected_da_lists, margin=1.1):
    max_time = None
    max_cost = None
    for da in collected_da_lists:
        if not da: continue
        for p in da:
            pt = solution_to_point(p)
            if pt is None: continue
            t, c = pt
            max_time = t if max_time is None else max(max_time, t)
            max_cost = c if max_cost is None else max(max_cost, c)
    if max_time is None or max_cost is None:
        return (20000.0, 20000.0)
    return (max_time * margin + 1.0, max_cost * margin + 1.0)


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


# ------------------ 保存指标与绘图 ------------------
# **修改点 4：在 save_per_run_and_plot 中加入 spread**
def save_per_run_and_plot(collected, reference_set, ref_point, out_dir="results_plots"):
    os.makedirs(out_dir, exist_ok=True)
    records = []
    for method, runs in collected.items():
        for ridx, da in enumerate(runs):
            # 接收新增的 spread 指标
            hv, igd, sp, avg_time, avg_cost, spread = compute_metrics_for_da(da, ref_point, reference_set)
            # 添加到记录字典中
            records.append({"method": method, "run_id": ridx, "hv": hv, "igd": igd, "sp": sp, "spread": spread,
                            "avg_time": avg_time, "avg_cost": avg_cost})

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, "per_run_metrics.csv"), index=False)

    # 箱线图和均值±CI（对空数据做保护）
    methods = sorted(df['method'].unique())
    # 循环中加入 "spread"
    for metric in ["hv", "igd", "sp", "spread"]:
        plt.figure(figsize=(max(6, 1.2 * len(methods)), 5))
        data = [df[df['method'] == m][metric].dropna().values for m in methods]
        safe_data = [d if len(d) > 0 else np.array([np.nan]) for d in data]
        plt.boxplot(safe_data, labels=methods, showmeans=True)
        # 针对 Spread 调整 ylabel
        display_metric = f"Spread (Δ)" if metric == "spread" else metric
        plt.ylabel(display_metric)
        plt.title(f"{display_metric} 箱线图")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{metric}_boxplot.png"))
        plt.close()

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
        plt.errorbar(x, np.nan_to_num(means), yerr=[np.nan_to_num(err_low), np.nan_to_num(err_up)], fmt='o', capsize=5)
        plt.xticks(x, methods, rotation=45, ha='right')
        plt.ylabel(display_metric)
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
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    cmap = plt.get_cmap("tab10")
    methods = sorted(per_method_pareto.keys())
    for i, method in enumerate(methods):
        pts = per_method_pareto[method]
        if pts is None or len(pts) == 0:
            continue
        # 按 time 排序，画线
        pts_sorted = pts[np.argsort(pts[:, 0])]
        ax.plot(pts_sorted[:, 0], pts_sorted[:, 1], label=method, color=cmap(i % 10), linewidth=2)
        ax.scatter(pts_sorted[:, 0], pts_sorted[:, 1], s=18, color=cmap(i % 10), alpha=0.9)

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


# ------------------ 主程序 ------------------
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
    # 若需要固定动作版本可按需添加
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

    # 逐算法、逐运行执行
    results_dir = pathlib.Path("results_archives")
    results_dir.mkdir(parents=True, exist_ok=True)

    for name, fn in algorithms.items():
        print(f">>> 正在运行 {name}...")
        for run_id, seed in enumerate(seeds):
            print(f"Running {name} run {run_id + 1}/{num_runs} (seed={seed})")
            fixed_action = int(name.split('-')[-1]) if name.startswith('RL-fixed-') else None

            # **修改点 5 - 运行算法：接收 spread_vals**
            da, hv_hist, final_sp, igd_vals, spread_vals = run_algorithm_once(name, fn, base_config,
                                                                              q_table_path=q_table_path,
                                                                              fixed_action=fixed_action,
                                                                              allow_learning=False, seed=seed)
            collected[name].append(da)
            # 保存单次运行结果以便复现：保存 archive + metadata
            save_obj = {
                "method": name,
                "run_id": run_id,
                "seed": seed,
                "archive": da,
                "hv_history": hv_hist,
                "final_sp": final_sp,
                "igd_values": igd_vals,
                "spread_values": spread_vals,  # <-- 保存 spread 历史记录
                "config": copy.deepcopy(base_config)
            }
            with open(results_dir / f"{name}_run{run_id}.pkl", "wb") as fh:
                pickle.dump(save_obj, fh)

    # 汇总所有的 DA 用于生成统一 reference set 与 ref point
    all_da = [da for runs in collected.values() for da in runs if da]
    reference_set = build_reference_set(all_da)
    np.save('reference_set.npy', reference_set)
    print(f"\n统一参考集生成完成，包含 {len(reference_set)} 个解")
    ref_point = build_unified_ref_point(all_da, margin=1.5)
    print(f"统一HV参考点: {ref_point}（对所有算法/运行使用相同参考点）")

    # ========== 新增：对每个算法合并其多次运行并做非支配筛选 ==========
    per_method_pareto = {}
    for name, runs in collected.items():
        # 合并该算法所有运行的解点
        merged_pts = []
        for da in runs:
            if not da:
                continue
            for sol in da:
                pt = solution_to_point(sol)
                if pt is not None:
                    merged_pts.append(pt)
        # 非支配筛选
        nd = nondominated_filter(merged_pts)
        per_method_pareto[name] = nd

    # 保存并画图
    save_and_plot_pareto_fronts(per_method_pareto, out_dir="pareto_fronts-M1-1", ref_point=ref_point,
                                scatter_all=False)

    # ========== 原有指标统计（修改：加入 Spread） ==========
    metrics = {}
    csv_rows = []
    for name, runs in collected.items():
        # **修改点 5 - 统计列表：新增 spread_list**
        hv_list, igd_list, sp_list, spread_list, avg_time_list, avg_cost_list = [], [], [], [], [], []
        for da in runs:
            # **修改点 5 - 计算指标：接收 spread**
            hv, igd, sp, avg_time, avg_cost, spread = compute_metrics_for_da(da, ref_point, reference_set)
            # **修改点 5 - 记录指标：记录 spread**
            hv_list.append(hv)
            igd_list.append(igd)
            sp_list.append(sp)
            spread_list.append(spread)
            avg_time_list.append(avg_time)
            avg_cost_list.append(avg_cost)

        # **修改点 5 - 汇总指标：传入 spread_list**
        metrics[name] = summarize_mean_std(hv_list, igd_list, sp_list, spread_list, avg_time_list, avg_cost_list)

        hv_m, hv_s = metrics[name]['HV']
        igd_m, igd_s = metrics[name]['IGD']
        sp_m, sp_s = metrics[name]['SP']
        spread_m, spread_s = metrics[name]['Spread']
        avg_time_m, avg_time_s = metrics[name]['AvgTime']
        avg_cost_m, avg_cost_s = metrics[name]['AvgCost']

        # **修改点 5 - 打印输出：加入 Spread**
        print(
            f"{name}: HV={hv_m:.4f}±{hv_s:.4f}, IGD={igd_m:.4f}±{igd_s:.4f}, SP={sp_m:.4f}±{sp_s:.4f}, Spread={spread_m:.4f}±{spread_s:.4f}, AvgTime={avg_time_m:.1f}±{avg_time_s:.1f}, AvgCost={avg_cost_m:.1f}±{avg_cost_s:.1f}")

        # **修改点 5 - CSV行数据：加入 Spread**
        csv_rows.append(
            [name, hv_m, hv_s, igd_m, igd_s, sp_m, sp_s, spread_m, spread_s, avg_time_m, avg_time_s, avg_cost_m,
             avg_cost_s])

    # 导出 CSV（统计表）
    # **修改点 5 - CSV表头：加入 Spread**
    headers = ['Method', 'HV_mean', 'HV_std', 'IGD_mean', 'IGD_std', 'SP_mean', 'SP_std', 'Spread_mean', 'Spread_std',
               'AvgTime_mean', 'AvgTime_std', 'AvgCost_mean', 'AvgCost_std']
    with open('comparison-1.0-4-M1-1.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_rows)

    # 导出 per-run 指标并绘图
    # save_per_run_and_plot 函数内部已更新，此处保持调用不变
    save_per_run_and_plot(collected, reference_set, ref_point, out_dir="results_plots_M1-1")

    print("全部完成。合并的非支配解已保存，统计结果也保存！")