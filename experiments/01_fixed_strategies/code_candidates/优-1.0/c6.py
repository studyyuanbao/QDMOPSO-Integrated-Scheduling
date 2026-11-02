import copy
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
    from moabc import run_moabc
except Exception:
    run_moabc = None
try:
    from mopso_basic import run_mopso
except Exception:
    run_mopso = None
try:
    from moaco import run_moaco
except Exception:
    run_moaco = None
try:
    from mogwo import run_mogwo
except Exception:
    run_mogwo = None

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

# ------------------ 单次运行算法（更稳健） ------------------
def run_algorithm_once(name, alg_fn, config, q_table_path=None, fixed_action=None, allow_learning=False, seed=None):
    """
    返回统一格式： (combined_archive_list, hv_history, final_sp, igd_values)
    - 对 RL 特殊处理（如果 name.startswith('RL')）
    - 对非 RL，尽量按约定的返回元组解析（容错）
    """
    # 固定随机性
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # 使用 deepcopy 的 config 副本传入算法，避免跨 run 污染
    cfg = copy.deepcopy(config)

    # RL 风格（pso_rl_new）通常有额外参数
    if name.startswith('RL'):
        # alg_fn 应该为 pso_rl_new.run_mopso
        # 它可能返回字典或元组。我们优先尝试字典格式。
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
            # 如果 alg_fn 不接受这些参数，回退为只传 cfg
            result = alg_fn(cfg)

        # 处理可能的返回格式
        if isinstance(result, dict):
            ca = result.get('convergence_archive', []) or []
            da = result.get('diversity_archive', []) or []
            hv_history = result.get('hv_history', []) or []
            final_sp = result.get('final_sp', 0.0)
            igd_values = result.get('igd_values', []) or []
            combined = list(da) + list(ca)
            return combined, hv_history, final_sp, igd_values
        elif isinstance(result, (list, tuple)):
            # 按约定解析元组（兼容旧实现）
            # 常见格式： (ca, da, all_fitness, global_min_time_list, global_min_cost_list, archive_size_history, q_table, hv_history, final_sp, igd_values)
            try:
                ca = result[0] if len(result) > 0 else []
                da = result[1] if len(result) > 1 else []
                hv_history = result[7] if len(result) > 7 else []
                final_sp = result[8] if len(result) > 8 else 0.0
                igd_values = result[9] if len(result) > 9 else []
            except Exception:
                ca, da, hv_history, final_sp, igd_values = [], [], [], 0.0, []
            combined = list(da) + list(ca)
            return combined, hv_history, final_sp, igd_values
        else:
            return [], [], 0.0, []
    else:
        # 非 RL 算法直接调用（多数返回元组）
        result = alg_fn(cfg)
        # 容错解析：期待至少前两个元素是 (convergence_archive, diversity_archive)
        if isinstance(result, (list, tuple)):
            # 尝试按常见格式解析
            ca = result[0] if len(result) > 0 else []
            da = result[1] if len(result) > 1 else []
            # hv_history 可能在索引 7
            hv_history = result[7] if len(result) > 7 else []
            final_sp = result[8] if len(result) > 8 else 0.0
            igd_values = result[9] if len(result) > 9 else []
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
        return np.empty((0,2))
    pts = np.array(all_points)
    keep = []
    for i in range(len(pts)):
        dominated = False
        for j in range(len(pts)):
            if i==j: continue
            # Pareto 支配（目标均为越小越好）
            if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                dominated = True
                break
        if not dominated: keep.append(i)
    return pts[keep]

# ------------------ 构建统一 HV 参考点 ------------------
def build_unified_ref_point(collected_da_lists, margin=1.1):
    max_time = None
    max_cost = None
    for da in collected_da_lists:
        if not da:
            continue
        for p in da:
            if hasattr(p, 'fitness') and isinstance(p.fitness, (list, tuple)) and len(p.fitness) >= 2:
                try:
                    t = float(p.fitness[0]); c = float(p.fitness[1])
                    if np.isnan(t) or np.isnan(c):
                        continue
                    max_time = t if max_time is None else max(max_time, t)
                    max_cost = c if max_cost is None else max(max_cost, c)
                except:
                    continue
    if max_time is None or max_cost is None:
        return (20000.0, 20000.0)
    return (max_time * margin + 1.0, max_cost * margin + 1.0)

# ------------------ 计算指标 ------------------
def compute_metrics_for_da(da, ref_point, reference_set_array):
    """
    返回 hv, igd, sp, avg_time, avg_cost
    注意：IGD 的计算采用 reference_set -> dist_to_solution_set 的平均
    """
    valid = []
    total_time = 0.0
    total_cost = 0.0
    for p in da:
        if hasattr(p, 'fitness') and isinstance(p.fitness, (list, tuple)) and len(p.fitness) >= 2:
            try:
                t = float(p.fitness[0]); c = float(p.fitness[1])
                if not (np.isnan(t) or np.isnan(c)):
                    valid.append((t, c))
                    total_time += t
                    total_cost += c
            except:
                continue
    n = len(valid)
    if n == 0:
        return 0.0, np.nan, 0.0, np.nan, np.nan

    # HV
    hv = calculate_hypervolume(valid, ref_point) if callable(calculate_hypervolume) else 0.0

    # IGD：reference_set_array: N x 2 array
    if reference_set_array is not None and len(reference_set_array) > 0:
        da_array = np.array(valid)
        # 对每个 reference 点，求到 da_array 的最小距离
        dists = []
        for ref_i in reference_set_array:
            ref_i = np.asarray(ref_i)
            # 形状校验
            if ref_i.size != 2:
                continue
            # 计算欧氏距离
            mins = np.min(np.linalg.norm(da_array - ref_i.reshape(1,2), axis=1))
            dists.append(mins)
        igd = float(np.mean(dists)) if dists else np.nan
    else:
        igd = np.nan

    # SP
    sp = calculate_spacing(valid) if callable(calculate_spacing) else 0.0

    avg_time = total_time / n
    avg_cost = total_cost / n
    return hv, igd, sp, avg_time, avg_cost

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
        # 确保每个方法至少有一个数据点，boxplot 要求这样
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
        # 对 NaN 值做保护（matplotlib 会失败）
        plt.errorbar(x, np.nan_to_num(means), yerr=[np.nan_to_num(err_low), np.nan_to_num(err_up)], fmt='o', capsize=5)
        plt.xticks(x, methods, rotation=45, ha='right'); plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{metric}_mean_ci.png"))
        plt.close()

    return df

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
    # for a in range(7): algorithms[f'RL-fixed-{a}'] = run_rl_mopso
    if run_moabc:
        algorithms['MOABC'] = run_moabc
    if run_moaco:
        algorithms['MOACO'] = run_moaco
    if run_plain_mopso:
        algorithms['MOPSO'] = run_plain_mopso
    if run_mogwo:
        algorithms['MOGWO'] = run_mogwo

    if not algorithms:
        raise SystemExit("未发现可运行算法")

    collected = {name: [] for name in algorithms.keys()}
    base_seed = 42
    seeds = [base_seed + i for i in range(num_runs)]
    print(f"使用种子序列: {seeds} (确保结果可重现)")

    # 逐算法、逐运行执行（每次传入 deepcopy(config) 以避免互相污染）
    for name, fn in algorithms.items():
        print(f">>> 正在运行 {name}...")
        for run_id, seed in enumerate(seeds):
            print(f"Running {name} run {run_id + 1}/{num_runs} (seed={seed})")
            fixed_action = int(name.split('-')[-1]) if name.startswith('RL-fixed-') else None
            # 注意：这里 run_algorithm_once 会 deepcopy 一份 config，所以传 base_config 也可以
            da, hv_hist, final_sp, igd_vals = run_algorithm_once(name, fn, base_config, q_table_path=q_table_path,
                                                                 fixed_action=fixed_action, allow_learning=False, seed=seed)
            collected[name].append(da)

    # 汇总所有的 DA 用于生成统一 reference set 与 ref point
    all_da = [da for runs in collected.values() for da in runs if da]
    reference_set = build_reference_set(all_da)
    np.save('reference_set.npy', reference_set)
    print(f"\n统一参考集生成完成，包含 {len(reference_set)} 个解")
    ref_point = build_unified_ref_point(all_da, margin=1.1)
    print(f"统一HV参考点: {ref_point}（对所有算法/运行使用相同参考点）")

    metrics = {}
    csv_rows = []
    for name, runs in collected.items():
        hv_list, igd_list, sp_list, avg_time_list, avg_cost_list = [], [], [], [], []
        for da in runs:
            hv, igd, sp, avg_time, avg_cost = compute_metrics_for_da(da, ref_point, reference_set)
            hv_list.append(hv); igd_list.append(igd); sp_list.append(sp); avg_time_list.append(avg_time); avg_cost_list.append(avg_cost)
        metrics[name] = summarize_mean_std(hv_list, igd_list, sp_list, avg_time_list, avg_cost_list)
        hv_m, hv_s = metrics[name]['HV']; igd_m, igd_s = metrics[name]['IGD']; sp_m, sp_s = metrics[name]['SP']
        avg_time_m, avg_time_s = metrics[name]['AvgTime']; avg_cost_m, avg_cost_s = metrics[name]['AvgCost']
        print(f"{name}: HV={hv_m:.4f}±{hv_s:.4f}, IGD={igd_m:.4f}±{igd_s:.4f}, SP={sp_m:.4f}±{sp_s:.4f}, AvgTime={avg_time_m:.1f}±{avg_time_s:.1f}, AvgCost={avg_cost_m:.1f}±{avg_cost_s:.1f}")
        csv_rows.append([name, hv_m, hv_s, igd_m, igd_s, sp_m, sp_s, avg_time_m, avg_time_s, avg_cost_m, avg_cost_s])

    # 导出 CSV
    headers = ['Method', 'HV_mean', 'HV_std', 'IGD_mean', 'IGD_std', 'SP_mean', 'SP_std', 'AvgTime_mean', 'AvgTime_std', 'AvgCost_mean', 'AvgCost_std']
    with open('comparison-1.0-4-all-c6.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f); writer.writerow(headers); writer.writerows(csv_rows)

    # 导出绘图
    save_per_run_and_plot(collected, reference_set, ref_point, out_dir="results_plots_all-c6")
