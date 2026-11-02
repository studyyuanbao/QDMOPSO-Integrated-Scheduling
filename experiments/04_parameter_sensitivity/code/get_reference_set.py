import numpy as np
from init import Config
from mopso_basic import run_mopso as run_plain_mopso
from pso_rl_new import run_mopso as run_rl_mopso

def nondominated_sort(solutions):
    """
    非支配排序，返回非支配解的索引
    """
    non_dominated = []
    for i, p in enumerate(solutions):
        dominated = False
        for j, q in enumerate(solutions):
            if i != j and dominates(q, p):  # 如果q支配p
                dominated = True
                break
        if not dominated:
            non_dominated.append(i)

    return np.array(non_dominated)


def dominates(a, b):
    """
    判断解a是否支配解b（即目标向量a在所有目标上都比b好）
    """
    return np.all(np.array(a) <= np.array(b)) and np.any(np.array(a) < np.array(b))


def get_reference_set(all_solutions):
    """
    从多个算法多次运行的解集中，生成参考集
    :param all_solutions: 所有算法多次运行结果的非支配解集
    :return: 最终参考集
    """
    # 合并所有解集
    combined_solutions = np.vstack(all_solutions)  # 如果解集是列表，转换成numpy数组

    # 进行非支配筛选
    non_dominated_indices = nondominated_sort(combined_solutions)
    reference_set = combined_solutions[non_dominated_indices]

    return reference_set


def run_times(config, num_runs=20, q_table_path=None):
    """
    使用相同的运行次数生成参考解：
    - 纯 MOPSO：直接按配置运行
    - RL-MOPSO：加载已下载的预训练模型（q_table_path），纯利用模式运行
    两者各运行 num_runs 次，然后合并生成参考集。
    """
    all_solutions = []  # 收集所有解集（来自两种算法）

    # 运行纯 MOPSO
    for run_id in range(num_runs):
        print(f"\n=== Plain MOPSO Run {run_id + 1}/{num_runs} ===")
        try:
            results = run_plain_mopso(config)
            # 旧版纯MOPSO返回元组：(_, da, ... hv_history, final_sp, igd_values)
            _, da, _, _, _, _, _, _, _, _ = results
            if da:
                all_solutions.append([p.fitness for p in da])
                print(f"Plain Run {run_id + 1}: 非支配解 {len(da)}")
            else:
                print(f"Plain Run {run_id + 1}: 无有效解")
        except Exception as e:
            print(f"Plain Run {run_id + 1} 出错: {e}")

    # 运行 RL-MOPSO（使用下载的预训练模型）
    for run_id in range(num_runs):
        print(f"\n=== RL-MOPSO Run {run_id + 1}/{num_runs} ===")
        try:
            result, _, _ = run_rl_mopso(
                config,
                total_iterations=getattr(config, 'num_iterations', 100),
                load_q_table_path=q_table_path,
                save_q_table_path=None,
                exploit_only=True
            )
            da = result['diversity_archive']
            if da:
                all_solutions.append([p.fitness for p in da])
                print(f"RL Run {run_id + 1}: 非支配解 {len(da)}")
            else:
                print(f"RL Run {run_id + 1}: 无有效解")
        except Exception as e:
            print(f"RL Run {run_id + 1} 出错: {e}")

    if not all_solutions:
        print("所有运行均无有效解，无法生成参考集")
        return None

    # 生成参考集
    try:
        combined_solutions = []
        for sol_list in all_solutions:
            combined_solutions.extend(sol_list)
        reference_set = get_reference_set(combined_solutions)
        np.save('reference_set.npy', reference_set)
        print(f"参考集生成完成，包含 {len(reference_set)} 个解")
        return reference_set
    except Exception as e:
        print(f"生成参考集时出错: {e}")
        return None


def load_reference_set(file_path='reference_set.npy'):
    """
    加载参考集
    :param file_path: 存储参考集的文件路径
    :return: 加载的参考集
    """
    return np.load(file_path)


if __name__ == '__main__':

    # 配置和算法
    config = Config()
    config.num_iterations = 100
    config.num_particles = 50
    # 使用已下载的预训练模型（与比较脚本保持一致）
    q_table_path = 'q_table13.pkl'

    print("Running MOPSO + RL-MOPSO to collect solutions and save reference set...")
    # 两个算法各跑 num_runs 次（相同次数），合并生成参考集
    reference_set = run_times(config, num_runs=2, q_table_path=q_table_path)

    # 加载之前保存的参考集
    loaded_reference_set = load_reference_set()
    print(f"Loaded Reference Set with {len(loaded_reference_set)} solutions")
