from collections import defaultdict
import numpy as np
# -------------------------
# Q-Learning Agent (fixed)
# -------------------------
class QLearningAgent:
    def __init__(self, actions, alpha=0.2, gamma=0.95, epsilon=0.5,
                 epsilon_min=0.1, epsilon_decay=0.98, total_iterations=1000,
                 weight_time=0.5, weight_cost=0.5,
                 ewma_alpha=0.2, switch_penalty=0.01, n_step: int = 0):
        self.actions = actions
        # Q-table initialised to slightly positive values to encourage exploitation after some learning
        self.q_table = defaultdict(lambda: [0.0 for _ in actions])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # keep total_iterations for optional diagnostics (not required for core Q-learning)
        self.total_iterations = total_iterations

        # tracking previous indicators (used in calculate_reward_based_on_improvement)
        self.prev_best_time = float('inf')
        self.prev_best_cost = float('inf')
        self.prev_hv = 0.0
        self.prev_igd = None
        self.prev_spacing = None
        self.prev_diversity = 0

        # history windows for smoothing baseline behaviour
        self.time_improv_history = []
        self.cost_improv_history = []
        self.hv_improv_history = []
        self.history_window = 50

        # cumulative reward (reset by caller per-episode)
        self.cumulative_reward = 0.0

        # scoring weights for state creation and (optionally) reward composition
        self.weight_time = weight_time
        self.weight_cost = weight_cost

        # global bests for normalization in state construction
        self.global_best_time = float('inf')
        self.global_best_cost = float('inf')

        # EWMA baseline for reward denoising and action-switch penalty
        self.ewma_alpha = ewma_alpha
        self.baseline_hv = 0.0
        self.switch_penalty = switch_penalty
        self.last_action = None
        
        # HV baseline-ratio 设置
        self.hv_baseline = None
        self.hv_baseline_samples = []
        self.hv_baseline_K = 5         # 前K代样本（可调）
        self.hv_levels = 7             # 等级数（0..6）
        self.hv_factor_min = 0.3
        self.hv_factor_max = 2.0
        self.hv_bins_fixed = None      # list of thresholds on log1p(hv) scale

        # SP（spacing）设置
        self.sp_levels = 3             # 等级数（0..2） 越大越好 -> higher level
        self.sp_bins_fixed = None      # list of thresholds (raw spacing scale)

        # 小辅助
        self.hv_history_norm = []      # 存 log1p(hv) 历史（如你已有可复用）
        self.sp_history = []           # 存 spacing 历史
        self.hv_window = 200           # 窗口大小，可调整（100~500 视迭代数）
        self.sp_window = 200
        self.hv_ewma = None

        self.prev_spacing = None

        # optional n-step return support (0 disables)
        self.n_step = max(0, int(n_step))
        self._traj_buffer = []  # (state, action_idx, reward, next_state)

    def choose_action(self, state):
        # epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        q_values = self.q_table[state]
        # if all Qs equal, break ties randomly to encourage exploration
        if all(q == q_values[0] for q in q_values):
            return np.random.choice(self.actions)
        return self.actions[int(np.argmax(q_values))]

    def update(self, state, action, reward, next_state):
        a_idx = self.actions.index(action)
        if self.n_step <= 1:
            q_predict = self.q_table[state][a_idx]
            q_target = reward + self.gamma * max(self.q_table[next_state])
            self.q_table[state][a_idx] += self.alpha * (q_target - q_predict)
        else:
            self._traj_buffer.append((state, a_idx, reward, next_state))
            if len(self._traj_buffer) >= self.n_step:
                G = 0.0
                for i in range(self.n_step):
                    G += (self.gamma ** i) * self._traj_buffer[i][2]
                s0, a0_idx, _, sn = self._traj_buffer[0]
                q_predict = self.q_table[s0][a0_idx]
                q_target = G + (self.gamma ** self.n_step) * max(self.q_table[sn])
                self.q_table[s0][a0_idx] += self.alpha * (q_target - q_predict)
                self._traj_buffer.pop(0)

    def get_state_indicators(self, hv_now: float, spacing_now: float, progress: float):
        """
        更稳健的状态特征构造：
        返回 (hv_slope, sp_trend, progress_bucket, hv_level, sp_level)

        设计要点：
        - 在 log1p 空间对 HV 做尺度化，但所有阈值/比值均使用相对变化（避免绝对量纲问题）
        - 使用两个不同的基线：短期 EWMA (hv_ewma) 用于斜率判定，长期 baseline_ewma 用于等级划分
        - 防止除零 / NaN，全部使用 eps 保护
        - spacing 的等级与趋势基于相对改变量（越小越好），并对 best_spacing 接近 0 做保护
        """

        eps = 1e-9

        # --- 1) 处理输入并缩放 HV（log1p）与 spacing 的基本合法化 ---
        hv_scaled = 0.0
        if hv_now is not None and not np.isnan(hv_now):
            hv_scaled = float(np.log1p(max(hv_now, 0.0)))
        # spacing 允许为 None/nan 的情况
        sp_val = None
        if spacing_now is not None and not np.isnan(spacing_now):
            sp_val = float(spacing_now)

        # 维护历史队列（保持兼容：原来使用 list）
        self.hv_history_norm.append(hv_scaled)
        self.sp_history.append(sp_val)
        if hasattr(self, "hv_window") and len(self.hv_history_norm) > self.hv_window:
            self.hv_history_norm.pop(0)
        if hasattr(self, "sp_window") and len(self.sp_history) > self.sp_window:
            self.sp_history.pop(0)

        # --- 2) 短期 EWMA（用于斜率/趋势判定） ---
        if self.hv_ewma is None:
            self.hv_ewma = hv_scaled
        else:
            self.hv_ewma = self.ewma_alpha * hv_scaled + (1.0 - self.ewma_alpha) * self.hv_ewma

        # 相对偏差（相对于 hv_ewma 的比值/相对差），更稳健
        rel_to_ewma = 0.0
        if abs(self.hv_ewma) > eps:
            rel_to_ewma = (hv_scaled - self.hv_ewma) / (abs(self.hv_ewma) + eps)

        # hv_slope 离散化：2 上升，1 平稳，0 下降
        if rel_to_ewma > 0.02:  # >2% 相对提升视为上升
            hv_slope = 2
        elif rel_to_ewma < -0.02:  # <-2% 视为下降
            hv_slope = 0
        else:
            hv_slope = 1

        # --- 3) spacing 趋势（越小越好）---
        if self.prev_spacing is None or sp_val is None:
            sp_trend = 0
        else:
            # 用相对变化判断（相对于上一代 spacing）
            denom = abs(self.prev_spacing) + eps
            rel_sp_delta = (sp_val - self.prev_spacing) / denom
            # 如果相对减少超过 0.001 (0.1%) 认为是改进
            if rel_sp_delta < -1e-3:
                sp_trend = 1  # 改善（变小）
            elif rel_sp_delta > 1e-3:
                sp_trend = -1  # 恶化（变大）
            else:
                sp_trend = 0  # 基本不变

        # --- 4) progress bucket（保持 5 段） ---
        try:
            progress_bucket = int(progress * 5)
        except Exception:
            progress_bucket = 0
        progress_bucket = max(0, min(progress_bucket, 4))

        # --- 5) 初始化/更新历史最优（保持原来语义：best_hv_scaled / best_spacing） ---
        if not hasattr(self, 'best_hv_scaled'):
            self.best_hv_scaled = -np.inf
        if not hasattr(self, 'best_spacing'):
            self.best_spacing = np.inf

        if hv_scaled is not None and not np.isnan(hv_scaled):
            if hv_scaled > self.best_hv_scaled:
                self.best_hv_scaled = hv_scaled

        if sp_val is not None:
            if sp_val < self.best_spacing:
                self.best_spacing = sp_val

        # --- 6) 长期 baseline（EWMA）用于 hv_level（解决“历史最优导致等级总是最低”的问题） ---
        # 确保在 __init__ 中已创建 self.hv_baseline_ewma 和 self.hv_baseline_alpha
        if not hasattr(self, "hv_baseline_ewma"):
            # 容错：若用户未在 __init__ 放置初始化，则在此创建
            self.hv_baseline_ewma = hv_scaled
            self.hv_baseline_alpha = getattr(self, "hv_baseline_alpha", 0.05)
        else:
            # 只有在 hv_scaled 有意义时才更新 baseline
            if hv_scaled is not None:
                self.hv_baseline_ewma = (self.hv_baseline_alpha * hv_scaled +
                                         (1.0 - self.hv_baseline_alpha) * self.hv_baseline_ewma)

        # 计算相对于长期 baseline 的相对差（比例）
        hv_rel_to_baseline = 0.0
        if self.hv_baseline_ewma is not None and abs(self.hv_baseline_ewma) > eps:
            hv_rel_to_baseline = (hv_scaled - self.hv_baseline_ewma) / (abs(self.hv_baseline_ewma) + eps)

        # hv_level 划分：使用相对差阈值（更直观）
        # 7 等级 (0..6)，阈值可调整
        if self.hv_baseline_ewma is None:
            hv_level = 0
        else:
            # 阈值：显著下降 / 轻微下降 / 基线附近 / 轻微上升 / 明显上升 / 非常明显上升
            if hv_rel_to_baseline < -0.20:
                hv_level = 0
            elif hv_rel_to_baseline < -0.05:
                hv_level = 1
            elif hv_rel_to_baseline < 0.05:
                hv_level = 2
            elif hv_rel_to_baseline < 0.20:
                hv_level = 3
            elif hv_rel_to_baseline < 0.50:
                hv_level = 4
            elif hv_rel_to_baseline < 1.0:
                hv_level = 5
            else:
                hv_level = 6

        # --- 7) spacing level（稳健 3-level 实现；保持原语义：0 差，1 中，2 优）---
        if sp_val is None or np.isinf(self.best_spacing) or np.isnan(self.best_spacing):
            sp_level = 1  # 中间等级（信息不足）
        else:
            # 用相对于 best_spacing 的比值（保护除零）
            sp_ratio = sp_val / (self.best_spacing + eps)
            # 设计：sp_ratio >= 1 -> 比历史最优差；0.8~1 稍差；<0.8 明显优
            if sp_ratio >= 1.0:
                sp_level = 0
            elif sp_ratio >= 0.8:
                sp_level = 1
            else:
                sp_level = 2

        # --- 8) 更新 prev 并返回 ---
        self.prev_spacing = sp_val
        self.prev_hv_raw = hv_now
        self.prev_hv = hv_scaled

        return (hv_slope, sp_trend, progress_bucket, hv_level, sp_level)

    def calculate_composite_reward(self,
                                   hv_prev: float, hv_now: float,
                                   sp_prev: float, sp_now: float,
                                   action,
                                   w_hv: float = 0.8, w_sp: float = 0.2):
        """
        简洁版组合奖励函数（固定权重）。
        - HV: log 比例差分，更稳健
        - Spacing: 越小越好
        - 动作切换惩罚
        """
        eps = 1e-9

        # HV（用 log 稳定化；期望传入原始HV）
        d_hv = 0.0
        if hv_prev is not None and hv_now is not None and not np.isnan(hv_prev) and not np.isnan(hv_now):
            if hv_prev < 1e-6:
                # 用绝对差或相对差的受限版本
                d_hv = hv_now - (hv_prev if hv_prev is not None else 0.0)
            else:
                d_hv = np.log((hv_now + eps) / (hv_prev + eps))

        # Spacing（越小越好）
        d_sp = 0.0
        if sp_prev is not None and sp_now is not None and not np.isnan(sp_prev) and not np.isnan(sp_now):
            d_sp = (sp_prev - sp_now) / (abs(sp_prev) + eps)

        # 组合奖励
        reward = w_hv * d_hv + w_sp * d_sp

        # 动作切换惩罚
        if hasattr(self, "last_action") and self.last_action is not None and action != self.last_action:
            penalty = getattr(self, "switch_penalty", 0.5)  # 默认0.5
            reward -= penalty
        self.last_action = action
        # 在此处也同步更新上一代原始HV，确保与外部一致
        self.prev_hv_raw = hv_now

        # 限幅
        reward = float(np.clip(reward, -20.0, 20.0))
        return reward

    # helper for diagnostics
    def flatten_qtable_values(self):
        return [q for qs in self.q_table.values() for q in qs]
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)