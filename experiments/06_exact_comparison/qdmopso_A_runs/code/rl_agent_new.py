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
        q_values = self.q_table.get(state, [0.0] * len(self.actions))
        # if all Qs equal, break ties randomly to encourage exploration
        if all(q == q_values[0] for q in q_values):
            return np.random.choice(self.actions)
        return self.actions[int(np.argmax(q_values))]

    def update(self, state, action, reward, next_state, terminal=False):
        a_idx = self.actions.index(action)
        if self.n_step <= 1:
            q_predict = self.q_table[state][a_idx]
            q_target = reward if terminal else reward + self.gamma * max(self.q_table[next_state])
            self.q_table[state][a_idx] += self.alpha * (q_target - q_predict)
        else:
            self._traj_buffer.append((state, a_idx, reward, next_state))
            if terminal:
                # Flush all remaining returns without bootstrapping beyond the episode.
                while self._traj_buffer:
                    G = sum((self.gamma ** i) * item[2] for i, item in enumerate(self._traj_buffer))
                    s0, a0_idx, _, _ = self._traj_buffer.pop(0)
                    self.q_table[s0][a0_idx] += self.alpha * (G - self.q_table[s0][a0_idx])
                return
            if len(self._traj_buffer) >= self.n_step:
                G = 0.0
                for i in range(self.n_step):
                    G += (self.gamma ** i) * self._traj_buffer[i][2]
                s0, a0_idx, _, sn = self._traj_buffer[0]
                q_predict = self.q_table[s0][a0_idx]
                q_target = G + (self.gamma ** self.n_step) * max(self.q_table[sn])
                self.q_table[s0][a0_idx] += self.alpha * (q_target - q_predict)
                self._traj_buffer.pop(0)

    def get_state_indicators(self, hv_now, spacing_now, progress):
        """Candidate: raw-HV deviations against previous short/long EWMAs.

        Thresholds retain their existing values and require later calibration.
        A missing/negative/nonfinite observation does not advance that baseline.
        Zero-to-positive uses a finite +1 sentinel, positive-to-zero gives -1.
        """
        def valid(value):
            return value is not None and np.isfinite(value) and value >= 0
        def relative(value, baseline):
            if baseline is None: return 0.0
            if baseline == 0: return 0.0 if value == 0 else 1.0
            return (value - baseline) / baseline
        short = getattr(self, 'hv_raw_short', None)
        long = getattr(self, 'hv_raw_long', None)
        if valid(hv_now):
            value = float(hv_now)
            ds, dl = relative(value, short), relative(value, long)
            hv_slope = 2 if ds > 0.02 else (0 if ds < -0.02 else 1)
            hv_level = next((i for i, threshold in enumerate([-0.20, -0.05, 0.05, 0.20, 0.50, 1.0]) if dl < threshold), 6)
            self.hv_raw_short = value if short is None else self.ewma_alpha * value + (1-self.ewma_alpha)*short
            self.hv_raw_long = value if long is None else 0.05*value + 0.95*long
        else:
            ds = dl = 0.0
            hv_slope, hv_level = 1, 2
        self.hv_short_deviation, self.hv_long_deviation = ds, dl
        baseline = getattr(self, 'sp_baseline_ewma', None)
        previous = self.prev_spacing
        if valid(spacing_now):
            value = float(spacing_now)
            delta = relative(value, previous)
            sp_trend = 1 if delta < -0.001 else (-1 if delta > 0.001 else 0)
            deviation = relative(value, baseline)
            sp_level = 2 if deviation < -0.05 else (0 if deviation > 0.05 else 1)
            self.sp_baseline_ewma = value if baseline is None else 0.05*value + 0.95*baseline
            self.prev_spacing = value
        else:
            sp_trend, sp_level = 0, 1
        progress_bucket = max(0, min(int(float(progress)*5),4)) if progress is not None and np.isfinite(progress) else 0
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

        # Bounded symmetric improvement; unavailable consecutive values give 0.
        d_sp = 0.0
        if (sp_prev is not None and sp_now is not None
                and np.isfinite(sp_prev) and np.isfinite(sp_now)
                and sp_prev >= 0 and sp_now >= 0):
            denominator = max(sp_prev, sp_now)
            if denominator > 0:
                d_sp = (sp_prev - sp_now) / denominator
        self.last_reward_components = {
            'hv': float(w_hv * d_hv), 'spacing': float(w_sp * d_sp),
            'switch_penalty': 0.0,
        }

        # 组合奖励
        reward = w_hv * d_hv + w_sp * d_sp

        # 动作切换惩罚
        if hasattr(self, "last_action") and self.last_action is not None and action != self.last_action:
            penalty = getattr(self, "switch_penalty", 0.5)  # 默认0.5
            reward -= penalty
            self.last_reward_components['switch_penalty'] = float(penalty)
        self.last_action = action
        # 在此处也同步更新上一代原始HV，确保与外部一致
        self.prev_hv_raw = hv_now

        self.last_reward_components['unclipped'] = float(reward)
        # 限幅
        reward = float(np.clip(reward, -20.0, 20.0))
        return reward

    # helper for diagnostics
    def flatten_qtable_values(self):
        return [q for qs in self.q_table.values() for q in qs]
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)