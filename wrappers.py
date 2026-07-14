"""Gymnasium observation and reward wrappers for the simplex environments.

- ``CompactObsWrapper`` replaces the full-tableau Dict observation with a small
  size-independent feature vector (so one policy generalizes across LP sizes).
- ``EmptyObsWrapper`` replaces the observation with a single constant feature,
  giving an information-free control baseline.
- ``BaselineRewardWrapper`` shapes the terminal reward by the pivot-count
  difference against a fixed baseline heuristic.

NOTE: ``CompactObsWrapper``'s feature layout is load-bearing — the compact
models encode these exact dimensions, so it must not change.
"""

import gymnasium as gym
import numpy as np
from collections import deque
from gymnasium import spaces

from simplex_solver import _pivot_col_heuristics, _pivot_row, _apply_pivot


class BaselineRewardWrapper(gym.Wrapper):
    """Shape reward by a reference strategy's iteration count on the same instance.

    On reset, runs `baseline_strategy` on a copy of the starting tableau and
    records its iteration count. On optimal termination, adds
    `coef * (baseline_nit - agent_nit)` (symmetric) or
    `coef * max(0, baseline_nit - agent_nit)` (wins_only) to the final reward.

    With `wins_only=False` (default), ties are neutral, wins positive, losses
    negative — which can push the agent toward "imitate the baseline" to
    avoid the negative tail. With `wins_only=True`, losses cost nothing
    extra beyond the per-step penalty, so the agent can safely gamble on
    non-baseline strategies in hopes of a win.
    """

    def __init__(self, env, baseline_strategy='steepest_edge', coef=5.0,
                 wins_only=False):
        super().__init__(env)
        self.baseline_strategy = baseline_strategy
        self.coef = float(coef)
        self.wins_only = bool(wins_only)
        self._baseline_nit = None

    def _compute_baseline(self):
        base = self.env.unwrapped
        T = base.T.copy()
        basis = base.basis.copy()
        tol = base.tol
        maxiter = getattr(base, "maxiter", 20000)
        use_bland = (self.baseline_strategy == 'blands_rule')
        nit = 0
        while nit < maxiter:
            ok, col = _pivot_col_heuristics(T, strategy=self.baseline_strategy, tol=tol)
            if not ok:
                return nit
            ok, row = _pivot_row(T, basis, col, phase=2, tol=tol, bland=use_bland)
            if not ok:
                return nit
            _apply_pivot(T, basis, row, col, tol=tol)
            if not np.isfinite(T[-1, -1]) or np.any(~np.isfinite(T)):
                return nit
            nit += 1
        return nit

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._baseline_nit = self._compute_baseline()
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if done and info.get("status") == "optimal" and self._baseline_nit is not None:
            agent_nit = int(info.get("nit", 0))
            diff = self._baseline_nit - agent_nit
            if self.wins_only:
                diff = max(0, diff)
            reward += self.coef * diff
        return obs, reward, done, truncated, info


class CompactObsWrapper(gym.ObservationWrapper):
    """Size-independent flat observation for the simplex pivot-strategy agent.

    Replaces the underlying Dict observation (which scales with LP dimensions)
    with a fixed-size feature vector built from:
      - last K actions (one-hot)
      - last K normalized delta-objective values
      - current objective and progress scalars
      - aggregate reduced-cost distribution stats

    This lets the same policy generalize across LP sizes.
    """

    def __init__(self, env, history_len: int = 5, num_strategies: int = None):
        super().__init__(env)
        self.history_len = int(history_len)
        if num_strategies is None:
            from config import NUM_PIVOT_STRATEGIES
            num_strategies = NUM_PIVOT_STRATEGIES
        self.num_strategies = int(num_strategies)

        # Feature layout:
        #   history_len * num_strategies : action history one-hot (oldest first)
        #   history_len                  : delta-obj history (normalized)
        #   3                            : current obj, nit_norm, degenerate_streak_norm
        #   4                            : reduced-cost stats (frac_neg, min, mean, std)
        #   2                            : candidate-column norm stats (mean, relative spread)
        #   2                            : ratio-test stats (min_ratio, frac_rows_at_min)
        self._n_features = (
            self.history_len * self.num_strategies
            + self.history_len
            + 3
            + 4
            + 2
            + 2
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._n_features,), dtype=np.float32
        )

        self._action_history = deque(maxlen=self.history_len)
        self._delta_history = deque(maxlen=self.history_len)
        self._initial_abs_obj = 1.0
        self._last_info = {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._action_history.clear()
        self._delta_history.clear()
        obj0 = float(np.asarray(obs["objective"]).item())
        self._initial_abs_obj = max(abs(obj0), 1e-6)
        self._last_info = {}
        return self.observation(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self._action_history.append(int(action))
        delta = float(np.asarray(obs["delta_objective"]).item())
        self._delta_history.append(delta / self._initial_abs_obj)
        self._last_info = info
        return self.observation(obs), reward, done, truncated, info

    def observation(self, obs):
        feats = np.zeros(self._n_features, dtype=np.float32)
        K = self.history_len
        S = self.num_strategies

        # Action history one-hot (oldest first, zero-padded)
        for t, a in enumerate(self._action_history):
            if 0 <= a < S:
                feats[t * S + a] = 1.0

        # Delta-obj history (oldest first, zero-padded)
        delta_offset = K * S
        for t, d in enumerate(self._delta_history):
            feats[delta_offset + t] = np.clip(d, -1e3, 1e3)

        base = delta_offset + K
        obj = float(np.asarray(obs["objective"]).item())
        feats[base + 0] = np.clip(obj / self._initial_abs_obj, -10.0, 10.0)
        feats[base + 1] = float(np.asarray(obs["nit_norm"]).item())
        deg = float(self._last_info.get("degenerate_streak", 0))
        feats[base + 2] = np.clip(deg / 200.0, 0.0, 5.0)

        rc = np.asarray(obs["reduced_costs"]).ravel()
        rc_stats_base = base + 3
        if rc.size > 0 and np.isfinite(rc).any():
            finite = rc[np.isfinite(rc)]
            neg = finite[finite < -1e-9]
            feats[rc_stats_base + 0] = neg.size / max(finite.size, 1)
            feats[rc_stats_base + 1] = np.clip(
                float(finite.min()) / self._initial_abs_obj, -10.0, 10.0
            )
            feats[rc_stats_base + 2] = np.clip(
                float(finite.mean()) / self._initial_abs_obj, -10.0, 10.0
            )
            feats[rc_stats_base + 3] = np.clip(
                float(finite.std()) / self._initial_abs_obj, 0.0, 10.0
            )

        # Tableau-based features (col norms + ratio test)
        tableau = np.asarray(obs["tableau"])
        if tableau.ndim == 2 and tableau.shape[0] > 1 and tableau.shape[1] > 1:
            T = tableau
            m_rows = T.shape[0] - 1
            tol = 1e-9
            neg_mask = rc < -tol
            col_feat_base = rc_stats_base + 4
            if neg_mask.any():
                cand = T[:m_rows, :-1][:, neg_mask]
                col_norms = np.linalg.norm(cand, axis=0)
                finite_norms = col_norms[np.isfinite(col_norms)]
                if finite_norms.size > 0:
                    mean_norm = float(finite_norms.mean())
                    std_norm = float(finite_norms.std())
                    feats[col_feat_base + 0] = np.clip(
                        mean_norm / np.sqrt(max(m_rows, 1)), 0.0, 10.0
                    )
                    feats[col_feat_base + 1] = np.clip(
                        std_norm / (mean_norm + 1e-9), 0.0, 10.0
                    )

                # Ratio test on the most-negative-RC column (what largest_coef would pick)
                best_j = int(np.argmin(rc))
                col = T[:m_rows, best_j]
                rhs = T[:m_rows, -1]
                pos = col > tol
                ratio_feat_base = col_feat_base + 2
                if pos.any():
                    ratios = np.where(pos, rhs / np.where(pos, col, 1.0), np.inf)
                    min_ratio = float(np.min(ratios))
                    if np.isfinite(min_ratio):
                        feats[ratio_feat_base + 0] = np.clip(
                            min_ratio / self._initial_abs_obj, -10.0, 10.0
                        )
                        tight = np.isfinite(ratios) & (
                            ratios <= min_ratio * (1.0 + 1e-5) + 1e-9
                        )
                        feats[ratio_feat_base + 1] = float(tight.sum()) / max(m_rows, 1)

        return feats


class EmptyObsWrapper(gym.ObservationWrapper):
    """Information-free observation — a single constant feature (always 0.0).

    Sanity-check baseline: the policy receives the SAME observation at every
    step of every episode, so it is forced to be state-independent. It can
    only learn one fixed action distribution (effectively "which single pivot
    rule is best on average"), never a per-state choice.

    If an agent trained with this wrapper matches a CompactObs agent, that is
    evidence the CompactObs agent wasn't actually using the tableau features —
    it had just collapsed to a fixed rule anyway.

    A constant length-1 vector is used rather than a genuinely zero-length
    (shape=(0,)) Box: information-wise they are identical (a constant input
    feeds only the first layer's bias), but the length-1 form avoids degenerate
    0-feature linear layers and SB3/Gym empty-array edge cases.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0, high=0.0, shape=(1,), dtype=np.float32
        )

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self.observation(None), info

    def step(self, action):
        _, reward, done, truncated, info = self.env.step(action)
        return self.observation(None), reward, done, truncated, info

    def observation(self, obs):
        return np.zeros(1, dtype=np.float32)

