"""Gymnasium environments exposing simplex pivoting as an RL problem.

Each environment wraps a simplex tableau: the observation encodes tableau /
progress features, an action selects one of the pivot-column heuristics from
``config.PIVOT_MAP``, and ``step`` applies that pivot and returns a shaped
reward (per-step cost, improvement bonus, degeneracy/loop penalties, terminal
success bonus). ``SecondPhasePivotingEnv`` is the base phase-2 environment;
``RandomMatrixEnv`` / ``LeducEnv`` build phase-2 tableaus for the two game
families, and the ``*FullPivotEnv`` variants let the agent play both phases.

NOTE: observation layout and the action map are load-bearing — the trained
models encode these exact dimensions, so they must not change.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import (
    PIVOT_MAP, NUM_PIVOT_STRATEGIES, USE_TWO_PHASE,
    USE_WEIGHTED_STEP_PENALTY, STEP_PENALTY_WEIGHTS,
    SCALE_PENALTY_BY_SIZE, REFERENCE_TABLEAU_ROWS,
)
from simplex_solver import (
    change_to_zero_sum_phase2_only,
    change_to_zero_sum, phase1solver, first_to_second, phase1_via_highs,
    _pivot_col_heuristics, _pivot_row, _apply_pivot,
)
from _linprog_utils import _parse_linprog, _get_Abc, _LPProblem
from matrix import Matrix


class SecondPhasePivotingEnv(gym.Env):
    def remove_artificial(self):
        for pivrow in [row for row in range(self.basis.size)
                       if self.basis[row] > self.T.shape[1] - 2]:
            non_zero_row = [col for col in range(self.T.shape[1] - 1)
                            if abs(self.T[pivrow, col]) > self.tol]
            if len(non_zero_row) > 0:
                pivcol = non_zero_row[0]
                _apply_pivot(self.T, self.basis, pivrow, pivcol, self.tol)
                self.nit += 1

    def __init__(self, T, basis):
        # Core state
        self.basis = basis
        self.T = T
        self.tol = 1e-7
        self.m = self.T.shape[1] - 1
        self._obs_clip = 1e4
        self._nan_penalty = 100.0
        self.remove_artificial()

        # Limits & counters
        self.maxiter = 20_000
        self.nit = 0

        # Solution buffer
        if len(self.basis[:self.m]) == 0:
            self.solution = np.empty(self.T.shape[1] - 1, dtype=np.float64)
        else:
            self.solution = np.empty(max(self.T.shape[1] - 1, max(self.basis[:self.m]) + 1),
                                     dtype=np.float64)

        # Gym spaces
        self.action_space = spaces.Discrete(NUM_PIVOT_STRATEGIES)

        # === Loop & progress bookkeeping ===
        self._seen_bases = set()
        self._last_obj = float(self.T[-1, -1])
        self._degenerate_streak = 0

        # === Reward coefficients ===
        self._step_penalty = 1.0
        self._improve_coef = 0.0
        self._degenerate_penalty = 0.0
        self._loop_penalty = 0.0
        self._success_bonus = 50.0
        self._improve_tol = 1e-12

        # === Sizes and last action ===
        self._n_vars = self.T.shape[1] - 1
        self._m_rows = self.T.shape[0] - 1
        self._last_action = -1

        # === Dict observation space ===
        self.observation_space = spaces.Dict({
            "tableau": spaces.Box(low=-np.inf, high=np.inf, shape=self.T.shape, dtype=np.float64),
            "basis_onehot": spaces.Box(low=0.0, high=1.0, shape=(self._n_vars,), dtype=np.float32),
            "reduced_costs": spaces.Box(low=-np.inf, high=np.inf, shape=(self._n_vars,), dtype=np.float64),
            "objective": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
            "delta_objective": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
            "nit_norm": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

        self.complete = False

    def _zero_obs(self):
        return {
            "tableau": np.zeros(self.T.shape, dtype=np.float64),
            "basis_onehot": np.zeros(self._n_vars, dtype=np.float32),
            "reduced_costs": np.zeros(self._n_vars, dtype=np.float64),
            "objective": np.zeros(1, dtype=np.float64),
            "delta_objective": np.zeros(1, dtype=np.float64),
            "nit_norm": np.zeros(1, dtype=np.float32),
        }

    def _basis_key(self):
        return tuple(int(i) for i in self.basis)

    def _basis_onehot(self):
        vec = np.zeros(self._n_vars, dtype=np.float32)
        for col in self.basis:
            c = int(col)
            if 0 <= c < self._n_vars:
                vec[c] = 1.0
        return vec

    def _last_action_onehot(self):
        v = np.zeros(int(NUM_PIVOT_STRATEGIES), dtype=np.float32)
        if 0 <= self._last_action < NUM_PIVOT_STRATEGIES:
            v[self._last_action] = 1.0
        return v

    def _nit_norm(self):
        return np.array([min(1.0, float(self.nit) / float(self.maxiter))], dtype=np.float32)

    def _get_obs(self):
        tableau = self.T.copy()
        rc = tableau[-1, :-1].copy()
        obj = float(tableau[-1, -1])
        delta = float(self._last_obj - obj)

        # Replace NaN/Inf, then clip to a bounded range so the NN doesn't blow up
        c = self._obs_clip
        np.nan_to_num(tableau, copy=False, nan=0.0, posinf=c, neginf=-c)
        np.nan_to_num(rc, copy=False, nan=0.0, posinf=c, neginf=-c)
        np.clip(tableau, -c, c, out=tableau)
        np.clip(rc, -c, c, out=rc)
        if not np.isfinite(obj):
            obj = 0.0
        if not np.isfinite(delta):
            delta = 0.0
        obj = float(np.clip(obj, -c, c))
        delta = float(np.clip(delta, -c, c))

        obs = {
            "tableau": tableau,
            "basis_onehot": self._basis_onehot(),
            "reduced_costs": rc,
            "objective": np.array([obj], dtype=np.float64),
            "delta_objective": np.array([delta], dtype=np.float64),
            "nit_norm": self._nit_norm(),
            # "degenerate_streak": np.array([float(self._degenerate_streak)], dtype=np.float32),
            # "loop_flag": np.array([1.0 if self._basis_key() in self._seen_bases else 0.0], dtype=np.float32),
            # "last_action_onehot": self._last_action_onehot(),
            # "shift_K": np.array([float(getattr(self, "K", 0.0) or 0.0)], dtype=np.float64),
        }
        return obs

    def reset(self, seed=None, **kwargs):
        self.nit = 0
        self._seen_bases.clear()
        self._last_obj = float(self.T[-1, -1])
        self._degenerate_streak = 0
        self._last_action = -1
        self._seen_bases.add(self._basis_key())
        return self._get_obs(), {}

    def _step_cost(self, strategy):
        w = STEP_PENALTY_WEIGHTS.get(strategy, 1.0) if USE_WEIGHTED_STEP_PENALTY else 1.0
        if SCALE_PENALTY_BY_SIZE:
            w *= self.T.shape[0] / REFERENCE_TABLEAU_ROWS
        return self._step_penalty * w

    def step(self, action):
        self._last_action = int(action)
        strategy = PIVOT_MAP[self._last_action]

        reward = -self._step_cost(strategy)
        done = False
        truncated = False

        pivcol_found, pivcol = _pivot_col_heuristics(self.T, strategy=strategy, tol=self.tol)
        if not pivcol_found:
            reward += self._success_bonus
            done = True
            info = {
                "status": "optimal",
                "nit": self.nit,
                "objective": float(self.T[-1, -1]),
                "degenerate_streak": self._degenerate_streak,
            }
            return self._get_obs(), reward, done, truncated, info

        use_bland = (strategy == 'blands_rule')
        pivrow_found, pivrow = _pivot_row(self.T, self.basis, pivcol, phase=2, tol=self.tol, bland=use_bland)
        if not pivrow_found:
            done = True
            info = {
                "status": "no_pivot_row",
                "nit": self.nit,
                "objective": float(self.T[-1, -1]),
                "degenerate_streak": self._degenerate_streak,
            }
            return self._get_obs(), reward, done, truncated, info

        old_obj = float(self.T[-1, -1])
        _apply_pivot(self.T, self.basis, pivrow, pivcol, tol=self.tol)
        self.nit += 1
        new_obj = float(self.T[-1, -1])

        # Terminate early if tableau became numerically corrupt: discard matrix,
        # assign a heavy negative reward, and return a zero observation so the
        # policy/value net never sees the corrupted tableau.
        if not np.isfinite(new_obj) or np.any(~np.isfinite(self.T)):
            done = True
            reward -= self._nan_penalty
            info = {
                "status": "numerical_error",
                "nit": self.nit,
                "objective": 0.0,
                "degenerate_streak": self._degenerate_streak,
            }
            return self._zero_obs(), reward, done, truncated, info

        delta = old_obj - new_obj
        if delta > self._improve_tol:
            reward += self._improve_coef * delta
            self._degenerate_streak = 0
        else:
            reward -= self._degenerate_penalty
            self._degenerate_streak += 1

        key = self._basis_key()
        if key in self._seen_bases:
            print("LOOP DETECTED")
            reward -= self._loop_penalty
            truncated = True
        else:
            self._seen_bases.add(key)

        if self.nit >= self.maxiter:
            done = True

        self._last_obj = new_obj

        info = {
            "status": "running",
            "nit": self.nit,
            "objective": new_obj,
            "delta_objective": delta,
            "degenerate": (delta <= self._improve_tol),
            "degenerate_streak": self._degenerate_streak,
            "pivcol": int(pivcol),
            "pivrow": int(pivrow),
            "strategy": strategy,
            "loop_detected": truncated,
        }
        return self._get_obs(), reward, done, truncated, info

    def render(self, mode='human'):
        print("Current Tableau:")
        print(self.T)
        print("Current Basis:", self.basis)
        print("Iterations:", self.nit)

    def close(self):
        pass


class FirstPhasePivotingEnv(gym.Env):

    def __init__(self, T, basis, nit0 = 0):
        self.basis = basis
        self.T = T
        self.tol = 1e-9
        self.maxiter = 5000
        self.nit0 = nit0
        self.nit = 0
        self.action_space = spaces.Discrete(NUM_PIVOT_STRATEGIES)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=self.T.shape, dtype=np.float64
        )
        self.complete = False

    def _get_obs(self):
        return self.T.copy()

    def reset(self, seed=None, **kwargs):
        self.nit = 0
        self.status = None
        return self._get_obs(), {}

    def step(self, action):
        strategy = PIVOT_MAP[int(action)]
        reward = -1
        done = False

        pivcol_found, pivcol = _pivot_col_heuristics(self.T, strategy=strategy, tol=self.tol)
        if not pivcol_found:
            done = True
            return self._get_obs(), reward, done, False, {}

        pivrow_found, pivrow = _pivot_row(self.T, self.basis, pivcol, phase=1, tol=self.tol)
        if not pivrow_found:
            done = True
            return self._get_obs(), reward, done, False, {}

        _apply_pivot(self.T, self.basis, pivrow, pivcol, tol=self.tol)
        self.nit += 1

        if self.nit >= self.maxiter:
            done = True

        return self._get_obs(), reward, done, False, {}

    def render(self, mode='human'):
        print("Current Tableau:")
        print(self.T)
        print("Current Basis:", self.basis)
        print("Iteration:", self.nit)

    def close(self):
        pass


class RandomMatrixEnv(SecondPhasePivotingEnv):
    def __init__(self, matrix: Matrix):
        self.matrix = matrix
        self.epsilon = matrix.epsilon
        self.K = None
        self.nit = 0
        self._phase1_nit = 0

        self._init_env()

        super().__init__(self.T, self.basis)

    def _init_env(self, seed=None):
        self.nit = 0
        self._phase1_nit = 0
        max_attempts = 20
        for attempt in range(max_attempts):
            try:
                perturbed_P = self.matrix.generate_perturbed_matrix()
                npMatrix = perturbed_P.base_P

                if USE_TWO_PHASE:
                    T, basis, av = change_to_zero_sum(npMatrix)
                    nit, status = phase1solver(T, basis)
                    if status != 0:
                        raise RuntimeError(f"Phase 1 failed with status {status}")
                    self._phase1_nit = nit
                    res = first_to_second(T, basis, av)
                    if res is None:
                        raise RuntimeError("Phase 1→2 transition failed")
                    self.T, self.basis = res
                    self.K = None
                else:
                    res = change_to_zero_sum_phase2_only(npMatrix)
                    if res is not None:
                        self.T, self.basis, self.K = res
                    else:
                        raise RuntimeError("Direct Phase 2 construction failed")
                return
            except Exception as e:
                print(f"[RandomMatrixEnv] Attempt {attempt + 1} failed: {e}")
                continue

        print(f"Too many unstable matrices for size {self.matrix.m}x{self.matrix.n}")
        raise RuntimeError("Failed to initialize a stable Phase 2 tableau.")

    def reset(self, seed=None, **kwargs):
        self._init_env(seed)
        self.nit = 0
        return super().reset(seed=seed)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        info["phase1_nit"] = self._phase1_nit
        info["total_nit"] = self._phase1_nit + self.nit
        return obs, reward, done, truncated, info


class LeducEnv(SecondPhasePivotingEnv):
    """
    Gym environment that generates sequence-form LPs from Leduc poker
    with non-uniform deck weights, then lets the RL agent pick pivot
    strategies in Phase 2 of the simplex method.
    """

    def __init__(self, game_name, alpha=2.0, num_ranks=3, seed=None):
        import pyspiel
        from leduc_experiment import build_sequence_form_matrices, sample_rank_weights

        self._game = pyspiel.load_game(game_name)
        self._alpha = alpha
        self._num_ranks = num_ranks
        self._build_matrices = build_sequence_form_matrices
        self._sample_weights = sample_rank_weights
        self._rng = np.random.default_rng(seed)

        self.K = None
        self.nit = 0
        self._phase1_nit = 0

        self._init_env()
        super().__init__(self.T, self.basis)

    def _init_env(self, seed=None):
        self.nit = 0
        self._phase1_nit = 0
        max_attempts = 20

        for attempt in range(max_attempts):
            try:
                weights = self._sample_weights(
                    alpha=self._alpha, num_ranks=self._num_ranks, rng=self._rng
                )
                A, E, e, F, f, *_ = self._build_matrices(
                    self._game, rank_weights=weights
                )

                # Build the sequence-form LP in standard form
                n_x, n_y, n_p = A.shape[0], A.shape[1], F.shape[0]
                c = np.concatenate([np.zeros(n_x), -f])
                A_eq = np.hstack([E, np.zeros((E.shape[0], n_p))])
                b_eq = e
                A_ub = np.hstack([-A.T, F.T])
                b_ub = np.zeros(n_y)
                bounds = [(0, None)] * n_x + [(None, None)] * n_p

                lp = _LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0=None, integrality=None)
                lp, solver_options = _parse_linprog(lp, None, meth='simplex')
                tol = solver_options.get('tol', 1e-9)
                A_std, b_std, c_std, c0, x0 = _get_Abc(lp, 0)

                n_rows, n_cols = A_std.shape
                neg = b_std < 0
                A_std[neg] *= -1
                b_std[neg] *= -1

                # Phase 1 tableau
                av = np.arange(n_rows) + n_cols
                basis = av.copy()
                row_constraints = np.hstack((A_std, np.eye(n_rows), b_std[:, np.newaxis]))
                row_objective = np.hstack((c_std, np.zeros(n_rows), c0))
                row_pseudo_objective = -row_constraints.sum(axis=0)
                row_pseudo_objective[av] = 0
                T = np.vstack((row_constraints, row_objective, row_pseudo_objective))

                # Solve Phase 1 with fixed strategy
                nit, status = phase1solver(T, basis, maxiter=50000)
                if status != 0:
                    raise RuntimeError(f"Phase 1 failed with status {status}")
                self._phase1_nit = nit

                res = first_to_second(T, basis, av)
                if res is None:
                    raise RuntimeError("Phase 1->2 transition failed")
                self.T, self.basis = res
                self.K = None
                return

            except Exception as exc:
                print(f"[LeducEnv] Attempt {attempt + 1} failed: {exc}")
                continue

        raise RuntimeError("LeducEnv: failed to build a stable Phase 2 tableau")

    def reset(self, seed=None, **kwargs):
        self._init_env(seed)
        self.nit = 0
        return super().reset(seed=seed)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        info["phase1_nit"] = self._phase1_nit
        info["total_nit"] = self._phase1_nit + self.nit
        return obs, reward, done, truncated, info


class FullPivotEnv(gym.Env):
    """Two-phase simplex env: agent picks pivot strategies for phase 1 AND phase 2.

    Starts at the un-pivoted phase-1 tableau (with pseudo-objective row and
    artificial variables). Each step applies the agent's chosen strategy. When
    the pseudo-objective reaches zero (feasibility), the env drops the pseudo
    row and artificial cols and continues in phase 2. Episode ends at phase-2
    optimum. Emits 31 compact features directly (no wrapper needed).
    """

    def __init__(self, matrix: Matrix, history_len: int = 5,
                 baseline_strategy: str = 'steepest_edge',
                 use_baseline: bool = True, baseline_coef: float = 5.0,
                 maxiter: int = 20_000, tol: float = 1e-7):
        from collections import deque
        self.matrix = matrix
        self.history_len = int(history_len)
        self.baseline_strategy = baseline_strategy
        self.use_baseline = use_baseline
        self.baseline_coef = float(baseline_coef)
        self.maxiter = maxiter
        self.tol = tol

        S = int(NUM_PIVOT_STRATEGIES)
        # +1 extra: phase indicator (1.0 = phase 1, 0.0 = phase 2)
        self._n_features = history_len * S + history_len + 3 + 4 + 2 + 2 + 1
        self.action_space = spaces.Discrete(S)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._n_features,), dtype=np.float32
        )

        self._action_hist = deque(maxlen=history_len)
        self._delta_hist = deque(maxlen=history_len)

        self._step_penalty = 1.0
        self._improve_coef = 10.0
        self._degenerate_penalty = 0.2
        self._success_bonus = 50.0
        self._improve_tol = 1e-12

        self._init_env()

    def _build_phase1(self):
        """Subclass hook. Returns (T, basis, av, rebuild_fn).

        `rebuild_fn` is a zero-arg callable that returns a fresh (T, basis, av)
        for the SAME underlying LP — used by `_compute_baseline` to replay the
        fixed strategy end-to-end without mutating the env's tableau.
        """
        perturbed = self.matrix.generate_perturbed_matrix()
        P = perturbed.base_P
        rebuild = lambda: change_to_zero_sum(P)
        T, basis, av = rebuild()
        return T, basis, av, rebuild

    def _init_env(self):
        for attempt in range(20):
            try:
                T, basis, av, payload = self._build_phase1()
                self.T, self.basis, self.av = T, basis, av
                self.phase = 1
                self.nit = 0
                self._phase1_nit = 0
                self._phase2_nit = 0
                self._last_obj = float(T[-1, -1])
                self._initial_abs_obj = max(abs(self._last_obj), 1e-6)
                self._degenerate_streak = 0
                self._action_hist.clear()
                self._delta_hist.clear()
                self._baseline_nit = self._compute_baseline(payload) if self.use_baseline else None
                return
            except Exception as e:
                print(f"[FullPivotEnv] Attempt {attempt + 1} failed: {e}")
                continue
        raise RuntimeError("FullPivotEnv: failed to initialize")

    def _compute_baseline(self, rebuild_fn):
        T, basis, av = rebuild_fn()
        strat = self.baseline_strategy
        n1, status = phase1solver(T, basis, maxiter=self.maxiter, tol=self.tol)
        if status != 0:
            return self.maxiter
        res = first_to_second(T, basis, av)
        if res is None:
            return self.maxiter
        T, basis = res
        n2 = 0
        while n2 < self.maxiter:
            ok, col = _pivot_col_heuristics(T, strategy=strat, tol=self.tol)
            if not ok:
                break
            ok, row = _pivot_row(T, basis, col, phase=2, tol=self.tol)
            if not ok:
                break
            _apply_pivot(T, basis, row, col, tol=self.tol)
            if not np.isfinite(T[-1, -1]) or np.any(~np.isfinite(T)):
                break
            n2 += 1
        return n1 + n2

    def _transition_to_phase2(self):
        self.T = np.delete(self.T[:-1, :], self.av, 1)
        self.phase = 2
        self._last_obj = float(self.T[-1, -1])
        # Rescale normalization to the phase-2 objective magnitude so phase-2
        # deltas give comparable reward signal.
        self._initial_abs_obj = max(abs(self._last_obj), 1e-6)

    def reset(self, seed=None, **kwargs):
        self._init_env()
        return self._compute_obs(), {}

    def _step_cost(self, strategy):
        w = STEP_PENALTY_WEIGHTS.get(strategy, 1.0) if USE_WEIGHTED_STEP_PENALTY else 1.0
        if SCALE_PENALTY_BY_SIZE:
            w *= self.T.shape[0] / REFERENCE_TABLEAU_ROWS
        return self._step_penalty * w

    def step(self, action):
        a = int(action)
        strategy = PIVOT_MAP[a]
        tol = self.tol
        reward = -self._step_cost(strategy)

        ok, col = _pivot_col_heuristics(self.T, strategy=strategy, tol=tol)
        if not ok:
            if self.phase == 1:
                if abs(self.T[-1, -1]) < 1e-7:
                    self._transition_to_phase2()
                    self._record_action(a, 0.0)
                    return self._compute_obs(), reward, False, False, {
                        "status": "phase_transition", "phase": 2,
                        "nit": self.nit, "phase1_nit": self._phase1_nit,
                    }
                return self._compute_obs(), reward - 100.0, True, False, {
                    "status": "infeasible", "nit": self.nit,
                }
            reward += self._success_bonus
            if self.use_baseline and self._baseline_nit is not None:
                reward += self.baseline_coef * (self._baseline_nit - self.nit)
            return self._compute_obs(), reward, True, False, {
                "status": "optimal", "nit": self.nit,
                "phase1_nit": self._phase1_nit, "phase2_nit": self._phase2_nit,
                "baseline_nit": self._baseline_nit,
            }

        ok, row = _pivot_row(self.T, self.basis, col, phase=self.phase, tol=tol)
        if not ok:
            return self._compute_obs(), reward - 50.0, True, False, {
                "status": "no_row", "nit": self.nit,
            }

        old_obj = float(self.T[-1, -1])
        _apply_pivot(self.T, self.basis, row, col, tol=tol)
        self.nit += 1
        if self.phase == 1:
            self._phase1_nit += 1
        else:
            self._phase2_nit += 1

        # Guard against numerical blow-up: NaN/Inf OR huge magnitudes
        _max_abs = 1e12
        if (not np.isfinite(self.T[-1, -1])
                or np.any(~np.isfinite(self.T))
                or np.max(np.abs(self.T)) > _max_abs):
            return self._compute_obs(), reward - 100.0, True, False, {
                "status": "numerical", "nit": self.nit,
            }

        new_obj = float(self.T[-1, -1])
        delta = old_obj - new_obj
        if delta > self._improve_tol:
            raw = self._improve_coef * delta / self._initial_abs_obj
            reward += float(np.clip(raw, -50.0, 50.0))
            self._degenerate_streak = 0
        else:
            reward -= self._degenerate_penalty
            self._degenerate_streak += 1

        if self.phase == 1 and abs(self.T[-1, -1]) < 1e-7:
            self._transition_to_phase2()

        self._last_obj = float(self.T[-1, -1])
        self._record_action(a, delta)

        if self.nit >= self.maxiter:
            return self._compute_obs(), reward, True, False, {
                "status": "maxiter", "nit": self.nit,
            }
        return self._compute_obs(), reward, False, False, {
            "status": "running", "phase": self.phase, "nit": self.nit,
        }

    def _record_action(self, action, delta):
        self._action_hist.append(int(action))
        self._delta_hist.append(float(delta) / self._initial_abs_obj)

    def _compute_obs(self):
        feats = np.zeros(self._n_features, dtype=np.float32)
        K = self.history_len
        S = int(NUM_PIVOT_STRATEGIES)

        for t, a in enumerate(self._action_hist):
            if 0 <= a < S:
                feats[t * S + a] = 1.0
        delta_offset = K * S
        for t, d in enumerate(self._delta_hist):
            feats[delta_offset + t] = np.clip(d, -1e3, 1e3)

        base = delta_offset + K
        obj = float(self.T[-1, -1])
        feats[base + 0] = np.clip(obj / self._initial_abs_obj, -10.0, 10.0)
        feats[base + 1] = min(1.0, self.nit / max(self.maxiter, 1))
        feats[base + 2] = np.clip(self._degenerate_streak / 200.0, 0.0, 5.0)

        rc = self.T[-1, :-1]
        rc_base = base + 3
        if rc.size > 0 and np.isfinite(rc).any():
            finite = rc[np.isfinite(rc)]
            neg = finite[finite < -1e-9]
            feats[rc_base + 0] = neg.size / max(finite.size, 1)
            feats[rc_base + 1] = np.clip(float(finite.min()) / self._initial_abs_obj, -10.0, 10.0)
            feats[rc_base + 2] = np.clip(float(finite.mean()) / self._initial_abs_obj, -10.0, 10.0)
            feats[rc_base + 3] = np.clip(float(finite.std()) / self._initial_abs_obj, 0.0, 10.0)

        k = 2 if self.phase == 1 else 1
        T_body = self.T[:-k, :-1] if self.T.shape[0] > k else self.T[:0, :]
        tol = 1e-9
        neg_mask = rc < -tol
        col_base = rc_base + 4
        if neg_mask.any() and T_body.shape[0] > 0:
            cand = T_body[:, neg_mask]
            col_norms = np.linalg.norm(cand, axis=0)
            finite_norms = col_norms[np.isfinite(col_norms)]
            if finite_norms.size > 0:
                mean_norm = float(finite_norms.mean())
                feats[col_base + 0] = np.clip(mean_norm / np.sqrt(max(T_body.shape[0], 1)), 0.0, 10.0)
                feats[col_base + 1] = np.clip(float(finite_norms.std()) / (mean_norm + 1e-9), 0.0, 10.0)

            best_j = int(np.argmin(rc))
            col = T_body[:, best_j]
            rhs = self.T[:-k, -1]
            pos = col > tol
            ratio_base = col_base + 2
            if pos.any():
                ratios = np.where(pos, rhs / np.where(pos, col, 1.0), np.inf)
                min_ratio = float(np.min(ratios))
                if np.isfinite(min_ratio):
                    feats[ratio_base + 0] = np.clip(min_ratio / self._initial_abs_obj, -10.0, 10.0)
                    tight = np.isfinite(ratios) & (ratios <= min_ratio * (1.0 + 1e-5) + 1e-9)
                    feats[ratio_base + 1] = float(tight.sum()) / max(T_body.shape[0], 1)

        # Phase indicator (last feature): 1.0 during phase 1, 0.0 during phase 2
        feats[-1] = 1.0 if self.phase == 1 else 0.0
        return feats


class LeducFullPivotEnv(FullPivotEnv):
    """Two-phase simplex env for Leduc poker sequence-form LPs.

    Same step / observation / reward machinery as `FullPivotEnv`, but the
    phase-1 tableau is constructed from a sampled Leduc LP (non-uniform deck
    via Dirichlet rank weights) rather than from a random payoff matrix.
    """

    def __init__(self, game_name: str, alpha: float = 2.0, num_ranks: int = 3,
                 history_len: int = 5, baseline_strategy: str = 'steepest_edge',
                 use_baseline: bool = True, baseline_coef: float = 5.0,
                 maxiter: int = 50_000, tol: float = 1e-7, seed=None):
        import pyspiel
        from leduc_experiment import build_sequence_form_matrices, sample_rank_weights
        self._game = pyspiel.load_game(game_name)
        self._alpha = float(alpha)
        self._num_ranks = int(num_ranks)
        self._build_seq_matrices = build_sequence_form_matrices
        self._sample_weights = sample_rank_weights
        self._rng = np.random.default_rng(seed)
        # `matrix` is unused in Leduc path — pass None
        super().__init__(
            matrix=None, history_len=history_len,
            baseline_strategy=baseline_strategy, use_baseline=use_baseline,
            baseline_coef=baseline_coef, maxiter=maxiter, tol=tol,
        )

    def _build_phase1(self):
        weights = self._sample_weights(
            alpha=self._alpha, num_ranks=self._num_ranks, rng=self._rng
        )
        A, E, e, F, f, *_ = self._build_seq_matrices(self._game, rank_weights=weights)
        n_x, n_y, n_p = A.shape[0], A.shape[1], F.shape[0]
        c = np.concatenate([np.zeros(n_x), -f])
        A_eq = np.hstack([E, np.zeros((E.shape[0], n_p))])
        A_ub = np.hstack([-A.T, F.T])
        bounds = [(0, None)] * n_x + [(None, None)] * n_p

        def rebuild():
            lp = _LPProblem(c, A_ub, np.zeros(n_y), A_eq, e, bounds, x0=None, integrality=None)
            lp2, _ = _parse_linprog(lp, None, meth='simplex')
            A_std, b_std, c_std, c0, _ = _get_Abc(lp2, 0)
            neg = b_std < 0
            A_std[neg] *= -1
            b_std[neg] *= -1
            n_rows, n_cols = A_std.shape
            av = np.arange(n_rows) + n_cols
            basis = av.copy()
            rc = np.hstack((A_std, np.eye(n_rows), b_std[:, np.newaxis]))
            ro = np.hstack((c_std, np.zeros(n_rows), c0))
            rp = -rc.sum(axis=0)
            rp[av] = 0
            T = np.vstack((rc, ro, rp))
            return T, basis, av

        T, basis, av = rebuild()
        return T, basis, av, rebuild


