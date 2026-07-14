"""Gymnasium environments exposing simplex pivoting as an RL problem.

Each environment wraps a simplex tableau: the observation encodes tableau /
progress features, an action selects one of the pivot-column heuristics from
``config.PIVOT_MAP``, and ``step`` applies that pivot and returns a shaped
reward (per-step cost, improvement bonus, degeneracy/loop penalties, terminal
success bonus). ``SecondPhasePivotingEnv`` is the base phase-2 environment;
``RandomMatrixEnv`` / ``LeducEnv`` build phase-2 tableaus for the two game
families.

NOTE: observation layout and the action map are load-bearing — the trained
models encode these exact dimensions, so they must not change.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import (
    PIVOT_MAP, NUM_PIVOT_STRATEGIES, USE_TWO_PHASE,
    USE_WEIGHTED_STEP_PENALTY, STEP_PENALTY_WEIGHTS,
)
from simplex_solver import (
    change_to_zero_sum_phase2_only,
    change_to_zero_sum, phase1solver, first_to_second,
    _pivot_col_heuristics, _pivot_row, _apply_pivot,
)
from _linprog_utils import _parse_linprog, _get_Abc, _LPProblem
from matrix import Matrix


class SecondPhasePivotingEnv(gym.Env):
    """Base phase-2 pivoting environment over a fixed simplex tableau.

    Actions pick a pivot rule from ``config.PIVOT_MAP``; ``step`` applies one
    pivot and rewards −step_cost per pivot plus a terminal success bonus.
    Subclasses (``RandomMatrixEnv``, ``LeducEnv``) resample a fresh LP on reset.
    """

    def remove_artificial(self):
        """Pivot any artificial variables remaining in the basis out of it."""
        for pivrow in [row for row in range(self.basis.size)
                       if self.basis[row] > self.T.shape[1] - 2]:
            non_zero_row = [col for col in range(self.T.shape[1] - 1)
                            if abs(self.T[pivrow, col]) > self.tol]
            if len(non_zero_row) > 0:
                pivcol = non_zero_row[0]
                _apply_pivot(self.T, self.basis, pivrow, pivcol, self.tol)
                self.nit += 1

    def __init__(self, T, basis):
        """Wrap an existing phase-2 tableau + basis and define the gym spaces."""
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
        """All-zero observation, returned after a numerical failure."""
        return {
            "tableau": np.zeros(self.T.shape, dtype=np.float64),
            "basis_onehot": np.zeros(self._n_vars, dtype=np.float32),
            "reduced_costs": np.zeros(self._n_vars, dtype=np.float64),
            "objective": np.zeros(1, dtype=np.float64),
            "delta_objective": np.zeros(1, dtype=np.float64),
            "nit_norm": np.zeros(1, dtype=np.float32),
        }

    def _basis_key(self):
        """Hashable basis fingerprint used for cycle detection."""
        return tuple(int(i) for i in self.basis)

    def _basis_onehot(self):
        """One-hot vector marking which variables are currently basic."""
        vec = np.zeros(self._n_vars, dtype=np.float32)
        for col in self.basis:
            c = int(col)
            if 0 <= c < self._n_vars:
                vec[c] = 1.0
        return vec

    def _nit_norm(self):
        """Iteration count normalized to [0, 1] by maxiter."""
        return np.array([min(1.0, float(self.nit) / float(self.maxiter))], dtype=np.float32)

    def _get_obs(self):
        """Build the Dict observation (tableau, reduced costs, progress scalars), NaN-safe and clipped."""
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
        }
        return obs

    def reset(self, seed=None, **kwargs):
        """Reset counters and cycle tracking on the CURRENT tableau (no resampling)."""
        self.nit = 0
        self._seen_bases.clear()
        self._last_obj = float(self.T[-1, -1])
        self._degenerate_streak = 0
        self._last_action = -1
        self._seen_bases.add(self._basis_key())
        return self._get_obs(), {}

    def _step_cost(self, strategy):
        """Per-pivot cost: 1.0, or the rule's calibrated weight when weighted penalties are on."""
        w = STEP_PENALTY_WEIGHTS.get(strategy, 1.0) if USE_WEIGHTED_STEP_PENALTY else 1.0
        return self._step_penalty * w

    def step(self, action):
        """Apply one pivot with the chosen rule; terminate on optimality, failure, or maxiter."""
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
        """Print the tableau, basis and iteration count."""
        print("Current Tableau:")
        print(self.T)
        print("Current Basis:", self.basis)
        print("Iterations:", self.nit)

    def close(self):
        """No resources to release."""
        pass


class RandomMatrixEnv(SecondPhasePivotingEnv):
    """Phase-2 env over perturbed zero-sum payoff matrices (matrix game mode).

    Each reset perturbs the base matrix by epsilon and rebuilds the phase-2
    tableau (directly, or via phase 1 when ``USE_TWO_PHASE``).
    """

    def __init__(self, matrix: Matrix):
        """Build the first tableau from ``matrix`` and initialize the base env."""
        self.matrix = matrix
        self.epsilon = matrix.epsilon
        self.K = None
        self.nit = 0
        self._phase1_nit = 0

        self._init_env()

        super().__init__(self.T, self.basis)

    def _init_env(self, seed=None):
        """Sample a perturbed matrix and build its phase-2 tableau (retrying unstable draws)."""
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
        """Resample a fresh LP instance and reset the base env on its tableau."""
        self._init_env(seed)
        self.nit = 0
        return super().reset(seed=seed)

    def step(self, action):
        """Step the base env, adding phase-1 / total pivot counts to the info dict."""
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
        """Load the OpenSpiel game and build the first sequence-form tableau."""
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
        """Sample Dirichlet deck weights, build the sequence-form LP, and solve phase 1."""
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
        """Resample a fresh LP instance and reset the base env on its tableau."""
        self._init_env(seed)
        self.nit = 0
        return super().reset(seed=seed)

    def step(self, action):
        """Step the base env, adding phase-1 / total pivot counts to the info dict."""
        obs, reward, done, truncated, info = super().step(action)
        info["phase1_nit"] = self._phase1_nit
        info["total_nit"] = self._phase1_nit + self.nit
        return obs, reward, done, truncated, info
