"""Manual, interactive comparison harness — NOT an automated test suite.

Despite the ``test_*`` function names, this file contains no assertions and is
not run by pytest. It is a scratch script for eyeballing, on a single perturbed
matrix, the pivot count and recovered game value of each fixed heuristic
(``test_fixed_strategies``) versus a loaded PPO agent (``test_rl``). Run it
directly (``python compare_strategies.py``); nothing imports it. The formal,
reproducible evaluation lives in ``experiment.py`` and ``results/``.
"""

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from matrix import Matrix
from simplex_solver import change_to_zero_sum_phase2_only, _pivot_col_heuristics, _pivot_row, _apply_pivot
from envs import RandomMatrixEnv

# Import constants
from config import (
    M, N, MIN_VAL, MAX_VAL, EPSILON,
    PIVOT_MAP, PIVOT_MAP_TEST, NUM_PIVOT_STRATEGIES_TEST,
)
from base_matrix import BASE_MATRIX


class TestRandomMatrixEnv(RandomMatrixEnv):
    """Test environment that uses PIVOT_MAP_TEST instead of PIVOT_MAP"""
    def step(self, action):
        self._last_action = int(action)
        # Use PIVOT_MAP_TEST for testing all heuristics
        strategy = PIVOT_MAP_TEST[self._last_action]

        reward = -self._step_penalty
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

def run_fixed_strategy(matrix: Matrix, action: int):
    env = TestRandomMatrixEnv(matrix)
    _, _ = env.reset()
    done = False
    truncated = False
    while not done and not truncated:
        _, _, done, truncated, _ = env.step(action)
    method = PIVOT_MAP_TEST[action]
    
    # Extract strategies
    first_player_strategy = extract_optimal_strategy(env.T, env.basis, M)
    second_player_strategy = extract_second_player_strategy(env.T, env.basis, M, N)
    

    game_value = compute_game_value_from_strategies(matrix, first_player_strategy, second_player_strategy)
    
    print(f"[{method.title()} Pivot] Steps: {env.nit}, Game Value: {game_value:.6f}")

def test_fixed_strategies(matrix: Matrix):
    for action in range(NUM_PIVOT_STRATEGIES_TEST):  # Use constant from config
        run_fixed_strategy(matrix, action)

def extract_optimal_strategy(T, basis, m):

    num_constraints = T.shape[0] - 1 
    num_vars = T.shape[1] - 1  
    

    x = np.zeros(num_vars)
    

    for row in range(num_constraints):
        var = basis[row]  
        if 0 <= var < num_vars:  
            x[var] = T[row, -1]  
    strategy = x[:m]
    
    total = strategy.sum()
    if total > 1e-8: 
        strategy = strategy / total
    else:
        strategy = np.ones(m) / m
    
    return strategy

def extract_second_player_strategy(T, basis, m, n):

    num_constraints = T.shape[0] - 1 
    num_vars = T.shape[1] - 1 
    objective_row = T[-1, :-1] 
    dual_vars = objective_row[-n:]
    dual_vars = np.abs(dual_vars)
    total = dual_vars.sum()
    if total > 1e-8:
        second_player_strategy = dual_vars / total
    else:
        second_player_strategy = np.ones(n) / n
    
    return second_player_strategy

def compute_game_value_from_strategies(matrix: Matrix, first_player_strategy, second_player_strategy):
    """
    Compute the game value using the optimal strategies and the original payoff matrix.
    For zero-sum games: Game Value = x^T * P * y
    
    Args:
        matrix: The original payoff matrix
        first_player_strategy: Optimal strategy for first player (row player)
        second_player_strategy: Optimal strategy for second player (column player)
    
    Returns:
        game_value: The computed game value
    """
    # Convert strategies to numpy arrays if they aren't already
    x = np.array(first_player_strategy)
    y = np.array(second_player_strategy)
    
    # Get the payoff matrix
    P = matrix.base_P
    
    # Compute game value: x^T * P * y
    game_value = x.T @ P @ y
    
    return game_value


def test_rl(matrix: Matrix):
    print("\n--- PPO Policy Evaluation ---")
    # print("Matrix P:")
    # print(pd.DataFrame(matrix.base_P).to_string(index=False, header=False))
    # print(matrix.base_P.tolist())

    # One of the shipped final models (see results/normal_form/models/).
    model_path = ("results/normal_form/models/"
                  "ppo_simplex_random_20000000_matrix40x40_min-1_max1_epsilon0.001_dict_weighted.zip")

    env = RandomMatrixEnv(matrix)
    model = PPO.load(model_path)
    obs, _ = env.reset()
    done = False
    truncated = False
    i = 0
    while not done and not truncated:

        action, _ = model.predict(obs, deterministic=True)
        print(f"[RL] Action: {PIVOT_MAP[int(action)]}")
        obs, _, done, truncated, _ = env.step(action)

    # Extract strategies
    first_player_strategy = extract_optimal_strategy(env.T, env.basis, M)
    second_player_strategy = extract_second_player_strategy(env.T, env.basis, M, N)
    
    # Compute game value using the correct formula from game theory
    game_value = compute_game_value_from_strategies(matrix, first_player_strategy, second_player_strategy)
    
    print(f"[RL] Game Value: {game_value:.6f}")
    # print("[RL] First Player Strategy:", first_player_strategy)
    # print("[RL] Second Player Strategy:", second_player_strategy)
    print(f"[RL] Steps Taken: {env.nit}")


if __name__ == "__main__":
    print(M,N)
    matrix = Matrix(m=M, n=N, low=MIN_VAL, high=MAX_VAL, epsilon=EPSILON, base_P=BASE_MATRIX)
    # print("Base matrix:")
    print(pd.DataFrame(matrix.base_P).to_string(index=False, header=False))

    test_matrix = matrix.generate_perturbed_matrix()
    # print("\nTesting Matrix:")
    print(pd.DataFrame(test_matrix.base_P).to_string(index=False, header=False))


    test_fixed_strategies(test_matrix)
    test_rl(test_matrix)
