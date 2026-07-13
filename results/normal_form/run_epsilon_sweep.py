"""Sweep TEST_EPSILON for a single model (dict_unweighted) on the
in-distribution test set.

TEST_EPSILON sets the perturbation magnitude of the in-distribution test
matrices (generate_test_matrices, mode="in_distribution"). The OOD set is
epsilon-independent (fresh uniform matrices), so we only re-run in-distribution.

The model was trained at EPSILON=0.001; this probes generalization to 10x and
100x larger perturbations than training.
"""
import os
import sys
import json
import contextlib

import numpy as np
from stable_baselines3 import PPO

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))  # results/normal_form -> repo root
sys.path.insert(0, REPO_ROOT)

import experiment
import config

# Use MATRIX per-pivot weights for weighted cost regardless of config's GAME_MODE.
experiment.STEP_PENALTY_WEIGHTS = config.STEP_PENALTY_WEIGHTS_MATRIX

MODEL_STEM = "ppo_simplex_random_20000000_matrix40x40_min-1_max1_epsilon0.001_dict_unweighted"
IS_COMPACT = False  # dict model
N_MATRICES = 200
SEED = 42
EPSILONS = [0.001, 0.01, 0.1]
MODELS_DIR = os.path.join(HERE, "models")
OUT_DIR = os.path.join(HERE, "evaluation", "epsilon_sweep")


def convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def run_in_distribution(n_matrices, seed, model):
    """In-distribution-only analogue of experiment.run_experiment."""
    np.random.seed(seed)
    strategies = list(experiment.PIVOT_MAP_TEST.values())
    all_methods = strategies + ["rl_agent"]

    matrices = experiment.generate_test_matrices(
        n_matrices, mode="in_distribution", base_matrix=None
    )
    rows = []
    for i, matrix_P in enumerate(matrices):
        tableau = experiment.prepare_tableau(matrix_P)
        if tableau is None:
            continue
        T, basis = tableau
        row = {"matrix_idx": i}
        for strategy in strategies:
            row[strategy] = experiment.run_fixed_strategy(T, basis, strategy)
        row["rl_agent"] = experiment.run_rl_agent(T, basis, model)
        rows.append(row)
    return {"in_distribution": rows}, all_methods


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    experiment.USE_COMPACT_OBS = IS_COMPACT

    model_path = os.path.join(MODELS_DIR, MODEL_STEM)
    model = PPO.load(model_path)

    for eps in EPSILONS:
        experiment.TEST_EPSILON = eps  # used by generate_test_matrices

        tag = f"eps{eps}"
        log_path = os.path.join(OUT_DIR, f"{MODEL_STEM}_{tag}.log")
        json_path = os.path.join(OUT_DIR, f"{MODEL_STEM}_{tag}.json")

        print(f"\n{'#' * 70}\n# TEST_EPSILON = {eps}  (training EPSILON = 0.001)\n{'#' * 70}",
              flush=True)

        with open(log_path, "w") as logf:
            with contextlib.redirect_stdout(logf):
                print(f"Model: {MODEL_STEM}")
                print(f"USE_COMPACT_OBS = {IS_COMPACT}")
                print(f"Training EPSILON = 0.001, TEST_EPSILON = {eps}")
                print(f"N_MATRICES = {N_MATRICES}, SEED = {SEED}")
                print("Mode: in_distribution only (OOD is epsilon-independent)")
                results, all_methods = run_in_distribution(N_MATRICES, SEED, model)
                experiment.analyze_results(results, all_methods)

        with open(json_path, "w") as f:
            json.dump(results, f, default=convert, indent=2)

        print(f"  -> {log_path}", flush=True)


if __name__ == "__main__":
    main()