"""Run experiment.py's evaluation pipeline on each model in final_models/.

The four models vary along two axes encoded in their filename:
  - obs:     "compact" (CompactObsWrapper, MlpPolicy) vs "dict" (Dict obs, MultiInputPolicy)
  - penalty: "weighted" vs "unweighted" step penalty (a training-time reward
             difference; evaluation reports BOTH nit and weighted_cost regardless)

The obs axis is the one that matters for *loading/running*: a compact model
expects the 31-feature flat observation, a dict model expects the Dict obs.
So we flip experiment.USE_COMPACT_OBS per model before running.
"""
import os
import sys
import json
import contextlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))  # results/normal_form -> repo root
sys.path.insert(0, REPO_ROOT)

import experiment
import config

# Report weighted cost with the MATRIX per-pivot weights regardless of the
# current GAME_MODE in config.py (experiment.py imports the auto-picked dict at
# module load). This lets matrix re-evaluation run correctly even while config
# is set to leduc mode.
experiment.STEP_PENALTY_WEIGHTS = config.STEP_PENALTY_WEIGHTS_MATRIX

N_MATRICES = 200
SEED = 42
MODELS_DIR = os.path.join(HERE, "models")
OUT_DIR = os.path.join(HERE, "evaluation")

MODELS = [
    "ppo_simplex_random_20000000_matrix40x40_min-1_max1_epsilon0.001_compact_unweighted",
    "ppo_simplex_random_20000000_matrix40x40_min-1_max1_epsilon0.001_compact_weighted",
    "ppo_simplex_random_20000000_matrix40x40_min-1_max1_epsilon0.001_dict_unweighted",
    "ppo_simplex_random_20000000_matrix40x40_min-1_max1_epsilon0.001_dict_weighted",
]


def convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for stem in MODELS:
        is_compact = "compact" in stem
        experiment.USE_COMPACT_OBS = is_compact  # flip before run_rl_agent uses it

        model_path = os.path.join(MODELS_DIR, stem)  # SB3 adds .zip
        log_path = os.path.join(OUT_DIR, stem + ".log")
        json_path = os.path.join(OUT_DIR, stem + ".json")

        print(f"\n{'#' * 70}")
        print(f"# {stem}")
        print(f"#   compact_obs={is_compact}")
        print(f"{'#' * 70}", flush=True)

        with open(log_path, "w") as logf:
            with contextlib.redirect_stdout(logf):
                print(f"Model: {stem}")
                print(f"USE_COMPACT_OBS = {is_compact}")
                print(f"N_MATRICES = {N_MATRICES}, SEED = {SEED}")
                print(f"Tested strategies: {list(experiment.PIVOT_MAP_TEST.values())}")
                results, all_methods = experiment.run_experiment(
                    N_MATRICES, SEED, model_path=model_path, base_matrix=None
                )
                experiment.analyze_results(results, all_methods)

        with open(json_path, "w") as f:
            json.dump(results, f, default=convert, indent=2)

        print(f"  -> log:  {log_path}")
        print(f"  -> json: {json_path}", flush=True)


if __name__ == "__main__":
    main()