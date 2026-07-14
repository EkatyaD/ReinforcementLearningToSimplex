# Normal-form (matrix game) results

RL-guided pivot selection on random perturbed **40×40 zero-sum payoff matrices**,
phase-2 only. Four PPO agents form a 2×2 design over the observation type and the
step-penalty shaping; each is compared against the five fixed pivot heuristics.

## Contents

| Path | What |
|---|---|
| `models/` | the four trained agents (20M steps each) |
| `evaluation/` | `eval_final_models.md` (full report) + per-model `.log`/`.json` + `epsilon_sweep/` |
| `base_matrix.py` | the 40×40 base matrix these results were trained/evaluated on (identical to the repo-root `base_matrix.py`) |
| `eval_normal_form.py` | re-run the evaluation of all four models |
| `run_epsilon_sweep.py` | optional generalization probe (sweeps the test perturbation ε) |

## The four models

| Model file (`…epsilon0.001_<tag>.zip`) | `USE_COMPACT_OBS` | `USE_WEIGHTED_STEP_PENALTY` | policy |
|---|:--:|:--:|---|
| `…_compact_unweighted` | `True`  | `False` | MlpPolicy |
| `…_compact_weighted`   | `True`  | `True`  | MlpPolicy |
| `…_dict_unweighted`    | `False` | `False` | MultiInputPolicy |
| `…_dict_weighted`      | `False` | `True`  | MultiInputPolicy |

`compact` = size-independent 31-feature observation; `dict` = full-tableau observation
(24 MB models, policy locked to the 40×40 shape). `weighted` scales the per-pivot penalty by
each rule's empirical wallclock cost (`STEP_PENALTY_WEIGHTS_MATRIX`); `unweighted` charges a
flat −1/pivot. See `evaluation/eval_final_models.md` for the full analysis and recommendation.

## Reproduce — evaluation

From the repo root (no config edits needed — the driver forces the matrix weights and flips
the observation flag per model):

```bash
python results/normal_form/eval_normal_form.py       # writes into results/normal_form/evaluation/
python results/normal_form/run_epsilon_sweep.py      # optional ε-sweep
```

## Reproduce — training

Set these in `config.py`, then run `python train.py` once per model:

```
GAME_MODE          = "matrix"
M = N              = 40
MIN_VAL, MAX_VAL   = -1, 1
EPSILON            = 0.001
TIMESTEPS          = 20_000_000
USE_FULL_PIVOT     = False        # agent plays phase 2 only
USE_TWO_PHASE      = False
```

Then pick the 2×2 cell for each model:

```
USE_COMPACT_OBS         = True/False   # compact vs dict
USE_WEIGHTED_STEP_PENALTY = True/False # weighted vs unweighted
```

The trained model is written to `models/ppo_simplex_random_20000000_matrix40x40_min-1_max1_epsilon0.001_<obs>_<penalty>.zip`
(top-level `models/`); move it into `results/normal_form/models/` to evaluate it here.
Training is heavy (20M steps) — see `../../cluster/simplexJob.sh` for the PBS/Metacentrum job.
