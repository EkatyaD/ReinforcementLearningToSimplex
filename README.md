# RL-guided simplex solver for zero-sum games

A reinforcement-learning agent (PPO) that learns to **dynamically select the pivot rule** at
each step of the two-phase simplex method, instead of committing to a single fixed heuristic.
At every pivot the agent chooses among **Bland's rule, largest coefficient, largest increase,
steepest edge, and random edge**, aiming to solve each linear program in fewer (or cheaper)
pivots than any one fixed rule.

Two problem families are supported via `GAME_MODE` in `config.py`:

- **`matrix`** — random perturbed 40×40 zero-sum payoff matrices (normal-form games).
- **`leduc`** — sequence-form LPs from Leduc poker with Dirichlet-sampled deck weights
  (requires `open_spiel` / `pyspiel`).

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # add open_spiel for the Leduc experiments
```

## Layout

```
config.py            central configuration (game mode, sizes, reward flags, model naming)
train.py             PPO training pipeline (honors GAME_MODE)
experiment.py        matrix-mode evaluation vs the 5 fixed heuristics
leduc_experiment_runner.py / experiment_leduc_fullpivot.py   Leduc evaluation
envs.py wrappers.py callbacks.py   gym environments, obs/reward wrappers, training callbacks
matrix.py base_matrix.py           payoff-matrix generation + the 40×40 base matrix
simplex_solver.py _linprog_utils.py io_utils.py   two-phase simplex + LP helpers
leduc_experiment.py  sequence-form LP construction for Leduc/Kuhn
cluster/             PBS/Metacentrum job scripts (training + evaluation) — reproducibility reference
calibration/         benchmarks that derive the per-rule STEP_PENALTY_WEIGHTS
results/             trained models + evaluations (see below)
```

## Reproducing the results

The two thesis result sets are self-contained under `results/`, each with its own README giving
the exact training config and one-command re-evaluation:

- **`results/normal_form/`** — the four 40×40 matrix agents (compact/dict × weighted/unweighted),
  their evaluations, and the base matrix. → [README](results/normal_form/README.md)
- **`results/leduc/`** — the final Leduc pair (compact weighted/unweighted, 30M steps) plus the
  empty-observation control baseline. → [README](results/leduc/README.md)

The evaluation drivers load the shipped models and re-run the comparison against the fixed
heuristics; training from scratch (20–30M steps) is heavy and best run on a cluster
(see `cluster/`).
