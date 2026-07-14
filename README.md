# RL-guided simplex solver for zero-sum games

A reinforcement-learning agent (PPO) that learns to **dynamically select the pivot rule** at
each simplex pivot, instead of committing to a single fixed heuristic. At every phase-2 pivot
the agent chooses among **largest coefficient, largest increase, and steepest edge**
(`PIVOT_MAP` in `config.py`), aiming to solve each linear program in fewer (or cheaper) pivots
than any one fixed rule. Evaluation compares it against **five** fixed heuristics — those three
plus the eval-only baselines **random edge** and **Bland's rule**. In all shipped results the
agent acts in phase 2 of the two-phase simplex method; phase 1 (when needed) is solved with a
fixed steepest-edge-style rule.

Two problem families are supported via `GAME_MODE` in `config.py`:

- **`matrix`** — random perturbed 40×40 zero-sum payoff matrices (normal-form games).
- **`leduc`** — sequence-form LPs from Leduc poker with Dirichlet-sampled deck weights
  (requires `open_spiel` / `pyspiel`).

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # curated direct dependencies
# For the Leduc experiments, also install OpenSpiel (not on PyPI as a wheel
# everywhere — see requirements.txt):
#   pip install open_spiel   # then verify: python -c "import pyspiel"
```

To reproduce the exact frozen environment the thesis results were produced with
(all transitive dependencies, CUDA wheels, etc.), install the lock file instead:

```bash
pip install -r requirements-lock.txt
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
                     (which parts are SciPy-derived and how they're used: see SCIPY_CODE.md)
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
