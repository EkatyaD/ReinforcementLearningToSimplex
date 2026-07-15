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

## Use of generative AI

Generative AI was used in the preparation of this repository:
for code review, refactoring (dead-code removal, deduplication, naming), writing
documentation (docstrings, READMEs, `SCIPY_CODE.md`), dependency and licensing
clean-up, and for verifying reproducibility (AI-assisted changes were checked by
re-running the evaluation pipeline and confirming the committed results are
reproduced byte-for-byte). Commits with AI assistance are marked with a
`Co-Authored-By` trailer in the git history. All changes were reviewed by the
author, who takes full responsibility for the contents of this repository.

## Install

> **Use Python 3.12 or 3.13.** The pinned dependency versions were frozen for
> this range and ship prebuilt wheels for it. On Python 3.14 (or newer) several
> pins have no wheel yet and pip tries to compile them from source — e.g.
> `pygame` fails without SDL dev headers. If you must stay on 3.14, install the
> curated deps *unpinned* (`pip install numpy scipy torch stable_baselines3
> gymnasium pandas matplotlib tensorboard tqdm rich`) to pull 3.14-compatible
> versions; note this is not the exact environment the results were produced in.

```bash
python3.12 -m venv venv && source venv/bin/activate   # or python3.13
pip install -r requirements.txt          # curated direct dependencies
# For the Leduc experiments, also install OpenSpiel (not on PyPI as a wheel
# everywhere — see requirements.txt):
#   pip install open_spiel   # then verify: python -c "import pyspiel"
```

To reproduce the exact frozen environment the thesis results were produced with
(all transitive dependencies, CUDA wheels, etc.), install the lock file instead
(also on Python 3.12/3.13):

```bash
pip install -r requirements-lock.txt
```

## Layout

```
config.py  simplex_solver.py  _linprog_utils.py            configuration + simplex core
envs.py  wrappers.py  matrix.py  base_matrix.py             the RL problem (envs, obs, LP instances)
train.py  callbacks.py  io_utils.py                         training
experiment.py  leduc_experiment.py  leduc_experiment_runner.py  compare_strategies.py   evaluation
calibration/  cluster/  results/                            supporting material + shipped results
```

### Configuration and simplex core

- **`config.py`** — every experiment knob, as module-level constants. `GAME_MODE`
  switches between the two problem families. `PIVOT_MAP` is the agent's 3-rule
  **training action space** (largest coefficient, largest increase, steepest edge);
  `PIVOT_MAP_TEST` adds the two evaluation-only baselines (random edge, Bland's
  rule). `STEP_PENALTY_WEIGHTS_{MATRIX,LEDUC}` are the empirically calibrated
  per-rule wallclock weights behind the "weighted cost" metric (auto-selected by
  game mode); `MODEL_RUN_TAG` encodes the observation/penalty configuration into
  model filenames so runs don't overwrite each other.
- **`simplex_solver.py`** — the simplex mathematics: the five pivot-column
  heuristics (`_pivot_col_heuristics`), the Harris two-pass ratio test
  (`_pivot_row`), pivot application (`_apply_pivot`), a fixed-rule Phase 1 solver
  (`phase1solver` + `first_to_second`), and the zero-sum tableau constructions —
  a direct Phase-2 route (`change_to_zero_sum_phase2_only`: positivity shift, a
  provably feasible trivial basis, canonicalization) and a general two-phase
  route with artificial variables (`change_to_zero_sum`).
- **`_linprog_utils.py`** — vendored SciPy module (BSD-3, see `LICENSES/`):
  LP validation and standard-form conversion (`_parse_linprog`, `_get_Abc`,
  `_LPProblem`). Which parts of the codebase are SciPy-derived and exactly how
  they are used is documented in [`SCIPY_CODE.md`](SCIPY_CODE.md).

### The RL problem

- **`envs.py`** — Gymnasium environments. `SecondPhasePivotingEnv` is the base:
  the state is a Phase-2 tableau, an action picks one of the 3 pivot rules, a step
  applies that pivot; reward is −cost per pivot plus a terminal success bonus, with
  basis-repeat cycle detection. `RandomMatrixEnv` resamples a perturbed payoff
  matrix each episode; `LeducEnv` resamples a Leduc sequence-form LP with
  Dirichlet-weighted decks (Phase 1 solved internally by a fixed rule).
- **`wrappers.py`** — observation wrappers. `CompactObsWrapper` replaces the
  full-tableau Dict observation with 31 size-independent features (action/progress
  history, reduced-cost and ratio-test statistics) so one policy works across LP
  sizes — mandatory for Leduc, where the tableau is ~483×965. `EmptyObsWrapper`
  feeds a constant observation: the information-free control baseline used to
  detect whether an agent actually uses the state or has collapsed to a fixed rule.
- **`matrix.py` / `base_matrix.py`** — `Matrix` holds a base payoff matrix and
  generates per-episode ε-perturbations of it; `base_matrix.py` stores the fixed
  40×40 base matrix the shipped agents were trained on (a copy is archived with
  the results in `results/normal_form/`).

### Training

- **`train.py`** — the training entry point: builds the vectorized env for the
  configured `GAME_MODE`, applies the observation wrapper, and trains PPO
  (stable-baselines3) with periodic checkpoints and a save-on-best callback.
  `python train.py` runs one configuration; `--grid-search` sweeps a small PPO
  hyperparameter grid.
- **`callbacks.py`** — the SB3 callbacks used above (checkpoint schedule,
  save-on-best by rolling episode length, per-rollout episode counter).
- **`io_utils.py`** — persists a freshly generated base matrix back into
  `base_matrix.py` (only used when training starts without a valid base matrix).

### Evaluation

- **`experiment.py`** — the matrix-mode evaluation. Builds two test sets
  (in-distribution: ε-perturbations of the training base matrix; out-of-
  distribution: fresh uniform matrices), solves every instance with all five
  fixed heuristics **and** the agent from identical starting tableaus, and
  reports pivot counts, weighted cost, per-instance win/tie/loss (including vs
  the best heuristic per instance) and Wilcoxon signed-rank tests. Also runnable
  standalone: `python experiment.py --model <path>`.
- **`leduc_experiment.py`** — constructs the sequence-form LP of Leduc/Kuhn poker
  from the OpenSpiel game tree (Koller–Megiddo–von Stengel realization-plan
  constraints), including the Dirichlet deck reweighting, plus a SciPy/HiGHS
  reference solver used to cross-check game values.
- **`leduc_experiment_runner.py`** — the Leduc analogue of `experiment.py`:
  samples Leduc LPs, runs heuristics + agent, same analysis.
- **`compare_strategies.py`** — a small manual harness: solves one perturbed
  matrix with every rule and a shipped agent and prints pivot counts + recovered
  game values (a quick sanity check, not part of the formal evaluation).

### Supporting material

- **`calibration/`** — the wallclock benchmarks used to derive the per-rule
  `STEP_PENALTY_WEIGHTS` (one script per problem family).
- **`cluster/`** — PBS/Metacentrum job scripts used for the actual training
  runs; kept as a reproducibility reference.
- **`results/`** — the shipped trained models and committed evaluations
  (see below).

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
