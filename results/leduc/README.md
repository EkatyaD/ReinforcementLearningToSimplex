# Leduc poker results

RL-guided pivot selection on **sequence-form LPs from Leduc poker** with Dirichlet-sampled
deck weights, phase-2 only. The final models are the compact-observation pair (weighted and
unweighted step penalty) at 30M steps.

## Contents

| Path | What |
|---|---|
| `models/` | the two final agents (`compact_{weighted,unweighted}`, 30M-step checkpoints) |
| `evaluation/` | `eval_new_models.md` (full report) + `eval_new_models.json` + `eval_run.log` |
| `eval_leduc.py` | re-run the evaluation of both models |
| `empty_obs_baseline/` | information-free control baseline (see its README) |

## The two models

Both use the **compact** observation (the full Leduc tableau is ~483×965, so a dict-obs policy
is infeasible — compact obs is auto-forced in leduc mode). They differ only in the step penalty:

| Model file | `USE_WEIGHTED_STEP_PENALTY` |
|---|:--:|
| `ppo_leduc_ckpt_30000000_alpha100.0_compact_weighted.zip`   | `True`  |
| `ppo_leduc_ckpt_30000000_alpha100.0_compact_unweighted.zip` | `False` |

Weighted cost uses `STEP_PENALTY_WEIGHTS_LEDUC`. Evaluation runs three sets: in-distribution
(α=100), out-of-distribution (α=10), and a uniform fair deck. See
`evaluation/eval_new_models.md` for the full analysis.

## Reproduce — evaluation

From the repo root (works regardless of the current `GAME_MODE` — the driver imports the Leduc
weights explicitly):

```bash
python results/leduc/eval_leduc.py       # writes into results/leduc/evaluation/
```

Requires `open_spiel` / `pyspiel` (Leduc game tree).

## Reproduce — training

Set these in `config.py`, then run `python train.py` once per model:

```
GAME_MODE       = "leduc"
LEDUC_GAME      = "leduc_poker(suit_isomorphism=true)"
LEDUC_ALPHA     = 100.0
LEDUC_NUM_RANKS = 3
TIMESTEPS       = 30_000_000
# USE_COMPACT_OBS is auto-forced True in leduc mode
USE_WEIGHTED_STEP_PENALTY = True   # -> compact_weighted;  False -> compact_unweighted
```

The 30M checkpoint is written to `models/ppo_leduc_ckpt_30000000_alpha100.0_compact_<penalty>.zip`
(top-level `models/`); move it into `results/leduc/models/` to evaluate it here. Training is
heavy — see `../../cluster/simplexJob.sh`.
