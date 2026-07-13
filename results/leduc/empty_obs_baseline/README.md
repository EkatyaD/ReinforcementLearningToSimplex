# Empty-observation baseline (control)

A sanity-check control for the Leduc agents. Here the observation is replaced by a **single
constant feature** (via `EmptyObsWrapper`), so the policy is **state-independent by
construction** — it cannot condition its pivot choice on the tableau and can only learn one
fixed action distribution ("which single rule is best on average").

If a real (compact-obs) agent's performance matched this baseline, that would be evidence it
had collapsed to a fixed rule rather than actually using the tableau. Running this baseline
prints each model's per-rule pivot usage, making the collapse explicit.

## Contents

| Path | What |
|---|---|
| `models/` | three empty-obs Leduc agents (weighted / unweighted checkpoints) |
| `eval_empty.py` | run all three on sampled Leduc LPs; report pivot-rule usage |

## Reproduce

From the repo root (requires `open_spiel` / `pyspiel`):

```bash
python results/leduc/empty_obs_baseline/eval_empty.py
```
