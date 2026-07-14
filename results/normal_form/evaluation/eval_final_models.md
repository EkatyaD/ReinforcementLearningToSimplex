# `final_models/` evaluation — four 20M-step models

Evaluation of the four checkpoints in `final_models/` with `experiment.py`
(via `run_final_models_experiment.py`). Same analysis pipeline and shared
fixed-strategy baselines as `metacentrum_training/eval_40x40.md`, but here all
four models are run on **both** the in-distribution and out-of-distribution test
sets, and the current (full-cycle) `STEP_PENALTY_WEIGHTS_MATRIX` weights are used
for both training-time shaping (for the `weighted` models) and evaluation.

## Setup

| | |
|---|---|
| Base matrix | `base_matrix.py` (40×40) |
| N instances | 200 in-distribution + 200 out-of-distribution |
| In-dist generation | perturbed base matrix at `TEST_EPSILON = EPSILON = 0.001` |
| OOD generation | fresh uniform `[-1, 1]` matrices (new base each draw) |
| Seed | 42 |
| Maxiter | 20,000 |
| Phase | Phase-2 only (`USE_TWO_PHASE=False`, direct canonical Phase-2 tableau) |
| Tableau shape | (42, 83) |
| Fixed-strategy runs | 5 strategies × 200 instances, shared across all 4 models |

### Weights used (`STEP_PENALTY_WEIGHTS_MATRIX`, full-cycle calibration)

| strategy | weight |
|---|---:|
| largest_coefficient | 1.00 |
| steepest_edge | 1.89 |
| largest_increase | 3.17 |
| random_edge | 0.90 |
| blands_rule | 0.66 |

### The four models (2×2 design)

| Model | Observation | Step penalty | Policy | Size |
|---|---|---|---|---:|
| `compact_unweighted` | compact (Box, 31 feat) | flat −1/pivot | MlpPolicy | 1.8 MB |
| `compact_weighted`   | compact (Box, 31 feat) | per-rule weighted | MlpPolicy | 1.8 MB |
| `dict_unweighted`    | dict (full tableau) | flat −1/pivot | MultiInputPolicy | 24 MB |
| `dict_weighted`      | dict (full tableau) | per-rule weighted | MultiInputPolicy | 24 MB |

All five fixed strategies **and** all four RL agents converged on 100% of
instances in both test sets; game values agreed within 1e-4 everywhere. The
comparison below is purely about efficiency. `Mean Δ` is the mean per-instance
reduction `heuristic − RL` (positive ⇒ RL fewer pivots / lower cost). `p` is the
Wilcoxon signed-rank p on nonzero paired diffs; `(heur)` marks the cases where
the heuristic wins the majority.

---

## Fixed-strategy baselines (shared across all 4 models)

### In-distribution — pivot count

| Method | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| largest_coefficient | 61.74 | 60 | 31 | 98 |
| largest_increase | 44.42 | 44 | 28 | 74 |
| **steepest_edge** | **32.72** | 32 | 25 | 48 |
| random_edge | 121.53 | 121 | 79 | 184 |
| blands_rule | 143.26 | 145 | 89 | 204 |

### In-distribution — weighted cost

| Method | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| **largest_coefficient** | **61.74** | 60.0 | 31.0 | 98.0 |
| **steepest_edge** | **61.83** | 60.5 | 47.3 | 90.7 |
| blands_rule | 94.55 | 95.7 | 58.7 | 134.6 |
| random_edge | 109.37 | 108.9 | 71.1 | 165.6 |
| largest_increase | 140.83 | 139.5 | 88.8 | 234.6 |

> As in `eval_40x40.md`: under the full-cycle weights, largest_coefficient and
> steepest_edge are essentially tied (61.7 vs 61.8) — the 40×40 LP family sits
> right at the SE/LC indifference point.

### Out-of-distribution — pivot count

| Method | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| largest_coefficient | 59.62 | 58 | 36 | 94 |
| largest_increase | 47.97 | 46 | 30 | 104 |
| **steepest_edge** | **37.04** | 37 | 19 | 57 |
| random_edge | 122.05 | 119 | 74 | 181 |
| blands_rule | 165.55 | 163 | 89 | 285 |

### Out-of-distribution — weighted cost

| Method | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| **largest_coefficient** | **59.62** | 58.0 | 36.0 | 94.0 |
| **steepest_edge** | **70.01** | 69.9 | 35.9 | 107.7 |
| blands_rule | 109.26 | 107.6 | 58.7 | 188.1 |
| random_edge | 109.84 | 107.1 | 66.6 | 162.9 |
| largest_increase | 152.05 | 145.8 | 95.1 | 329.7 |

Bar to beat: **steepest_edge** on pivot count, **largest_coefficient** on weighted cost.

---

## Model 1: `compact_unweighted`

### In-distribution — pivot count (RL mean **29.29**, median 28, min 22, max 51)

| Opponent | RL wins | Ties | RL loses | Mean Δ | p |
|---|---:|---:|---:|---:|---:|
| largest_coefficient | 199 | 0 | 1 | +32.45 | <1e-6 |
| largest_increase | 185 | 5 | 10 | +15.13 | <1e-6 |
| steepest_edge | 176 | 10 | 14 | +3.42 | <1e-6 |
| random_edge | 200 | 0 | 0 | +92.23 | <1e-6 |
| blands_rule | 200 | 0 | 0 | +113.97 | <1e-6 |

vs best-per-instance: **90.5%** win+tie (169/12/19), mean reduction +8.4%.

### In-distribution — weighted cost (RL mean **61.89**, median 58.74, min 45.0, max 121.9)

| Opponent | RL wins | Ties | RL loses | Mean Δ | p |
|---|---:|---:|---:|---:|---:|
| largest_coefficient | 99 | 0 | 101 | −0.15 | 0.67 (heur) |
| largest_increase | 199 | 0 | 1 | +78.94 | <1e-6 |
| steepest_edge | 120 | 0 | 80 | −0.05 | 0.006 |
| random_edge | 199 | 0 | 1 | +47.49 | <1e-6 |
| blands_rule | 188 | 0 | 12 | +32.67 | <1e-6 |

vs best-per-instance: **36.0%** win (72/0/128), mean reduction −13.3%.

### OOD — pivot count (RL mean **39.04**, median 39, min 18, max 63)

| Opponent | RL wins | Ties | RL loses | Mean Δ | p |
|---|---:|---:|---:|---:|---:|
| largest_coefficient | 193 | 2 | 5 | +20.57 | <1e-6 |
| largest_increase | 164 | 5 | 31 | +8.93 | <1e-6 |
| steepest_edge | 58 | 16 | 126 | −2.00 | <1e-6 (heur) |
| random_edge | 200 | 0 | 0 | +83.00 | <1e-6 |
| blands_rule | 200 | 0 | 0 | +126.51 | <1e-6 |

vs best-per-instance: **34.5%** win+tie (52/17/131), mean reduction −7.4%.

### OOD — weighted cost (RL mean **88.72**, median 88.74, min 37.8, max 163.7)

| Opponent | RL wins | Ties | RL loses | Mean Δ | p |
|---|---:|---:|---:|---:|---:|
| largest_coefficient | 18 | 0 | 182 | −29.10 | <1e-6 (heur) |
| largest_increase | 198 | 0 | 2 | +63.33 | <1e-6 |
| steepest_edge | 16 | 0 | 184 | −18.71 | <1e-6 (heur) |
| random_edge | 158 | 0 | 42 | +21.12 | <1e-6 |
| blands_rule | 151 | 0 | 49 | +20.54 | <1e-6 |

vs best-per-instance: **2.0%** win (4/0/196), mean reduction −56.6%.

---

## Model 2: `compact_weighted`

### In-distribution — pivot count (RL mean **42.47**, median 42, min 27, max 72)

| Opponent | RL wins | Ties | RL loses | Mean Δ | p |
|---|---:|---:|---:|---:|---:|
| largest_coefficient | 178 | 1 | 21 | +19.27 | <1e-6 |
| largest_increase | 109 | 7 | 84 | +1.96 | 0.008 |
| steepest_edge | 23 | 6 | 171 | −9.76 | <1e-6 (heur) |
| random_edge | 200 | 0 | 0 | +79.06 | <1e-6 |
| blands_rule | 200 | 0 | 0 | +100.80 | <1e-6 |

vs best-per-instance: **12.0%** win+tie (18/6/176), mean reduction −34.3%.

### In-distribution — weighted cost (RL mean **49.10**, median 48.07, min 32.5, max 83.6)

| Opponent | RL wins | Ties | RL loses | Mean Δ | p |
|---|---:|---:|---:|---:|---:|
| largest_coefficient | 153 | 0 | 47 | +12.64 | <1e-6 |
| largest_increase | 200 | 0 | 0 | +91.73 | <1e-6 |
| steepest_edge | 174 | 0 | 26 | +12.74 | <1e-6 |
| random_edge | 200 | 0 | 0 | +60.28 | <1e-6 |
| blands_rule | 199 | 0 | 1 | +45.46 | <1e-6 |

vs best-per-instance: **71.5%** win (143/0/57), mean reduction +9.6%.

### OOD — pivot count (RL mean **52.48**, median 52, min 22, max 78)

| Opponent | RL wins | Ties | RL loses | Mean Δ | p |
|---|---:|---:|---:|---:|---:|
| largest_coefficient | 144 | 7 | 49 | +7.14 | <1e-6 |
| largest_increase | 61 | 5 | 134 | −4.51 | <1e-6 (heur) |
| steepest_edge | 8 | 5 | 187 | −15.44 | <1e-6 (heur) |
| random_edge | 200 | 0 | 0 | +69.57 | <1e-6 |
| blands_rule | 200 | 0 | 0 | +113.08 | <1e-6 |

vs best-per-instance: **4.5%** win+tie (5/4/191), mean reduction −45.9%.

### OOD — weighted cost (RL mean **58.47**, median 57.62, min 26.5, max 90.5)

| Opponent | RL wins | Ties | RL loses | Mean Δ | p |
|---|---:|---:|---:|---:|---:|
| largest_coefficient | 113 | 0 | 87 | +1.14 | 0.12 |
| largest_increase | 200 | 0 | 0 | +93.58 | <1e-6 |
| steepest_edge | 161 | 0 | 39 | +11.53 | <1e-6 |
| random_edge | 200 | 0 | 0 | +51.37 | <1e-6 |
| blands_rule | 199 | 0 | 1 | +50.79 | <1e-6 |

vs best-per-instance: **49.0%** win (98/0/102), mean reduction −3.1%.

---

## Model 3: `dict_unweighted`

### In-distribution — pivot count (RL mean **26.28**, median 25, min 20, max 49)

| Opponent | RL wins | Ties | RL loses | Mean Δ | p |
|---|---:|---:|---:|---:|---:|
| largest_coefficient | 200 | 0 | 0 | +35.46 | <1e-6 |
| largest_increase | 194 | 2 | 4 | +18.14 | <1e-6 |
| steepest_edge | 185 | 3 | 12 | +6.43 | <1e-6 |
| random_edge | 200 | 0 | 0 | +95.25 | <1e-6 |
| blands_rule | 200 | 0 | 0 | +116.98 | <1e-6 |

vs best-per-instance: **93.0%** win+tie (183/3/14), mean reduction +17.5%.

### In-distribution — weighted cost (RL mean **51.03**, median 48.84, min 37.2, max 96.9)

| Opponent | RL wins | Ties | RL loses | Mean Δ | p |
|---|---:|---:|---:|---:|---:|
| largest_coefficient | 156 | 0 | 44 | +10.71 | <1e-6 |
| largest_increase | 200 | 0 | 0 | +89.80 | <1e-6 |
| steepest_edge | 179 | 0 | 21 | +10.80 | <1e-6 |
| random_edge | 200 | 0 | 0 | +58.34 | <1e-6 |
| blands_rule | 198 | 0 | 2 | +43.52 | <1e-6 |

vs best-per-instance: **74.0%** win (148/0/52), mean reduction +6.3%.

### OOD — pivot count (RL mean **43.95**, median 44, min 19, max 72)

| Opponent | RL wins | Ties | RL loses | Mean Δ | p |
|---|---:|---:|---:|---:|---:|
| largest_coefficient | 182 | 2 | 16 | +15.67 | <1e-6 |
| largest_increase | 123 | 5 | 72 | +4.02 | 3e-6 |
| steepest_edge | 32 | 7 | 161 | −6.91 | <1e-6 (heur) |
| random_edge | 200 | 0 | 0 | +78.10 | <1e-6 |
| blands_rule | 200 | 0 | 0 | +121.61 | <1e-6 |

vs best-per-instance: **16.5%** win+tie (27/6/167), mean reduction −21.9%.

### OOD — weighted cost (RL mean **86.23**, median 86.34, min 39.8, max 135.7)

| Opponent | RL wins | Ties | RL loses | Mean Δ | p |
|---|---:|---:|---:|---:|---:|
| largest_coefficient | 7 | 0 | 193 | −26.61 | <1e-6 (heur) |
| largest_increase | 198 | 0 | 2 | +65.82 | <1e-6 |
| steepest_edge | 28 | 0 | 172 | −16.22 | <1e-6 (heur) |
| random_edge | 169 | 0 | 31 | +23.61 | <1e-6 |
| blands_rule | 160 | 0 | 40 | +23.03 | <1e-6 |

vs best-per-instance: **1.5%** win (3/0/197), mean reduction −52.5%.

---

## Model 4: `dict_weighted`

### In-distribution — pivot count (RL mean **27.59**, median 27, min 21, max 44)

| Opponent | RL wins | Ties | RL loses | Mean Δ | p |
|---|---:|---:|---:|---:|---:|
| largest_coefficient | 200 | 0 | 0 | +34.15 | <1e-6 |
| largest_increase | 190 | 1 | 9 | +16.83 | <1e-6 |
| steepest_edge | 177 | 3 | 20 | +5.12 | <1e-6 |
| random_edge | 200 | 0 | 0 | +93.93 | <1e-6 |
| blands_rule | 200 | 0 | 0 | +115.67 | <1e-6 |

vs best-per-instance: **87.5%** win+tie (171/4/25), mean reduction +13.1%.

### In-distribution — weighted cost (RL mean **40.11**, median 38.74, min 28.6, max 67.9)

| Opponent | RL wins | Ties | RL loses | Mean Δ | p |
|---|---:|---:|---:|---:|---:|
| largest_coefficient | 185 | 0 | 15 | +21.63 | <1e-6 |
| largest_increase | 200 | 0 | 0 | +100.72 | <1e-6 |
| steepest_edge | 197 | 0 | 3 | +21.72 | <1e-6 |
| random_edge | 200 | 0 | 0 | +69.26 | <1e-6 |
| blands_rule | 200 | 0 | 0 | +54.45 | <1e-6 |

vs best-per-instance: **91.5%** win (183/0/17), mean reduction +26.4%.

### OOD — pivot count (RL mean **42.44**, median 43, min 22, max 63)

| Opponent | RL wins | Ties | RL loses | Mean Δ | p |
|---|---:|---:|---:|---:|---:|
| largest_coefficient | 185 | 2 | 13 | +17.18 | <1e-6 |
| largest_increase | 128 | 15 | 57 | +5.53 | <1e-6 |
| steepest_edge | 38 | 10 | 152 | −5.39 | <1e-6 (heur) |
| random_edge | 200 | 0 | 0 | +79.61 | <1e-6 |
| blands_rule | 200 | 0 | 0 | +123.11 | <1e-6 |

vs best-per-instance: **21.5%** win+tie (32/11/157), mean reduction −17.1%.

### OOD — weighted cost (RL mean **67.92**, median 68.54, min 37.1, max 92.9)

| Opponent | RL wins | Ties | RL loses | Mean Δ | p |
|---|---:|---:|---:|---:|---:|
| largest_coefficient | 56 | 0 | 144 | −8.31 | <1e-6 (heur) |
| largest_increase | 200 | 0 | 0 | +84.13 | <1e-6 |
| steepest_edge | 115 | 0 | 85 | +2.08 | 0.002 |
| random_edge | 197 | 0 | 3 | +41.92 | <1e-6 |
| blands_rule | 194 | 0 | 6 | +41.34 | <1e-6 |

vs best-per-instance: **22.0%** win (44/0/156), mean reduction −20.3%.

---

## Side-by-side summary

### In-distribution

| Model | Pivot mean | Pivot win+tie vs best | Weighted mean | Weighted win vs best | Δ vs SE pivots | Δ vs LC weighted |
|---|---:|---:|---:|---:|---:|---:|
| compact_unweighted | 29.29 | 90.5% | 61.89 | 36.0% | +3.4 | −0.2 |
| compact_weighted | 42.47 | 12.0% | 49.10 | 71.5% | −9.8 | +12.6 |
| **dict_unweighted** | **26.28** | **93.0%** | 51.03 | 74.0% | +6.4 | +10.7 |
| **dict_weighted** | 27.59 | 87.5% | **40.11** | **91.5%** | +5.1 | +21.6 |

### Out-of-distribution

| Model | Pivot mean | Pivot win+tie vs best | Weighted mean | Weighted win vs best | Δ vs SE pivots | Δ vs LC weighted |
|---|---:|---:|---:|---:|---:|---:|
| compact_unweighted | 39.04 | 34.5% | 88.72 | 2.0% | −2.0 | −29.1 |
| **compact_weighted** | 52.48 | 4.5% | **58.47** | **49.0%** | −15.4 | +1.1 |
| dict_unweighted | 43.95 | 16.5% | 86.23 | 1.5% | −6.9 | −26.6 |
| dict_weighted | 42.44 | 21.5% | 67.92 | 22.0% | −5.4 | −8.3 |

(`Δ vs SE pivots` / `Δ vs LC weighted` = mean per-instance reduction against the
strongest fixed rule on each metric; positive ⇒ RL better.)

## Interpretation

1. **The step-penalty weighting cleanly selects the objective.** Unweighted
   models minimize raw pivot count (dict_unweighted is the pivot champion at
   26.28, beating the per-instance best heuristic 93% of the time); weighted
   models minimize wallclock cost (dict_weighted is the weighted champion at
   40.11, beating the per-instance best 91.5% of the time). On the *other*
   metric each is clearly worse — e.g. compact_unweighted has the lowest
   in-dist pivots of the compact pair but the highest weighted cost (61.89,
   essentially tied with the baselines and losing the weighted comparison).

2. **`dict` beats `compact` in-distribution on both metrics.** Full-tableau
   observations win on pivots (26.28 vs 29.29 unweighted) and on weighted cost
   (40.11 vs 49.10 weighted) — at 13× the model size and a policy locked to the
   40×40 shape. This matches the `eval_40x40.md` finding that observation type
   matters more than the exact weighting.

3. **Best single model in-distribution: `dict_weighted`.** It is the only model
   strong on *both* metrics in-distribution (87.5% pivot win+tie and 91.5%
   weighted win vs the per-instance best), and it cuts weighted cost 34.5% below
   steepest_edge and 32.1% below largest_coefficient (medians). Its weighted
   mean (40.11) reproduces the `eval_40x40.md` `dict_weighted` result (40.90).

4. **Generalization is the consistent weak spot.** OOD, **no model beats the
   per-instance best heuristic** on either metric — the fixed rules are strong
   on fresh matrices (SE ≈37 pivots, LC ≈60 weighted), and policies trained on
   the narrow ε=0.001 base-matrix distribution transfer only partially. All four
   still dominate the weak rules (random_edge, blands_rule) and beat the
   *average* heuristic; they just can't outrun the per-instance best.

5. **OOD wallclock favors `compact_weighted`.** It is the only model that holds
   near par with largest_coefficient OOD (58.47 vs 59.62, 49% win, Δ +1.1) — the
   size-independent observation generalizes best, exactly as designed, even
   though it loses to `dict` in-distribution. The unweighted models collapse OOD
   on weighted cost (compact 88.72 / dict 86.23, ~2% win) because they happily
   spend expensive SE / largest_increase pivots that the unweighted objective
   never penalized.

## Recommendation

- **Fewest iterations, in-distribution → `dict_unweighted`** (26.28 mean, 93% win+tie).
- **Lowest wallclock, in-distribution → `dict_weighted`** (40.11 mean, 91.5% win; best all-round).
- **Robustness to unseen distributions → `compact_weighted`** (only model near par OOD on weighted cost).
- Single overall pick: **`dict_weighted`** — strongest on the metric that matters
  (wallclock cost) while staying competitive on pivot count.

## Caveats

- OOD matrices come from a *different generator* (fresh uniform `[-1,1]`), not a
  larger perturbation of the training base — a hard transfer test.
- The `weighted` models were trained with these same full-cycle weights, so
  (unlike the `eval_40x40.md` models) there is **no train/eval weight mismatch**
  here — the weighted-cost numbers judge them on their actual training objective.
- The `clip_range` / `lr_schedule` deserialization warnings on load are benign
  (training-only objects; they do not affect `model.predict`).
- Weighted-cost numbers are wallclock estimates for the (42, 83) Phase-2 tableau;
  meaningful relative to each other, not as real seconds.

## Artifacts

`final_models_results/<model>.log` (full printed analysis) and `<model>.json`
(per-instance status / nit / weighted_cost / objective), produced by
`run_final_models_experiment.py`.