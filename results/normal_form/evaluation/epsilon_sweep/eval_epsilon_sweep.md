# TEST_EPSILON sweep — `dict_unweighted`

How well does the `dict_unweighted` agent (trained at `EPSILON = 0.001`) hold up
when the **in-distribution test perturbation** is scaled 10× and 100× larger than
training? `TEST_EPSILON` controls the magnitude of the perturbation applied to the
base matrix when generating in-distribution test matrices; the model and base
matrix are unchanged, only the test set moves further from training.

| | |
|---|---|
| Model | `ppo_simplex_random_20000000_..._dict_unweighted` (dict obs, flat −1/pivot) |
| Training EPSILON | 0.001 |
| TEST_EPSILON | 0.001, 0.01, 0.1 |
| N instances | 200 in-distribution, seed 42 |
| Mode | in-distribution only (OOD set is epsilon-independent) |

All 5 fixed strategies and the RL agent converged on 100% of instances at every
ε; game values consistent within 1e-4. Driver: `run_epsilon_sweep.py`.

## Summary

### Pivot count

| ε (test) | RL mean | steepest_edge mean | RL vs SE (W/T/L) | win+tie vs best | mean red. vs best |
|---:|---:|---:|---|---:|---:|
| 0.001 | **26.28** | 32.72 | 185 / 3 / 12 (RL) | **93.0%** | +17.5% |
| 0.01  | 28.57 | 32.80 | 154 / 11 / 35 (RL) | 81.5% | +11.2% |
| 0.1   | 31.73 | 31.36 | 86 / 24 / 90 (heur, p=0.23) | 52.5% | −2.8% |

### Weighted cost

| ε (test) | RL mean | LC mean | SE mean | win vs best | mean red. vs best |
|---:|---:|---:|---:|---:|---:|
| 0.001 | **51.03** | 61.74 | 61.83 | **74.0%** | +6.3% |
| 0.01  | 56.74 | 62.59 | 61.98 | 49.0% | −3.7% |
| 0.1   | 63.14 | 61.80 | 59.27 | 24.5% | −16.4% |

(For reference, the same model on the fully OOD set — fresh uniform matrices —
scored 43.95 pivots / 86.23 weighted, far worse than even ε=0.1.)

## What happens

1. **The agent degrades smoothly and monotonically as ε grows.** RL pivot mean
   climbs 26.3 → 28.6 → 31.7 and weighted mean 51.0 → 56.7 → 63.1, while the
   fixed heuristics barely move (steepest_edge stays ~31–33 pivots; LC ~62
   weighted). The LP family isn't getting harder — the *agent* is drifting off
   the distribution it memorized around the base matrix.

2. **ε = 0.01 (10× training): still a solid win.** The agent keeps beating the
   per-instance best heuristic 81.5% of the time on pivots and is roughly at par
   on weighted cost (49% win, −3.7% mean). Generalization to a 10× larger
   perturbation is essentially intact.

3. **ε = 0.1 (100× training): the edge is gone.** On pivot count the agent is now
   statistically tied with steepest_edge (86 W / 24 T / 90 L, Wilcoxon p=0.23 —
   no significant difference), and win+tie vs the per-instance best drops to
   52.5%. On weighted cost it now *loses* to both LC (114/200) and SE (121/200),
   −16.4% vs the per-instance best. It still crushes the weak rules
   (largest_increase, random_edge, blands_rule) by 30–55% everywhere.

4. **Structured perturbation ≫ fresh uniform.** Even at ε=0.1 the perturbed-base
   test set (31.7 pivots) is much easier for the agent than the fully OOD
   uniform set (43.95). Scaling the perturbation moves the test toward, but does
   not reach, the difficulty of genuinely new matrices — the base-matrix
   structure still helps the agent a lot.

## Takeaway

The `dict_unweighted` policy generalizes comfortably to ~10× its training
perturbation, then loses its advantage by ~100×, converging to roughly
steepest_edge-level performance rather than collapsing. The degradation is
graceful (no convergence failures, no blow-ups), but it confirms the agent's
gains are tied to the training distribution: at ε=0.1 a practitioner would do
just as well running steepest_edge directly.

## Artifacts

`final_models_results/epsilon_sweep/*_eps{0.001,0.01,0.1}.log` (full analysis)
and matching `.json` (per-instance results).