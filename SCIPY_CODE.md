# How the SciPy-derived code works and where it is used

This repository contains two kinds of SciPy-derived code, both under the BSD
3-Clause license (full text: [`LICENSES/SCIPY-BSD-3.txt`](LICENSES/SCIPY-BSD-3.txt)):

1. **`_linprog_utils.py`** — a vendored copy of SciPy's
   `scipy/optimize/_linprog_util.py` (LP problem parsing and standard-form
   conversion). Not original to this thesis.
2. **Pivot mechanics in `simplex_solver.py`** — `_pivot_row`, `_apply_pivot`
   and the `phase1solver` loop are *adapted from* SciPy's dense tableau simplex
   implementation (`scipy.optimize._linprog_simplex`, the legacy
   `method="simplex"` solver), modified as described below.

Everything else — the pivot-rule heuristics, the zero-sum tableau
constructions, the gym environments, training and evaluation — is thesis code.

---

## 1. The vendored module: `_linprog_utils.py`

Only three names are imported from it (see `simplex_solver.py`, `envs.py`,
`leduc_experiment.py`):

| Name | What it does |
|---|---|
| `_LPProblem` | A namedtuple bundling one LP: `(c, A_ub, b_ub, A_eq, b_eq, bounds, x0, integrality)`. Used as the container handed to the two functions below. |
| `_parse_linprog(lp, options, meth)` | Input validation and clean-up: checks shapes/dtypes, converts `bounds` to a canonical `(n, 2)` array, resolves solver options (notably the tolerance `tol`). Returns the cleaned `_LPProblem` plus the options dict. |
| `_get_Abc(lp, c0)` | Converts the general-form LP into **standard form**: minimize `cᵀx` subject to `A x = b`, `x ≥ 0`. It shifts variables with finite lower bounds to zero, substitutes variables that only have an upper bound, **splits free variables into a difference of two non-negative variables** (`x = x⁺ − x⁻`), and appends one slack variable per inequality row (`A = [A_ub | I ; A_eq | 0]`). Returns `A, b, c, c0, x0`, where `c0` is the constant objective offset collected by the substitutions. |

The rest of the vendored module (`_presolve`, `_autoscale`, `_postsolve`,
`_check_result`, …) is **not used** by this project; the file is kept whole so
it stays diffable against upstream SciPy.

Why standard form matters here: the simplex tableau format used throughout the
project assumes equality constraints with non-negative variables. The
game-theoretic LPs are naturally stated with inequalities and one *free*
variable (the game value `v`), so `_get_Abc` is what turns them into something
a tableau can represent — the free `v` becomes `v⁺ − v⁻`, and each "column
strategy payoff ≤ v" inequality gains a slack.

## 2. Adapted pivot mechanics in `simplex_solver.py`

| Function | Origin and changes |
|---|---|
| `_apply_pivot(T, basis, pivrow, pivcol, tol)` | Near-verbatim from SciPy: Gauss–Jordan elimination on the pivot element (normalize the pivot row, subtract its multiples from every other row), updating `basis[pivrow] = pivcol`. The numerical-stability warning message is SciPy's. |
| `_pivot_row(T, basis, pivcol, phase, tol, bland)` | The ratio test (choose the leaving row). SciPy's exact minimum-ratio test was replaced with a **Harris-style two-pass test**: pass 1 finds the minimum ratio `q_min = min(b_i / a_i,pivcol)` over rows with positive pivot-column entries; pass 2 forms an "admissible band" of rows with ratio ≤ `(1+η)·q_min` (η = 1e-7) and picks the numerically strongest pivot (largest column entry) in the band. With `bland=True` it instead picks the smallest basis index within the band. The `phase` argument keeps SciPy's convention that a phase-1 tableau carries **two** bottom rows (objective + pseudo-objective) to exclude from the test. |
| `phase1solver(T, basis, …)` | Adapted from the phase-1 portion of SciPy's `_solve_simplex` loop: repeatedly pick an entering column, run the ratio test, apply the pivot, until the pseudo-objective can no longer improve. The column rule is fixed to the project's `steepest_edge` heuristic instead of SciPy's Dantzig rule. |
| `_pivot_col_heuristics(T, strategy, tol)` | Thesis code. Generalizes SciPy's binary column choice (Dantzig vs. Bland) to the five named rules the experiments compare: `largest_coefficient` (= Dantzig), `blands_rule` (leftmost negative reduced cost), `largest_increase`, `steepest_edge`, `random_edge`. |

## 3. End-to-end: from a zero-sum game to a tableau

**Matrix mode, default path (`USE_TWO_PHASE = False`, all shipped normal-form
results).** The SciPy plumbing is *bypassed*: `change_to_zero_sum_phase2_only`
builds the standard-form arrays `A, b, c` for the shifted game directly (the
positivity shift `K` guarantees a trivial basic feasible solution — all slacks
plus one strategy variable), and `build_phase2_tableau_canonical` assembles the
canonical phase-2 tableau from the chosen basis (`B⁻¹N`, `B⁻¹b`, reduced costs).
No phase 1, no artificial variables, no `_get_Abc`.

**Matrix mode, two-phase path (`USE_TWO_PHASE = True`), and Leduc (always).**

```
game LP  ──_LPProblem──▶  _parse_linprog  ──▶  _get_Abc  ──▶  A, b, c (standard form)
                                                              │
      change_to_zero_sum / LeducEnv._init_env: flip negative-b rows,
      add one artificial variable per row (initial basis), append the
      objective row and the phase-1 pseudo-objective row
                                                              │
                    phase1solver (drive artificials to zero) ─┘
                                                              │
      first_to_second: drop the pseudo row, delete artificial columns;
      the env's remove_artificial pivots any still-basic artificial out
                                                              ▼
                              phase-2 tableau the RL agent pivots on
```

For matrix games the LP is the minimax LP (`max v` s.t. `xᵀP ≥ v·1`, `Σx = 1`);
for Leduc it is the sequence-form LP built in `leduc_experiment.py`
(Koller–Megiddo–von Stengel realization-plan constraints). Both enter the same
SciPy plumbing and the same phase-1 machinery.

## 4. Summary of provenance

| Component | Provenance |
|---|---|
| `_linprog_utils.py` (all of it) | SciPy, vendored verbatim (BSD-3) |
| `_apply_pivot`, `_pivot_row` skeleton, `phase1solver` loop | SciPy `_linprog_simplex`, adapted |
| Pivot-rule heuristics, Harris band, zero-sum/tableau constructions, sequence-form LP builder, gym envs, wrappers, training, evaluation | Thesis code (MIT) |
