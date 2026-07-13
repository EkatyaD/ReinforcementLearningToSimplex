import argparse
import importlib.util
import numpy as np
import json
from collections import defaultdict

from stable_baselines3 import PPO
from scipy.stats import wilcoxon

from matrix import Matrix
from simplex_solver import (
    change_to_zero_sum_phase2_only,
    change_to_zero_sum, phase1solver, first_to_second,
    _pivot_col_heuristics, _pivot_row, _apply_pivot,
)
from envs import SecondPhasePivotingEnv
from wrappers import CompactObsWrapper
from config import (
    M, N, MIN_VAL, MAX_VAL, EPSILON, TEST_EPSILON, TIMESTEPS,
    MODEL_NAME_TEMPLATE, PIVOT_MAP, PIVOT_MAP_TEST,
    PIVOT_STRATEGY_NAMES, USE_TWO_PHASE, USE_COMPACT_OBS,
    STEP_PENALTY_WEIGHTS,
)
from base_matrix import BASE_MATRIX


def load_base_matrix_from_file(path):
    """Load BASE_MATRIX from an arbitrary base_matrix.py file."""
    spec = importlib.util.spec_from_file_location("custom_base_matrix", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BASE_MATRIX

MAXITER = 20_000
TOL = 1e-9


# ---------------------------------------------------------------------------
# Tableau preparation
# ---------------------------------------------------------------------------

def prepare_tableau(matrix_P):
    """Build Phase 2 tableau from a payoff matrix and remove artificial variables.

    Returns (T, basis) ready for pivoting, or None on failure.
    """
    if USE_TWO_PHASE:
        T, basis, av = change_to_zero_sum(matrix_P)
        nit, status = phase1solver(T, basis)
        if status != 0:
            return None
        res = first_to_second(T, basis, av)
        if res is None:
            return None
        T, basis = res
    else:
        res = change_to_zero_sum_phase2_only(matrix_P)
        if res is None:
            return None
        T, basis, K = res
    # Remove artificial variables from basis (mirrors SecondPhasePivotingEnv.remove_artificial)
    for pivrow in [row for row in range(basis.size) if basis[row] > T.shape[1] - 2]:
        non_zero_cols = [col for col in range(T.shape[1] - 1) if abs(T[pivrow, col]) > TOL]
        if non_zero_cols:
            _apply_pivot(T, basis, pivrow, non_zero_cols[0], TOL)
    return T, basis


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_fixed_strategy(T, basis, strategy):
    """Run a single fixed pivot strategy on a copy of the tableau.

    Returns dict with status, nit, weighted_cost, objective.
    `weighted_cost` is `nit * STEP_PENALTY_WEIGHTS[strategy]` — the sum of
    per-pivot wallclock-cost weights (constant for a fixed strategy).
    """
    T = T.copy()
    basis = basis.copy()
    nit = 0
    seen_bases = {tuple(int(i) for i in basis)}
    w = float(STEP_PENALTY_WEIGHTS.get(strategy, 1.0))

    while nit < MAXITER:
        found, pivcol = _pivot_col_heuristics(T, strategy=strategy, tol=TOL)
        if not found:
            return {"status": "optimal", "nit": nit, "weighted_cost": nit * w,
                    "objective": float(T[-1, -1])}

        use_bland = (strategy == "blands_rule")
        found, pivrow = _pivot_row(T, basis, pivcol, phase=2, tol=TOL, bland=use_bland)
        if not found:
            return {"status": "no_pivot_row", "nit": nit, "weighted_cost": nit * w,
                    "objective": float(T[-1, -1])}

        _apply_pivot(T, basis, pivrow, pivcol, tol=TOL)
        nit += 1

        key = tuple(int(i) for i in basis)
        if key in seen_bases:
            return {"status": "loop", "nit": nit, "weighted_cost": nit * w,
                    "objective": float(T[-1, -1])}
        seen_bases.add(key)

    return {"status": "maxiter", "nit": nit, "weighted_cost": nit * w,
            "objective": float(T[-1, -1])}


def run_rl_agent(T, basis, model):
    """Run the trained RL agent on a copy of the tableau.

    Returns dict with status, nit, weighted_cost, objective.
    `weighted_cost` is the sum of `STEP_PENALTY_WEIGHTS[PIVOT_MAP[action]]`
    over the actions the agent actually picked — a wallclock-cost analogue
    of the raw pivot count.
    """
    base_env = SecondPhasePivotingEnv(T.copy(), basis.copy())
    env = CompactObsWrapper(base_env) if USE_COMPACT_OBS else base_env
    obs, _ = env.reset()
    done = False
    truncated = False
    weighted_cost = 0.0
    info = {}

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        a = int(action)
        # Look up the rule the agent picked; only count the pivot toward the
        # weighted cost if the env actually applies it (i.e. status != optimal).
        strategy = PIVOT_MAP.get(a)
        prev_nit = base_env.nit
        obs, _, done, truncated, info = env.step(action)
        if base_env.nit > prev_nit and strategy is not None:
            weighted_cost += float(STEP_PENALTY_WEIGHTS.get(strategy, 1.0))

    status = info.get("status", "unknown")
    if truncated and not done:
        status = "loop"

    return {"status": status, "nit": base_env.nit,
            "weighted_cost": weighted_cost,
            "objective": float(base_env.T[-1, -1])}


# ---------------------------------------------------------------------------
# Test matrix generation
# ---------------------------------------------------------------------------

def generate_test_matrices(n_matrices, mode="in_distribution", base_matrix=None):
    """Generate a list of raw payoff matrices (np.ndarray).

    mode="in_distribution":  perturbations of the training base matrix.
    mode="out_of_distribution": fresh random matrices (new base each time).
    base_matrix: optional override for the base matrix (default: BASE_MATRIX from base_matrix.py).
    """
    if base_matrix is None:
        base_matrix = BASE_MATRIX

    matrices = []



    if mode == "in_distribution":
        # Use TEST_EPSILON for the perturbation magnitude. When TEST_EPSILON
        # equals EPSILON (the default), the test set matches the training
        # distribution; when it differs, this probes generalization to a
        # different perturbation magnitude than the agent was trained on.
        base = Matrix(m=M, n=N, min=MIN_VAL, max=MAX_VAL,
                      epsilon=TEST_EPSILON, base_P=base_matrix)
        for _ in range(n_matrices):
            perturbed = base.generate_perturbed_matrix()
            matrices.append(perturbed.base_P)

    elif mode == "out_of_distribution":
        for _ in range(n_matrices):
            mat = Matrix(m=M, n=N, min=MIN_VAL, max=MAX_VAL, epsilon=0.0)
            mat.generateMatrix(mode="uniform")
            matrices.append(mat.base_P)

    return matrices


# ---------------------------------------------------------------------------
# Main experiment loop-
# ---------------------------------------------------------------------------

def run_experiment(n_matrices, seed, model_path=None, base_matrix=None):
    np.random.seed(seed)

    if model_path is None:
        model_path = MODEL_NAME_TEMPLATE.format(
            steps=TIMESTEPS, m=M, n=N,
            min=MIN_VAL, max=MAX_VAL, eps=EPSILON,
        )
    # SB3 appends .zip automatically, so strip it if present in template
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    strategies = list(PIVOT_MAP_TEST.values())
    all_methods = strategies + ["rl_agent"]

    results = {}
    for mode in ["in_distribution", "out_of_distribution"]:
        print(f"\n{'=' * 60}")
        print(f"  {mode.replace('_', ' ').upper()} ({n_matrices} matrices)")
        print(f"{'=' * 60}")

        matrices = generate_test_matrices(n_matrices, mode=mode, base_matrix=base_matrix)
        rows = []

        for i, matrix_P in enumerate(matrices):
            tableau = prepare_tableau(matrix_P)
            if tableau is None:
                print(f"  Matrix {i}: failed to build tableau, skipping")
                continue

            T, basis = tableau
            row = {"matrix_idx": i}

            for strategy in strategies:
                row[strategy] = run_fixed_strategy(T, basis, strategy)

            row["rl_agent"] = run_rl_agent(T, basis, model)
            rows.append(row)

            if (i + 1) % 50 == 0:
                print(f"  Completed {i + 1}/{n_matrices}")

        results[mode] = rows
        print(f"  Done: {len(rows)} matrices solved")

    return results, all_methods


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_metric(rows, n_total, all_methods, strategies, metric_key, metric_label):
    """Print iteration-style stats for a single metric.

    metric_key="nit" reports raw pivot count.
    metric_key="weighted_cost" reports sum of per-pivot wallclock-cost weights
    (calibrated by benchmark_pivot_cost.py and stored in STEP_PENALTY_WEIGHTS).
    """
    print(f"\n  --- {metric_label} ---")

    # --- Stats table ---
    iters = defaultdict(list)
    for row in rows:
        for method in all_methods:
            if row[method]["status"] == "optimal":
                iters[method].append(row[method][metric_key])

    print(f"    {'Method':25s} {'Mean':>9s} {'Median':>9s} "
          f"{'Min':>8s} {'Max':>8s} {'N':>5s}")
    for method in all_methods:
        if iters[method]:
            arr = np.array(iters[method], dtype=float)
            label = "RL Agent" if method == "rl_agent" else method
            print(f"    {label:25s} {arr.mean():9.2f} {float(np.median(arr)):9.2f} "
                  f"{arr.min():8.2f} {arr.max():8.2f} {len(arr):5d}")

    # --- Head-to-head: RL vs each heuristic ---
    print(f"\n    Head-to-head: RL Agent vs each heuristic ({metric_label})")
    print(f"    {'Heuristic':25s} {'RL wins':>8s} {'Ties':>8s} "
          f"{'RL loses':>9s} {'N/A':>5s}")

    for strategy in strategies:
        wins, ties, losses, na = 0, 0, 0, 0
        paired_rl, paired_heur = [], []

        for row in rows:
            rl_ok = row["rl_agent"]["status"] == "optimal"
            h_ok = row[strategy]["status"] == "optimal"

            if rl_ok and h_ok:
                rl_n = row["rl_agent"][metric_key]
                h_n = row[strategy][metric_key]
                if rl_n < h_n:
                    wins += 1
                elif rl_n == h_n:
                    ties += 1
                else:
                    losses += 1
                paired_rl.append(rl_n)
                paired_heur.append(h_n)
            elif rl_ok and not h_ok:
                wins += 1
            elif not rl_ok and h_ok:
                losses += 1
            else:
                na += 1

        print(f"    {strategy:25s} {wins:8d} {ties:8d} {losses:9d} {na:5d}")

        if len(paired_rl) >= 10:
            diffs = np.array(paired_rl, dtype=float) - np.array(paired_heur, dtype=float)
            nonzero = diffs[diffs != 0]
            if len(nonzero) >= 10:
                median_diff = float(np.median(nonzero))
                alt = "less" if median_diff < 0 else "greater"
                stat, p = wilcoxon(nonzero, alternative=alt)
                direction = "RL better" if median_diff < 0 else "Heuristic better"
                print(f"      Wilcoxon p={p:.6f} ({direction})")

    # --- Reduction: RL vs each heuristic ---
    print(f"\n    {metric_label} reduction: RL Agent vs each heuristic "
          f"(paired both-converged)")
    print(f"    {'Heuristic':25s} {'Mean %':>8s} {'Median %':>9s} "
          f"{'Mean abs':>11s} {'P25 %':>7s} {'P75 %':>7s} {'N':>5s}")

    for strategy in strategies:
        pct_reductions = []
        abs_reductions = []
        for row in rows:
            rl_ok = row["rl_agent"]["status"] == "optimal"
            h_ok = row[strategy]["status"] == "optimal"
            if rl_ok and h_ok:
                rl_n = row["rl_agent"][metric_key]
                h_n = row[strategy][metric_key]
                if h_n > 0:
                    pct_reductions.append((h_n - rl_n) / h_n * 100)
                abs_reductions.append(h_n - rl_n)

        if pct_reductions:
            arr = np.array(pct_reductions)
            abs_arr = np.array(abs_reductions, dtype=float)
            print(f"    {strategy:25s} {arr.mean():+7.1f}% {float(np.median(arr)):+8.1f}% "
                  f"{abs_arr.mean():+10.2f} {np.percentile(arr, 25):+6.1f}% "
                  f"{np.percentile(arr, 75):+6.1f}% {len(arr):5d}")
        else:
            print(f"    {strategy:25s}  {'(no paired data)':>40s}")

    # --- Reduction: RL vs best-per-instance heuristic ---
    print(f"\n    {metric_label} reduction: RL Agent vs best heuristic per instance")

    pct_vs_best = []
    abs_vs_best = []
    for row in rows:
        rl_ok = row["rl_agent"]["status"] == "optimal"
        best_n = None
        for strategy in strategies:
            if row[strategy]["status"] == "optimal":
                cand = row[strategy][metric_key]
                if best_n is None or cand < best_n:
                    best_n = cand
        if rl_ok and best_n is not None and best_n > 0:
            rl_n = row["rl_agent"][metric_key]
            pct_vs_best.append((best_n - rl_n) / best_n * 100)
            abs_vs_best.append(best_n - rl_n)

    if pct_vs_best:
        arr = np.array(pct_vs_best)
        abs_arr = np.array(abs_vs_best, dtype=float)
        print(f"      Mean reduction:   {arr.mean():+.1f}%  ({abs_arr.mean():+.2f})")
        print(f"      Median reduction: {float(np.median(arr)):+.1f}%  "
              f"({float(np.median(abs_arr)):+.2f})")
        print(f"      P25 / P75:        {np.percentile(arr, 25):+.1f}% / "
              f"{np.percentile(arr, 75):+.1f}%")
        print(f"      Paired instances: {len(arr)}")
    else:
        print(f"      (no paired data)")

    # --- Win/Tie/Loss vs best per instance ---
    print(f"\n    Win/Tie/Loss: RL Agent vs best heuristic per instance ({metric_label}):")
    wins, ties, losses, na = 0, 0, 0, 0
    for row in rows:
        rl_ok = row["rl_agent"]["status"] == "optimal"
        best_n = None
        for strategy in strategies:
            if row[strategy]["status"] == "optimal":
                cand = row[strategy][metric_key]
                if best_n is None or cand < best_n:
                    best_n = cand
        if rl_ok and best_n is not None:
            rl_n = row["rl_agent"][metric_key]
            if rl_n < best_n: wins += 1
            elif rl_n == best_n: ties += 1
            else: losses += 1
        elif rl_ok and best_n is None:
            wins += 1
        elif not rl_ok and best_n is not None:
            losses += 1
        else:
            na += 1

    total_decided = wins + ties + losses
    print(f"      Wins: {wins:4d}  Ties: {ties:4d}  "
          f"Losses: {losses:4d}  N/A: {na:4d}")
    if total_decided > 0:
        print(f"      Win rate: {wins / total_decided * 100:.1f}%  "
              f"Win+Tie rate: {(wins + ties) / total_decided * 100:.1f}%")


def analyze_results(results, all_methods):
    strategies = [m for m in all_methods if m != "rl_agent"]

    for mode, rows in results.items():
        if not rows:
            continue

        n_total = len(rows)
        print(f"\n{'=' * 60}")
        print(f"  RESULTS: {mode.replace('_', ' ').upper()} ({n_total} matrices)")
        print(f"{'=' * 60}")

        # --- Convergence rates (mode-level, metric-independent) ---
        convergence = defaultdict(int)
        for row in rows:
            for method in all_methods:
                if row[method]["status"] == "optimal":
                    convergence[method] += 1

        print(f"\n  Convergence rates:")
        for method in all_methods:
            rate = convergence[method] / n_total * 100
            label = "RL Agent" if method == "rl_agent" else method
            print(f"    {label:25s}: {convergence[method]:4d}/{n_total} ({rate:.1f}%)")

        # --- Per-metric analysis (run twice) ---
        analyze_metric(rows, n_total, all_methods, strategies,
                       metric_key="nit", metric_label="Pivot count")
        analyze_metric(rows, n_total, all_methods, strategies,
                       metric_key="weighted_cost",
                       metric_label="Weighted cost (sum of STEP_PENALTY_WEIGHTS)")

        # --- Game value consistency check (mode-level) ---
        n_inconsistent = 0
        for row in rows:
            objectives = []
            for method in all_methods:
                if row[method]["status"] == "optimal":
                    objectives.append(row[method]["objective"])
            if objectives and (max(objectives) - min(objectives)) > 1e-4:
                n_inconsistent += 1

        if n_inconsistent > 0:
            print(f"\n  WARNING: {n_inconsistent} matrices had inconsistent "
                  f"game values across methods (diff > 1e-4)")
        else:
            print(f"\n  Game value consistency: all methods agree within 1e-4")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():

    n_matrices = 200  # Number of test matrices per test set
    seed = 42  # Random seed for reproducibility
    save = None  # Path to save raw results (e.g., "results.json")
    base_matrix = None  # Path to base_matrix.py, or None to use current base_matrix.py
    model = MODEL_NAME_TEMPLATE.format(steps=TIMESTEPS, m=M, n=N, min=MIN_VAL, max=MAX_VAL, eps=EPSILON)
    # Standalone smoke entry only. The canonical 4-model evaluation is
    # results/normal_form/eval_normal_form.py; this points at one shipped model.
    model = "results/normal_form/models/ppo_simplex_random_20000000_matrix40x40_min-1_max1_epsilon0.001_dict_weighted.zip"

    # Load custom base matrix if specified
    # base_matrix = None
    if base_matrix:
        base_matrix = load_base_matrix_from_file(base_matrix)
        print(f"Using custom base matrix from: {base_matrix}")

    print(f"Matrix size: {M}x{N}")
    print(f"Test matrices per set: {n_matrices}")
    print(f"Seed: {seed}")
    print(f"Training EPSILON:      {EPSILON}")
    print(f"Test     EPSILON:      {TEST_EPSILON}"
          f"{'  (matches training)' if TEST_EPSILON == EPSILON else '  (OOD perturbation magnitude)'}")
    print(f"RL training strategies: {list(PIVOT_MAP.values())}")
    print(f"All tested strategies:  {list(PIVOT_MAP_TEST.values())}")
    print(f"Per-pivot weights for weighted-cost metric (STEP_PENALTY_WEIGHTS):")
    for k, v in STEP_PENALTY_WEIGHTS.items():
        print(f"    {k:<25s} {v}")

    results, all_methods = run_experiment(n_matrices, seed,
                                          model_path=model,
                                          base_matrix=base_matrix)
    analyze_results(results, all_methods)

    if save:
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(save, "w") as f:
            json.dump(results, f, default=convert, indent=2)
        print(f"\nRaw results saved to {save}")


if __name__ == "__main__":
    main()