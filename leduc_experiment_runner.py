"""
Leduc experiment runner: evaluate a trained PPO model against fixed pivot
strategies on sequence-form LPs built from Leduc poker with perturbed deck
priors. Mirrors the structure of experiment.py (matrix-mode runner).

All settings are read from config.py (no CLI flags). Edit the constants in
`main()` below to control which model to load and how many test LPs to sample.
"""

import json
import numpy as np
from collections import defaultdict

import pyspiel
from stable_baselines3 import PPO
from scipy.stats import wilcoxon

from simplex_solver import (
    phase1solver, first_to_second,
    _pivot_col_heuristics, _pivot_row, _apply_pivot,
)
from envs import SecondPhasePivotingEnv
from _linprog_utils import _parse_linprog, _get_Abc, _LPProblem
from leduc_experiment import build_sequence_form_matrices, sample_rank_weights
from config import (
    LEDUC_GAME, LEDUC_ALPHA, LEDUC_NUM_RANKS,
    TIMESTEPS, PIVOT_MAP, PIVOT_MAP_TEST,
    STEP_PENALTY_WEIGHTS, MODEL_RUN_TAG,
)


MAXITER = 50_000
TOL = 1e-7


# ---------------------------------------------------------------------------
# Tableau construction (mirrors LeducEnv._init_env)
# ---------------------------------------------------------------------------

def build_leduc_tableau(game, rank_weights):
    """Build a Phase 2 tableau for one Leduc LP. Returns (T, basis, phase1_nit)
    or None on failure."""
    A, E, e, F, f, *_ = build_sequence_form_matrices(game, rank_weights=rank_weights)

    n_x, n_y, n_p = A.shape[0], A.shape[1], F.shape[0]
    c = np.concatenate([np.zeros(n_x), -f])
    A_eq = np.hstack([E, np.zeros((E.shape[0], n_p))])
    b_eq = e
    A_ub = np.hstack([-A.T, F.T])
    b_ub = np.zeros(n_y)
    bounds = [(0, None)] * n_x + [(None, None)] * n_p

    lp = _LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0=None, integrality=None)
    lp, solver_options = _parse_linprog(lp, None, meth='simplex')
    A_std, b_std, c_std, c0, x0 = _get_Abc(lp, 0)

    n_rows, n_cols = A_std.shape
    neg = b_std < 0
    A_std[neg] *= -1
    b_std[neg] *= -1

    av = np.arange(n_rows) + n_cols
    basis = av.copy()
    row_constraints = np.hstack((A_std, np.eye(n_rows), b_std[:, np.newaxis]))
    row_objective = np.hstack((c_std, np.zeros(n_rows), c0))
    row_pseudo_objective = -row_constraints.sum(axis=0)
    row_pseudo_objective[av] = 0
    T = np.vstack((row_constraints, row_objective, row_pseudo_objective))

    nit, status = phase1solver(T, basis, maxiter=MAXITER)
    if status != 0:
        return None
    phase1_nit = nit

    res = first_to_second(T, basis, av)
    if res is None:
        return None
    T, basis = res

    # Mirror SecondPhasePivotingEnv.remove_artificial to clean up basis
    for pivrow in [row for row in range(basis.size) if basis[row] > T.shape[1] - 2]:
        non_zero_cols = [col for col in range(T.shape[1] - 1)
                         if abs(T[pivrow, col]) > TOL]
        if non_zero_cols:
            _apply_pivot(T, basis, pivrow, non_zero_cols[0], TOL)

    return T, basis, phase1_nit


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_fixed_strategy(T, basis, strategy):
    """Run a fixed strategy. Returns status, nit, weighted_cost, objective.
    weighted_cost = nit * STEP_PENALTY_WEIGHTS[strategy] (constant per pivot)."""
    T = T.copy()
    basis = basis.copy()
    nit = 0
    seen = {tuple(int(i) for i in basis)}
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

        if not np.all(np.isfinite(T)):
            return {"status": "numerical_error", "nit": nit,
                    "weighted_cost": nit * w, "objective": 0.0}

        key = tuple(int(i) for i in basis)
        if key in seen:
            return {"status": "loop", "nit": nit, "weighted_cost": nit * w,
                    "objective": float(T[-1, -1])}
        seen.add(key)

    return {"status": "maxiter", "nit": nit, "weighted_cost": nit * w,
            "objective": float(T[-1, -1])}


def run_rl_agent(T, basis, model, use_compact=True):
    """Run the trained RL agent. Returns status, nit, weighted_cost, objective.
    weighted_cost = sum of STEP_PENALTY_WEIGHTS[PIVOT_MAP[action]] over the
    rules the agent actually picked. Leduc models use compact obs by default."""
    from wrappers import CompactObsWrapper
    base_env = SecondPhasePivotingEnv(T.copy(), basis.copy())
    env = CompactObsWrapper(base_env) if use_compact else base_env
    obs, _ = env.reset()
    done = False
    truncated = False
    info = {}
    weighted_cost = 0.0
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        a = int(action)
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
# Test LP sampling
# ---------------------------------------------------------------------------

def sample_leduc_tableaus(game, n_matrices, alpha, rng, uniform=False):
    """Sample n_matrices Leduc LP tableaus with the given alpha (or a uniform deck
    if uniform=True, in which case alpha is ignored)."""
    tableaus = []
    attempts = 0
    max_attempts = n_matrices * 10
    while len(tableaus) < n_matrices and attempts < max_attempts:
        attempts += 1
        if uniform:
            weights = np.full(LEDUC_NUM_RANKS, 1.0 / LEDUC_NUM_RANKS)
        else:
            weights = sample_rank_weights(alpha=alpha, num_ranks=LEDUC_NUM_RANKS, rng=rng)
        try:
            tab = build_leduc_tableau(game, weights)
        except Exception as exc:
            print(f"  [sample] build failure: {exc}")
            continue
        if tab is None:
            continue
        T, basis, phase1_nit = tab
        tableaus.append({"T": T, "basis": basis, "weights": weights,
                         "phase1_nit": phase1_nit})
        if uniform and len(tableaus) == 1:
            # Uniform deck is deterministic — one instance is enough.
            break
    if len(tableaus) < n_matrices:
        print(f"  [sample] only built {len(tableaus)}/{n_matrices} (after {attempts} attempts)")
    return tableaus


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(model_path, n_matrices, alpha_in, alpha_out, seed):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    print(f"Loading model: {model_path}")
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]
    model = PPO.load(model_path)
    # Detect obs type from the loaded model so we wrap with CompactObsWrapper
    # only when the policy expects a flat Box (vs the Dict tableau).
    use_compact = (type(model.observation_space).__name__ == "Box")
    print(f"  Model obs space: {type(model.observation_space).__name__} "
          f"-> use_compact={use_compact}")

    game = pyspiel.load_game(LEDUC_GAME)

    strategies = list(PIVOT_MAP_TEST.values())
    all_methods = strategies + ["rl_agent"]

    test_sets = {
        "in_distribution":     {"alpha": alpha_in,  "n": n_matrices, "uniform": False},
        "out_of_distribution": {"alpha": alpha_out, "n": n_matrices, "uniform": False},
        "uniform_deck":        {"alpha": None,      "n": 1,          "uniform": True},
    }

    results = {}
    for mode, cfg in test_sets.items():
        print(f"\n{'=' * 60}")
        label = f" alpha={cfg['alpha']}" if not cfg["uniform"] else " (fair deck)"
        print(f"  {mode.upper()}{label} | target={cfg['n']} LP(s)")
        print(f"{'=' * 60}")

        tableaus = sample_leduc_tableaus(game, cfg["n"], cfg["alpha"], rng,
                                         uniform=cfg["uniform"])
        rows = []
        for i, tab in enumerate(tableaus):
            T, basis = tab["T"], tab["basis"]
            row = {"matrix_idx": i,
                   "weights": tab["weights"].tolist(),
                   "phase1_nit": tab["phase1_nit"]}

            for strategy in strategies:
                row[strategy] = run_fixed_strategy(T, basis, strategy)
            row["rl_agent"] = run_rl_agent(T, basis, model, use_compact=use_compact)
            rows.append(row)

            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{len(tableaus)}")

        results[mode] = rows
        print(f"  Done: {len(rows)} LP(s) solved")

    return results, all_methods


# ---------------------------------------------------------------------------
# Analysis (mirrors experiment.py)
# ---------------------------------------------------------------------------

def analyze_metric(rows, n_total, all_methods, strategies, metric_key, metric_label):
    """Print stats / head-to-heads / reductions for a single metric.
    metric_key='nit' = raw pivot count.
    metric_key='weighted_cost' = sum of per-pivot wallclock weights."""
    print(f"\n  --- {metric_label} ---")

    iters = defaultdict(list)
    for row in rows:
        for m in all_methods:
            if row[m]["status"] == "optimal":
                iters[m].append(row[m][metric_key])

    print(f"    {'Method':25s} {'Mean':>9} {'Median':>9} {'Min':>8} {'Max':>8} {'N':>5}")
    for m in all_methods:
        if iters[m]:
            arr = np.array(iters[m], dtype=float)
            label = "RL Agent" if m == "rl_agent" else m
            print(f"    {label:25s} {arr.mean():9.2f} {float(np.median(arr)):9.2f} "
                  f"{arr.min():8.2f} {arr.max():8.2f} {len(arr):5d}")

    if n_total < 2:
        return

    print(f"\n    Head-to-head: RL Agent vs each heuristic ({metric_label})")
    print(f"    {'Heuristic':25s} {'RL wins':>8} {'Ties':>5} "
          f"{'RL loses':>9} {'N/A':>5}")
    for s in strategies:
        wins, ties, losses, na = 0, 0, 0, 0
        paired_rl, paired_h = [], []
        for row in rows:
            rl_ok = row["rl_agent"]["status"] == "optimal"
            h_ok = row[s]["status"] == "optimal"
            if rl_ok and h_ok:
                rl_n, h_n = row["rl_agent"][metric_key], row[s][metric_key]
                if rl_n < h_n: wins += 1
                elif rl_n == h_n: ties += 1
                else: losses += 1
                paired_rl.append(rl_n); paired_h.append(h_n)
            elif rl_ok and not h_ok: wins += 1
            elif not rl_ok and h_ok: losses += 1
            else: na += 1
        print(f"    {s:25s} {wins:8d} {ties:5d} {losses:9d} {na:5d}")
        if len(paired_rl) >= 10:
            diffs = np.array(paired_rl, dtype=float) - np.array(paired_h, dtype=float)
            nz = diffs[diffs != 0]
            if len(nz) >= 10:
                med = float(np.median(nz))
                alt = "less" if med < 0 else "greater"
                stat, p = wilcoxon(nz, alternative=alt)
                direction = "RL better" if med < 0 else "Heuristic better"
                print(f"      Wilcoxon p={p:.6f} ({direction})")

    print(f"\n    {metric_label} reduction: RL vs each heuristic (paired, both converged)")
    print(f"    {'Heuristic':25s} {'Mean %':>8} {'Median %':>9} "
          f"{'Mean abs':>11} {'N':>5}")
    for s in strategies:
        pct, absr = [], []
        for row in rows:
            if row["rl_agent"]["status"] == "optimal" and row[s]["status"] == "optimal":
                rl_n, h_n = row["rl_agent"][metric_key], row[s][metric_key]
                if h_n > 0:
                    pct.append((h_n - rl_n) / h_n * 100)
                absr.append(h_n - rl_n)
        if pct:
            arr, abs_arr = np.array(pct), np.array(absr, dtype=float)
            print(f"    {s:25s} {arr.mean():+7.1f}% {float(np.median(arr)):+8.1f}% "
                  f"{abs_arr.mean():+10.2f} {len(arr):5d}")

    # vs best-per-instance
    pct_best, abs_best = [], []
    wins, ties, losses, na = 0, 0, 0, 0
    for row in rows:
        rl_ok = row["rl_agent"]["status"] == "optimal"
        best = None
        for s in strategies:
            if row[s]["status"] == "optimal":
                cand = row[s][metric_key]
                if best is None or cand < best:
                    best = cand
        if rl_ok and best is not None:
            rl_n = row["rl_agent"][metric_key]
            if best > 0:
                pct_best.append((best - rl_n) / best * 100)
            abs_best.append(best - rl_n)
            if rl_n < best: wins += 1
            elif rl_n == best: ties += 1
            else: losses += 1
        elif rl_ok and best is None: wins += 1
        elif not rl_ok and best is not None: losses += 1
        else: na += 1

    print(f"\n    RL vs best heuristic per instance ({metric_label}): "
          f"W {wins}  T {ties}  L {losses}  N/A {na}")
    if pct_best:
        arr = np.array(pct_best)
        abs_arr = np.array(abs_best, dtype=float)
        print(f"      Mean reduction: {arr.mean():+.1f}%  ({abs_arr.mean():+.2f})   "
              f"Median: {float(np.median(arr)):+.1f}%   N={len(arr)}")


def analyze_results(results, all_methods):
    strategies = [m for m in all_methods if m != "rl_agent"]

    for mode, rows in results.items():
        if not rows:
            continue

        n_total = len(rows)
        print(f"\n{'=' * 60}")
        print(f"  RESULTS: {mode.upper()} ({n_total} LP(s))")
        print(f"{'=' * 60}")

        # Convergence (mode-level, metric-independent)
        conv = defaultdict(int)
        for row in rows:
            for m in all_methods:
                if row[m]["status"] == "optimal":
                    conv[m] += 1
        print(f"\n  Convergence rates:")
        for m in all_methods:
            rate = conv[m] / n_total * 100
            label = "RL Agent" if m == "rl_agent" else m
            print(f"    {label:25s}: {conv[m]:4d}/{n_total} ({rate:.1f}%)")

        # Per-metric analysis (run twice)
        analyze_metric(rows, n_total, all_methods, strategies,
                       metric_key="nit", metric_label="Pivot count")
        analyze_metric(rows, n_total, all_methods, strategies,
                       metric_key="weighted_cost",
                       metric_label="Weighted cost (sum of STEP_PENALTY_WEIGHTS)")

        # Game-value consistency
        n_inconsistent = 0
        for row in rows:
            objs = [row[m]["objective"] for m in all_methods if row[m]["status"] == "optimal"]
            if objs and (max(objs) - min(objs)) > 1e-3:
                n_inconsistent += 1
        if n_inconsistent:
            print(f"\n  WARNING: {n_inconsistent} LPs had inconsistent game values (diff > 1e-3)")
        else:
            print(f"\n  Game value consistency: all methods agree within 1e-3")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    # Edit these to drive the experiment.
    n_matrices = 50                            # LPs per test set (in/out distribution)
    seed = 42
    alpha_in = LEDUC_ALPHA                     # training distribution
    alpha_out = max(1.0, LEDUC_ALPHA / 10.0)   # broader perturbation for OOD
    # Default model path uses the active MODEL_RUN_TAG from config. Override
    # `model_path` below to point at a different run.
    model_path = f"models/ppo_leduc_{TIMESTEPS}_alpha{LEDUC_ALPHA}_{MODEL_RUN_TAG}.zip"
    save = None                                # e.g. "leduc_results.json"

    print(f"Game:        {LEDUC_GAME}")
    print(f"alpha_in:    {alpha_in}  (training distribution)")
    print(f"alpha_out:   {alpha_out}  (out-of-distribution)")
    print(f"n_matrices:  {n_matrices} per non-uniform test set")
    print(f"seed:        {seed}")
    print(f"RL training strategies: {list(PIVOT_MAP.values())}")
    print(f"All tested strategies:  {list(PIVOT_MAP_TEST.values())}")
    print(f"Per-pivot weights for weighted-cost metric (STEP_PENALTY_WEIGHTS):")
    for k, v in STEP_PENALTY_WEIGHTS.items():
        print(f"    {k:<25s} {v}")

    results, all_methods = run_experiment(
        model_path=model_path,
        n_matrices=n_matrices,
        alpha_in=alpha_in,
        alpha_out=alpha_out,
        seed=seed,
    )
    analyze_results(results, all_methods)

    if save:
        def convert(o):
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return o
        with open(save, "w") as f:
            json.dump(results, f, default=convert, indent=2)
        print(f"\nRaw results saved to {save}")


if __name__ == "__main__":
    main()
