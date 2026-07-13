"""Evaluate the RL full-pivot agent against fixed heuristics on Leduc LPs.

For each of N sampled Leduc instances:
  - Build the phase-1 tableau from Dirichlet-sampled deck weights.
  - Run each fixed strategy end-to-end (phase 1 + phase 2). Record total pivots.
  - Run the RL agent end-to-end (deterministic policy). Record total pivots.
  - Compare.

The RL agent uses `LeducFullPivotEnv` with `use_baseline=False` (no extra work
per reset) for pure evaluation. Fixed methods run on COPIES of the env's
starting tableau, so all methods solve the same LP per test.
"""
import argparse
import warnings
import numpy as np
from stable_baselines3 import PPO

from envs import LeducFullPivotEnv
from simplex_solver import (
    first_to_second, _pivot_col_heuristics, _pivot_row, _apply_pivot,
)
from config import LEDUC_GAME, LEDUC_ALPHA, LEDUC_NUM_RANKS, TIMESTEPS, PIVOT_MAP


def run_fixed_full(T, basis, av, strategy, maxiter=50000, tol=1e-7):
    """Run phase 1 + phase 2 end-to-end with a fixed pivot strategy.

    Returns (status, nit1, nit2). status in {"optimal","phase1_fail","phase2_unbounded","maxiter"}.
    """
    use_bland = (strategy == 'blands_rule')
    # Phase 1 loop (uses pseudo-obj row T[-1])
    n1 = 0
    while n1 < maxiter:
        ok, col = _pivot_col_heuristics(T, strategy=strategy, tol=tol)
        if not ok:
            break
        ok, row = _pivot_row(T, basis, col, phase=1, tol=tol, bland=use_bland)
        if not ok:
            return "phase1_unbounded", n1, 0
        _apply_pivot(T, basis, row, col, tol=tol)
        if not np.isfinite(T[-1, -1]) or np.any(~np.isfinite(T)):
            return "phase1_numerical", n1, 0
        n1 += 1
    if n1 >= maxiter:
        return "phase1_maxiter", n1, 0

    # Transition
    res = first_to_second(T, basis, av)
    if res is None:
        return "phase1->2_fail", n1, 0
    T, basis = res

    # Phase 2 loop
    n2 = 0
    while n2 < maxiter:
        ok, col = _pivot_col_heuristics(T, strategy=strategy, tol=tol)
        if not ok:
            return "optimal", n1, n2
        ok, row = _pivot_row(T, basis, col, phase=2, tol=tol, bland=use_bland)
        if not ok:
            return "phase2_unbounded", n1, n2
        _apply_pivot(T, basis, row, col, tol=tol)
        if not np.isfinite(T[-1, -1]) or np.any(~np.isfinite(T)):
            return "phase2_numerical", n1, n2
        n2 += 1
    return "phase2_maxiter", n1, n2


def run_rl_agent(env, model):
    """Run RL agent end-to-end. Returns (status, nit1, nit2)."""
    obs, _ = env.reset()
    done = truncated = False
    info = {}
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, info = env.step(int(action))
    return info.get("status", "unknown"), info.get("phase1_nit", 0), info.get("phase2_nit", 0)


def evaluate(n_matrices=50, seed=42, timesteps=TIMESTEPS, maxiter=50_000):
    warnings.filterwarnings("ignore")

    model_path = f"models/ppo_leduc_{timesteps}_alpha{LEDUC_ALPHA}.zip"
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    env = LeducFullPivotEnv(
        game_name=LEDUC_GAME, alpha=LEDUC_ALPHA, num_ranks=LEDUC_NUM_RANKS,
        use_baseline=False, seed=seed, maxiter=maxiter,
    )
    strategies = ["largest_coefficient", "largest_increase", "steepest_edge"]

    results = {s: [] for s in strategies}
    results["rl_agent"] = []
    statuses = {s: [] for s in results}

    for i in range(n_matrices):
        # Reset env → samples new weights, builds initial phase-1 tableau
        # We run RL FIRST so env keeps its clean starting tableau as reference
        # but we need to snapshot it before RL mutates.
        obs, _ = env.reset()
        T0 = env.T.copy()
        basis0 = env.basis.copy()
        av0 = env.av.copy()

        # Fixed heuristics first (on copies)
        for strat in strategies:
            status, n1, n2 = run_fixed_full(T0.copy(), basis0.copy(), av0.copy(), strat, maxiter=env.maxiter)
            results[strat].append((n1 + n2) if status == "optimal" else None)
            statuses[strat].append(status)

        # RL agent — replay from the same env (already reset)
        done = truncated = False
        info = {}
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(int(action))
        if info.get("status") == "optimal":
            results["rl_agent"].append(int(info.get("phase1_nit", 0)) + int(info.get("phase2_nit", 0)))
        else:
            results["rl_agent"].append(None)
        statuses["rl_agent"].append(info.get("status", "unknown"))

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{n_matrices}")

    print()
    print("=" * 70)
    print(f"RESULTS on {n_matrices} Leduc LPs (alpha={LEDUC_ALPHA})")
    print("=" * 70)
    print(f"{'Method':25s} {'Conv.':>6s} {'Mean':>8s} {'Median':>8s} {'Min':>5s} {'Max':>5s}")
    for method in list(results.keys()):
        converged = [x for x in results[method] if x is not None]
        conv_rate = len(converged) / n_matrices
        if converged:
            arr = np.array(converged)
            print(f"{method:25s} {conv_rate*100:5.1f}% {arr.mean():8.1f} {np.median(arr):8.1f} "
                  f"{arr.min():5d} {arr.max():5d}")
        else:
            print(f"{method:25s} 0 converged")

    # Head-to-head: RL vs each
    print()
    print("Head-to-head: RL Agent vs each heuristic (paired, both converged)")
    print(f"{'Heuristic':25s} {'Wins':>6s} {'Ties':>6s} {'Losses':>7s}  Mean Δ%")
    rl = results["rl_agent"]
    for strat in strategies:
        fs = results[strat]
        wins = ties = losses = 0
        diffs = []
        for r, f in zip(rl, fs):
            if r is None or f is None:
                continue
            if r < f:
                wins += 1
            elif r > f:
                losses += 1
            else:
                ties += 1
            diffs.append((f - r) / f * 100)
        mean_diff = float(np.mean(diffs)) if diffs else 0.0
        print(f"{strat:25s} {wins:>6d} {ties:>6d} {losses:>7d}  {mean_diff:+.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="Number of Leduc test instances")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timesteps", type=int, default=TIMESTEPS, help="Timesteps tag on the saved model")
    ap.add_argument("--maxiter", type=int, default=50_000, help="Per-phase iter cap")
    args = ap.parse_args()
    evaluate(n_matrices=args.n, seed=args.seed, timesteps=args.timesteps, maxiter=args.maxiter)


if __name__ == "__main__":
    main()
