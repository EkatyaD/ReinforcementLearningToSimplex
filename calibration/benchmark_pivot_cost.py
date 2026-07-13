"""Benchmark per-strategy column-selection wallclock for the simplex pivot rules.

Goal: derive empirical weights for the per-step penalty so the RL agent's
return reflects the real differential cost of picking each pivot strategy.

For each LP size, we:
  1. Build several Phase-2 tableaus from perturbed matrices.
  2. Walk them with steepest_edge, snapshotting the tableau every K steps.
  3. On each snapshot, time every strategy with `repeats` calls (after warm-up).

Same-snapshot comparison isolates per-call cost from trajectory differences.
"""

import argparse
import time
import numpy as np

from matrix import Matrix
from simplex_solver import (
    change_to_zero_sum, phase1solver, first_to_second,
    _pivot_col_heuristics, _pivot_row, _apply_pivot,
)
from config import MIN_VAL, MAX_VAL, EPSILON


TOL = 1e-9
STRATEGIES = [
    "largest_coefficient",
    "largest_increase",
    "steepest_edge",
    "random_edge",
    "blands_rule",
]


def prepare_phase2(P):
    T, basis, av = change_to_zero_sum(P)
    _, status = phase1solver(T, basis)
    if status != 0:
        return None
    res = first_to_second(T, basis, av)
    if res is None:
        return None
    T, basis = res
    for pivrow in [r for r in range(basis.size) if basis[r] > T.shape[1] - 2]:
        nzc = [c for c in range(T.shape[1] - 1) if abs(T[pivrow, c]) > TOL]
        if nzc:
            _apply_pivot(T, basis, pivrow, nzc[0], TOL)
    return T, basis


def collect_snapshots(T, basis, every=4, max_snapshots=8, maxiter=2000):
    snaps = []
    nit = 0
    while nit < maxiter and len(snaps) < max_snapshots:
        if nit % every == 0:
            snaps.append(T.copy())
        ok, col = _pivot_col_heuristics(T, strategy="steepest_edge", tol=TOL)
        if not ok:
            break
        ok, row = _pivot_row(T, basis, col, phase=2, tol=TOL)
        if not ok:
            break
        _apply_pivot(T, basis, row, col, tol=TOL)
        nit += 1
    return snaps


def time_strategy(T, strategy, repeats, warmup=10):
    for _ in range(warmup):
        _pivot_col_heuristics(T, strategy=strategy, tol=TOL)
    t0 = time.perf_counter()
    for _ in range(repeats):
        _pivot_col_heuristics(T, strategy=strategy, tol=TOL)
    return (time.perf_counter() - t0) / repeats * 1e6  # µs/call


def make_base(m, n, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(low=-1, high=2, size=(m, n)).astype(np.float64)


def run_size(m, n, num_matrices, snaps_per_matrix, repeats, seed):
    base = make_base(m, n, seed)
    matrix = Matrix(m=m, n=n, min=MIN_VAL, max=MAX_VAL, epsilon=EPSILON, base_P=base)

    snapshots = []
    for k in range(num_matrices):
        P = matrix.generate_perturbed_matrix().base_P
        prep = prepare_phase2(P)
        if prep is None:
            continue
        T, basis = prep
        snapshots.extend(collect_snapshots(T, basis, max_snapshots=snaps_per_matrix))

    per_strategy = {s: [] for s in STRATEGIES}
    for T in snapshots:
        for s in STRATEGIES:
            per_strategy[s].append(time_strategy(T, s, repeats=repeats))
    return per_strategy, len(snapshots)


def report(size, per_strategy, num_snapshots):
    means = {s: float(np.mean(per_strategy[s])) for s in STRATEGIES}
    ref = min(means.values())
    print(f"\nSize {size[0]}x{size[1]}  ({num_snapshots} snapshots, "
          f"reference = {min(means, key=means.get)})")
    print(f"{'strategy':<22} {'µs/call':>10} {'std':>10} {'norm':>8}")
    for s in STRATEGIES:
        std = float(np.std(per_strategy[s]))
        print(f"{s:<22} {means[s]:>10.3f} {std:>10.3f} {means[s]/ref:>8.3f}")
    return means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, default="20,40,60,80",
                        help="Comma-separated list of square sizes")
    parser.add_argument("--num-matrices", type=int, default=4)
    parser.add_argument("--snaps-per-matrix", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sizes = [(int(s), int(s)) for s in args.sizes.split(",")]
    print(f"Repeats={args.repeats}, matrices/size={args.num_matrices}, "
          f"snaps/matrix={args.snaps_per_matrix}, seed={args.seed}")

    per_size_means = {}
    for m, n in sizes:
        per, ns = run_size(m, n, args.num_matrices, args.snaps_per_matrix,
                           args.repeats, args.seed)
        per_size_means[(m, n)] = report((m, n), per, ns)

    # Cross-size scaling table: how each strategy's cost grows with size,
    # relative to its own value at the smallest size.
    print("\nScaling vs smallest size (per-strategy growth factor):")
    smallest = sizes[0]
    header = "strategy".ljust(22) + "".join(f"{s[0]:>10}" for s in sizes)
    print(header)
    for st in STRATEGIES:
        base_val = per_size_means[smallest][st]
        row = st.ljust(22) + "".join(
            f"{per_size_means[s][st]/base_val:>10.2f}" for s in sizes
        )
        print(row)


if __name__ == "__main__":
    main()
