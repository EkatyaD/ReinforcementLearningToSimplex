"""Per-strategy column-selection wallclock on Leduc sequence-form LPs.

Same methodology as benchmark_pivot_cost.py: walk each phase-2 tableau with
steepest_edge, snapshot every K pivots, then time every strategy on the same
snapshot.
"""

import argparse
import numpy as np

from envs import LeducEnv
from benchmark_pivot_cost import STRATEGIES, time_strategy, collect_snapshots
from config import LEDUC_GAME, LEDUC_ALPHA, LEDUC_NUM_RANKS


def main():
    """CLI entry: benchmark all pivot strategies on sampled Leduc phase-2 tableaus."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-instances", type=int, default=2)
    parser.add_argument("--snaps-per-instance", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=LEDUC_ALPHA)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = LeducEnv(
        game_name=LEDUC_GAME,
        alpha=args.alpha,
        num_ranks=LEDUC_NUM_RANKS,
        seed=args.seed,
    )

    snapshots = []
    for i in range(args.num_instances):
        env.reset()
        print(f"[reset {i+1}/{args.num_instances}] tableau shape={env.T.shape}",
              flush=True)
        T, basis = env.T.copy(), env.basis.copy()
        snapshots.extend(collect_snapshots(T, basis,
                                           max_snapshots=args.snaps_per_instance))

    print(f"\nLeduc: game={LEDUC_GAME}, alpha={args.alpha}, "
          f"ranks={LEDUC_NUM_RANKS}, instances={args.num_instances}, "
          f"snapshots={len(snapshots)}, repeats={args.repeats}", flush=True)

    per_strategy = {s: [] for s in STRATEGIES}
    for k, T in enumerate(snapshots):
        for s in STRATEGIES:
            per_strategy[s].append(time_strategy(T, s, repeats=args.repeats))
        print(f"  snapshot {k+1}/{len(snapshots)} timed", flush=True)

    means = {s: float(np.mean(per_strategy[s])) for s in STRATEGIES}
    ref_lc = means["largest_coefficient"]
    ref_bl = means["blands_rule"]
    print(f"\n{'strategy':<22} {'µs/call':>12} {'std':>12} {'norm/lc':>10} {'norm/bl':>10}")
    for s in STRATEGIES:
        std = float(np.std(per_strategy[s]))
        print(f"{s:<22} {means[s]:>12.2f} {std:>12.2f} "
              f"{means[s]/ref_lc:>10.3f} {means[s]/ref_bl:>10.3f}")


if __name__ == "__main__":
    main()
