"""Evaluate the empty-observation Leduc models: trace which pivot rule the
agent uses most often across real Leduc phase-2 LPs.

The observation is a single constant feature, so the policy is
state-independent by construction. We still run it end-to-end on sampled LPs
and count per-rule pivots to report the dominant rule (and confirm collapse).
"""
import sys, os, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
# results/leduc/empty_obs_baseline -> repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO_ROOT)
warnings.filterwarnings('ignore')

import numpy as np
from collections import defaultdict
import pyspiel
from stable_baselines3 import PPO

from leduc_experiment_runner import sample_leduc_tableaus
from envs import SecondPhasePivotingEnv
from wrappers import EmptyObsWrapper
from config import LEDUC_GAME, LEDUC_ALPHA, PIVOT_MAP, STEP_PENALTY_WEIGHTS_LEDUC

MODELS_DIR = os.path.join(HERE, 'models')
MODELS = [
    'ppo_leduc_ckpt_4000000_alpha100.0_empty_unweighted.zip',
    'ppo_leduc_ckpt_4000000_alpha100.0_empty_weighted.zip',
    'ppo_leduc_5000000_alpha100.0_empty_weighted.zip',
]
N_LP = 30
SEED = 42
WEIGHTS = STEP_PENALTY_WEIGHTS_LEDUC


def run_rl_traced(T, basis, model):
    """Run the empty-obs agent on one tableau, counting pivots per rule."""
    base_env = SecondPhasePivotingEnv(T.copy(), basis.copy())
    env = EmptyObsWrapper(base_env)
    obs, _ = env.reset()
    done = truncated = False
    info = {}
    counts = defaultdict(int)
    weighted = 0.0
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        strat = PIVOT_MAP.get(int(action))
        prev = base_env.nit
        obs, _, done, truncated, info = env.step(action)
        if base_env.nit > prev and strat is not None:
            counts[strat] += 1
            weighted += float(WEIGHTS.get(strat, 1.0))
    status = info.get('status', 'unknown')
    if truncated and not done:
        status = 'loop'
    return {'status': status, 'nit': base_env.nit,
            'weighted_cost': weighted, 'action_counts': dict(counts)}


def main():
    """Run every empty-obs model on shared sampled LPs and report rule usage."""
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)
    game = pyspiel.load_game(LEDUC_GAME)

    print(f"Sampling {N_LP} Leduc LPs (alpha={LEDUC_ALPHA})...", flush=True)
    tabs = sample_leduc_tableaus(game, N_LP, LEDUC_ALPHA, rng, uniform=False)

    for fname in MODELS:
        path = os.path.join(MODELS_DIR, fname)
        model = PPO.load(path[:-4])
        obs_shape = getattr(model.observation_space, 'shape', None)
        total = defaultdict(int)
        nits, wcosts, conv = [], [], 0
        for tab in tabs:
            r = run_rl_traced(tab['T'], tab['basis'], model)
            for k, v in r['action_counts'].items():
                total[k] += v
            if r['status'] == 'optimal':
                conv += 1
                nits.append(r['nit'])
                wcosts.append(r['weighted_cost'])
        grand = sum(total.values())
        print(f"\n=== {fname} ===")
        print(f"obs space: {model.observation_space}  shape={obs_shape}")
        print(f"converged: {conv}/{N_LP}"
              + (f" | mean pivots {np.mean(nits):.1f} | mean wcost {np.mean(wcosts):.1f}"
                 if nits else ""))
        print("pivot-rule usage (most common first):")
        for k in sorted(total, key=lambda x: -total[x]):
            print(f"  {k:22s} {total[k]:7d} ({total[k]/max(grand,1)*100:5.1f}%)")


if __name__ == '__main__':
    main()
