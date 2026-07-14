
"""Evaluate the final Leduc PPO models in results/leduc/models/ against the 5
fixed pivot heuristics on sequence-form Leduc LPs.

Reuses the tableau construction + fixed/RL runners from
`leduc_experiment_runner.py`. Fixed-strategy results are computed ONCE per test
set and shared across all models (they don't depend on the model), so only the
RL agent is re-run per model. The agent's per-rule action counts are traced so
we can detect collapse-to-a-single-rule.

Weighted cost uses STEP_PENALTY_WEIGHTS_LEDUC explicitly (imported from config),
so the report is correct regardless of the current GAME_MODE.

Outputs (written into results/leduc/evaluation/):
  eval_new_models.json  (raw per-LP results)
  eval_new_models.md    (formatted report)
"""
import sys, os, json, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))  # results/leduc -> repo root
sys.path.insert(0, REPO_ROOT)
warnings.filterwarnings('ignore')

import numpy as np
from collections import defaultdict
import pyspiel
from stable_baselines3 import PPO
from scipy.stats import wilcoxon

import leduc_experiment_runner as runner
from leduc_experiment_runner import sample_leduc_tableaus, run_fixed_strategy
from envs import SecondPhasePivotingEnv
from wrappers import CompactObsWrapper
from config import (
    LEDUC_GAME, LEDUC_ALPHA,
    PIVOT_MAP, PIVOT_MAP_TEST, STEP_PENALTY_WEIGHTS_LEDUC,
)

# Force the runner's weighted-cost to use the Leduc per-pivot weights.
runner.STEP_PENALTY_WEIGHTS = STEP_PENALTY_WEIGHTS_LEDUC
WEIGHTS = STEP_PENALTY_WEIGHTS_LEDUC

# Cap iterations at 20k: the weak rules (blands_rule, random_edge) either
# converge well under this (~1300 pivots) or cycle to the cap — they don't
# converge in the 20k–50k window, so this halves runtime without changing the
# convergence tally. Reflected in the report's "Maxiter" line.
MAXITER = 20_000
runner.MAXITER = MAXITER

MODELS_DIR = os.path.join(HERE, 'models')
OUT_DIR = os.path.join(HERE, 'evaluation')
MODELS = [
    'ppo_leduc_ckpt_30000000_alpha100.0_compact_weighted.zip',
    'ppo_leduc_ckpt_30000000_alpha100.0_compact_unweighted.zip',
]
N_LP = 50
SEED = 42
ALPHA_IN = LEDUC_ALPHA            # 100.0
ALPHA_OUT = max(1.0, LEDUC_ALPHA / 10.0)  # 10.0


def run_rl_traced(T, basis, model, use_compact):
    """Run the agent; also record how many pivots used each rule."""
    base_env = SecondPhasePivotingEnv(T.copy(), basis.copy())
    env = CompactObsWrapper(base_env) if use_compact else base_env
    obs, _ = env.reset()
    done = truncated = False
    info = {}
    weighted = 0.0
    counts = defaultdict(int)
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        strat = PIVOT_MAP.get(int(action))
        prev = base_env.nit
        obs, _, done, truncated, info = env.step(action)
        if base_env.nit > prev and strat is not None:
            weighted += float(WEIGHTS.get(strat, 1.0))
            counts[strat] += 1
    status = info.get('status', 'unknown')
    if truncated and not done:
        status = 'loop'
    return {'status': status, 'nit': base_env.nit, 'weighted_cost': weighted,
            'objective': float(base_env.T[-1, -1]), 'action_counts': dict(counts)}


# ---------------------------------------------------------------------------
# Build shared test sets (fixed-strategy results computed once)
# ---------------------------------------------------------------------------

def build_testset(game, n, alpha, rng, strategies):
    """Sample n Leduc LPs and solve each with every fixed strategy (shared across models)."""
    tabs = sample_leduc_tableaus(game, n, alpha, rng, uniform=False)
    rows = []
    for i, tab in enumerate(tabs):
        T, basis = tab['T'], tab['basis']
        row = {'idx': i, 'phase1_nit': int(tab['phase1_nit']),
               'fixed': {}, 'rl': {}}
        for s in strategies:
            row['fixed'][s] = run_fixed_strategy(T, basis, s)
        row['_T'] = T
        row['_basis'] = basis
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"    built+fixed {i+1}/{len(tabs)}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Analysis helpers -> markdown
# ---------------------------------------------------------------------------

def metric_tables(rows, model_key, strategies, metric, label):
    """Return markdown for one metric (pivot count or weighted cost)."""
    methods = strategies + ['rl']
    def val(row, m):
        """Result dict for method m in this row (fixed strategy or the RL agent)."""
        d = row['fixed'][m] if m in strategies else row['rl'][model_key]
        return d
    out = [f"### {label}\n"]
    # stats
    out.append("| Method | Mean | Median | Min | Max | N |")
    out.append("|---|---:|---:|---:|---:|---:|")
    stat_rows = []
    for m in methods:
        vals = [val(r, m)[metric] for r in rows if val(r, m)['status'] == 'optimal']
        if vals:
            a = np.array(vals, float)
            stat_rows.append((m, a.mean(), float(np.median(a)), a.min(), a.max(), len(a)))
    # sort by mean ascending for readability, but keep rl flagged
    for m, mean, med, mn, mx, n in sorted(stat_rows, key=lambda x: x[1]):
        name = '**RL Agent**' if m == 'rl' else m
        b = '**' if m == 'rl' else ''
        out.append(f"| {name} | {b}{mean:.2f}{b} | {b}{med:.2f}{b} | {mn:.2f} | {mx:.2f} | {n} |")
    out.append("")
    # head-to-head RL vs each
    out.append("**Head-to-head (RL vs each):**\n")
    out.append("| Heuristic | RL wins | Ties | RL loses | Wilcoxon p | Direction |")
    out.append("|---|---:|---:|---:|---:|---|")
    for s in strategies:
        wins = ties = losses = 0
        prl, ph = [], []
        for r in rows:
            rl_ok = r['rl'][model_key]['status'] == 'optimal'
            h_ok = r['fixed'][s]['status'] == 'optimal'
            if rl_ok and h_ok:
                rl_n, h_n = r['rl'][model_key][metric], r['fixed'][s][metric]
                if rl_n < h_n: wins += 1
                elif rl_n == h_n: ties += 1
                else: losses += 1
                prl.append(rl_n); ph.append(h_n)
            elif rl_ok and not h_ok: wins += 1
            elif not rl_ok and h_ok: losses += 1
        pstr, direction = '—', '—'
        if len(prl) >= 10:
            diffs = np.array(prl, float) - np.array(ph, float)
            nz = diffs[diffs != 0]
            if len(nz) >= 10:
                med = float(np.median(nz))
                alt = 'less' if med < 0 else 'greater'
                try:
                    _, p = wilcoxon(nz, alternative=alt)
                    pstr = f"{p:.6f}"
                except Exception:
                    pstr = 'n/a'
                direction = 'RL better' if med < 0 else 'Heuristic better'
            else:
                direction = 'tie'
        elif wins == 0 and losses == 0:
            direction = f'tie ({ties}/{ties})'
        out.append(f"| {s} | {wins} | {ties} | {losses} | {pstr} | {direction} |")
    out.append("")
    # vs best-per-instance
    wins = ties = losses = 0
    pct = []
    for r in rows:
        rl_ok = r['rl'][model_key]['status'] == 'optimal'
        best = None
        for s in strategies:
            if r['fixed'][s]['status'] == 'optimal':
                c = r['fixed'][s][metric]
                best = c if best is None else min(best, c)
        if rl_ok and best is not None:
            rl_n = r['rl'][model_key][metric]
            if best > 0:
                pct.append((best - rl_n) / best * 100)
            if rl_n < best: wins += 1
            elif rl_n == best: ties += 1
            else: losses += 1
    meanpct = f"{np.mean(pct):+.1f}%" if pct else 'n/a'
    medpct = f"{float(np.median(pct)):+.1f}%" if pct else 'n/a'
    out.append(f"**RL vs best-per-instance:** W {wins} / T {ties} / L {losses} — "
               f"mean reduction {meanpct}, median {medpct}\n")
    return '\n'.join(out)


def convergence_table(rows, model_key, strategies):
    """Markdown table of how many LPs each method solved to optimality."""
    methods = strategies + ['rl']
    conv = {}
    for m in methods:
        if m == 'rl':
            conv[m] = sum(1 for r in rows if r['rl'][model_key]['status'] == 'optimal')
        else:
            conv[m] = sum(1 for r in rows if r['fixed'][m]['status'] == 'optimal')
    n = len(rows)
    out = ["| Method | Converged |", "|---|---|"]
    for m in sorted(methods, key=lambda x: -conv[x]):
        name = '**RL Agent**' if m == 'rl' else m
        b = '**' if m == 'rl' else ''
        out.append(f"| {name} | {b}{conv[m]} / {n}{b} |")
    return '\n'.join(out)


def action_summary(rows, model_key):
    """One-line summary of the agent's pivot-rule usage (detects rule collapse)."""
    total = defaultdict(int)
    for r in rows:
        for k, v in r['rl'][model_key].get('action_counts', {}).items():
            total[k] += v
    grand = sum(total.values())
    if grand == 0:
        return "_no pivots recorded_"
    parts = []
    for k in sorted(total, key=lambda x: -total[x]):
        parts.append(f"`{k}` {total[k]} ({total[k]/grand*100:.1f}%)")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Re-evaluate both shipped Leduc models, writing eval_new_models.{json,md}."""
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)
    game = pyspiel.load_game(LEDUC_GAME)
    strategies = list(PIVOT_MAP_TEST.values())

    print("Building test sets (fixed strategies computed once)...", flush=True)
    print("  in-distribution alpha=%.1f" % ALPHA_IN, flush=True)
    indist = build_testset(game, N_LP, ALPHA_IN, rng, strategies)
    print("  out-of-distribution alpha=%.1f" % ALPHA_OUT, flush=True)
    ood = build_testset(game, N_LP, ALPHA_OUT, rng, strategies)
    testsets = {'in_distribution': indist, 'out_of_distribution': ood}

    # Run each model
    model_meta = {}
    for fname in MODELS:
        path = os.path.join(MODELS_DIR, fname)
        print(f"\nLoading {fname}", flush=True)
        model = PPO.load(path[:-4] if path.endswith('.zip') else path)
        use_compact = type(model.observation_space).__name__ == 'Box'
        model_meta[fname] = {'use_compact': use_compact,
                             'obs': type(model.observation_space).__name__}
        for setname, rows in testsets.items():
            for i, r in enumerate(rows):
                r['rl'][fname] = run_rl_traced(r['_T'], r['_basis'], model, use_compact)
            print(f"  {setname}: agent done ({len(rows)} LPs)", flush=True)

    # ---- emit JSON (strip heavy tableaus) ----
    dump = {}
    for setname, rows in testsets.items():
        dump[setname] = []
        for r in rows:
            rr = {'idx': r['idx'], 'phase1_nit': r['phase1_nit'],
                  'fixed': r['fixed'], 'rl': r['rl']}
            dump[setname].append(rr)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'eval_new_models.json'), 'w') as f:
        json.dump(dump, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print("\nWrote eval_new_models.json", flush=True)

    # ---- emit markdown ----
    md = []
    md.append("# Leduc evaluation — models in `results/leduc/models/`\n")
    md.append("Generated by `results/leduc/eval_leduc.py`.\n")
    md.append("## Setup\n")
    md.append("| | |")
    md.append("|---|---|")
    md.append(f"| Game | `{LEDUC_GAME}` |")
    md.append(f"| LPs per test set | {N_LP} |")
    md.append(f"| Seed | {SEED} |")
    md.append(f"| In-distribution α | {ALPHA_IN} (Dirichlet rank weights, near-uniform deck) |")
    md.append(f"| Out-of-distribution α | {ALPHA_OUT} (broader deck perturbation) |")
    md.append("| Phase 1 | `phase1solver` (steepest_edge, hardcoded) — agent acts in Phase 2 only |")
    md.append(f"| Maxiter | {MAXITER:,} |")
    md.append(f"| Action space (training `PIVOT_MAP`) | {list(PIVOT_MAP.values())} |")
    md.append(f"| Tested heuristics (`PIVOT_MAP_TEST`) | {strategies} |\n")
    md.append("Per-pivot weights for weighted cost (`STEP_PENALTY_WEIGHTS_LEDUC`, current calibration):\n")
    md.append("| strategy | weight |")
    md.append("|---|---:|")
    for k, v in WEIGHTS.items():
        md.append(f"| {k} | {v:.2f} |")
    md.append("")

    # comparison summary across models (in-distribution, pivot + weighted means for RL)
    md.append("## Model comparison summary\n")
    md.append("RL-agent mean pivot count / mean weighted cost, and dominant rule. "
              "Lower is better.\n")
    md.append("| Model | obs | set | RL mean pivots | RL mean wcost | RL converged | dominant rule(s) |")
    md.append("|---|---|---|---:|---:|---|---|")
    for fname in MODELS:
        for setname, rows in testsets.items():
            piv = [r['rl'][fname]['nit'] for r in rows if r['rl'][fname]['status'] == 'optimal']
            wc = [r['rl'][fname]['weighted_cost'] for r in rows if r['rl'][fname]['status'] == 'optimal']
            conv = len(piv)
            mp = f"{np.mean(piv):.1f}" if piv else 'n/a'
            mw = f"{np.mean(wc):.1f}" if wc else 'n/a'
            short = fname.replace('ppo_leduc_', '').replace('_alpha100.0', '').replace('.zip', '')
            md.append(f"| `{short}` | {model_meta[fname]['obs']} | {setname} | {mp} | {mw} | "
                      f"{conv}/{len(rows)} | {action_summary(rows, fname)} |")
    md.append("")

    # per-model detail
    for fname in MODELS:
        short = fname.replace('.zip', '')
        md.append(f"## {short}\n")
        md.append(f"- obs space: `{model_meta[fname]['obs']}` "
                  f"(compact wrapper: {model_meta[fname]['use_compact']})\n")
        for setname, rows in testsets.items():
            md.append(f"### {setname}\n")
            md.append(f"**Convergence (within {MAXITER:,} pivots):**\n")
            md.append(convergence_table(rows, fname, strategies))
            md.append("")
            md.append(f"**Agent action distribution:** {action_summary(rows, fname)}\n")
            md.append(metric_tables(rows, fname, strategies, 'nit', 'Pivot count'))
            md.append(metric_tables(rows, fname, strategies, 'weighted_cost',
                                    'Weighted cost (Σ STEP_PENALTY_WEIGHTS_LEDUC)'))
            # game value consistency
            bad = 0
            for r in rows:
                objs = [r['fixed'][s]['objective'] for s in strategies
                        if r['fixed'][s]['status'] == 'optimal']
                if r['rl'][fname]['status'] == 'optimal':
                    objs.append(r['rl'][fname]['objective'])
                if objs and (max(objs) - min(objs)) > 1e-3:
                    bad += 1
            md.append(f"_Game-value consistency: {'all agree within 1e-3' if bad == 0 else f'{bad} LPs disagree (>1e-3)'}._\n")

    with open(os.path.join(OUT_DIR, 'eval_new_models.md'), 'w') as f:
        f.write('\n'.join(md))
    print("Wrote eval_new_models.md", flush=True)


if __name__ == '__main__':
    main()