#!/usr/bin/env python3

import numpy as np
import pyspiel
from scipy.optimize import linprog

from _linprog_utils import _parse_linprog, _get_Abc, _LPProblem
from simplex_solver import (
    phase1solver, first_to_second,
    _pivot_col_heuristics, _pivot_row, _apply_pivot,
)


def _infoset_id(state, player):
    return state.information_state_string(player)


def _sorted_sequences(seqs):
    return sorted(seqs, key=lambda s: (len(s), s))


def _card_rank(card_id, num_suits=2):
    """Map card ID to rank index. Leduc: cards 0,1->J(0), 2,3->Q(1), 4,5->K(2)."""
    return card_id // num_suits


def _reweight_chance_outcomes(outcomes, rank_weights, num_suits=2):
    """
    Reweight chance outcomes based on rank weights.

    Each remaining card's probability becomes proportional to its rank weight,
    preserving the card-removal structure (only available cards are considered).
    """
    weighted = []
    total = 0.0
    for action, orig_prob in outcomes:
        rank = _card_rank(action, num_suits)
        w = rank_weights[rank] * orig_prob
        weighted.append((action, w))
        total += w

    if total <= 0:
        n = len(outcomes)
        return [(a, 1.0 / n) for a, _ in outcomes]

    return [(a, w / total) for a, w in weighted]


def build_sequence_form_matrices(game, rank_weights=None):
    """
    Build sequence-form matrices for a 2-player game.

    Parameters
    ----------
    game : pyspiel.Game
    rank_weights : array-like of shape (num_ranks,), optional
        Weights for each card rank (J, Q, K). If None, use the game's default
        (uniform) chance probabilities.
    """
    if game.num_players() != 2:
        raise ValueError("Expected 2-player game")

    num_suits = 2
    root = game.new_initial_state()

    p0_sequences = {()}
    p1_sequences = {()}

    p0_infoset_parent = {}
    p1_infoset_parent = {}

    p0_infoset_actions = {}
    p1_infoset_actions = {}

    payoff_accum = {}

    def record_infoset(player, I, parent_seq, legal_actions):
        if player == 0:
            if I in p0_infoset_parent and p0_infoset_parent[I] != parent_seq:
                raise RuntimeError("Perfect recall violated for player 0")
            p0_infoset_parent.setdefault(I, parent_seq)
            p0_infoset_actions.setdefault(I, set()).update(int(a) for a in legal_actions)
        else:
            if I in p1_infoset_parent and p1_infoset_parent[I] != parent_seq:
                raise RuntimeError("Perfect recall violated for player 1")
            p1_infoset_parent.setdefault(I, parent_seq)
            p1_infoset_actions.setdefault(I, set()).update(int(a) for a in legal_actions)

    def dfs(state, seq0, seq1, chance_prob):
        if state.is_terminal():
            v0 = float(state.returns()[0])
            payoff_accum[(seq0, seq1)] = payoff_accum.get((seq0, seq1), 0.0) + chance_prob * v0
            return

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            if rank_weights is not None:
                outcomes = _reweight_chance_outcomes(outcomes, rank_weights, num_suits)
            for a, p in outcomes:
                child = state.clone()
                child.apply_action(a)
                dfs(child, seq0, seq1, chance_prob * float(p))
            return

        player = state.current_player()
        legal = state.legal_actions(player)
        I = _infoset_id(state, player)

        if player == 0:
            record_infoset(0, I, seq0, legal)
            for a in legal:
                a = int(a)
                new_seq0 = seq0 + ((I, a),)
                p0_sequences.add(new_seq0)
                child = state.clone()
                child.apply_action(a)
                dfs(child, new_seq0, seq1, chance_prob)
        else:
            record_infoset(1, I, seq1, legal)
            for a in legal:
                a = int(a)
                new_seq1 = seq1 + ((I, a),)
                p1_sequences.add(new_seq1)
                child = state.clone()
                child.apply_action(a)
                dfs(child, seq0, new_seq1, chance_prob)

    dfs(root, (), (), 1.0)

    p0_infoset_actions = {I: tuple(sorted(v)) for I, v in p0_infoset_actions.items()}
    p1_infoset_actions = {I: tuple(sorted(v)) for I, v in p1_infoset_actions.items()}

    p0_sequences = _sorted_sequences(p0_sequences)
    p1_sequences = _sorted_sequences(p1_sequences)

    p0_index = {s: i for i, s in enumerate(p0_sequences)}
    p1_index = {s: i for i, s in enumerate(p1_sequences)}

    A = np.zeros((len(p0_sequences), len(p1_sequences)))
    for (s0, s1), val in payoff_accum.items():
        A[p0_index[s0], p1_index[s1]] += val

    def build_constraints(seq_index, infoset_parent, infoset_actions):
        infosets = sorted(infoset_parent.keys())
        M = np.zeros((1 + len(infosets), len(seq_index)))
        rhs = np.zeros(1 + len(infosets))

        M[0, seq_index[()]] = 1.0
        rhs[0] = 1.0

        for r, I in enumerate(infosets, start=1):
            parent_seq = infoset_parent[I]
            M[r, seq_index[parent_seq]] = 1.0
            for a in infoset_actions.get(I, ()):
                child_seq = parent_seq + ((I, int(a)),)
                M[r, seq_index[child_seq]] = -1.0

        return M, rhs

    E, e = build_constraints(p0_index, p0_infoset_parent, p0_infoset_actions)
    F, f = build_constraints(p1_index, p1_infoset_parent, p1_infoset_actions)

    return A, E, e, F, f, p0_index, p1_index, p0_infoset_parent, p1_infoset_parent, p0_infoset_actions, p1_infoset_actions


def solve_zero_sum_sequence_form_lp(A, E, e, F, f):
    """Solve sequence-form LP using scipy/HiGHS (reference solver)."""
    n_x = A.shape[0]
    n_y = A.shape[1]

    c = np.concatenate([np.zeros(n_x), -f])

    A_eq = np.hstack([E, np.zeros((E.shape[0], F.shape[0]))])
    b_eq = e

    A_ub = np.hstack([-A.T, F.T])
    b_ub = np.zeros(n_y)

    bounds = [(0, None)] * n_x + [(None, None)] * F.shape[0]

    sol = linprog(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if sol.status != 0:
        raise RuntimeError("LP failed")

    x = sol.x[:n_x]
    y = -sol.ineqlin.marginals
    v = float(x @ A @ y)

    return x, y, v


def solve_sequence_form_simplex(A, E, e, F, f, strategy='steepest_edge'):
    """
    Solve the sequence-form LP using the project's two-phase simplex.

    The LP is: min [0; -f]^T [x; p]
               s.t. [E, 0][x; p] = e
                    [-A^T, F^T][x; p] <= 0
                    x >= 0, p free

    After _get_Abc converts to standard form, the column layout is:
        [x (n_x) | p+ (n_p) | p- (n_p) | slacks (n_y)]

    Player 1's realization plan y is recovered from the dual variables
    of the inequality constraints (= negative reduced costs of slacks).

    Returns
    -------
    x : player 0 realization plan
    y : player 1 realization plan
    v : game value
    total_nit : total pivot count (phase 1 + phase 2)
    """
    n_x = A.shape[0]
    n_y = A.shape[1]
    n_p = F.shape[0]

    c = np.concatenate([np.zeros(n_x), -f])
    A_eq = np.hstack([E, np.zeros((E.shape[0], n_p))])
    b_eq = e
    A_ub = np.hstack([-A.T, F.T])
    b_ub = np.zeros(n_y)
    bounds = [(0, None)] * n_x + [(None, None)] * n_p

    lp = _LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0=None, integrality=None)
    lp, solver_options = _parse_linprog(lp, None, meth='simplex')
    tol = solver_options.get('tol', 1e-9)
    c0 = 0
    A_std, b_std, c_std, c0, x0 = _get_Abc(lp, c0)

    n_rows, n_cols = A_std.shape

    neg = b_std < 0
    A_std[neg] *= -1
    b_std[neg] *= -1

    # Phase 1 tableau
    av = np.arange(n_rows) + n_cols
    basis = av.copy()

    row_constraints = np.hstack((A_std, np.eye(n_rows), b_std[:, np.newaxis]))
    row_objective = np.hstack((c_std, np.zeros(n_rows), c0))
    row_pseudo_objective = -row_constraints.sum(axis=0)
    row_pseudo_objective[av] = 0
    T = np.vstack((row_constraints, row_objective, row_pseudo_objective))

    # Phase 1 (large LP — need higher iteration limit)
    nit1, status1 = phase1solver(T, basis, maxiter=50000)
    if status1 != 0:
        raise RuntimeError(f"Phase 1 failed: status {status1}")

    # Phase 1 -> Phase 2
    res = first_to_second(T, basis, av)
    if res is None:
        raise RuntimeError("Phase 1->2 transition failed")
    T, basis = res

    # Phase 2 with configurable pivot strategy
    nit2 = 0
    max_iter = 50000
    while nit2 < max_iter:
        pivcol_found, pivcol = _pivot_col_heuristics(T, strategy=strategy, tol=tol)
        if not pivcol_found:
            break
        pivrow_found, pivrow = _pivot_row(T, basis, pivcol, phase=2, tol=tol)
        if not pivrow_found:
            raise RuntimeError("Unbounded LP in Phase 2")
        _apply_pivot(T, basis, pivrow, pivcol, tol=tol)
        nit2 += 1

    return nit1, nit2


def sample_rank_weights(alpha=2.0, num_ranks=3, rng=None):
    """Sample rank weights from Dirichlet(alpha, ..., alpha)."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.dirichlet(np.full(num_ranks, alpha))


def realization_to_behavioral(realization, seq_index, infoset_parent, infoset_actions, fallback="uniform", eps=1e-15):
    policy = {}

    for I in sorted(infoset_parent.keys()):
        parent_seq = infoset_parent[I]
        parent_val = realization[seq_index[parent_seq]]

        actions = infoset_actions.get(I, ())
        if not actions:
            continue

        if parent_val <= eps:
            if fallback == "uniform":
                p = 1.0 / len(actions)
                policy[I] = {a: p for a in actions}
            elif fallback == "zeros":
                policy[I] = {a: 0.0 for a in actions}
            continue

        probs = {}
        for a in actions:
            child_seq = parent_seq + ((I, a),)
            probs[a] = realization[seq_index[child_seq]] / parent_val

        s = sum(probs.values())
        if s > 0:
            for a in probs:
                probs[a] /= s

        policy[I] = probs

    return policy


def run_experiment(game_name, rank_names, num_ranks, n_samples=5, alpha=2.0, seed=42):
    """Run the non-uniform deck experiment for a given game."""
    import warnings
    warnings.filterwarnings('ignore')

    game = pyspiel.load_game(game_name)
    rng = np.random.default_rng(seed)

    strategies = ['blands_rule', 'largest_coefficient', 'largest_increase',
                  'steepest_edge']

    # --- Uniform deck (reference) ---
    print(f"\n{'=' * 60}")
    print(f"GAME: {game_name}")
    print(f"{'=' * 60}")
    print("UNIFORM DECK (reference)")

    (A, E, e, F, f, p0_idx, p1_idx,
     p0_ip, p1_ip, p0_ia, p1_ia) = build_sequence_form_matrices(game)

    x_ref, y_ref, v_ref = solve_zero_sum_sequence_form_lp(A, E, e, F, f)
    print(f"  payoff matrix: {A.shape[0]}x{A.shape[1]} sequences")
    print(f"  LP constraints: {E.shape[0] + A.shape[1]} rows")
    print(f"  scipy value: {v_ref:.6f}")

    for strat in strategies:
        nit1, nit2 = solve_sequence_form_simplex(A, E, e, F, f, strategy=strat)
        print(f"  simplex ({strat:>21s}): phase1={nit1:5d}, phase2={nit2:5d}, total={nit1+nit2:5d}")

    # --- Non-uniform deck samples ---
    print(f"\nNON-UNIFORM DECK SAMPLES (Dirichlet alpha={alpha})")
    print("-" * 60)

    for trial in range(n_samples):
        weights = sample_rank_weights(alpha=alpha, num_ranks=num_ranks, rng=rng)
        print(f"\nTrial {trial + 1}: weights = "
              + ", ".join(f"{rank_names[i]}={weights[i]:.3f}" for i in range(num_ranks)))

        (A, E, e, F, f, p0_idx, p1_idx,
         p0_ip, p1_ip, p0_ia, p1_ia) = build_sequence_form_matrices(game, rank_weights=weights)

        x_ref, y_ref, v_ref = solve_zero_sum_sequence_form_lp(A, E, e, F, f)
        print(f"  scipy value: {v_ref:.6f}")

        for strat in strategies:
            nit1, nit2 = solve_sequence_form_simplex(A, E, e, F, f, strategy=strat)
            print(f"  simplex ({strat:>21s}): phase1={nit1:5d}, phase2={nit2:5d}, total={nit1+nit2:5d}")

        pi0 = realization_to_behavioral(x_ref, p0_idx, p0_ip, p0_ia)
        print(f"  P0 strategy (first 4 infosets):")
        for I in sorted(pi0.keys())[:4]:
            probs = pi0[I]
            prob_str = ", ".join(f"a{a}={p:.3f}" for a, p in sorted(probs.items()))
            print(f"    {I}: {prob_str}")


def main():
    # Kuhn poker: 3 ranks (J, Q, K), 1 suit each → 13x13 LP (fast)
    run_experiment("kuhn_poker", ['J', 'Q', 'K'], num_ranks=3, n_samples=5)

    # Leduc poker with suit isomorphism: 3 ranks, 2 suits → 337x337 LP (moderate)
    run_experiment("leduc_poker(suit_isomorphism=true)", ['J', 'Q', 'K'], num_ranks=3, n_samples=3)


if __name__ == "__main__":
    main()