"""Two-phase simplex core: pivot-column heuristics, the ratio test, pivot
application, a Phase 1 solver, and the tableau-construction routines that turn
a zero-sum game matrix into a Phase 2 tableau.

This module holds the thesis' own simplex logic. The five pivot-column
heuristics compared in the experiments (``largest_coefficient``,
``largest_increase``, ``steepest_edge``, ``random_edge``, ``blands_rule``) are
implemented here (see ``_pivot_col_heuristics``); the RL agent's training
action space is the 3-rule subset in ``config.PIVOT_MAP``, the remaining two
serve as evaluation-only baselines. LP standard-form plumbing is imported from
``_linprog_utils`` (vendored from SciPy — see that file's header). The gym
environments that drive these primitives live in ``envs.py``.
"""

import numpy as np
from warnings import warn

# The following helpers are vendored from SciPy (BSD-3, see _linprog_utils.py):
# LP parsing/standard-form conversion, not original to this thesis.
from _linprog_utils import _parse_linprog, _get_Abc, _LPProblem


def potential_increase(T, col_index, cost_j, tol=1e-9):
    """Textbook greatest-improvement score for entering column `col_index`.

    Score is `-c_j * theta_j` where `theta_j = min(b_i / a_ij)` over rows
    with a_ij > tol — the actual step the simplex takes (binding row =
    ratio test minimum, same quantity that `_pivot_row` computes).
    """
    col = T[:-1, col_index]
    mask = col > tol
    if not np.any(mask):
        return None
    return float(-cost_j * (T[:-1, -1][mask] / col[mask]).min())

def pivot_col_random_edge(T, ma, tol=1e-9):
    """
    Textbook Random Edge rule: pick uniformly at random from columns with
    negative reduced cost. O(|cand|) — no per-candidate ratio test.
    """
    col_candidates = np.ma.nonzero(ma < 0)[0]
    if col_candidates.size == 0:
        return False, np.nan
    return True, int(np.random.choice(col_candidates))


def pivot_col_largest_increase(T, ma, tol=1e-9):
    cost_row = T[-1, :-1]  # objective coefficients excluding the RHS

    # Among negative entries, find the column with largest improvement
    col_candidates = np.ma.nonzero(ma < 0)[0]
    best_increase = -float('inf')
    best_col = None

    for j in col_candidates:
        cost_j = cost_row[j]
        inc = potential_increase(T, j, cost_j, tol=tol)
        if inc is not None and inc > best_increase:
            best_increase = inc
            best_col = j

    if best_col is None:
        # Means for every negative cost_j, no row was feasible => unbounded
        return False, np.nan

    return True, best_col

def _pivot_col_heuristics(T, strategy, tol=1e-9):
    """
    Pick pivot column using one of the supported heuristics.
    """
    ma = np.ma.masked_where(T[-1, :-1] >= -tol, T[-1, :-1], copy=False)
    if ma.count() == 0:
        return False, np.nan

    if strategy == 'blands_rule':
        # Bland's rule: choose the leftmost negative coefficient
        col = np.nonzero(np.logical_not(np.atleast_1d(ma.mask)))[0][0]
        return True, col

    elif strategy == 'largest_coefficient':
        col = np.ma.nonzero(ma == ma.min())[0][0]
        return True,  col

    elif strategy == 'largest_increase':
        res, col = pivot_col_largest_increase(T, ma, tol)
        return res, col

    elif strategy == 'random_edge':
        return pivot_col_random_edge(T, ma, tol)

    elif strategy == 'steepest_edge':
        cost_row = T[-1, :-1]
        best_ratio = -np.inf
        best_col = None
        col_candidates = np.ma.nonzero(ma < 0)[0]  # j where cost_j < -tol

        for j in col_candidates:
            cost_j = cost_row[j]
            col_vector = T[:-1, j]  # exclude the objective row
            col_norm = np.linalg.norm(col_vector)

            if col_norm <= tol:
                continue  # avoid division by near-zero or zero norm

            ratio = abs(cost_j) / col_norm

            if ratio > best_ratio:
                best_ratio = ratio
                best_col = j

        if best_col is None:
            return False, np.nan
        return True, best_col

def _pivot_row(T, basis, pivcol, phase, tol=1e-9, bland=False):
    """
    Harris two-pass ratio test for selecting the leaving row.

    Parameters
    ----------
    T : ndarray
        Simplex tableau.
    basis : ndarray
        Current basis indices (used only for Bland tie-break if requested).
    pivcol : int
        Entering column index.
    phase : int
        1 or 2; controls how many bottom rows are excluded from the ratio test.
    tol : float
        Positivity tolerance for the pivot column entries.
    bland : bool
        If True, apply Bland's rule: among rows attaining the exact minimum
        ratio, pick the one whose basic variable has the smallest index
        (preserves the anti-cycling guarantee).
        If False (default), choose the row with the largest pivot element
        within the Harris eta-band of near-minimum ratios (numerical
        stability).

    Returns
    -------
    (found: bool, pivrow: int)
    """
    # Exclude objective rows at bottom (same as your original code)
    k = 2 if phase == 1 else 1
    col = T[:-k, pivcol]

    # Eligible rows have strictly positive pivot column entries
    eligible = col > tol
    if not np.any(eligible):
        return False, np.nan

    b = T[:-k, -1]

    # Compute ratios only for eligible rows
    q = np.full_like(col, np.inf, dtype=np.float64)
    q[eligible] = b[eligible] / col[eligible]

    # Pass 1: find robust minimum ratio
    q_min = np.min(q)
    if not np.isfinite(q_min):
        return False, np.nan

    # Pass 2: define admissible set near the minimum
    # eta: small relative tolerance; you can tweak 1e-7..1e-4 if needed
    eta = 1e-7
    threshold = (1.0 + eta) * q_min

    admissible = (q <= threshold) & np.isfinite(q)
    rows = np.where(admissible)[0]
    if rows.size == 0:
        # Fallback: strict minimizer (should rarely happen)
        rows = np.array([int(np.argmin(q))])

    if bland:
        # True Bland's rule: among rows attaining the exact minimum ratio, pick
        # the one whose basic variable has the smallest index. Using the exact
        # minimizer set (not the Harris eta-band, which can select a row with a
        # strictly larger ratio) preserves feasibility and Bland's anti-cycling
        # guarantee.
        min_rows = np.where(q == q_min)[0]
        idx = min_rows[np.argmin(np.take(basis, min_rows))]
    else:
        # Choose numerically strongest pivot among admissible rows:
        # largest pivot element in the entering column
        pivots = col[rows]
        idx = rows[np.argmax(pivots)]

    return True, int(idx)


def _apply_pivot(T, basis, pivrow, pivcol, tol=1e-9):

    basis[pivrow] = pivcol
    pivval = T[pivrow, pivcol]
    T[pivrow] /= pivval
    col = T[:, pivcol].copy()
    col[pivrow] = 0.0
    T -= np.outer(col, T[pivrow])

    # The selected pivot should never lead to a pivot value less than the tol.
    if np.isclose(pivval, tol, atol=0, rtol=1e4):
        message = (
            f"The pivot operation produces a pivot value of:{pivval: .1e}, "
            "which is only slightly greater than the specified "
            f"tolerance{tol: .1e}. This may lead to issues regarding the "
            "numerical stability of the simplex method. "
            "Removing redundant constraints, changing the pivot strategy "
            "via Bland's rule or increasing the tolerance may "
            "help reduce the issue.")
        warn(message, stacklevel=5)



def phase1solver(T, basis,
                   maxiter=1000, tol=1e-9, nit0=0):
    nit = nit0
    status = 0
    phase=1
    bland=False
    message = ''
    complete = False
    m = T.shape[1] - 2
    if len(basis[:m]) == 0:
        solution = np.empty(T.shape[1] - 1, dtype=np.float64)
    else:
        solution = np.empty(max(T.shape[1] - 1, max(basis[:m]) + 1),
                            dtype=np.float64)

    while not complete:
        # Find the pivot column
        pivcol_found, pivcol = _pivot_col_heuristics(T, strategy='steepest_edge', tol=tol)
        if not pivcol_found:
            pivcol = np.nan
            pivrow = np.nan
            status = 0
            complete = True
        else:
            # Find the pivot row
            pivrow_found, pivrow = _pivot_row(T, basis, pivcol, phase, tol, bland)
            if not pivrow_found:
                status = 3
                complete = True


        if not complete:
            if nit >= maxiter:
                # Iteration limit exceeded
                status = 1
                complete = True
            else:
                _apply_pivot(T, basis, pivrow, pivcol, tol)
                nit += 1
    return nit, status


def change_to_zero_sum(GameMatrix):
    m, n = GameMatrix.shape
    c = np.append(np.zeros(m), -1)
    A_ub = np.hstack([-GameMatrix.T, np.ones((n, 1))])
    b_ub = np.zeros(n)
    A_eq = np.append(np.ones(m), 0).reshape(1, -1)
    b_eq = [1]
    bounds = [(0, None)] * m + [(None, None)]

    lp = _LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0=None, integrality=None)
    options = None

    lp, solver_options = _parse_linprog(lp, options, meth='simplex')
    tol = solver_options.get('tol', 1e-9)
    c0 = 0
    A, b, c, c0, x0 = _get_Abc(lp, c0)

    n, m = A.shape

    # All constraints must have b >= 0.
    is_negative_constraint = np.less(b, 0)
    A[is_negative_constraint] *= -1
    b[is_negative_constraint] *= -1

    # As all cons   traints are equality constraints the artificial variables
    # will also be basic variables.
    av = np.arange(n) + m
    basis = av.copy()

    row_constraints = np.hstack((A, np.eye(n), b[:, np.newaxis]))
    row_objective = np.hstack((c, np.zeros(n), c0))
    row_pseudo_objective = -row_constraints.sum(axis=0)
    row_pseudo_objective[av] = 0
    T = np.vstack((row_constraints, row_objective, row_pseudo_objective))

    return T, basis, av

# change from first phase to second
def first_to_second(T, basis, av):
    status = 0
    # Adaptive tolerance based on matrix size
    m = len(basis) - len(av)  # number of original variables
    base_tol = 1e-9
    adaptive_tol = base_tol * max(1, m / 10)  # Increase tolerance for larger matrices
    
    if abs(T[-1, -1]) < adaptive_tol:
        # Remove the pseudo-objective row from the tableau
        T = T[:-1, :]
        # Remove the artificial variable columns from the tableau
        T = np.delete(T, av, 1)
    else:
        # Failure to find a feasible starting point
        status = 2
        messages = {0: "Optimization terminated successfully.",
                    1: "Iteration limit reached.",
                    2: "Optimization failed. Unable to find a feasible"
                       " starting point.",
                    3: "Optimization failed. The problem appears to be unbounded.",
                    4: "Optimization failed. Singular matrix encountered."}
        messages[status] = (
            "Phase 1 of the simplex method failed to find a feasible "
            "solution. The pseudo-objective function evaluates to "
            f"{abs(T[-1, -1]):.1e} "
            f"which exceeds the required tolerance of {adaptive_tol} for a solution to be "
            "considered 'close enough' to zero to be a basic solution. "
            "Consider increasing the tolerance to be greater than "
            f"{abs(T[-1, -1]):.1e}. "
            "If this tolerance is unacceptably large the problem may be "
            "infeasible."
        )

    if status == 0:
        # Phase 2
        # nit2, status = phase2solver(T, n, basis)
        return T, basis

    else:
        print("[solve_zero_sum] Phase 1 failed or LP is numerically unstable.")
        print(f"[solve_zero_sum] Pseudo-objective: {T[-1, -1]:.2e} vs adaptive_tol {adaptive_tol:.1e}, status={status}")
        return None

def build_phase2_tableau_canonical(A, b, c, basis, tol=1e-12):
    """
    Given equality-form constraints A x = b (with x >= 0), an objective c^T x,
    and a chosen basis 'basis' (one column index per row), return the canonical
    Phase-2 simplex tableau:
        [ I | B^{-1}N | B^{-1}b ]
        [ 0 |  r_N     |   z0   ]
    where r_N = c_N - c_B B^{-1} N, z0 = c_B^T B^{-1} b.
    Assumes 'basis' selects a nonsingular square submatrix B of A.

    Args:
        A : (m x n) ndarray
        b : (m,) ndarray
        c : (n,) ndarray
        basis : (m,) ndarray of int column indices
        tol : float, numerical guard

    Returns:
        T  : ((m+1) x (n+1)) ndarray, canonical tableau
        basis : unchanged, but now consistent with T’s identity block
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    basis = np.asarray(basis, dtype=int)

    m, n = A.shape
    if basis.size != m:
        raise ValueError("basis length must equal number of rows m")

    # Partition columns into basis and nonbasis
    B_cols = basis
    N_cols = np.array([j for j in range(n) if j not in set(B_cols)], dtype=int)

    # Extract blocks
    B = A[:, B_cols]              # (m x m)
    N = A[:, N_cols]              # (m x (n-m))
    c_B = c[B_cols]               # (m,)
    c_N = c[N_cols]               # (n-m,)

    # Invert B (or solve)
    try:
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError as e:
        raise ValueError("Chosen basis is singular; pick a different feasible basis.") from e

    # Canonical blocks
    B_inv_N = B_inv @ N           # (m x (n-m))
    B_inv_b = B_inv @ b           # (m,)

    # Objective reduction
    r_N = c_N - c_B @ B_inv_N     # (n-m,)
    z0  = c_B @ B_inv_b           # scalar

    # Assemble the full tableau columns in original variable order
    # Start with zeros; we will fill basis and nonbasis locations.
    top_left = np.zeros((m, n))
    # Put identity in basis columns
    for i, col in enumerate(B_cols):
        top_left[i, col] = 1.0
    # Put B^{-1}N into nonbasis columns
    for j_in, col in enumerate(N_cols):
        top_left[:, col] = B_inv_N[:, j_in]

    # RHS
    rhs = B_inv_b.reshape(-1, 1)

    # Objective row in original variable order
    obj_row = np.zeros(n)
    # Basic reduced costs are zero by construction
    for j_in, col in enumerate(N_cols):
        obj_row[col] = r_N[j_in]

    # Build tableau [ A' | b' ; obj | z0 ]
    T = np.vstack([
        np.hstack([top_left, rhs]),
        np.hstack([obj_row, np.array([z0])])
    ])

    # Tiny cleanup: zero-out near
    # -zeros for numerical neatness
    T[np.abs(T) < tol] = 0.0
    return T, basis

def change_to_zero_sum_phase2_only(GameMatrix):
    """
    Convert a zero-sum game matrix directly to a canonical Phase 2 tableau
    (same output shape as first_to_second would produce after removing AVs).
    Returns (T, basis, K).
    """
    m, n = GameMatrix.shape

    # Shift so B >= 0 and keep K for value unshift later
    min_element = np.min(GameMatrix)
    K = max(0.0, -min_element + 1e-6)
    B = GameMatrix + K

    # Standard-form A, b, c
    slack_matrix = np.eye(n)
    A_constraints = np.hstack([-B.T, np.ones((n, 1)), slack_matrix])
    A_sum         = np.hstack([np.ones(m), np.zeros(1), np.zeros(n)])
    A = np.vstack([A_constraints, A_sum])

    b = np.zeros(n + 1)
    b[-1] = 1.0

    c = np.zeros(m + 1 + n)
    c[m] = -1.0

    # Basis: all slacks + x0 (same reasoning as above)
    basis = np.empty(n + 1, dtype=int)
    basis[:n] = m + 1 + np.arange(n)
    basis[-1] = 0

    # Canonicalize
    T, basis = build_phase2_tableau_canonical(A, b, c, basis)
    return T, basis, K


if __name__ == '__main__':
    A = np.array([[-1, -1],
                  [-2, 3],
                  ])
    change_to_zero_sum(A)

