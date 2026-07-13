import gymnasium as gym
import numpy as np
from warnings import warn
import scipy.sparse as sps
from collections import namedtuple

from gymnasium import spaces
from stable_baselines3.common.type_aliases import GymEnv

from _linprog_utils import (
    _parse_linprog, _presolve, _get_Abc, _LPProblem, _autoscale,
    _postsolve, _check_result, _display_summary)

from config import PIVOT_MAP, NUM_PIVOT_STRATEGIES

## NOTE: Environments moved to envs.py (SecondPhasePivotingEnv, FirstPhasePivotingEnv)



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

def _pivot_col(T, tol=1e-9, bland=False):

    ma = np.ma.masked_where(T[-1, :-1] >= -tol, T[-1, :-1], copy=False)
    if ma.count() == 0:
        return False, np.nan
    if bland:
        # ma.mask is sometimes 0d
        return True, np.nonzero(np.logical_not(np.atleast_1d(ma.mask)))[0][0]
    return True, np.ma.nonzero(ma == ma.min())[0][0]


# def _pivot_row(T, basis, pivcol, phase, tol=1e-9, bland=False):

#     if phase == 1:
#         k = 2
#     else:
#         k = 1
#     ma = np.ma.masked_where(T[:-k, pivcol] <= tol, T[:-k, pivcol], copy=False)
#     if ma.count() == 0:
#         return False, np.nan
#     mb = np.ma.masked_where(T[:-k, pivcol] <= tol, T[:-k, -1], copy=False)
#     q = mb / ma
#     min_rows = np.ma.nonzero(q == q.min())[0]
#     if bland:
#         return True, min_rows[np.argmin(np.take(basis, min_rows))]
#     return True, min_rows[0]
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
        If True, tie-break among near-minimum ratios using smallest basis index.
        If False (default), choose the row with the largest pivot in the admissible set.

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
        # Bland tie-break among admissible rows: smallest basis index
        idx = rows[np.argmin(np.take(basis, rows))]
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
        # print("\n" + "="*80)
        # print("WARNING: NUMERICAL STABILITY ISSUE DETECTED")
        # print("="*80)
        
        # print(f"\nPIVOT OPERATION DETAILS:")
        # print(f"  • Pivot value: {pivval:.6e}")
        # print(f"  • Tolerance: {tol:.6e}")
        # print(f"  • Ratio (pivot/tolerance): {pivval/tol:.2f}")
        # print(f"  • Pivot row: {pivrow}")
        # print(f"  • Pivot column: {pivcol}")
        
        # print(f"\nTABLEAU INFORMATION:")
        # print(f"  • Tableau shape: {T.shape}")
        # print(f"  • Number of rows: {T.shape[0]}")
        # print(f"  • Number of columns: {T.shape[1]}")
        
        # print(f"\nBASIS INFORMATION:")
        # print(f"  • Current basis: {basis}")
        # print(f"  • Basis size: {len(basis)}")
        
        # print(f"\nPIVOT ELEMENT CONTEXT:")
        # print(f"  • Pivot element value: {pivval:.6e}")
        # print(f"  • Pivot row before normalization:")
        # pivot_row_before = T[pivrow] * pivval  # Reconstruct original row
        # print(f"    {pivot_row_before}")
        
        # print(f"\nNEARBY ELEMENTS (same row):")
        # for col in range(T.shape[1]):
        #     if col != pivcol:
        #         val = T[pivrow, col] * pivval  # Reconstruct original value
        #         if abs(val) > tol/10:  # Show elements close to tolerance
        #             print(f"    Column {col}: {val:.6e}")
        
        # print(f"\nNEARBY ELEMENTS (same column):")
        # for row in range(T.shape[0]):
        #     if row != pivrow:
        #         val = T[row, pivcol]
        #         if abs(val) > tol/10:  # Show elements close to tolerance
        #             print(f"    Row {row}: {val:.6e}")
        
        # print("="*80)
        # print("END OF WARNING")
        # print("="*80 + "\n")
        
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

def phase1_via_highs(A_std, b_std, c_std, c0=0.0, tol=1e-7):
    """Solve phase 1 via scipy HiGHS, return phase-2 tableau in project format.

    Uses linprog(c=0) to find a feasible vertex, extracts a basis of m LI columns
    preferring positive-x columns, and materializes T = [B^{-1}A | B^{-1}b;
    reduced_costs | c0 - c_B^T B^{-1} b] via sparse LU.
    """
    from scipy.optimize import linprog
    from scipy.linalg import qr as sp_qr
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu

    n_rows, n_cols = A_std.shape

    sol = linprog(
        c=np.zeros(n_cols), A_eq=A_std, b_eq=b_std,
        bounds=[(0, None)] * n_cols, method='highs',
    )
    if sol.status != 0:
        raise RuntimeError(f"HiGHS phase 1 failed: status={sol.status} ({sol.message})")

    x = np.maximum(sol.x, 0.0)
    pos_idx = np.where(x > tol)[0]
    zero_idx = np.where(x <= tol)[0]
    if pos_idx.size > n_rows:
        raise RuntimeError(f"HiGHS returned non-vertex: {pos_idx.size} > {n_rows}")

    needed = n_rows - pos_idx.size
    if needed > 0:
        if pos_idx.size > 0:
            Q_pos, _ = sp_qr(A_std[:, pos_idx], mode='economic')
            Mz = A_std[:, zero_idx]
            R = Mz - Q_pos @ (Q_pos.T @ Mz)
        else:
            R = A_std[:, zero_idx]
        _, _, piv = sp_qr(R, pivoting=True, mode='economic')
        basis = np.concatenate([pos_idx, zero_idx[piv[:needed]]]).astype(int)
    else:
        basis = pos_idx.astype(int)

    B_sparse = csc_matrix(A_std[:, basis])
    rhs = np.column_stack([A_std, b_std[:, None]])
    lu = splu(B_sparse)
    Binv_rhs = lu.solve(rhs)

    c_B = c_std[basis]
    reduced = c_std - c_B @ Binv_rhs[:, :-1]
    obj_slot = c0 - c_B @ Binv_rhs[:, -1]

    T = np.empty((n_rows + 1, n_cols + 1), dtype=np.float64)
    T[:n_rows, :] = Binv_rhs
    T[-1, :-1] = reduced
    T[-1, -1] = obj_slot

    return T, basis, int(sol.nit)


def change_to_zero_sum_direct_phase2(GameMatrix):
    """
    Convert a zero-sum game matrix directly to a canonical Phase 2 tableau
    without Phase 1 by constructing a trivial feasible BFS and then
    canonicalizing the tableau.
    Returns (T, basis, None).
    """
    m, n = GameMatrix.shape

    # Shift so B >= 0
    min_element = np.min(GameMatrix)
    K = max(0.0, -min_element + 1e-6)
    B = GameMatrix + K

    # Build standard-form A, b, c for:
    #  -B^T x + v + s = 0  (n rows)
    #   1^T x        = 1  (1 row)
    slack_matrix = np.eye(n)
    A_constraints = np.hstack([-B.T, np.ones((n, 1)), slack_matrix])     # n x (m+1+n)
    A_sum         = np.hstack([np.ones(m), np.zeros(1), np.zeros(n)])     # 1 x (m+1+n)
    A = np.vstack([A_constraints, A_sum])                                 # (n+1) x (m+1+n)

    b = np.zeros(n + 1)
    b[-1] = 1.0

    c = np.zeros(m + 1 + n)
    c[m] = -1.0   # minimize -v

    # Choose a provably feasible, nonsingular basis:
    #  - First n rows: take all slacks s_j basic (columns m+1 ... m+n)
    #  - Last row (sum): take x_0 basic (column 0)
    # This yields B = [[I_n, -B^T[:,0]]; [0,...,0, 1]], invertible,
    # and basic values x0=1, s = B^T[:,0] >= 0 (since B>=0).
    basis = np.empty(n + 1, dtype=int)
    basis[:n] = m + 1 + np.arange(n)  # all slacks
    basis[-1] = 0                     # x[0]

    # Canonicalize
    T, basis = build_phase2_tableau_canonical(A, b, c, basis)

    # No artificial variables in this path
    return T, basis, None


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

def build_trivial_bfs_zero_sum_game(GameMatrix):
    """
    Build a canonical Phase 2 tableau for the zero-sum game using the
    trivial pure strategy x = e_0 as a feasible BFS.
    Returns (T, basis).
    """
    m, n = GameMatrix.shape

    # Shift so B >= 0
    min_element = np.min(GameMatrix)
    K = max(0.0, -min_element + 1e-6)
    B = GameMatrix + K

    # Standard-form A, b, c as above
    slack_matrix = np.eye(n)
    A_constraints = np.hstack([-B.T, np.ones((n, 1)), slack_matrix])
    A_sum         = np.hstack([np.ones(m), np.zeros(1), np.zeros(n)])
    A = np.vstack([A_constraints, A_sum])

    b = np.zeros(n + 1)
    b[-1] = 1.0

    c = np.zeros(m + 1 + n)
    c[m] = -1.0

    # Same feasible, nonsingular basis: all slacks + x0
    basis = np.empty(n + 1, dtype=int)
    basis[:n] = m + 1 + np.arange(n)
    basis[-1] = 0

    # Canonicalize
    T, basis = build_phase2_tableau_canonical(A, b, c, basis)
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

