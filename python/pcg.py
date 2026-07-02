"""Preconditioned conjugate gradient solvers.

``pcg`` is a faithful port of the Mathematica ``PCGSolve`` reference (the
classical Hestenes-Stiefel PCG with Fletcher-Reeves-type beta,
cf. Saad, "Iterative Methods for Sparse Linear Systems", 2nd ed.,
Algorithm 9.1). ``flexible_pcg`` replaces beta with the Polak-Ribiere
form of Notay, "Flexible Conjugate Gradients", SIAM J. Sci. Comput. 22(4),
2000, which tolerates nonlinear / nonsymmetric preconditioners such as a
neural operator.
"""

import numpy as np


def pcg(A, b, M=None, tol=1e-10, maxiter=2000):
    """Preconditioned conjugate gradient, ported from Mathematica ``PCGSolve``.

    State transition per iteration (identical to the reference)::

        Ap    = A p
        alpha = rz / (p . Ap)                       # Saad Alg. 9.1, line 3
        x    += alpha p
        r    -= alpha Ap
        z     = M(r)
        beta  = (r . z) / rz_old                    # Fletcher-Reeves form
        p     = z + beta p

    with initialization ``x0 = 0``, ``r0 = b``, ``p0 = z0 = M(b)``.

    Parameters
    ----------
    A : scipy.sparse matrix or numpy.ndarray
        Symmetric positive definite system matrix.
    b : numpy.ndarray
        Right-hand side.
    M : callable or None, optional
        Preconditioner ``z = M(r)`` approximating ``A^{-1} r``.
        ``None`` means identity (plain CG).
    tol : float, optional
        Stop when the relative residual ``||r_k|| / ||b|| <= tol``.
    maxiter : int, optional
        Maximum number of iterations.

    Returns
    -------
    x : numpy.ndarray
        Approximate solution.
    res_hist : list of float
        Relative residuals ``||r_k||/||b||`` starting with 1.0 and with one
        entry appended per iteration, mirroring the ``Sow``/``Reap`` history
        of the Mathematica reference. Iterations performed =
        ``len(res_hist) - 1``.
    """
    if M is None:
        M = lambda r: r  # noqa: E731 -- identity preconditioner

    b = np.asarray(b, dtype=np.float64)
    bnorm = np.linalg.norm(b)
    res_hist = [1.0]

    x = np.zeros_like(b)
    r = b.copy()
    z = M(r)
    p = z.copy()
    rz = r @ z

    for _ in range(maxiter):
        Ap = A @ p
        alpha = rz / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        relres = np.linalg.norm(r) / bnorm
        res_hist.append(relres)
        z = M(r)
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
        if relres <= tol:
            break

    return x, res_hist


def flexible_pcg(A, b, M, tol=1e-10, maxiter=2000):
    """Flexible PCG with Polak-Ribiere beta (Notay 2000).

    Identical to :func:`pcg` except for the search-direction update::

        beta = z_{k+1} . (r_{k+1} - r_k) / (z_k . r_k)     # Notay (2000)

    (the Polak-Ribiere formula), which retains local orthogonality when the
    preconditioner ``M`` varies between iterations — e.g. a nonlinear or
    nonsymmetric neural preconditioner — at the cost of one extra stored
    vector.

    Parameters
    ----------
    A : scipy.sparse matrix or numpy.ndarray
        Symmetric positive definite system matrix.
    b : numpy.ndarray
        Right-hand side.
    M : callable
        (Possibly nonlinear) preconditioner ``z = M(r)``.
    tol : float, optional
        Stop when ``||r_k|| / ||b|| <= tol``.
    maxiter : int, optional
        Maximum number of iterations.

    Returns
    -------
    x : numpy.ndarray
        Approximate solution.
    res_hist : list of float
        Relative residuals starting with 1.0, one entry per iteration.
    """
    b = np.asarray(b, dtype=np.float64)
    bnorm = np.linalg.norm(b)
    res_hist = [1.0]

    x = np.zeros_like(b)
    r = b.copy()
    z = M(r)
    p = z.copy()
    rz = r @ z

    for _ in range(maxiter):
        Ap = A @ p
        alpha = rz / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        relres = np.linalg.norm(r_new) / bnorm
        res_hist.append(relres)
        z_new = M(r_new)
        beta = (z_new @ (r_new - r)) / rz  # Polak-Ribiere (Notay 2000)
        p = z_new + beta * p
        r = r_new
        rz = r @ z_new
        if relres <= tol:
            break

    return x, res_hist
