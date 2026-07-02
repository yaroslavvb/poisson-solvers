"""Classical preconditioners as callables ``z = M(r)`` approximating ``A^{-1} r``.

Each factory returns a function suitable for the ``M`` argument of
``pcg``/``flexible_pcg`` in :mod:`pcg`.
"""

import numpy as np
import scipy.sparse.linalg as spla


def identity():
    """Identity preconditioner (``M = I``, i.e. plain CG).

    Returns
    -------
    callable
        ``r -> r``.
    """
    return lambda r: r


def jacobi(A):
    """Jacobi (diagonal) preconditioner ``M = diag(A)^{-1}``.

    Note: for the constant-coefficient Dirichlet Laplacian
    (``poisson_2d``), ``diag(A)`` is the constant ``4/h^2``, so this is a
    positive scalar multiple of the identity and PCG convergence is
    iteration-identical to plain CG.

    Parameters
    ----------
    A : scipy.sparse matrix
        System matrix with strictly positive diagonal.

    Returns
    -------
    callable
        ``r -> r / diag(A)``.
    """
    inv_diag = 1.0 / A.diagonal()
    return lambda r: r * inv_diag


def ilu(A, **kw):
    """Incomplete LU preconditioner via :func:`scipy.sparse.linalg.spilu`.

    Parameters
    ----------
    A : scipy.sparse matrix
        System matrix (converted to CSC for the factorization).
    **kw
        Passed through to ``spilu`` (e.g. ``drop_tol``, ``fill_factor``).

    Returns
    -------
    callable
        ``r -> ILU_solve(r)`` returning float64 arrays.
    """
    fac = spla.spilu(A.tocsc(), **kw)
    return lambda r: fac.solve(np.asarray(r, dtype=np.float64))
