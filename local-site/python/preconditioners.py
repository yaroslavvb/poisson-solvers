"""Classical preconditioners as callables ``z = M(r)`` approximating ``A^{-1} r``.

Each factory returns a function suitable for the ``M`` argument of
``pcg``/``flexible_pcg`` in :mod:`pcg`.

Also hosts the shared dense building blocks :func:`ic0` (zero-fill incomplete
Cholesky, the truncated-regression/Vecchia factor of reports 09/10/11) and
:func:`block_average_matrix` (piecewise-constant coarse regressors), factored
out of ``experiments/grid_regressions_multiscale.py`` so that
``experiments/richardson_ar.py`` reuses the identical implementations.
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


def ic0(Ad):
    """Zero-fill incomplete Cholesky: L kept on tril(A)'s sparsity pattern.

    Standard IC(0) recurrence (Saad, Iterative Methods, 2nd ed., Sec. 10.3.5),
    computed dense here for inspectability:

        L[i,j] = (A[i,j] - sum_{m<j} L[i,m] L[j,m]) / L[j,j]   (i,j) in pattern
        L[i,i] = sqrt(A[i,i] - sum_{m<i} L[i,m]^2)

    Statistical reading (reports 09/10): exact chol(A) whitens u ~ N(0,A^{-1})
    by regressing each u_i on ALL its successors in the elimination order,
    with coefficients -L[j,i]/L[i,i] spread over the whole bandwidth-n
    wavefront. IC(0) truncates that regression to the stencil successors only
    (the tril(A) pattern) -- exactly a Vecchia approximation of the Gaussian
    field: keep p(u_i | stencil subset) instead of the full conditional.
    Schafer, Katzfuss & Owhadi, "Sparse Cholesky factorization by
    Kullback-Leibler minimization" (SIAM J. Sci. Comput. 43(3), 2021) show the
    covariance-side truncated regression is the KL-optimal factor on the
    pattern; grid_regressions_multiscale.py part 5 measures how close IC(0)
    gets to it on the grid (they coincide exactly on the 1-D chain, where
    there is no fill-in).
    """
    N = Ad.shape[0]
    pattern = np.tril(Ad) != 0.0
    L = np.zeros_like(Ad)
    for i in range(N):
        for j in np.nonzero(pattern[i, : i + 1])[0]:  # ascending, ends at j=i
            s = Ad[i, j] - L[i, :j] @ L[j, :j]
            if j == i:
                L[i, i] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return L


def block_average_matrix(n, bs):
    """N x (n/bs)^2 block-average matrix: column (bi,bj) puts 1/bs^2 on its
    bs x bs block (row-major coarse index bi*(n//bs)+bj)."""
    nb = n // bs
    Z = np.zeros((n * n, nb * nb))
    for bi in range(nb):
        for bj in range(nb):
            col = bi * nb + bj
            for di in range(bs):
                for dj in range(bs):
                    Z[(bi * bs + di) * n + (bj * bs + dj), col] = 1.0 / bs**2
    return Z
