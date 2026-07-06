"""Discrete Poisson operators and Gaussian-random-field right-hand sides.

Faithful Python port of the Mathematica reference construction::

    n = 32; h = 1.0/(n + 1);
    d1 = SparseArray[{Band[{1,1}]->2., Band[{2,1}]->-1., Band[{1,2}]->-1.}, {n,n}];
    A  = (KroneckerProduct[d1, id] + KroneckerProduct[id, d1])/h^2;

All matrices act on grid functions flattened in row-major (C) order, which
matches Mathematica's ``Flatten``: the flat index of grid node ``(i, j)`` is
``k = i*n + j`` with the first axis varying slowest.
"""

import numpy as np
import scipy.sparse as sp


def laplacian_1d(n):
    """1-D discrete Laplacian stencil matrix (no ``1/h^2`` scaling).

    Tridiagonal matrix with 2 on the diagonal and -1 on the two
    off-diagonals — the ``[-1, 2, -1]`` second-difference stencil for a
    homogeneous Dirichlet problem on ``n`` interior nodes.

    Parameters
    ----------
    n : int
        Number of interior grid points.

    Returns
    -------
    scipy.sparse.csr_matrix
        ``n x n`` symmetric positive definite tridiagonal matrix.
    """
    d = sp.diags(
        [-np.ones(n - 1), 2.0 * np.ones(n), -np.ones(n - 1)],
        offsets=[-1, 0, 1],
        format="csr",
    )
    return d


def poisson_2d(n):
    """2-D discrete Laplacian on the unit square with Dirichlet BCs.

    Standard 5-point finite-difference discretization of ``-Delta u = f``
    on an ``n x n`` interior grid with mesh width ``h = 1/(n+1)``::

        A = (kron(d1, I) + kron(I, d1)) / h^2

    mirroring the Mathematica ``KroneckerProduct`` construction. ``A`` is
    symmetric positive definite with constant diagonal ``4/h^2``.

    Parameters
    ----------
    n : int
        Interior grid points per dimension; the system has size ``n^2``.

    Returns
    -------
    scipy.sparse.csr_matrix
        ``n^2 x n^2`` SPD matrix.
    """
    h = 1.0 / (n + 1)
    d1 = laplacian_1d(n)
    eye = sp.identity(n, format="csr")
    a = (sp.kron(d1, eye) + sp.kron(eye, d1)) / h**2
    return a.tocsr()


def variable_poisson_2d(n, contrast=100.0):
    """Variable-coefficient diffusion operator ``-div(a grad u)``.

    5-point finite-volume discretization on the unit square with Dirichlet
    boundary conditions, coefficient::

        a(x, y) = 1         for x < 0.5
        a(x, y) = contrast  for x >= 0.5

    Grid nodes are ``x_i = i*h``, ``y_j = j*h`` with ``h = 1/(n+1)`` and
    flat index ``k = i*n + j`` (x along the slow axis). Face transmissibility
    in x uses the harmonic mean of the adjacent nodal coefficients,
    ``w_{i+1/2} = 2 a_i a_{i+1} / (a_i + a_{i+1})``, which preserves flux
    continuity across the material interface (cf. LeVeque, "Finite Difference
    Methods for ODEs and PDEs", SIAM 2007, Sec. 2.15). Since ``a`` depends
    only on x, the y-direction face coefficient at node ``(i, j)`` is just
    ``a_i`` and the operator assembles as::

        A = ( kron(Tx, I) + kron(diag(a), d1) ) / h^2

    where ``Tx`` is the harmonic-mean tridiagonal in x and ``d1`` the
    constant-coefficient stencil from :func:`laplacian_1d`.

    The high-contrast jump makes ``diag(A)`` strongly non-constant, so a
    Jacobi preconditioner is no longer a scalar multiple of the identity
    (unlike :func:`poisson_2d`).

    Parameters
    ----------
    n : int
        Interior grid points per dimension.
    contrast : float, optional
        Coefficient value in the half-domain ``x >= 0.5`` (default 100.0).

    Returns
    -------
    scipy.sparse.csr_matrix
        ``n^2 x n^2`` SPD matrix.
    """
    h = 1.0 / (n + 1)
    x = (np.arange(1, n + 1)) * h
    a_nodes = np.where(x < 0.5, 1.0, contrast)

    # Face transmissibilities in x: n+1 faces; boundary faces take the
    # adjacent nodal value (a is constant near each boundary).
    w = np.empty(n + 1)
    w[0] = a_nodes[0]
    w[-1] = a_nodes[-1]
    w[1:-1] = 2.0 * a_nodes[:-1] * a_nodes[1:] / (a_nodes[:-1] + a_nodes[1:])

    tx = sp.diags(
        [-w[1:-1], w[:-1] + w[1:], -w[1:-1]],
        offsets=[-1, 0, 1],
        format="csr",
    )
    eye = sp.identity(n, format="csr")
    a = (sp.kron(tx, eye) + sp.kron(sp.diags(a_nodes), laplacian_1d(n))) / h**2
    return a.tocsr()


def grf_rhs(n, alpha=2.0, tau=3.0, seed=42):
    """Gaussian random field right-hand side, standardized and flattened.

    Samples a mean-zero GRF with Matern-like spectral density
    ``(|k|^2 + tau^2)^(-alpha/2)`` — i.e. covariance ``(-Delta + tau^2 I)^(-alpha)``
    up to normalization, as used for Poisson benchmark data in Li et al.,
    "Fourier Neural Operator for Parametric PDEs" (ICLR 2021). Mirrors the
    Mathematica pipeline::

        freqs    = RotateRight[Range[-n/2, n/2-1], n/2] * n
        spectrum = 1/(Outer[Plus, freqs^2, freqs^2] + tau^2)^(alpha/2)
        SeedRandom[42];
        noise    = RandomVariate[NormalDistribution[], {n,n,2}] . {1, I}
        grfRaw   = Re@InverseFourier[noise*spectrum, FourierParameters -> {-1,-1}]
        b        = Standardize@Flatten@grfRaw

    Notes
    -----
    * NumPy's PCG64 generator cannot bit-match Mathematica's ``SeedRandom[42]``
      stream; this replicates the distribution and pipeline exactly, not the
      individual draws.
    * Mathematica ``InverseFourier`` with ``FourierParameters -> {-1,-1}``
      applies no prefactor, while ``np.fft.ifft2`` divides by ``n^2``; both
      use the ``e^{+2*pi*i k.x}`` kernel, so the transforms differ only by the
      overall constant ``n^2``, which the final standardization (mean 0,
      sample std 1) removes.
    * ``Standardize`` uses the sample standard deviation, hence ``ddof=1``.

    Parameters
    ----------
    n : int
        Grid points per dimension (even, matching ``Range[-n/2, n/2-1]``).
    alpha : float, optional
        Spectral decay exponent (default 2.0).
    tau : float, optional
        Inverse length-scale / mass parameter (default 3.0).
    seed : int, optional
        Seed for ``np.random.default_rng`` (default 42).

    Returns
    -------
    numpy.ndarray
        Shape ``(n*n,)``, mean 0, sample std 1, row-major flattening.
    """
    # RotateRight[Range[-n/2, n/2-1], n/2] * n -> [0, 1, ..., n/2-1, -n/2, ..., -1] * n
    f = np.roll(np.arange(-n // 2, n // 2), n // 2) * n
    spectrum = 1.0 / (f[:, None] ** 2 + f[None, :] ** 2 + tau**2) ** (alpha / 2.0)

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, n, 2)) @ np.array([1.0, 1.0j])
    field = np.real(np.fft.ifft2(noise * spectrum))

    flat = field.ravel()  # row-major, matching Mathematica Flatten
    return (flat - flat.mean()) / flat.std(ddof=1)
