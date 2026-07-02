"""Randomized Nystrom preconditioner.

Implements the randomized Nystrom approximation (Algorithm 2.1) and the
Nystrom preconditioner (Eq. 5.3) of Frangella, Tropp & Udell,
"Randomized Nystrom Preconditioning", arXiv:2110.02820, for solving
``(A + mu*I) x = b`` with symmetric PSD ``A`` by preconditioned CG.

The approximation ``A_nys = U diag(lams) U^T`` satisfies
``0 <= A_nys <= A`` in the PSD order and ``lams[j] <= lambda_j(A)``
(their Lemma 2.1). Only matvecs with ``A`` are needed to build it; the
apply cost is ``O(n * rank)`` with no factorization of ``A``.

Honest expectation for this repo's benchmark: the 2-D Laplacian's spectrum
decays SLOWLY from the top (no spectral gap — eigenvalue counts grow like
the area of a quarter disk in frequency space), which is the adversarial
case for Nystrom preconditioning. The paper targets fast-decay /
regularized problems where the effective dimension d_eff(mu) is small
(their Theorem 5.1); here deflating ``rank`` modes out of ``n`` trims the
condition number only modestly, so expect modest speedups that grow with
rank rather than the O(1)-kappa behavior seen on ridge-regression spectra.
"""

import numpy as np
import scipy.linalg as sla


class NystromPreconditioner:
    """Randomized Nystrom preconditioner for ``(A + mu*I) x = b``.

    Construction follows Algorithm 2.1 of arXiv:2110.02820 exactly:

    1. ``Omega = randn(n, rank)``           (Gaussian test matrix)
    2. ``Omega = qr(Omega, 0)``             (thin QR — stabilizes the sketch)
    3. ``Y = A @ Omega``                    (``rank`` matvecs; only access to A)
    4. ``nu = eps(||Y||_F)``                (ulp-size stabilization shift)
    5. ``Y_nu = Y + nu * Omega``            (equivalent to sketching A + nu*I)
    6. ``C = chol(Omega^T Y_nu)``           (upper-triangular Cholesky)
    7. ``B = Y_nu C^{-1}``                  (triangular solve from the right)
    8. ``U, Sigma, _ = svd(B, 0)``          (thin SVD of B)
    9. ``lams = max(0, Sigma^2 - nu)``      (remove shift; clip keeps PSD)

    The preconditioner is Eq. (5.3) with ``lam_ell`` the smallest retained
    Nystrom eigenvalue::

        P^{-1} = (lam_ell + mu) U (Lam + mu I)^{-1} U^T + (I - U U^T)

    applied matrix-free as ``P^{-1} v = U (d * (U^T v)) + v`` with
    ``d_i = (lam_ell + mu)/(lams_i + mu) - 1``. The ``(lam_ell + mu)``
    scaling makes ``P`` act as the identity on ``range(U)^perp`` in a
    spectrally consistent way, mimicking the optimal rank-``ell``
    preconditioner (their Sec. 5.1); Proposition 5.3 bounds the resulting
    ``kappa(P^{-1/2} A_mu P^{-1/2})``.

    ``mu = 0`` guard: eigenvalue clipping in step 9 can produce exact zeros,
    for which Eq. (5.3) with ``mu = 0`` is singular. We DROP nonpositive
    Nystrom eigenvalues (reduce the effective rank) rather than floor them:
    dropping treats the unresolved directions as part of the orthogonal
    complement, on which ``P`` already acts as the identity — consistent
    with the structure of Eq. (5.3) — whereas flooring at ``nu`` would
    inject an arbitrary machine-precision scale into the spectrum. If every
    eigenvalue is dropped (``A_nys = 0`` and ``mu = 0``, outside the
    paper's theory) the preconditioner degrades to the identity.

    Parameters
    ----------
    A : scipy.sparse matrix
        Symmetric PSD system matrix (must support ``A @ dense_matrix``).
        Positive definite is required for a sound preconditioner at
        ``mu = 0``.
    rank : int
        Sketch size ``ell`` (number of columns of the test matrix), with
        ``1 <= rank <= n``. Theory (their Theorem 5.1) prescribes
        ``ell = 2*ceil(1.5*d_eff(mu)) + 1`` for E[kappa] < 28.
    mu : float, optional
        Regularization: the target system is ``(A + mu*I) x = b``
        (default 0.0, valid for positive definite ``A``).
    seed : int, optional
        Seed for the Gaussian test matrix (default 0).

    Attributes
    ----------
    U : numpy.ndarray, shape (n, rank)
        Orthonormal Nystrom eigenvector estimates (descending eigenvalues).
    lams : numpy.ndarray, shape (rank,)
        Nystrom eigenvalue estimates, descending; ``lams[j] <= lambda_j(A)``.
    mu : float
        Regularization parameter.
    rank_eff : int
        Retained eigenpairs after the ``mu = 0`` drop (== ``rank`` unless
        clipping produced zeros at ``mu = 0``).
    lam_ell : float
        Smallest retained eigenvalue, the ``lambda_hat_ell`` of Eq. (5.3)
        (0.0 in the degenerate all-dropped case).
    """

    def __init__(self, A, rank, mu=0.0, seed=0):
        n = A.shape[0]
        if not 1 <= rank <= n:
            raise ValueError(f"rank must be in [1, {n}], got {rank}")
        if mu < 0:
            raise ValueError(f"mu must be >= 0, got {mu}")

        rng = np.random.default_rng(seed)
        omega = rng.standard_normal((n, rank))          # step 1
        omega, _ = np.linalg.qr(omega)                  # step 2: thin QR
        y = A @ omega                                   # step 3: rank matvecs
        nu = np.spacing(np.linalg.norm(y, "fro"))       # step 4: eps(||Y||_F)

        # Steps 5-6, with the paper's fallback: if Cholesky of the core
        # matrix fails, increase nu and retry (Algorithm 2.1, remark on
        # step 6). lower=False consumes the upper triangle, matching
        # MATLAB's chol on the (numerically near-symmetric) core.
        for _ in range(4):
            y_nu = y + nu * omega                       # step 5
            try:
                c = sla.cholesky(omega.T @ y_nu, lower=False)  # step 6
                break
            except np.linalg.LinAlgError:
                nu *= 100.0
        else:
            raise np.linalg.LinAlgError(
                "Nystrom core matrix not positive definite even after "
                f"increasing the shift to nu={nu:.3e}; is A symmetric PSD?"
            )

        # Step 7: B = Y_nu C^{-1}  <=>  C^T B^T = Y_nu^T.
        b_mat = sla.solve_triangular(c, y_nu.T, trans="T", lower=False).T
        u, sigma, _ = np.linalg.svd(b_mat, full_matrices=False)  # step 8
        lams = np.maximum(0.0, sigma**2 - nu)           # step 9

        self.U = u
        self.lams = lams
        self.mu = float(mu)

        # mu = 0 guard (see class docstring): drop nonpositive eigenvalues.
        # sigma (hence lams) is descending, so positives form a prefix.
        k = rank if mu > 0 else int(np.count_nonzero(lams > 0.0))
        self.rank_eff = k
        if k == 0:
            # A_nys = 0 with mu = 0: theory does not apply; P^{-1} = I.
            self.lam_ell = 0.0
            self._u_act = None
            self._d = None
        else:
            self.lam_ell = float(lams[k - 1])
            self._u_act = u[:, :k]
            # d_i = (lam_ell + mu)/(lams_i + mu) - 1, cf. Eq. (5.3).
            self._d = (self.lam_ell + self.mu) / (lams[:k] + self.mu) - 1.0

    def apply(self, r):
        """Apply ``P^{-1}`` to a vector (Eq. 5.3 of arXiv:2110.02820).

        ``P^{-1} r = U [(lam_ell + mu)(Lam + mu I)^{-1} - I] U^T r + r``,
        costing ``O(n * rank_eff)``.

        Parameters
        ----------
        r : numpy.ndarray
            Vector of length ``n`` (float64).

        Returns
        -------
        numpy.ndarray
            ``P^{-1} r``, float64.
        """
        r = np.asarray(r, dtype=np.float64)
        if self.rank_eff == 0:
            return r.copy()
        w = self._u_act.T @ r
        return self._u_act @ (self._d * w) + r

    __call__ = apply
