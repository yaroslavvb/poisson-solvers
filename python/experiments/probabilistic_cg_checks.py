"""Numerical verification for the probabilistic-CG explainer (CG as Gaussian inference).

Machine-checks every claim and number for the page that fuses the ICML 2026
probabilistic-numerics tutorial's linear-solver formulation (Hennig, Pfoertner,
Weiland, "Probabilistic Numerics", ICML Seoul 06/07/2026, pp. 7-20) with
Shewchuk's deterministic geometry ("An Introduction to the Conjugate Gradient
Method Without the Agonizing Pain", 1994, SS3/SS6/SS7-8).

Notation (slides pp. 9-14): prior p(x) = N(mu_0 = 0, Sigma_0); an action s_i
observes bbar_i = s_i^T b = y_i^T x with y_i := A^T s_i (a delta likelihood).
Rank-one conditioning:

    v_i  = Sigma_{i-1} y_i / sqrt(y_i^T Sigma_{i-1} y_i)
    xi_i = (bbar_i - y_i^T mu_{i-1}) / sqrt(y_i^T Sigma_{i-1} y_i)
    mu_i = mu_{i-1} + xi_i v_i        Sigma_i = Sigma_{i-1} - v_i v_i^T

Proposition (slides pp. 13-14): the v_i are orthonormal in the Mahalanobis
inner product <u,v>_{Sigma_0} = u^T Sigma_0^{-1} v -- iterative conditioning
implements Gram-Schmidt on the {y_i} in (R^D, <.,.>_{Sigma_0}), so
Sigma_i = Sigma_0 - sum_j v_j v_j^T and mu_i = sum_j xi_j v_j.

Two arenas:
  * Shewchuk's 2x2: A = [[3,2],[2,6]], b = (2,-8), x* = (2,-2), eigs {2,7}.
    The sqrt cancels inside the mu/Sigma updates, so the whole conditioning
    sequence is RATIONAL and is run in exact Fractions.
  * The suite's Dirichlet chain: A = laplacian_1d(32)/h^2, h = 1/33, with the
    green-tents heater/chiller rhs: +2 W at node 7 (s = 7/33) and -1.5 W at
    node 25 (s = 25/33), entries f/h (green-tents SS superposition sources).

Checks:
 1. sequential rank-one conditioning == batch Gaussian conditioning
    (2x2 exact Fractions; chain to float precision; Sigma_0 in {I, A^{-1}}).
 2. Gram-Schmidt proposition: <v_i,v_j>_{Sigma_0} = delta_ij, Sigma_i =
    Sigma_0 - V V^T, mu_i = V xi  (both arenas x both priors).
 3. THE HEADLINE (slides' Corner Case 2 [Dennis Jr & Turner 1987; Cockayne,
    Oates, Ipsen, Girolami 2019 BayesCG]): prior Sigma_0 = A^{-1} + greedy
    actions s_n = r_{n-1} ==> mu_n is the CG iterate of python/pcg.py,
    iterate-for-iterate; v_n = p_n/sqrt(p_n^T A p_n); conjugacy == Mahalanobis
    orthonormality. Exact on the 2x2 (mu_1 = (34/83,-136/83), mu_2 = x*).
 4. uncertainty accounting (prior A^{-1}): tr(A Sigma_n) = D - n exactly;
    E||x-mu_n||_A^2 = tr(A Sigma_n) (Monte Carlo over the prior, fixed
    actions); S_n^T(A mu_n - b) = 0 (explored directions residual-free,
    Wenger et al. NeurIPS 2022); actual CG A-norm error vs the posterior band.
 5. Corner Case 1 bridge [Hennig SIOPT 2015]: coordinate actions s_n = e_n
    with prior A^{-1} ==> R_ij = v_i^T y_j is upper triangular with
    R^T R = A and R^T == chol(A): conditioning = Cholesky
    (update_views_explainer's Shawe-Taylor section).
 6. policy race (greedy / coordinate / fixed random, same prior A^{-1}):
    tr(A Sigma_k) drops by exactly 1 per step for ALL policies (identical
    uncertainty budget) while the A-norm ERROR curves differ wildly --
    greedy spends the same budget where THIS b lives.
 7. posterior-ellipse widget data (2x2): prior 1-sigma ellipse of N(0,A^{-1}),
    the degenerate segment after observation 1 (exact Sigma_1, x* ON the
    segment line), the point after observation 2; CG path + 8-step steepest-
    descent zigzag from the same x0 = 0 (Shewchuk SS6).
 8. calibration: K = 200 draws x ~ N(0, A^{-1}), 5 fixed actions:
    (x - mu_5)^T Sigma_5^+ (x - mu_5) ~ chi^2_{27} (mean + 90% coverage).

Run from the repo root:
    uv run python python/experiments/probabilistic_cg_checks.py

Expected output: all lines PASS; writes results/probabilistic_cg.json with
every number and trajectory the report quotes.
"""
import json
import math
import os
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcg import pcg
from poisson import laplacian_1d

np.set_printoptions(precision=6, suppress=True)
results = {"checks": []}


def ok(name, cond, err=None):
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {name}" + (f"   (max err {err:.3g})" if err is not None else ""))
    results["checks"].append({"name": name, "pass": bool(cond),
                              **({"max_err": float(err)} if err is not None else {})})


# =============================================================================
# the probabilistic linear solver (slides pp. 9-12), float version
# =============================================================================
def prob_solve(A, b, Sigma0, actions, nsteps, stop_rtol=0.0):
    """Sequential rank-one conditioning. `actions(k, mu, Sigma, r)` -> s_k."""
    D = len(b)
    mu = np.zeros(D)
    Sigma = Sigma0.copy()
    mus, Sigmas, V, xis, S_used = [mu.copy()], [Sigma.copy()], [], [], []
    bnorm = np.linalg.norm(b)
    for k in range(nsteps):
        r = b - A @ mu
        if stop_rtol and np.linalg.norm(r) / bnorm < stop_rtol:
            break
        s = actions(k, mu, Sigma, r)
        y = A.T @ s
        Sy = Sigma @ y
        den = float(y @ Sy)
        if den <= 0:
            break
        sq = math.sqrt(den)
        v = Sy / sq
        xi = float(s @ b - y @ mu) / sq
        mu = mu + xi * v
        Sigma = Sigma - np.outer(v, v)
        mus.append(mu.copy()); Sigmas.append(Sigma.copy())
        V.append(v); xis.append(xi); S_used.append(np.asarray(s, float))
    return {"mus": mus, "Sigmas": Sigmas, "V": np.array(V).T if V else np.zeros((D, 0)),
            "xi": np.array(xis), "S": np.array(S_used).T if S_used else np.zeros((D, 0))}


def batch_posterior(A, b, Sigma0, S):
    """One-shot Gaussian conditioning on S^T b = S^T A x (slides p. 11)."""
    Y = A.T @ S
    W = Sigma0 @ Y
    G = Y.T @ W                                # y_i^T Sigma_0 y_j
    K = np.linalg.solve(G, W.T)
    return W @ np.linalg.solve(G, S.T @ b), Sigma0 - W @ K


# =============================================================================
# exact-Fraction linear algebra for the 2x2 arena
# =============================================================================
def fvec(v): return [Fraction(x) for x in v]
def fmat(M): return [[Fraction(x) for x in row] for row in M]
def fdot(u, v): return sum(a * c for a, c in zip(u, v))
def fmatvec(M, v): return [fdot(row, v) for row in M]
def fsub(M, N): return [[a - c for a, c in zip(r1, r2)] for r1, r2 in zip(M, N)]
def fscale_outer(u, w, den): return [[ui * wj / den for wj in w] for ui in u]
def finv2(M):
    det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    return [[M[1][1] / det, -M[0][1] / det], [-M[1][0] / det, M[0][0] / det]]
def fstr(x): return str(x)  # Fraction -> "p/q"


def prob_step_exact(A, b, mu, Sigma, s):
    """One rank-one conditioning step, rational form (the sqrt cancels)."""
    y = fmatvec(A, s)                          # A symmetric: A^T s = A s
    Sy = fmatvec(Sigma, y)
    den = fdot(y, Sy)
    xi_over_sq = (fdot(s, b) - fdot(y, mu)) / den
    mu2 = [m + xi_over_sq * w for m, w in zip(mu, Sy)]
    Sigma2 = fsub(Sigma, fscale_outer(Sy, Sy, den))
    return mu2, Sigma2, Sy, den


# =============================================================================
# arenas
# =============================================================================
# -- Shewchuk 2x2 (exact) --
A2f = fmat([[3, 2], [2, 6]])
b2f = fvec([2, -8])
A2 = np.array([[3.0, 2.0], [2.0, 6.0]])
b2 = np.array([2.0, -8.0])
S02f = finv2(A2f)                              # prior covariance A^{-1}, exact
xstar2f = fmatvec(S02f, b2f)
ev2 = sorted(np.linalg.eigvalsh(A2))
ok("Shewchuk arena: x* == (2,-2) exact and eig(A) == {2, 7}",
   xstar2f == [Fraction(2), Fraction(-2)]
   and abs(ev2[0] - 2) < 1e-12 and abs(ev2[1] - 7) < 1e-12)

# -- Dirichlet chain n = 32, green-tents heater/chiller rhs --
n = 32
h = 1.0 / (n + 1)
Ad = (laplacian_1d(n) / h**2).toarray()
invA = np.linalg.inv(Ad)
j_hot, j_cold, f_hot, f_cold = 6, 24, 2.0, -1.5   # nodes 7 and 25 (1-based)
b = np.zeros(n)
b[j_hot] = f_hot / h
b[j_cold] = f_cold / h
x_star = np.linalg.solve(Ad, b)
xstar_Anorm = math.sqrt(x_star @ Ad @ x_star)

# =============================================================================
# 1. sequential rank-one conditioning == batch conditioning
# =============================================================================
# exact, 2x2, rational actions s1 = (1,2), s2 = (3,-1), both priors
for prior_name, S0f in (("I", fmat([[1, 0], [0, 1]])), ("A^{-1}", S02f)):
    mu, Sig = [Fraction(0), Fraction(0)], [row[:] for row in S0f]
    seq = []
    for s in (fvec([1, 2]), fvec([3, -1])):
        mu, Sig, _, _ = prob_step_exact(A2f, b2f, mu, Sig, s)
        seq.append(([m for m in mu], [row[:] for row in Sig]))
    exact_all = True
    for k, Scols in enumerate(([fvec([1, 2])], [fvec([1, 2]), fvec([3, -1])])):
        Y = [fmatvec(A2f, s) for s in Scols]                    # list of y_i
        W = [fmatvec(S0f, y) for y in Y]                        # Sigma0 y_i
        G = [[fdot(Y[i], W[j]) for j in range(k + 1)] for i in range(k + 1)]
        Ginv = finv2(G) if k == 1 else [[1 / G[0][0]]]
        rhs = [fdot(s, b2f) for s in Scols]
        coef = [sum(Ginv[i][j] * rhs[j] for j in range(k + 1)) for i in range(k + 1)]
        mu_b = [sum(coef[i] * W[i][d] for i in range(k + 1)) for d in range(2)]
        Sig_b = [[S0f[a][c] - sum(W[i][a] * Ginv[i][j] * W[j][c]
                                  for i in range(k + 1) for j in range(k + 1))
                  for c in range(2)] for a in range(2)]
        exact_all &= (mu_b == seq[k][0]) and (Sig_b == seq[k][1])
    ok(f"2x2 exact Fractions: sequential v_i/xi_i == batch conditioning after 1 and 2 obs "
       f"(Sigma_0 = {prior_name})", exact_all)

# float, chain, 8 random actions, both priors
rng = np.random.default_rng(7)
S8 = rng.standard_normal((n, 8))
batch_errs = {}
for prior_name, S0 in (("I", np.eye(n)), ("A^{-1}", invA)):
    run = prob_solve(Ad, b, S0, lambda k, mu, Sig, r: S8[:, k], 8)
    err = 0.0
    for k in (4, 8):
        mu_b, Sig_b = batch_posterior(Ad, b, S0, S8[:, :k])
        err = max(err,
                  np.abs(run["mus"][k] - mu_b).max() / np.abs(x_star).max(),
                  np.abs(run["Sigmas"][k] - Sig_b).max() / np.abs(S0).max())
    batch_errs[prior_name] = err
    ok(f"chain: sequential == batch conditioning after 4 and 8 random actions "
       f"(Sigma_0 = {prior_name}, rel)", err < 1e-12, err)

# =============================================================================
# 2. Gram-Schmidt proposition (slides pp. 13-14), both arenas x both priors
# =============================================================================
Srand_full = rng.standard_normal((n, n))
S2rand = rng.standard_normal((2, 2))
gs_errs = {}
for arena, A_, b_, S0s, Sacts, D in (
        ("2x2", A2, b2, (("I", np.eye(2)), ("A^{-1}", np.linalg.inv(A2))), S2rand, 2),
        ("chain", Ad, b, (("I", np.eye(n)), ("A^{-1}", invA)), Srand_full, n)):
    for prior_name, S0 in S0s:
        run = prob_solve(A_, b_, S0, lambda k, mu, Sig, r: Sacts[:, k], D)
        V, xi = run["V"], run["xi"]
        S0inv = np.linalg.inv(S0)
        e_orth = np.abs(V.T @ S0inv @ V - np.eye(V.shape[1])).max()
        e_sig = max(np.abs(run["Sigmas"][i] - (S0 - V[:, :i] @ V[:, :i].T)).max()
                    for i in range(D + 1)) / np.abs(S0).max()
        e_mu = max(np.abs(run["mus"][i] - V[:, :i] @ xi[:i]).max()
                   for i in range(D + 1)) / np.abs(run["mus"][-1]).max()
        err = max(e_orth, e_sig, e_mu)
        gs_errs[f"{arena}, Sigma_0={prior_name}"] = {
            "orthonormality": e_orth, "Sigma_decomp": e_sig, "mu_decomp": e_mu}
        ok(f"Gram-Schmidt proposition on {arena}, Sigma_0 = {prior_name}: "
           f"<v_i,v_j>_Sigma0 == delta_ij, Sigma_i == Sigma_0 - VV^T, mu_i == V xi",
           err < 1e-9, err)

# =============================================================================
# 3. THE HEADLINE: prior A^{-1} + greedy residual actions == CG (Corner Case 2)
# =============================================================================
# reference CG iterates from the house solver
x_hist = []
x_cg, res_hist = pcg(Ad, b, tol=1e-16, maxiter=n, x_hist=x_hist)
n_cg = len(x_hist) - 1

greedy = prob_solve(Ad, b, invA, lambda k, mu, Sig, r: r, n, stop_rtol=1e-13)
n_greedy = len(greedy["mus"]) - 1
kmax = min(n_cg, n_greedy)
err_it = max(np.abs(greedy["mus"][k] - x_hist[k]).max() for k in range(kmax + 1))
err_it /= np.abs(x_star).max()
ok(f"HEADLINE: greedy actions s_n = r_n-1 + prior A^-1 ==> mu_n == pcg iterate x_n, "
   f"n = 0..{kmax} (chain, rel)", err_it < 1e-10, err_it)

# v_n == p_n / sqrt(p_n^T A p_n): CG directions from iterate differences
Vcg = np.zeros((n, kmax))
for k in range(1, kmax + 1):
    d = x_hist[k] - x_hist[k - 1]              # alpha_k p_k, alpha_k > 0
    Vcg[:, k - 1] = d / math.sqrt(d @ Ad @ d)
err_v = np.abs(greedy["V"][:, :kmax] - Vcg).max()
ok("HEADLINE: v_n == p_n / sqrt(p_n^T A p_n) -- the A-normalized CG search directions, all n",
   err_v < 1e-8, err_v)

# conjugacy == Mahalanobis orthonormality: the SAME matrix in two vocabularies
Gcg = Vcg.T @ Ad @ Vcg                         # Shewchuk's p_i^T A p_j, normalized
Gpn = greedy["V"][:, :kmax].T @ Ad @ greedy["V"][:, :kmax]   # slides' <v_i,v_j>_Sigma0
err_same = np.abs(Gcg - Gpn).max()
err_conj = np.abs(Gcg - np.eye(kmax)).max()
ok("HEADLINE: conjugacy p_i^T A p_j = 0  ==  Mahalanobis <v_i,v_j>_Sigma0 = delta_ij "
   "(same numbers, two vocabularies)", max(err_same, err_conj) < 1e-8,
   max(err_same, err_conj))

# ---- exact 2x2: CG in Fractions, then the probabilistic solver in Fractions --
r0 = b2f
p1 = r0[:]
Ap1 = fmatvec(A2f, p1)
alpha1 = fdot(r0, r0) / fdot(p1, Ap1)                        # 17/83
x1 = [alpha1 * p for p in p1]                                # (34/83, -136/83)
r1 = [r - alpha1 * ap for r, ap in zip(r0, Ap1)]             # (336/83, 84/83)
beta2 = fdot(r1, r1) / fdot(r0, r0)                          # 1764/6889
p2 = [r + beta2 * p for r, p in zip(r1, p1)]
Ap2 = fmatvec(A2f, p2)
alpha2 = fdot(r1, r1) / fdot(p2, Ap2)                        # 83/238
x2 = [x + alpha2 * p for x, p in zip(x1, p2)]
ok("2x2 exact CG: alpha_1 = 17/83, x_1 = (34/83, -136/83), r_1 = (336/83, 84/83), "
   "beta_2 = 1764/6889, alpha_2 = 83/238, x_2 == x* == (2,-2)",
   alpha1 == Fraction(17, 83) and x1 == [Fraction(34, 83), Fraction(-136, 83)]
   and r1 == [Fraction(336, 83), Fraction(84, 83)] and beta2 == Fraction(1764, 6889)
   and alpha2 == Fraction(83, 238) and x2 == [Fraction(2), Fraction(-2)])

muf, Sigf = [Fraction(0), Fraction(0)], [row[:] for row in S02f]
rf = [bi - v for bi, v in zip(b2f, fmatvec(A2f, muf))]
mu1f, Sig1f, Sy1, den1 = prob_step_exact(A2f, b2f, muf, Sigf, rf)   # obs 1
rf1 = [bi - v for bi, v in zip(b2f, fmatvec(A2f, mu1f))]
mu2f, Sig2f, Sy2, den2 = prob_step_exact(A2f, b2f, mu1f, Sig1f, rf1)  # obs 2
cross1 = Sy1[0] * p1[1] - Sy1[1] * p1[0]
cross2 = Sy2[0] * p2[1] - Sy2[1] * p2[0]
ok("2x2 exact probabilistic: mu_1 == x_1, mu_2 == x* == (2,-2), Sigma_2 == 0, and "
   "Sigma_(n-1) y_n parallel to p_n at both steps (exact Fractions)",
   mu1f == x1 and mu2f == [Fraction(2), Fraction(-2)]
   and Sig2f == [[Fraction(0)] * 2] * 2 and cross1 == 0 and cross2 == 0)

Sig1_expect = fmat([[Fraction(242, 581), Fraction(-55, 581)],
                    [Fraction(-55, 581), Fraction(25, 1162)]])
ok("2x2 exact: Sigma_1 == (1/1162) [[484,-110],[-110,25]] == (509/1162) uu^T, "
   "u = (22,-5)/sqrt(509)", Sig1f == Sig1_expect)
tr0 = fdot([A2f[0][0], A2f[0][1]], [S02f[0][0], S02f[1][0]]) \
    + fdot([A2f[1][0], A2f[1][1]], [S02f[0][1], S02f[1][1]])
trA = lambda S: sum(A2f[i][0] * S[0][i] + A2f[i][1] * S[1][i] for i in range(2))
ok("2x2 exact: tr(A Sigma_n) == 2, 1, 0 for n = 0, 1, 2 (one unit per observation)",
   trA(S02f) == 2 and trA(Sig1f) == 1 and trA(Sig2f) == 0)

# =============================================================================
# 4. uncertainty accounting on the chain (prior A^{-1})
# =============================================================================
traces_greedy = [float(np.trace(Ad @ S)) for S in greedy["Sigmas"]]
err_tr = max(abs(traces_greedy[k] - (n - k)) for k in range(len(traces_greedy)))
ok("tr(A Sigma_n) == 32 - n for every n (greedy): one unit of A-weighted "
   "uncertainty removed per observation", err_tr < 1e-8, err_tr)

# E ||x - mu_n||_A^2 == tr(A Sigma_n): Monte Carlo over the prior, FIXED actions
K_mc = 2000
L_chol = np.linalg.cholesky(Ad)
rng_mc = np.random.default_rng(3)
X = np.linalg.solve(L_chol.T, rng_mc.standard_normal((n, K_mc))).T  # rows ~ N(0, A^{-1})
S5 = rng_mc.standard_normal((n, 5))
err_mc = 0.0
mc_table = []
for k in range(6):
    if k == 0:
        MU = np.zeros_like(X)
    else:
        Y = Ad @ S5[:, :k]
        G = Y.T @ invA @ Y
        P = invA @ Y @ np.linalg.solve(G, Y.T)  # mu_k = P x (since S^T b = Y^T x)
        MU = X @ P.T
    E = X - MU
    mean_sq = float(np.einsum('ij,jk,ik->i', E, Ad, E).mean())
    mc_table.append({"k": k, "mc_mean": mean_sq, "trace": float(n - k)})
    err_mc = max(err_mc, abs(mean_sq - (n - k)) / (n - k))
ok("E ||x - mu_n||_A^2 == tr(A Sigma_n) over the prior (K = 2000 draws, fixed "
   "actions, n = 0..5, rel tol 5%)", err_mc < 0.05, err_mc)

# explored directions are residual-free: S_n^T (A mu_n - b) == 0 (all policies)
def residual_free_err(run):
    e = 0.0
    for k in range(1, len(run["mus"])):
        Sk = run["S"][:, :k]
        rk = Ad @ run["mus"][k] - b
        colnorm = np.linalg.norm(Sk, axis=0)
        e = max(e, np.abs(Sk.T @ rk).max() / (colnorm.max() * np.linalg.norm(b)))
    return e

coord = prob_solve(Ad, b, invA, lambda k, mu, Sig, r: np.eye(n)[k], n)
rng_pol = np.random.default_rng(11)
Sfix = rng_pol.standard_normal((n, n))
randp = prob_solve(Ad, b, invA, lambda k, mu, Sig, r: Sfix[:, k], n)
err_rf = max(residual_free_err(run) for run in (greedy, coord, randp))
ok("S_n^T (A mu_n - b) == 0 for all n, all three policies (explored subspace "
   "residual-free, Wenger et al. 2022)", err_rf < 1e-9, err_rf)

# actual CG A-norm error vs the posterior band sqrt(tr(A Sigma_n))
def a_norm_errs(run):
    return [math.sqrt(max((x_star - m) @ Ad @ (x_star - m), 0.0)) for m in run["mus"]]

errA_greedy = a_norm_errs(greedy)
band = [math.sqrt(n - k) for k in range(len(greedy["mus"]))]
mono = all(errA_greedy[k + 1] < errA_greedy[k] + 1e-14 for k in range(len(errA_greedy) - 1))
ok("actual CG A-norm error decreases monotonically (recorded against the "
   "posterior band sqrt(tr(A Sigma_n)) -- the band is the PRIOR-average, not per-b)",
   mono)

# =============================================================================
# 5. Corner Case 1: coordinate actions == Cholesky (Hennig SIOPT 2015)
# =============================================================================
Vc = coord["V"]
R = Vc.T @ Ad                                  # R_ij = v_i^T y_j, y_j = A e_j
Lc = np.linalg.cholesky(Ad)
scale = np.abs(Lc).max()
err_tri = np.abs(np.tril(R, -1)).max() / scale
err_chol = np.abs(R.T - Lc).max() / scale
err_rr = np.abs(R.T @ R - Ad).max() / np.abs(Ad).max()
ok("Corner Case 1: coordinate actions e_1..e_32 + prior A^-1 ==> R_ij = v_i^T y_j "
   "is upper triangular and R^T == chol(A) (signs included), R^T R == A",
   max(err_tri, err_chol, err_rr) < 1e-12, max(err_tri, err_chol, err_rr))
err_vtri = np.abs(np.tril(Vc, -1)).max()
ok("Corner Case 1: V is upper triangular (Gram-Schmidt of e_1..e_n in the "
   "A-inner product stays in span{e_1..e_n}) and A^-1 == V V^T",
   max(err_vtri, np.abs(Vc @ Vc.T - invA).max() / np.abs(invA).max()) < 1e-10,
   max(err_vtri, np.abs(Vc @ Vc.T - invA).max() / np.abs(invA).max()))

# =============================================================================
# 6. policy race: identical uncertainty budget, wildly different error
# =============================================================================
runs = {"greedy": greedy, "coordinate": coord, "random": randp}
race = {}
for name, run in runs.items():
    tr_curve = [float(np.trace(Ad @ S)) for S in run["Sigmas"]]
    race[name] = {"trace": tr_curve, "errA": a_norm_errs(run),
                  "n_steps": len(run["mus"]) - 1}
err_budget = max(abs(race[p]["trace"][k] - (n - k))
                 for p in race for k in range(len(race[p]["trace"])))
ok("policy race: tr(A Sigma_k) == 32 - k for greedy, coordinate AND random -- "
   "identical uncertainty budget, 1 unit per step regardless of policy",
   err_budget < 1e-8, err_budget)
eg, ec, er = (race[p]["errA"][8] for p in ("greedy", "coordinate", "random"))
eg16, ec16, er16 = (race[p]["errA"][16] for p in ("greedy", "coordinate", "random"))
ok(f"policy race: the A-norm ERRORS differ wildly: k = 8 greedy {eg:.3e} vs "
   f"coordinate {ec:.3e} / random {er:.3e} (>= 3x); k = 16 greedy {eg16:.3e} vs "
   f"{ec16:.3e} / {er16:.3e} (>= 8x)",
   eg < er / 3 and eg < ec / 3 and eg16 < er16 / 8 and eg16 < ec16 / 8)
final_errs = [race[p]["errA"][-1] for p in race]
ok("policy race: all three policies collapse to x* at k = 32 (full rank => "
   "posterior is a point) -- same destination, different journeys",
   max(final_errs) / xstar_Anorm < 1e-7, max(final_errs) / xstar_Anorm)

# =============================================================================
# 7. posterior-ellipse widget data (2x2)
# =============================================================================
# prior N(0, A^{-1}): 1-sigma axes sqrt(1/2) along (2,-1)/sqrt5, sqrt(1/7) along (1,2)/sqrt5
w_pr, Q_pr = np.linalg.eigh(np.linalg.inv(A2))
ok("2x2 prior ellipse: eigvals of A^-1 == {1/7, 1/2} with axes (1,2), (2,-1) "
   "(1-sigma semi-axes 0.37796, 0.70711)",
   abs(w_pr[0] - 1 / 7) < 1e-12 and abs(w_pr[1] - 1 / 2) < 1e-12
   and abs(abs(Q_pr[:, 0] @ np.array([1, 2]) / math.sqrt(5)) - 1) < 1e-12
   and abs(abs(Q_pr[:, 1] @ np.array([2, -1]) / math.sqrt(5)) - 1) < 1e-12)

detS1 = Sig1f[0][0] * Sig1f[1][1] - Sig1f[0][1] * Sig1f[1][0]
y1f = fmatvec(A2f, b2f)
measured_var = fdot(y1f, fmatvec(Sig1f, y1f))
ok("2x2 after obs 1: det Sigma_1 == 0 and y_1^T Sigma_1 y_1 == 0 exactly -- the "
   "credible ellipse collapses to a segment (variance along the measured "
   "direction is spent)", detS1 == 0 and measured_var == 0)
dstar = [xs - m for xs, m in zip([Fraction(2), Fraction(-2)], mu1f)]
cross_seg = dstar[0] * Fraction(-5) - dstar[1] * Fraction(22)
ok("2x2 after obs 1: x* - mu_1 == (6/83)(22,-5) -- the true solution lies "
   "EXACTLY on the collapsed segment line", cross_seg == 0
   and dstar == [Fraction(132, 83), Fraction(-30, 83)])
seg_sigma = float(Fraction(509, 1162)) ** 0.5
dist_star = math.sqrt(sum(float(d) ** 2 for d in dstar))

# steepest descent zigzag from the SAME x0 = 0 (Shewchuk SS6)
sd_path = [np.zeros(2)]
xk = np.zeros(2)
for _ in range(8):
    rk = b2 - A2 @ xk
    xk = xk + float(rk @ rk) / float(rk @ A2 @ rk) * rk
    sd_path.append(xk.copy())
sd_errA = [math.sqrt((np.array([2., -2.]) - p) @ A2 @ (np.array([2., -2.]) - p))
           for p in sd_path]
sd_orth = max(abs(float((b2 - A2 @ sd_path[k]) @ (b2 - A2 @ sd_path[k + 1])))
              for k in range(7))
sd_ratios = [sd_errA[k + 1] / sd_errA[k] for k in range(8)]
ok("2x2 steepest descent from x0 = 0: successive residuals orthogonal (zigzag), "
   "constant contraction ||e||_A ratio, still 8 steps short of x* while CG is "
   "EXACT in 2", sd_orth < 1e-10 and sd_errA[8] > 1e-4
   and max(sd_ratios) - min(sd_ratios) < 1e-10)

# =============================================================================
# 8. calibration: Mahalanobis statistic ~ chi^2_{D-i} (slides' claim, empirical)
# =============================================================================
K_cal, n_obs = 200, 5
dof = n - n_obs
rng_cal = np.random.default_rng(0)
Xc = np.linalg.solve(L_chol.T, rng_cal.standard_normal((n, K_cal))).T
Scal = rng_cal.standard_normal((n, n_obs))     # FIXED design, same for every draw
Ycal = Ad @ Scal
Gcal = Ycal.T @ invA @ Ycal
Pcal = invA @ Ycal @ np.linalg.solve(Gcal, Ycal.T)
Sig5 = invA - invA @ Ycal @ np.linalg.solve(Gcal, Ycal.T @ invA)
w5, Q5 = np.linalg.eigh(Sig5)
rank5 = int((w5 > w5.max() * 1e-10).sum())
winv = np.where(w5 > w5.max() * 1e-10, 1.0 / np.maximum(w5, 1e-300), 0.0)
Ec = Xc - Xc @ Pcal.T
Zc = Ec @ Q5
stats = (Zc ** 2 * winv).sum(axis=1)
q90 = float(chi2_dist.ppf(0.9, dof))
cov90 = float((stats <= q90).mean())
mean_stat, var_stat = float(stats.mean()), float(stats.var(ddof=1))
ok(f"calibration: rank(Sigma_5) == {dof} and Mahalanobis stat ~ chi^2_{dof}: "
   f"empirical mean {mean_stat:.2f} vs {dof} (rel tol 10%), 90% coverage "
   f"{cov90:.3f} (tol +/-0.07)", rank5 == dof
   and abs(mean_stat - dof) / dof < 0.10 and abs(cov90 - 0.9) < 0.07)

npass = sum(c["pass"] for c in results["checks"])

# =============================================================================
# every number / trajectory the report quotes
# =============================================================================
r8 = lambda v: float(f"{v:.10g}")
lst = lambda a: [r8(float(x)) for x in np.asarray(a).ravel()]
mat = lambda M: [[r8(float(x)) for x in row] for row in np.asarray(M)]

results["quoted"] = {
    "notation": {
        "prior": "p(x) = N(0, Sigma_0)",
        "action_observation": "bbar_i = s_i^T b = y_i^T x,  y_i = A^T s_i",
        "v_i": "Sigma_{i-1} y_i / sqrt(y_i^T Sigma_{i-1} y_i)",
        "xi_i": "(bbar_i - y_i^T mu_{i-1}) / sqrt(y_i^T Sigma_{i-1} y_i)",
        "source": "Hennig, Pfoertner, Weiland, Probabilistic Numerics tutorial, "
                  "ICML Seoul 06/07/2026, pp. 7-20; Shewchuk 1994 SS3/SS6/SS7-8",
    },
    "shewchuk_2x2": {
        "A": [[3, 2], [2, 6]], "b": [2, -8], "x_star": [2, -2],
        "eigenvalues": [2, 7],
        "Sigma0_Ainv": [["3/7", "-1/7"], ["-1/7", "3/14"]],
        "cg_exact": {
            "alpha_1": "17/83", "x_1": ["34/83", "-136/83"],
            "x_1_float": [float(Fraction(34, 83)), float(Fraction(-136, 83))],
            "r_1": ["336/83", "84/83"], "beta_2": "1764/6889",
            "p_2": [fstr(p2[0]), fstr(p2[1])], "alpha_2": "83/238",
            "x_2": ["2", "-2"],
        },
        "prob_exact": {
            "mu_1": [fstr(v) for v in mu1f], "mu_2": ["2", "-2"],
            "Sigma_1": [[fstr(v) for v in row] for row in Sig1f],
            "Sigma_1_over_1162": [[484, -110], [-110, 25]],
            "Sigma_1_rank1": "Sigma_1 = (509/1162) u u^T, u = (22,-5)/sqrt(509)",
            "trace_A_Sigma": [2, 1, 0],
            "xi_sq": [fstr(Fraction(68) ** 2 / den1), fstr(fdot(rf1, rf1) ** 2 / den2)],
            "v1_float": lst(np.array([2.0, -8.0]) / math.sqrt(332.0)),
            "den_1": fstr(den1), "den_2": fstr(den2),
        },
        "ellipse_prior": {
            "eigvals": ["1/2", "1/7"],
            "semi_axes": [r8(math.sqrt(0.5)), r8(math.sqrt(1 / 7))],
            "axis_dirs": [[2, -1], [1, 2]],
            "axis_dirs_normalized": mat(np.array([[2, -1], [1, 2]]) / math.sqrt(5)),
        },
        "ellipse_obs1": {
            "center": [fstr(v) for v in mu1f],
            "center_float": lst([float(v) for v in mu1f]),
            "segment_dir": [22, -5],
            "segment_dir_normalized": lst(np.array([22, -5]) / math.sqrt(509)),
            "variance_along": "509/1162",
            "one_sigma_half_length": r8(seg_sigma),
            "x_star_on_segment": "x* - mu_1 = (6/83)(22,-5), distance "
                                 f"{r8(dist_star)} = {r8(dist_star / seg_sigma)} sigma",
            "collapsed_dir_v1": lst(np.array([2.0, -8.0]) / math.sqrt(332.0)),
        },
        "ellipse_obs2": {"center": [2, -2], "note": "posterior is a point"},
        "cg_path": [[0, 0], [float(Fraction(34, 83)), float(Fraction(-136, 83))], [2, -2]],
        "sd_path": mat(np.array(sd_path)),
        "sd_errA": lst(sd_errA),
        "sd_contraction_per_step": r8(sd_ratios[0]),
        "sd_residual_orthogonality": r8(sd_orth),
    },
    "chain": {
        "n": n, "h": h, "A": "laplacian_1d(32)/h^2",
        "rhs": {"node_hot": j_hot + 1, "s_hot": (j_hot + 1) * h, "f_hot": f_hot,
                "node_cold": j_cold + 1, "s_cold": (j_cold + 1) * h, "f_cold": f_cold,
                "scaling": "b[j] = f/h (green-tents physical-watts convention)"},
        "b": lst(b), "x_star": lst(x_star),
        "x_star_A_norm": r8(xstar_Anorm),
        "prior_band_at_0": r8(math.sqrt(n)),
        "cg_iterations": n_cg, "greedy_steps": n_greedy,
        "pcg_res_hist": lst(res_hist),
        "headline_iterate_rel_err": r8(err_it),
        "v_vs_p_max_err": r8(err_v),
        "conjugacy_equals_mahalanobis_err": r8(err_same),
        "conjugacy_offdiag_err": r8(err_conj),
        "greedy_mus": [lst(m) for m in greedy["mus"]],
        "sequential_vs_batch_rel_err": {k: r8(v) for k, v in batch_errs.items()},
        "gram_schmidt_errs": {k: {kk: r8(vv) for kk, vv in d.items()}
                              for k, d in gs_errs.items()},
    },
    "uncertainty": {
        "trace_curve_greedy": lst(traces_greedy),
        "band_sqrt_trace": lst(band),
        "errA_greedy": lst(errA_greedy),
        "trace_identity_max_err": r8(err_tr),
        "mc_identity": [{"k": t["k"], "mc_mean": r8(t["mc_mean"]),
                         "trace": t["trace"]} for t in mc_table],
        "mc_K": K_mc, "mc_rel_err": r8(err_mc),
        "residual_free_max_err": r8(err_rf),
        "honest_note": "tr(A Sigma_n) = E||x-mu_n||_A^2 averages over the PRIOR; "
                       "the actual error for this particular b is one draw and "
                       "sits below the band once the greedy policy locks onto it",
    },
    "corner_case_1": {
        "R_upper_triangular_err": r8(err_tri),
        "R_T_equals_chol_err": r8(err_chol),
        "RTR_equals_A_err": r8(err_rr),
        "V_upper_triangular_err": r8(err_vtri),
        "chol_diag": lst(np.diag(Lc)),
        "R_diag": lst(np.diag(R)),
        "R": mat(R), "chol_A_lower": mat(Lc),
        "citation": "Hennig SIOPT 2015; = update_views_explainer Shawe-Taylor/Cholesky",
    },
    "policy_race": {
        p: {"trace": lst(race[p]["trace"]), "errA": lst(race[p]["errA"]),
            "errA_rel": lst(np.array(race[p]["errA"]) / xstar_Anorm),
            "n_steps": race[p]["n_steps"]} for p in race},
    "policy_race_summary": {
        "budget_identical_max_err": r8(err_budget),
        "errA_at_k8": {"greedy": r8(eg), "coordinate": r8(ec), "random": r8(er)},
        "errA_at_k16": {"greedy": r8(eg16), "coordinate": r8(ec16), "random": r8(er16)},
        "ratio_coord_over_greedy_k8": r8(ec / eg),
        "ratio_random_over_greedy_k8": r8(er / eg),
        "ratio_coord_over_greedy_k16": r8(ec16 / eg16),
        "ratio_random_over_greedy_k16": r8(er16 / eg16),
        "teaching_point": "identical tr(A Sigma_k) = 32-k budgets; greedy spends "
                          "its budget where THIS b lives",
    },
    "calibration": {
        "K": K_cal, "n_obs": n_obs, "dof": dof, "rank_Sigma5": rank5,
        "mean_stat": r8(mean_stat), "expected_mean": dof,
        "var_stat": r8(var_stat), "expected_var": 2 * dof,
        "coverage_90": r8(cov90), "chi2_q90": r8(q90),
    },
    "checks_total": len(results["checks"]),
    "checks_passed": npass,
    "max_err_all_checks": max(c.get("max_err", 0.0) for c in results["checks"]),
}

out = Path(__file__).resolve().parents[2] / "results" / "probabilistic_cg.json"
out.write_text(json.dumps(results, indent=2))
size = os.path.getsize(out)
print(f"\n{npass}/{len(results['checks'])} checks passed")
print(f"wrote {out} ({size/1024:.0f} KB)")

# ---- console tables ---------------------------------------------------------
print("\n2x2 exact numbers (Shewchuk arena, prior A^-1, greedy = CG):")
print(f"  alpha_1 = 17/83, x_1 = mu_1 = (34/83, -136/83) = ({float(Fraction(34,83)):.6f}, {float(Fraction(-136,83)):.6f})")
print(f"  r_1 = (336/83, 84/83), beta_2 = 1764/6889, alpha_2 = 83/238, x_2 = mu_2 = (2, -2)")
print(f"  Sigma_1 = (1/1162)[[484,-110],[-110,25]], tr(A Sigma) = 2 -> 1 -> 0")
print("\npolicy race on the chain (A-norm error, identical budget tr = 32-k):")
print(f"{'k':>4}{'greedy':>12}{'coordinate':>12}{'random':>12}{'budget':>9}")
for k in (0, 2, 4, 8, 16, 24, 31, 32):
    row = [race[p]["errA"][k] if k < len(race[p]["errA"]) else float('nan')
           for p in ("greedy", "coordinate", "random")]
    print(f"{k:>4}{row[0]:>12.3e}{row[1]:>12.3e}{row[2]:>12.3e}{32-k:>9}")
