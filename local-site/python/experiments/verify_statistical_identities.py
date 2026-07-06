"""Numerical verification of every identity asserted in reports/09-stiffness-as-precision.md.

Checks 19 exact statements about the 1-D Dirichlet Laplacian A = d1/h^2 (n = 8):
the Green's-function / Brownian-bridge covariance identity, full-conditional
regressions, Jacobi's iteration matrix = the regression matrix B, the
(I - B).D2 precision identity, Cholesky-as-sequential-regression facts in both
orderings (incl. chol(Sigma) = P.inv(L)'.P), QR/Cholesky duality on the
difference operator, the modified-Cholesky (Pourahmadi) construction, and the
DST-I eigensystem.

Run from the repo root:
    uv run python python/experiments/verify_statistical_identities.py

Expected output: 19 lines, all PASS.
"""
import numpy as np
from numpy.linalg import cholesky, inv, pinv, qr, eigvals
np.set_printoptions(precision=4, suppress=True)
ok = lambda name, cond: print(f"{'PASS' if cond else 'FAIL'}: {name}")

n = 8; h = 1.0/(n+1)
L1 = np.diag(2.0*np.ones(n)) + np.diag(-1.0*np.ones(n-1),1) + np.diag(-1.0*np.ones(n-1),-1)
A = L1 / h**2
S = inv(A)                      # covariance
x = (np.arange(1,n+1))*h        # physical nodes

# 1. A^-1 = h * BrownianBridge covariance min(s,t)-st
G = np.minimum.outer(x,x) - np.outer(x,x)
ok("A^-1 == h * (min(s,t)-st)", np.allclose(S, h*G))

# 2. full conditionals: E[u_i|rest] coeffs = -A_ij/A_ii ; Var = 1/A_ii = h^2/2
i = 3
Sii = S[i,i]; Sio = np.delete(S[i,:],i); Soo = np.delete(np.delete(S,i,0),i,1)
w = Sio @ inv(Soo)              # regression coefficients of x_i on rest
ok("cond coeffs == -A_ij/A_ii", np.allclose(w, -np.delete(A[i,:],i)/A[i,i]))
ok("cond var == 1/A_ii == h^2/2", np.isclose(Sii - Sio@inv(Soo)@Sio, 1/A[i,i]) and np.isclose(1/A[i,i], h*h/2))

# 3. Jacobi iteration matrix == regression matrix B; rho(B) = cos(pi h)
B = np.eye(n) - np.diag(1/np.diag(A)) @ A
ok("Jacobi matrix rows = neighbor-average coeffs (1/2)", np.allclose(B[i,i-1],0.5) and np.allclose(B[i,i+1],0.5))
ok("rho(B) == cos(pi h)", np.isclose(max(abs(eigvals(B))), np.cos(np.pi*h)))

# 4. notebook identity Q = (I - B_col) . D2 on a random full-rank data matrix
rng = np.random.default_rng(0); X = rng.standard_normal((12,5))
C = X.T @ X
Bcol = np.zeros((5,5)); resid2 = np.zeros(5)
for j in range(5):
    Xo = np.delete(X,j,1)
    beta = pinv(Xo) @ X[:,j]
    Bcol[np.arange(5)!=j, j] = beta
    r = X[:,j] - Xo@beta; resid2[j] = r@r
Q = (np.eye(5)-Bcol) @ np.diag(1/resid2)
ok("precision == (I - B).D2  (notebook)", np.allclose(Q, inv(C)))
ok("diag(Q) == 1/residual^2   (notebook)", np.allclose(np.diag(inv(C)), 1/resid2))
ok("X.inv(X'X) == pinv(X')    (notebook)", np.allclose(X@inv(C), pinv(X.T)))

# 5. chol(A)=L lower: regression on successor, coeff = i/(i+1), innovation sd = 1/L_ii
L = cholesky(A)
mask = np.ones_like(L, bool); np.fill_diagonal(mask, False)
mask[np.arange(1,n), np.arange(0,n-1)] = False
ok("L is lower-bidiagonal (no fill on a chain)", np.allclose(L[mask], 0))
coef = np.array([-L[i+1,i]/L[i,i] for i in range(n-1)])
ok("coef of x_i on x_{i+1} == i/(i+1)", np.allclose(coef, np.arange(1,n)/np.arange(2,n+1)))
# direct check: regression of x_i on x_{i+1..n} under Sigma
i = 2
Sio = S[i,i+1:]; Soo = S[i+1:,i+1:]
wsucc = Sio @ inv(Soo)
ok("only immediate successor matters (Markov)", np.allclose(wsucc[1:],0) and np.isclose(wsucc[0], (i+1)/(i+2)))
ok("innovation sd == 1/L_ii", np.isclose(np.sqrt(S[i,i]-Sio@inv(Soo)@Sio), 1/L[i,i]))

# 6. chol(Sigma) == P L^-T P with P = reversal
P = np.eye(n)[::-1]
ok("chol(Sigma) = P.inv(L).T.P (reverse order)", np.allclose(cholesky(P@S@P), P@inv(L).T@P))
ok("sampling factor = inv(L).T:  (L^-T)(L^-T)' == Sigma", np.allclose(inv(L).T@inv(L), S))

# 7. QR of difference matrix D: A = D'D, R == L' (sign-fixed), Q = D R^-1
D = np.zeros((n+1,n))
for kk in range(n):
    D[kk,kk] = 1.0; D[kk+1,kk] = -1.0
D /= h
ok("A == D'D", np.allclose(A, D.T@D))
Qq,R = qr(D)
sgn = np.diag(np.sign(np.diag(R))); R = sgn@R; Qq = Qq@sgn
ok("R from QR(D) == chol(A)^T", np.allclose(R, L.T))
ok("Q = D.R^-1 orthonormal (whitened design)", np.allclose((D@inv(R)).T@(D@inv(R)), np.eye(n)))

# 8. Pourahmadi mchol: T = I - Phi (regress on predecessors) => T Sigma T' diagonal
T = np.eye(n)
for ii in range(1,n):
    w = S[ii,:ii] @ inv(S[:ii,:ii])
    T[ii,:ii] = -w
Dd = T@S@T.T
ok("(I-Phi) Sigma (I-Phi)' diagonal (modified Cholesky)", np.allclose(Dd, np.diag(np.diag(Dd))))

# 9. DST eigen: eigvecs sin(k pi x_i), eigvals 4 sin^2(k pi h/2)/h^2
k = np.arange(1,n+1)
lam = 4*np.sin(k*np.pi*h/2)**2/h**2
V = np.sin(np.outer(x,k)*np.pi)
ok("sine vectors are eigenvectors with stated eigenvalues", np.allclose(A@V, V@np.diag(lam)))
