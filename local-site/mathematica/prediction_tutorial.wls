#!/usr/bin/env wolframscript
(* ::Package:: *)

(*
mathematica/prediction_tutorial.wls

EXACT-ARITHMETIC companion to the report-15 prediction tutorial: the n = 5
Dirichlet chain A = tridiag(-1,2,-1)/h^2, h = 1/6, done entirely in rationals
-- no N[] anywhere, so every PASS below is an identity, not an approximation.

Function names and construction style mirror the companion notebook
whitening_inverse_transposed.nb (~/git0/newton/):
  B            hollow two-sided regression matrix; x - B.x removes each
               coordinate's prediction from all the others
               (notebook: getB, "B matrix, 5.6 of Pourahmadi")
  D2           diagonal of INVERSE residual variances; here the residual
               (full-conditional) variance is 1/A_ii, so D2 = diag(A)
               (notebook: D2, "D2 matrix of 5.6")
  precisionMat (iiMat - B).D2 == A         (notebook: precisionMat)
  phiL2R[i]    regress coordinate i on predecessors 1..i-1   (notebook: phiL2R)
  phiR2L[i]    regress coordinate i on successors  i+1..n    (notebook: phiR2L)
  mchol        modified Cholesky Sigma = C.diag.diag.C^T, C unit lower
               triangular                                     (notebook: mchol)
  imchol       Inverse[mchol[cov]]                            (notebook: imchol)

One convention note: the notebook works with data matrices X (rows = samples)
and multiplies ROW vectors on the right (x - x.B, x - x.phiL2R), so its
coefficient matrices (getB's B, coefsLeftToRight, coefsChol) are the
TRANSPOSES of the ones here. In that column convention (iiMat - B).D2 == A
holds for any SPD A; for the row-convention B built below the general
identity is D2.(iiMat - B) == A, and the two orders coincide on this chain
only because diag(A) = 72 iiMat makes B symmetric (report 10 Sec. 3.1).
This script works with COLUMN vectors and the population covariance Sigma
directly -- the notebook's PseudoInverse[Y1'.Y1].(Y1'.Y2) normal equations
become Inverse[Sigma_pred,pred].Sigma_pred,target -- so PhiL2R is strictly
LOWER triangular, residual = (iiMat - PhiL2R).x, and C = Inverse[iiMat - PhiL2R].

Prints the same quantities as the Python (float) version of steps 1-4 so the
two can be cross-checked line by line.

Run from repo root:  wolframscript -file mathematica/prediction_tutorial.wls
Expected: 13 lines PASS, exit code 0.
*)

nFail = 0; nPass = 0;
ok[name_, cond_] := If[cond === True,
  nPass++; Print["PASS: ", name],
  nFail++; Print["FAIL: ", name]];
printRows[label_, m_] := (Print[label];
  Do[Print["  ", ToString[m[[i]], InputForm]], {i, Length[m]}];);

n = 5;
h = 1/6;                        (* exact: h = 1/(n+1) *)
iiMat = IdentityMatrix[n];      (* notebook: iiMat *)
xs = Table[i*h, {i, n}];        (* physical nodes x_i = i h *)

(* ---------------- Step 1: exact A and Sigma = Inverse[A] ---------------- *)

Print["== Step 1: n = 5 chain, A = tridiag(-1,2,-1)/h^2, h = 1/6, exact =="];
A = Table[Which[i == j, 2, Abs[i - j] == 1, -1, True, 0], {i, n}, {j, n}]/h^2;
Sigma = Inverse[A];             (* exact covariance of the discrete bridge *)

Print["h = ", ToString[h, InputForm]];
printRows["A rows (InputForm rationals):", A];
printRows["Sigma = Inverse[A] rows (InputForm rationals):", Sigma];

(* Brownian-bridge covariance: Sigma_ij = h x_i (1 - x_j) on the upper
   triangle i <= j, i.e. h * min(x_i,x_j) * (1 - max(x_i,x_j))  (report 09) *)
bridge = h*Table[Min[i, j]/6*(1 - Max[i, j]/6), {i, n}, {j, n}];
ok["Sigma == h Min[x_i,x_j](1 - Max[x_i,x_j])  (Brownian-bridge covariance)",
  Sigma === bridge];

(* ------- Step 2: hollow two-sided B, D2, precisionMat = (iiMat-B).D2 ------- *)

Print[""];
Print["== Step 2: two-sided regression matrix B and precisionMat (notebook: getB, D2, precisionMat) =="];

(* B: each row i = coefficients predicting u_i from all other coordinates;
   hollow (zero diagonal).  (notebook: getB -- there built by per-coordinate
   least squares on data X; on the chain it collapses to I - diag(A)^-1.A,
   the 1/2-on-neighbors pattern = the Jacobi predictor, report 12 Sec. 2) *)
B = iiMat - Inverse[DiagonalMatrix[Diagonal[A]]] . A;
printRows["B rows (hollow, 1/2 on chain neighbors):", B];

expB = Table[If[Abs[i - j] == 1, 1/2, 0], {i, n}, {j, n}];
ok["B == iiMat - Inverse[diag(A)].A has the hollow 1/2 neighbor pattern",
  B === expB];

(* rows of B really are the exact full-conditional regressions from Sigma:
   w_i = Sigma_i,rest . Inverse[Sigma_rest,rest]  (notebook: phi / getB) *)
twoSidedOK = True; condVarOK = True;
Do[
  others = Delete[Range[n], i];
  w = Flatten[Sigma[[{i}, others]] . Inverse[Sigma[[others, others]]]];
  twoSidedOK = twoSidedOK && (w === B[[i, others]]);
  condVar = Sigma[[i, i]] - w . Sigma[[others, i]];
  condVarOK = condVarOK && (condVar === 1/A[[i, i]]) && (condVar === h^2/2),
  {i, n}];
ok["every row of B == regression of u_i on the rest computed from Sigma",
  twoSidedOK];
ok["full-conditional variance == 1/A_ii == h^2/2 == 1/72 at every i",
  condVarOK];

(* D2 = diagonal of inverse residual variances (notebook: D2 = 1/diag(R'.R));
   residual variance of the full conditional is 1/A_ii, so D2 = diag(A) *)
D2 = DiagonalMatrix[Diagonal[A]];
Print["D2 diagonal (inverse residual variances): ", ToString[Diagonal[D2], InputForm]];

(* notebook: precisionMat = (iiMat - B).D2, Assert[precisionMat == Inverse[cov@X]];
   this order is valid here because B is symmetric -- see header convention note *)
precisionMat = (iiMat - B) . D2;
ok["A == (iiMat - B).D2 exactly  (notebook: precisionMat identity)",
  A === precisionMat];

(* -------- Step 3: one-sided regressions phiL2R / phiR2L and mchol -------- *)

Print[""];
Print["== Step 3: one-sided regressions (notebook: phiL2R, phiR2L) and modified Cholesky (notebook: mchol) =="];

(* Coefficients to predict coordinate i from predecessors 1..i-1.
   (notebook: phiL2R -- there p = PseudoInverse[Y1'.Y1].(Y1'.Y2) on data X;
   population analogue on Sigma below.  PadRight as in the notebook.) *)
phiL2R[i_, cov_] := Module[{coefs},
  coefs = Switch[i,
    1, {},   (* empty list: no predecessors for the first coordinate *)
    _, Flatten[Inverse[cov[[1 ;; i - 1, 1 ;; i - 1]]] . cov[[1 ;; i - 1, {i}]]]];
  PadRight[coefs, n]];

(* Coefficients to predict coordinate i from successors i+1..n.
   (notebook: phiR2L, PadLeft as in the notebook.) *)
phiR2L[i_, cov_] := Module[{coefs},
  coefs = Switch[i,
    n, {},   (* empty list: no successors for the last coordinate *)
    _, Flatten[Inverse[cov[[i + 1 ;; n, i + 1 ;; n]]] . cov[[i + 1 ;; n, {i}]]]];
  PadLeft[coefs, n]];

(* rows = coefficients (column-vector convention); the notebook's
   coefsLeftToRight/coefsRightToLeft are the transposes of these *)
PhiL2R = Table[phiL2R[i, Sigma], {i, n}];
PhiR2L = Table[phiR2L[i, Sigma], {i, n}];
printRows["PhiL2R rows (predict i from predecessors; strictly lower):", PhiL2R];
printRows["PhiR2L rows (predict i from successors; strictly upper):", PhiR2L];

(* Markov chain: only the immediate neighbor loads.  Predecessor coefficient
   (1-x_i)/(1-x_{i-1}) = (n+1-i)/(n+2-i) -> 4/5, 3/4, 2/3, 1/2;
   successor coefficient x_i/x_{i+1} = i/(i+1) -> 1/2, 2/3, 3/4, 4/5  (report 09) *)
expPhiL2R = Table[If[j == i - 1, (n + 1 - i)/(n + 2 - i), 0], {i, n}, {j, n}];
expPhiR2L = Table[If[j == i + 1, i/(i + 1), 0], {i, n}, {j, n}];
ok["phiL2R loads only on immediate predecessor with (n+1-i)/(n+2-i) = 4/5, 3/4, 2/3, 1/2",
  PhiL2R === expPhiL2R];
ok["phiR2L loads only on immediate successor with i/(i+1) = 1/2, 2/3, 3/4, 4/5",
  PhiR2L === expPhiR2L];

(* notebook: mchol -- returns unit lower triangular C and sets global "diag"
   such that cov = C.diag.diag.C^T *)
mchol[cov_] := Module[{},
  chol = Transpose[CholeskyDecomposition[cov]];
  diag = DiagonalMatrix[Diagonal[chol]];
  chol . Inverse[diag]];

(* notebook: imchol -- inverse modified Cholesky *)
imchol[cov_] := Module[{}, Inverse[mchol@cov]];

(* CholeskyDecomposition introduces exact radicals Sqrt[r]; the unit-triangular
   factor of a rational SPD matrix is rational, so Simplify recovers exact
   fractions (still no N[] -- Simplify of exact radicals is exact) *)
cc = Simplify[mchol[Sigma]];
ds = Simplify[diag . diag];   (* innovation variances, notebook's diag.diag *)
printRows["C = mchol[Sigma] rows (unit lower triangular):", cc];
Print["Ds = diag.diag innovation variances: ", ToString[Diagonal[ds], InputForm]];

(* notebook: coefsChol = iiMat - imchol[cov]^T == coefsLeftToRight; in the
   column convention that is exactly C == Inverse[iiMat - PhiL2R] *)
ok["mchol C == Inverse[iiMat - PhiL2R]  (Cholesky == sequential regression)",
  cc === Inverse[iiMat - PhiL2R]];
ok["Sigma == C.Ds.Transpose[C] exactly  (notebook: mchol identity)",
  Sigma === cc . ds . Transpose[cc]];

(* ------------- Step 4: whitening -- (iiMat - PhiL2R) decorrelates ------------- *)

Print[""];
Print["== Step 4: whitening by the one-sided predictor =="];

W = iiMat - PhiL2R;             (* residual map: innovations = W.u *)
white = W . Sigma . Transpose[W];
printRows["(iiMat - PhiL2R).Sigma.Transpose[iiMat - PhiL2R] rows:", white];
ok["(iiMat - PhiL2R).Sigma.Transpose[iiMat - PhiL2R] is exactly diagonal",
  DiagonalMatrixQ[white]];
ok["whitened diagonal == Ds innovation variances",
  white === ds];
Print["innovation variances (exact fractions): ", ToString[Diagonal[white], InputForm]];

(* ------------- Step 5: perfect prediction solves in one pass ------------- *)

Print[""];
Print["== Step 5: perfect prediction solves A.x = e_3 in one predict-correct pass =="];

b = UnitVector[n, 3];
xStar = Sigma . b;              (* exact solution: column 3 of Sigma *)
Print["b  = ", ToString[b, InputForm]];
Print["x* = Sigma.e_3 = ", ToString[xStar, InputForm]];

(* normal equations of the perfect one-sided predictor ARE the operator
   (report 12 Sec. 2): A = W^T.Inverse[Ds].W with W = iiMat - PhiL2R *)
M = Transpose[W] . Inverse[ds] . W;
ok["A == Transpose[iiMat - PhiL2R].Inverse[Ds].(iiMat - PhiL2R)  (normal equations)",
  A === M];

(* one predict-correct pass from x0 = 0: x1 = x0 + M^-1.(b - A.x0), applied
   as two exact unit-triangular solves -- prediction becomes substitution *)
x0 = ConstantArray[0, n];
r0 = b - A . x0;
y = LinearSolve[Transpose[W], r0];   (* backward solve, exact *)
x1 = x0 + LinearSolve[W, ds . y];    (* forward solve, exact *)
Print["x1 after one predict-correct pass = ", ToString[x1, InputForm]];
ok["one predict-correct pass from x0 = 0 reproduces x* exactly",
  x1 === xStar];

(* ---------------------------------- summary ---------------------------------- *)

Print[""];
Print[nPass, " PASS, ", nFail, " FAIL (all arithmetic exact rational)"];
Exit[If[nFail > 0, 1, 0]];
