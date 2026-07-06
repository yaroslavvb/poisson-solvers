#!/usr/bin/env wolframscript
(* ::Package:: *)

(*
grid8_regressions.wls

Small-grid (n = 8, N = 64) companion to reports/09-stiffness-as-precision.md
and reports/10 (fluctuation-dissipation): reproduces the user's n = 8 snippets
in the repo convention and exports the two ArrayPlots they made plus the
Cholesky-band plot.

Convention (matches the suite): n = 8 interior nodes per side, h = 1/(n+1) = 1/9,
2-D A = Kronecker sum / h^2 (N = n^2 = 64).  u ~ N(0, A^-1); the regression
matrix B = I - diag(A)^-1 A (Jacobi iteration matrix) stores the
full-conditional regressions with stencil coefficients 1/4 and conditional
variance 1/A_ii = h^2/4.  phiR2L = chol(A): L = Transpose[CholeskyDecomposition[A]]
is the regression-on-successors factor, banded with bandwidth n = 8 and fill
inside the band (this fill is what Vecchia / IC(0) truncates).

Checks (PASS/FAIL style follows python/experiments/verify_statistical_identities.py):
  1. B nonzero pattern = 5-point stencil off-diagonals, coefficients exactly 1/4;
  2. notebook identity A == (I - B).D2, D2 = diag(A), to machine precision;
  3. conditional variance 1/A_ii == h^2/4;
  4. Inverse[A] elementwise positive (discrete Green's function);
  5. L = CholeskyDecomposition[A]^T has bandwidth exactly n = 8, with fill
     strictly inside the band (nonzeros of L outside A's lower pattern).

Run from repo root:  wolframscript -file mathematica/grid8_regressions.wls
Exports: figures/mma_grid8_inverse.png, figures/mma_grid8_B.png,
         figures/mma_grid8_cholL.png
*)

ok[name_, cond_] := Print[If[cond, "PASS", "FAIL"], ": ", name];
relErr[x_, y_] := Norm[Flatten[Normal[x - y]]]/Norm[Flatten[Normal[y]]];

(* ================= 1. Build A and B (user's snippet) ================= *)

n = 8;
h = 1.0/(n + 1);                                   (* h = 1/9 *)
nn = n^2;                                          (* N = 64 *)

d1 = SparseArray[
    {Band[{1, 1}] -> 2., Band[{2, 1}] -> -1., Band[{1, 2}] -> -1.},
    {n, n}];
id = IdentityMatrix[n, SparseArray];
A = (KroneckerProduct[d1, id] + KroneckerProduct[id, d1])/h^2;

B = IdentityMatrix[nn] -
    Inverse[DiagonalMatrix[Diagonal[A]]] . A;      (* Inverse[] densifies *)
Bsp = SparseArray[Chop[B, 1.*^-14]];               (* recover sparse pattern *)

(* ================= 2. Checks ================= *)

(* (i) nonzero pattern = 4-neighbor (5-point stencil) pattern, coefficients 1/4 *)
stencilPos = Select[A["NonzeroPositions"], #[[1]] =!= #[[2]] &]; (* off-diag support *)
offVals = Bsp["NonzeroValues"];
maxDiag = Max[Abs[Diagonal[B]]];
Print["B (N = ", nn, "): ", Length[Bsp["NonzeroPositions"]],
  " nonzeros (5-point off-diagonal pattern has ", Length[stencilPos],
  "),  max |coeff - 1/4| = ", Max[Abs[offVals - 1/4]],
  ",  max |diagonal| = ", maxDiag];
ok["B nonzeros = 4-neighbor stencil pattern, coefficients exactly 1/4",
  Sort[Bsp["NonzeroPositions"]] === Sort[stencilPos] &&
  Max[Abs[offVals - 1/4]] == 0. && maxDiag < 1.*^-14];

(* (ii) notebook identity A == (I - B).D2, D2 = diag(A) *)
D2 = DiagonalMatrix[Diagonal[A]];
Print["notebook identity: relative error ||A - (I-B).D2||_F / ||A||_F = ",
  relErr[(IdentityMatrix[nn] - B) . D2, A]];
ok["notebook identity A == (I - B).D2 to machine precision",
  relErr[(IdentityMatrix[nn] - B) . D2, A] < 1.*^-14];

(* (iii) conditional variance 1/A_ii = h^2/4 *)
Print["conditional variance: 1/A_ii = ", 1/A[[1, 1]],
  "  (theory h^2/4 = ", h^2/4, ")"];
ok["conditional variance 1/A_ii == h^2/4",
  Abs[1/A[[1, 1]] - h^2/4]/(h^2/4) < 1.*^-14];

(* (iv) Inverse[A] elementwise positive (discrete Green's function / covariance) *)
Ainv = Inverse[Normal[A]];
Print["Inverse[A]: min entry = ", Min[Ainv], ",  max entry = ", Max[Ainv]];
ok["Inverse[A] elementwise positive", Min[Ainv] > 0.];

(* (v) Cholesky: upper U, L = U^T; bandwidth n = 8 with fill inside the band *)
U = CholeskyDecomposition[A];
L = Transpose[Normal[U]];
Lsp = SparseArray[Chop[L, 1.*^-12]];
bandwidthL = Max[(#[[1]] - #[[2]]) & /@ Lsp["NonzeroPositions"]];
lowerPatA = Select[A["NonzeroPositions"], #[[1]] >= #[[2]] &];
fillPos = Complement[Lsp["NonzeroPositions"], lowerPatA];
nFill = Length[fillPos];
Print["L = chol(A)^T: ", Length[Lsp["NonzeroPositions"]],
  " nonzeros, bandwidth = ", bandwidthL,
  ";  A's lower pattern has ", Length[lowerPatA],
  " nonzeros -> ", nFill, " fill-in entries inside the band"];
ok["L has bandwidth exactly n = 8",
  bandwidthL == n];
ok["fill-in stays inside the band and is nonempty (Vecchia truncates it)",
  nFill > 0 && Max[(#[[1]] - #[[2]]) & /@ fillPos] <= n];

(* ================= 3. Figures ================= *)

Export["figures/mma_grid8_inverse.png",
  ArrayPlot[Ainv, ColorFunction -> "SunsetColors", PlotRangePadding -> None,
    ImageSize -> 500,
    PlotLabel ->
      "Inverse[A]: covariance with 8x8 block structure (n = 8, N = 64)"],
  ImageResolution -> 150];

Export["figures/mma_grid8_B.png",
  ArrayPlot[B, ImageSize -> 500,
    PlotLabel ->
      "regression matrix B = I - diag(A)^-1 A   (n = 8, N = 64, coeffs 1/4)"],
  ImageResolution -> 150];

Export["figures/mma_grid8_cholL.png",
  ArrayPlot[Log10[Abs[L] + 1.*^-16], ColorFunction -> "SunsetColors",
    PlotRangePadding -> None, ImageSize -> 500,
    PlotLabel ->
      "Log10|L|, L = CholeskyDecomposition[A]^T: filled band, bandwidth 8"],
  ImageResolution -> 150];

Print["Exported figures/mma_grid8_inverse.png, figures/mma_grid8_B.png, ",
  "figures/mma_grid8_cholL.png"];
