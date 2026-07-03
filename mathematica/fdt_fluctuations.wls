#!/usr/bin/env wolframscript
(* ::Package:: *)

(*
fdt_fluctuations.wls

Fluctuation-dissipation companion to reports/09-stiffness-as-precision.md:
validates the B-matrix regression-extraction snippets and gives a compact
Mathematica FDT demo in the repo's notation.

Convention (matches the suite): n interior nodes, h = 1/(n+1),
1-D A = tridiag(-1,2,-1)/h^2, 2-D A = Kronecker sum / h^2 (N = n^2).
u ~ N(0, A^-1) is a Brownian bridge on the grid; the regression matrix
B = I - diag(A)^-1 A (Jacobi iteration matrix) stores the full-conditional
regressions, with the notebook identity A == (I - B).D2, D2 = diag(A).

Checks (PASS/FAIL style follows python/experiments/verify_statistical_identities.py):
  1. B's nonzero pattern = neighbor pattern, coefficients 1/2 (1D), 1/4 (2D);
     notebook identity A == (I - B).D2 to machine precision;
     row-vs-column convention: B symmetric here (constant diagonal), and in
     general B_col == Transpose[B_row] (checked on a random SPD matrix).
  2. FDT demo, 1D n = 32: 20000 samples of N(0, A^-1); empirical covariance
     vs Inverse[A]; sample regression of the center coordinate on its two
     neighbors -> coefficients ~ 1/2, residual variance ~ h^2/2 = 1/A_ii.

Run from repo root:  wolframscript -file mathematica/fdt_fluctuations.wls
Exports: figures/mma_fdt_covariance.png, figures/mma_fdt_B_2d.png
*)

ok[name_, cond_] := Print[If[cond, "PASS", "FAIL"], ": ", name];
relErr[x_, y_] := Norm[Flatten[Normal[x - y]]]/Norm[Flatten[Normal[y]]];

(* ================= 1. Regression-matrix extraction ================= *)

n = 32;
h = 1.0/(n + 1);

(* --- 1D: user's snippet, exactly in substance --- *)
A = SparseArray[
    {Band[{1, 1}] -> 2., Band[{2, 1}] -> -1., Band[{1, 2}] -> -1.},
    {n, n}]/h^2;
B = IdentityMatrix[n] - Inverse[DiagonalMatrix[Diagonal[A]]] . A;

(* (i) nonzero pattern = chain-neighbor pattern, coefficients 1/2 *)
neighborMask1D = Table[Boole[Abs[i - j] == 1], {i, n}, {j, n}];
offVals1D = Pick[Flatten[B], Flatten[neighborMask1D], 1];
Print["1D B: max |off-diagonal coeff - 1/2| = ",
  Max[Abs[offVals1D - 1/2]],
  ",  max |entry off the neighbor pattern| = ",
  Max[Abs[Pick[Flatten[B], Flatten[neighborMask1D], 0]]]];
ok["1D B nonzeros = chain neighbors, coefficients exactly 1/2",
  Max[Abs[offVals1D - 1/2]] == 0. &&
  Max[Abs[Pick[Flatten[B], Flatten[neighborMask1D], 0]]] < 1.*^-14];

(* (ii) notebook identity A == (I - B).D2, D2 = diag(A) *)
D2 = DiagonalMatrix[Diagonal[A]];
Print["1D notebook identity: relative error ||A - (I-B).D2||_F / ||A||_F = ",
  relErr[(IdentityMatrix[n] - B) . D2, A]];
ok["1D notebook identity A == (I - B).D2 to machine precision",
  relErr[(IdentityMatrix[n] - B) . D2, A] < 1.*^-14];

(* --- 2D via the Kronecker sum --- *)
d1 = SparseArray[
    {Band[{1, 1}] -> 2., Band[{2, 1}] -> -1., Band[{1, 2}] -> -1.},
    {n, n}];
id = IdentityMatrix[n, SparseArray];
A2 = (KroneckerProduct[d1, id] + KroneckerProduct[id, d1])/h^2;  (* N = n^2 *)
nn = n^2;
B2 = IdentityMatrix[nn, SparseArray] -
     Inverse[DiagonalMatrix[Diagonal[A2]]] . A2;   (* Inverse[] densifies *)
B2sp = SparseArray[Chop[B2, 1.*^-14]];             (* recover sparse pattern *)

(* (i) nonzero pattern = 4-neighbor (5-point stencil) pattern, coefficients 1/4 *)
stencilPos = Select[A2["NonzeroPositions"], #[[1]] =!= #[[2]] &]; (* off-diag support *)
offVals2D = B2sp["NonzeroValues"];
maxDiag2D = Max[Abs[Diagonal[B2]]];
Print["2D B: ", Length[B2sp["NonzeroPositions"]],
  " nonzeros (4-neighbor pattern has ", Length[stencilPos],
  "),  max |coeff - 1/4| = ", Max[Abs[offVals2D - 1/4]],
  ",  max |diagonal| = ", maxDiag2D];
ok["2D B nonzeros = 4-neighbor pattern, coefficients exactly 1/4",
  Sort[B2sp["NonzeroPositions"]] === Sort[stencilPos] &&
  Max[Abs[offVals2D - 1/4]] == 0. && maxDiag2D < 1.*^-14];

(* (ii) notebook identity in 2D *)
D22 = DiagonalMatrix[Diagonal[A2]];
Print["2D notebook identity: relative error ||A - (I-B).D2||_F / ||A||_F = ",
  relErr[(IdentityMatrix[nn, SparseArray] - B2) . D22, A2]];
ok["2D notebook identity A == (I - B).D2 to machine precision",
  relErr[(IdentityMatrix[nn, SparseArray] - B2) . D22, A2] < 1.*^-14];

(* (iii) row vs column convention.
   Row (PDE/Jacobi) convention:  B_row = I - Inverse[D].A, row i regresses u_i on the rest.
   Notebook column convention:   B_col = I - A.Inverse[D], column j predicts u_j.
   For symmetric A these are transposes: B_col == Transpose[B_row]; when the
   diagonal of A is constant (2/h^2 in 1D, 4/h^2 in 2D) B is symmetric, so the
   two conventions coincide. *)
ok["1D B symmetric (constant diagonal 2/h^2): row & column conventions coincide",
  relErr[Transpose[B], B] < 1.*^-14];
ok["2D B symmetric (constant diagonal 4/h^2): row & column conventions coincide",
  Max[Abs[B2 - Transpose[B2]]] == 0.];

SeedRandom[7];
X5 = RandomReal[{-1, 1}, {5, 5}];
M5 = X5 . Transpose[X5] + 5.*IdentityMatrix[5];   (* SPD, non-constant diagonal *)
Dm = DiagonalMatrix[Diagonal[M5]];
Brow = IdentityMatrix[5] - Inverse[Dm] . M5;
Bcol = IdentityMatrix[5] - M5 . Inverse[Dm];
Print["random SPD 5x5: ||Bcol - Transpose[Brow]||_F = ",
  Norm[Bcol - Transpose[Brow], "Frobenius"],
  ",  ||Brow - Transpose[Brow]||_F = ",
  Norm[Brow - Transpose[Brow], "Frobenius"], " (B itself not symmetric)"];
ok["general SPD with differing diagonals: B_col == Transpose[B_row]",
  Norm[Bcol - Transpose[Brow], "Frobenius"] < 1.*^-14 &&
  Norm[Brow - Transpose[Brow], "Frobenius"] > 0.01];

(* ================= 2. FDT demo: sample and regress ================= *)

nSamples = 20000;
sigma = Inverse[Normal[A]];               (* Brownian-bridge covariance h(min-st) *)
sigma = (sigma + Transpose[sigma])/2;     (* symmetrize round-off *)
SeedRandom[42];
samples = RandomVariate[
  MultinormalDistribution[ConstantArray[0., n], sigma], nSamples];

empCov = Covariance[samples];
covErr = Norm[empCov - sigma, "Frobenius"]/Norm[sigma, "Frobenius"];
Print["empirical covariance (", nSamples,
  " samples) vs Inverse[A]: relative Frobenius error = ", covErr];
ok["empirical covariance matches Inverse[A] to Monte Carlo accuracy (< 5%)",
  covErr < 0.05];

(* regression of the center coordinate on its two neighbors *)
c = n/2;                                  (* center node, i = 16 *)
y = samples[[All, c]];
Xn = samples[[All, {c - 1, c + 1}]];
beta = LeastSquares[Xn, y];
resid = y - Xn . beta;
residVar = Variance[resid];
Print["regression u_", c, " ~ u_", c - 1, ", u_", c + 1,
  ":  coefficients = ", beta, "  (theory: {1/2, 1/2})"];
Print["residual variance = ", residVar, "  (theory h^2/2 = 1/A_ii = ",
  h^2/2, ")"];
ok["sample regression coefficients ~ {1/2, 1/2} (within 0.02)",
  Max[Abs[beta - 1/2]] < 0.02];
ok["sample residual variance ~ h^2/2 (within 5%)",
  Abs[residVar - h^2/2]/(h^2/2) < 0.05];

(* ================= 3. Figures ================= *)

Export["figures/mma_fdt_covariance.png",
  GraphicsRow[{
    ArrayPlot[sigma, ColorFunction -> "SunsetColors", PlotRangePadding -> None,
      PlotLabel -> "Inverse[A]  (Brownian-bridge covariance)"],
    ArrayPlot[empCov, ColorFunction -> "SunsetColors", PlotRangePadding -> None,
      PlotLabel -> "empirical covariance (20000 samples)"]},
    ImageSize -> 900],
  ImageResolution -> 150];

Export["figures/mma_fdt_B_2d.png",
  ArrayPlot[B2, ImageSize -> 500,
    PlotLabel ->
      "2D regression matrix B = I - diag(A)^-1 A   (n = 32, N = 1024)"],
  ImageResolution -> 150];

Print["Exported figures/mma_fdt_covariance.png, figures/mma_fdt_B_2d.png"];
