#!/usr/bin/env wolframscript
(* ::Package:: *)

(*
poisson_pcg.wls

2-D Poisson problem -Laplace[u] = b on the unit square, Dirichlet BCs,
discretized with the standard 5-point stencil on an n x n interior grid.
Right-hand side b is a Gaussian random field with Matern-like spectrum
1/(|k|^2 + tau^2)^alpha (cf. Li et al., "Fourier Neural Operator for
Parametric PDEs", 2020, Appendix: GRF sampling).

Solves with (preconditioned) conjugate gradients:
  - no preconditioner (Identity)
  - Jacobi (diagonal) preconditioner
PCG recursion follows Saad, "Iterative Methods for Sparse Linear
Systems" (2nd ed.), Algorithm 9.1.

Run from repo root:  wolframscript -file mathematica/poisson_pcg.wls
Exports: figures/mma_source_grf.png, figures/mma_solution.png,
         figures/mma_convergence.png
*)

(* ---------------- Problem setup ---------------- *)

n = 32;
h = 1.0/(n + 1);

(* 1-D Dirichlet Laplacian, tridiagonal (2, -1, -1), no 1/h^2 factor *)
d1 = SparseArray[
  {Band[{1, 1}] -> 2.0, Band[{2, 1}] -> -1.0, Band[{1, 2}] -> -1.0},
  {n, n}];
id = IdentityMatrix[n, SparseArray];

(* 2-D Laplacian: A = (d1 (x) I + I (x) d1) / h^2, SPD, size n^2 *)
A = (KroneckerProduct[d1, id] + KroneckerProduct[id, d1])/h^2;

(* ---------------- Gaussian random field RHS ---------------- *)

alpha = 2.0;
tau = 3.0;
freqs = N@RotateRight[Range[-n/2, n/2 - 1], n/2]*n;
spectrum = 1.0/(Outer[Plus, freqs^2, freqs^2] + tau^2)^(alpha/2.0);
SeedRandom[42];
noise = RandomVariate[NormalDistribution[], {n, n, 2}] . {1, I};
grfRaw = Re@InverseFourier[noise*spectrum, FourierParameters -> {-1, -1}];
b = Standardize@Flatten@grfRaw;
grf = ArrayReshape[b, {n, n}];

(* ---------------- Preconditioned conjugate gradients ---------------- *)

(*
PCGSolve[matA, vecB, precondFunc] -> {x, resHist}
precondFunc: r -> z ~ matA^-1 . r.  resHist starts at 1.0, then one
relative residual ||r_k||/||b|| per iteration (collected via Sow/Reap).
*)
PCGSolve[matA_, vecB_, precondFunc_, maxIter_ : 1000, tol_ : 10^-10] :=
  Module[{bNorm = Norm[vecB], pcgStep, result},
    pcgStep[{x_, r_, p_, rz_, _}] :=
      Module[{Ap = matA . p, aStep, xNew, rNew, zNew, rzNew, relRes},
        aStep = rz/(p . Ap);          (* alpha_k = (r.z)/(p.Ap) *)
        xNew = x + aStep*p;
        rNew = r - aStep*Ap;
        relRes = Norm[rNew]/bNorm;
        Sow[relRes];
        zNew = precondFunc[rNew];
        rzNew = rNew . zNew;
        {xNew, rNew, zNew + (rzNew/rz)*p, rzNew, relRes}];
    result = Reap[
      Sow[1.0];
      NestWhile[pcgStep,
        {0.*vecB, vecB, precondFunc[vecB], vecB . precondFunc[vecB], 1.0},
        (#[[5]] > tol &), 1, maxIter]];
    {result[[1, 1]], result[[2, 1]]}];

(* ---------------- Solves ---------------- *)

{xNone, resNone} = PCGSolve[A, b, Identity];
Print["CG (no preconditioner):  ", Length[resNone] - 1,
  " iterations, final relres = ", Last[resNone]];

(* Normal: keep the reciprocal off the sparse background element *)
invDiag = 1.0/Normal[Diagonal[A]];
{xJacobi, resJacobi} = PCGSolve[A, b, invDiag*# &];
Print["PCG (Jacobi):            ", Length[resJacobi] - 1,
  " iterations, final relres = ", Last[resJacobi]];

Print["Solution mismatch ||xNone - xJacobi|| = ", Norm[xNone - xJacobi]];

(* ---------------- Figures ---------------- *)

(* Wolfram 15.0 has no built-in "Viridis"; Blend of matplotlib anchors *)
viridis = Blend[{RGBColor[0.267, 0.005, 0.329], RGBColor[0.283, 0.157, 0.471],
    RGBColor[0.243, 0.286, 0.537], RGBColor[0.192, 0.407, 0.556],
    RGBColor[0.149, 0.510, 0.557], RGBColor[0.122, 0.620, 0.537],
    RGBColor[0.208, 0.718, 0.475], RGBColor[0.431, 0.808, 0.345],
    RGBColor[0.993, 0.906, 0.144]}, #] &;

Export["figures/mma_source_grf.png",
  ArrayPlot[grf, ColorFunction -> viridis, ImageSize -> 450,
    PlotLabel -> "GRF source term b"],
  ImageResolution -> 150];

Export["figures/mma_solution.png",
  ArrayPlot[ArrayReshape[xJacobi, {n, n}], ColorFunction -> "SunsetColors",
    ImageSize -> 450, PlotLabel -> "Solution u (Jacobi PCG)"],
  ImageResolution -> 150];

Export["figures/mma_convergence.png",
  ListLogPlot[{resNone, resJacobi}, Joined -> True,
    PlotLegends -> {"No preconditioner", "Jacobi"},
    Frame -> True, FrameLabel -> {"Iteration", "Relative Residual"},
    ImageSize -> 450, PlotRange -> All],
  ImageResolution -> 150];

Print["Exported figures/mma_source_grf.png, figures/mma_solution.png, figures/mma_convergence.png"];
