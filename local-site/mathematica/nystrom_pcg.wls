#!/usr/bin/env wolframscript
(* ::Package:: *)

(*
nystrom_pcg.wls

Randomized Nystrom preconditioner for CG, following Frangella, Tropp &
Udell, "Randomized Nystrom Preconditioning", SIAM J. Matrix Anal. Appl.
44(2), 2023:
  - Algorithm 2.1 (stabilized single-pass Nystrom approximation
    A_nys = U diag(lams) U^T from sketch Y = A.Omega),
  - Eq. (5.3) preconditioner apply (P itself is Eq. (1.3)) with mu = 0:
      P^-1 r = lamL * U.((U^T r)/lams) + (r - U.(U^T r)),
    where lamL is the smallest retained positive Nystrom eigenvalue.

Same 32 x 32 Poisson problem and PCG driver as poisson_pcg.wls.

Run from repo root:  wolframscript -file mathematica/nystrom_pcg.wls
Exports: figures/mma_nystrom_convergence.png
*)

(* ---------------- Problem setup (identical to poisson_pcg.wls) ---------------- *)

n = 32;
h = 1.0/(n + 1);
d1 = SparseArray[
  {Band[{1, 1}] -> 2.0, Band[{2, 1}] -> -1.0, Band[{1, 2}] -> -1.0},
  {n, n}];
id = IdentityMatrix[n, SparseArray];
A = (KroneckerProduct[d1, id] + KroneckerProduct[id, d1])/h^2;

alpha = 2.0;
tau = 3.0;
freqs = N@RotateRight[Range[-n/2, n/2 - 1], n/2]*n;
spectrum = 1.0/(Outer[Plus, freqs^2, freqs^2] + tau^2)^(alpha/2.0);
SeedRandom[42];
noise = RandomVariate[NormalDistribution[], {n, n, 2}] . {1, I};
grfRaw = Re@InverseFourier[noise*spectrum, FourierParameters -> {-1, -1}];
b = Standardize@Flatten@grfRaw;

(* ---------------- PCG driver (identical to poisson_pcg.wls) ---------------- *)

PCGSolve[matA_, vecB_, precondFunc_, maxIter_ : 1000, tol_ : 10^-10] :=
  Module[{bNorm = Norm[vecB], pcgStep, result},
    pcgStep[{x_, r_, p_, rz_, _}] :=
      Module[{Ap = matA . p, aStep, xNew, rNew, zNew, rzNew, relRes},
        aStep = rz/(p . Ap);
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

(* ---------------- Randomized Nystrom preconditioner ---------------- *)

(*
NystromPreconditioner[matA, ell, seed] -> {applyFunc, lams}
Stabilized single-pass Nystrom (Frangella-Tropp-Udell Alg. 2.1):
Gaussian sketch Omega (N x ell); Y = A.Omega; shift
nu = Sqrt[N] eps ||Y||_F; B = Y_nu C^-1 with C^T C = Omega^T Y_nu
(Cholesky); eigenpairs from thin SVD of B; lams = sigma^2 - nu, clipped
at 0.  Rank-deficient directions (lams == 0) are dropped from U.
*)
NystromPreconditioner[matA_, ell_, seed_ : 0] :=
  Module[{nn = Length[matA], Omega, Y, nu, Ynu, cholC, B, U, S, V,
      lams, pos, lamL},
    SeedRandom[seed];
    Omega = RandomVariate[NormalDistribution[], {nn, ell}];
    Y = matA . Omega;
    nu = Sqrt[nn]*$MachineEpsilon*Norm[Y, "Frobenius"];
    Ynu = Y + nu*Omega;
    (* Omega^T.Ynu is symmetric in exact arithmetic; symmetrize for Cholesky *)
    cholC = CholeskyDecomposition[
      (Transpose[Omega] . Ynu + Transpose[Transpose[Omega] . Ynu])/2];
    (* B = Ynu.Inverse[cholC] via triangular solve: cholC^T.B^T = Ynu^T *)
    B = Transpose[LinearSolve[Transpose[cholC], Transpose[Ynu]]];
    {U, S, V} = SingularValueDecomposition[B, ell];
    lams = Map[Max[0.0, #] &, Diagonal[S]^2 - nu];
    pos = Flatten[Position[lams, _?(# > 0 &)]];
    U = U[[All, pos]];
    lams = lams[[pos]];
    lamL = Min[lams];   (* mu = 0: shift by smallest retained eigenvalue *)
    (* Slot syntax (not Function[r, ...]): a named parameter would be
       captured by the pattern variable r_ inside PCGSolve's pcgStep. *)
    {With[{ur = Transpose[U] . #}, lamL*U . (ur/lams) + (# - U . ur)] &,
     lams}];

(* ---------------- Solves and comparison ---------------- *)

ell = 128;
{nysApply, nysLams} = NystromPreconditioner[A, ell, 0];
Print["Nystrom sketch size ell = ", ell, ", retained rank = ",
  Length[nysLams], ", lam range = [", Min[nysLams], ", ", Max[nysLams], "]"];

{xNone, resNone} = PCGSolve[A, b, Identity];
Print["CG (no preconditioner):  ", Length[resNone] - 1,
  " iterations, final relres = ", Last[resNone]];

{xNys, resNys} = PCGSolve[A, b, nysApply];
Print["PCG (Nystrom, ell=128):  ", Length[resNys] - 1,
  " iterations, final relres = ", Last[resNys]];

Print["Solution mismatch ||xNone - xNys|| = ", Norm[xNone - xNys]];

(* ---------------- Figure ---------------- *)

Export["figures/mma_nystrom_convergence.png",
  ListLogPlot[{resNone, resNys}, Joined -> True,
    PlotLegends -> {"No preconditioner", "Nystrom (ell=128)"},
    Frame -> True, FrameLabel -> {"Iteration", "Relative Residual"},
    ImageSize -> 450, PlotRange -> All],
  ImageResolution -> 150];

Print["Exported figures/mma_nystrom_convergence.png"];
