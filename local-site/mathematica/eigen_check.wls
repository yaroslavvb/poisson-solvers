#!/usr/bin/env wolframscript
(* ::Package:: *)

(*
eigen_check.wls

Verifies the discrete Laplacian spectra against the analytic formulas
(see e.g. LeVeque, "Finite Difference Methods for ODEs and PDEs", Sec 2.10):

  1-D (no 1/h^2 scaling):  lam_k = 2 - 2 Cos[k Pi/(n+1)],  k = 1..n
  2-D:                     Lam_{k,l} = (lam_k + lam_l)/h^2

and checks the condition number of A against the analytic prediction
  cond(A) = lam_n/lam_1 = Cot[Pi/(2(n+1))]^2.

Run from repo root:  wolframscript -file mathematica/eigen_check.wls
*)

n = 32;
h = 1.0/(n + 1);

d1 = SparseArray[
  {Band[{1, 1}] -> 2.0, Band[{2, 1}] -> -1.0, Band[{1, 2}] -> -1.0},
  {n, n}];
id = IdentityMatrix[n, SparseArray];
A = (KroneckerProduct[d1, id] + KroneckerProduct[id, d1])/h^2;

(* ---- 1-D check ---- *)
lamAnalytic1d = Table[2.0 - 2.0*Cos[k*Pi/(n + 1)], {k, n}];
eigs1d = Eigenvalues[Normal[d1]];
dev1d = Max[Abs[Sort[eigs1d] - Sort[lamAnalytic1d]]];
Print["1-D: max |eig(d1) - (2 - 2 Cos[k Pi/(n+1)])| = ", dev1d];

(* ---- 2-D check (dense 1024 x 1024, fine at this size) ---- *)
lamAnalytic2d = Flatten@Outer[Plus, lamAnalytic1d, lamAnalytic1d]/h^2;
eigs2d = Eigenvalues[Normal[A]];
dev2d = Max[Abs[Sort[eigs2d] - Sort[lamAnalytic2d]]];
Print["2-D: max |eig(A) - (lam_k + lam_l)/h^2|     = ", dev2d];

(* ---- Condition number ---- *)
condComputed = Max[eigs2d]/Min[eigs2d];
condAnalytic = N[Cot[Pi/(2 (n + 1))]^2];
Print["cond(A) computed = ", condComputed];
Print["cond(A) analytic Cot[Pi/(2(n+1))]^2 = ", condAnalytic];
Print["cond deviation = ", Abs[condComputed - condAnalytic]];
