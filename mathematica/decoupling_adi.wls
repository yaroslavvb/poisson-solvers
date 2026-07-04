#!/usr/bin/env wolframscript
(* decoupling_adi.wls — Kronecker separability + single-shift ADI companion (n = 32).

   House convention (reports 09..12): h = 1/(n+1), d1 = tridiag(-1,2,-1),
   A = (kron(d1,I) + kron(I,d1))/h^2.  Here H = kron(I,d1)/h^2 (horizontal
   couplings), V = kron(d1,I)/h^2 (vertical couplings).

   1. A == H + V exactly, and H.V == V.H exactly: a Kronecker sum of commuting
      terms is simultaneously diagonalizable — the algebraic meaning of
      "separable".
   2. With sigma = Sqrt[lam_1 lam_n], the single-shift ADI preconditioner
      Minv = 2 sigma (H + sigma I)^-1 (V + sigma I)^-1 has preconditioned
      spectrum 2 sigma (lam_i + lam_j)/((lam_i + sigma)(lam_j + sigma));
      kappa drops from 440.69 to ~ Sqrt[kappa]/2 (the square-root effect).
   3. Semiseparable 1D check: (d1/h^2)^-1_{ij} = h x_i (1 - x_j) for i <= j
      (the Brownian-bridge covariance h(min(s,t) - s t) on the grid).
   4. Figure: sorted spectrum of A (rescaled) vs sorted spectrum of Minv.A.

   Run from repo root:  wolframscript -file mathematica/decoupling_adi.wls  *)

fails = 0;
check[name_, cond_] := (If[cond, Print["PASS: ", name], fails++; Print["FAIL: ", name]];);

repo = If[$InputFileName =!= "", DirectoryName[ExpandFileName[$InputFileName], 2], Directory[]];

n = 32; h = 1/(n + 1);              (* exact rational *)
id = IdentityMatrix[n, SparseArray];
d1 = SparseArray[{Band[{1, 1}] -> 2, Band[{1, 2}] -> -1, Band[{2, 1}] -> -1}, {n, n}];

(* ---- 1. Kronecker sum: A = H + V with [H, V] = 0 (exact arithmetic) ---- *)
A = (KroneckerProduct[d1, id] + KroneckerProduct[id, d1])/h^2;
H = KroneckerProduct[id, d1]/h^2;   (* couples neighbors within a row  *)
V = KroneckerProduct[d1, id]/h^2;   (* couples neighbors across rows   *)

devSum     = Max[Abs[Normal[A - (H + V)]]];          (* exact rationals -> exact 0 *)
devCommute = Max[Abs[Normal[H.V - V.H]]];
Print["max|A - (H+V)|   (exact) = ", devSum];
Print["max|H.V - V.H|   (exact) = ", devCommute];
check["A == H + V exactly (Kronecker sum)", devSum === 0];
check["H.V == V.H exactly (commute => simultaneously diagonalizable = separable)", devCommute === 0];

(* ---- 2. Single-shift ADI: Minv = 2 sigma (H+sigma)^-1 (V+sigma)^-1 ---- *)
lam[k_] := (2 - 2 Cos[k Pi h])/h^2;                  (* 1D DST-I eigenvalues *)
sigma = Sqrt[lam[1] lam[n]];                          (* geometric-mean shift *)
Print["lam_1 = ", N[lam[1]], "   lam_n = ", N[lam[n]], "   sigma = Sqrt[lam_1 lam_n] = ", N[sigma]];

An = N[Normal[A]]; Hn = N[Normal[H]]; Vn = N[Normal[V]];
sig = N[sigma]; id2 = IdentityMatrix[n^2];
Minv = 2 sig Inverse[Hn + sig id2].Inverse[Vn + sig id2];
MA = Minv.An;

(* H, V, A all commute, so Minv.A is symmetric up to roundoff *)
asym = Max[Abs[MA - Transpose[MA]]];
Print["max|Minv.A - (Minv.A)^T| = ", ScientificForm[asym, 3]];
eigPre = Sort[Eigenvalues[(MA + Transpose[MA])/2]];   (* dense symmetric eigensolve *)

closed = Sort[N[Flatten[Table[
    2 sigma (lam[i] + lam[j])/((lam[i] + sigma) (lam[j] + sigma)), {i, n}, {j, n}]]]];
relDev = Max[Abs[(eigPre - closed)/closed]];
Print["max rel dev, eig(Minv.A) vs closed form 2s(li+lj)/((li+s)(lj+s)) = ", ScientificForm[relDev, 3]];
check["preconditioned spectrum matches closed form (rel dev < 1e-10)", relDev < 10^-10];

eigA = Sort[Eigenvalues[An]];
kappaBefore = Last[eigA]/First[eigA];
kappaAfter  = Last[eigPre]/First[eigPre];
Print["kappa(A)        = ", N[kappaBefore, 6], "   (house value 440.69; = lam_n/lam_1)"];
Print["kappa(Minv.A)   = ", N[kappaAfter, 6]];
Print["Sqrt[kappa(A)]/2 = ", N[Sqrt[kappaBefore]/2, 6], "   <- square-root effect: one ADI double-sweep"];
check["kappa(A) == 440.69 (2 dp)", Abs[kappaBefore - 440.69] < 0.005];
check["kappa(Minv.A) ~ Sqrt[kappa]/2  (within 2%)", Abs[kappaAfter/(Sqrt[kappaBefore]/2) - 1] < 0.02];

(* ---- 3. Semiseparable inverse in 1D: Green's function of d1/h^2 ---- *)
T = Normal[d1]/h^2;                                   (* exact 1D operator *)
S = Inverse[T];                                       (* exact rational inverse *)
x = Table[i h, {i, n}];
devSS = Max[Table[Abs[S[[i, j]] - h x[[i]] (1 - x[[j]])], {i, n}, {j, i, n}]];  (* upper triangle *)
Print["max|Inverse(d1/h^2)_ij - h x_i (1 - x_j)|, i<=j  (exact) = ", devSS];
check["1D inverse is semiseparable: h x_i (1-x_j) on upper triangle", devSS === 0];

(* ---- 4. Figure: clustering of the preconditioned spectrum ---- *)
figPath = FileNameJoin[{repo, "figures", "mma_decoupling_spectrum.png"}];
plot = ListLogPlot[{eigA/Last[eigA], eigPre},
   PlotStyle -> {Directive[ColorData[97, 1], PointSize[0.006]],
                 Directive[ColorData[97, 2], PointSize[0.006]]},
   PlotLegends -> Placed[{"eig(A) / max eig(A)   (kappa = 440.69)",
                          "eig(M\!\(\*SuperscriptBox[\(\), \(-1\)]\)A), single-shift ADI (kappa = 10.5)"}, {0.62, 0.25}],
   AxesLabel -> {"index (sorted)", "eigenvalue (log)"},
   PlotLabel -> "Single-shift ADI: spectrum collapses from 3 decades to [0.17, 1.83]",
   PlotRange -> All, ImageSize -> 700];
Export[figPath, plot, ImageResolution -> 150];
Print["wrote ", figPath, "  (", FileByteCount[figPath], " bytes)"];
check["figure exported nonzero", FileExistsQ[figPath] && FileByteCount[figPath] > 0];

Print[If[fails === 0, "ALL CHECKS PASS (7/7)", ToString[fails] <> " CHECK(S) FAILED"]];
Exit[If[fails === 0, 0, 1]];
