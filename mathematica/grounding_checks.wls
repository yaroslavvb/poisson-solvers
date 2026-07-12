#!/usr/bin/env wolframscript
(* ::Package:: *)

(*
mathematica/grounding_checks.wls

EXACT-ARITHMETIC companion to the grounding / boundary-condition tutorial:
the n = 6 free path Laplacian L = B^T.B (unit resistors, Vishnoi sign
convention), its pseudoinverse, its grounded inverses GroundedInverse[L,k]
for every k, and the Robin family L + kappa e_1 e_1^T -- all in exact
rationals, no N[] anywhere, so every PASS is an identity.

The user's predicates/functions are reproduced verbatim:
  TriangularRankOneQ[m]   every ALIGNED corner block m[[1;;j, j;;n]]
                          (rows 1..j, cols j..n) has rank <= 1
  TriangularRankQ[m, r]   same with rank <= r -- the SEMISEPARABLE-RANK
                          certificate, identical to report 14's
                          off-diagonal-block rank on the chain
  GroundedInverse[l, k]   delete row/col k, invert, re-embed with a zero
                          row/col at k  (reference electrode at node k)

Verified below:
  1. the two printed findings -- GroundedInverse[L,1] is triangular-rank-1
     and PseudoInverse[L] is triangular-rank-2 -- plus the sharper facts
     that the pseudoinverse is NOT rank-1 and that rank 2 is attained.
  2. GroundedInverse[L,k] is triangular-rank-1 for EVERY k, with exact
     closed form  G_k[i,j] = k - Max[i,j] (i,j < k),  Min[i,j] - k
     (i,j > k),  0 otherwise -- block-diagonal across the separator k.
  3. GroundedInverse[L,1][[2;;,2;;]] == Min[i,j] exactly: the discrete
     Brownian-MOTION kernel (one pin = motion, two pins = bridge).
  4. gauge identity PseudoInverse[L] == C.GroundedInverse[L,k].C for every
     k, with C = I - 11^T/6: all groundings coincide after double
     centering, and the centering is what adds the +1 rank.
  5. R_eff(i,j) = G_ii + G_jj - 2 G_ij == Abs[i-j] from the pseudoinverse
     AND from every grounding: effective resistance is gauge-invariant
     (series unit resistors, cf. electrical_networks.wls Reff(1,6) = 5).
  6. Robin: Inverse[L + kappa e_1 e_1^T] == Min[i,j] - 1 + 1/kappa exactly
     for kappa in {1/10, 1, 10}, triangular-rank-1 for every kappa, single
     pair of AFFINE factors u_i = i - 1 + 1/kappa, v_j = 1 (Gantmacher-
     Krein); kappa -> infinity recovers the grounded inverse, kappa -> 0
     blows up along the constants.

Run from repo root:  wolframscript -file mathematica/grounding_checks.wls
Expected: 26 lines PASS, exit code 0.
*)

nFail = 0; nPass = 0;
ok[name_, cond_] := If[cond === True,
  nPass++; Print["PASS: ", name],
  nFail++; Print["FAIL: ", name]];
printRows[label_, m_] := (Print[label];
  Do[Print["  ", ToString[m[[i]], InputForm]], {i, Length[m]}];);

n = 6;

(* ------------- Step 0: the user's construction, verbatim ------------- *)

Print["== Step 0: n = 6 free path Laplacian L = B^T.B, exact =="];

g = PathGraph[Range[n]];
B = -Transpose@IncidenceMatrix@DirectedGraph[g, "Acyclic"];
   (* Vishnoi sign: row of B is e_u - e_v for edge u -> v *)
L = Transpose[B] . B;              (* free (Neumann) Laplacian *)
B = Normal[B]; L = Normal[L];      (* dense, so === compares structurally *)

(* the user's predicates, verbatim *)
TriangularRankOneQ[m_] := AllTrue[Range[n], MatrixRank[m[[1 ;; #, # ;; n]]] <= 1 &];
TriangularRankQ[m_, r_] := AllTrue[Range[n], MatrixRank[m[[1 ;; #, # ;; n]]] <= r &];

(* the user's grounded inverse, verbatim: delete row/col k, invert,
   re-embed with a zero row/col at k *)
GroundedInverse[l_, k_] := Module[{keep = Delete[Range[Length[l]], k], out},
  out = ConstantArray[0, {Length[l], Length[l]}];
  out[[keep, keep]] = Inverse[l[[keep, keep]]];
  out];

printRows["B rows (edge-vertex incidence, Vishnoi sign):", B];
printRows["L rows (free Laplacian):", L];

ok["B rows are exactly e_i - e_{i+1} for the 5 path edges",
  B === Table[UnitVector[n, i] - UnitVector[n, i + 1], {i, n - 1}]];
ok["Diagonal[L] == {1,2,2,2,2,1}  (degree-1 corners ARE the insulation)",
  Diagonal[L] === {1, 2, 2, 2, 2, 1}];
ok["zero row sums: L.ones == 0  (constants in the kernel, Neumann)",
  L . ConstantArray[1, n] === ConstantArray[0, n]];
ok["MatrixRank[L] == 5  (kernel is exactly the constants)",
  MatrixRank[L] === 5];

(* ------------- Step 1: the two printed findings, plus sharper ------------- *)

Print[""];
Print["== Step 1: the two findings -- grounded inverse rank 1, pseudoinverse rank 2 =="];

G1 = GroundedInverse[L, 1];
Lp = PseudoInverse[L];             (* exact rationals *)

printRows["GroundedInverse[L,1] rows:", G1];
printRows["PseudoInverse[L] rows (InputForm rationals):", Lp];

ok["FINDING 1: TriangularRankOneQ[GroundedInverse[L,1]] == True",
  TriangularRankOneQ[G1] === True];
ok["FINDING 2: TriangularRankQ[PseudoInverse[L], 2] == True",
  TriangularRankQ[Lp, 2] === True];
ok["sharper: TriangularRankOneQ[PseudoInverse[L]] == False  (not rank 1)",
  TriangularRankOneQ[Lp] === False];
ok["sharper: some corner block of PseudoInverse[L] has rank EXACTLY 2",
  Max[Table[MatrixRank[Lp[[1 ;; j, j ;; n]]], {j, n}]] === 2];

(* ------- Step 2: every grounding is rank 1, with exact closed form ------- *)

Print[""];
Print["== Step 2: GroundedInverse[L,k] for every k -- rank 1, Markov separator split =="];

(* grounding at k splits the chain at the separator k: each block is a
   Brownian-motion kernel run AWAY from k, and the i<k<j corner is zero *)
gkClosed[k_] := Table[Which[
    i < k && j < k, k - Max[i, j],
    i > k && j > k, Min[i, j] - k,
    True, 0], {i, n}, {j, n}];

rank1AllK = True; closedAllK = True; ginvAllK = True; blockAllK = True;
Do[
  gk = GroundedInverse[L, k];
  rank1AllK = rank1AllK && TriangularRankOneQ[gk];
  closedAllK = closedAllK && (gk === gkClosed[k]);
  ginvAllK = ginvAllK && (L . gk . L === L);
  blockAllK = blockAllK &&
    AllTrue[Flatten[Table[gk[[i, j]], {i, 1, k - 1}, {j, k + 1, n}]], # === 0 &],
  {k, n}];

printRows["GroundedInverse[L,3] rows (interior grounding, block diagonal):",
  GroundedInverse[L, 3]];

ok["TriangularRankOneQ[GroundedInverse[L,k]] == True for EVERY k = 1..6",
  rank1AllK];
ok["closed form G_k[i,j] == k-Max[i,j] | Min[i,j]-k | 0 for every k (exact)",
  closedAllK];
ok["interior grounding splits the chain: G_k[i,j] == 0 whenever i < k < j",
  blockAllK];
ok["each G_k is a g-inverse: L.G_k.L == L exactly for every k",
  ginvAllK];

(* ------------- Step 3: one pin = Brownian motion, exact integers ------------- *)

Print[""];
Print["== Step 3: GroundedInverse[L,1] is the discrete Brownian-motion kernel =="];

minMat = Table[Min[i, j], {i, n - 1}, {j, n - 1}];
printRows["GroundedInverse[L,1][[2;;,2;;]] rows (should be Min[i,j]):",
  G1[[2 ;;, 2 ;;]]];
ok["GroundedInverse[L,1][[2;;,2;;]] == Table[Min[i,j]] exactly (integers)",
  G1[[2 ;;, 2 ;;]] === minMat];
ok["equivalently G_1[i,j] == Min[i,j] - 1 on all of 1..6 (zero row/col 1)",
  G1 === Table[Min[i, j] - 1, {i, n}, {j, n}]];

(* ------------- Step 4: the gauge identity L^+ == C.G_k.C ------------- *)

Print[""];
Print["== Step 4: gauge identity -- all groundings coincide after centering =="];

CC = IdentityMatrix[n] - ConstantArray[1/6, {n, n}];
printRows["C = I - 11^T/6 rows:", CC];

gaugeAllK = True;
Do[gaugeAllK = gaugeAllK && (Lp === CC . GroundedInverse[L, k] . CC), {k, n}];
ok["PseudoInverse[L] == C.GroundedInverse[L,k].C exactly for EVERY k",
  gaugeAllK];
ok["Lp is the mean-zero gauge: Lp.ones == 0 and ones.Lp == 0",
  Lp . ConstantArray[1, n] === ConstantArray[0, n] &&
  ConstantArray[1, n] . Lp === ConstantArray[0, n]];

(* ------------- Step 5: R_eff is grounding-invariant, == |i-j| ------------- *)

Print[""];
Print["== Step 5: effective resistance R_eff(i,j) == Abs[i-j], any gauge =="];

reff[G_, i_, j_] := G[[i, i]] + G[[j, j]] - 2 G[[i, j]];
distMat = Table[Abs[i - j], {i, n}, {j, n}];

ok["R_eff from PseudoInverse[L] == Abs[i-j] for all pairs (series resistors)",
  Table[reff[Lp, i, j], {i, n}, {j, n}] === distMat];
ok["R_eff from GroundedInverse[L,1] == Abs[i-j] for all pairs",
  Table[reff[GroundedInverse[L, 1], i, j], {i, n}, {j, n}] === distMat];
ok["R_eff from GroundedInverse[L,4] == Abs[i-j] for all pairs",
  Table[reff[GroundedInverse[L, 4], i, j], {i, n}, {j, n}] === distMat];
ok["R_eff identical from ALL SIX groundings AND the pseudoinverse",
  AllTrue[Range[n],
    Table[reff[GroundedInverse[L, #], i, j], {i, n}, {j, n}] === distMat &]];
ok["R_eff(1,6) == 5  (electrical_networks.wls: path P6 series value)",
  reff[Lp, 1, 6] === 5];

(* ------------- Step 6: Robin -- grounding through a resistor ------------- *)

Print[""];
Print["== Step 6: Robin family L + kappa e_1 e_1^T, kappa in {1/10, 1, 10} =="];

robinRank1 = True; robinClosed = True; robinFactored = True;
Do[
  MK = L + kappa Outer[Times, UnitVector[n, 1], UnitVector[n, 1]];
  RK = Inverse[MK];                (* exact rationals *)
  robinRank1 = robinRank1 && TriangularRankOneQ[RK];
  (* closed form: Green matrix Min[i,j] - 1 + 1/kappa; the 1/kappa shift
     along ones^T ones is the leak through the grounding resistor *)
  robinClosed = robinClosed &&
    (RK === Table[Min[i, j] - 1 + 1/kappa, {i, n}, {j, n}]);
  (* single-pair (Gantmacher-Krein) factorization on the upper triangle:
     u_i = i - 1 + 1/kappa satisfies the Robin row (1+kappa)u_1 == u_2,
     v_j = 1 satisfies the free right end v_5 == v_6; both AFFINE in the
     index, intercept of u depends on kappa *)
  uK = Table[i - 1 + 1/kappa, {i, n}];
  vK = ConstantArray[1, n];
  robinFactored = robinFactored &&
    AllTrue[Flatten[Table[{i, j}, {i, n}, {j, i, n}], 1],
      RK[[#[[1]], #[[2]]]] === uK[[#[[1]]]] vK[[#[[2]]]] &] &&
    ((1 + kappa) uK[[1]] === uK[[2]]) && (vK[[5]] === vK[[6]]),
  {kappa, {1/10, 1, 10}}];

printRows["Inverse[L + (1/10) e_1 e_1^T] rows:",
  Inverse[L + (1/10) Outer[Times, UnitVector[n, 1], UnitVector[n, 1]]]];
printRows["Inverse[L + 1 e_1 e_1^T] rows (== Min[i,j]):",
  Inverse[L + 1 Outer[Times, UnitVector[n, 1], UnitVector[n, 1]]]];
printRows["Inverse[L + 10 e_1 e_1^T] rows:",
  Inverse[L + 10 Outer[Times, UnitVector[n, 1], UnitVector[n, 1]]]];

ok["TriangularRankOneQ[Inverse[L + kappa e_1 e_1^T]] for kappa = 1/10, 1, 10",
  robinRank1];
ok["closed form Inverse == Min[i,j] - 1 + 1/kappa exactly, all three kappa",
  robinClosed];
ok["upper triangle factors as AFFINE u_i v_j with u_i = i-1+1/kappa, v_j = 1",
  robinFactored];
Print["  exhibited pair for kappa = 1:  u = ",
  ToString[Range[n], InputForm], ",  v = ",
  ToString[ConstantArray[1, n], InputForm],
  "  (G[i,j] = u[Min] v[Max] = Min[i,j])"];

(* Robin interpolates the 0-pin and 1-pin ends of the family *)
ok["kappa -> infinity limit: Inverse - (1/kappa) 11^T == GroundedInverse[L,1] for kappa = 1/10, 1, 10",
  AllTrue[{1/10, 1, 10},
    Inverse[L + # Outer[Times, UnitVector[n, 1], UnitVector[n, 1]]] -
        (1/#) ConstantArray[1, {n, n}] === G1 &]];
ok["kappa -> 0 blow-up is along the constants: Inverse == G_1 + (1/kappa) 11^T",
  With[{kappa = 1/10},
    Inverse[L + kappa Outer[Times, UnitVector[n, 1], UnitVector[n, 1]]] ===
      G1 + (1/kappa) ConstantArray[1, {n, n}]]];

(* ---------------------------------- summary ---------------------------------- *)

Print[""];
Print[nPass, " PASS, ", nFail, " FAIL (all arithmetic exact rational)"];
Exit[If[nFail > 0, 1, 0]];
