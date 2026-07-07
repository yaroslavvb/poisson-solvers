(* ::Package:: *)

(* electrical_networks_grid3.wls -- Vishnoi Ch.4 quantities on the 3x3 GRID
   graph, exact arithmetic, machine-checked.  Companion to
   electrical_networks.wls (same machinery, standalone file); see that file
   for the section-by-section map to the monograph.

   Grid specifics verified here beyond the generic Ch.4 checks:
     - tau(3x3 grid) == 192 spanning trees (classical value, via Matrix-Tree)
     - edges fall into exactly TWO Reff classes by symmetry:
       8 outer edges (corner <-> side-midpoint) and 4 inner (side <-> center)
     - Foster: 8 R_outer + 4 R_inner == n-1 == 8
     - corner-to-opposite-corner Reff computed two independent ways
     - the free grid Laplacian here is the Neumann/ungrounded cousin of the
       suite's Dirichlet operator poisson_2d (grounding the boundary makes L
       nonsingular and turns L^+ quadratic forms into the Green's function
       of reports 09/14)

   Vishnoi sign convention throughout: B_{e,i} = +1 at the TAIL, -1 at the
   HEAD of directed edge e (the negative of Mathematica's IncidenceMatrix,
   checked).  Run:  wolframscript -file mathematica/electrical_networks_grid3.wls *)

nPass = 0; nFail = 0;
ok[name_, cond_] := If[TrueQ[cond],
   nPass++; Print["PASS: ", name],
   nFail++; Print["FAIL: ", name]];

incidenceB[dg_Graph] := With[{vl = VertexList[dg], el = EdgeList[dg]},
   Table[Which[v === e[[1]], 1, v === e[[2]], -1, True, 0], {e, el}, {v, vl}]];

treeCount[L_] := Det[L[[2 ;;, 2 ;;]]];

pathFlow[dg_Graph, path_List] := Module[{el = EdgeList[dg], f},
   f = ConstantArray[0, Length[el]];
   Do[With[{a = path[[k]], b = path[[k + 1]]},
       Module[{iF = FirstPosition[el, DirectedEdge[a, b], {0}][[1]],
               iR = FirstPosition[el, DirectedEdge[b, a], {0}][[1]]},
         If[iF =!= 0, f[[iF]] += 1, f[[iR]] -= 1]]],
      {k, Length[path] - 1}];
   f];

analyze[g0_Graph, name_String, s_, t_] := Module[
   {dg, vl, el, n, m, B, L, Lp, es, et, cext, v, curr, Reff, Pi, eigs,
    tau, pInT, fstar, energy, altPath, falt},
   dg = DirectedGraph[g0, "Acyclic"];
   vl = VertexList[dg]; el = EdgeList[dg];
   n = Length[vl]; m = Length[el];
   Print["\n================  ", name, "  (n = ", n, ", m = ", m,
         ")  ================"];

   B = incidenceB[dg];
   L = Transpose[B] . B;
   Print["edges (chosen orientation): ", el];
   Print["B  = ", MatrixForm[B]];
   Print["L  = B^T.B = ", MatrixForm[L]];
   ok[name <> ": Lemma 4.1, B^T.B == KirchhoffMatrix (Laplacian)",
      L === Normal[KirchhoffMatrix[UndirectedGraph[dg]]]];
   ok[name <> ": Mathematica sign convention is the negative of Vishnoi's",
      B === -Transpose[Normal[IncidenceMatrix[dg]]]];

   Lp = PseudoInverse[L];
   ok[name <> ": L^+ exact (rational) and symmetric",
      And[Element[Flatten[Lp], Rationals], Lp === Transpose[Lp]]];

   es = UnitVector[n, s]; et = UnitVector[n, t];
   cext = es - et;
   v = Lp . cext;
   curr = B . v;
   Print["c_ext = e_", s, " - e_", t, ";  voltages v = L^+ c_ext = ", v];
   Print["edge currents i = B.v = ", curr];
   ok[name <> ": Kirchhoff's law B^T.i == c_ext", Transpose[B] . curr === cext];

   Reff = Table[With[{i = e[[1]], j = e[[2]]},
        (UnitVector[n, VertexIndex[dg, i]] - UnitVector[n, VertexIndex[dg, j]])
        . Lp .
        (UnitVector[n, VertexIndex[dg, i]] - UnitVector[n, VertexIndex[dg, j]])],
      {e, el}];
   Print["Reff per edge = ", Reff];
   Pi = B . Lp . Transpose[B];
   ok[name <> ": Prop 4.2, Pi symmetric", Pi === Transpose[Pi]];
   ok[name <> ": Prop 4.3, Pi^2 == Pi (exact)", Pi . Pi === Pi];
   eigs = Sort[Eigenvalues[Pi]];
   ok[name <> ": Prop 4.4, eigenvalues of Pi are {0,1} with n-1 ones, m-n+1 zeros",
      eigs === Sort[Join[ConstantArray[0, m - n + 1], ConstantArray[1, n - 1]]]];
   ok[name <> ": Pi(e,e) == Reff(e) for every edge", Diagonal[Pi] === Reff];
   ok[name <> ": Foster's theorem, Sum_e Reff(e) == n-1 == tr(Pi)",
      Total[Reff] === n - 1 && Tr[Pi] === n - 1];

   tau = treeCount[L];
   pInT = Table[With[{Bd = Delete[B, ei]},
        1 - treeCount[Transpose[Bd] . Bd]/tau], {ei, m}];
   Print["spanning trees tau(G) = ", tau, ";  P[e in T] = ", pInT];
   ok[name <> ": Thm 4.5, P[e in T] == Reff(e) for every edge", pInT === Reff];

   fstar = B . Lp . cext;
   energy = fstar . fstar;
   Print["unit ", s, ",", t, "-flow f* = B.L^+(e_s - e_t) = ", fstar];
   ok[name <> ": Prop 4.6, E(f*) == (e_s-e_t)^T L^+ (e_s-e_t) == Reff(s,t)",
      energy === cext . Lp . cext];
   ok[name <> ": f* is a unit s,t-flow (B^T.f* == e_s - e_t)",
      Transpose[B] . fstar === cext];
   ok[name <> ": f* lies in the image of Pi (Pi.f* == f*)", Pi . fstar === fstar];
   altPath = Last[SortBy[FindPath[UndirectedGraph[dg], vl[[s]], vl[[t]], Infinity, All],
                         Length]];
   falt = pathFlow[dg, altPath];
   Print["competitor path flow along ", altPath, ": f = ", falt,
         ",  E = ", falt . falt, "  vs  E(f*) = ", energy];
   ok[name <> ": competitor is a unit s,t-flow", Transpose[B] . falt === cext];
   ok[name <> ": Thm 4.7, E(f*) <= E(path flow), via ||Pi f||^2 <= ||f||^2",
      And[energy <= falt . falt,
          (Pi . falt) . (Pi . falt) <= falt . falt,
          (Pi . falt) . (Pi . falt) === cext . Lp . cext]];
   {dg, B, L, Lp, Reff, tau}];

(* ---- the 3x3 grid ------------------------------------------------------- *)
g = GridGraph[{3, 3}];

(* identify structure by degree, independent of Mathematica's vertex order *)
corners = Select[VertexList[g], VertexDegree[g, #] == 2 &];
sides   = Select[VertexList[g], VertexDegree[g, #] == 3 &];
center  = First@Select[VertexList[g], VertexDegree[g, #] == 4 &];
ok["3x3 grid: 4 corners (deg 2), 4 side-midpoints (deg 3), 1 center (deg 4)",
   Length[corners] == 4 && Length[sides] == 4 && IntegerQ[center]];

sCorner = First[corners];
tCorner = First@Select[corners, GraphDistance[g, sCorner, #] == 4 &];
Print["corner-to-opposite-corner demo: s = ", sCorner, ", t = ", tCorner];

{dg, B, L, Lp, Reff, tau} =
  analyze[g, "Grid 3x3", VertexIndex[g, sCorner], VertexIndex[g, tCorner]];

(* ---- grid-specific checks ------------------------------------------------ *)
ok["Grid 3x3: tau == 192 spanning trees (classical value)", tau === 192];

(* exactly two Reff classes: 8 outer (corner-side) edges, 4 inner (side-center) *)
el = EdgeList[dg];
outerQ = Table[MemberQ[corners, e[[1]]] || MemberQ[corners, e[[2]]], {e, el}];
routerVals = DeleteDuplicates@Pick[Reff, outerQ];
rinnerVals = DeleteDuplicates@Pick[Reff, outerQ, False];
Print["Reff classes: outer (corner-side) = ", routerVals,
      " x", Count[outerQ, True],
      ";  inner (side-center) = ", rinnerVals, " x", Count[outerQ, False]];
ok["Grid 3x3: exactly two Reff classes by symmetry (8 outer, 4 inner)",
   Length[routerVals] == 1 && Length[rinnerVals] == 1 &&
   Count[outerQ, True] == 8 && Count[outerQ, False] == 4];
ok["Grid 3x3: Foster by classes, 8 R_outer + 4 R_inner == 8",
   8 First[routerVals] + 4 First[rinnerVals] === 8];
ok["Grid 3x3: outer edges are harder to bypass, R_outer > R_inner",
   First[routerVals] > First[rinnerVals]];

(* corner-to-opposite-corner Reff two independent ways: quadratic form vs the
   voltage drop of the solved network *)
Module[{n9 = 9, si = VertexIndex[g, sCorner], ti = VertexIndex[g, tCorner],
        cext, v, r1, r2},
  cext = UnitVector[n9, si] - UnitVector[n9, ti];
  v = Lp . cext;
  r1 = cext . Lp . cext;
  r2 = v[[si]] - v[[ti]];
  Print["Reff(corner, opposite corner) = ", r1];
  ok["Grid 3x3: Reff(s,t) via quadratic form == voltage drop v_s - v_t",
     r1 === r2];
  ok["Grid 3x3: Reff(corner, center) identical for all four corners (symmetry)",
     Length[DeleteDuplicates@Table[
        With[{ci = VertexIndex[g, c], mi = VertexIndex[g, center]},
          (UnitVector[n9, ci] - UnitVector[n9, mi]) . Lp .
          (UnitVector[n9, ci] - UnitVector[n9, mi])], {c, corners}]] == 1];
  ];

(* ---- Sec 4.4 weighted, on the same grid ---------------------------------- *)
Module[{vl, el9, n9, m9, Bw, w, W, Wh, Lw, Lwp, PiW, si, ti, cext, v, curr,
        ReffW, fstar, energy, tauW, pInT},
  vl = VertexList[dg]; el9 = EdgeList[dg]; n9 = Length[vl]; m9 = Length[el9];
  Print["\n================  Weighted grid 3x3 (Sec 4.4)  ================"];
  Bw = B;
  (* perfect-square conductances, deliberately asymmetric *)
  w = {1, 4, 1, 9, 4, 1, 16, 9, 4, 1, 25, 4}[[;; m9]];
  W = DiagonalMatrix[w]; Wh = DiagonalMatrix[Sqrt[w]];
  Print["weights w = ", w, "  (r_e = 1/w_e)"];
  Lw = Transpose[Bw] . W . Bw;
  Lwp = PseudoInverse[Lw];

  si = VertexIndex[g, sCorner]; ti = VertexIndex[g, tCorner];
  cext = UnitVector[n9, si] - UnitVector[n9, ti];
  v = Lwp . cext; curr = W . Bw . v;
  ok["Weighted grid: Kirchhoff B^T.i == c_ext", Transpose[Bw] . curr === cext];

  ReffW = Table[With[{i = e[[1]], j = e[[2]]},
       (UnitVector[n9, VertexIndex[dg, i]] - UnitVector[n9, VertexIndex[dg, j]])
       . Lwp .
       (UnitVector[n9, VertexIndex[dg, i]] - UnitVector[n9, VertexIndex[dg, j]])],
     {e, el9}];
  PiW = Wh . Bw . Lwp . Transpose[Bw] . Wh;
  ok["Weighted grid: Pi_w symmetric and Pi_w^2 == Pi_w (exact)",
     And[PiW === Transpose[PiW], PiW . PiW === PiW]];
  ok["Weighted grid: Pi_w(e,e) == w_e Reff(e)", Diagonal[PiW] === w ReffW];
  ok["Weighted grid Foster: Sum_e w_e Reff(e) == n-1 == tr(Pi_w)",
     Total[w ReffW] === n9 - 1 && Tr[PiW] === n9 - 1];

  tauW = treeCount[Lw];
  pInT = Table[With[{Wd = ReplacePart[W, {ei, ei} -> 0]},
       1 - treeCount[Transpose[Bw] . Wd . Bw]/tauW], {ei, m9}];
  Print["tau_w(G) = ", tauW, ";  P[e in T] = ", pInT];
  ok["Weighted grid Thm 4.5: P[e in T] == w_e Reff(e) == Pi_w(e,e)",
     pInT === w ReffW];

  fstar = W . Bw . Lwp . cext;
  energy = Total[fstar^2/w];
  Print["weighted corner-to-corner unit flow f* = ", fstar,
        ";  E(f*) = ", energy];
  ok["Weighted grid Prop 4.6: E(f*) == (e_s-e_t)^T L^+ (e_s-e_t)",
     energy === cext . Lwp . cext];
  ok["Weighted grid: f* is a unit s,t-flow", Transpose[Bw] . fstar === cext];
  ];

Print["\n== summary: ", nPass, " PASS, ", nFail, " FAIL =="];
If[nFail > 0, Exit[1]];
