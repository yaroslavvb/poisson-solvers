(* ::Package:: *)

(* electrical_networks.wls -- Graphs as electrical networks: every quantity of
   Vishnoi, "Lx = b" (Foundations and Trends in TCS), Chapter 4, in EXACT
   arithmetic, with machine checks.

   Quantities implemented (section numbers from the monograph):
     4.1  incidence matrix B (Vishnoi sign convention), Laplacian L = B^T.B
          (Lemma 4.1), network solve v = L^+ c_ext, Ohm i = B.v,
          Kirchhoff B^T.i = c_ext
     4.2  effective resistances Reff(e) = (e_i-e_j)^T L^+ (e_i-e_j),
          the projection Pi = B.L^+.B^T (Props 4.2-4.4: symmetric, Pi^2 = Pi,
          eigenvalues in {0,1}, rank n-1), Pi(e,e) = Reff(e),
          spanning-tree marginals P[e in T] = Reff(e) (Thm 4.5, verified via
          Matrix-Tree determinants and edge deletion),
          Foster's theorem Sum_e Reff(e) = n-1 (= tr Pi = rank Pi)
     4.3  electrical unit s,t-flow fstar = B.L^+.(e_s-e_t), its energy
          E(fstar) = Reff(s,t) (Prop 4.6), and Thm 4.7: fstar minimizes
          energy among all unit s,t-flows (explicit competitor checked)
     4.4  weighted graphs: L = B^T.W.B, Pi_w = W^(1/2).B.L^+.B^T.W^(1/2),
          i = W.B.v, f*_w = W.B.L^+.(e_s-e_t),
          E(f*_w) = Sum_e f_e^2/w_e = Reff(s,t), P[e in T] = w_e Reff(e)

   Sign convention: Vishnoi's B has +1 at the TAIL and -1 at the HEAD of each
   directed edge; Mathematica's IncidenceMatrix uses the opposite sign, so we
   build B directly from the edge list (and check B == -IncidenceMatrix^T).
   All quantities below are invariant under B -> -B except the per-edge signs
   of currents/flows, which simply follow the chosen orientation.

   Relation to the poisson-solvers suite: for the path graph, L is the FREE
   (ungrounded) 1-D Laplacian d1 with the two boundary rows relaxed -- singular
   with kernel 1, hence the pseudoinverse; the reports' A = d1/h^2 is the
   grounded (Dirichlet) version whose inverse is the Brownian-bridge covariance
   (report 09).  Reff(s,t) here plays the role the Green's function plays there.

   Run:  wolframscript -file mathematica/electrical_networks.wls            *)

nPass = 0; nFail = 0;
ok[name_, cond_] := If[TrueQ[cond],
   nPass++; Print["PASS: ", name],
   nFail++; Print["FAIL: ", name]];

(* ---- 4.1: incidence matrix, Vishnoi convention ------------------------- *)
incidenceB[dg_Graph] := With[{vl = VertexList[dg], el = EdgeList[dg]},
   Table[Which[v === e[[1]], 1, v === e[[2]], -1, True, 0], {e, el}, {v, vl}]];

(* Matrix-Tree: number (weighted: generating polynomial) of spanning trees *)
treeCount[L_] := Det[L[[2 ;;, 2 ;;]]];

(* one simple s,t-path turned into a unit flow on the oriented edges *)
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

   (* B and L = B^T.B (Lemma 4.1) *)
   B = incidenceB[dg];
   L = Transpose[B] . B;
   Print["edges (chosen orientation): ", el];
   Print["B  = ", MatrixForm[B]];
   Print["L  = B^T.B = ", MatrixForm[L]];
   ok[name <> ": Lemma 4.1, B^T.B == KirchhoffMatrix (Laplacian)",
      L === Normal[KirchhoffMatrix[UndirectedGraph[dg]]]];
   ok[name <> ": Mathematica sign convention is the negative of Vishnoi's",
      B === -Transpose[Normal[IncidenceMatrix[dg]]]];

   (* exact pseudoinverse; rows/cols of B are orthogonal to 1 *)
   Lp = PseudoInverse[L];
   ok[name <> ": L^+ exact (rational) and symmetric",
      And[Element[Flatten[Lp], Rationals], Lp === Transpose[Lp]]];

   (* network solve: unit current in at s, out at t *)
   es = UnitVector[n, s]; et = UnitVector[n, t];
   cext = es - et;
   v = Lp . cext;            (* voltages, defined up to an additive constant *)
   curr = B . v;             (* Ohm's law, unit resistances: i = B.v         *)
   Print["c_ext = e_", s, " - e_", t, ";  voltages v = L^+ c_ext = ", v];
   Print["edge currents i = B.v = ", curr];
   ok[name <> ": Kirchhoff's law B^T.i == c_ext", Transpose[B] . curr === cext];

   (* 4.2: effective resistances and Pi *)
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

   (* Thm 4.5: spanning-tree marginals via Matrix-Tree + edge deletion:
      P[e in T] = 1 - tau(G - e)/tau(G)  ==  Reff(e)                       *)
   tau = treeCount[L];
   pInT = Table[With[{Bd = Delete[B, ei]},
        1 - treeCount[Transpose[Bd] . Bd]/tau], {ei, m}];
   Print["spanning trees tau(G) = ", tau, ";  P[e in T] = ", pInT];
   ok[name <> ": Thm 4.5, P[e in T] == Reff(e) for every edge", pInT === Reff];

   (* 4.3: electrical flow, energy, optimality *)
   fstar = B . Lp . cext;
   energy = fstar . fstar;
   Print["unit ", s, ",", t, "-flow f* = B.L^+(e_s - e_t) = ", fstar];
   ok[name <> ": Prop 4.6, E(f*) == (e_s-e_t)^T L^+ (e_s-e_t) == Reff(s,t)",
      energy === cext . Lp . cext];
   ok[name <> ": f* is a unit s,t-flow (B^T.f* == e_s - e_t)",
      Transpose[B] . fstar === cext];
   ok[name <> ": f* lies in the image of Pi (Pi.f* == f*)", Pi . fstar === fstar];
   (* competitor: route the whole unit along one simple path (longest found) *)
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
   {B, L, Lp}];

(* ---- run on three graphs ------------------------------------------------ *)
n = 6;
g = GraphData[{"Path", n}];        (* the user's starting point            *)
analyze[g, "Path P6", 1, 6];
ok["Path P6: series resistors, Reff(1,6) == 5",
   With[{dg = DirectedGraph[g, "Acyclic"]},
     With[{B = incidenceB[dg], cext = UnitVector[6, 1] - UnitVector[6, 6]},
       cext . PseudoInverse[Transpose[B] . B] . cext === 5]]];

analyze[CycleGraph[6], "Cycle C6", 1, 4];       (* Reff(edge) = 5/6; two routes *)
analyze[CompleteGraph[4], "Complete K4", 1, 2]; (* Reff(edge) = 1/2; tau = 16   *)

(* ---- 4.4: weighted graphs, on C6 with square weights -------------------- *)
Module[{dg, vl, el, n6, m6, B, w, W, Wh, L, Lp, PiW, es, et, cext, v, curr,
        Reff, fstar, energy, tau, pInT},
  dg = DirectedGraph[CycleGraph[6], "Acyclic"];
  vl = VertexList[dg]; el = EdgeList[dg]; n6 = Length[vl]; m6 = Length[el];
  Print["\n================  Weighted C6 (Sec 4.4)  ================"];
  B = incidenceB[dg];
  w = {1, 4, 9, 4, 1, 16};                (* perfect squares -> exact W^(1/2) *)
  W = DiagonalMatrix[w]; Wh = DiagonalMatrix[Sqrt[w]];
  Print["weights w = ", w, "  (resistances r_e = 1/w_e = ", 1/w, ")"];
  L = Transpose[B] . W . B;
  Lp = PseudoInverse[L];
  Print["L = B^T.W.B = ", MatrixForm[L]];

  es = UnitVector[n6, 1]; et = UnitVector[n6, 4]; cext = es - et;
  v = Lp . cext; curr = W . B . v;        (* Ohm with conductances: i = W.B.v *)
  ok["Weighted: Kirchhoff B^T.i == c_ext", Transpose[B] . curr === cext];

  Reff = Table[With[{i = e[[1]], j = e[[2]]},
       (UnitVector[n6, VertexIndex[dg, i]] - UnitVector[n6, VertexIndex[dg, j]])
       . Lp .
       (UnitVector[n6, VertexIndex[dg, i]] - UnitVector[n6, VertexIndex[dg, j]])],
     {e, el}];
  Print["Reff per edge = ", Reff];
  PiW = Wh . B . Lp . Transpose[B] . Wh;
  ok["Weighted: Pi_w = W^(1/2).B.L^+.B^T.W^(1/2) symmetric and Pi_w^2 == Pi_w",
     And[PiW === Transpose[PiW], PiW . PiW === PiW]];
  ok["Weighted: Pi_w(e,e) == w_e Reff(e)", Diagonal[PiW] === w Reff];
  ok["Weighted Foster: Sum_e w_e Reff(e) == n-1 == tr(Pi_w)",
     Total[w Reff] === n6 - 1 && Tr[PiW] === n6 - 1];

  tau = treeCount[L];                     (* weighted Matrix-Tree polynomial *)
  pInT = Table[With[{Wd = ReplacePart[W, {ei, ei} -> 0]},
       1 - treeCount[Transpose[B] . Wd . B]/tau], {ei, m6}];
  Print["tau_w(G) = ", tau, ";  P[e in T] = ", pInT];
  ok["Weighted Thm 4.5: P[e in T] == w_e Reff(e) == Pi_w(e,e)",
     pInT === w Reff];

  fstar = W . B . Lp . cext;              (* f* = W.B.L^+(e_s - e_t)          *)
  energy = Total[fstar^2/w];              (* E = Sum_e r_e f_e^2              *)
  Print["weighted unit 1,4-flow f* = ", fstar, ";  E(f*) = ", energy];
  ok["Weighted Prop 4.6: E(f*) == (e_s-e_t)^T L^+ (e_s-e_t)",
     energy === cext . Lp . cext];
  ok["Weighted: f* is a unit s,t-flow", Transpose[B] . fstar === cext];
  ];

Print["\n== summary: ", nPass, " PASS, ", nFail, " FAIL =="];
If[nFail > 0, Exit[1]];
