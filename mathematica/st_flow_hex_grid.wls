#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Unit electrical s-t flow on a finite honeycomb grid.

   The graph consists of the vertices and edges of an n x n patch of regular
   hexagons.  Interior graph vertices therefore have degree 3, and the faces
   of the graph are hexagons (rather than triangles between hex-cell centers).

   Run from the repository root:
     wolframscript -file mathematica/st_flow_hex_grid.wls

   Output:
     figures/mma_st_flow_hex_grid.png
*)

n = 3;
nodeRadius = 0.15;
pinv = PseudoInverse;

(* Flat-top unit hexagons.  The center formula is the standard axial-coordinate
   embedding; neighboring cells share one complete edge. *)
hexOffsets = {
  {1, 0}, {1/2, Sqrt[3]/2}, {-1/2, Sqrt[3]/2},
  {-1, 0}, {-1/2, -Sqrt[3]/2}, {1/2, -Sqrt[3]/2}
};
centers = Flatten[
  Table[{3 q/2, Sqrt[3] (u + q/2)}, {q, 0, n - 1}, {u, 0, n - 1}],
  1
];
cells = Function[center, (center + #) & /@ hexOffsets] /@ centers;
cellEdges = Flatten[
  (Partition[Append[#, First[#]], 2, 1] &) /@ cells,
  1
];

(* Canonical endpoint order removes the edges shared by adjacent hexagons. *)
edges = DeleteDuplicates[Sort /@ cellEdges];
pts = DeleteDuplicates[Flatten[edges, 1]];
drawPts = N[pts];
drawEdges = N[edges];

(* Opposite boundary vertices; replace these two values to move the terminals. *)
orderedPts = SortBy[pts, {First, Last}];
s = First[orderedPts];
t = Last[orderedPts];

(* B = -Transpose[IncidenceMatrix[g]], in Vishnoi's convention: +1 at
   the arbitrarily chosen tail and -1 at the head. *)
B = Table[
   Which[v === e[[1]], 1, v === e[[2]], -1, True, 0],
   {e, edges}, {v, pts}
];

source = Boole[# === s] - Boole[# === t] & /@ pts;
X = pinv[Transpose[B]];          (* minimum-norm free response matrix *)
f = X . source;                 (* electrical unit s-t flow             *)

If[Transpose[B] . f =!= source,
  Print["FAIL: Kirchhoff law Transpose[B].f == e_s-e_t did not hold"];
  Exit[1]
];

energy = f . f;
L = Transpose[B] . B;
effectiveResistance = source . pinv[L] . source;
If[energy =!= effectiveResistance,
  Print["FAIL: E(f) != R_eff(s,t)"];
  Exit[1]
];

maxF = Max[Abs[f]];

(* Decimal edge labels are easier to read on this dense graph.  Replace
   flowLabel below by TraditionalForm[value] to display exact fractions. *)
flowLabel[value_] := NumberForm[N[value], {3, 2}];

flowGraphic = Graphics[
  {
    (* Base grid. *)
    GrayLevel[0.82], AbsoluteThickness[2], Line /@ drawEdges,

    (* Directed flow arrows and magnitude labels. *)
    MapThread[
      Function[{edge, value},
        With[
          {magnitude = Abs[value],
           p = If[value >= 0, edge, Reverse[edge]]},
          With[
            {d = Normalize[p[[2]] - p[[1]]]},
            {
              Blend[
                {RGBColor[0.95, 0.70, 0.24], RGBColor[0.78, 0.20, 0.12]},
                N[magnitude/maxF]
              ],
              Thickness[0.005 + 0.018 N[magnitude/maxF]],
              Arrowheads[0.026],
              Arrow[{
                p[[1]] + nodeRadius d,
                p[[2]] - nodeRadius d
              }],
              Text[
                Framed[
                  Style[flowLabel[magnitude], 9, Bold, Black],
                  Background -> White, FrameStyle -> None, FrameMargins -> 0
                ],
                Mean[p] + 0.105 {-d[[2]], d[[1]]}
              ]
            }
          ]
        ]
      ],
      {drawEdges, f}
    ],

    (* Nodes: label only the injection and extraction sites to avoid covering
       the flow values on the denser six-neighbor grid. *)
    EdgeForm[Directive[GrayLevel[0.18], AbsoluteThickness[1.5]]],
    MapThread[
      {
        Which[
          #1 === s, RGBColor[0.86, 0.29, 0.24],
          #1 === t, RGBColor[0.20, 0.43, 0.72],
          True, White
        ],
        Disk[#2, nodeRadius]
      } &,
      {pts, drawPts}
    ],
    MapThread[
      If[#1 === s || #1 === t,
        Text[Style[If[#1 === s, "s", "t"], 12, Bold, White], #2],
        Nothing
      ] &,
      {pts, drawPts}
    ]
  },
  PlotRangePadding -> Scaled[0.05],
  AspectRatio -> Automatic,
  ImageSize -> 820,
  Background -> White
];

figure = Column[
  {
    Style["Unit electrical flow on a honeycomb grid", 19, Bold],
    Style[
      Row[{"one ampere:  s = ", s, "  to  t = ", t,
           "    |V| = ", Length[pts], ", |E| = ", Length[edges]}],
      12, GrayLevel[0.25]
    ],
    flowGraphic,
    Style[
      Row[{"E(f*) = R_eff(s,t) = ", TraditionalForm[energy]}],
      12, Bold
    ],
    Spacer[18]
  },
  Alignment -> Center,
  Spacings -> 0.6
];

Print["vertices = ", Length[pts], "; edges = ", Length[edges]];
Print["PASS: Kirchhoff Transpose[B].f = e_s - e_t"];
Print["PASS: E(f*) = R_eff(s,t) = ", energy];

If[! DirectoryQ["figures"],
  CreateDirectory["figures", CreateIntermediateDirectories -> True]
];
Export[
  "figures/mma_st_flow_hex_grid.png",
  figure,
  ImageResolution -> 180
];
Print["Exported figures/mma_st_flow_hex_grid.png"];
