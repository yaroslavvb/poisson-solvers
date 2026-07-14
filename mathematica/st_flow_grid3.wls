#!/usr/bin/env wolframscript
(* ::Package:: *)

(*
  st_flow_grid3.wls

  Visualize the unit electrical s-t flow on a 3 x 3 grid, following
  Vishnoi (2013), Section 4.3.  Every edge has unit resistance.

  Vishnoi's incidence convention is used: B[[e, v]] is +1 at the tail and
  -1 at the head of oriented edge e.  Consequently

      L = Transpose[B].B,
      v = PseudoInverse[L].(e_s - e_t),
      fStar = B.v = B.PseudoInverse[L].(e_s - e_t).

  The arbitrary additive constant in v is fixed by the Moore-Penrose
  pseudoinverse (Mean[v] == 0).  Arrow direction shows the sign of fStar;
  arrow width and edge labels show its magnitude.

  Run from the repository root:
    wolframscript -file mathematica/st_flow_grid3.wls

  Output:
    figures/mma_st_flow_grid3.png
*)

ClearAll[incidenceMatrix, electricalGridFlow, flowArrow, voltageColor];

incidenceMatrix[vertices_List, edges_List] :=
  Table[
    Which[v === e[[1]], 1, v === e[[2]], -1, True, 0],
    {e, edges}, {v, vertices}
  ];

voltageColor[x_, {xmin_, xmax_}] :=
  Blend[
    {RGBColor[0.20, 0.43, 0.72], RGBColor[0.96, 0.96, 0.94],
     RGBColor[0.86, 0.29, 0.24]},
    Rescale[N[x], {N[xmin], N[xmax]}]
  ];

flowArrow[edge_, value_, maxFlow_, nodeRadius_] := Module[
  {tail, head, direction, start, finish, midpoint, normal, labelPosition,
   relativeMagnitude, color, width},
  {tail, head} = If[value >= 0, List @@ edge, Reverse[List @@ edge]];
  direction = Normalize[N[head - tail]];
  start = N[tail] + nodeRadius direction;
  finish = N[head] - nodeRadius direction;
  midpoint = (N[tail] + N[head])/2;
  normal = {-direction[[2]], direction[[1]]};
  labelPosition = midpoint + 0.14 normal;
  relativeMagnitude = N[Abs[value]/maxFlow];
  color = Blend[
    {RGBColor[0.95, 0.70, 0.24], RGBColor[0.78, 0.20, 0.12]},
    relativeMagnitude
  ];
  width = 0.008 + 0.026 relativeMagnitude;
  {
    color, Thickness[width], Arrowheads[0.035], Arrow[{start, finish}],
    Text[
      Framed[
        Style[TraditionalForm[Abs[value]], 12, Bold, Black],
        Background -> White, FrameStyle -> None, FrameMargins -> 1
      ],
      labelPosition
    ]
  }
];

electricalGridFlow[n_Integer, s : {_, _}, t : {_, _}] := Module[
  {vertices, horizontalEdges, verticalEdges, edges, b, laplacian, source,
   voltage, fStar, energy, effectiveResistance, maxFlow, nodeRadius = 0.13,
   vMinMax, plotRange, baseGrid, arrows, nodes, graphic, figure},

  vertices = Flatten[Table[{x, y}, {y, 1, n}, {x, 1, n}], 1];
  If[! MemberQ[vertices, s] || ! MemberQ[vertices, t] || s === t,
    Print["s and t must be distinct vertices of the grid."];
    Exit[1]
  ];

  (* Fix an orientation only to define B: east on horizontal edges and north
     on vertical edges.  Negative computed flows are drawn in reverse. *)
  horizontalEdges = Flatten[
    Table[DirectedEdge[{x, y}, {x + 1, y}], {y, 1, n}, {x, 1, n - 1}]
  ];
  verticalEdges = Flatten[
    Table[DirectedEdge[{x, y}, {x, y + 1}], {x, 1, n}, {y, 1, n - 1}]
  ];
  edges = Join[horizontalEdges, verticalEdges];

  b = incidenceMatrix[vertices, edges];
  laplacian = Transpose[b] . b;
  source = UnitVector[Length[vertices], First@FirstPosition[vertices, s]] -
    UnitVector[Length[vertices], First@FirstPosition[vertices, t]];
  voltage = PseudoInverse[laplacian] . source;
  fStar = b . voltage;
  energy = fStar . fStar;
  effectiveResistance = source . PseudoInverse[laplacian] . source;

  (* Exact checks of Kirchhoff's law and Proposition 4.6. *)
  If[Transpose[b] . fStar =!= source,
    Print["FAIL: B^T.fStar != e_s - e_t"];
    Exit[1]
  ];
  If[energy =!= effectiveResistance,
    Print["FAIL: E(fStar) != (e_s-e_t)^T.L^+.(e_s-e_t)"];
    Exit[1]
  ];

  maxFlow = Max[Abs[fStar]];
  vMinMax = MinMax[voltage];
  plotRange = {{0.45, n + 0.55}, {0.45, n + 0.55}};

  baseGrid = {
    GrayLevel[0.82], AbsoluteThickness[2],
    Line /@ (List @@@ edges)
  };
  arrows = MapThread[
    flowArrow[#1, #2, maxFlow, nodeRadius] &,
    {edges, fStar}
  ];
  nodes = MapIndexed[
    Function[{vertex, position},
      With[
        {i = First[position], role = Which[vertex === s, "s", vertex === t, "t", True, ""]},
        {
          EdgeForm[Directive[GrayLevel[0.18], AbsoluteThickness[1.5]]],
          voltageColor[voltage[[i]], vMinMax], Disk[N[vertex], nodeRadius],
          Text[
            Style[If[role === "", ToString[vertex], role], 12, Bold,
              If[Abs[N[voltage[[i]]]] > 0.35 Max[Abs[N[voltage]]], White, Black]],
            N[vertex]
          ],
          Text[
            Style[Row[{"v = ", TraditionalForm[voltage[[i]]]}], 11, Black],
            N[vertex] + {0, -0.23}
          ]
        }
      ]
    ],
    vertices
  ];

  graphic = Graphics[
    {baseGrid, arrows, nodes},
    PlotRange -> plotRange, PlotRangePadding -> Scaled[0.03],
    AspectRatio -> 1, ImageSize -> 620, Background -> White
  ];

  figure = Column[
    {
      Style["Unit electrical flow on a 3 x 3 grid", 19, Bold],
      Style[
        Row[{"one ampere injected at s = ", s,
             "; removed at t = ", t, "  (all resistances = 1)"}],
        12, GrayLevel[0.25]
      ],
      graphic,
      Grid[
        {{
          Style[Row[{"Energy  E(f*) = ", TraditionalForm[energy]}], 12, Bold],
          Spacer[28],
          Style[Row[{"Effective resistance  R_eff(s,t) = ",
                     TraditionalForm[effectiveResistance]}], 12, Bold]
        }},
        Alignment -> Center
      ],
      Style[
        "Arrow direction is the current direction; width and label encode |f*_e|.",
        11, GrayLevel[0.35]
      ]
    },
    Alignment -> Center, Spacings -> 0.7
  ];

  <|
    "Vertices" -> vertices, "Edges" -> edges, "B" -> b,
    "L" -> laplacian, "Voltage" -> voltage, "Flow" -> fStar,
    "Energy" -> energy, "EffectiveResistance" -> effectiveResistance,
    "Figure" -> figure
  |>
];

(* Opposite corners: top-left source and bottom-right sink. *)
result = electricalGridFlow[3, {1, 3}, {3, 1}];

Print["voltages (vertex -> v): ",
  Thread[result["Vertices"] -> result["Voltage"]]];
Print["oriented edge flows (edge -> f*): ",
  Thread[result["Edges"] -> result["Flow"]]];
Print["PASS: Kirchhoff B^T.f* = e_s - e_t"];
Print["PASS: E(f*) = R_eff(s,t) = ", result["Energy"]];

If[! DirectoryQ["figures"],
  CreateDirectory["figures", CreateIntermediateDirectories -> True]
];
Export[
  "figures/mma_st_flow_grid3.png",
  result["Figure"],
  ImageResolution -> 180
];
Print["Exported figures/mma_st_flow_grid3.png"];
