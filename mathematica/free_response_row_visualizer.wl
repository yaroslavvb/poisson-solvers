(* ::Package:: *)

(*
  Interactive row visualizer for

      X = PseudoInverse[Transpose[B]] = B . PseudoInverse[Transpose[B] . B].

  B has one oriented edge per row (+1 at its tail, -1 at its head), so
  X is edges x vertices.  Selecting row e displays

      response = X[[e]] = PseudoInverse[L] . B[[e]],

  the mean-zero vertex-potential response to one unit injected at the tail
  of edge e and removed at its head.  The optional arrows display the branch
  currents B.response driven by that unit dipole.

  Evaluate this file in a Mathematica notebook, or use

      Get["mathematica/free_response_row_visualizer.wl"]

  to return the default 3 x 3 interface.  Call
  FreeResponseRowVisualizer[n] directly to build another grid size.
*)

ClearAll[
  FreeResponseData, FreeResponseRowVisualizer, signedResponseColor,
  responseFlowArrow, exactLabel
];

FreeResponseData[n_Integer?Positive] /; n >= 2 := Module[
  {vertices, edges, b, laplacian, x},

  vertices = Tuples[Range[n], 2];
  edges = Select[
    Subsets[vertices, {2}],
    Total[Abs[#[[1]] - #[[2]]]] == 1 &
  ];
  b = Table[
    Which[v === edge[[1]], 1, v === edge[[2]], -1, True, 0],
    {edge, edges}, {v, vertices}
  ];
  laplacian = Transpose[b] . b;
  x = PseudoInverse[Transpose[b]];

  <|
    "GridSize" -> n,
    "Vertices" -> vertices,
    "Edges" -> edges,
    "B" -> b,
    "L" -> laplacian,
    "X" -> x,
    "IdentityChecks" -> <|
      "X == B.L+" -> (x === b . PseudoInverse[laplacian]),
      "B^T.X == centering projector" ->
        (Transpose[b] . x ===
          IdentityMatrix[Length[vertices]] -
            ConstantArray[1, {Length[vertices], Length[vertices]}]/
              Length[vertices]),
      "X.1 == 0" ->
        (x . ConstantArray[1, Length[vertices]] ===
          ConstantArray[0, Length[edges]])
    |>
  |>
];

signedResponseColor[value_, scale_] := Blend[
  {
    RGBColor[0.18, 0.43, 0.72],
    RGBColor[0.97, 0.97, 0.94],
    RGBColor[0.84, 0.25, 0.20]
  },
  Clip[(N[value]/scale + 1)/2, {0, 1}]
];

exactLabel[value_] := TraditionalForm[value];

responseFlowArrow[edge_, value_, maxFlow_, nodeRadius_] := Module[
  {orientedEdge, direction, start, finish, midpoint, normal, relative,
   color, width},

  orientedEdge = If[value >= 0, edge, Reverse[edge]];
  direction = N[orientedEdge[[2]] - orientedEdge[[1]]];
  start = N[orientedEdge[[1]]] + nodeRadius direction;
  finish = N[orientedEdge[[2]]] - nodeRadius direction;
  midpoint = Mean[N[orientedEdge]];
  normal = {-direction[[2]], direction[[1]]};
  relative = If[maxFlow == 0, 0., N[Abs[value]/maxFlow]];
  color = Blend[
    {RGBColor[0.95, 0.70, 0.24], RGBColor[0.75, 0.16, 0.10]},
    relative
  ];
  width = 0.007 + 0.021 relative;

  {
    color,
    Thickness[width],
    Arrowheads[0.032],
    Arrow[{start, finish}],
    Text[
      Framed[
        Style[exactLabel[Abs[value]], 10, Bold, Black],
        Background -> White,
        FrameStyle -> None,
        FrameMargins -> 1
      ],
      midpoint + 0.13 normal
    ]
  }
];

FreeResponseRowVisualizer[n_Integer : 3] /; n >= 2 := Module[
  {data, vertices, edges, b, x, m, vertexCount, rowLabels, nodeRadius,
   plotRange, matrixHeader},

  data = FreeResponseData[n];
  vertices = data["Vertices"];
  edges = data["Edges"];
  b = data["B"];
  x = data["X"];
  m = Length[edges];
  vertexCount = Length[vertices];
  nodeRadius = 0.13;
  plotRange = {{0.45, n + 0.55}, {0.45, n + 0.55}};
  rowLabels = Table[
    k -> Row[{"row ", k, ":  ", edges[[k, 1]], "  ",
       Style["\[Rule]", 14], "  ", edges[[k, 2]]}],
    {k, m}
  ];
  matrixHeader = Prepend[Style[#, 9, Bold] & /@ vertices, "edge / vertex"];

  DynamicModule[{selectedRow = 1, showCurrents = True, showMatrix = False},
    Column[
      {
        Style["Rows of the free-response matrix  X = pinv(B\[Transpose])", 18,
          Bold],
        Style[
          Row[{
            "Select an oriented edge. Its row is the mean-zero vertex ",
            "response to +1 at the tail and -1 at the head."
          }],
          11, GrayLevel[0.30]
        ],
        Row[
          {
            Style["Selected row:  ", 11, Bold],
            PopupMenu[Dynamic[selectedRow], rowLabels, ImageSize -> 245],
            Spacer[22],
            Checkbox[Dynamic[showCurrents]],
            Style[" show induced branch currents", 11]
          }
        ],

        Dynamic[
          Module[
            {edge, tail, head, response, flow, responseScale, maxFlow,
             baseEdges, clickTargets, selectedEdge, arrows, nodes, graphic,
             sourceVector, conservation, energy},

            edge = edges[[selectedRow]];
            {tail, head} = edge;
            response = x[[selectedRow]];
            flow = b . response;
            responseScale = Max[Abs[N[response]]];
            If[responseScale == 0, responseScale = 1.];
            maxFlow = Max[Abs[flow]];
            sourceVector = b[[selectedRow]];
            conservation = Transpose[b] . flow;
            energy = response . sourceVector;

            baseEdges = {
              GrayLevel[0.80], AbsoluteThickness[2], Line /@ edges
            };
            clickTargets = MapIndexed[
              Function[{anEdge, position},
                With[{k = First[position]},
                  EventHandler[
                    {
                      Directive[GrayLevel[0.2], Opacity[0.001],
                        AbsoluteThickness[14]],
                      Line[anEdge]
                    },
                    {"MouseClicked" :> (selectedRow = k)},
                    PassEventsDown -> True
                  ]
                ]
              ],
              edges
            ];
            selectedEdge = {
              Directive[RGBColor[0.22, 0.22, 0.22], Opacity[0.75],
                AbsoluteThickness[7]],
              Line[edge]
            };
            arrows = If[
              TrueQ[showCurrents],
              MapThread[
                responseFlowArrow[#1, #2, maxFlow, nodeRadius] &,
                {edges, flow}
              ],
              {}
            ];
            nodes = MapIndexed[
              Function[{vertex, position},
                With[
                  {i = First[position], value = response[[First[position]]],
                   role = Which[
                     vertex === tail, "+1 source",
                     vertex === head, "-1 sink",
                     True, ""
                   ]},
                  {
                    If[
                      role === "",
                      Nothing,
                      {
                        FaceForm[None],
                        EdgeForm[Directive[
                          If[vertex === tail,
                            RGBColor[0.72, 0.12, 0.08],
                            RGBColor[0.10, 0.30, 0.65]
                          ],
                          AbsoluteThickness[4]
                        ]],
                        Disk[N[vertex], nodeRadius + 0.045]
                      }
                    ],
                    EdgeForm[Directive[GrayLevel[0.15], AbsoluteThickness[1.4]]],
                    FaceForm[signedResponseColor[value, responseScale]],
                    Tooltip[
                      Disk[N[vertex], nodeRadius],
                      Column[{
                        Row[{"vertex: ", vertex}],
                        Row[{"X[[", selectedRow, ", ", i, "]] = ",
                          exactLabel[value]}],
                        If[role === "", Nothing, Row[{"input: ", role}]]
                      }]
                    ],
                    Text[
                      Style[
                        Which[
                          vertex === tail, "+1",
                          vertex === head, "-1",
                          True, ToString[vertex]
                        ],
                        11, Bold,
                        If[Abs[N[value]] > 0.55 responseScale, White, Black]
                      ],
                      N[vertex]
                    ],
                    Text[
                      Framed[
                        Style[exactLabel[value], 9, Black],
                        Background -> White,
                        FrameStyle -> None,
                        FrameMargins -> 0
                      ],
                      N[vertex] + {0, -0.22}
                    ]
                  }
                ]
              ],
              vertices
            ];

            graphic = Graphics[
              {baseEdges, selectedEdge, arrows, clickTargets, nodes},
              PlotRange -> plotRange,
              PlotRangePadding -> Scaled[0.03],
              AspectRatio -> 1,
              ImageSize -> 620,
              Background -> White
            ];

            Column[
              {
                Style[
                  Row[{
                    "row ", selectedRow, " selects  ", tail,
                    "  (+1)  \[Rule]  ", head, "  (-1)"
                  }],
                  13, Bold
                ],
                graphic,
                Grid[
                  {
                    {
                      Style["selected row  X[[e]]", 10, Bold],
                      Row[exactLabel /@ response, "   "]
                    },
                    {
                      Style["mean / gauge", 10, Bold],
                      exactLabel[Mean[response]]
                    },
                    {
                      Style["Kirchhoff check", 10, Bold],
                      Row[{
                        "B\[Transpose](B response) = b_e:  ",
                        conservation === sourceVector
                      }]
                    },
                    {
                      Style["energy / voltage drop", 10, Bold],
                      Row[{
                        exactLabel[flow . flow], " = ", exactLabel[energy],
                        " = response(tail) - response(head)"
                      }]
                    }
                  },
                  Alignment -> Left,
                  Dividers -> {False, {False, True, False, False, False}},
                  Spacings -> {1.2, 0.65}
                ],
                Style[
                  "Click any grid edge to select its row. Vertex color and labels " <>
                    "show the row of X; arrows show the current response B.X[[e]].",
                  10, GrayLevel[0.35]
                ],
                Row[{
                  Style["blue", 10, Bold, RGBColor[0.18, 0.43, 0.72]],
                  Style[" = negative response     ", 10, GrayLevel[0.35]],
                  Style["white", 10, Bold, GrayLevel[0.45]],
                  Style[" = zero     ", 10, GrayLevel[0.35]],
                  Style["red", 10, Bold, RGBColor[0.84, 0.25, 0.20]],
                  Style[" = positive response; colored rings mark the input endpoints.",
                    10, GrayLevel[0.35]]
                }]
              },
              Alignment -> Center,
              Spacings -> 0.65
            ]
          ],
          TrackedSymbols :> {selectedRow, showCurrents}
        ],

        OpenerView[
          {
            Style["Show the complete matrix X (selected row highlighted)", 11,
              Bold],
            Dynamic[
              Grid[
                Prepend[
                  MapIndexed[
                    Function[{matrixRow, position},
                      With[{k = First[position]},
                        Prepend[
                          Map[
                            If[k === selectedRow,
                              Item[exactLabel[#], Background -> RGBColor[1., 0.94, 0.72]],
                              exactLabel[#]
                            ] &,
                            matrixRow
                          ],
                          Item[
                            Row[{k, ": ", edges[[k, 1]], "\[Rule]", edges[[k, 2]]}],
                            Background -> If[k === selectedRow,
                              RGBColor[1., 0.88, 0.55], GrayLevel[0.94]]
                          ]
                        ]
                      ]
                    ],
                    x
                  ],
                  matrixHeader
                ],
                Frame -> All,
                Alignment -> Center,
                ItemSize -> All,
                BaseStyle -> {FontSize -> 9}
              ],
              TrackedSymbols :> {selectedRow}
            ]
          },
          Dynamic[showMatrix]
        ],
        Style[
          Row[{
            "Dimensions: ", m, " edges \[Times] ", vertexCount,
            " vertices.  Global checks: ", data["IdentityChecks"]
          }],
          9, GrayLevel[0.42]
        ]
      },
      Alignment -> Center,
      Spacings -> 0.8
    ]
  ]
];

If[$FrontEnd === Null,
  Print[
    "Loaded FreeResponseRowVisualizer. Open this file in Mathematica and " <>
    "evaluate FreeResponseRowVisualizer[3] for the interactive interface."
  ],
  FreeResponseRowVisualizer[3]
]
