(* ::Package:: *)

(* Interactive CG dashboard for a 1D Poisson (electrostatics) problem.
   Reference source for the JavaScript port in cg-explorer/ .
   NOTE: the Manipulate at the end requires the notebook front end;
   run this file inside Mathematica, not via wolframscript. *)

(* ==== 1. Setup the 1D Poisson Problem ==== *)
n = 256;
h = 1.0/(n + 1);

(* 1D Discrete Laplacian (-1,2,-1 stencil) *)
A = SparseArray[{Band[{1, 1}] -> 2.0, Band[{2, 1}] -> -1.0,
     Band[{1, 2}] -> -1.0}, {n, n}]/h^2;

(* ==== 2. Source Term (Point Sources) ==== *)
b = ConstantArray[0.0, n];
b[[1]] = 1.0;
b[[128]] = 0.0;
b[[256]] = -1.0;

(* ==== 3. Standard Conjugate Gradient Solver (No Preconditioner) ==== *)
CGSolve[matA_, vecB_, maxIter_ : 1000, tol_ : 10^-2] :=
  Module[{bNorm = Norm[vecB], cgStep, result},
   (* Define a single CG mathematical state transition *)
   cgStep[{x_, r_, p_, rDotr_, _}] :=
    Module[{Ap = matA . p, \[Alpha]Step, stepTaken, xNew, rNew,
      rDotrNew, pNew, relRes},
     \[Alpha]Step = rDotr/(p . Ap);
     stepTaken = \[Alpha]Step*p; (* Exact step (\[CapitalDelta]u) taken this iteration *)
     xNew = x + stepTaken;
     rNew = r - \[Alpha]Step*Ap;
     relRes = Norm[rNew]/bNorm;
     (* Save: {Solution, Residual, Step Taken, Relative Error} *)
     Sow[{xNew, rNew, stepTaken, relRes}];
     rDotrNew = rNew . rNew;
     pNew = rNew + (rDotrNew/rDotr)*p;
     {xNew, rNew, pNew, rDotrNew, relRes}];
   result =
    Reap[(* Sow the Initial State (Iteration 0) with a 0-vector for the step *)
     Sow[{0.*vecB, vecB, 0.*vecB, 1.0}];
     NestWhile[cgStep, {0.*vecB, vecB, vecB, vecB . vecB, 1.0},
      (#[[5]] > tol &), 1, maxIter]];
   (* Returns {Final Solution, Full Iteration History} *)
   {result[[1, 1]], result[[2, 1]]}];

(* ==== 4. Run the Solver ==== *)
Print["Solving Standard CG..."];
{xFinal, history} = CGSolve[A, b];
resHistory = history[[All, 4]];
Print["Total Iterations: ", Length[resHistory] - 1];

(* ==== 5. Interactive 2x2 Visualization Dashboard ==== *)
plotOpts = Sequence[ImageSize -> 330, Frame -> True, GridLines -> Automatic];

(* Helper functions to precompute fixed PlotRanges *)
padRange[limits_] :=
  limits + {-0.05, 0.05}*Max[10^-6, limits[[2]] - limits[[1]]];
symRange[limits_] :=
  Module[{m = Max[Abs[limits], 10^-6]}, {-m, m}*1.05]; (* Keeps 0 centered *)

(* Compute global limits to freeze axes and prevent auto-scaling jumps *)
xPlotRange = padRange[MinMax[Flatten[history[[All, 1]]]]];
rPlotRange = symRange[MinMax[Flatten[history[[All, 2]]]]];
stepPlotRange = symRange[MinMax[Flatten[history[[All, 3]]]]];

Manipulate[
 Module[{currX, currR, currStepSize, currRes, plot1, plot2, plot3, plot4},
  (* Retrieve state from the history matrix *)
  currX = history[[step, 1]];
  currR = history[[step, 2]];
  currStepSize = history[[step, 3]];
  currRes = resHistory[[step]];
  plot1 = ListLinePlot[currX, DataRange -> {0, 1}, PlotRange -> xPlotRange,
    PlotStyle -> Directive[AbsoluteThickness[2], Darker[Blue]],
    PlotLabel -> "Solved Field (u) | Step " <> ToString[step - 1], Evaluate@plotOpts];
  plot2 = ListLinePlot[currR, DataRange -> {0, 1}, PlotRange -> rPlotRange,
    PlotStyle -> Directive[AbsoluteThickness[1.5], Darker[Red]],
    PlotLabel -> "Residual (r) | Step " <> ToString[step - 1], Evaluate@plotOpts];
  plot3 = ListLinePlot[currStepSize, DataRange -> {0, 1}, PlotRange -> stepPlotRange,
    PlotStyle -> Directive[AbsoluteThickness[1.5], Darker[Green]],
    PlotLabel -> "Step Taken (\[CapitalDelta]u) | Step " <> ToString[step - 1], Evaluate@plotOpts];
  plot4 = ListLogPlot[resHistory, Joined -> True, PlotRange -> All,
    PlotStyle -> Directive[AbsoluteThickness[2], Gray],
    FrameLabel -> {"Iteration Step", "Relative Residual (Log)"},
    PlotLabel -> "Convergence Tracker", Evaluate@plotOpts,
    Epilog -> {Red, PointSize[0.04], Point[{step, currRes}]}];
  Grid[{{plot1, plot2}, {plot3, plot4}}, Spacings -> {1, 1.5}]],
 {{step, 1, "Select Iteration Step:"}, 1, Length[history], 1, Appearance -> "Labeled"}]
