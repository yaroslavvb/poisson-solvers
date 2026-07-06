# poisson-solvers

Implementation and study of **preconditioned conjugate gradients for the 2-D Poisson equation**: a Mathematica reference program ported line-by-line to Python, then used as a fully-instrumented testbed for classical (Jacobi, ILU), randomized ([Nyström, arXiv:2110.02820](https://arxiv.org/abs/2110.02820)), and neural ([NPO, arXiv:2502.01337](https://arxiv.org/abs/2502.01337)) preconditioners.

**Problem**: $A = (d_1 \otimes I + I \otimes d_1)/h^2$ — the 5-point Dirichlet Laplacian on a $32\times32$ interior grid ($h = 1/(n+1)$, $N = n^2 = 1024$, $\kappa(A) = 440.69$ exactly) — solved against a standardized Gaussian-random-field right-hand side to $\|r\|/\|b\| \le 10^{-10}$. Plus a variable-coefficient variant (100:1 coefficient jump) that makes Jacobi nontrivial.

**Why**: at this scale every quantity is exactly computable (analytic spectrum, dense eigensolves, direct-solve ground truth), so each preconditioner's behavior can be *explained*, not just measured — including two instructive failures: Nyström losing to plain CG on a flat-topped spectrum, and plain PCG stalling on a nonlinear neural preconditioner.

## Interactive demo & rendered reports

- **[Iterative solvers explorer](https://yaroslavvb.github.io/poisson-solvers/cg-explorer/)** — interactive dashboard for a 1D heat-conduction Poisson problem (heater at one end, chiller at the other): scrub through the iteration history of CG, SOR, gradient descent, and CG with a toy in-browser neural preconditioner ([arXiv:2502.01337](https://arxiv.org/abs/2502.01337), trained by [python/neural/train_npo_1d.py](python/neural/train_npo_1d.py)).
- **[Hierarchical solver race](https://yaroslavvb.github.io/poisson-solvers/interactive/hierarchical-solvers.html)** — five solvers (GD, CG, PCG with the *actual* HODLR rank-2/rank-8 compressed inverses, damped Richardson) racing live in the browser on the hot/cold-rod problem of [report 14](reports/14-hierarchical-inverse.md) §5; runs locally too ([interactive/hierarchical-solvers.html](interactive/hierarchical-solvers.html), serve the repo root over HTTP).
- **[Rendered report suite](https://yaroslavvb.github.io/poisson-solvers/)** — the reports below as web pages (GitHub Pages).

## Reading offline

The whole suite (reports, figures, both interactive pages, vendored MathJax) is mirrored as a self-contained static site in `local-site/` — open `local-site/index.html` directly in a browser (`file://`, no server, no network needed). Regenerate it with `uv run python tools/build_local_site.py`.

## Quickstart

Python env is managed by `uv` (Python 3.12; numpy, scipy, matplotlib, torch). Run from the repo root:

```bash
uv run python python/experiments/run_all.py        # full benchmark -> results/results.json + figures
uv run python python/experiments/spectra.py        # eigenvalue verification + kappa(n) scaling
uv run python python/neural/train_npo.py           # train the neural preconditioner (~216 s CPU)
uv run python python/neural/eval_npo.py            # evaluate it (FCG vs plain PCG vs CG)

wolframscript -file mathematica/poisson_pcg.wls    # Mathematica reference run
wolframscript -file mathematica/eigen_check.wls    # analytic-spectrum cross-check
wolframscript -file mathematica/nystrom_pcg.wls    # Wolfram Nystrom implementation
```

All Python runs are bit-deterministic (fixed seeds) and reproduce the committed `results/*.json` exactly.

## Results at a glance

Canonical problem, from [results/results.json](results/results.json):

| method | iterations | wall (setup + solve) [s] | note |
|---|---:|---:|---|
| CG (none) | 116 | 0.0012 | baseline |
| CG (Jacobi) | 116 | 0.0012 | provably identical to plain CG (constant diagonal) |
| CG (ILU) | **5** | 0.0012 | near-direct factorization at this scale |
| CG (Nyström, ranks 16–256) | 123–119 | 0.0026–0.026 | *worse* than plain CG — flat-top spectrum is its adversarial case |
| **FCG (NPO, Notay)** | **30** | 0.030 | 3.87× fewer iterations via spectral clustering (12.56 spread vs κ = 440.69) |
| CG (NPO, plain PCG) | 2000 | 1.68 | **did not converge** — negative control: FR-β breaks on a nonlinear M |

Variable-coefficient problem (contrast 100): CG 771 iterations → Jacobi-CG **137** (5.6×).

## Reports

Start with [reports/00-overview.md](reports/00-overview.md) (full repo map, run instructions, annotated reading order), then:

1. [01 — Code Walkthrough](reports/01-code-walkthrough.md) — Mathematica reference + Python port, divergence ledger
2. [02 — The Eigenvalue Story](reports/02-eigenvalues.md) — closed-form spectra, DST-I, exact κ, verified to float64
3. [03 — The GRF Right-Hand Side](reports/03-gaussian-random-fields.md) — spectral sampler, Matérn interpretation, Re(IFFT) proof
4. [04 — Krylov and PCG](reports/04-krylov-and-pcg.md) — CG theory, √κ bound vs reality, flexible CG (Notay)
5. [05 — Classical Preconditioners](reports/05-classical-preconditioners.md) — Jacobi (theorem + 5.6× contrast case), ILU, road to multigrid
6. [06 — Neural Preconditioning Operator](reports/06-neural-preconditioner.md) — NPO paper digest, toy NAMG, 116 → 30, why FCG is required
7. [07 — Randomized Nyström Preconditioning](reports/07-nystrom-preconditioning.md) — exact implementation, the instructive negative result
8. [08 — Consolidated Results](reports/08-results.md) — full matrix, timings, sanity checks, limitations, next steps
9. [09 — The Statistical Dictionary](reports/09-stiffness-as-precision.md) — stiffness matrix = precision matrix: Green's function as Brownian-bridge covariance, solvers as Gaussian inference, preconditioners as surrogate models (incomplete Cholesky = Vecchia, Nyström = factor analysis)
10. [10 — Kick It, or Watch It Jitter](reports/10-fluctuation-dissipation.md) — fluctuation–dissipation for the discrete Laplacian: kick response = jitter covariance, the Cholesky factors read out of thermal snapshots, and a Vecchia preconditioner learned from noise alone that cuts the canonical solve 116 → 30 (κ 440.69 → 13.7)
11. [11 — Predict Thy Neighbor, Subtract the Average](reports/11-regressions-and-multiscale.md) — the 09/10 dictionary worked on the 8×8 grid: Cholesky fill as wavefront regressions, the IC(0)-vs-Vecchia gap measured (~22%), block averages as coarse regressors, and an additive two-level IC(0)+coarse preconditioner that takes the hot/cold-rod solve 76 → 32 (κ 440.69 → 11.05)
12. [12 — The Preconditioner Is an Autoregressive Predictor](reports/12-autoregressive-preconditioning.md) — the synthesis: CG removed, every preconditioner run as the same stationary predict-and-correct Richardson iteration, quality read as ρ(I − CA) — perfect two-sided regressions scheduled synchronously = Jacobi (ρ = cos πh), sequentially = Gauss–Seidel (rate exponent exactly doubles), the perfect causal predictor solving in one step, and the truncation ladder down to a mesh-independent two-grid ρ = 0.357 (17 sweeps vs Jacobi's 4777)
13. [13 — Preconditioning Is Decoupling](reports/13-preconditioning-as-decoupling.md) — the capstone: coupling = cross-partials = off-diagonal precision = conditional dependence, and every preconditioner is a scheme for splitting one entangled minimization into (nearly) independent subproblems — five decoupling axes (coordinates, frequency, direction, space, scale) raced on one ladder (ADI cuts κ 440.69 → 10.52 = 0.50√κ; block-Jacobi(2) leaves 960 of 1024 eigenvalues at exactly 1 and CG needs just 12 iterations), plus the measured GD-vs-CG verdict — two distinct eigenvalues mean CG finishes in exactly 2 steps even at κ = 10⁶ (GD: 11.5 million): clusters, not range, are CG's currency
14. [14 — The Hierarchical Structure of the Inverse](reports/14-hierarchical-inverse.md) — the structural sequel: conditional independence across a separator is a *rank bound* on covariance blocks ($\Sigma_{LR} = \Sigma_{LI}\Sigma_{II}^{-1}\Sigma_{IR}$, rank ≤ |I|), measured as a machine-precision cliff at exactly the separator width 32 at every level of a HODLR partition of $A^{-1}$ — so the dense inverse compresses to $O(Nr\log N)$ (8× at rank 8) and runs as an apply-ready preconditioner (κ 440.69 → 3.85 at rank 8; rank 16 takes the iteration lead, 11 vs block-Jacobi's 12) — plus an **interactive in-browser solver race** ([interactive/hierarchical-solvers.html](interactive/hierarchical-solvers.html)) driving the actual exported rank-2/rank-8 blocks against GD and CG
15. [15 — Preconditioning Is Approximate Prediction](reports/15-preconditioning-as-prediction.md) — **the tutorial capstone**: the statistical arc of 09–14 replayed as eight worked steps on the smallest possible examples — the $n=5$ chain in **exact rational arithmetic** (every matrix printed whole: the bridge covariance $5/216 \dots 1/24$, the $1/2$-on-each-neighbor conditionals, sequential-regression coefficients $4/5, 3/4, 2/3, 1/2$, perfect prediction solving in one pass), then the $4\times4$ grid where truncation first bites (IC(0) = truncated Vecchia measured coefficient-by-coefficient, 19% of the wavefront row dropped; measured Richardson tail = $\rho(I - M^{-1}A)$ to five decimals) — ending with the suite's oldest asserted identity made *executable*: PCG **is** CG in the predictor's whitened coordinates, trajectories coinciding to $3.9\times10^{-16}$ at every iterate; every displayed number machine-generated by twin companions ([Python](python/experiments/prediction_tutorial.py), 43 PASS; [Wolfram](mathematica/prediction_tutorial.wls), 13 PASS, all-rational). New readers: this is the on-ramp — read it early

## Layout

- `mathematica/` — reference `.wls` scripts (problem, PCG, eigen checks, Nyström)
- `python/` — `poisson.py`, `pcg.py`, `preconditioners.py`, `nystrom.py`, `neural/`, `experiments/`
- `reports/` — the report suite (00–15)
- `interactive/` — self-contained browser demos (`hierarchical-solvers.html`, `adi-sweep.html`)
- `results/` — JSON summaries + NPO checkpoint (deterministic, reproducible)
- `figures/` — PNGs at dpi=150; `mma_*` are Mathematica exports
