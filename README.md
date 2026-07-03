# poisson-solvers

Implementation and study of **preconditioned conjugate gradients for the 2-D Poisson equation**: a Mathematica reference program ported line-by-line to Python, then used as a fully-instrumented testbed for classical (Jacobi, ILU), randomized ([Nyström, arXiv:2110.02820](https://arxiv.org/abs/2110.02820)), and neural ([NPO, arXiv:2502.01337](https://arxiv.org/abs/2502.01337)) preconditioners.

**Problem**: $A = (d_1 \otimes I + I \otimes d_1)/h^2$ — the 5-point Dirichlet Laplacian on a $32\times32$ interior grid ($h = 1/(n+1)$, $N = n^2 = 1024$, $\kappa(A) = 440.69$ exactly) — solved against a standardized Gaussian-random-field right-hand side to $\|r\|/\|b\| \le 10^{-10}$. Plus a variable-coefficient variant (100:1 coefficient jump) that makes Jacobi nontrivial.

**Why**: at this scale every quantity is exactly computable (analytic spectrum, dense eigensolves, direct-solve ground truth), so each preconditioner's behavior can be *explained*, not just measured — including two instructive failures: Nyström losing to plain CG on a flat-topped spectrum, and plain PCG stalling on a nonlinear neural preconditioner.

## Interactive demo & rendered reports

- **[Iterative solvers explorer](https://yaroslavvb.github.io/poisson-solvers/cg-explorer/)** — interactive dashboard for a 1D heat-conduction Poisson problem (heater at one end, chiller at the other): scrub through the iteration history of CG, SOR, gradient descent, and CG with a toy in-browser neural preconditioner ([arXiv:2502.01337](https://arxiv.org/abs/2502.01337), trained by [python/neural/train_npo_1d.py](python/neural/train_npo_1d.py)).
- **[Rendered report suite](https://yaroslavvb.github.io/poisson-solvers/)** — the reports below as web pages (GitHub Pages).

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

## Layout

- `mathematica/` — reference `.wls` scripts (problem, PCG, eigen checks, Nyström)
- `python/` — `poisson.py`, `pcg.py`, `preconditioners.py`, `nystrom.py`, `neural/`, `experiments/`
- `reports/` — the report suite (00–11)
- `results/` — JSON summaries + NPO checkpoint (deterministic, reproducible)
- `figures/` — PNGs at dpi=150; `mma_*` are Mathematica exports
