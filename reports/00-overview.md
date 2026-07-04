# 00 — Overview: Start Here

## What this repo is

An implementation and study of **preconditioned conjugate gradients for the 2-D Poisson equation**, grown out of a Mathematica reference program ([mathematica/poisson_pcg.wls](../mathematica/poisson_pcg.wls)) and ported line-by-line to Python. On one fully-instrumented toy problem — the 5-point Dirichlet Laplacian on a $32\times32$ interior grid ($N = 1024$), solved against a Gaussian-random-field right-hand side — it benchmarks four families of preconditioner inside the same PCG/FCG harness:

1. **Classical**: identity, Jacobi, ILU ([python/preconditioners.py](../python/preconditioners.py));
2. **Randomized**: the Nyström preconditioner of Frangella–Tropp–Udell, [arXiv:2110.02820](https://arxiv.org/abs/2110.02820) ([python/nystrom.py](../python/nystrom.py));
3. **Neural**: a toy Neural Preconditioning Operator (NPO) after Li, Xiao, Lai & Wang, [arXiv:2502.01337](https://arxiv.org/abs/2502.01337) — a ~50k-parameter attention-multigrid network trained to approximate the inverse ([python/neural/](../python/neural/));
4. **Nothing**, as the baseline plain CG.

Notation used everywhere: $d_1$ is the 1-D tridiagonal $[-1,2,-1]$ stencil (no $1/h^2$); $A = (d_1\otimes I + I\otimes d_1)/h^2$ with $h = 1/(n+1)$, $n = 32$, $N = n^2 = 1024$; $\kappa(A) = 440.69$; $b$ = `grf_rhs(32, alpha=2.0, tau=3.0, seed=42)`; tolerance $\|r_k\|/\|b\| \le 10^{-10}$; iterations = `len(res_hist) - 1`.

The punchline of the study is that each method behaves exactly as its theory predicts *once the problem's spectrum is understood*: ILU nearly factorizes the toy problem and wins outright; Jacobi is provably a no-op on constant coefficients and worth 5.6× on a 100:1 coefficient jump; Nyström *loses* to plain CG because the Laplacian's flat-topped spectrum is its adversarial case; and the NPO cuts iterations 3.87× via eigenvalue clustering — but only inside *flexible* CG, because a ReLU network is not a fixed SPD matrix and plain PCG demonstrably stalls on it.

## Repo map

```
mathematica/
  poisson_pcg.wls        # reference program: problem + GRF RHS + functional PCG (NestWhile), CG vs Jacobi
  eigen_check.wls        # verifies spec(d1), spec(A), kappa against closed forms
  nystrom_pcg.wls        # independent Wolfram implementation of the Nystrom preconditioner (ell = 128)
  fdt_fluctuations.wls   # independent Wolfram check of the FDT identities + exact B-pattern checks (report 10)
  grid8_regressions.wls  # independent Wolfram check of the 8x8 grid regressions + the two ArrayPlots (report 11)
  decoupling_adi.wls     # independent Wolfram check of report 13: ADI closed-form spectrum, semiseparable identity (exact), dense kappas
python/
  poisson.py             # laplacian_1d, poisson_2d, variable_poisson_2d, grf_rhs
  pcg.py                 # pcg (Fletcher-Reeves PCG) + flexible_pcg (Notay Polak-Ribiere FCG)
  preconditioners.py     # identity(), jacobi(A), ilu(A, **kw) factories -> callables z = M(r)
  nystrom.py             # NystromPreconditioner class (Algorithm 1 of arXiv:2110.02820, mu >= 0)
  neural/
    npo.py               # NPO network (NAMG-lite transformer) + NPOPreconditioner checkpoint wrapper
    train_npo.py         # trains the NPO (3 losses, 400 epochs) -> results/npo_checkpoint.pt
    eval_npo.py          # held-out eval: CG vs FCG(NPO) vs plain-PCG(NPO) negative control
  experiments/
    run_baseline.py      # CG vs Jacobi on the canonical problem -> results/baseline.json
    spectra.py           # eigenvalue verification + kappa(n) scaling -> results/spectra.json
    run_nystrom.py       # rank sweep 16..256 + exact preconditioned spectra -> results/nystrom.json
    npo_spectrum.py      # column-linearized NPO spectrum diagnostics -> results/npo_spectrum.json
    run_all.py           # consolidated benchmark + sanity checks -> results/results.json
    verify_statistical_identities.py  # machine-checks every identity of report 09 (19 PASS)
    fdt_fluctuations.py  # fluctuation-dissipation experiments of report 10 -> results/fdt.json + fdt_* figures
    grid_regressions_multiscale.py  # report 11 grid regressions + two-level experiments -> results/grid_multiscale.json + grid8_*/twolevel_* figures
    richardson_ar.py     # report 12 stationary Richardson ladder (perfect AR + truncations, no CG) -> results/richardson_ar.json + richardson_* figures
    decoupling.py        # report 13 decoupling axes (diagonal/DST/ADI/space) + GD-vs-CG -> results/decoupling.json + decoupling_* figures
reports/                 # this report suite (00..13)
results/                 # JSON summaries, npo_checkpoint.pt, npo_training_history.json
figures/                 # all PNGs (dpi=150); mma_* are the Mathematica exports
pyproject.toml, uv.lock  # Python env (3.12: numpy, scipy, matplotlib, torch)
```

## How to run everything

All Python scripts are run from the repo root with `uv`:

```bash
# Python experiments (deterministic; regenerate results/*.json and figures/*.png)
uv run python python/experiments/run_baseline.py   # CG vs Jacobi baseline
uv run python python/experiments/spectra.py        # eigenvalue verification + kappa scaling
uv run python python/experiments/run_nystrom.py    # Nystrom rank sweep + exact kappas
uv run python python/neural/train_npo.py           # train the NPO (~216 s CPU) -> checkpoint
uv run python python/neural/eval_npo.py            # NPO evaluation (needs checkpoint)
uv run python python/experiments/npo_spectrum.py   # NPO linearized-spectrum study (needs checkpoint)
uv run python python/experiments/run_all.py        # consolidated matrix + hard sanity assertions
uv run python python/experiments/verify_statistical_identities.py  # report 09 identity checks (19 PASS)
uv run python python/experiments/fdt_fluctuations.py               # report 10 FDT experiments (52 PASS)
uv run python python/experiments/grid_regressions_multiscale.py    # report 11 grid + multiscale checks (36 PASS)
uv run python python/experiments/richardson_ar.py                  # report 12 Richardson ladder (53 PASS)
uv run python python/experiments/decoupling.py                     # report 13 decoupling checks (44 PASS)

# Mathematica cross-checks (Wolfram 15.0)
wolframscript -file mathematica/poisson_pcg.wls    # reference CG/Jacobi run + mma_* figures
wolframscript -file mathematica/eigen_check.wls    # analytic-spectrum verification
wolframscript -file mathematica/nystrom_pcg.wls    # Wolfram Nystrom run (ell = 128)
wolframscript -file mathematica/fdt_fluctuations.wls  # report 10 Wolfram cross-check (10 PASS)
wolframscript -file mathematica/grid8_regressions.wls  # report 11 Wolfram cross-check (6 PASS)
wolframscript -file mathematica/decoupling_adi.wls  # report 13 Wolfram cross-check (7 PASS)
```

Everything Python-side is bit-deterministic (GRF seed 42, training seeds 100–139, Nyström seed 0, torch seed 0); reruns reproduce the committed JSONs exactly. The Mathematica runs use a different RNG stream, so they are the *same distribution but a different draw*: 115 CG iterations vs Python's 116 — statistically equivalent, deliberately not bit-matched (divergence ledger in [01-code-walkthrough.md](01-code-walkthrough.md) §3).

## Headline results

Canonical problem `poisson_2d(32)`, $b$ = GRF (seed 42), tol $10^{-10}$ — from [results/results.json](../results/results.json):

| method | iterations | final relres | solve wall [s] | setup [s] | converged |
|---|---:|---:|---:|---:|---|
| CG (none) | 116 | 6.67e-11 | 0.00117 | — | yes |
| CG (Jacobi) | 116 | 6.67e-11 | 0.00119 | ~0 | yes (provably ≡ plain CG) |
| CG (ILU, `spilu` defaults) | **5** | 6.21e-13 | 0.00024 | 0.00100 | yes |
| CG (Nyström, rank 16) | 123 | 9.52e-11 | 0.00184 | 0.00074 | yes |
| CG (Nyström, rank 64) | 123 | 8.01e-11 | 0.00188 | 0.00418 | yes |
| CG (Nyström, rank 128) | 122 | 7.73e-11 | 0.00200 | 0.01156 | yes |
| CG (Nyström, rank 256) | 119 | 9.18e-11 | 0.00233 | 0.02384 | yes |
| **FCG (NPO, Notay)** | **30** | 7.16e-11 | 0.02766 | 0.00272 | yes |
| CG (NPO, plain PCG — negative control) | 2000 | 9.65e-06 | 1.68 | 0.00272 | **no — stalls at ~1e-5** |

Variable-coefficient problem `variable_poisson_2d(32, contrast=100)` (same RHS):

| method | iterations | final relres |
|---|---:|---:|
| CG (none) | 771 | 8.98e-11 |
| CG (Jacobi) | **137** | 9.22e-11 |

Key spectral numbers: $\kappa(A) = 440.69$ (exact, matches $\cot^2\frac{\pi}{2(n+1)}$ to all printed digits); Nyström exact preconditioned $\kappa$ = 439.62 / 434.52 / 426.58 / 407.46 for ranks 16/64/128/256; NPO linearized spectrum entirely in the right half-plane with modulus spread **12.56** (35× tighter than $\kappa(A)$), 98.1% of eigenvalues within $[0.5,2]\times$ median. Every converged solve matches `scipy` `spsolve` to ≤ 3.1e-11 relative error.

## Reading order

The reports are written to be read in numerical order; each is self-contained but cross-linked.

**[01 — Code Walkthrough](01-code-walkthrough.md).** Construct-by-construct dissection of the Mathematica reference program (`Band`/`SparseArray`, the `RotateRight` frequency trick, the `NestWhile`/`Sow`/`Reap` functional PCG) and its Python port. Includes the complete divergence ledger — every place the port differs (RNG stream, FFT normalization, `ddof`) and why results are statistically equivalent but not bit-identical across languages.

**[02 — The Eigenvalue Story](02-eigenvalues.md).** Derives the full spectral theory from scratch: sine eigenvectors of $d_1$ from the three-term recurrence, DST-I diagonalization, the Kronecker-sum eigenpair theorem, and the exact $\kappa(A) = \cot^2\frac{\pi}{2(n+1)} \approx 0.41\,n^2$. Everything is verified to float64 precision by both `spectra.py` and `eigen_check.wls`, and §8 previews what each preconditioner does to this spectrum.

**[03 — The GRF Right-Hand Side](03-gaussian-random-fields.md).** Line-by-line derivation of the spectral GRF sampler: the fftshift-ordered frequency grid, the effective Matérn parameters ($\nu = 1$, $\tau_{\text{eff}} = \tau/n$), and a proof that `Re(IFFT)` of non-Hermitian complex noise yields a legitimate stationary GRF at exactly half variance. Quantifies the torus-vs-Dirichlet-box mismatch and why it is harmless for solver benchmarking.

**[04 — Krylov and PCG](04-krylov-and-pcg.md).** CG as $A$-norm minimization over Krylov subspaces, the exact 5-line recurrence and its line-for-line match with the Mathematica `pcgStep`, preconditioning as a change of inner product, and the proof of PCG's invariance under $M \to cM$. Derives the Chebyshev $\sqrt{\kappa}$ bound (predicts 249–281 iterations; measured 116) and the Notay flexible-CG $\beta$ that the nonlinear NPO requires.

**[05 — Classical Preconditioners](05-classical-preconditioners.md).** Jacobi and ILU in depth: the theorem that Jacobi ≡ plain CG on constant coefficients (residual histories agree to 4.4e-16), the variable-coefficient contrast where Jacobi earns 5.6× (κ drops 17767 → 428), and ILU's 5-iteration near-direct solve at toy scale, with its honest caveats. Ends with why multigrid — not any of these — is the right asymptotic answer for Poisson, and why the NPO paper builds a *neural multigrid*.

**[06 — Neural Preconditioning Operator](06-neural-preconditioner.md).** Digest of the NPO paper (three losses, NAMG architecture, theory) and the toy reimplementation: the $h^2 A$ training-scale trick, training on 40 GRFs + recorded CG residuals, and the headline 116 → 30 (3.87×, closely matching the paper's 113 → 34). The linearized-spectrum study shows the win comes from clustering (spread 12.56, 98.1% within a factor-2 band) despite 57% nonsymmetry and 43% nonlinearity — plus the deliberate plain-PCG stall that shows why flexible CG is non-negotiable.

**[07 — Randomized Nyström Preconditioning](07-nystrom-preconditioning.md).** The stabilized Nyström construction and preconditioner, implemented exactly and verified against dense eigensolves. The instructive negative result: all ranks take *more* iterations than plain CG (119–123 vs 116) despite provably smaller exact $\kappa$, because the Laplacian's flat-topped spectrum caps even *optimal* rank-256 deflation at $\kappa = 298$ — and the report maps precisely the (ridge/kernel) regime where the method actually delivers its $\mathbb{E}\,\kappa < 28$ guarantee.

**[08 — Consolidated Results](08-results.md).** The full method matrix, wall-time accounting (setup vs solve, µs/iteration), the cross-method κ/clustering table, sanity checks and anomalies (including the reported-not-asserted Nyström rank-16/64 tie), what wins in which regime, limitations, and next steps. If you read only one report, read this one.

**[09 — The Statistical Dictionary](09-stiffness-as-precision.md).** A coda that rereads the suite in the language of Gaussian inference: the stiffness matrix as a GMRF precision matrix (whose inverse is a Brownian-bridge covariance in closed form), Jacobi/Gauss–Seidel as conditional-expectation sweeps (Gibbs sampling minus the noise), Cholesky and QR as sequential regression and whitening, CG as conditioning on precision-uncorrelated measurements, and preconditioning as fitting a tractable surrogate Gaussian — incomplete Cholesky as the Vecchia approximation, Nyström as factor analysis — which makes 05's Jacobi no-op and 07's negative result corollaries of one statistical picture. Every identity is machine-verified by [verify_statistical_identities.py](../python/experiments/verify_statistical_identities.py).

**[10 — Kick It, or Watch It Jitter](10-fluctuation-dissipation.md).** The physics companion to 09, running the dictionary in the fluctuation–dissipation direction: hold the discretized rod at temperature $k_BT$ and show that everything a solver needs is measurable from the equilibrium jitter alone — the kick response $A^{-1}e_j$ equals the fluctuation covariance column, regressions on thermal snapshots recover the Jacobi matrix $B$, both Cholesky factors and their closed-form bridge coefficients, and an IC(0)-pattern Vecchia preconditioner fitted purely to snapshots cuts the canonical 2-D solve from 116 to 30 iterations ($\kappa$ 440.69 → 13.7). Along the way it measures the Euler–Maruyama sampler's exact $O(\Delta t)$ bias, identifies critical slowing down / MCMC mixing / solver stall as one phenomenon with $\kappa$ as its dimensionless face, and reads 2-D fill-in as walk-sum marginalization. Machine-checked by [fdt_fluctuations.py](../python/experiments/fdt_fluctuations.py) (52 checks) and independently by [fdt_fluctuations.wls](../mathematica/fdt_fluctuations.wls) (10 checks).

**[11 — Predict Thy Neighbor, Subtract the Average](11-regressions-and-multiscale.md).** The worked-example companion to 09/10: the whole regression dictionary run on the 8×8 grid — the smallest 2-D problem where preconditioning is real — reading $A^{-1}$ and $B$ pixel by pixel, the Green's function as a discounted sum over lattice random walks, Cholesky fill as elimination-wavefront regressions, and the first explicit measurement of the IC(0)-vs-Vecchia coefficient gap (~22% where fill matters). It then builds the multiscale alternative from regression on block averages ($R^2$: 15% for one global mean → 57% for sixteen) and races an additive two-level IC(0)+coarse preconditioner on a hot/cold-rod $n=32$ problem — 76 → 32 iterations, $\kappa$ 440.69 → 11.05, with error-field snapshots showing exactly which smooth mode each method leaves behind. Machine-checked by [grid_regressions_multiscale.py](../python/experiments/grid_regressions_multiscale.py) (36 checks) and independently by [grid8_regressions.wls](../mathematica/grid8_regressions.wls) (6 checks).

**[12 — The Preconditioner Is an Autoregressive Predictor](12-autoregressive-preconditioning.md).** The synthesis report 09–11 build toward: CG is removed entirely and every preconditioner is run as the *same* stationary predict-and-correct Richardson iteration, so each model's quality is exposed as one number — $\rho(I - CA)$, the fraction of structure the predictor fails to explain per sweep. The trichotomy at its core: the perfect two-sided regressions applied synchronously to stale neighbors are Jacobi ($\rho = \cos\pi h = 0.9955$, provably identical to optimally damped Richardson); the same weights applied sequentially with fresh values are Gauss–Seidel ($\rho = \cos^2\pi h$ — the rate exponent exactly doubles); and the perfect one-sided (causal) predictor $\Phi, d^2$ — built two independent ways, its wavefront support, screening decay, and Schur/Dirichlet-to-Neumann identities dissected weight by weight — solves in one step because triangularity turns prediction into back-substitution. The truncation ladder then quantifies everything in between: IC(0) at $\rho = 0.9697$ with a measured two-phase slope, coarse-only Galerkin projections that stall at exactly the energy fraction outside the coarse space, block-spin vs bilinear coarse variables (110 vs 17 sweeps), and mesh-independence of the two-grid cycle ($\rho \approx 0.357$ at $n = 16/32/64$) against the $1-\rho = O(h^2)$ critical slowing down of every local method. Machine-checked by [richardson_ar.py](../python/experiments/richardson_ar.py) (53 checks, ~6 s).

**[13 — Preconditioning Is Decoupling](13-preconditioning-as-decoupling.md).** The capstone reading of the suite: the difficulty of minimizing $J(u) = \tfrac12 u^\top Au - b^\top u$ *is* the coupling — the cross-partials of the energy = the off-diagonal precision entries = the conditional dependencies of the Gibbs field (three vocabularies, one number, FD-verified) — and every preconditioner is a scheme for splitting one entangled minimization into subproblems that can be optimized (nearly) separately. Five axes of decoupling are run as one ladder on the hot/cold-rod problem: **coordinates** (a scalar diagonal decouples nothing — Jacobi-GD ≡ GD, 1998 = 1998 iterations), **frequency** (the DST-I/KL rotation is the exact decoupler: one-pass solve to $1.3\times10^{-15}$), **direction** (one ADI double-sweep exploits $A = H + V$ with $[H,V] = 0$ for the measured square-root effect: $\kappa$ 440.69 → 10.52 = 0.50$\sqrt{\kappa}$, closed-form spectrum to $3.4\times10^{-15}$), **space** (a 2-column separator gives $\mathrm{Cov}(L,R\mid I) = 0$ exactly; block-Jacobi(2) leaves 960 of 1024 eigenvalues at exactly 1 and CG needs 12 iterations on the rank-32 interface remainder), and **scale** ([11](11-regressions-and-multiscale.md)'s two-level rebuilt verbatim, 31 iterations). A semiseparable interlude reads the exact rank-1 triangles of the 1-D inverse ($h\,x_i(1-x_j)$ — [09](09-stiffness-as-precision.md)'s Brownian-bridge kernel) as the covariance-side origin of low-rank far-field structure. The closing GD-vs-CG section measures *why CG is fast*: GD pays the spectrum's **range** — rate $(\kappa-1)/(\kappa+1)$ to 7–8 digits, counts growing linearly in $\kappa$ up to 11.5M iterations at $\kappa = 10^6$ — while CG pays the number of **clusters** (two distinct eigenvalues ⟹ exactly 2 iterations at every coupling strength; three clusters ⟹ 18), because $A$-orthogonal directions are an on-the-fly sequential decoupler; the two mechanisms compose, with $\sqrt{\kappa}$ bounding all four ladder methods but spectral *shape* deciding within the bound (ADI burns 88% of its Chebyshev budget, block-Jacobi(2) 23%). Machine-checked by [decoupling.py](../python/experiments/decoupling.py) (44 checks, ~2.4 s) and independently by [decoupling_adi.wls](../mathematica/decoupling_adi.wls) (7 checks, exact integer arithmetic + dense eigensolves).
