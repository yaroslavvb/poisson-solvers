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
reports/                 # this report suite (00..08)
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

# Mathematica cross-checks (Wolfram 15.0)
wolframscript -file mathematica/poisson_pcg.wls    # reference CG/Jacobi run + mma_* figures
wolframscript -file mathematica/eigen_check.wls    # analytic-spectrum verification
wolframscript -file mathematica/nystrom_pcg.wls    # Wolfram Nystrom run (ell = 128)
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
