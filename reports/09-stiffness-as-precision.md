# The Stiffness Matrix Is a Precision Matrix

### A statistical dictionary for solving $-u'' = f$, from Green's functions to preconditioners

*A statistical companion to the suite: it rereads [01](01-code-walkthrough.md), [03](03-gaussian-random-fields.md), [04](04-krylov-and-pcg.md), [05](05-classical-preconditioners.md), and [07](07-nystrom-preconditioning.md) in the vocabulary of Gaussian inference. The setting is the 1-D factor $d_1/h^2$ of the suite's Kronecker-sum operator — the chain, where every object has a closed form; §6–7 say exactly what changes on the 2-D grid. Regression-identity notation follows a companion Mathematica notebook (`whitening_inverse_transposed.nb`, Pourahmadi's parameterization) and the Gram–Schmidt / QR / Cholesky duality of Shawe-Taylor & Cristianini §5.2. Every identity asserted below is verified numerically by [python/experiments/verify_statistical_identities.py](../python/experiments/verify_statistical_identities.py) (19 checks, $n = 8$).*

---

## 1. One problem, two vocabularies

Take the steady 1D heat (Poisson) equation on $(0,1)$ with Dirichlet walls held at zero temperature,

$$-u''(x) = f(x), \qquad u(0) = u(1) = 0,$$

discretized on $n$ interior nodes $x_i = ih$, $h = 1/(n+1)$, giving the familiar system $Au = b$ with

$$A = \frac{1}{h^2}\,\mathrm{tridiag}(-1,\, 2,\, -1).$$

The numerical analyst calls $A$ the **stiffness matrix** (its FEM pedigree) or the discrete Laplacian, and calls solving $Au=b$ "the linear solve." The statistician looks at the same symbols and reads off a Gaussian distribution. The solution $u$ is the unique minimizer of the discrete Dirichlet energy

$$J(u) = \tfrac{1}{2}u^\top A u - b^\top u,$$

and exponentiating the negative energy gives a probability density:

$$p(u) \propto \exp\!\big(-\tfrac{1}{2}u^\top A u + b^\top u\big) = \mathcal{N}\!\big(u \mid A^{-1}b,\; A^{-1}\big).$$

This is the **information form** (canonical / natural-parameter form) of a Gaussian: the pair $(A, b)$ consists of the **precision matrix** and the **information vector** (also called the potential vector or shift). The moment form is $(\Sigma, \mu) = (A^{-1}, A^{-1}b)$. So the single most compressive statement of the whole dictionary is:

> **Solving $Au = b$ is converting a Gaussian from natural parameters to moments.** The solution $u^\star = A^{-1}b$ is the posterior mean; the energy landscape around it has curvature $A$; and every linear solver is, whether it knows it or not, an inference algorithm on this Gaussian.

Two aliases worth keeping in mind: physicists call $\tfrac12 u^\top A u$ the (discrete Dirichlet) energy and $p(u)$ its Gibbs measure; in graphical-model language $(A,b)$ parameterize a **Gaussian Markov random field (GMRF)**, with $A$'s sparsity pattern as the graph.

---

## 2. $A^{-1}$ is a covariance matrix — specifically, a Brownian bridge

The map $b \mapsto A^{-1}b$ is the discrete **Green's function**, and the Green's function of $-d^2/dx^2$ with Dirichlet boundary conditions on $(0,1)$ is

$$G(s,t) = \min(s,t) - st.$$

That kernel is *exactly* the covariance function of the **Brownian bridge** $B_t = W_t - tW_1$ (Brownian motion pinned to zero at both ends — the continuum limit of "zero temperature at both walls"). The discrete identity is clean and exact:

$$(A^{-1})_{ij} = h\,\big(\min(x_i, x_j) - x_i x_j\big) = h\,G(x_i, x_j).$$

So the covariance of the Gaussian whose precision is the stiffness matrix is the Green's function times a quadrature weight. The prior $u \sim \mathcal N(0, A^{-1})$ *is* a Brownian bridge sampled on the grid, and $u^\star = A^{-1}b$ is the bridge's conditional mean given the source — a kriging/smoothing operation with the bridge kernel.

This one identity explains the central structural asymmetry of the subject:

**Sparse precision, dense covariance.** $A$ is tridiagonal but $A^{-1}$ is completely dense. Statistically this is the difference between conditional and marginal structure. Zeros of the precision encode **conditional independence**: $A_{ij} = 0 \iff u_i \perp u_j \mid u_{\text{rest}}$. The three-point stencil says the temperature field is a Markov chain in space — given your two neighbors, the rest of the rod tells you nothing. But *marginal* correlations are long-range: $\mathrm{Corr}(u_i, u_j) > 0$ for every pair, because a point heat source warms the entire rod. The PDE stencil is the conditional-independence graph; the Green's function is the marginal dependence. This is why the statistician (Rue & Held's GMRF program) and the numerical analyst make the same move for the same reason: **never form the inverse; work with the sparse precision and apply its inverse implicitly.**

---

## 3. Every row of $A$ is a regression (and Jacobi's iteration matrix is the notebook's $B$)

The companion notebook's central identity — asserted there as `precisionMat == (I - B).D2` — is the finite-sample version of the textbook Gaussian fact that the precision matrix stores the **full-conditional regressions**. For $u \sim \mathcal N(\mu, A^{-1})$,

$$\mathbb E[u_i \mid u_{-i}] = \mu_i - \frac{1}{A_{ii}}\sum_{j \ne i} A_{ij}(u_j - \mu_j), \qquad \mathrm{Var}(u_i \mid u_{-i}) = \frac{1}{A_{ii}}.$$

In notebook notation: collect into column $j$ of a hollow matrix $B$ the coefficients that predict coordinate $j$ from all the others ($B_{ij} = -A_{ij}/A_{jj}$ for $i \ne j$), and let the diagonal `D2` store the inverse residual variances $1/\sigma_j^2 = A_{jj}$; then $A = (I - B)\,\cdot$`D2` exactly — the notebook's `precisionMat` — and $\mathrm{diag}(A) = $ inverse residual variances — the notebook's `Diagonal[precisionMat] == 1/Diagonal[R'.R]` assertion, verified here too. The precision matrix is *literally a stack of regressions*: divide row $i$ by its diagonal and negate the off-diagonals, and you are reading regression coefficients.

Apply this to the Laplacian row $\tfrac{1}{h^2}(-u_{i-1} + 2u_i - u_{i+1}) = b_i$ and the regression is charming:

$$\mathbb E[u_i \mid u_{-i}] = \frac{u_{i-1} + u_{i+1}}{2} + \frac{h^2}{2}b_i, \qquad \mathrm{Var}(u_i \mid u_{-i}) = \frac{h^2}{2}.$$

Each node's best prediction is the average of its neighbors (plus a source lift). This is the discrete **mean-value property** of harmonic functions wearing a statistics costume: "harmonic" and "martingale" are the same condition.

Now the classical iterative solvers fall out as inference procedures. The **Jacobi iteration** $u \leftarrow u - \mathrm{diag}(A)^{-1}(Au - b)$ has iteration matrix $I - \mathrm{diag}(A)^{-1}A$, which is *exactly the regression matrix $B$* — Jacobi simultaneously replaces every coordinate by its conditional expectation. **Gauss–Seidel** does the same sweep sequentially, using fresh values as it goes: it is a **Gibbs sampler with the noise deleted**, a systematic-scan sweep of conditional means. Add the correct noise $\mathcal N(0, h^2/2)$ to each Gauss–Seidel update and you have, exactly, the Gibbs sampler for $\mathcal N(A^{-1}b, A^{-1})$. This is not just an analogy: the splitting-solver and the Gibbs sampler converge under the same condition and at the same asymptotic rate, $\rho(B)$ (Goodman & Sokal 1989; Fox & Parker 2017 make the equivalence a theorem for general matrix splittings).

The rate is where statistics gives the cleanest *explanation of failure*. For our chain, $\rho(B) = \cos(\pi h) \approx 1 - \pi^2 h^2/2$: each variable is *almost perfectly predicted by its neighbors* — smoothness of the field means the conditional-mean map is nearly the identity, each sweep extracts a sliver of new information, and Jacobi/Gibbs mix in $O(n^2)$ sweeps. And it explains the suite's headline no-op — CG(Jacobi) identical to plain CG, residual histories agreeing to 4.4e-16 ([05 — Classical Preconditioners](05-classical-preconditioners.md)): the Jacobi *preconditioner* $M = \mathrm{diag}(A)$ is an **independence model with homogeneous conditional variances**. It carries zero information about the correlation structure — when the diagonal is constant ($2/h^2$ on the chain, $4/h^2$ on the 2-D grid) it is a scalar, and PCG is invariant to scalar rescalings of $M$ (the invariance proof is in [04](04-krylov-and-pcg.md)). The variable-coefficient contrast problem, where Jacobi earns 5.6×, is the converse: heterogeneous conditional variances are real statistical information, and an independence model that gets them right is no longer vacuous. A statistical model that says "everything is i.i.d." cannot help you whiten a Brownian bridge.

---

## 4. Square roots: design matrices, Cholesky as sequential regression, and why the whitener is an inverse transpose

### 4.1 The stiffness matrix is a Gram matrix

Let $D \in \mathbb R^{(n+1)\times n}$ be the scaled forward-difference (incidence) matrix, $(Du)_k = (u_k - u_{k-1})/h$ with the boundary values grounded. Then

$$A = D^\top D, \qquad J(u) = \tfrac12\|Du\|^2 - b^\top u.$$

So $A$ is the Gram matrix ("uncentered covariance," the notebook's $\mathrm{cov}[X] = X^\top X$) of a design matrix, and $Au = b$ is a set of **normal equations**: the solve is ridgeless least squares in which the "design matrix" is a differential operator and the penalty is roughness. Under the prior, $Du$ is white noise conditioned to sum to zero ($\mathrm{Cov}(Du) = D(D^\top D)^{-1}D^\top$ is the orthogonal projector onto $\{\text{increments summing to }0\}$) — the increments of a bridge.

### 4.2 Cholesky = sequential regression (mchol, phiL2R, phiR2L)

The notebook's `mchol` computes the **modified Cholesky decomposition** $\Sigma = C D^2 C^\top$ with $C$ unit lower triangular — Pourahmadi's regression parameterization of a covariance matrix. Its content: fix an ordering of the variables and write each as a regression on its *predecessors*,

$$u_i = \sum_{j<i}\varphi_{ij}\,u_j + \sigma_i\,\varepsilon_i, \qquad \varepsilon_i \sim \mathcal N(0,1)\ \text{i.i.d.},$$

i.e. $(I - \Phi)\,u = D_\sigma\,\varepsilon$ with $\Phi$ strictly lower triangular. Then $\Sigma = (I-\Phi)^{-1}D_\sigma^2(I-\Phi)^{-\top}$ (so $C = (I-\Phi)^{-1}$) and the precision factors as $A = (I-\Phi)^\top D_\sigma^{-2}(I-\Phi)$. The $\varepsilon_i$ are the **innovations**; the $\sigma_i^2$ are the residual variances of the sequential regressions — verified above: computing $\Phi$ by literally running the regressions makes $(I-\Phi)\Sigma(I-\Phi)^\top$ diagonal. Your `phiL2R` (predict coordinate $i$ from earlier coordinates, "3.26 of Pourahmadi") builds exactly this $\Phi$; `phiR2L` builds its time-reversed twin.

The two directions are not decoration — they are the ordinary and reversed Cholesky factorizations. With $A = LL^\top$ ($L$ lower), the whitening map $z = L^\top u$ reads, row by row, $z_i = L_{ii}u_i + L_{i+1,i}u_{i+1}$: the regression of $u_i$ on its *successors* (phiR2L), coefficient $-L_{i+1,i}/L_{ii}$, innovation s.d. $1/L_{ii}$. For the bridge the coefficients have a closed form with a transparent meaning: $-L_{i+1,i}/L_{ii} = i/(i+1)$, which is just linear interpolation between the known left wall (zero, at distance $i$) and the conditioning neighbor $u_{i+1}$ — pinned diffusion, verified numerically ($\tfrac12, \tfrac23, \tfrac34, \tfrac45, \dots$). Meanwhile the Cholesky factor of the *covariance* runs the regressions forward on predecessors (phiL2R). The exact relationship, verified above with the reversal permutation $P$:

$$\mathrm{chol}(A^{-1}) = P\,L^{-\top}P.$$

**Cholesky of the inverse is the inverse transpose of the Cholesky, read in the opposite variable order.** The two factorizations tell one regression story in the two possible directions of the Markov chain. This is the notebook's title made precise, and it is why Rue's classic GMRF sampler never forms $\mathrm{chol}(\Sigma)$: to draw $u \sim \mathcal N(0, A^{-1})$ you factor the sparse precision $A = LL^\top$ and back-substitute,

$$u = L^{-\top}z, \qquad z\sim\mathcal N(0,I).$$

Sampling is **coloring** by $L^{-\top}$; solving/whitening is applying $L^\top$ (or $A$). One triangular matrix, used in the two directions.

Because the graph is a chain, $L$ is lower **bidiagonal — Cholesky creates no fill-in on a tree**. Hold that thought; it is the entire story of why 1D is easy and 2D/3D is where preconditioning becomes real (§6).

### 4.3 QR and the Shawe-Taylor duality, in whitening language

Shawe-Taylor & Cristianini's §5.2 duality says: Gram–Schmidt on the feature vectors gives $X' = QR$, and the *same* $R$ appears as the Cholesky factor of the kernel matrix, $K = XX' = R'R$ — "Cholesky is the dual implementation of Gram–Schmidt." For our problem the "data matrix" is the difference operator: QR-factor $D = QR$ and (verified, after fixing signs) $R = L^\top = \mathrm{chol}(A)^\top$, with

$$Q = D R^{-1} = D L^{-\top}.$$

Read that as a statistician: **the orthonormal $Q$ of the QR decomposition is the design matrix whitened by the inverse transpose of the Cholesky factor of its Gram matrix.** Gram–Schmidt *is* whitening; $R$ stores the regression coefficients and innovation scales it took to do it; and computing $R$ via QR on $D$ rather than Cholesky on $A = D^\top D$ is the numerically stable "square-root method" that avoids squaring the condition number (the same reason the notebook's singular-matrix experiments found $\mathrm{pinv}(X'X)$ brittle and preferred $\mathrm{pinv}(X)\mathrm{pinv}(X)^\top$, with Tikhonov as the robust fallback — forming the Gram matrix is where ill-conditioning bites). The notebook's remaining assertion, $X(X^\top X)^{-1} = \mathrm{pinv}(X^\top)$ — verified — is the same object from a third angle: "dividing by the covariance" is the transpose of the pseudoinverse, the map that turns the design into its own dual (bi-orthogonal) basis.

One more member of the family: whiteners of $\Sigma$ form an orthogonal orbit $\{U\Sigma^{-1/2} : U^\top U = I\}$. Cholesky whitening picks the *triangular* (causal, ordered) representative; PCA whitening picks the eigenbasis representative — which for our $A$ is the **discrete sine transform**, since $A$'s eigenvectors are $\sin(k\pi x)$ with eigenvalues $4\sin^2(k\pi h/2)/h^2$ (verified), i.e. the **Karhunen–Loève basis of the Brownian bridge** (the DST-I diagonalization of [02 — The Eigenvalue Story](02-eigenvalues.md)); and ZCA whitening $\Sigma^{-1/2}$ is the unique *symmetric* representative — the polar factor of any whitener, the one closest to the identity (Kessy, Lewandowski & Strimmer 2018).

---

## 5. Solvers as inference algorithms

**Direct solve (Thomas algorithm) = Kalman smoothing = belief propagation.** Elimination on a tridiagonal $A$ is Gaussian belief propagation on a chain: each Schur-complement step $A_{22} - A_{21}A_{11}^{-1}A_{12}$ is a **marginalization** of the eliminated variable, the forward sweep is Kalman filtering (accumulate information left to right), and back-substitution is the Rauch–Tung–Striebel smoothing pass. Message passing is exact on trees for the same reason Cholesky has no fill-in on trees: marginalizing a leaf creates no new dependencies. The $O(n)$ tridiagonal solve and the Kalman smoother are one algorithm.

**Conjugate gradients = sequential regression on uncorrelated directions.** CG's defining property, $p_i^\top A p_j = 0$, says the search directions are orthogonal in the $A$-inner product — which is precisely the statement that the linear functionals $p_i^\top A u$ are **uncorrelated random variables** under $u \sim \mathcal N(0, A^{-1})$: $\mathrm{Cov}(p_i^\top Au,\, p_j^\top Au) = p_i^\top A A^{-1} A\, p_j = p_i^\top A p_j$. CG is Gram–Schmidt run in the Mahalanobis geometry of the precision, extracting one new *independent measurement* $p_k^\top b$ per iteration and regressing the solution on the measurements so far; the $A$-norm optimality of $x_k$ over the Krylov space is least-squares optimality of that regression. This reading is a theorem in probabilistic numerics: with prior $u \sim \mathcal N(x_0, A^{-1})$, the posterior means after conditioning on the CG search directions *are* the CG iterates (Cockayne, Oates, Ipsen & Girolami 2019; Hennig 2015).

**Spectral solve = KL rotation.** Diagonalizing with the DST solves the system by rotating to the Karhunen–Loève coordinates where the bridge's components are independent, dividing by the per-mode variances, and rotating back — PCA whitening as a solver.

---

## 6. Preconditioning = fitting a tractable surrogate Gaussian (the regression recipe)

Here is where the dictionary pays rent. A preconditioner is an SPD $M$ with two properties: $M^{-1}r$ is cheap, and $M \approx A$. Statistically: **$M$ is the precision of a surrogate Gaussian model, and PCG is CG run in the surrogate's whitened coordinates.** Factor $M = L_M L_M^\top$; split-preconditioned CG solves

$$\big(L_M^{-1} A\, L_M^{-\top}\big)\,\tilde u = L_M^{-1}b,$$

and $L_M^{-1}AL_M^{-\top}$ is the true precision expressed in coordinates the *surrogate* believes are white. If the surrogate were exact, this matrix is $I$ and CG converges in one step. In general, $\kappa(M^{-1}A)$ — the CG convergence constant — is a **model-mismatch statistic**: the spread of variances the true model assigns to directions the surrogate has normalized to unit variance. Building a preconditioner is a statistical modeling problem: *choose a family of Gaussians you can factor and solve cheaply; fit the best member to $A$; whiten with it.*

The regression view makes the construction mechanical, because §3–4 showed that a precision matrix *is* a collection of regressions. The recipe: fix an elimination ordering; for each variable $i$ choose a small conditioning set $s(i)$ among the not-yet-eliminated variables; compute the coefficients $c_{i}$ and residual variance $\sigma_i^2$ of the regression of $u_i$ on $u_{s(i)}$ (exactly `phiR2L`, restricted to $s(i)$); place $1/\sigma_i$ on the diagonal and $-c_i/\sigma_i$ in the sparse off-diagonal slots of a triangular factor $U$; and take $M = U^\top U$, applied via two sparse triangular solves. If $s(i)$ is everything, the regressions are exact, $U = L^\top$, and $M = A$. Every classical preconditioner is a truncation policy for these regressions:

**Jacobi** keeps no regressors at all: the independence surrogate, retaining only conditional variances. On constant-diagonal $A$ it is a scalar — provably inert, per §3.

**Gauss–Seidel / SSOR** as a preconditioner applies one (or one symmetrized) sweep of conditional-mean updates: an embedded pass of the noise-free Gibbs sampler, with the Fox–Parker equivalence quantifying exactly how much whitening one sweep buys.

**Incomplete Cholesky, IC(0)** keeps only regressors allowed by $A$'s own sparsity pattern: each variable regressed only on its stencil neighbors among the survivors. In spatial statistics this surrogate has a name — the **Vecchia approximation** (Vecchia 1988), the workhorse of large-scale Gaussian-process inference: replace the joint density by $\prod_i p(u_i \mid u_{s(i)})$ with small conditioning sets. The identification is sharp: the sparse triangular factor minimizing the KL divergence from the true Gaussian, subject to the sparsity pattern, is given in closed form by exactly these local regressions (Schäfer, Katzfuss & Owhadi 2021). So "compute the preconditioner from the regression view" is not a metaphor — the KL-optimal sparse factor *is* the truncated-regression factor, and incomplete Cholesky is maximum-likelihood fitting of a sparse autoregression to the field.

**Pivoted / partial decompositions** truncate in the other direction: keep all regressors but only $T \ll n$ regressions. This is Algorithm 5.12 of Shawe-Taylor & Cristianini verbatim — partial Gram–Schmidt / incomplete pivoted Cholesky, where the residual array $d_i$ tracks each variable's **unexplained variance** and the pivot `[a, I(j+1)] = max(d)` is greedy **forward-selection regression**: always adjoin the variable the current model predicts worst. (Their $\nu_j = \sqrt{d_j}$ is the innovation standard deviation; their Proposition 5.11 — rank of the data equals rank of $K$ — is the statement that the Gaussian is supported on a $T$-dimensional subspace.) The resulting surrogate, low-rank factor plus diagonal residual, is a **factor-analysis model**: "the field is a few common factors plus independent noise" — the Nyström / inducing-point approximation of GP regression, recycled as a preconditioner — exactly what [07 — Randomized Nyström Preconditioning](07-nystrom-preconditioning.md) implements. Its instructive negative result is the factor-analysis reading made empirical: the Laplacian's flat-topped spectrum is a field with *no dominant common factors*, so a low-rank-plus-diagonal surrogate has nothing to capture, and even optimal rank-256 deflation barely dents $\kappa$. Their *deflation* (eq. 5.8) is the complementary move: project out directions already known — statistically, condition on the coarse modes and precondition only the residual field.

Why does none of this matter in 1D? Because the chain has no fill-in: IC(0) *is* complete Cholesky, the exact solve is $O(n)$ Kalman smoothing, and preconditioning is theater. The subject becomes real on the suite's 2-D grid — where the ILU result (5 iterations, near-direct at toy scale, [05](05-classical-preconditioners.md)) shows how much of the fill a good incomplete factorization can afford to keep at $N = 1024$ — because eliminating a variable marries its neighbors: **fill-in is marginalization-induced dependence** — integrate out an interior node of a grid and its former neighbors become directly coupled. Approximating the fill (which is all IC, Vecchia orderings, and nested dissection are doing) means choosing which of those induced regressions to keep; multigrid answers hierarchically, with coarse grids as a multiscale statistical model whose coarse-level solves are inference on aggregated variables. The statistical framing tells you *what any good preconditioner must capture*: the long-range marginal correlations that the bridge's Green's function makes unavoidable, using only short-range conditional machinery.

---

## 7. The heat-equation and Matérn coda (and the operator-learning punchline)

Everything above concerned the equilibrium problem; the time-dependent heat equation $u_t = u_{xx}$ adds one more page to the dictionary. The heat semigroup $e^{t\Delta}$ is convolution with a Gaussian kernel of variance $2t$ — running the PDE forward is Gaussian blurring, i.e. adding independent Gaussian increments in KL space, mode $k$ decaying at rate $\lambda_k$. Implicit time stepping makes the statistics explicit: one backward-Euler step solves

$$(I + \Delta t\, A)\,u^{+} = u,$$

and $I + \Delta t\,A = \Delta t\,(\tau^2 - \Delta_h)$ with $\tau^2 = 1/\Delta t$ is precisely a **Whittle–Matérn precision operator**. The suite's GRF right-hand side ([03 — The GRF Right-Hand Side](03-gaussian-random-fields.md), effective Matérn smoothness $\nu = 1$) — spectrum $(|\omega|^2 + \tau^2)^{-\alpha/2}$ applied to white noise — is the spectral implementation of the SPDE construction $(\tau^2 - \Delta)^{\alpha/2}\,g = \mathcal W$ (Whittle 1954; Lindgren, Rue & Lindström 2011): Matérn fields are exactly the fields whose precision is a power of a shifted Laplacian, which is *why* Matérn GPs admit sparse GMRF representations and fast solvers at all. Note the symmetry with §4.2: the generator **colors** white noise by a covariance square root in the KL basis; the solver **whitens** by the precision. Same factor, opposite directions, and "inverse transposed" is the hinge between them.

For the neural preconditioner of [06 — Neural Preconditioning Operator](06-neural-preconditioner.md) this closes a loop. §6 said a preconditioner is a surrogate covariance; a network trained to approximate the inverse is therefore doing *covariance estimation* — learning the kriging map of the field — and its measured success mode, spectral clustering rather than raw $\kappa$ reduction, is what a good-but-inexact covariance model looks like to CG. More broadly, for operator learning trained on pairs $(f, u = A^{-1}f)$ with GRF forcing $f \sim \mathcal N(0, \Sigma_f)$, the pairs are *jointly Gaussian* with $\mathrm{Cov}(u) = A^{-1}\Sigma_f A^{-\top}$ and cross-covariance $\Sigma_f A^{-\top}$; the Bayes-optimal map from $f$ to $u$ is the linear kriging operator, which here is $A^{-1}$ itself, and the smoothing it applies — two extra powers of spectral decay per solve — is why solved fields are so much tamer than their sources.

---

## 8. The dictionary

| Numerical linear algebra / PDE | Statistics / probability |
|---|---|
| Stiffness matrix $A$ (discrete $-\Delta$, Dirichlet) | Precision (inverse covariance) matrix of a GMRF; natural parameter |
| Right-hand side $b$; solving $Au=b$ | Information (potential) vector; converting natural parameters to moments — posterior mean |
| Dirichlet energy $\tfrac12 u^\top Au - b^\top u$ | Negative log-density of $\mathcal N(A^{-1}b, A^{-1})$ |
| Stencil / sparsity pattern of $A$ | Conditional-independence (Markov) graph |
| $A^{-1}$; Green's function $\min(s,t)-st$ | Covariance matrix; Brownian-bridge kernel |
| Row equation $u_i = \tfrac12(u_{i-1}+u_{i+1}) + \tfrac{h^2}{2}b_i$ | Full-conditional regression: $\mathbb E[u_i \mid u_{-i}]$, variance $1/A_{ii}$ |
| Notebook: $A = $ `(I − B).D2` | Precision assembled from regress-on-the-rest coefficients and inverse residual variances |
| Jacobi iteration matrix $I - \mathrm{diag}(A)^{-1}A$ | The regression matrix $B$; simultaneous conditional-mean update |
| Gauss–Seidel / SOR sweep | Systematic-scan Gibbs sampler with the noise removed (same rate: Goodman–Sokal, Fox–Parker) |
| $\rho(\text{iteration matrix}) \to 1$ as $h \to 0$ | Neighbors nearly determine each node; vanishing innovation per sweep |
| $A = D^\top D$, $D$ = difference operator | Normal equations of a regression whose design matrix is the operator; roughness penalty |
| Cholesky $A = LL^\top$ | Sequential (autoregressive) factorization; regressions on successors (`phiR2L`); innovations rep. |
| `mchol`: $\Sigma = CD^2C^\top$ (Pourahmadi) | Regressions on predecessors (`phiL2R`); modified Cholesky = ordered AR model |
| $\mathrm{chol}(A^{-1}) = P L^{-\top} P$ | Same regressions read in the reversed variable order |
| Back-substitution $u = L^{-\top}z$ | Sampling from the GMRF (coloring); Rue's algorithm |
| $z = L^\top u$; and QR: $Q = DL^{-\top}$ | Whitening; Gram–Schmidt = whitening by the inverse-transposed Cholesky factor |
| QR of $D$ vs Cholesky of $D^\top D$ | Square-root method vs normal equations (stability; the notebook's pinv experiments) |
| $X(X^\top X)^{-1} = \mathrm{pinv}(X^\top)$ | "Dividing by covariance" = dual basis = transposed pseudoinverse |
| DST diagonalization; eigenpairs $\sin(k\pi x)$, $4\sin^2(k\pi h/2)/h^2$ | Karhunen–Loève expansion of the bridge; PCA whitening |
| $\Sigma^{-1/2}$ (symmetric square root) | ZCA whitening; polar factor of any whitener |
| Thomas algorithm; elimination + back-substitution | Kalman filter + RTS smoother; Gaussian belief propagation on the chain |
| Schur complement; fill-in | Marginalization; dependence induced by integrating variables out |
| CG; $A$-conjugate directions | Sequential regression on measurements uncorrelated under $\mathcal N(0,A^{-1})$; BayesCG posterior means |
| Preconditioner $M \approx A$; $\kappa(M^{-1}A)$ | Tractable surrogate Gaussian; whitening under the surrogate; model-mismatch statistic |
| Jacobi preconditioner | Independence surrogate (conditional variances only) — inert when the diagonal is constant |
| Incomplete Cholesky IC(0) | Vecchia approximation; KL-optimal sparse autoregression (Schäfer–Katzfuss–Owhadi) |
| Pivoted partial Cholesky (Alg. 5.12; pivot on $d_i$) | Forward-selection regression on unexplained variance; Nyström / inducing points; factor analysis (low-rank + diagonal) |
| Deflation of known eigenvectors (eq. 5.8) | Conditioning on known factors / coarse modes |
| Multigrid coarse-grid correction | Hierarchical model; inference on aggregated (marginalized) variables |
| Backward Euler operator $I + \Delta t\,A$ | Whittle–Matérn precision, $\tau^2 = 1/\Delta t$ |
| Spectral GRF generator $(|\omega|^2+\tau^2)^{-\alpha/2}\cdot$noise | SPDE sampling $(\tau^2-\Delta)^{\alpha/2}g = \mathcal W$ (Lindgren–Rue–Lindström) |
| Solution operator $A^{-1}$ acting on random $f$ | Kriging map; Bayes-optimal linear operator for the joint $(f,u)$ Gaussian |

---

## 9. Pointers

The GMRF half of the dictionary — precision sparsity as Markov structure, sampling by $L^{-\top}$, Cholesky orderings — is Rue & Held, *Gaussian Markov Random Fields* (2005); the Matérn/SPDE bridge is Whittle (1954) and Lindgren, Rue & Lindström (JRSS-B 2011). The regression parameterization of covariance ($\Phi$, $D^2$, modified Cholesky) that the companion notebook implements is Pourahmadi (Biometrika 1999; Statistical Science 2011 review). The sampler–solver equivalence is Goodman & Sokal (1989) and Fox & Parker (Bernoulli 2017); the CG-as-Bayesian-inference results are Hennig (SIOPT 2015) and Cockayne, Oates, Ipsen & Girolami (Bayesian Analysis 2019). Vecchia (JRSS-B 1988) introduced the truncated-conditioning surrogate; Schäfer, Katzfuss & Owhadi (SISC 2021) prove the KL-optimality of sparse-Cholesky-by-local-regression that makes "preconditioner = fitted autoregression" exact. The whitening taxonomy (Cholesky / PCA / ZCA and the orthogonal orbit) is Kessy, Lewandowski & Strimmer (American Statistician 2018). And the QR–Cholesky–Gram–Schmidt duality with the pivoted partial algorithm is Shawe-Taylor & Cristianini, *Kernel Methods for Pattern Analysis*, §5.2.
