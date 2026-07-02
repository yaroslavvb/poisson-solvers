"""Train the toy Neural Preconditioning Operator (NPO) on 32x32 Poisson.

Follows Li et al., arXiv:2502.01337v2, Sec. 3.3, 5.1.1-5.1.2:

* Dataset: right-hand sides ``b`` sampled from a Gaussian random field
  (``poisson.grf_rhs``, seeds 100, 101, ...); residual vectors ``r_k``
  recorded along baseline (unpreconditioned) CG runs via ``pcg.pcg``
  (tol 1e-10, cap 100 iterations, paper Sec. 5.1.2).
* Losses (all scale-free; total = sum of the three, paper Table 3 ablation):
    condition loss (Eq. 9):  mean_i ||(I - A M(r_i)) r_i||^2 / ||r_i||^2
                             over recorded CG residuals r_i;
    residual  loss (Eq. 10): mean_i ||A M(b_i) - b_i||^2 / ||b_i||^2
                             over GRF right-hand sides b_i;
    data loss:               mean_i ||M(v_i) - x_i||^2 / ||x_i||^2 with the
                             exact targets x_i = A^{-1} v_i (sparse LU) for
                             every sample v_i — for a recorded CG residual
                             r_k this target is exactly the error
                             e_k = x* - x_k of the recorded partial state,
                             since A e_k = r_k.

SCALING (see also npo.py): the network is trained to invert the SCALED
matrix ``A_hat = h^2 * A`` (spectrum ~[0.018, 8] for n=32 — friendly float32
numerics). PCG is invariant to positive scaling of M, so M ~ A_hat^{-1}
serves directly as a preconditioner for A. All training samples are
normalized to unit norm; the runtime wrapper (``NPOPreconditioner``)
applies the same normalization.

Saves ``results/npo_checkpoint.pt`` and ``results/npo_training_history.json``.
Run from the repo root: ``uv run python python/neural/train_npo.py``.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcg import pcg
from poisson import grf_rhs, poisson_2d
from npo import NPO

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- configuration -----------------------------------------------------
N_GRID = 32
NUM_RHS = 40                       # GRF right-hand sides, seeds 100..139
FIRST_SEED = 100
RECORD_ITERS = (1, 2, 4, 8, 16, 32, 64)   # CG residual snapshots per run
CG_TOL = 1e-10                     # baseline-run settings (paper Sec. 5.1.2)
CG_MAXITER = 100
MODEL_CONFIG = {"width": 32, "num_coarse": 64, "num_heads": 4, "ffn_mult": 4}
EPOCHS = 400
BATCH = 32
LR = 2e-3                          # peak lr after linear warmup
LR_MIN = 5e-5
WARMUP_EPOCHS = 10


class _RecordingIdentity:
    """Identity preconditioner that records every residual pcg hands it.

    ``pcg`` evaluates ``z = M(r)`` once at initialization (r = b) and once
    per iteration, so ``residuals[k]`` is the residual after k iterations.
    """

    def __init__(self):
        self.residuals = []

    def __call__(self, r):
        self.residuals.append(r.copy())
        return r


def build_dataset(a_hat):
    """Assemble (inputs, exact targets, rhs-mask) as float32 tensors.

    Every sample is unit-normalized; targets are ``A_hat^{-1} v`` via a
    single sparse LU factorization (the data-loss targets — exact solves
    for the rhs samples, exact CG errors for the residual samples).
    """
    lu = spla.splu(a_hat.tocsc())
    inputs, targets, is_rhs = [], [], []
    for k in range(NUM_RHS):
        b = grf_rhs(N_GRID, alpha=2.0, tau=3.0, seed=FIRST_SEED + k)
        rec = _RecordingIdentity()
        # Baseline Krylov run on the scaled system; A_hat and A generate the
        # same Krylov spaces so the recorded residuals equal those of a run
        # on (A, b).
        pcg(a_hat, b, M=rec, tol=CG_TOL, maxiter=CG_MAXITER)
        samples = [(b, True)]
        samples += [
            (rec.residuals[i], False)
            for i in RECORD_ITERS
            if i < len(rec.residuals)
        ]
        for v, rhs_flag in samples:
            u = v / np.linalg.norm(v)
            inputs.append(u)
            targets.append(lu.solve(u))
            is_rhs.append(rhs_flag)
    x = torch.tensor(np.array(inputs, dtype=np.float32))
    z = torch.tensor(np.array(targets, dtype=np.float32))
    return (
        x.reshape(-1, 1, N_GRID, N_GRID),
        z.reshape(-1, 1, N_GRID, N_GRID),
        torch.tensor(is_rhs),
    )


def main():
    torch.manual_seed(0)
    t0 = time.time()

    h = 1.0 / (N_GRID + 1)
    a = poisson_2d(N_GRID)
    a_hat = (a * h**2).tocsr()  # scaled operator, spectrum ~[0.018, 8]

    x_all, z_all, rhs_all = build_dataset(a_hat)
    n_samples = x_all.shape[0]
    print(f"dataset: {n_samples} samples "
          f"({int(rhs_all.sum())} GRF rhs, {int((~rhs_all).sum())} CG residuals)")

    # A_hat as a float32 torch CSR matrix for the loss matvecs.
    coo = a_hat.tocoo()
    a_t = torch.sparse_coo_tensor(
        np.vstack([coo.row, coo.col]),
        coo.data.astype(np.float32),
        coo.shape,
    ).coalesce().to_sparse_csr()

    model = NPO(**MODEL_CONFIG)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params} parameters, config {MODEL_CONFIG}")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [
            torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=0.1, total_iters=WARMUP_EPOCHS
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=LR_MIN
            ),
        ],
        milestones=[WARMUP_EPOCHS],
    )

    history = []
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_samples)
        sums = {"total": 0.0, "condition": 0.0, "residual": 0.0, "data": 0.0}
        n_batches = 0
        for start in range(0, n_samples, BATCH):
            idx = perm[start : start + BATCH]
            xb, zb, rhs_b = x_all[idx], z_all[idx], rhs_all[idx]

            out = model(xb)
            of = out.flatten(1)                     # M(v), (B, N)
            xf = xb.flatten(1)                      # v, unit norm
            zf = zb.flatten(1)                      # A_hat^{-1} v

            # ||A_hat M(v) - v||^2; inputs have unit norm so this is already
            # the normalized quantity of paper Eq. 9 / Eq. 10.
            av = torch.sparse.mm(a_t, of.t()).t()
            op_err = (av - xf).pow(2).sum(dim=1)
            cond_mask, rhs_mask = ~rhs_b, rhs_b
            l_cond = op_err[cond_mask].mean() if cond_mask.any() else of.sum() * 0.0
            l_res = op_err[rhs_mask].mean() if rhs_mask.any() else of.sum() * 0.0
            # Data loss with exact targets, normalized by ||target||^2.
            l_data = ((of - zf).pow(2).sum(dim=1)
                      / zf.pow(2).sum(dim=1)).mean()

            loss = l_cond + l_res + l_data
            opt.zero_grad()
            loss.backward()
            opt.step()

            sums["total"] += float(loss)
            sums["condition"] += float(l_cond)
            sums["residual"] += float(l_res)
            sums["data"] += float(l_data)
            n_batches += 1
        sched.step()

        entry = {k: v / n_batches for k, v in sums.items()}
        entry["epoch"] = epoch
        entry["lr"] = sched.get_last_lr()[0]
        history.append(entry)
        if epoch % 25 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:4d}  total {entry['total']:.4e}  "
                  f"cond {entry['condition']:.4e}  "
                  f"res {entry['residual']:.4e}  "
                  f"data {entry['data']:.4e}  "
                  f"({time.time() - t0:.0f}s)")

    wall = time.time() - t0
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    torch.save(
        {"config": MODEL_CONFIG, "state_dict": model.state_dict(),
         "n_grid": N_GRID},
        results_dir / "npo_checkpoint.pt",
    )
    with open(results_dir / "npo_training_history.json", "w") as f:
        json.dump(
            {
                "config": {
                    "model": MODEL_CONFIG, "n_grid": N_GRID,
                    "num_rhs": NUM_RHS, "first_seed": FIRST_SEED,
                    "record_iters": list(RECORD_ITERS),
                    "num_samples": int(n_samples), "epochs": EPOCHS,
                    "batch": BATCH, "lr": LR, "lr_min": LR_MIN,
                    "warmup_epochs": WARMUP_EPOCHS,
                    "n_params": int(n_params),
                },
                "wall_time_seconds": wall,
                "history": history,
            },
            f,
            indent=2,
        )

    final = history[-1]
    print(f"final losses: total {final['total']:.4e}  "
          f"condition {final['condition']:.4e}  "
          f"residual {final['residual']:.4e}  data {final['data']:.4e}")
    print(f"training wall time: {wall:.1f}s")
    print("saved results/npo_checkpoint.pt and results/npo_training_history.json")


if __name__ == "__main__":
    main()
