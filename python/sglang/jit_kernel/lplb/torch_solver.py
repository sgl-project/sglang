"""
IPM LP Solver — Pure PyTorch reference implementation.

Solves: min c^T x  subject to  Ax = b, x >= 0
using a barrier (interior point) method with 5 iterations.

This mirrors the CUDA kernel in csrc/minilp.cu but runs entirely
in PyTorch for testing, prototyping, and as a fallback when the
CUDA kernel is not available.
"""

import torch


def solve_ipm(
    A: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    num_iters: int = 5,
) -> torch.Tensor:
    """
    Barrier-method Interior Point solver for standard-form LP.

    Args:
        A: Constraint matrix, shape (NC, NV), float32, on CUDA.
        b: RHS vector, shape (NC,), float32, on CUDA.
        c: Objective coefficients, shape (NV,), float32, on CUDA.
        num_iters: Number of barrier iterations (default 5).

    Returns:
        x: Solution vector, shape (NV,), float32.
            x[0..n-1] are token distribution ratios for redundant expert copies.
            Returns 0.5 for all entries if the solver fails to converge.
    """
    nc, nv = A.shape
    assert b.shape == (nc,), f"b shape mismatch: {b.shape} vs ({nc},)"
    assert c.shape == (nv,), f"c shape mismatch: {c.shape} vs ({nv},)"

    x = torch.ones(nv, device=A.device, dtype=torch.float32)

    for _ in range(num_iters):
        # Scaled constraint matrix: ax2[i][j] = A[i][j] * x[j]^2
        x_sq = x * x
        ax2 = A * x_sq.unsqueeze(0)  # (NC, NV)

        # KKT system: ax2a is (NC, NC) symmetric positive definite
        ax2a = ax2 @ A.t()  # (NC, NC)

        # KKT RHS
        ax2c = ax2 @ c  # (NC,)

        # Cholesky solve: ax2a @ delta = ax2c
        # Add small regularization to avoid singular matrix
        ax2a.diagonal().add_(1e-6)
        try:
            delta = torch.linalg.solve(ax2a, ax2c)  # (NC,)
        except torch.linalg.LinAlgError:
            return torch.full((nv,), 0.5, device=A.device, dtype=torch.float32)

        # Direction: r = delta^T @ A, then d = x * (c - r)
        r = A.t() @ delta  # (NV,)
        d = x * (c - r)  # (NV,)

        # Line search
        d_max = d.max()
        alpha = 0.999 / d_max

        # Update
        x = x * (1.0 - alpha * d)

    # Convergence check
    residual = A @ x - b
    max_residual = residual.abs().max()
    avail = (d_max < 0.1) and (0 <= x[-1] < 1e-4) and (max_residual < 0.05)

    if not avail:
        return torch.full((nv,), 0.5, device=A.device, dtype=torch.float32)

    return x
