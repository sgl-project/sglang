"""
IPM LP Solver — dispatches to the fused cuBLASDx+cuSolverDx kernel when
available, falls back to a pure-PyTorch reference implementation otherwise.

Solves: min c^T x  subject to  Ax = b, x >= 0
using a barrier (interior point) method with 5 iterations.

The torch fallback mirrors the CUDA kernel in csrc/minilp.cu. Both backends
expose the same 4-argument signature so users (e.g. LPLBSolver._solve_torch)
get the fused kernel automatically without code changes.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# Backend dispatch state (checked on first call, cached afterwards)
_BACKEND_CHECKED = False
_FUSED_AVAILABLE = False
_FUSED_SOLVE_IPM = None  # type: ignore[assignment]
_FUSED_WARMUP = None  # type: ignore[assignment]
_FUSED_ASSERT_FITS = None  # type: ignore[assignment]
_FUSED_REJECTED_SHAPES: set = set()  # (nc, nv) shapes that don't fit / failed

_REQUIRE_FUSED_CACHED: Optional[bool] = None


def _require_fused() -> bool:
    """Whether --lplb-require-fused is set; read once and cached at first call."""
    global _REQUIRE_FUSED_CACHED
    if _REQUIRE_FUSED_CACHED is None:
        from sglang.srt.server_args import get_global_server_args

        _REQUIRE_FUSED_CACHED = bool(get_global_server_args().lplb_require_fused)
    return _REQUIRE_FUSED_CACHED


def _init_fused_backend() -> None:
    """Resolve the fused backend once. Records WHY it's disabled when it is."""
    global _BACKEND_CHECKED, _FUSED_AVAILABLE
    global _FUSED_SOLVE_IPM, _FUSED_WARMUP, _FUSED_ASSERT_FITS

    if _BACKEND_CHECKED:
        return
    _BACKEND_CHECKED = True

    if not torch.cuda.is_available():
        logger.info("LPLB fused solver disabled: CUDA not available")
        return

    cap = torch.cuda.get_device_capability()
    if cap[0] < 9:
        logger.info(
            f"LPLB fused solver disabled: GPU SM {cap[0]}.{cap[1]} < 9.0 "
            "(requires Hopper or newer)"
        )
        return

    try:
        from sglang.jit_kernel.lplb.cublasdx_solver import (
            solve_ipm as fused_solve_ipm,
            warmup as fused_warmup,
        )
        from sglang.jit_kernel.lplb.shmem_budget import assert_fits
    except ImportError as e:
        logger.info(
            f"LPLB fused solver disabled: {e}. "
            "Install with: pip install 'nvmath-python[cu12-dx]==0.9.0' 'numba-cuda>=0.28.1'"
        )
        return

    _FUSED_SOLVE_IPM = fused_solve_ipm
    _FUSED_WARMUP = fused_warmup
    _FUSED_ASSERT_FITS = assert_fits
    _FUSED_AVAILABLE = True
    logger.info("LPLB fused solver enabled (cuBLASDx + cuSolverDx)")


def warmup(nc: int, nv: int, num_iters: int = 5, device: str = "cuda") -> None:
    """Pre-JIT-compile the fused kernel for a given (NC, NV) shape.

    Call this once per unique shape at solver construction time to hide the
    20-40s JIT compilation cost. Safe to call even when the fused backend
    is unavailable — becomes a no-op.
    """
    _init_fused_backend()
    if not _FUSED_AVAILABLE:
        return
    try:
        _FUSED_ASSERT_FITS(nc, nv, gpu="h100")
    except ValueError as e:
        logger.info(f"LPLB fused solver: shape (NC={nc}, NV={nv}) rejected: {e}")
        _FUSED_REJECTED_SHAPES.add((nc, nv))
        return
    try:
        _FUSED_WARMUP(nc, nv, num_iters=num_iters, device=device)
        logger.info(f"LPLB fused solver: warmed up for (NC={nc}, NV={nv})")
    except Exception as e:  # pragma: no cover
        logger.warning(
            f"LPLB fused solver: warmup failed for (NC={nc}, NV={nv}): {e}. "
            "Falling back to torch IPM for this shape."
        )
        _FUSED_REJECTED_SHAPES.add((nc, nv))


def solve_ipm(
    A: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    num_iters: int = 5,
) -> torch.Tensor:
    """
    Barrier-method Interior Point solver for standard-form LP.

    Dispatches to the fused cuBLASDx+cuSolverDx kernel when available
    (Hopper+ GPU with nvmath-python[cu12-dx] installed, shape fits in shmem).
    Falls back to the PyTorch reference implementation otherwise.

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

    _init_fused_backend()
    if (
        _FUSED_AVAILABLE
        and A.is_cuda
        and A.dtype == torch.float32
        and (nc, nv) not in _FUSED_REJECTED_SHAPES
    ):
        try:
            return _FUSED_SOLVE_IPM(A, b, c, num_iters=num_iters)
        except Exception as e:  # pragma: no cover
            if _require_fused():
                raise RuntimeError(
                    f"--lplb-require-fused is set but the fused solver raised "
                    f"at runtime for (NC={nc}, NV={nv}): {e!r}."
                ) from e
            logger.warning(
                f"LPLB fused solver: runtime failure for (NC={nc}, NV={nv}): {e}. "
                "Falling back to torch IPM for this shape."
            )
            _FUSED_REJECTED_SHAPES.add((nc, nv))

    if _require_fused():
        if not _FUSED_AVAILABLE:
            reason = (
                "fused backend unavailable (no CUDA, GPU SM < 9.0, or "
                "nvmath-python[cu12-dx] / numba-cuda not importable)"
            )
        elif not A.is_cuda:
            reason = f"input A is on {A.device}, not CUDA"
        elif A.dtype != torch.float32:
            reason = f"input A dtype is {A.dtype}, not torch.float32"
        elif (nc, nv) in _FUSED_REJECTED_SHAPES:
            reason = (
                f"shape (NC={nc}, NV={nv}) was rejected earlier "
                "(exceeded shmem budget or failed at runtime)"
            )
        else:
            reason = "dispatch predicate fell through unexpectedly"
        raise RuntimeError(
            f"--lplb-require-fused is set but the fused solver was skipped: "
            f"{reason}. (NC={nc}, NV={nv})."
        )

    return _solve_ipm_torch(A, b, c, num_iters=num_iters)


def _solve_ipm_torch(
    A: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    num_iters: int = 5,
) -> torch.Tensor:
    """Pure-PyTorch reference IPM — fallback when the fused backend is unavailable."""
    nc, nv = A.shape
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
