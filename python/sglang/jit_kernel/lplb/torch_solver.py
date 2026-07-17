"""IPM LP Solver entry point — dispatches to the fused JIT CUDA kernel.

Solves: min c^T x  subject to  Ax = b, x >= 0
using a barrier (interior point) method with 5 iterations.

The fused kernel lives in ``cuda_solver`` (CUDA C++ via ``load_jit``,
backed by cuBLASDx GEMMs + cuSolverDx POSV). This
module is the public-facing import surface for callers (``LPLBSolver``)
and resolves/caches the backend on first use.

LPLB requires Hopper-class hardware and a Math-DX install with cuBLASDx
headers plus the cuSolverDx device-link artifact. If either is missing,
``warmup`` and ``solve_ipm`` raise — there is no
silent fallback.
"""

import logging

import torch

logger = logging.getLogger(__name__)


# Backend dispatch state (resolved on first call, cached afterwards)
_BACKEND_CHECKED = False
_FUSED_AVAILABLE = False
_FUSED_SOLVE_IPM = None  # type: ignore[assignment]
_FUSED_WARMUP = None  # type: ignore[assignment]
_FUSED_ASSERT_FITS = None  # type: ignore[assignment]


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
        from sglang.jit_kernel.lplb.cuda_solver import solve_ipm as fused_solve_ipm
        from sglang.jit_kernel.lplb.cuda_solver import warmup as fused_warmup
        from sglang.jit_kernel.lplb.shmem_budget import assert_fits
    except ImportError as e:
        logger.info(
            f"LPLB fused solver disabled: {e}. "
            "Install a full Math-DX distribution with cuBLASDx/cuSolverDx "
            "or set MATHDX_HOME to an extracted archive."
        )
        return

    _FUSED_SOLVE_IPM = fused_solve_ipm
    _FUSED_WARMUP = fused_warmup
    _FUSED_ASSERT_FITS = assert_fits
    _FUSED_AVAILABLE = True
    logger.info("LPLB fused solver enabled (CUDA C++ via load_jit, cuBLASDx)")


def _unavailable_reason() -> str:
    if not torch.cuda.is_available():
        return "CUDA is not available"
    cap = torch.cuda.get_device_capability()
    if cap[0] < 9:
        return f"GPU SM {cap[0]}.{cap[1]} < 9.0 (requires Hopper or newer)"
    return (
        "Math-DX cuBLASDx/cuSolverDx headers or cuSolverDx device-link "
        "artifact not found — set MATHDX_HOME to a full Math-DX archive"
    )


def warmup(nc: int, nv: int, num_iters: int = 5, device: str = "cuda") -> None:
    """Pre-JIT-compile the fused kernel for a given (NC, NV) shape.

    Call once per unique shape at solver construction time to hide the
    20-40s JIT compilation cost. Raises if the fused backend is
    unavailable, the shape exceeds the shmem budget, or the kernel
    fails to compile/launch.
    """
    _init_fused_backend()
    if not _FUSED_AVAILABLE:
        raise RuntimeError(f"LPLB fused solver unavailable: {_unavailable_reason()}")
    _FUSED_ASSERT_FITS(nc, nv, gpu="h100")
    _FUSED_WARMUP(nc, nv, num_iters=num_iters, device=device)


def solve_ipm(
    A: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    num_iters: int = 5,
) -> torch.Tensor:
    """Barrier-method Interior Point solver for standard-form LP.

    Dispatches to the JIT-compiled CUDA C++ kernel (Hopper+ GPU with
    Math-DX cuBLASDx/cuSolverDx headers and cuSolverDx device-link
    artifact, reachable via ``MATHDX_HOME``). Raises if the fused backend is
    unavailable or the inputs aren't on CUDA in float32.

    Args:
        A: Constraint matrix, shape (NC, NV), float32, on CUDA.
        b: RHS vector, shape (NC,), float32, on CUDA.
        c: Objective coefficients, shape (NV,), float32, on CUDA.
        num_iters: Number of barrier iterations (default 5).

    Returns:
        x: Solution vector, shape (NV,), float32. The kernel writes 0.5
        for every entry on non-convergence.
    """
    nc, nv = A.shape
    assert b.shape == (nc,), f"b shape mismatch: {b.shape} vs ({nc},)"
    assert c.shape == (nv,), f"c shape mismatch: {c.shape} vs ({nv},)"

    _init_fused_backend()
    if not _FUSED_AVAILABLE:
        raise RuntimeError(f"LPLB fused solver unavailable: {_unavailable_reason()}")
    if not A.is_cuda:
        raise RuntimeError(
            f"LPLB fused solver requires CUDA tensors; got A on {A.device}."
        )
    if A.dtype != torch.float32:
        raise RuntimeError(
            f"LPLB fused solver requires float32; got A.dtype={A.dtype}."
        )
    return _FUSED_SOLVE_IPM(A, b, c, num_iters=num_iters)


def solve_ipm_torch_reference(
    A: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    num_iters: int = 5,
) -> torch.Tensor:
    """Pure-torch reference for the fused IPM kernel — testing only.

    Mirrors the barrier-method iteration in ``csrc/lplb/ipm.cuh``
    step-for-step so the two can be compared numerically:

      x <- 1
      for _ in range(num_iters):
          ax2  = A * x^2            # (NC, NV)
          ax2a = ax2 @ A^T          # (NC, NC) KKT matrix
          delta = solve(ax2a, ax2 @ c)
          r     = delta^T @ A       # (NV,)
          d     = x * (c - r)
          alpha = 0.999 / d_max  (or 1.0 if d_max <= 1e-9)
          x    *= 1 - alpha * d
      write 0.5 everywhere on non-convergence.

    NOT bit-equivalent to the kernel: the kernel factors/solves the KKT
    system with cuSolverDx POSV while this uses ``torch.linalg.solve`` (LU).
    The two agree to a small tolerance
    (the numerical difference being the whole point of the comparison
    test). This function is never on the production path — the fused
    kernel is the only LP solver at runtime.
    """
    nc, nv = A.shape
    assert b.shape == (nc,), f"b shape mismatch: {b.shape} vs ({nc},)"
    assert c.shape == (nv,), f"c shape mismatch: {c.shape} vs ({nv},)"

    x = torch.ones(nv, device=A.device, dtype=torch.float32)
    d_max = torch.tensor(0.0, device=A.device, dtype=torch.float32)
    for _ in range(num_iters):
        ax2 = A * (x * x).unsqueeze(0)  # (NC, NV)
        ax2a = ax2 @ A.t()  # (NC, NC)
        ax2c = ax2 @ c  # (NC,)
        # Match the kernel's 1e-12 pivot clamp via a tiny diagonal jitter so
        # a (near-)singular KKT system stays solvable instead of raising.
        ax2a = ax2a + 1e-12 * torch.eye(nc, device=A.device, dtype=torch.float32)
        delta = torch.linalg.solve(ax2a, ax2c)  # (NC,)
        r = A.t() @ delta  # (NV,)
        d = x * (c - r)  # (NV,)
        d_max = d.max()
        alpha = 0.999 / d_max if d_max > 1e-9 else torch.tensor(1.0, device=A.device)
        x = x * (1.0 - alpha * d)

    max_residual = (A @ x - b).abs().max()
    converged = (d_max < 0.1) and (0 <= x[-1] < 1e-4) and (max_residual < 0.05)
    if not converged:
        return torch.full((nv,), 0.5, device=A.device, dtype=torch.float32)
    return x
