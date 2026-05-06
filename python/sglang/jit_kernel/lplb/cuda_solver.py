"""JIT-compiled CUDA Interior Point Method LP solver.

Replaces the Numba/nvmath-python implementation in ``cublasdx_solver.py``.
The kernel is a single-block fused IPM defined in
``csrc/lplb/ipm.cuh`` and compiled per ``(NC, NV, BLOCK_DIM, SM_VER,
NUM_ITERS)`` tuple via sglang's ``tvm-ffi`` ``load_jit``.

Per-call CPU overhead is dominated by the pybind11 dispatch + four
``data_ptr()`` calls (~5–10 µs total), versus ~500–700 µs for the prior
Numba path (numba dispatcher chain + ``as_cuda_array`` per tensor).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    get_jit_cuda_arch,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

DEFAULT_BLOCK_DIM = 256
# Per-element kernels (post-LP dispatch) saturate easily — also 256.
DISPATCH_BLOCK_DIM = 256
DEFAULT_NUM_ITERS = 5


def _sm_ver() -> int:
    arch = get_jit_cuda_arch()
    return arch.major * 100 + arch.minor * 10


@cache_once
def _ipm_module(
    nc: int, nv: int, block_dim: int, num_iters: int, sm_ver: int
) -> "Module":
    """JIT-compile the IPM kernel for one shape. Cached for the process lifetime."""
    args = make_cpp_args(nc, nv, block_dim, sm_ver, num_iters)
    # The kernel uses cuBLASDx (header-only) for the GEMMs and a hand-written
    # block-level Cholesky for the POSV. No -rdc=true / static-lib linkage
    # required, so sglang's tvm-ffi load_jit handles the build with the
    # default flags.
    return load_jit(
        "lplb_ipm",
        *args,
        cuda_files=["lplb/ipm.cuh"],
        cuda_wrappers=[("ipm_solve", f"ipm_solve<{args}>")],
        extra_dependencies=["mathdx"],
    )


def warmup(
    nc: int,
    nv: int,
    num_iters: int = DEFAULT_NUM_ITERS,
    device: str = "cuda",
) -> None:
    """JIT-compile the kernel for ``(nc, nv)`` so the first real solve isn't
    paying the compile cost. Raises on compile or launch failure.
    """
    module = _ipm_module(nc, nv, DEFAULT_BLOCK_DIM, num_iters, _sm_ver())
    # Trigger any first-call lazy initialization.
    A = torch.zeros(nc, nv, dtype=torch.float32, device=device)
    b = torch.zeros(nc, dtype=torch.float32, device=device)
    c = torch.zeros(nv, dtype=torch.float32, device=device)
    result = torch.empty(nv, dtype=torch.float32, device=device)
    module.ipm_solve(A, b, c, result)
    logger.info(f"LPLB CUDA IPM solver: warmed up for (NC={nc}, NV={nv})")


def solve_ipm(
    A: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    num_iters: int = DEFAULT_NUM_ITERS,
    result: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Run the fused single-SM IPM kernel.

    cuBLASDx GEMMs + hand-written block Cholesky, dispatched per the
    module docstring.

    Args:
        A: Constraint matrix, shape ``(NC, NV)``, float32, on CUDA.
        b: RHS vector, shape ``(NC,)``, float32, on CUDA.
        c: Objective coefficients, shape ``(NV,)``, float32, on CUDA.
        num_iters: Number of barrier iterations (default 5).
        result: Optional pre-allocated ``(NV,)`` float32 CUDA buffer to write
            into. When omitted the kernel allocates a fresh result tensor
            (~20 µs of CPU overhead). Passing in a long-lived buffer skips
            that alloc on every solve.

    Returns:
        x: Solution vector, shape ``(NV,)``, float32. The kernel writes 0.5
        for every entry on non-convergence (matches the prior Numba behavior).
    """
    assert A.is_cuda and b.is_cuda and c.is_cuda
    assert A.dtype == torch.float32
    nc, nv = A.shape
    assert b.shape == (nc,), f"b shape mismatch: {b.shape} vs ({nc},)"
    assert c.shape == (nv,), f"c shape mismatch: {c.shape} vs ({nv},)"

    module = _ipm_module(nc, nv, DEFAULT_BLOCK_DIM, num_iters, _sm_ver())
    if result is None:
        result = torch.empty(nv, dtype=torch.float32, device=A.device)
    module.ipm_solve(A, b, c, result)
    return result


@cache_once
def _prep_module(
    nc: int,
    nv: int,
    num_single: int,
    num_red_log: int,
    num_gpus: int,
    block_dim: int,
) -> "Module":
    args = make_cpp_args(nc, nv, num_single, num_red_log, num_gpus, block_dim)
    return load_jit(
        "lplb_lp_prep",
        *args,
        cuda_files=["lplb/lp_prep.cuh"],
        cuda_wrappers=[("lp_prep", f"lp_prep<{args}>")],
    )


def prep_lp_inputs(
    A_full: torch.Tensor,
    b: torch.Tensor,
    t1: torch.Tensor,
    global_counts: torch.Tensor,
    log_single: torch.Tensor,
    log_replicated: torch.Tensor,
    B1: torch.Tensor,
    A_base_row_sum: torch.Tensor,
) -> None:
    """Replace the 8 torch ops that built the IPM inputs with one CUDA kernel.

    Writes into the caller-provided ``A_full`` (last column only), ``b``,
    and ``t1`` buffers. ``A_full``'s first ``NV-1`` columns must already
    hold ``A_base.copy_()`` from solver init — this kernel does not touch
    them.
    """
    nc, nv = A_full.shape
    num_single = log_single.shape[0]
    num_red_log = log_replicated.shape[0]
    num_gpus, _ns = B1.shape
    module = _prep_module(nc, nv, num_single, num_red_log, num_gpus, DEFAULT_BLOCK_DIM)
    module.lp_prep(
        A_full, b, t1, global_counts, log_single, log_replicated, B1, A_base_row_sum
    )


@cache_once
def _post_module(
    num_logical: int,
    max_copies: int,
    num_single: int,
    num_red_phy: int,
    block_dim: int,
) -> "Module":
    args = make_cpp_args(num_logical, max_copies, num_single, num_red_phy, block_dim)
    return load_jit(
        "lplb_lp_post",
        *args,
        cuda_files=["lplb/lp_post.cuh"],
        cuda_wrappers=[("lp_post", f"lp_post<{args}>")],
    )


def extract_log2phy_prob(
    log2phy_prob: torch.Tensor,
    x: torch.Tensor,
    t1: torch.Tensor,
    phy_single: torch.Tensor,
    phy_replicated: torch.Tensor,
    log2phy: torch.Tensor,
) -> None:
    """Replace the 5 torch ops that turned the IPM output into log2phy_prob
    with one CUDA kernel. Writes into the caller-provided ``log2phy_prob``
    buffer of shape ``(num_logical, max_copies)``.
    """
    num_logical, max_copies = log2phy_prob.shape
    num_single = phy_single.shape[0]
    num_red_phy = phy_replicated.shape[0]
    module = _post_module(
        num_logical, max_copies, num_single, num_red_phy, DEFAULT_BLOCK_DIM
    )
    module.lp_post(log2phy_prob, x, t1, phy_single, phy_replicated, log2phy)


@cache_once
def _dispatch_module(max_copies: int, block_dim: int) -> "Module":
    args = make_cpp_args(max_copies, block_dim)
    return load_jit(
        "lplb_dispatch_probability",
        *args,
        cuda_files=["lplb/dispatch_probability.cuh"],
        cuda_wrappers=[("dispatch_probability", f"dispatch_probability<{args}>")],
    )


def dispatch_probability(
    topk_ids: torch.Tensor,
    log2phy_prob: torch.Tensor,
    log2phy_map: torch.Tensor,
) -> torch.Tensor:
    """Replace the 7 torch ops in `_topk_ids_logical_to_physical_probability`
    with a single per-token-per-slot CUDA kernel.

    Args:
        topk_ids: (num_tokens, topk) int32, on CUDA. Logical expert ids from
            the router.
        log2phy_prob: (num_logical, max_copies) float32. LP solver output.
        log2phy_map: (num_logical, max_copies) int32. -1 entries are
            unused replicas; treated as 0-weight in the multinomial.

    Returns:
        Physical topk ids tensor with the same shape as ``topk_ids``.
    """
    original_shape = topk_ids.shape
    flat_ids = topk_ids.reshape(-1).contiguous().to(torch.int32)
    n = flat_ids.shape[0]
    num_logical, max_copies = log2phy_prob.shape
    assert log2phy_map.shape == (num_logical, max_copies)
    map32 = log2phy_map.contiguous().to(torch.int32)

    out = torch.empty(n, dtype=torch.int32, device=topk_ids.device)
    random_vals = torch.rand(n, dtype=torch.float32, device=topk_ids.device)
    module = _dispatch_module(max_copies, DISPATCH_BLOCK_DIM)
    module.dispatch_probability(out, flat_ids, log2phy_prob, map32, random_vals)
    return out.view(original_shape).to(topk_ids.dtype)
