"""Tiny bf16 GEMM for skinny ``x[m, k] @ w[n, k].T`` with a handful of rows.

Two kernels behind one entry point, picked by whichever of N / K is the tiny
dimension: the N variant spreads K across a block and walks N, the K variant
reduces K inside a warp and spreads N across the grid. See
:func:`tiny_gemm_bf16`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# NOTE: CUDA constraints
_WARP_THREADS: int = 32
_MAX_BLOCK_THREADS: int = 1024
# Mirrors GEMMTraitN in tiny_gemm.cuh: one thread owns kBytes = 32 bytes of K,
# so the block is K / 16 threads on every arch (k_unroll absorbs the vector-width
# difference between Hopper and Blackwell).
_N_GEMM_ELEMS_PER_THREAD: int = 16
# Mirrors GEMMTraitK: the K variant always uses 16-byte vectors.
_K_GEMM_ELEMS_PER_THREAD: int = 8


def _prefer_k_variant(n: int, k: int) -> bool:
    """Find the **tiny** dimension"""
    return k <= n


@cache_once
def _jit_tiny_gemm_module(
    n: int, k: int, max_m: int, n_split: int, out_dtype: torch.dtype
) -> Module:
    use_k_variant = _prefer_k_variant(n, k)
    args = make_cpp_args(n, k, max_m, n_split, out_dtype, is_arch_support_pdl())
    name = "tiny_k_gemm" if use_k_variant else "tiny_n_gemm"
    kernel = "TinyKGemmKernel" if use_k_variant else "TinyNGemmKernel"
    return load_jit(
        name,
        *args,
        cuda_files=["gemm/tiny_gemm.cuh"],
        cuda_wrappers=[("run", f"{kernel}<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _get_num_sm() -> int:
    device = torch.cuda.current_device()
    return torch.cuda.get_device_properties(device).multi_processor_count


def _n_gemm_block_threads(k: int) -> int:
    return k // _N_GEMM_ELEMS_PER_THREAD


def _k_gemm_reduction_lanes(k: int) -> int:
    return k // _K_GEMM_ELEMS_PER_THREAD


def _supports_n_variant(k: int, max_m: int) -> bool:
    block = _n_gemm_block_threads(k)
    return (
        k % _N_GEMM_ELEMS_PER_THREAD == 0
        and block % _WARP_THREADS == 0
        and block <= _MAX_BLOCK_THREADS
        and block >= max_m
    )


def _supports_k_variant(n: int, k: int) -> bool:
    lanes = _k_gemm_reduction_lanes(k)
    return (
        k % _K_GEMM_ELEMS_PER_THREAD == 0
        and k > _K_GEMM_ELEMS_PER_THREAD
        and _WARP_THREADS % lanes == 0
        and n % (_WARP_THREADS // lanes) == 0
    )


@cache_once
def _default_n_gemm_n_split(n: int, k: int, max_m: int) -> int:
    split_cap = _n_gemm_block_threads(k) // max_m
    candidates = [d for d in range(1, min(n, split_cap) + 1) if n % d == 0]
    if not candidates:
        raise RuntimeError(
            f"tiny_gemm: no valid n_split for N={n}, K={k}, max_m={max_m};"
            " try lower max_m or align the N dimension"
        )
    # try to fit into 1 wave
    sm_count = _get_num_sm()
    for split in candidates:
        if n // split <= sm_count:
            return split
    return candidates[-1]


@cache_once
def _default_k_gemm_n_split(n: int, k: int) -> int:
    lanes = _k_gemm_reduction_lanes(k)
    candidates = [
        d
        for d in range(1, min(n, _MAX_BLOCK_THREADS // lanes) + 1)
        if n % d == 0
        and d * lanes % _WARP_THREADS == 0
        and d * lanes <= _MAX_BLOCK_THREADS
    ]
    # try to fit into 1 wave
    if not candidates:
        raise RuntimeError(f"tiny_gemm: no valid n_split for N={n}, K={k}")
    sm_count = _get_num_sm()
    for d in candidates:
        if n // d <= sm_count:
            return d
    return candidates[-1]


@register_custom_op(op_name="tiny_gemm_bf16", mutates_args=["out"])
def _tiny_gemm_custom_op(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    max_m: int,
    n_split: int,
) -> None:
    n, k = w.shape
    module = _jit_tiny_gemm_module(n, k, max_m, n_split, out.dtype)
    module.run(x, w, out)


@cache_once
def can_use_tiny_gemm(n: int, k: int, max_m: int = 16) -> bool:
    """Whether :func:`tiny_gemm_bf16` can serve ``[m, k] @ [n, k].T`` for
    ``m <= max_m``. Callers fall back to a general GEMM when this is False."""
    if _prefer_k_variant(n, k):
        return _supports_k_variant(n, k)
    else:
        return _supports_n_variant(k, max_m)


@debug_kernel_api
def tiny_gemm_bf16(
    x: torch.Tensor,
    w: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    out_dtype: Optional[torch.dtype] = None,
    n_split: Optional[int] = None,
    max_m: int = 16,
) -> torch.Tensor:
    """
    Equal to `torch.nn.functional.linear(x, w)`.
    Call :func:`can_use_tiny_gemm` first: shapes outside the supported set raise
    rather than falling back.

    :param x: Shape [m, k], must be bf16_t, `m <= max_m`
    :param w: Shape [n, k], must be bf16_t
    """
    n, k = w.shape
    if out is None:
        out_dtype = torch.bfloat16 if out_dtype is None else out_dtype
        out = torch.empty((x.shape[0], n), dtype=out_dtype, device=x.device)
    else:
        assert out_dtype is None or out_dtype == out.dtype
    if n_split is None:
        n_split = (
            _default_k_gemm_n_split(n, k)
            if _prefer_k_variant(n, k)
            else _default_n_gemm_n_split(n, k, max_m)
        )
    _tiny_gemm_custom_op(x, w, out, max_m, n_split)
    return out
