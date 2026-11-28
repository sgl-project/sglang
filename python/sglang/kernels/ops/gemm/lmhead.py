"""lm_head shard GEMM for few rows of x on SM100: ``out[N, W] = x[N, K] @ w[W, K].T``, bf16 in, fp32 accumulation,
bf16 out, for weights far larger than L2 (the DFlash2 draft lm_head shard, W = V / TP + 256).

- ``lmhead_mma`` (N <= 16): mma.sync, FlashInfer's warp-split-K structure (csrc/gemm/lmhead_mma.cuh).
- ``lmhead_umma`` (N <= 128): UMMA (tcgen05.mma), one whole weight tile per CTA, tile height chosen so the tiles fill
  one wave (cuBLAS's structure; csrc/gemm/lmhead_umma.cuh).
- ``lmhead_gemm``: the dispatch between them, cuBLAS above N = 128.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


# mma.sync kernel: BLOCK_M = 16 weight rows x BLOCK_N = 8 / 16 rows of x per CTA, BLOCK_K = 256 per stage.
_MMA_BLOCK_M, _MMA_BLOCK_K = 16, 256
_CTA_RESERVED_SMEM = 1024  # shared memory the driver reserves per CTA on SM100


def _get_mma_block_n(n: int) -> int:
    assert 0 <= n <= 16, n
    return 8 if n <= 8 else 16


def _get_mma_ctas_per_sm(block_n: int, num_stages: int) -> int:
    # sizeof(MMASmem): the stage ring, the epilogue partials, two mbarriers per stage.
    smem = (
        num_stages * (_MMA_BLOCK_M + block_n) * _MMA_BLOCK_K * 2
        + 4 * block_n * (_MMA_BLOCK_M + 4) * 4
        + 16 * num_stages
    )
    threads = (2 + 4 + block_n // 8) * 32
    arch = current_arch()
    return min(
        arch.smem_per_sm // (smem + _CTA_RESERVED_SMEM),
        arch.max_threads_per_sm // threads,
    )


@cache_once
def _get_mma_num_stages(n: int, w_rows: int, hidden: int) -> int:
    """The deepest ring (>= 2 stages) that keeps the whole grid resident in one wave; when no ring fits, the
    deepest ring at 3 CTAs per SM (several waves, balanced by the block scheduler). A nearly empty last wave costs
    about a CTA's whole duration (TP16 N = 16: 3 stages, 4 CTAs per SM, 1.08 waves: 29.9 us vs 24.9)."""
    block_n = _get_mma_block_n(n)
    ctas = w_rows // _MMA_BLOCK_M
    num_sms = current_arch().num_sms
    max_stages = hidden // _MMA_BLOCK_K
    one_wave = [
        s
        for s in range(2, max_stages + 1)
        if _get_mma_ctas_per_sm(block_n, s) * num_sms >= ctas
    ]
    if one_wave:
        return max(one_wave)
    return max(
        s for s in range(2, max_stages + 1) if _get_mma_ctas_per_sm(block_n, s) >= 3
    )


@cache_once
def _jit_mma_module(block_n: int, num_stages: int, hidden: int) -> Module:
    args = make_cpp_args(block_n, num_stages, hidden, is_arch_support_pdl())
    return load_jit(
        "lmhead_mma",
        *args,
        cuda_files=["gemm/lmhead_mma.cuh"],
        cuda_wrappers=[("run", f"sglang::LMHeadMMAKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def lmhead_mma(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    num_stages: Optional[int] = None,
) -> torch.Tensor:
    """N <= 16."""
    n, hidden = x.shape
    if num_stages is None:
        num_stages = _get_mma_num_stages(n, w.shape[0], hidden)
    if out is None:
        out = x.new_empty((n, w.shape[0]))
    _jit_mma_module(_get_mma_block_n(n), num_stages, hidden).run(x, w, out)
    return out


# UMMA kernel: BLOCK_M = 64 / 128 rows of x (UMMA M) x one whole tile of BLOCK_N weight rows (UMMA N) per CTA.
_UMMA_BLOCK_K = 64


def _get_umma_block_m(n: int) -> int:
    assert 0 <= n <= 128, n
    return 64 if n <= 64 else 128


@cache_once
def _get_umma_block_n(w_rows: int, block_m: int) -> int:
    """The smallest legal UMMA N (a multiple of 8 for M = 64, of 16 otherwise) whose tile count fits one wave."""
    num_sms = current_arch().num_sms
    align = 8 if block_m == 64 else 16
    return min(256, -(-(-(-w_rows // num_sms)) // align) * align)


@cache_once
def _get_umma_num_stages(block_m: int, block_n: int) -> int:
    stage = (block_m + block_n) * _UMMA_BLOCK_K * 2
    # 256 B covers the mbarriers and the TMEM address slot after the ring.
    return min(16, (current_arch().smem_per_block_optin - 256) // stage)


@cache_once
def _jit_umma_module(block_m: int, block_n: int, num_stages: int) -> Module:
    args = make_cpp_args(block_m, block_n, num_stages, is_arch_support_pdl())
    return load_jit(
        "lmhead_umma",
        *args,
        cuda_files=["gemm/lmhead_umma.cuh"],
        cuda_wrappers=[("run", f"sglang::LMHeadUMMAKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def lmhead_umma(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    num_stages: Optional[int] = None,
) -> torch.Tensor:
    """N <= 128."""
    n = x.shape[0]
    if block_m is None:
        block_m = _get_umma_block_m(n)
    if block_n is None:
        block_n = _get_umma_block_n(w.shape[0], block_m)
    if num_stages is None:
        num_stages = _get_umma_num_stages(block_m, block_n)
    if out is None:
        out = x.new_empty((n, w.shape[0]))
    _jit_umma_module(block_m, block_n, num_stages).run(x, w, out)
    return out


def lmhead_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``x[N, K] @ w[W, K].T``: mma.sync for N <= 16, UMMA for N <= 128 (measured on GB300 at TP8 / TP16 shards);
    cuBLAS above N = 128 or for a shard too tall for one wave of 256-row tiles (TP4). Our kernels launch with PDL
    where the arch supports it; cuBLAS decides for itself (its kernels carry the PDL launch attribute).
    """
    n, w_rows = x.shape[0], w.shape[0]
    if n <= 16:
        return lmhead_mma(x, w, out=out)
    if n <= 128 and w_rows <= 256 * current_arch().num_sms:
        return lmhead_umma(x, w, out=out)
    return torch.matmul(x, w.T, out=out)
