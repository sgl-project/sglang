"""Skinny bf16 GEMV on ROCm for decode row counts: [M, K] x [N, K] as fp32 split-K
partials whose fixed-order sum is batch invariant (the DeepSeek-V4 router and indexer head
weights)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_gfx95_supported, is_hip

# the GEMV walks 16-row tiles inside one program, so every M runs the same tile (batch invariance)
ROCM_ROUTER_MAX_TOKENS = 64

_BLOCK_M = 16
_BLOCK_N = 16
_BLOCK_K = 512
_MAX_SPLIT_K = 32


def rocm_gemv_split_k_max_tokens(*, n: int, k: int, weight_dtype: torch.dtype) -> int:
    """Rows up to which rocm_router_gemv_split_k serves an [M, k] @ [n, k].T bf16
    GEMV (one 16-wide N tile per 512 of K, 16-row tiles), -1 when the device or the shape
    rules it out."""
    if not (is_hip() and is_gfx95_supported()):
        return -1
    if weight_dtype != torch.bfloat16:
        return -1
    if n <= 0 or n % _BLOCK_N != 0:
        return -1
    if k % _BLOCK_K != 0 or not 0 < k // _BLOCK_K <= _MAX_SPLIT_K:
        return -1
    return ROCM_ROUTER_MAX_TOKENS


@triton.jit
def _router_gemv_split_k_kernel(
    x_ptr,
    w_ptr,
    part_ptr,
    M,
    stride_xm,
    stride_wn,
    stride_ps,
    stride_pm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M_TILES: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    w = tl.load(w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None])
    # the weight tile is read once; every row tile is the same BLOCK_M x BLOCK_K dot
    for t in tl.static_range(M_TILES):
        offs_m = t * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = offs_m < M
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        acc = tl.dot(x, w)
        tl.store(
            part_ptr
            + pid_k * stride_ps
            + offs_m[:, None] * stride_pm
            + offs_n[None, :],
            acc,
            mask=row_mask[:, None],
        )


def rocm_router_gemv_split_k(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x[M, K] @ w[N, K].T as fp32 split-K partials [K // 512, M, N], summed over dim 0 in
    order by rocm_router_reduce_partials or the fused gate."""
    M, K = x.shape
    N, K_w = w.shape
    assert K == K_w and K % _BLOCK_K == 0 and N % _BLOCK_N == 0
    assert 0 < M <= ROCM_ROUTER_MAX_TOKENS, (
        f"{M} rows: the split-K GEMV serves at most {ROCM_ROUTER_MAX_TOKENS}"
    )
    assert x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
    assert x.stride(1) == 1 and w.stride(1) == 1
    split_k = K // _BLOCK_K
    partials = torch.empty((split_k, M, N), dtype=torch.float32, device=x.device)
    _router_gemv_split_k_kernel[(N // _BLOCK_N, split_k)](
        x,
        w,
        partials,
        M,
        x.stride(0),
        w.stride(0),
        partials.stride(0),
        partials.stride(1),
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        M_TILES=triton.cdiv(M, _BLOCK_M),
        num_warps=4,
    )
    return partials


@triton.jit
def _reduce_partials_kernel(
    part_ptr,
    out_ptr,
    M,
    stride_ps,
    stride_pm,
    stride_om,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs // N
    n = offs % N
    mask = m < M
    acc = tl.load(part_ptr + m * stride_pm + n, mask=mask, other=0.0)
    for s in tl.static_range(1, SPLIT_K):
        acc += tl.load(
            part_ptr + s * stride_ps + m * stride_pm + n, mask=mask, other=0.0
        )
    tl.store(out_ptr + m * stride_om + n, acc, mask=mask)


def rocm_router_reduce_partials(partials: torch.Tensor, out: torch.Tensor) -> None:
    """out[M, N] = partials[0] + partials[1] + ... in that order, in fp32:
    the same sum the fused gate computes."""
    split_k, M, N = partials.shape
    assert out.shape == (M, N) and out.dtype == torch.float32 and out.stride(1) == 1
    block = 1024
    _reduce_partials_kernel[(triton.cdiv(M * N, block),)](
        partials,
        out,
        M,
        partials.stride(0),
        partials.stride(1),
        out.stride(0),
        N=N,
        SPLIT_K=split_k,
        BLOCK=block,
        num_warps=4,
    )
