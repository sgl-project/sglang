"""Kimi-K3 MLA extend-with-prefix K/V assembly without the concat (ROCm).

Replaces index_select + strided fp8 casts + contiguous + cat around kv_b_proj:
one Triton gather dequantizes the latent rows into the GEMM input and writes
k_pe (broadcast to all heads) into its slot of K, and one copy places k_nope;
V stays a view of the GEMM output. Bit-identical (e4m3fn -> bf16 is exact).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _k3_prefix_gather_kernel(
    cache_ptr,  # [P, LORA + ROPE] (fp8 or bf16), row stride s_cache
    idx_ptr,  # [N] int32/int64
    kvc_ptr,  # [N, LORA] bf16 dense out (GEMM input)
    kpe_ptr,  # k_pe destination: element (n, h, d) at n*s_kpe_n + h*s_kpe_h + d
    N,
    s_cache,
    s_kpe_n,
    s_kpe_h,
    LORA: tl.constexpr,
    ROPE: tl.constexpr,
    H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    idx = tl.load(idx_ptr + rows, mask=rmask, other=0).to(tl.int64)
    src = cache_ptr + idx[:, None] * s_cache
    rows64 = rows.to(tl.int64)[:, None]

    offs_l = tl.arange(0, LORA)[None, :]
    lat = tl.load(src + offs_l, mask=rmask[:, None], other=0.0)
    tl.store(
        kvc_ptr + rows64 * LORA + offs_l,
        lat.to(kvc_ptr.dtype.element_ty),
        mask=rmask[:, None],
    )

    if H > 0:  # H == 0: latent only
        offs_r = tl.arange(0, ROPE)[None, :]
        pe = tl.load(src + LORA + offs_r, mask=rmask[:, None], other=0.0)
        pe = pe.to(kpe_ptr.dtype.element_ty)
        dst = kpe_ptr + rows64 * s_kpe_n + offs_r
        for h in tl.static_range(H):
            tl.store(dst + h * s_kpe_h, pe, mask=rmask[:, None])


@triton.jit
def _k3_prefix_knope_copy_kernel(
    kv_ptr,  # [R, *] rows of the GEMM output, row stride s_kv
    k_ptr,  # [R, *] rows of K, row stride s_k
    R,
    s_kv,
    s_k,
    DN: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = (pid * BLOCK_R + tl.arange(0, BLOCK_R)).to(tl.int64)
    rmask = rows < R
    offs = tl.arange(0, DN)[None, :]
    x = tl.load(kv_ptr + rows[:, None] * s_kv + offs, mask=rmask[:, None])
    tl.store(k_ptr + rows[:, None] * s_k + offs, x, mask=rmask[:, None])


def _gather_dequant(
    cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kvc: torch.Tensor,
    kpe_dst: Optional[torch.Tensor],
    kv_lora_rank: int,
    rope_dim: int,
) -> None:
    """kvc <- cache[kv_indices, :lora]; kpe_dst[:, h] <- cache[kv_indices, lora:]."""
    n = kv_indices.shape[0]
    if n == 0:
        return
    cache2d = cache.view(cache.shape[0], -1)
    assert cache2d.shape[1] == kv_lora_rank + rope_dim and cache2d.stride(1) == 1
    if kpe_dst is None:
        kpe_dst = kvc.new_empty((0, 0, 1))
    assert kpe_dst.dim() == 3 and kpe_dst.stride(2) == 1
    block_n = 8
    _k3_prefix_gather_kernel[(triton.cdiv(n, block_n),)](
        cache2d,
        kv_indices,
        kvc,
        kpe_dst,
        n,
        cache2d.stride(0),
        kpe_dst.stride(0),
        kpe_dst.stride(1),
        LORA=kv_lora_rank,
        ROPE=rope_dim,
        H=kpe_dst.shape[1],
        BLOCK_N=block_n,
        num_warps=4,
    )


def k3_mla_prefix_kv(
    cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_b_proj: torch.nn.Module,
    num_heads: int,
    nope_dim: int,
    v_dim: int,
    kv_lora_rank: int,
    rope_dim: int,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (k [N, H, dn+rope], v [N, H, dv]) for the prefix rows
    ``cache[kv_indices]``; v is a strided view (unit last stride)."""
    n = kv_indices.shape[0]
    dev = cache.device
    kvc = torch.empty((n, kv_lora_rank), dtype=dtype, device=dev)
    k = torch.empty((n, num_heads, nope_dim + rope_dim), dtype=dtype, device=dev)
    _gather_dequant(cache, kv_indices, kvc, k[:, :, nope_dim:], kv_lora_rank, rope_dim)
    kv = kv_b_proj(kvc)[0]
    assert kv.is_contiguous()
    kv = kv.view(n, num_heads, nope_dim + v_dim)
    v = kv[:, :, nope_dim:]
    rows = n * num_heads
    if rows > 0:
        block_r = 32
        _k3_prefix_knope_copy_kernel[(triton.cdiv(rows, block_r),)](
            kv,
            k,
            rows,
            nope_dim + v_dim,
            nope_dim + rope_dim,
            DN=nope_dim,
            BLOCK_R=block_r,
            num_warps=4,
        )
    return k, v
