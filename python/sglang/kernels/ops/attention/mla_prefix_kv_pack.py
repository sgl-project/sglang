"""MLA cached-prefix K/V expansion for an FP8 prefill kernel, in two launches.

An extend over a cached prefix re-expands the whole latent cache:
    gather:  kvc = bf16(cache[idx, :R]), k_pe = cache[idx, R:]     (fp8 cache)
    kv_b_proj GEMM on kvc  -> kv [L, H, Dn + Dv]
    pack:    k = fp8(cat(kv[:, h, :Dn], k_pe)), v = fp8(kv[:, h, Dn:])
             for output heads h' = 0 .. Hp - 1 tiled from h = h' % H
replacing index_select, two casts, a contiguous copy, the k_pe cat, the head
repeat and two fp8 casts. The casts are plain round-to-nearest conversions, as
torch's .to(float8_e4m3fn) does for in-range values; k_pe keeps its cached bytes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_latent_kernel(
    cache_ptr,  # [slots, R + P] fp8 rows
    idx_ptr,  # [L]
    kvc_ptr,  # [L, R] bf16
    pe_ptr,  # [L, P] fp8
    L,
    stride_cache,
    R: tl.constexpr,
    P: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_L + tl.arange(0, BLOCK_L)
    row_mask = rows < L
    src = tl.load(idx_ptr + rows, mask=row_mask, other=0).to(tl.int64)
    o_r = tl.arange(0, R)
    lat = tl.load(
        cache_ptr + src[:, None] * stride_cache + o_r[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    tl.store(
        kvc_ptr + rows[:, None].to(tl.int64) * R + o_r[None, :],
        lat.to(kvc_ptr.dtype.element_ty),
        mask=row_mask[:, None],
    )
    o_p = tl.arange(0, P)
    pe = tl.load(
        cache_ptr + src[:, None] * stride_cache + R + o_p[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    tl.store(
        pe_ptr + rows[:, None].to(tl.int64) * P + o_p[None, :],
        pe,
        mask=row_mask[:, None],
    )


@triton.jit
def _pack_kv_kernel(
    kv_ptr,  # [L, H, DN + DV] bf16, row stride stride_kv
    pe_ptr,  # [L, P] fp8
    k_ptr,  # [L, HP, DN + P] fp8
    v_ptr,  # [L, HP, DV] fp8
    L,
    stride_kv,
    H: tl.constexpr,
    HP: tl.constexpr,
    DN: tl.constexpr,
    DV: tl.constexpr,
    P: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    h = tl.program_id(1)
    rows = tl.program_id(0) * BLOCK_L + tl.arange(0, BLOCK_L)
    row_mask = rows[:, None] < L
    rows64 = rows[:, None].to(tl.int64)
    o_n = tl.arange(0, DN)
    o_v = tl.arange(0, DV)
    o_p = tl.arange(0, P)
    src = kv_ptr + rows64 * stride_kv + h * (DN + DV)
    k_nope = tl.load(src + o_n[None, :], mask=row_mask, other=0.0).to(
        k_ptr.dtype.element_ty
    )
    v = tl.load(src + DN + o_v[None, :], mask=row_mask, other=0.0).to(
        v_ptr.dtype.element_ty
    )
    pe = tl.load(pe_ptr + rows64 * P + o_p[None, :], mask=row_mask, other=0.0)
    # Output head h, and h + H while it is a tiled padding head.
    for rep in tl.static_range(0, HP, H):
        if rep + h < HP:
            k_row = k_ptr + (rows64 * HP + rep + h) * (DN + P)
            tl.store(k_row + o_n[None, :], k_nope, mask=row_mask)
            tl.store(k_row + DN + o_p[None, :], pe, mask=row_mask)
            tl.store(
                v_ptr + (rows64 * HP + rep + h) * DV + o_v[None, :], v, mask=row_mask
            )


def gather_prefix_latent(
    kv_cache: torch.Tensor, kv_indices: torch.Tensor, kv_lora_rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """kv_cache [slots, 1, R + P] fp8, kv_indices [L] -> (bf16 latent [L, R],
    fp8 rope part [L, P])."""
    cache = kv_cache.view(kv_cache.shape[0], -1)
    width = cache.shape[1]
    rope = width - kv_lora_rank
    num_rows = kv_indices.shape[0]
    kvc = torch.empty(
        (num_rows, kv_lora_rank), dtype=torch.bfloat16, device=kv_cache.device
    )
    pe = torch.empty((num_rows, rope), dtype=kv_cache.dtype, device=kv_cache.device)
    if num_rows:
        block_l = 16
        _gather_latent_kernel[(triton.cdiv(num_rows, block_l),)](
            cache,
            kv_indices,
            kvc,
            pe,
            num_rows,
            cache.stride(0),
            R=kv_lora_rank,
            P=rope,
            BLOCK_L=block_l,
            num_warps=4,
        )
    return kvc, pe


def pack_prefix_kv_fp8(
    kv: torch.Tensor,
    k_pe: torch.Tensor,
    *,
    num_heads: int,
    num_heads_padded: int,
    nope_dim: int,
    v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """kv [L, H * (Dn + Dv)] bf16 (kv_b_proj output), k_pe [L, P] fp8 ->
    (k [L, Hp, Dn + P], v [L, Hp, Dv]) in k_pe's fp8 dtype, heads tiled to Hp."""
    num_rows = kv.shape[0]
    rope = k_pe.shape[1]
    assert kv.shape[1] == num_heads * (nope_dim + v_dim) and kv.stride(1) == 1
    assert k_pe.is_contiguous() and num_heads_padded >= num_heads
    k = torch.empty(
        (num_rows, num_heads_padded, nope_dim + rope),
        dtype=k_pe.dtype,
        device=kv.device,
    )
    v = torch.empty(
        (num_rows, num_heads_padded, v_dim), dtype=k_pe.dtype, device=kv.device
    )
    if num_rows:
        block_l = 16
        _pack_kv_kernel[(triton.cdiv(num_rows, block_l), num_heads)](
            kv,
            k_pe,
            k,
            v,
            num_rows,
            kv.stride(0),
            H=num_heads,
            HP=num_heads_padded,
            DN=nope_dim,
            DV=v_dim,
            P=rope,
            BLOCK_L=block_l,
            num_warps=4,
        )
    return k, v
