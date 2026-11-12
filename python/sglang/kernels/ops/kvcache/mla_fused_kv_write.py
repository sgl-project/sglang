# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""The absorbed MLA path's RoPE, q / k concat and FP8 KV-row store in one launch (Triton attention branch)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_e4m3fn_bytes(x):
    # torch's BF16 -> float8_e4m3fn cast: round to nearest even, saturate to +-448, NaN -> 0x7f.
    return x.to(tl.float32).to(tl.float8e4nv).to(tl.uint8, bitcast=True)


@triton.jit
def _rotate_pair(x, y, cos, sin):
    # rope.cuh's out_x = x * cos - y * sin and out_y = x * sin + y * cos as NVCC contracts them in
    # fused_rope_kernel (out_x fuses x * cos, out_y fuses y * cos): bitwise to apply_rope_with_cos_sin_cache_inplace.
    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    out_x = tl.fma(xf, cos, -(yf * sin))
    out_y = tl.fma(yf, cos, xf * sin)
    return out_x.to(tl.bfloat16), out_y.to(tl.bfloat16)


@triton.jit
def _mla_concat_qk_store_fp8_kv_kernel(
    q_nope,
    q_nope_stride_t,
    q_nope_stride_h,
    q_pe,
    q_pe_stride_t,
    q_pe_stride_h,
    k_nope,
    k_nope_stride_t,
    k_pe,
    k_pe_stride_t,
    loc,
    positions,
    cos_sin_cache,
    q_out,
    k_out,
    kv_rows,
    NUM_HEADS: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    ROPE: tl.constexpr,
    USE_GDC: tl.constexpr,
):
    if USE_GDC:
        # Launched with PDL: every input is read and every row stored after the wait.
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    # Program (t, h < NUM_HEADS) writes q row (t, h); program (t, NUM_HEADS) writes k row t and its KV row.
    t = tl.program_id(0)
    h = tl.program_id(1)
    nope = tl.arange(0, NOPE_DIM)
    pair = tl.arange(0, ROPE_DIM // 2)  # interleaved (GPT-J) rope pairs: elements 2p, 2p + 1
    row_dim: tl.constexpr = NOPE_DIM + ROPE_DIM
    if h < NUM_HEADS:
        nope_src = q_nope + t * q_nope_stride_t + h * q_nope_stride_h
        pe_src = q_pe + t * q_pe_stride_t + h * q_pe_stride_h
        dst = q_out + (t * NUM_HEADS + h) * row_dim
    else:
        nope_src = k_nope + t * k_nope_stride_t
        pe_src = k_pe + t * k_pe_stride_t
        dst = k_out + t * row_dim
    x = tl.load(nope_src + nope)
    pe_even = tl.load(pe_src + 2 * pair)
    pe_odd = tl.load(pe_src + 2 * pair + 1)
    if ROPE:
        # cos_sin_cache: fp32 [positions, ROPE_DIM], cos | sin halves
        cs = cos_sin_cache + tl.load(positions + t).to(tl.int64) * ROPE_DIM
        pe_even, pe_odd = _rotate_pair(
            pe_even, pe_odd, tl.load(cs + pair), tl.load(cs + ROPE_DIM // 2 + pair)
        )
    tl.store(dst + nope, x)
    tl.store(dst + NOPE_DIM + 2 * pair, pe_even)
    tl.store(dst + NOPE_DIM + 2 * pair + 1, pe_odd)
    if h == NUM_HEADS:
        kv = kv_rows + tl.load(loc + t).to(tl.int64) * row_dim
        tl.store(kv + nope, _fp8_e4m3fn_bytes(x))
        tl.store(kv + NOPE_DIM + 2 * pair, _fp8_e4m3fn_bytes(pe_even))
        tl.store(kv + NOPE_DIM + 2 * pair + 1, _fp8_e4m3fn_bytes(pe_odd))


def mla_concat_qk_store_fp8_kv(
    *,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    kv_rows: torch.Tensor,
    loc: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
    cos_sin_cache: Optional[torch.Tensor] = None,
    pdl: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """q = cat(q_nope, q_pe), k = cat(k_nope, k_pe) and kv_rows[loc] = k as float8_e4m3fn bytes, bitwise equal
    to two torch.cat calls plus MLATokenToKVPool.set_kv_buffer on an FP8 pool. With cos_sin_cache, q_pe and k_pe
    are first rotated as RotaryEmbedding.forward_cuda does in place (interleaved pairs, positions); the inputs are
    left untouched. pdl: launch with programmatic dependent launch (same output).

    q_nope [T, H, nope] and q_pe [T, H, rope], k_nope and k_pe [T, 1, *]: BF16, any token / head strides, unit
    stride in the last dim. kv_rows: one layer's uint8 pool buffer [slots, 1, nope + rope]; loc [T] slot ids.
    Duplicate slots in loc store in an unspecified order, as the index_put they replace does.
    """
    num_tokens, num_heads, nope_dim = q_nope.shape
    rope_dim = q_pe.shape[-1]
    row_dim = nope_dim + rope_dim
    assert q_pe.shape == (num_tokens, num_heads, rope_dim)
    assert k_nope.shape == (num_tokens, 1, nope_dim)
    assert k_pe.shape == (num_tokens, 1, rope_dim)
    assert all(x.dtype == torch.bfloat16 for x in (q_nope, q_pe, k_nope, k_pe))
    assert all(x.stride(-1) == 1 for x in (q_nope, q_pe, k_nope, k_pe))
    assert kv_rows.dtype == torch.uint8 and kv_rows.is_contiguous()
    assert kv_rows.shape[-1] == row_dim and kv_rows.numel() == kv_rows.shape[0] * row_dim
    assert loc.shape == (num_tokens,)
    rope = cos_sin_cache is not None
    if rope:
        assert positions.shape == (num_tokens,)
        assert cos_sin_cache.dtype == torch.float32 and cos_sin_cache.is_contiguous()
        assert cos_sin_cache.shape[-1] == rope_dim
    q = q_nope.new_empty(num_tokens, num_heads, row_dim)
    k = k_nope.new_empty(num_tokens, 1, row_dim)
    if num_tokens > 0:
        _mla_concat_qk_store_fp8_kv_kernel[(num_tokens, num_heads + 1)](
            q_nope,
            q_nope.stride(0),
            q_nope.stride(1),
            q_pe,
            q_pe.stride(0),
            q_pe.stride(1),
            k_nope,
            k_nope.stride(0),
            k_pe,
            k_pe.stride(0),
            loc,
            positions if rope else loc,
            cos_sin_cache if rope else kv_rows,
            q,
            k,
            kv_rows,
            NUM_HEADS=num_heads,
            NOPE_DIM=nope_dim,
            ROPE_DIM=rope_dim,
            ROPE=rope,
            USE_GDC=pdl,
            num_warps=4,
            **({"launch_pdl": True} if pdl else {}),
        )
    return q, k
