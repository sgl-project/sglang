# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional

import torch
import triton
import triton.language as tl

STEP5_HEAD_DIM = 192
STEP5_REGION_TOKENS = 8
STEP5_SELECTED_REGIONS = 513

_QKV_WIDTHS = (1920, 3456)
_QKV_GATE_WIDTHS = (1928, 3472)


@triton.jit
def _norm_rope_head(
    row,
    weight_ptr,
    rotary_row,
    eps,
    HALF: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    d = tl.arange(0, BLOCK_D)
    in_head = d < D
    partner = tl.where(d < HALF, d + HALF, tl.where(d < 2 * HALF, d - HALF, d))
    x = tl.load(row + d, mask=in_head, other=0.0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
    w = tl.load(weight_ptr + d, mask=in_head, other=0.0)
    normed = (x * inv_rms * (w + 1.0)).to(tl.bfloat16).to(tl.float32)
    x_partner = tl.load(row + partner, mask=in_head, other=0.0).to(tl.float32)
    w_partner = tl.load(weight_ptr + partner, mask=in_head, other=0.0)
    normed_partner = (
        (x_partner * inv_rms * (w_partner + 1.0)).to(tl.bfloat16).to(tl.float32)
    )
    freq = tl.where(d < HALF, d, d - HALF)
    rotary = d < 2 * HALF
    cos = tl.load(rotary_row + freq, mask=rotary, other=1.0).to(tl.float32)
    sin = tl.load(rotary_row + HALF + freq, mask=rotary, other=0.0).to(tl.float32)
    return tl.where(
        d < HALF,
        normed * cos - normed_partner * sin,
        tl.where(rotary, normed_partner * sin + normed * cos, normed),
    ).to(tl.bfloat16)


@triton.jit
def _qknorm_rope_kernel(
    qkv_ptr,
    q_weight_ptr,
    k_weight_ptr,
    rotary_ptr,
    positions_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    gate_ptr,
    k_cache_ptr,
    v_cache_ptr,
    loc_ptr,
    qkv_stride,
    rotary_stride,
    eps,
    HEADS: tl.constexpr,
    HALF: tl.constexpr,
    GATED: tl.constexpr,
    STORE_CACHE: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    in_head = d < D
    row = qkv_ptr + token * qkv_stride + head * D
    rotary_row = rotary_ptr + tl.load(positions_ptr + token) * rotary_stride
    if head < HEADS:
        out = _norm_rope_head(row, q_weight_ptr, rotary_row, eps, HALF, D, BLOCK_D)
        tl.store(q_ptr + (token * HEADS + head) * D + d, out, mask=in_head)
    else:
        out = _norm_rope_head(row, k_weight_ptr, rotary_row, eps, HALF, D, BLOCK_D)
        tl.store(k_ptr + token * D + d, out, mask=in_head)
        v = tl.load(row + D + d, mask=in_head, other=0.0)
        tl.store(v_ptr + token * D + d, v, mask=in_head)
        if STORE_CACHE:
            loc = tl.load(loc_ptr + token)
            live = in_head & (loc >= 0)
            tl.store(k_cache_ptr + loc * D + d, out, mask=live)
            tl.store(v_cache_ptr + loc * D + d, v, mask=live)
        if GATED:
            g = tl.arange(0, BLOCK_HEADS)
            gate = tl.load(row + 2 * D + g, mask=g < HEADS, other=0.0)
            tl.store(gate_ptr + token * HEADS + g, gate, mask=g < HEADS)


def step5_qknorm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    rotary_cache: torch.Tensor,
    *,
    eps: float,
    kv_write: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-centered RMSNorm then NeoX RoPE on Q/K of a packed [Q|K|V] row.

    QKV is BF16 [T,(Qheads+2)*192]; weights are FP32 [192]; the rotary cache is
    BF16 [capacity, span] holding [cos|sin]. Norm output is rounded to BF16
    before the FP32 rotation. ``kv_write`` is ``(k_cache, v_cache, loc)`` with
    BF16 [slots,192] caches; rows with ``loc < 0`` are not stored.
    """
    if qkv.ndim != 2 or qkv.shape[1] not in _QKV_WIDTHS:
        raise ValueError("Step5 QKV must have shape [tokens,1920 or 3456]")
    q, k, v, _ = _qknorm_rope(
        qkv,
        q_weight,
        k_weight,
        positions,
        rotary_cache,
        heads=qkv.shape[1] // STEP5_HEAD_DIM - 2,
        gated=False,
        eps=eps,
        kv_write=kv_write,
    )
    return q, k, v


def step5_qknorm_rope_gated(
    qkv_gate: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    rotary_cache: torch.Tensor,
    *,
    eps: float,
    kv_write: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``step5_qknorm_rope`` over [Q|K|V|head gate]; also returns the gate [T,Qheads]."""
    if qkv_gate.ndim != 2 or qkv_gate.shape[1] not in _QKV_GATE_WIDTHS:
        raise ValueError("Step5 QKV with gate must have shape [tokens,1928 or 3472]")
    return _qknorm_rope(
        qkv_gate,
        q_weight,
        k_weight,
        positions,
        rotary_cache,
        heads=(qkv_gate.shape[1] - 2 * STEP5_HEAD_DIM) // (STEP5_HEAD_DIM + 1),
        gated=True,
        eps=eps,
        kv_write=kv_write,
    )


def _qknorm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    rotary_cache: torch.Tensor,
    *,
    heads: int,
    gated: bool,
    eps: float,
    kv_write: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if rotary_cache.ndim != 2 or rotary_cache.shape[1] not in (64, 192):
        raise ValueError("Step5 main rotary cache must have width 64 or 192")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Step5 Q/K norm needs a positive eps")
    if qkv.stride(1) != 1 or not rotary_cache.is_contiguous():
        raise ValueError("Step5 Q/K norm requires row-major QKV and rotary cache")
    if q_weight.dtype != torch.float32 or k_weight.dtype != torch.float32:
        raise ValueError("Step5 Q/K norm weights must be FP32")
    tokens = qkv.shape[0]
    q = qkv.new_empty((tokens, heads, STEP5_HEAD_DIM))
    k = qkv.new_empty((tokens, 1, STEP5_HEAD_DIM))
    v = qkv.new_empty((tokens, 1, STEP5_HEAD_DIM))
    gate = qkv.new_empty((tokens, heads)) if gated else None
    k_cache, v_cache, loc = kv_write if kv_write is not None else (None, None, None)
    if tokens:
        _qknorm_rope_kernel[(tokens, heads + 1)](
            qkv,
            q_weight,
            k_weight,
            rotary_cache,
            positions,
            q,
            k,
            v,
            gate,
            k_cache,
            v_cache,
            loc,
            qkv.stride(0),
            rotary_cache.stride(0),
            eps,
            HEADS=heads,
            HALF=rotary_cache.shape[1] // 2,
            GATED=gated,
            STORE_CACHE=kv_write is not None,
            D=STEP5_HEAD_DIM,
            BLOCK_D=256,
            BLOCK_HEADS=triton.next_power_of_2(heads),
            num_warps=4,
        )
    return q, k, v, gate


@triton.jit
def _head_gate_kernel(
    output_ptr,
    attn_output_ptr,
    gate_ptr,
    gate_stride_row,
    gate_stride_head,
    hidden_dim: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    offsets = block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offsets < hidden_dim
    head = offsets // HEAD_DIM
    attn_off = row * hidden_dim + offsets
    attn = tl.load(attn_output_ptr + attn_off, mask=mask, other=0.0).to(tl.float32)
    gate_off = row * gate_stride_row + head * gate_stride_head
    gate = tl.load(gate_ptr + gate_off, mask=mask, other=0.0).to(tl.float32)
    sigmoid = tl.sigmoid(gate).to(gate_ptr.dtype.element_ty).to(tl.float32)
    tl.store(output_ptr + attn_off, attn * sigmoid, mask=mask)


def step5_head_gate(
    attn_output: torch.Tensor, gate: torch.Tensor, *, head_dim: int
) -> torch.Tensor:
    """Apply per-head gates with a materialized gate-dtype sigmoid rounding."""
    num_tokens, num_heads = gate.shape
    hidden_dim = num_heads * head_dim
    assert attn_output.shape == (num_tokens, hidden_dim)
    out = torch.empty_like(attn_output)
    block_h = 1024 if num_tokens < 1024 else 2048
    _head_gate_kernel[(num_tokens, triton.cdiv(hidden_dim, block_h))](
        out,
        attn_output,
        gate,
        gate.stride(0),
        gate.stride(1),
        hidden_dim=hidden_dim,
        HEAD_DIM=head_dim,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return out


@triton.jit
def _sparse_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    packed_ptr,
    counts_ptr,
    lengths_ptr,
    out_ptr,
    k_scale_ptr,
    v_scale_ptr,
    q_stride_row,
    q_stride_head,
    kv_stride,
    packed_stride,
    out_stride_row,
    out_stride_head,
    scale,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    REGION: tl.constexpr,
    FP8_KV: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BLOCK_H)
    d = tl.arange(0, BLOCK_D)
    n = tl.arange(0, BLOCK_N)
    head_mask = h < HEADS
    dim_mask = d < D
    count = tl.load(counts_ptr + row)
    length = tl.load(lengths_ptr + row)

    q = tl.load(
        q_ptr + row * q_stride_row + h[:, None] * q_stride_head + d[None, :],
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    if FP8_KV:
        k_scale = tl.load(k_scale_ptr)
        v_scale = tl.load(v_scale_ptr)

    m = tl.full((BLOCK_H,), float("-inf"), tl.float32)
    l = tl.zeros((BLOCK_H,), tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)
    for start in range(0, count * REGION, BLOCK_N):
        token = start + n
        region = token // REGION
        offset = token % REGION
        selected = (region < count) & (length > 0)
        entry = tl.load(
            packed_ptr + row * packed_stride + region, mask=selected, other=0
        )
        slot = (entry >> 32) * REGION + offset
        logical = entry - ((entry >> 32) << 32)
        valid = selected & (logical + offset < length)
        kv_mask = valid[:, None] & dim_mask[None, :]
        kv_off = slot[:, None] * kv_stride + d[None, :]
        k = tl.load(k_ptr + kv_off, mask=kv_mask, other=0.0)
        v = tl.load(v_ptr + kv_off, mask=kv_mask, other=0.0)
        if FP8_KV:
            k = (k.to(tl.float32) * k_scale).to(tl.bfloat16)
            v = (v.to(tl.float32) * v_scale).to(tl.bfloat16)

        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(valid[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m - m_safe)
        p = tl.exp(s - m_safe[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new

    out = acc / tl.where(l > 0, l, 1.0)[:, None]
    tl.store(
        out_ptr + row * out_stride_row + h[:, None] * out_stride_head + d[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=head_mask[:, None] & dim_mask[None, :],
    )


def step5_sparse_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    packed: torch.Tensor,
    counts: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Attention of each query row over its selected 8-token regions.

    q/out are [T,heads,192]; k_cache/v_cache are [slots,1,192] BF16, or FP8 with
    FP32 [1] scales. packed is int64 [T,>=513] with (physical_slot // 8) in the
    high word and the region's first logical position in the low word; counts
    [T] bounds the live entries and lengths [T] is the causal KV length, so a
    row with length 0 writes zeros.
    """
    rows, heads, dim = q.shape
    if dim != STEP5_HEAD_DIM or k_cache.shape[-1] != dim or v_cache.shape[-1] != dim:
        raise ValueError("Step5 sparse attention requires head dim 192")
    if packed.dtype != torch.int64 or packed.shape[0] != rows:
        raise ValueError("Step5 sparse attention requires int64 packed [T,513]")
    fp8_kv = k_cache.dtype == torch.float8_e4m3fn
    if fp8_kv and (k_scale is None or v_scale is None):
        raise ValueError("Step5 FP8 KV requires K and V scales")
    if q.stride(2) != 1 or out.stride(2) != 1 or k_cache.stride(-1) != 1:
        raise ValueError("Step5 sparse attention requires contiguous head dim")
    if rows:
        _sparse_attention_kernel[(rows,)](
            q,
            k_cache,
            v_cache,
            packed,
            counts,
            lengths,
            out,
            k_scale,
            v_scale,
            q.stride(0),
            q.stride(1),
            k_cache.stride(0),
            packed.stride(0),
            out.stride(0),
            out.stride(1),
            scale,
            HEADS=heads,
            D=dim,
            REGION=STEP5_REGION_TOKENS,
            FP8_KV=fp8_kv,
            BLOCK_H=max(16, triton.next_power_of_2(heads)),
            BLOCK_D=256,
            BLOCK_N=64,
            num_warps=8,
        )
    return out
