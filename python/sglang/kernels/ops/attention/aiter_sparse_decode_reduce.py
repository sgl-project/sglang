# SPDX-License-Identifier: Apache-2.0
"""Split-KV combine for aiter's gfx950 sparse decode kernel, bitwise ``_pa_decode_sparse_reduce``,
with the DeepSeek-V4 inverse RoPE folded in."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _aiter_sparse_decode_reduce_kernel(
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    sink_ptr,
    out_ptr,
    fr_ptr,
    pos_ptr,
    pm_stride_t,
    pm_stride_s,
    pa_stride_t,
    pa_stride_s,
    pa_stride_h,
    out_stride_t,
    out_stride_h,
    fr_stride_pos,
    NUM_SPLITS: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    HAS_SINK: tl.constexpr,
    INV_ROPE: tl.constexpr,
):
    """Grid (num_queries, num_heads). Partials: ``part_m`` / ``part_l``
    [T, S, H] fp32 (``m`` in the base-2 exponent domain), ``part_acc``
    [T, S, H, D] fp32 -- the layout aiter's gluon main kernel stores."""
    RCP_LN2: tl.constexpr = 1.4426950408889634
    t = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    pm_base = t * pm_stride_t + h
    pa_base = t * pa_stride_t + h * pa_stride_h

    neg_inf = float("-inf")
    m_final = tl.full((), neg_inf, tl.float32)
    for s in tl.static_range(NUM_SPLITS):
        m_s = tl.load(part_m_ptr + pm_base + s * pm_stride_s)
        m_final = tl.maximum(m_final, m_s)
    if HAS_SINK:
        sink = tl.load(sink_ptr + h).to(tl.float32)
        m_final = tl.maximum(m_final, sink * RCP_LN2)

    l_final = tl.full((), 0.0, tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)
    for s in tl.static_range(NUM_SPLITS):
        m_s = tl.load(part_m_ptr + pm_base + s * pm_stride_s)
        l_s = tl.load(part_l_ptr + pm_base + s * pm_stride_s)
        w = tl.exp2(m_s - m_final)
        # gluon contracts l + w * l_s into one fma; Triton leaves the scalar chain as mul + add
        l_final = tl.fma(w, l_s, l_final)
        acc_s = tl.load(part_acc_ptr + pa_base + s * pa_stride_s + d)
        acc = tl.fma(w, acc_s, acc)
    if HAS_SINK:
        l_final = l_final + tl.exp2(sink * RCP_LN2 - m_final)

    out = (acc / l_final).to(out_ptr.dtype.element_ty)
    if INV_ROPE:
        # the model's standalone kernel reads the bf16-rounded output, so rotate the rounded values
        x = out.to(tl.float32)
        pos = tl.load(pos_ptr + t)
        is_rope = d >= D - RD
        cos_idx = ((d - (D - RD)) // 2) * 2
        cos = tl.load(fr_ptr + pos * fr_stride_pos + cos_idx, mask=is_rope, other=0.0)
        sin = tl.load(
            fr_ptr + pos * fr_stride_pos + cos_idx + 1, mask=is_rope, other=0.0
        )
        x_sin = x * sin
        even = d % 2 == 0
        x_neg = tl.where(even, -x_sin, x_sin)
        x_neg = tl.reshape(x_neg, (D // 2, 2))
        x_neg = tl.flip(x_neg, 1)
        x_rot = tl.reshape(x_neg, (D,))
        roped = tl.fma(x, cos, x_rot)
        out = tl.where(is_rope, roped.to(out_ptr.dtype.element_ty), out)
    tl.store(out_ptr + t * out_stride_t + h * out_stride_h + d, out)


def aiter_sparse_split_reduce(
    part_acc: torch.Tensor,
    part_m: torch.Tensor,
    part_l: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    out_dtype: torch.dtype = torch.bfloat16,
    inv_rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Combine ``pa_decode_sparse(..., skip_reduce=True)``'s partials into the
    ``[T, H, D]`` output. ``inv_rope`` is ``(freqs_real [max_pos, rope_dim] fp32,
    positions [T])``: the last ``rope_dim`` of every head are inverse-rotated."""
    T, S, H, D = part_acc.shape
    assert part_m.shape == (T, S, H) and part_l.shape == (T, S, H), (
        part_m.shape,
        part_l.shape,
    )
    assert part_acc.stride(3) == 1 and part_m.stride(2) == 1 and part_l.stride(2) == 1
    assert part_m.stride() == part_l.stride()
    assert triton.next_power_of_2(D) == D and S >= 1
    out = torch.empty((T, H, D), dtype=out_dtype, device=part_acc.device)
    if T == 0:
        return out
    has_sink = attn_sink is not None
    if inv_rope is not None:
        freqs_real, positions = inv_rope
        rope_dim = freqs_real.shape[1]
        assert freqs_real.dtype == torch.float32 and freqs_real.stride(1) == 1
        assert positions.shape == (T,), positions.shape
        assert rope_dim % 2 == 0 and rope_dim <= D
    else:
        freqs_real = positions = out  # unread placeholders
        rope_dim = 0
    _aiter_sparse_decode_reduce_kernel[(T, H)](
        part_m,
        part_l,
        part_acc,
        attn_sink if has_sink else out,
        out,
        freqs_real,
        positions,
        part_m.stride(0),
        part_m.stride(1),
        part_acc.stride(0),
        part_acc.stride(1),
        part_acc.stride(2),
        out.stride(0),
        out.stride(1),
        freqs_real.stride(0) if inv_rope is not None else 0,
        NUM_SPLITS=S,
        D=D,
        RD=rope_dim,
        HAS_SINK=has_sink,
        INV_ROPE=inv_rope is not None,
        num_warps=1,
    )
    return out
