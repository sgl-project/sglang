"""One-launch SSMax scaling for Mistral-Large-4: factor = log(pos + 1) * s[head] + b[head].

The eager forms launch five to seven elementwise kernels per layer, which dominate
small-batch decode where every kernel costs a few microseconds regardless of size.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _ssmax_query_kernel(
    q_ptr,
    out_ptr,
    pos_ptr,
    s_ptr,
    b_ptr,
    stride_qt,
    stride_qh,
    stride_ot,
    stride_oh,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    pos = tl.load(pos_ptr + t).to(tl.float32)
    factor = tl.log(1.0 + pos) * tl.load(s_ptr + h) + tl.load(b_ptr + h)
    # The eager form casts the factor to the query dtype before multiplying.
    factor = factor.to(out_ptr.dtype.element_ty).to(tl.float32)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    q = tl.load(q_ptr + t * stride_qt + h * stride_qh + offs, mask=mask).to(tl.float32)
    tl.store(
        out_ptr + t * stride_ot + h * stride_oh + offs,
        (q * factor).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _ssmax_gate_kernel(
    w_ptr,
    qs_ptr,
    out_ptr,
    pos_ptr,
    s_ptr,
    b_ptr,
    scale,
    H,
    stride_wt,
    stride_wh,
    stride_qst,
    stride_qsh,
    HAS_Q_SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    t = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    pos = tl.load(pos_ptr + t).to(tl.float32)
    factor = tl.log(1.0 + pos) * tl.load(s_ptr + offs, mask=mask) + tl.load(
        b_ptr + offs, mask=mask
    )
    w = tl.load(w_ptr + t * stride_wt + offs * stride_wh, mask=mask).to(tl.float32)
    if HAS_Q_SCALE:
        qs = tl.load(qs_ptr + t * stride_qst + offs * stride_qsh, mask=mask)
        w = (w * qs) * scale
    tl.store(out_ptr + t * H + offs, w * factor, mask=mask)


def ssmax_scale_query(
    q: torch.Tensor, positions: torch.Tensor, s: torch.Tensor, b: torch.Tensor
) -> Optional[torch.Tensor]:
    """q * (log1p(pos) * s + b).to(q.dtype) for q of shape (tokens, heads, dim).

    Returns None when the layout is not one this kernel handles."""
    if q.dim() != 3 or q.stride(-1) != 1 or positions.shape[0] != q.shape[0]:
        return None
    tokens, heads, dim = q.shape
    out = torch.empty_like(q)
    if tokens == 0:
        return out
    _ssmax_query_kernel[(tokens, heads)](
        q,
        out,
        positions,
        s,
        b,
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        D=dim,
        BLOCK_D=triton.next_power_of_2(dim),
    )
    return out


def ssmax_scale_gates(
    weights: torch.Tensor,
    positions: torch.Tensor,
    s: torch.Tensor,
    b: torch.Tensor,
    q_scale: Optional[torch.Tensor] = None,
    scale: float = 1.0,
) -> Optional[torch.Tensor]:
    """Float32 (tokens, heads, 1) gates: ((w * q_scale) * scale) * factor, or w * factor
    without q_scale. weights is (tokens, heads) or (tokens, heads, 1); q_scale is
    (tokens, heads, 1). Returns None when the layout is not one this kernel handles."""
    if weights.dim() == 3:
        if weights.shape[-1] != 1:
            return None
        weights = weights[..., 0]
    if weights.dim() != 2 or positions.shape[0] < weights.shape[0]:
        return None
    tokens, heads = weights.shape
    if q_scale is not None:
        if q_scale.shape != (tokens, heads, 1) or q_scale.dtype != torch.float32:
            return None
        qs, stride_qst, stride_qsh = q_scale, q_scale.stride(0), q_scale.stride(1)
    else:
        qs, stride_qst, stride_qsh = weights, 0, 0
    out = torch.empty(
        (tokens, heads, 1), dtype=torch.float32, device=weights.device
    )
    if tokens == 0:
        return out
    _ssmax_gate_kernel[(tokens,)](
        weights,
        qs,
        out,
        positions,
        s,
        b,
        scale,
        heads,
        weights.stride(0),
        weights.stride(1),
        stride_qst,
        stride_qsh,
        HAS_Q_SCALE=q_scale is not None,
        BLOCK_H=triton.next_power_of_2(heads),
    )
    return out
