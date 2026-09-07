"""Fused rope-tail + fp4 fake-quant for the DeepSeek-V4.1 low-ratio path.

`fake_quant_fp4(rope_tail(x, freqs, rd))` is 41 eager pointwise ops and appears at
16 call sites per decode step (4 compressor latents, 4 indexer keys, 8 indexer
queries), which measured 656 `vectorized_elementwise_kernel` launches per step --
9.3% of a bs=1 decode step at 1.32 us each, i.e. almost entirely launch latency.

This reproduces the two functions exactly, including the bf16 round-trip that
`rope_tail` performs on the rotated tail before `fake_quant_fp4` upcasts again,
and the round-half-to-even of `torch.round`.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

FP4_MAX = 6.0
FP4_AMAX_FLOOR = 6 * (2.0**-126)


@triton.jit
def _rope_tail_fake_quant_fp4_kernel(
    x_ptr,
    f_ptr,
    out_ptr,
    x_stride_r,
    out_stride_r,
    f_stride_t,
    rows_per_token,
    D: tl.constexpr,
    RD: tl.constexpr,
    BLK: tl.constexpr,
    AMAX_FLOOR: tl.constexpr,
    INVERSE: tl.constexpr,
):
    r = tl.program_id(0)
    t = r // rows_per_token
    offs = tl.arange(0, D)
    v = tl.load(x_ptr + r * x_stride_r + offs).to(tl.float32)

    # ---- rope_tail: adjacent pairs of the last RD features as one complex number
    head_len = D - RD
    in_tail = offs >= head_len
    pos = offs - head_len
    j = pos // 2
    is_im = (pos % 2) == 1
    re = tl.load(x_ptr + x_stride_r * r + head_len + 2 * j, mask=in_tail, other=0.0).to(
        tl.float32
    )
    im = tl.load(
        x_ptr + x_stride_r * r + head_len + 2 * j + 1, mask=in_tail, other=0.0
    ).to(tl.float32)
    # freqs arrives as the real view of a complex tensor, i.e. real and imag
    # interleaved on the last axis, so index 2*j / 2*j+1 rather than assuming a
    # unit stride -- `view_as_real(f)[..., 0]` has stride 2 along j, which silently
    # read the wrong coefficient for every tail element.
    fr = tl.load(f_ptr + t * f_stride_t + 2 * j, mask=in_tail, other=1.0)
    fi = tl.load(f_ptr + t * f_stride_t + 2 * j + 1, mask=in_tail, other=0.0)
    if INVERSE:
        fi = -fi
    rot = tl.where(is_im, re * fi + im * fr, re * fr - im * fi)
    # rope_tail casts the rotated tail back to x.dtype before the cat; the head
    # never leaves it. Reproduce that rounding or the quant sees different input.
    rot = rot.to(tl.bfloat16).to(tl.float32)
    v = tl.where(in_tail, rot, v)

    # ---- fake_quant_fp4: per-BLK ue8m0 scale, e2m1 round-trip
    vb = tl.reshape(v, (D // BLK, BLK))
    amax = tl.max(tl.abs(vb), axis=1)
    amax = tl.maximum(amax, AMAX_FLOOR) * (1.0 / 6.0)
    # ceil_pow2 on the IEEE bits, exact at powers of two
    bits = amax.to(tl.int32, bitcast=True)
    expo = ((bits >> 23) & 0xFF) - 127
    expo = expo + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = ((expo + 127) << 23).to(tl.float32, bitcast=True)

    s = vb / scale[:, None]
    s = tl.minimum(tl.maximum(s, -6.0), 6.0)
    mag = tl.abs(s)
    step = tl.where(mag < 2.0, 0.5, tl.where(mag < 4.0, 1.0, 2.0))
    # torch.round is round-half-to-even; torch.sign(0) is 0
    sgn = tl.where(s > 0, 1.0, tl.where(s < 0, -1.0, 0.0))
    q = libdevice.rint(mag / step) * step * sgn
    out = tl.reshape(q * scale[:, None], (D,))
    tl.store(out_ptr + r * out_stride_r + offs, out.to(out_ptr.dtype.element_ty))


def rope_tail_fake_quant_fp4(
    x: torch.Tensor,
    freqs: torch.Tensor,
    rope_dim: int,
    inverse: bool = False,
    block_size: int = 32,
) -> torch.Tensor:
    """One kernel for ``fake_quant_fp4(rope_tail(x, freqs, rope_dim))``.

    x: [T, ..., D] contiguous in the last dim; freqs: complex [T, rope_dim // 2].
    """
    x = x.contiguous()
    assert x.shape[-1] % block_size == 0
    assert rope_dim % 2 == 0 and rope_dim <= x.shape[-1]
    d = x.shape[-1]
    x2 = x.reshape(-1, d)
    rows = x2.shape[0]
    rows_per_token = rows // x.shape[0]
    f_real = torch.view_as_real(freqs.contiguous()).contiguous()
    out = torch.empty_like(x)
    if rows == 0:
        return out
    _rope_tail_fake_quant_fp4_kernel[(rows,)](
        x2,
        f_real,
        out.reshape(-1, d),
        x2.stride(0),
        d,
        f_real.stride(0),
        rows_per_token,
        D=d,
        RD=rope_dim,
        BLK=block_size,
        AMAX_FLOOR=FP4_AMAX_FLOOR,
        INVERSE=inverse,
        num_warps=4,
    )
    return out
