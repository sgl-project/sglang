"""Fused RoPE tail and fp4 fake-quant for the DeepSeek-V4.1 low-ratio path.

Preserves the bf16 round-trip after RoPE and before quantization, and the
round-half-to-even behavior of torch.round.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

FP4_MAX = 6.0
FP4_AMAX_FLOOR = 6 * (2.0**-126)


@triton.jit
def rope_tail_fake_quant_fp4_row(
    x_row_ptr,
    f_row_ptr,
    D: tl.constexpr,
    RD: tl.constexpr,
    BLK: tl.constexpr,
    AMAX_FLOOR: tl.constexpr,
    INVERSE: tl.constexpr,
    COMPRESSED_KV: tl.constexpr,
):
    """One row of ``rope_tail_fake_quant_fp4`` as an fp32 ``[D]`` vector: ``x_row_ptr`` points
    at the row, ``f_row_ptr`` at the token's ``[RD // 2, 2]`` real view of the complex freqs."""
    offs = tl.arange(0, D)
    v = tl.load(x_row_ptr + offs).to(tl.float32)

    # ---- rope_tail: adjacent pairs of the last RD features as one complex number
    head_len = D - RD
    in_tail = offs >= head_len
    pos = offs - head_len
    j = pos // 2
    is_im = (pos % 2) == 1
    re = tl.load(x_row_ptr + head_len + 2 * j, mask=in_tail, other=0.0).to(tl.float32)
    im = tl.load(x_row_ptr + head_len + 2 * j + 1, mask=in_tail, other=0.0).to(
        tl.float32
    )
    # freqs is a real/imag-interleaved view with stride 2 between complex pairs;
    # index 2*j / 2*j+1 to preserve that layout.
    fr = tl.load(f_row_ptr + 2 * j, mask=in_tail, other=1.0)
    fi = tl.load(f_row_ptr + 2 * j + 1, mask=in_tail, other=0.0)
    if INVERSE:
        fi = -fi
    rot = tl.where(is_im, re * fi + im * fr, re * fr - im * fi)
    # rope_tail casts the rotated tail back to x.dtype before the cat; the head
    # never leaves it. Reproduce that rounding or the quant sees different input.
    rot = rot.to(tl.bfloat16).to(tl.float32)
    v = tl.where(in_tail, rot, v)

    # ---- FP4 round-trip, with a separate scale format for compressed KV.
    vb = tl.reshape(v, (D // BLK, BLK))
    amax = tl.max(tl.abs(vb), axis=1)
    if COMPRESSED_KV:
        scale = tl.minimum(tl.maximum(amax * (1.0 / 6.0), 2.0**-9), 448.0)
        scale = scale.to(tl.float8e4nv).to(tl.float32)
        s = tl.div_rn(vb, scale[:, None])
    else:
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
    return tl.reshape(q * scale[:, None], (D,))


@triton.jit
def _rope_tail_fake_quant_fp4_kernel(
    x_ptr,
    f_ptr,
    pos_ptr,
    out_ptr,
    x_stride_r,
    out_stride_r,
    f_stride_t,
    rows_per_token,
    num_pos,
    D: tl.constexpr,
    RD: tl.constexpr,
    BLK: tl.constexpr,
    AMAX_FLOOR: tl.constexpr,
    INVERSE: tl.constexpr,
    COMPRESSED_KV: tl.constexpr,
    HAS_POS: tl.constexpr,
):
    r = tl.program_id(0)
    t = r // rows_per_token
    if HAS_POS:
        # freqs is the whole table: row t reads its position's entry (the freqs[positions] gather folded in)
        t = tl.load(pos_ptr + t).to(tl.int64)
        # the caller owns positions < num_pos; an out-of-range row reads entry 0 instead of past the table
        t = tl.where((t >= 0) & (t < num_pos), t, 0)
    out = rope_tail_fake_quant_fp4_row(
        x_ptr + r * x_stride_r,
        f_ptr + t * f_stride_t,
        D=D,
        RD=RD,
        BLK=BLK,
        AMAX_FLOOR=AMAX_FLOOR,
        INVERSE=INVERSE,
        COMPRESSED_KV=COMPRESSED_KV,
    )
    tl.store(
        out_ptr + r * out_stride_r + tl.arange(0, D), out.to(out_ptr.dtype.element_ty)
    )


def rope_tail_fake_quant_fp4(
    x: torch.Tensor,
    freqs: torch.Tensor,
    rope_dim: int,
    inverse: bool = False,
    block_size: int = 32,
    *,
    compressed_kv: bool = False,
    positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """RoPE and FP4 round-trip: per-16 E4M3 for compressed KV, per-32 UE8M0 otherwise.

    x: [T, ..., D] contiguous in the last dim; freqs: complex [T, rope_dim // 2], or with
    ``positions`` ([T] int) the whole table that row t reads at ``positions[t]``.
    """
    x = x.contiguous()
    if compressed_kv:
        block_size = 16
    assert x.shape[-1] % block_size == 0
    assert rope_dim % 2 == 0 and rope_dim <= x.shape[-1]
    d = x.shape[-1]
    x2 = x.reshape(-1, d)
    rows = x2.shape[0]
    out = torch.empty_like(x)
    if rows == 0:
        return out
    rows_per_token = rows // x.shape[0]
    f_real = torch.view_as_real(freqs.contiguous()).contiguous()
    if positions is not None:
        assert positions.dim() == 1 and positions.shape[0] == x.shape[0], (
            positions.shape,
            x.shape,
        )
        positions = positions.contiguous()
    _rope_tail_fake_quant_fp4_kernel[(rows,)](
        x2,
        f_real,
        positions if positions is not None else f_real,
        out.reshape(-1, d),
        x2.stride(0),
        d,
        f_real.stride(0),
        rows_per_token,
        f_real.shape[0],
        D=d,
        RD=rope_dim,
        BLK=block_size,
        AMAX_FLOOR=FP4_AMAX_FLOOR,
        INVERSE=inverse,
        COMPRESSED_KV=compressed_kv,
        HAS_POS=positions is not None,
        num_warps=4,
    )
    return out
