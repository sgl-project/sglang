# SPDX-License-Identifier: Apache-2.0
"""Bit-exact fused rotate-half RoPE for bf16 ``(B, S, H, D)`` activations.

Replaces the eager ERNIE-Image per-projection chain

    ``cos/sin -> chunk -> cat(-x2, x1) -> two muls + add -> cat(tail)``

(~7 kernels per q/k, including two full-width concats) with one Triton
kernel, reproducing every aten bf16 rounding boundary bit for bit:

- ``out[i]        = round(round(x1 * cos1) + round(-x2 * sin1))``
- ``out[i + R/2]  = round(round(x2 * cos2) + round( x1 * sin2))``
- columns past the rotary span are copied through unchanged (the eager
  path concatenates them back untouched).

``cos``/``sin`` are precomputed once per forward as ``(B * S, rot_dim)``
bf16 rows — the same values the eager chain materializes per layer via
``torch.cos(freqs).to(dtype)`` — so the per-layer trigonometry disappears
as well.  Negation, the fp32 products and the single-rounded add match
aten elementwise semantics exactly (no reductions are involved), which is
what makes a lossless default-on mount possible; callers still verify the
first call against the eager chain and fall back on any mismatch (see
``ernie_image.py``).
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.kernels.ops.diffusion.triton.numerics import round_bf16_to_fp32
from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _rope_rotate_half_kernel(
    out_ptr,
    x_ptr,
    cos_ptr,
    sin_ptr,
    heads,
    D: tl.constexpr,
    ROT: tl.constexpr,
    HALF: tl.constexpr,
    H_BLOCK: tl.constexpr,
    HALF_BLOCK: tl.constexpr,
    TAIL_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)  # one program per (batch, seq) row
    base = row * heads * D
    hs = tl.arange(0, H_BLOCK)[:, None]
    hmask = hs < heads
    cols = tl.arange(0, HALF_BLOCK)[None, :]
    cmask = cols < HALF
    m = hmask & cmask

    off1 = base + hs * D + cols
    off2 = off1 + HALF
    x1 = tl.load(x_ptr + off1, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + off2, mask=m, other=0.0).to(tl.float32)
    cos1 = tl.load(cos_ptr + row * ROT + cols, mask=cmask, other=0.0).to(tl.float32)
    cos2 = tl.load(cos_ptr + row * ROT + HALF + cols, mask=cmask, other=0.0).to(
        tl.float32
    )
    sin1 = tl.load(sin_ptr + row * ROT + cols, mask=cmask, other=0.0).to(tl.float32)
    sin2 = tl.load(sin_ptr + row * ROT + HALF + cols, mask=cmask, other=0.0).to(
        tl.float32
    )

    # Each product is rounded to bf16 like the eager mul; the store rounds
    # the fp32 add exactly once, like the eager add.
    out1 = round_bf16_to_fp32(x1 * cos1) + round_bf16_to_fp32(-x2 * sin1)
    out2 = round_bf16_to_fp32(x2 * cos2) + round_bf16_to_fp32(x1 * sin2)
    tl.store(out_ptr + off1, out1, mask=m)
    tl.store(out_ptr + off2, out2, mask=m)

    if D > ROT:
        tcols = ROT + tl.arange(0, TAIL_BLOCK)[None, :]
        tmask = hmask & (tcols < D)
        toff = base + hs * D + tcols
        tail = tl.load(x_ptr + toff, mask=tmask, other=0.0)
        tl.store(out_ptr + toff, tail, mask=tmask)


def can_use_fused_rope_rotate_half(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
    if x.dtype is not torch.bfloat16 or not x.is_cuda:
        return False
    if x.dim() != 4 or not x.is_contiguous():
        return False
    rows = x.shape[0] * x.shape[1]
    rot = cos.shape[-1]
    return (
        cos.dtype is torch.bfloat16
        and sin.dtype is torch.bfloat16
        and cos.is_cuda
        and cos.device == x.device
        and sin.device == x.device
        and cos.shape == (rows, rot)
        and sin.shape == (rows, rot)
        and cos.is_contiguous()
        and sin.is_contiguous()
        and rot % 2 == 0
        and 0 < rot <= x.shape[-1]
    )


def _fake_rope_rotate_half(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(x)


@register_custom_op(
    op_name="triton_fused_rope_rotate_half_bitexact",
    mutates_args=[],
    fake_impl=_fake_rope_rotate_half,
)
def fused_rope_rotate_half_bitexact(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Rotate-half RoPE over the leading ``cos.shape[-1]`` columns of ``x``.

    ``x`` is ``(B, S, H, D)``; ``cos``/``sin`` are ``(B * S, rot_dim)`` rows.
    Bit-exact vs the eager chunk/neg/cat/mul/add chain.
    """
    batch, seq_len, heads, head_dim = x.shape
    rot = cos.shape[-1]
    half = rot // 2
    out = torch.empty_like(x)
    tail = head_dim - rot
    with torch.cuda.device(x.device):
        _rope_rotate_half_kernel[(batch * seq_len,)](
            out,
            x,
            cos,
            sin,
            heads,
            D=head_dim,
            ROT=rot,
            HALF=half,
            H_BLOCK=triton.next_power_of_2(heads),
            HALF_BLOCK=triton.next_power_of_2(half),
            TAIL_BLOCK=triton.next_power_of_2(max(tail, 1)),
        )
    return out
