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

from sglang.kernels.ops.diffusion.common.numerics import round_bf16_to_fp32
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

    ``x`` is ``(B, S, H, D)``; ``cos``/``sin`` are contiguous
    ``(B * S, rot_dim)`` rows or ``(B, S, 1, rot_dim)`` broadcast tables.
    Bit-exact vs the eager chunk/neg/cat/mul/add chain.
    """
    if not (
        x.is_cuda and x.dtype is torch.bfloat16 and x.ndim == 4 and x.is_contiguous()
    ):
        raise RuntimeError("rotate-half RoPE expects contiguous BF16 CUDA [B, S, H, D]")
    batch, seq_len, heads, head_dim = x.shape
    if cos.ndim not in (2, 4):
        raise RuntimeError("cos must have shape [B * S, rot_dim] or [B, S, 1, rot_dim]")
    rot = cos.shape[-1]
    if not (0 < rot <= head_dim and rot % 2 == 0):
        raise RuntimeError("rot_dim must be positive, even and no larger than head_dim")
    table_shape = (batch * seq_len, rot) if cos.ndim == 2 else (batch, seq_len, 1, rot)
    device = x.device
    for name, tensor in (("cos", cos), ("sin", sin)):
        if not (
            tensor.dtype is torch.bfloat16
            and tensor.device == device
            and tensor.shape == table_shape
            and tensor.is_contiguous()
        ):
            raise RuntimeError(
                f"{name} must be contiguous {table_shape} with x's dtype/device"
            )
    half = rot // 2
    out = torch.empty_like(x)
    tail = head_dim - rot
    with torch.cuda.device(device):
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
