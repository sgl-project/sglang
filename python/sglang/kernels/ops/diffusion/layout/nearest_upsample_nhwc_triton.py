# SPDX-License-Identifier: Apache-2.0
"""Bit-exact channels_last nearest upsample for the Wan-family VAE decoders.

``nn.Upsample(scale_factor=2, mode="nearest-exact")`` on a channels_last
(NHWC) input dispatches to aten's ``upsample_nearest2d_nhwc_out_frame``. On an
H200 with a ``[1, 192, 240, 416]`` bf16 input that kernel takes 0.458 ms
against 0.190 ms for aten's own NCHW kernel on the same bytes; this gather
takes 0.061 ms. The Wan / Qwen-Image VAE decoders hit the aten NHWC kernel
once per up block per chunk as soon as they run channels_last end-to-end.

Numerical contract: bit-exact vs ``nn.Upsample`` in ``nearest`` and
``nearest-exact`` mode for integer scale factors. Write an output index as
``i = k * f + r`` with ``0 <= r <= f - 1``. ``nearest`` reads ``floor(i / f)
= k``; ``nearest-exact`` reads ``floor((i + 0.5) / f) = floor(k + (r + 0.5) /
f) = k`` because ``(r + 0.5) / f < 1``. Both therefore read input ``i // f``,
so the op is a pure gather that never touches a value, and the result is
bitwise identical for any dtype the predicate admits (bf16 / fp16 / fp32, the
ones the tests cover). The kernel walks the output in its NHWC memory order,
so the stores and the gathered loads are both contiguous along ``C``.

Verified (``torch.equal`` vs ``F.interpolate``): ``[1, 192, 240, 416]``,
``[4, 192, 120, 208]``, ``[1, 96, 480, 832]``, ``[1, 3, 5, 7]`` at factor 2
and ``[2, 3, 5, 7]`` at factor ``(3, 2)`` for all three dtypes; end-to-end
inside the Wan 2.1 (81 frames, 480x832) and Qwen-Image (1024x1024) decoders.
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

_MAX_INT32 = 2**31 - 1
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@triton.jit
def _nearest_upsample_nhwc_kernel(
    x_ptr,
    out_ptr,
    total,
    C,
    out_h,
    out_w,
    fh,
    fw,
    sxn,
    sxh,
    sxw,
    IDX64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if IDX64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    # Output is dense NHWC: offs = ((n * out_h + h) * out_w + w) * C + c.
    c = offs % C
    t = offs // C
    w = t % out_w
    t = t // out_w
    h = t % out_h
    n = t // out_h
    src = n * sxn + (h // fh) * sxh + (w // fw) * sxw + c
    vals = tl.load(x_ptr + src, mask=mask)
    tl.store(out_ptr + offs, vals, mask=mask)


def _integer_scale(scale) -> tuple[int, int] | None:
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    if not isinstance(scale, (tuple, list)) or len(scale) != 2:
        return None
    out = []
    for s in scale:
        f = float(s)
        if f < 1.0 or f != int(f):
            return None
        out.append(int(f))
    return out[0], out[1]


def can_use_nearest_upsample_nhwc(x: torch.Tensor, scale_factor, mode: str) -> bool:
    """True when ``F.interpolate(x, scale_factor=..., mode=...)`` is a plain
    integer-factor gather on a dense channels_last 4D tensor."""
    return (
        isinstance(x, torch.Tensor)
        and x.is_cuda
        and not torch.is_grad_enabled()
        and not x.requires_grad
        and mode in ("nearest", "nearest-exact")
        and x.dim() == 4
        and x.numel() > 0
        and x.dtype in _SUPPORTED_DTYPES
        # Dense NHWC. For C > 1 this fixes stride(C) == 1, which the kernel
        # relies on by adding the channel index unscaled; for C == 1 that index
        # is always 0, so no separate stride check is needed.
        and x.is_contiguous(memory_format=torch.channels_last)
        and _integer_scale(scale_factor) is not None
    )


def nearest_upsample_nhwc(x: torch.Tensor, scale_factor) -> torch.Tensor:
    """Integer-factor nearest upsample of a channels_last ``[N, C, H, W]``
    tensor, returned dense channels_last. Bit-exact vs ``nn.Upsample`` in
    ``nearest`` and ``nearest-exact`` modes; raises on unsupported input."""
    factors = _integer_scale(scale_factor)
    if factors is None:
        raise ValueError(f"scale_factor must be integer-valued, got {scale_factor}")
    if not (
        x.is_cuda
        and x.dim() == 4
        and x.is_contiguous(memory_format=torch.channels_last)
    ):
        raise ValueError("nearest_upsample_nhwc needs a CUDA channels_last 4D tensor")
    fh, fw = factors
    n, c, h, w = x.shape
    out_h, out_w = h * fh, w * fw
    out = torch.empty(
        (n, c, out_h, out_w),
        device=x.device,
        dtype=x.dtype,
        memory_format=torch.channels_last,
    )
    total = out.numel()
    if total == 0:
        return out
    sxn, _, sxh, sxw = x.stride()
    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
    with torch.get_device_module().device(x.device):
        _nearest_upsample_nhwc_kernel[grid](
            x,
            out,
            total,
            c,
            out_h,
            out_w,
            fh,
            fw,
            sxn,
            sxh,
            sxw,
            IDX64=total >= _MAX_INT32 or x.numel() >= _MAX_INT32,
            BLOCK=BLOCK,
        )
    return out
