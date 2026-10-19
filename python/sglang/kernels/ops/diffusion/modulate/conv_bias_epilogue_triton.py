# SPDX-License-Identifier: Apache-2.0
"""Bit-exact conv bias epilogue for channels_last VAE decoders.

PyTorch's cuDNN convolution path does not fuse the bias: ``F.conv{2,3}d``
runs ``cudnn_convolution`` and then ``output.add_(bias.view(1, C, 1, ...))``
as a separate broadcast kernel. On a channels_last(_3d) output that broadcast
add is not vectorised by aten (measured 1.5 TB/s on H200 for a
``[1, 96, 4, 480, 832]`` bf16 tensor, against 4.1 TB/s for this kernel), and
in the Wan 2.1 VAE decoder it costs ~10% of the whole decode.

Two entry points, both exact replicas of the aten arithmetic:

- ``conv_bias_epilogue(x, bias)`` computes ``x.dtype(float(x) + float(bias))``
  per element: one fp32 add, one rounding, exactly ``x.add_(bias)``.
- ``conv_bias_epilogue(x, bias, residual)`` additionally adds the residual the
  way the eager ``conv(x) + h`` chain does: ``x.dtype(x.dtype(x + bias) + h)``,
  i.e. the intermediate is rounded to ``x.dtype`` before the second add, so
  the result is bitwise identical to the two-op chain while reading ``x`` and
  ``h`` once and writing once.

``bias`` may be fp32 while ``x`` is half precision (the autocast case); it is
cast to ``x.dtype`` first, which is what autocast does before calling the conv.
Layout: the output is ``empty_like(x)`` so it keeps ``x``'s dense
channels_last(_3d) strides, exactly like the in-place aten ``add_``.

Verified (``torch.equal`` vs aten): ``[1, 96, 4, 480, 832]``,
``[1, 192, 4, 240, 416]``, ``[1, 384, 2, 120, 208]``, ``[1, 384, 1, 60, 104]``
bf16 and the 4D ``[4, 96, 480, 832]`` conv2d output; end-to-end inside the
Wan 2.1 decoder (81 frames, 480x832).
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

_MAX_INT32 = 2**31 - 1
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@triton.jit
def _conv_bias_epilogue_kernel(
    x_ptr,
    bias_ptr,
    res_ptr,
    out_ptr,
    total,
    C,
    HAS_RES: tl.constexpr,
    IDX64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if IDX64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    # Dense channels_last: the channel is the fastest-varying index.
    c = offs % C
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask).to(tl.float32)
    # aten add_: fp32 opmath, one rounding to x.dtype.
    y = (x + b).to(out_ptr.dtype.element_ty)
    if HAS_RES:
        # Eager ``conv_out + h`` rounds the biased conv output first, then
        # adds the residual with a second rounding.
        h = tl.load(res_ptr + offs, mask=mask).to(tl.float32)
        y = (y.to(tl.float32) + h).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs, y, mask=mask)


def _dense_channels_last(x: torch.Tensor) -> bool:
    """Dense NHWC / NDHWC with ``C > 1`` and canonical strides on every dim
    (``C == 1`` is also NCHW-contiguous, and the kernel adds the channel index
    unscaled, so it needs ``stride(C) == 1`` unambiguously)."""
    if x.dim() == 4:
        n, c, h, w = x.shape
        return c > 1 and x.stride() == (h * w * c, 1, w * c, c)
    if x.dim() == 5:
        n, c, t, h, w = x.shape
        return c > 1 and x.stride() == (t * h * w * c, 1, h * w * c, w * c, c)
    return False


def _bias_ok(x: torch.Tensor, bias: torch.Tensor) -> bool:
    return (
        isinstance(bias, torch.Tensor)
        and bias.is_cuda
        and bias.device == x.device
        and bias.numel() == x.shape[1]
        and (bias.dtype == x.dtype or bias.dtype == torch.float32)
    )


def can_use_conv_bias_epilogue(
    x: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> bool:
    """True when ``x.add_(bias)`` (optionally ``+ residual``) can be replaced by
    the fused kernel with identical values and layout. Never raises."""
    return (
        isinstance(x, torch.Tensor)
        and x.is_cuda
        and not (
            torch.is_grad_enabled()
            and (x.requires_grad or (residual is not None and residual.requires_grad))
        )
        and x.dim() in (4, 5)
        and x.numel() > 0
        and x.dtype in _SUPPORTED_DTYPES
        and _dense_channels_last(x)
        and _bias_ok(x, bias)
        and (
            residual is None
            or (
                isinstance(residual, torch.Tensor)
                and residual.device == x.device
                and residual.dtype == x.dtype
                and residual.shape == x.shape
                and residual.stride() == x.stride()
            )
        )
    )


def conv_bias_epilogue(
    x: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x.dtype(x + bias)`` or ``x.dtype(x.dtype(x + bias) + residual)`` on a
    dense channels_last tensor, bit-exact vs aten. Raises on unsupported input;
    guard with :func:`can_use_conv_bias_epilogue`."""
    if not can_use_conv_bias_epilogue(x, bias, residual):
        raise ValueError(
            "unsupported input for conv_bias_epilogue: needs a CUDA dense "
            f"channels_last bf16/fp16/fp32 4D/5D tensor with C > 1, got shape "
            f"{tuple(x.shape)} strides {tuple(x.stride())} dtype {x.dtype}"
        )
    # Same cast autocast applies to the bias before the conv call.
    bias = bias.reshape(-1).to(x.dtype).contiguous()
    out = torch.empty_like(x)
    total = out.numel()
    BLOCK = 4096
    grid = (triton.cdiv(total, BLOCK),)
    with torch.get_device_module().device(x.device):
        _conv_bias_epilogue_kernel[grid](
            x,
            bias,
            x if residual is None else residual,
            out,
            total,
            x.shape[1],
            HAS_RES=residual is not None,
            IDX64=total >= _MAX_INT32,
            BLOCK=BLOCK,
        )
    return out
