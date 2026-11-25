# SPDX-License-Identifier: Apache-2.0
"""Conv bias + residual add in one pass, bit-exact against aten's two adds.

``(y + bias.view(1, C, 1, ...)) + h`` for a conv output ``y`` produced without its
bias: aten adds bf16 tensors in fp32 and rounds once per op, and so does the
kernel, so the fused result equals the eager chain bit for bit while the bias
pass's read + write of ``y`` disappears. Supports channels_last-dense tensors
(``C % 8 == 0``) and NCHW / NCDHW contiguous tensors (spatial ``% 8 == 0``);
``h`` must share ``y``'s shape and strides. bf16, CUDA only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.diffusion.norm.channel_rmsnorm_silu_nhwc_jit import (
    is_channels_last_dense,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_VEC = 8
_ALIGN = 2 * _VEC


@cache_once
def _jit_module(channels_inner: bool) -> Module:
    args = make_cpp_args(channels_inner)
    return load_jit(
        "bias_residual_add",
        *args,
        cuda_files=["diffusion/bias_residual_add.cuh"],
        cuda_wrappers=[("run", f"bias_residual_add::Kernel<{args}>::run")],
    )


def _same_layout(y: torch.Tensor, h: torch.Tensor) -> bool:
    # strides of size-1 dims are arbitrary (a squeezed frame dim differs between
    # a permuted view and a fresh tensor) and do not change the element order
    return all(
        hs == ys for size, hs, ys in zip(y.shape, h.stride(), y.stride()) if size > 1
    )


def _layout(y: torch.Tensor, bias: torch.Tensor, h: torch.Tensor) -> str | None:
    if not (
        y.is_cuda
        and torch.version.hip is None
        and not torch.compiler.is_compiling()
        and y.ndim in (4, 5)
        and y.numel() > 0
        and y.dtype is torch.bfloat16
        and h.dtype is torch.bfloat16
        and bias.dtype is torch.bfloat16
        and h.device == y.device
        and bias.device == y.device
        and h.shape == y.shape
        and _same_layout(y, h)
        and bias.numel() == y.shape[1]
        and bias.is_contiguous()
        and y.data_ptr() % _ALIGN == 0
        and h.data_ptr() % _ALIGN == 0
        and bias.data_ptr() % _ALIGN == 0
    ):
        return None
    channels = y.shape[1]
    if is_channels_last_dense(y):
        return "inner" if channels % _VEC == 0 else None
    if y.is_contiguous():
        spatial = y.numel() // (y.shape[0] * channels)
        return "outer" if spatial % _VEC == 0 else None
    return None


def can_use_bias_residual_add(
    y: torch.Tensor, bias: torch.Tensor, h: torch.Tensor
) -> bool:
    return _layout(y, bias, h) is not None


def _fake(y, bias, h):
    return torch.empty_like(y)


@register_custom_op(op_name="bias_residual_add", mutates_args=[], fake_impl=_fake)
def bias_residual_add(
    y: torch.Tensor, bias: torch.Tensor, h: torch.Tensor
) -> torch.Tensor:
    """``(y + bias) + h`` with aten's per-op bf16 rounding; output keeps ``y``'s strides."""
    layout = _layout(y, bias, h)
    assert layout is not None
    out = torch.empty_like(y)
    channels = y.shape[1]
    if layout == "inner":
        dims = (0, *range(2, y.ndim), 1)
        flat = lambda t: t.permute(*dims).reshape(-1)  # noqa: E731
        spatial = 1
    else:
        flat = lambda t: t.view(-1)  # noqa: E731
        spatial = y.numel() // (y.shape[0] * channels)
    with torch.cuda.device(y.device):
        _jit_module(layout == "inner").run(
            flat(out), flat(y), flat(h), bias.reshape(-1), channels, spatial
        )
    return out


__all__ = ["bias_residual_add", "can_use_bias_residual_add"]
