"""Fused SAM window-unpartition and residual-add kernel wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_window_unpartition_add_module(hidden_size: int) -> Module:
    args = make_cpp_args(hidden_size, is_arch_support_pdl())
    return load_jit(
        "window_unpartition_add",
        *args,
        cuda_files=["elementwise/window_unpartition_add.cuh"],
        cuda_wrappers=[("window_unpartition_add", f"window_unpartition_add<{args}>")],
    )


@cache_once
def _platform_supported() -> bool:
    from sglang.kernels.selector import select_kernel
    from sglang.kernels.spec import PlatformInfo

    spec = select_kernel("elementwise.window_unpartition_add")
    return spec.is_available(PlatformInfo.detect())


def can_use_window_unpartition_add(
    windows: torch.Tensor, shortcut: torch.Tensor, window_size: int
) -> bool:
    """Whether ``window_unpartition_add`` accepts these inputs; else use eager ops."""
    if not (_platform_supported() and windows.is_cuda and shortcut.is_cuda):
        return False
    if windows.dtype != torch.bfloat16 or shortcut.dtype != torch.bfloat16:
        return False
    if windows.dim() != 4 or shortcut.dim() != 4 or not windows.is_contiguous():
        return False
    batch, height, width, hidden = shortcut.shape
    per_image = -(-height // window_size) * -(-width // window_size)
    if hidden % 8 != 0 or windows.shape != (
        batch * per_image,
        window_size,
        window_size,
        hidden,
    ):
        return False
    _, stride_h, stride_w, stride_c = shortcut.stride()
    bhwc = stride_c == 1 and stride_w == hidden and stride_h == width * hidden
    width_major = stride_w == 1 and stride_h == width
    # Flat element indices are uint32.
    return (bhwc or width_major) and shortcut.numel() < 2**32


def window_unpartition_add(
    windows: torch.Tensor,
    shortcut: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """Restore SAM windows and add the residual without padded intermediates.

    ``windows`` is ``[num_windows, window_size, window_size, C]`` and
    ``shortcut`` ``[B, H, W, C]`` (BHWC or BCHW-permuted), bf16 on CUDA with
    ``C % 8 == 0``; the output keeps ``shortcut``'s strides.
    """
    output = torch.empty_strided(
        shortcut.shape, shortcut.stride(), dtype=shortcut.dtype, device=shortcut.device
    )
    module = _jit_window_unpartition_add_module(shortcut.shape[-1])
    module.window_unpartition_add(output, windows, shortcut, window_size)
    return output
