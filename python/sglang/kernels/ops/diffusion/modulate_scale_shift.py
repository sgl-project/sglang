"""Fused adaLN modulate: ``x * (1 + scale) + shift`` in one CUDA kernel.

Numerical contract: the kernel reproduces each eager op's
fp32-opmath/round-to-storage-dtype boundary (fp16/bf16), so its output is
bit-exact vs the eager chain (``torch.equal``) and needs no quality gate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_ALIGN_BYTES = 16


@cache_once
def _jit_modulate_scale_shift_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "diffusion_modulate_scale_shift",
        *args,
        cuda_files=["diffusion/modulate_scale_shift.cuh"],
        cuda_wrappers=[
            (
                "modulate_scale_shift",
                "sglang_modulate_scale_shift::"
                f"ModulateScaleShiftKernel<{args}>::run",
            ),
        ],
    )


def _fake_impl(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    return torch.empty_like(x)


@register_custom_op(
    op_name="diffusion_modulate_scale_shift",
    mutates_args=[],
    fake_impl=_fake_impl,
)
def _modulate_scale_shift_custom_op(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    module = _jit_modulate_scale_shift_module(x.dtype)
    module.modulate_scale_shift(out, x, scale, shift)
    return out


def _aligned(t: torch.Tensor) -> bool:
    return t.data_ptr() % _ALIGN_BYTES == 0


def can_use_modulate_scale_shift_cuda(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> bool:
    if (
        x.dtype not in _SUPPORTED_DTYPES
        or scale.dtype != x.dtype
        or shift.dtype != x.dtype
        or not (x.is_cuda and scale.is_cuda and shift.is_cuda)
        or not (x.device == scale.device == shift.device)
        or x.dim() != 3
        or scale.dim() != 2
        or shift.shape != scale.shape
        or scale.shape != (x.shape[0], x.shape[-1])
        or not (x.is_contiguous() and scale.is_contiguous() and shift.is_contiguous())
        or x.numel() == 0
    ):
        return False
    vec = _ALIGN_BYTES // x.element_size()
    return (
        x.shape[-1] % vec == 0 and _aligned(x) and _aligned(scale) and _aligned(shift)
    )


def modulate_scale_shift_cuda(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    """Fused ``x * (1 + scale[:, None]) + shift[:, None]`` (bit-exact vs eager)."""
    if not can_use_modulate_scale_shift_cuda(x, scale, shift):
        raise RuntimeError("unsupported input for modulate_scale_shift CUDA")
    return _modulate_scale_shift_custom_op(x, scale, shift)
