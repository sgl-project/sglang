"""Bit-exact temporal blend/write and RGB denormalization for H3 on SM100+.

Temporal blend preserves aten's reciprocal-first ramp and separate FP32
multiply/add rounding. Denormalization preserves subtraction/division rounding
and NaNs. The blend destination must not alias either input; denormalization
supports in-place writes only to contiguous, caller-owned frames.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.diffusion.common.platform import is_cuda_sm_at_least
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_MIN_SM = (10, 0)
# Must match kVecWidth / kVecAlignment in csrc/diffusion/minimax_h3_vae_output.cuh.
_VEC_WIDTH = 4
_VEC_ALIGNMENT = 16


@cache_once
def _jit_minimax_h3_vae_output_module(vectorized: bool) -> Module:
    args = make_cpp_args(vectorized)
    return load_jit(
        "diffusion_minimax_h3_vae_output",
        *args,
        cuda_files=["diffusion/minimax_h3_vae_output.cuh"],
        cuda_wrappers=[
            (
                "temporal_blend_write",
                f"minimax_h3_vae_output::TemporalBlendWriteKernel<{args}>::run",
            ),
            (
                "denorm_clamp",
                f"minimax_h3_vae_output::DenormClampKernel<{args}>::run",
            ),
        ],
    )


def _vectorizable(*tensors: torch.Tensor) -> bool:
    # Mirror of the launcher's 128-bit precondition: innermost stride 1,
    # `W % 4 == 0`, every other stride and the data pointer 16-byte aligned.
    return all(
        t.stride(-1) == 1
        and t.shape[-1] % _VEC_WIDTH == 0
        and all(s % _VEC_WIDTH == 0 for s in t.stride()[:-1])
        and t.data_ptr() % _VEC_ALIGNMENT == 0
        for t in tensors
    )


@register_custom_op(
    op_name="diffusion_minimax_h3_vae_temporal_blend_write", mutates_args=["out"]
)
def _temporal_blend_write_out(
    part: torch.Tensor,
    overlap: torch.Tensor,
    out: torch.Tensor,
    start: int,
    count: int,
    blend_extent: int,
    vectorized: bool,
) -> None:
    module = _jit_minimax_h3_vae_output_module(vectorized)
    module.temporal_blend_write(part, overlap, out, start, count, blend_extent)


@register_custom_op(
    op_name="diffusion_minimax_h3_vae_denorm_clamp", mutates_args=["out"]
)
def _denorm_clamp_out(
    value: torch.Tensor,
    out: torch.Tensor,
    mean0: float,
    mean1: float,
    mean2: float,
    std0: float,
    std1: float,
    std2: float,
    vectorized: bool,
) -> None:
    module = _jit_minimax_h3_vae_output_module(vectorized)
    module.denorm_clamp(value, out, mean0, mean1, mean2, std0, std1, std2)


def minimax_h3_vae_output_device_supported(device: torch.device | int) -> bool:
    """Whether ``device`` can run these kernels at all (CUDA SM100 or newer);
    the ``can_use_*`` predicates check the tensors on top of this."""
    return is_cuda_sm_at_least(_MIN_SM, device)


def minimax_h3_vae_temporal_blend_write(
    part: torch.Tensor,
    overlap: torch.Tensor | None,
    *,
    blend_extent: int,
    out: torch.Tensor,
    start: int,
    count: int,
) -> None:
    """Write ``part[:, :, :count]`` into ``out[:, :, start:start+count]``, blending
    its leading frames against ``overlap`` when one is given."""
    if overlap is None:
        # The launcher never reads `overlap` at extent 0; pass `part` for the shape check.
        overlap, extent = part, 0
    else:
        extent = min(int(blend_extent), overlap.shape[2], part.shape[2])
    _temporal_blend_write_out(
        part,
        overlap,
        out,
        int(start),
        int(count),
        extent,
        _vectorizable(part, overlap, out),
    )


def can_use_minimax_h3_vae_temporal_blend_write(
    part: torch.Tensor,
    overlap: torch.Tensor | None,
    *,
    blend_extent: int,
    out: torch.Tensor,
    start: int,
    count: int,
) -> bool:
    if not isinstance(part, torch.Tensor):
        return False
    tensors = (part, out) if overlap is None else (part, overlap, out)
    if not all(
        isinstance(t, torch.Tensor)
        and t.is_cuda
        and t.dtype == torch.float32
        and t.ndim == 5
        and t.device == part.device
        and t.shape[:2] == part.shape[:2]
        and t.shape[3:] == part.shape[3:]
        for t in tensors
    ):
        return False
    if not (
        blend_extent >= 0
        and 0 <= count <= part.shape[2]
        and start >= 0
        and start + count <= out.shape[2]
    ):
        return False
    out_storage = out.untyped_storage().data_ptr()
    if any(t.untyped_storage().data_ptr() == out_storage for t in tensors[:-1]):
        return False
    return (
        is_cuda_sm_at_least(_MIN_SM, part.device) and not torch.compiler.is_compiling()
    )


def minimax_h3_vae_denorm_clamp(
    value: torch.Tensor,
    *,
    mean: Sequence[float],
    std: Sequence[float],
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``clamp((value - mean[c]) / std[c], 0, 1)`` as one launch; NaN preserved.

    ``out`` is either None (a fresh contiguous buffer) or ``value`` itself,
    which rewrites a contiguous input in place.
    """
    mean = [float(v) for v in mean]
    std = [float(v) for v in std]
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("mean and std must hold one value per RGB channel")
    if out is None:
        out = torch.empty(value.shape, dtype=torch.float32, device=value.device)
    elif out is not value:
        raise ValueError("out must be None or the input tensor itself")
    _denorm_clamp_out(value, out, *mean, *std, _vectorizable(value, out))
    return out


def can_use_minimax_h3_vae_denorm_clamp(
    value: torch.Tensor,
    *,
    mean: Sequence[float],
    std: Sequence[float],
    out: torch.Tensor | None = None,
) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and value.is_cuda
        and value.dtype == torch.float32
        and value.ndim == 5
        and value.shape[1] == 3
        and value.numel() > 0
        and (out is None or (out is value and value.is_contiguous()))
        and len(mean) == 3
        and len(std) == 3
        and is_cuda_sm_at_least(_MIN_SM, value.device)
        and not torch.compiler.is_compiling()
    )


__all__ = [
    "can_use_minimax_h3_vae_denorm_clamp",
    "can_use_minimax_h3_vae_temporal_blend_write",
    "minimax_h3_vae_denorm_clamp",
    "minimax_h3_vae_output_device_supported",
    "minimax_h3_vae_temporal_blend_write",
]
