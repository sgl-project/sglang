from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_ltx2_post_rms_modulate_module() -> Module:
    return load_jit(
        "diffusion_ltx2_post_rms_modulate",
        cuda_files=["diffusion/ltx2_post_rms_modulate.cuh"],
        cuda_wrappers=[
            (
                "ltx2_post_rms_modulate",
                "sglang_ltx2_post_rms_modulate::LTX2PostRMSModulateKernel::run",
            ),
            (
                "ltx2_post_rms_dual_modulate",
                "sglang_ltx2_post_rms_modulate::LTX2PostRMSDualModulateKernel::run",
            ),
        ],
    )


def _is_sm100_or_newer(x: torch.Tensor) -> bool:
    if not x.is_cuda:
        return False
    try:
        return torch.cuda.get_device_capability(x.device)[0] >= 10
    except RuntimeError:
        return False


def _supported_param(x: torch.Tensor, param: torch.Tensor) -> bool:
    return (
        param.is_cuda
        and param.device == x.device
        and param.dtype == torch.bfloat16
        and param.ndim == 3
        and param.shape[0] in (1, x.shape[0])
        and param.shape[1] in (1, x.shape[1])
        and param.shape[2] == x.shape[2]
        and param.stride(-1) == 1
    )


def can_use_ltx2_post_rms_modulate_cuda(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> bool:
    return (
        _is_sm100_or_newer(x)
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim == 3
        and x.is_contiguous()
        and x.shape[-1] % 8 == 0
        and _supported_param(x, scale)
        and _supported_param(x, shift)
    )


def can_use_ltx2_post_rms_dual_modulate_cuda(
    x: torch.Tensor,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
) -> bool:
    return (
        can_use_ltx2_post_rms_modulate_cuda(x, scale0, shift0)
        and _supported_param(x, scale1)
        and _supported_param(x, shift1)
    )


def _fake_single_impl(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(x)


def _fake_dual_impl(
    x: torch.Tensor,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(x)


@register_custom_op(
    op_name="diffusion_ltx2_post_rms_modulate",
    mutates_args=[],
    fake_impl=_fake_single_impl,
)
def _ltx2_post_rms_modulate_custom_op(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(x)
    _jit_ltx2_post_rms_modulate_module().ltx2_post_rms_modulate(out, x, scale, shift)
    return out


@register_custom_op(
    op_name="diffusion_ltx2_post_rms_dual_modulate",
    mutates_args=[],
    fake_impl=_fake_dual_impl,
)
def _ltx2_post_rms_dual_modulate_custom_op(
    x: torch.Tensor,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out0 = torch.empty_like(x)
    out1 = torch.empty_like(x)
    _jit_ltx2_post_rms_modulate_module().ltx2_post_rms_dual_modulate(
        out0, out1, x, scale0, shift0, scale1, shift1
    )
    return out0, out1


def ltx2_post_rms_modulate_cuda(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    if not can_use_ltx2_post_rms_modulate_cuda(x, scale, shift):
        raise RuntimeError("unsupported input for LTX2 post-RMS modulation CUDA")
    return _ltx2_post_rms_modulate_custom_op(x, scale, shift)


def ltx2_post_rms_dual_modulate_cuda(
    x: torch.Tensor,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not can_use_ltx2_post_rms_dual_modulate_cuda(x, scale0, shift0, scale1, shift1):
        raise RuntimeError("unsupported input for LTX2 post-RMS dual modulation CUDA")
    return _ltx2_post_rms_dual_modulate_custom_op(x, scale0, shift0, scale1, shift1)
