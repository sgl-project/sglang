from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_HIDDEN = 3072
_ALIGN = 32


def _aligned(t: torch.Tensor) -> bool:
    return t.data_ptr() % _ALIGN == 0


def _blackwell_or_newer(device: torch.device) -> bool:
    return (
        torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] >= 10
    )


def _qwen_activation(t, like=None) -> bool:
    return (
        isinstance(t, torch.Tensor)
        and t.is_cuda
        and t.dtype == torch.bfloat16
        and t.ndim == 3
        and t.shape[0] == 1
        and t.shape[-1] == _HIDDEN
        and t.numel() > 0
        and t.is_contiguous()
        and _aligned(t)
        and (like is None or (t.device == like.device and t.shape == like.shape))
    )


def _row_bf16(t, device: torch.device):
    if (
        not isinstance(t, torch.Tensor)
        or t.dtype != torch.bfloat16
        or not t.is_cuda
        or t.device != device
        or t.ndim < 1
        or t.stride(-1) != 1
    ):
        return None
    if t.shape == (_HIDDEN,):
        row = t
    elif t.shape in ((1, _HIDDEN), (1, 1, _HIDDEN)):
        row = t.reshape(_HIDDEN)
    else:
        return None
    return row if _aligned(row) else None


@cache_once
def norm_scale_shift_module() -> Module:
    return load_jit(
        "qwen_image_norm_scale_shift_native",
        cuda_files=["diffusion/norm_scale_shift.cuh"],
        cuda_wrappers=[
            (
                "qwen_image_nss_bf16_row",
                "sglang_norm_scale_shift::QwenImageNormScaleShiftKernel::run",
            ),
            (
                "qwen_image_srnss_bf16_row",
                "sglang_norm_scale_shift::"
                "QwenImageScaleResidualNormScaleShiftKernel::run",
            ),
        ],
    )


_module = norm_scale_shift_module


def try_fused_norm_scale_shift(x, weight, bias, scale, shift, norm_type, eps):
    if norm_type != "layer" or weight is not None or bias is not None:
        return None
    if not _qwen_activation(x) or not _blackwell_or_newer(x.device):
        return None

    scale = _row_bf16(scale, x.device)
    shift = _row_bf16(shift, x.device)
    if scale is None or shift is None:
        return None

    y = torch.empty_like(x)
    _module().qwen_image_nss_bf16_row(
        y.view(-1, _HIDDEN), x.view(-1, _HIDDEN), scale, shift, float(eps)
    )
    return y


def try_fused_scale_residual_norm_scale_shift(
    residual, x, gate, weight, bias, scale, shift, norm_type, eps
):
    if norm_type != "layer" or weight is not None or bias is not None:
        return None
    if not (
        _qwen_activation(x)
        and _qwen_activation(residual, x)
        and _blackwell_or_newer(x.device)
    ):
        return None

    gate = _row_bf16(gate, x.device)
    scale = _row_bf16(scale, x.device)
    shift = _row_bf16(shift, x.device)
    if gate is None or scale is None or shift is None:
        return None

    y = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    _module().qwen_image_srnss_bf16_row(
        y.view(-1, _HIDDEN),
        residual_out.view(-1, _HIDDEN),
        residual.view(-1, _HIDDEN),
        x.view(-1, _HIDDEN),
        gate,
        scale,
        shift,
        float(eps),
    )
    return y, residual_out
