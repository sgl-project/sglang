"""Qwen-Image native fast path for diffusion norm-scale-shift ops.

This module intentionally covers only the production Qwen-Image pattern:
bf16 activations, ``B == 1``, ``D == 3072``, layer norm, bf16 row-broadcast
scale/shift, and no affine weight/bias. Unsupported inputs return ``None`` so
the caller's original CuTe-DSL implementation handles them unchanged.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

_BF16 = torch.bfloat16
_ALIGN = 32
_HIDDEN = 3072

_ENABLED = os.environ.get("SGLANG_NSS_NATIVE_DISABLE", "0").lower() not in (
    "1",
    "true",
    "yes",
)


def set_native_enabled(enabled: bool) -> None:
    global _ENABLED
    _ENABLED = enabled


def native_enabled() -> bool:
    return _ENABLED


def _blackwell_or_newer(device: torch.device) -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 10


def _aligned(t: torch.Tensor) -> bool:
    return t.data_ptr() % _ALIGN == 0


def _row_bf16(t, B: int, S: int, D: int, device: torch.device):
    if not isinstance(t, torch.Tensor):
        return None
    if t.dtype != _BF16 or not t.is_cuda or t.device != device:
        return None
    if t.ndim >= 1 and t.stride(-1) != 1:
        return None
    if t.ndim == 1 and t.shape == (D,):
        return t if _aligned(t) else None
    if t.ndim == 2 and t.shape == (1, D):
        v = t.reshape(D)
        return v if v.is_contiguous() and _aligned(v) else None
    if t.ndim == 3 and B == 1 and t.shape == (1, 1, D):
        v = t.reshape(D)
        return v if v.is_contiguous() and _aligned(v) else None
    return None


def _activation_ok(t, D: int) -> bool:
    return (
        isinstance(t, torch.Tensor)
        and t.is_cuda
        and t.dtype == _BF16
        and t.ndim == 3
        and t.shape[0] == 1
        and t.shape[-1] == D
        and t.numel() > 0
        and t.is_contiguous()
        and _aligned(t)
    )


def _qwen_shape_ok(x: torch.Tensor) -> bool:
    return (
        x.ndim == 3
        and x.shape[0] == 1
        and x.shape[-1] == _HIDDEN
        and _activation_ok(x, _HIDDEN)
        and _blackwell_or_newer(x.device)
    )


@cache_once
def _module():
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
                "sglang_norm_scale_shift::QwenImageScaleResidualNormScaleShiftKernel::run",
            ),
        ],
    )


def try_fused_norm_scale_shift(
    x, weight, bias, scale, shift, norm_type, eps
) -> Optional[torch.Tensor]:
    if not _ENABLED or norm_type != "layer" or weight is not None or bias is not None:
        return None
    if not (isinstance(x, torch.Tensor) and _qwen_shape_ok(x)):
        return None

    B, S, D = x.shape
    sc = _row_bf16(scale, B, S, D, x.device)
    sh = _row_bf16(shift, B, S, D, x.device)
    if sc is None or sh is None:
        return None

    y = torch.empty_like(x)
    _module().qwen_image_nss_bf16_row(y.view(B * S, D), x.view(B * S, D), sc, sh, float(eps))
    return y


def try_fused_scale_residual_norm_scale_shift(
    residual, x, gate, weight, bias, scale, shift, norm_type, eps
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if not _ENABLED or norm_type != "layer" or weight is not None or bias is not None:
        return None
    if not (
        isinstance(x, torch.Tensor)
        and isinstance(residual, torch.Tensor)
        and residual.shape == x.shape
        and residual.dtype == x.dtype
        and residual.is_cuda
        and residual.device == x.device
        and _qwen_shape_ok(x)
        and _activation_ok(residual, _HIDDEN)
    ):
        return None

    B, S, D = x.shape
    g = _row_bf16(gate, B, S, D, x.device)
    sc = _row_bf16(scale, B, S, D, x.device)
    sh = _row_bf16(shift, B, S, D, x.device)
    if g is None or sc is None or sh is None:
        return None

    y = torch.empty_like(x)
    res_out = torch.empty_like(x)
    _module().qwen_image_srnss_bf16_row(
        y.view(B * S, D),
        res_out.view(B * S, D),
        residual.view(B * S, D),
        x.view(B * S, D),
        g,
        sc,
        sh,
        float(eps),
    )
    return y, res_out
