# SPDX-License-Identifier: Apache-2.0
"""Shared Ascend normalization and dynamic INT8 fusion."""

from functools import lru_cache

import torch
import torch.nn.functional as F


@lru_cache(maxsize=None)
def _supports_add_rmsnorm_int8(device: torch.device, dtype: torch.dtype, width: int):
    try:
        x = torch.zeros((1, width), device=device, dtype=dtype)
        weight = torch.ones(width, device=device, dtype=dtype)
        data, _, residual, scale, _ = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
            x, x, weight, epsilon=1e-6
        )
        return (
            data.dtype == torch.int8
            and data.shape == x.shape
            and residual.dtype == dtype
            and residual.shape == x.shape
            and scale.dtype == torch.float32
            and scale.shape == (1,)
        )
    except torch.OutOfMemoryError:
        raise
    except (AttributeError, NotImplementedError, RuntimeError):
        return False


def _try_add_rmsnorm_int8(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
):
    if (
        x.device.type != "npu"
        or x.dtype not in (torch.float16, torch.bfloat16)
        or x.ndim != 2
        or x.numel() == 0
        or residual.shape != x.shape
        or residual.dtype != x.dtype
        or residual.device != x.device
        or weight is None
        or weight.shape != (x.shape[-1],)
        or weight.dtype != x.dtype
        or weight.device != x.device
    ):
        return None
    if not _supports_add_rmsnorm_int8(x.device, x.dtype, x.shape[-1]):
        return None
    from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
        _INT8Activation,
    )

    data, _, residual_out, scales, _ = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
        x, residual, weight, epsilon=eps
    )
    return _INT8Activation(data, scales, x.dtype), residual_out


def rms_norm_npu(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float | None,
    residual: torch.Tensor | None = None,
    *,
    quantize: bool = False,
    preserve_residual_rounding: bool = False,
):
    if eps is None:
        eps = torch.finfo(x.dtype).eps
    if residual is not None:
        if quantize:
            quantized = _try_add_rmsnorm_int8(x, residual, weight, eps)
            if quantized is not None:
                return quantized
        if preserve_residual_rounding:
            # Preserve the model's low-precision residual addition.
            x = residual = residual + x
        else:
            out, _, residual_out = torch.ops.npu.npu_add_rms_norm(
                residual, x, weight, eps
            )
            return out, residual_out

    if preserve_residual_rounding and (
        x.dtype not in (torch.float16, torch.bfloat16)
        or weight.dtype != x.dtype
        or weight.device != x.device
        or x.numel() == 0
        or not hasattr(torch.ops.npu, "npu_rms_norm")
    ):
        out = F.rms_norm(x, (weight.shape[-1],), weight, eps)
    else:
        out = torch.ops.npu.npu_rms_norm(x, weight, eps)[0]
    return out if residual is None else (out, residual)
