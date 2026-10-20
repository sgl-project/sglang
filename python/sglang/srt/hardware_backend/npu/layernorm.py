# SPDX-License-Identifier: Apache-2.0
"""Shared Ascend normalization and dynamic INT8 fusion."""

from functools import lru_cache

import torch


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


def try_add_rmsnorm_int8(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float | None,
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

    if eps is None:
        eps = torch.finfo(x.dtype).eps
    data, _, residual_out, scales, _ = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
        x, residual, weight, epsilon=eps
    )
    return _INT8Activation(data, scales, x.dtype), residual_out
