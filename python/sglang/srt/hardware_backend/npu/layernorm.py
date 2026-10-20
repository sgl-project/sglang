# SPDX-License-Identifier: Apache-2.0
"""Shared Ascend normalization and dynamic INT8 fusion."""

from abc import ABC, abstractmethod
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


class BaseNormalization(ABC):
    @abstractmethod
    def _apply_norm(self, *args, **kwargs): ...


class NPURMSNorm(BaseNormalization):
    def _apply_norm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float | None,
        residual: torch.Tensor | None = None,
        *,
        preserve_residual_rounding: bool = False,
    ):
        if eps is None:
            eps = torch.finfo(x.dtype).eps
        if residual is not None:
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


class NPURMSNormQuant(NPURMSNorm):
    @staticmethod
    def is_supported(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor | None
    ) -> bool:
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
            return False
        return _supports_add_rmsnorm_int8(x.device, x.dtype, x.shape[-1])

    def _apply_norm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float | None,
        residual: torch.Tensor | None = None,
        *,
        preserve_residual_rounding: bool = False,
    ):
        if residual is None or not self.is_supported(x, residual, weight):
            return super()._apply_norm(
                x,
                weight,
                eps,
                residual,
                preserve_residual_rounding=preserve_residual_rounding,
            )
        if eps is None:
            eps = torch.finfo(x.dtype).eps
        from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
            _INT8Activation,
        )

        data, _, residual_out, scales, _ = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
            x, residual, weight, epsilon=eps
        )
        return _INT8Activation(data, scales, x.dtype), residual_out


def get_rmsnorm_variant(quantize: bool = False) -> BaseNormalization:
    return NPURMSNormQuant() if quantize else NPURMSNorm()
