# SPDX-License-Identifier: Apache-2.0
"""Native Ascend RMSNorm for the MiniMax H3 transformer."""

import torch
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm


class MiniMaxH3RMSNorm(RMSNorm):
    def __init__(
        self,
        normalized_shape,
        eps=None,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ):
        norm = nn.RMSNorm(normalized_shape, eps, elementwise_affine, device, dtype)
        super().__init__(norm.normalized_shape[-1], eps=eps, weight_dtype=dtype)
        self.weight = norm.weight
        self.normalized_shape = norm.normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is not None:
            quantized = self._forward_npu_quant(x, residual)
            if quantized is not None:
                return quantized
            # Preserve the refiner's original low-precision residual addition.
            x = residual = residual + x
        if (
            x.device.type == "npu"
            and x.dtype in (torch.float16, torch.bfloat16)
            and self.weight is not None
            and self.weight.device == x.device
            and self.weight.dtype == x.dtype
            and x.numel() > 0
            and hasattr(torch.ops.npu, "npu_rms_norm")
        ):
            eps = self.eps if self.eps is not None else torch.finfo(x.dtype).eps
            out = torch.ops.npu.npu_rms_norm(x, self.weight, eps)[0]
        else:
            out = torch.nn.functional.rms_norm(
                x, self.normalized_shape, self.weight, self.eps
            )
        return out if residual is None else (out, residual)
