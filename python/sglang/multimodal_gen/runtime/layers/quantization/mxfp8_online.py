# SPDX-License-Identifier: Apache-2.0
"""Online MXFP8 for diffusion linears: ``--quantization mxfp8`` on a bf16 checkpoint.

Weights quantize at load to e4m3 with one E8M0 scale per 32 elements along K
(FlashInfer's block quant, scales in the 128x4 swizzled layout); activations
take the same block quant per call; the GEMM is cuBLASLt's block-scaled
``torch.nn.functional.scaled_mm``. ``apply`` also takes a prequantized
``(fp8, swizzled scales)`` tuple from a fused producer. Layers with K not a
multiple of 32, N not a multiple of 16, or fp32 params keep the per-channel
fp8 path.
"""

from typing import Optional

import torch
from torch.nn import Module
from torch.nn.functional import ScalingType, SwizzleType, scaled_mm

from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8LinearMethod

_E8M0 = torch.float8_e8m0fnu


def mxfp8_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[rows, K] bf16 -> (e4m3 [rows, K], E8M0 scales in the swizzled layout)."""
    from flashinfer import mxfp8_quantize as _quantize

    q, scale = _quantize(x.contiguous(), True)
    return q, scale.view(_E8M0)


def mxfp8_scaled_mm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    bias: Optional[torch.Tensor],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    return scaled_mm(
        a,
        weight.t(),
        scale_a=a_scale,
        scale_recipe_a=ScalingType.BlockWise1x32,
        swizzle_a=SwizzleType.SWIZZLE_32_4_4,
        scale_b=weight_scale,
        scale_recipe_b=ScalingType.BlockWise1x32,
        swizzle_b=SwizzleType.SWIZZLE_32_4_4,
        bias=bias,
        output_dtype=output_dtype,
    )


class MXFP8OnlineLinearMethod(Fp8LinearMethod):
    def __init__(self, quant_config) -> None:
        super().__init__(quant_config)
        # online: bf16 weights load as-is and quantize here; the inherited
        # method must not expect block-serialized scales
        self.block_quant = False

    def process_weights_after_loading(self, layer: Module) -> None:
        layer.mxfp8 = (
            not self.use_marlin
            and layer.weight.dtype in (torch.bfloat16, torch.float16)
            and layer.weight.shape[1] % 32 == 0
            and layer.weight.shape[0] % 16 == 0
        )
        if not layer.mxfp8:
            super().process_weights_after_loading(layer)
            return
        qweight, scale = mxfp8_quantize(layer.weight.data)
        layer.weight = torch.nn.Parameter(qweight, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
        layer.input_scale = None

    def accepts_mxfp8_input(self, layer: Module) -> bool:
        return bool(layer.mxfp8)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not layer.mxfp8:
            return super().apply(layer, x, bias)
        if isinstance(x, tuple):
            a, a_scale = x
            lead, output_dtype = a.shape[:-1], torch.bfloat16
        else:
            lead, output_dtype = x.shape[:-1], x.dtype
            a, a_scale = mxfp8_quantize(x.reshape(-1, x.shape[-1]))
        out = mxfp8_scaled_mm(
            a.reshape(-1, a.shape[-1]),
            a_scale,
            layer.weight,
            layer.weight_scale,
            bias=bias,
            output_dtype=output_dtype,
        )
        return out.view(*lead, out.shape[-1])


__all__ = ["MXFP8OnlineLinearMethod", "mxfp8_quantize", "mxfp8_scaled_mm"]
