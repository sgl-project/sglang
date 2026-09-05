# SPDX-License-Identifier: Apache-2.0
"""Online MXFP8 for diffusion linears: ``--quantization mxfp8`` on a bf16 checkpoint.

Weights quantize at load to e4m3 with one E8M0 scale per 32 elements along K
(scales in the cuBLASLt 128x4 swizzled layout, ``mxfp8_quantize_swizzled``);
activations take the same block quant per call unless the producer hands over
a prequantized ``(fp8, swizzled scales)`` tuple; the GEMM is cuBLASLt's
block-scaled ``torch.nn.functional.scaled_mm``. Layers with K not a multiple
of 32, N not a multiple of 16, fp32 params, or pre-Blackwell GPUs keep the
per-channel fp8 path.
"""

from typing import Optional

import torch
from torch.nn import Module
from torch.nn.functional import ScalingType, SwizzleType, scaled_mm

from sglang.kernels.ops.diffusion import mxfp8_quantize_swizzled
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8LinearMethod

_E8M0 = torch.float8_e8m0fnu


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
        # SRTFp8Config(use_mxfp8) sets a block size; the fallback path loads bf16
        self.block_quant = False

    def process_weights_after_loading(self, layer: Module) -> None:
        layer.mxfp8 = (
            not self.use_marlin
            and layer.weight.is_cuda
            and torch.cuda.get_device_capability(layer.weight.device)[0] >= 10
            and layer.weight.dtype in (torch.bfloat16, torch.float16)
            and layer.weight.shape[1] % 32 == 0
            and layer.weight.shape[0] % 16 == 0
        )
        if not layer.mxfp8:
            super().process_weights_after_loading(layer)
            return
        qweight, scale = mxfp8_quantize_swizzled(layer.weight.data.contiguous())
        layer.weight = torch.nn.Parameter(qweight, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale.view(_E8M0), requires_grad=False)
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
            a_scale = a_scale.view(_E8M0)
            lead, output_dtype = a.shape[:-1], torch.bfloat16
        else:
            lead, output_dtype = x.shape[:-1], x.dtype
            a, a_scale = mxfp8_quantize_swizzled(
                x.reshape(-1, x.shape[-1]).contiguous()
            )
            a_scale = a_scale.view(_E8M0)
        out = mxfp8_scaled_mm(
            a.reshape(-1, a.shape[-1]),
            a_scale,
            layer.weight,
            layer.weight_scale,
            bias=bias,
            output_dtype=output_dtype,
        )
        return out.view(*lead, out.shape[-1])


__all__ = ["MXFP8OnlineLinearMethod", "mxfp8_scaled_mm"]
