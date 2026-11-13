# SPDX-License-Identifier: Apache-2.0
"""Online MXFP8 for diffusion linears: ``--quantization mxfp8`` on a bf16 checkpoint.

Weights quantize at load to e4m3 with one E8M0 scale per 32 elements along K
(scales in the cuBLASLt 128x4 swizzled layout, ``mxfp8_quantize_swizzled``);
activations take the same block quant per call unless the producer hands over
a prequantized ``(fp8, swizzled scales)`` tuple; the GEMM is cuBLASLt's
block-scaled ``torch.nn.functional.scaled_mm``. On CUDA, layers with K not a
multiple of 32, N not a multiple of 16, non-bf16 params, or pre-Blackwell GPUs
keep the per-channel fp8 path.

On Intel XPU the bf16 weights quantize once at load and activations per call,
both with sgl-kernel's group-32 UE8M0 quantizer (row-major E8M0 [rows, K/32]
scales), and the GEMM is the block-scaled ``torch.nn.functional.scaled_mm``
(BlockWise1x32); see srt's ``xpu_mxfp8_scaled_mm_linear``. N needs no
alignment there; layers with K not a multiple of 32 or non-bf16 params keep
the per-channel fp8 path.
"""

from typing import Optional

import torch
from torch.nn import Module
from torch.nn.functional import ScalingType, SwizzleType, scaled_mm

from sglang.kernels.ops.quantization.mxfp8_swizzled_triton import (
    can_use_mxfp8_swizzled,
    mxfp8_quantize_swizzled,
)
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8LinearMethod
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.srt.layers.quantization.fp8_utils import (
    XPU_MXFP8_GROUP_SIZE,
    xpu_mxfp8_quantize,
    xpu_mxfp8_scaled_mm_linear,
)

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
        layer.xpu_mxfp8 = (
            current_platform.is_xpu()
            and layer.weight.dtype == torch.bfloat16
            and layer.weight.shape[1] % XPU_MXFP8_GROUP_SIZE == 0
        )
        if layer.xpu_mxfp8:
            qweight, scale = xpu_mxfp8_quantize(layer.weight.data)
            layer.weight = torch.nn.Parameter(qweight, requires_grad=False)
            layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
            layer.input_scale = None
            layer.mxfp8 = False
            return
        layer.mxfp8 = (
            not self.use_marlin
            and can_use_mxfp8_swizzled(layer.weight)
            and torch.cuda.get_device_capability(layer.weight.device)[0] >= 10
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
        if layer.xpu_mxfp8:
            return xpu_mxfp8_scaled_mm_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                bias=bias,
            )
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
