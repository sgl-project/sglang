# SPDX-License-Identifier: Apache-2.0
"""``--quantization fp8_per_tensor``: online fp8 with one scale per tensor.

Weights quantize per tensor at load, activations take a dynamic per-tensor
scale, the GEMM runs on cuBLASLt (``torch._scaled_mm``). ``apply`` also takes
a prequantized ``(fp8, scale)`` tuple from a fused activation + quant kernel.
Layers with unaligned K/N or fp32 params keep the per-channel path."""

from typing import Optional

import torch
from torch.nn import Module

from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
)
from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear, input_to_float8


class Fp8PerTensorConfig(Fp8Config):
    @classmethod
    def get_name(cls) -> str:
        return "fp8_per_tensor"

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        method = super().get_quant_method(layer, prefix)
        if isinstance(method, Fp8LinearMethod):
            return Fp8PerTensorLinearMethod(self)
        return method


class Fp8PerTensorLinearMethod(Fp8LinearMethod):
    def process_weights_after_loading(self, layer: Module) -> None:
        # torch._scaled_mm wants 16-aligned K and N and a bf16/fp16 activation
        layer.fp8_per_tensor = (
            not self.quant_config.is_checkpoint_fp8_serialized
            and not self.block_quant
            and not self.use_marlin
            and layer.weight.dtype in (torch.bfloat16, torch.float16)
            and layer.weight.shape[0] % 16 == 0
            and layer.weight.shape[1] % 16 == 0
        )
        if not layer.fp8_per_tensor:
            super().process_weights_after_loading(layer)
            return
        qweight, weight_scale = input_to_float8(layer.weight.data)
        layer.weight = torch.nn.Parameter(qweight.t(), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            weight_scale.reshape(1), requires_grad=False
        )
        layer.input_scale = None

    def accepts_fp8_per_tensor_input(self, layer: Module) -> bool:
        return bool(layer.fp8_per_tensor)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not layer.fp8_per_tensor:
            return super().apply(layer, x, bias)
        if isinstance(x, tuple):
            qinput, x_scale = x
            return apply_fp8_linear(
                input=qinput,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                input_scale=x_scale,
                bias=bias,
                cutlass_fp8_supported=False,
                pad_output=False,
                pre_quant_output_dtype=torch.bfloat16,
            )
        return apply_fp8_linear(
            input=x.contiguous(),
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            bias=bias,
            cutlass_fp8_supported=False,
            use_per_token_if_dynamic=False,
            pad_output=False,
            compressed_tensor_quant=True,
        )


__all__ = ["Fp8PerTensorConfig", "Fp8PerTensorLinearMethod"]
