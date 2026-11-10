"""V100-oriented W8A16 storage quantization for native diffusion linears.

Weights are quantized per output channel after their TP shard has been loaded.
The forward path dequantizes only that rank-local matrix to FP16 and dispatches
the GEMM through PyTorch/cuBLAS, which uses Volta FP16 Tensor Cores.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.models.parameter import ModelWeightParameter
from torch import nn


class V100W8A16Config(QuantizationConfig):
    """Online symmetric INT8 weight storage with FP16 activations/compute."""

    requires_fp16_source = True

    def __init__(self, ignored_layers: list[str] | None = None) -> None:
        super().__init__()
        self.ignored_layers = tuple(ignored_layers or ())

    @classmethod
    def get_name(cls) -> str:
        return "v100_w8a16"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "V100W8A16Config":
        return cls(
            ignored_layers=cls.get_from_keys_or(
                config, ["ignored_layers", "modules_to_not_convert"], None
            )
        )

    def get_quant_method(
        self, layer: nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        if any(pattern in prefix for pattern in self.ignored_layers):
            return UnquantizedLinearMethod()
        return V100W8A16LinearMethod(self)


class V100W8A16LinearMethod(LinearMethodBase):
    """Per-row W8 storage followed by rank-local FP16 dequantization."""

    def __init__(self, quant_config: V100W8A16Config) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        if params_dtype != torch.float16:
            raise ValueError(
                f"v100_w8a16 requires FP16 source parameters, got {params_dtype}."
            )
        output_size_per_partition = sum(output_partition_sizes)
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.logical_widths = output_partition_sizes
        layer.orig_dtype = params_dtype
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )
        layer.register_parameter("weight", weight)

    @torch.no_grad()
    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if layer.weight.dtype == torch.int8:
            return
        if layer.weight.dtype != torch.float16:
            raise ValueError(
                f"v100_w8a16 expected loaded FP16 weights, got {layer.weight.dtype}."
            )

        weight_fp32 = layer.weight.float()
        amax = weight_fp32.abs().amax(dim=1)
        scale = (amax / 127.0).clamp_min(torch.finfo(torch.float32).tiny)
        qweight = torch.round(weight_fp32 / scale.unsqueeze(1))
        qweight = qweight.clamp_(-127, 127).to(torch.int8)

        layer.weight = nn.Parameter(qweight.contiguous(), requires_grad=False)
        layer.weight_scale = nn.Parameter(
            scale.to(torch.float16).contiguous(), requires_grad=False
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if layer.weight.dtype != torch.int8 or not hasattr(layer, "weight_scale"):
            raise RuntimeError(
                "v100_w8a16 weights were used before post-load quantization"
            )
        x = x.to(torch.float16)
        dequant_weight = layer.weight.to(torch.float16)
        dequant_weight.mul_(layer.weight_scale.unsqueeze(1))
        if bias is not None and bias.dtype != torch.float16:
            bias = bias.to(torch.float16)
        if not x.is_cuda:
            return F.linear(x, dequant_weight, bias)

        # H3 was trained and released in BF16. On Volta, a finite BF16 linear
        # result can exceed FP16's range even though cuBLAS accumulates the
        # products in FP32. Request an FP32 output explicitly, then saturate at
        # the reduced-precision boundary so one overflow cannot poison global
        # packed attention with NaNs.
        input_shape = x.shape
        x_2d = x.reshape(-1, input_shape[-1])
        out = torch.mm(x_2d, dequant_weight.t(), out_dtype=torch.float32)
        if bias is not None:
            out.add_(bias.float())
        fp16_max = torch.finfo(torch.float16).max
        return (
            out.clamp_(-fp16_max, fp16_max)
            .to(torch.float16)
            .view(*input_shape[:-1], dequant_weight.shape[0])
        )


__all__ = ["V100W8A16Config", "V100W8A16LinearMethod"]
