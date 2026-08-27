"""Online W8A8 INT8 quantization for diffusion linear layers on Ascend NPU."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.multimodal_gen.runtime.layers.linear import LinearBase, LinearMethodBase
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.models.parameter import ModelWeightParameter
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs
from sglang.srt.hardware_backend.npu.quantization.online_quantization import (
    get_npu_online_integer_quant_spec,
    npu_dynamic_quantize_weight,
    npu_format_online_dense_scale,
    npu_format_online_dense_weight,
)
from sglang.srt.layers.quantization.utils import is_layer_skipped


class NPUOnlineW8A8DiffusionConfig(QuantizationConfig):
    """Quantize full-precision diffusion transformer linears to W8A8."""

    def __init__(
        self,
        ignored_layers: list[str] | None = None,
        packed_modules_mapping: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__()
        if not current_platform.is_npu():
            raise ValueError("w8a8_int8 diffusion quantization requires Ascend NPU")
        self.ignored_layers = ignored_layers or []
        self.packed_modules_mapping = packed_modules_mapping or {}

    @classmethod
    def get_name(cls) -> str:
        return "w8a8_int8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NPUOnlineW8A8DiffusionConfig":
        return cls(
            ignored_layers=cls.get_from_keys_or(config, ["ignored_layers"], None)
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        if not isinstance(layer, LinearBase):
            return None
        if is_layer_skipped(
            prefix,
            self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
        ):
            from sglang.multimodal_gen.runtime.layers.linear import (
                UnquantizedLinearMethod,
            )

            return UnquantizedLinearMethod()
        return NPUOnlineW8A8DiffusionLinearMethod(self)

    def get_scaled_act_names(self) -> list[str]:
        return []


class NPUOnlineW8A8DiffusionLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: NPUOnlineW8A8DiffusionConfig) -> None:
        self.quant_config = quant_config
        self.spec = get_npu_online_integer_quant_spec("w8a8_int8")
        assert self.spec is not None

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        if weight.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "w8a8_int8 diffusion quantization requires FP16 or BF16 "
                f"weights, got {weight.dtype}"
            )
        if not weight.is_npu:
            weight = weight.to(f"npu:{torch.npu.current_device()}")
        quantized_weight, weight_scale = npu_dynamic_quantize_weight(
            weight, self.spec
        )
        weight = Parameter(
            npu_format_online_dense_weight(quantized_weight, self.spec),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.weight = weight
        layer.weight_scale = Parameter(
            npu_format_online_dense_scale(weight_scale, self.spec),
            requires_grad=False,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_shape = x.shape
        output_dtype = x.dtype
        if output_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "w8a8_int8 diffusion quantization requires FP16 or BF16 "
                f"activations, got {output_dtype}"
            )
        quantized_x, dynamic_scale = torch.ops.npu.npu_dynamic_quant(
            x.reshape(-1, x.shape[-1]),
            dst_type=self.spec.activation_dtype,
        )
        output = torch.ops.npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=dynamic_scale.flatten(),
            bias=bias.to(torch.float32) if bias is not None else None,
            output_dtype=output_dtype,
        )
        return output.reshape(*input_shape[:-1], output.shape[-1])
