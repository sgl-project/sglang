"""Online W8A8 INT8 quantization for diffusion linear layers on Ascend NPU."""

from __future__ import annotations

from typing import Any, Optional

import torch

from sglang.multimodal_gen.runtime.layers.linear import LinearBase, LinearMethodBase
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.models.parameter import ModelWeightParameter
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUW8A8Int8DynamicLinearMethod,
    _INT8Activation,
)
from sglang.srt.layers.quantization.utils import _module_path_match, is_layer_skipped


class NPUOnlineW8A8DiffusionConfig(QuantizationConfig):
    """Quantize full-precision diffusion transformer linears to W8A8."""

    def __init__(
        self,
        ignored_layers: list[str] | None = None,
        packed_modules_mapping: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__()
        if not current_platform.is_npu():
            raise ValueError("w8a8_int diffusion quantization requires Ascend NPU")
        self.ignored_layers = ignored_layers or []
        self.packed_modules_mapping = packed_modules_mapping or {}

    @classmethod
    def get_name(cls) -> str:
        return "w8a8_int"

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
    def from_config(cls, config: dict[str, Any]) -> NPUOnlineW8A8DiffusionConfig:
        return cls(
            ignored_layers=cls.get_from_keys_or(config, ["ignored_layers"], None)
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        if not isinstance(layer, LinearBase):
            return None
        if any(
            _module_path_match(ignored, prefix) for ignored in self.ignored_layers
        ) or is_layer_skipped(
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
    accepts_int8_input = True

    def __init__(self, quant_config: NPUOnlineW8A8DiffusionConfig) -> None:
        self.quant_config = quant_config
        self.kernel = NPUW8A8Int8DynamicLinearMethod(online_weights=True)

    def quantize_swiglu(self, x: torch.Tensor) -> _INT8Activation | None:
        return self.kernel.quantize_swiglu(x)

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

    def _process_weight_on_load(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        if weight.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "w8a8_int diffusion quantization requires FP16 or BF16 "
                f"weights, got {weight.dtype}"
            )
        torch.npu.config.allow_internal_format = True
        self.kernel.process_weights_after_loading(layer)
        set_weight_attrs(layer.weight, {"input_dim": 1, "output_dim": 0})
        layer._w8a8_weight_processed = True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not getattr(layer, "_w8a8_weight_processed", False):
            self._process_weight_on_load(layer)
        # Convert GEMM bias once instead of on every denoising step.
        if layer.bias is not None and not layer.skip_bias_add:
            layer.bias.data = layer.bias.data.to(torch.float32)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | _INT8Activation,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(x, _INT8Activation):
            input_shape = x.data.shape
            output_dtype = x.source_dtype
        else:
            input_shape = x.shape
            output_dtype = x.dtype
        if output_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "w8a8_int diffusion quantization requires FP16 or BF16 "
                f"activations, got {output_dtype}"
            )
        if not isinstance(x, _INT8Activation):
            x = x.reshape(-1, x.shape[-1])
        output = self.kernel.apply(layer, x, bias)
        return output.reshape(*input_shape[:-1], output.shape[-1])
