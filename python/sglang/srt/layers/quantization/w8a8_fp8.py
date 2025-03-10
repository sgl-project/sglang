from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.linear import LinearMethodBase
from sglang.srt.layers.parameter import ChannelQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    cutlass_fp8_supported,
    normalize_e4m3fn_to_e4m3fnuz,
)
from sglang.srt.utils import is_hip


class W8A8Fp8Config(QuantizationConfig):
    """Config class for W8A8 FP8 Quantization.

    - Weight: static, per-channel, symmetric
    - Activation: dynamic, per-token, symmetric
    """

    def __init__(self):
        pass

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @classmethod
    def get_name(self) -> str:
        return "w8a8_fp8"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "W8A8Fp8Config":
        return cls()

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        from sglang.srt.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            return W8A8Fp8LinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class W8A8Fp8LinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: W8A8Fp8Config):
        self.cutlass_fp8_supported = cutlass_fp8_supported()
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight
        weight_scale = layer.weight_scale.detach()
        if is_hip():
            weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                weight=weight, weight_scale=weight_scale
            )
        layer.weight = Parameter(weight.t(), requires_grad=False)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs
    ):

        weight_loader = extra_weight_attrs.get("weight_loader")
        self.logical_widths = output_partition_sizes

        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        return apply_fp8_linear(
            x,
            layer.weight,
            layer.weight_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported,
        )
