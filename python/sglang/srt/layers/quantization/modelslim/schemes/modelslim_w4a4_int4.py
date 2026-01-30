# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

import torch

from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPU_W4A4DynamicLinearMethod,
)
from sglang.srt.layers.parameter import PerTensorScaleParameter
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimScheme
from sglang.srt.utils import set_weight_attrs


class ModelSlimW4A4Int4(ModelSlimScheme):

    def __init__(
        self,
        quant_config: Dict[str, any],
        prefix: str,
    ):
        self.quant_config = quant_config
        self.is_dynamic = self.quant_config[prefix + ".weight"] == "W4A4_DYNAMIC"
        self.kernel = NPU_W4A4DynamicLinearMethod()

    @staticmethod
    def get_weight(
        input_size: int, output_size: int, params_dtype: torch.dtype
    ) -> Dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size, dtype=torch.int8)}
        return params_dict

    @staticmethod
    def get_perchannel_param(
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size, 1, dtype=params_dtype)
        return params_dict

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        weight_dict = {
            "weight": torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=torch.int8
            )
        }
        for weight_name, weight_param in weight_dict.items():
            param = torch.nn.Parameter(weight_param, requires_grad=False)
            set_weight_attrs(param, {"input_dim": 1, "output_dim": 0})
            layer.register_parameter(weight_name, param)
            set_weight_attrs(param, extra_weight_attrs)

        pertensor_dict = {}
        for pertensor_name, pertensor_param in pertensor_dict.items():
            param = PerTensorScaleParameter(
                data=pertensor_param, weight_loader=weight_loader
            )
            # disable warning
            param.ignore_warning = True
            layer.register_parameter(pertensor_name, param)

        perchannel_dict = {}
        perchannel_dict["weight_scale"] = torch.empty(
            output_size_per_partition, 1, dtype=params_dtype
        )
        perchannel_dict["weight_offset"] = torch.empty(
            output_size_per_partition, 1, dtype=params_dtype
        )
        for perchannel_name, perchannel_param in perchannel_dict.items():
            param = torch.nn.Parameter(perchannel_param, requires_grad=False)
            set_weight_attrs(param, {"output_dim": 0})
            layer.register_parameter(perchannel_name, param)
            set_weight_attrs(param, extra_weight_attrs)

    def process_weights_after_loading(self, layer):
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.kernel.apply(layer, x, bias)
