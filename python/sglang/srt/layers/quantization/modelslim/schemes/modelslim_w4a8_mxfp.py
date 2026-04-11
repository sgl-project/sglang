# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional

import torch

from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUW4A8MxFpLinearMethod
)
from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
)
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimLinearScheme


class ModelSlimW4A8MxFp(ModelSlimLinearScheme):

    def __init__(
        self,
        quant_config: Dict[str, any],
        prefix: str,
    ):
        self.quant_config = quant_config
        self.kernel = NPUW4A8MxFpLinearMethod()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight_loader = extra_weight_attrs.get("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)

        weight = ModelWeightParameter(
            data=torch.empty(
                (output_size_per_partition, input_size_per_partition // 2), dtype=torch.uint8
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        scale_dtype = torch.uint8
        block_k = 32
        weight_scale = BlockQuantScaleParameter(
            data=torch.zeros(
                output_size_per_partition,
                (input_size_per_partition + block_k - 1) // block_k,
                dtype=scale_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.kernel.apply(layer, x, bias)
