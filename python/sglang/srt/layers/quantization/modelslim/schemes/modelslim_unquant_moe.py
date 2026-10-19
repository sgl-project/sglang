"""Floating-point projection groups in mixed ModelSlim MoE layers."""

import torch

from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUUnquantMoEMethod
from sglang.srt.layers.quantization.modelslim.schemes.modelslim_scheme import (
    ModelSlimMoEScheme,
)
from sglang.srt.utils import set_weight_attrs


class ModelSlimUnquantizedMoE(ModelSlimMoEScheme):
    def __init__(self, quant_config, weight_prefix):
        self.weight_prefix = weight_prefix
        self.kernel = NPUUnquantMoEMethod()

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        **extra_weight_attrs,
    ):
        if self.weight_prefix == "w13":
            shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        else:
            shape = (num_experts, hidden_size, intermediate_size_per_partition)
        weight = torch.nn.Parameter(
            torch.empty(shape, dtype=layer.moe_runner_config.params_dtype),
            requires_grad=False,
        )
        layer.register_parameter(f"{self.weight_prefix}_weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter(f"{self.weight_prefix}_weight_scale", None)
        layer.register_parameter(f"{self.weight_prefix}_weight_offset", None)

    def process_weights_after_loading(self, layer):
        self.kernel.process_weights_after_loading(layer, self.weight_prefix)
