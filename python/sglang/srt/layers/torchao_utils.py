"""
Common utilities for torchao.
"""

import torch
from torchao.quantization import (
    int4_weight_only,
    int8_dynamic_activation_int8_weight,
    int8_weight_only,
    quantize_,
)


def torchao_quantize_param_data(param, torchao_config):
    dummy_linear = torch.nn.Linear(param.shape[1], param.shape[0], bias=False)
    dummy_linear.weight = param
    if "int8wo" in torchao_config:
        quantize_(dummy_linear, int8_weight_only())
    elif "int8dq" in torchao_config:
        quantize_(dummy_linear, int8_dynamic_activation_int8_weight())
    elif "int4wo" in torchao_config:
        group_size = int(torchao_config.split("-")[-1])
        assert group_size in [
            32,
            64,
            128,
            256,
        ], f"int4wo groupsize needs to be one of [32, 64, 128, 256] but got {group_size}"
        quantize_(dummy_linear, int4_weight_only(group_size=group_size))
    elif "fp8wo" in torchao_config:
        from torchao.quantization import float8_weight_only

        # this requires newer hardware
        # [rank0]: AssertionError: fp8e4nv data type is not supported on CUDA arch < 89
        quantize_(dummy_linear, float8_weight_only())
    return dummy_linear.weight
