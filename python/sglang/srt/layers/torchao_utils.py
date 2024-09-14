"""
Common utilities for torchao.
"""

import torch
from typing import Dict, Set


def torchao_quantize_param_data(param: torch.Tensor, torchao_config: str):
    """Quantize a Tensor with torchao quantization specified by torchao_config

    Args:
       `param`: weight parameter of the linear module
       `torchao_config`: type of quantization and their arguments we want to use to
        quantize the Tensor, e.g. int4wo-128 means int4 weight only quantization with group_size
        128
    """
    # Lazy import to suppress some warnings
    from torchao.quantization import (
        int4_weight_only,
        int8_dynamic_activation_int8_weight,
        int8_weight_only,
        quantize_,
    )

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


def quantize_params_with_suffixes_(params_dict: Dict[str, torch.Tensor], param_suffixes: Set[str], torchao_config: str) -> None:
    """A util function used for quantizing the weight parameters after they are loaded

    Args:
      `params_dict`: dictionary mapping from param_name to the parameter Tensor
      `param_suffixes`: a set of suffixes, we'll quantize the Tensor matching these suffixes

    Returns:
       None, the `params_dict` is modified inplace
    """
    for param_suffix in param_suffixes:
        for name in params_dict:
            param = params_dict[name]
            if param_suffix in name and param.ndim == 2:
                params_dict[name] = torchao_quantize_param_data(
                    param, torchao_config
                )
