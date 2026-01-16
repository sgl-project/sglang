"""
Common utilities for torchao.
"""

import logging
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


def proj_filter(
    module: torch.nn.Module,
    fqn: str,
):
    """Filter function for quantizing projection layers."""
    return "proj" in fqn


# TODO: implement a more general filter function
def proj_filter_conv3d(
    module: torch.nn.Module,
    fqn: str,
):
    if isinstance(module, torch.nn.Conv3d):
        logger.warning(f"Quantize: skipping {fqn} because it's a Conv3d")
        return False
    return "proj" in fqn


def apply_torchao_config_to_model(
    model: torch.nn.Module,
    torchao_config: str,
    filter_fn: Optional[Callable] = proj_filter,
):
    """Quantize a modelwith torchao quantization specified by torchao_config

    Args:
       `model`: a model to be quantized based on torchao_config
       `torchao_config` (str): type of quantization and their arguments we want to use to
        quantize the model, e.g. int4wo-128 means int4 weight only quantization with group_size
        128
    """
    # Lazy import to suppress some warnings
    from torchao.quantization import (
        float8_dynamic_activation_float8_weight,
        float8_weight_only,
        int4_weight_only,
        int8_dynamic_activation_int8_weight,
        int8_weight_only,
        quantize_,
    )
    from torchao.quantization.observer import PerRow, PerTensor

    if torchao_config == "" or torchao_config is None:
        return model
    elif "int8wo" in torchao_config:
        quantize_(model, int8_weight_only(), filter_fn=proj_filter_conv3d)
    elif "int8dq" in torchao_config:
        quantize_(model, int8_dynamic_activation_int8_weight(), filter_fn=filter_fn)
    elif "int4wo" in torchao_config:
        group_size = int(torchao_config.split("-")[-1])
        assert group_size in [
            32,
            64,
            128,
            256,
        ], f"int4wo groupsize needs to be one of [32, 64, 128, 256] but got {group_size}"
        quantize_(model, int4_weight_only(group_size=group_size), filter_fn=filter_fn)
    elif "fp8wo" in torchao_config:
        # this requires newer hardware
        # [rank0]: AssertionError: fp8e4nv data type is not supported on CUDA arch < 89
        quantize_(model, float8_weight_only(), filter_fn=proj_filter_conv3d)
    elif "fp8dq" in torchao_config:
        granularity = torchao_config.split("-")[-1]
        GRANULARITY_MAP = {
            "per_row": PerRow(),
            "per_tensor": PerTensor(),
        }
        assert (
            granularity in GRANULARITY_MAP
        ), f"Supported granularity are: {GRANULARITY_MAP.keys()}, got {granularity}"
        quantize_(
            model,
            float8_dynamic_activation_float8_weight(
                granularity=GRANULARITY_MAP[granularity]
            ),
            filter_fn=proj_filter_conv3d,
        )
    else:
        raise ValueError(f"Unexpected config: {torchao_config}")

    return model
