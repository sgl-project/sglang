# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/fp8_utils.py
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import torch

from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    PerTensorScaleParameter,
)


def create_fp8_scale_parameter(
    parameter_type: torch.nn.Parameter,
    output_partition_sizes: list[int],
    input_size_per_partition: int,
    block_size: list[int] | None,
    weight_loader: Callable | None,
) -> torch.nn.Parameter:
    """Create scale parameter based on quantization strategy."""
    if parameter_type == ChannelQuantScaleParameter:
        scale = parameter_type(
            data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
    elif parameter_type == BlockQuantScaleParameter:
        assert block_size is not None
        block_n, block_k = block_size[0], block_size[1]
        output_size_per_partition = sum(output_partition_sizes)
        scale = parameter_type(
            data=torch.empty(
                (output_size_per_partition + block_n - 1) // block_n,
                (input_size_per_partition + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
    elif parameter_type == PerTensorScaleParameter:
        scale = parameter_type(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
    else:
        raise ValueError(f"Unknown parameter type: {parameter_type}")

    scale[:] = torch.finfo(torch.float32).min
    return scale
