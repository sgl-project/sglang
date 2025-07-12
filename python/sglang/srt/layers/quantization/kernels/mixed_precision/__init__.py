# Adapted from https://github.com/vllm-project/vllm/tree/v0.8.2/vllm/model_executor/layers/quantization/kernels/mixed_precision/__init__.py

from typing import List, Mapping, Tuple, Union, Optional, NamedTuple

import torch

from sglang.srt.layers.quantization.kernels.mixed_precision.marlin import MarlinLinearKernel
from sglang.srt.layers.quantization.kernels.mixed_precision.MPLinearKernel import (
    MPLinearKernel, MPLinearLayerConfig
)

_POSSIBLE_KERNELS: list[type[MPLinearKernel]] = [
    MarlinLinearKernel,
]

def choose_mp_linear_kernel(
        config: MPLinearLayerConfig,
        compute_capability: Optional[int] = None) -> type[MPLinearKernel]:
    """
    Choose an MPLinearKernel that can implement the given config for the given
     compute capability. Attempts to choose the best kernel in terms of 
     performance.

    Args:
        config (MPLinearLayerConfig): Description of the linear layer to be 
          implemented.
        compute_capability (Optional[int], optional): The compute capability of
          the target device, if None uses `current_platform` to get the compute 
          capability. Defaults to None.

    Raises:
        ValueError: If no kernel can implement the given config.

    Returns:
        type[MPLinearKernel]: Chosen kernel.
    """
    compute_capability = torch.cuda.get_device_capability()
    compute_capability = compute_capability[0] * 10 + compute_capability[1]
    failure_reasons = []
    for kernel in _POSSIBLE_KERNELS:
        if kernel.get_min_capability() > compute_capability:
            failure_reasons.append(
                f"{kernel.__name__} requires capability "
                f"{kernel.get_min_capability()}, current compute capability "
                f"is {compute_capability}")
            continue

        can_implement, failure_reason = kernel.can_implement(config)
        if can_implement:
            return kernel
        else:
            failure_reasons.append(
                f' {kernel.__name__} cannot implement due to: {failure_reason}'
            )

    raise ValueError(
        "Failed to find a kernel that can implement the "\
        "WNA16 linear layer. Reasons: \n"
        + '\n'.join(failure_reasons))