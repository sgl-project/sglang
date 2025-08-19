# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.runtime import driver


# ===== Supporting utils, CUDA and TMA =====
class CudaUtils:
    @staticmethod
    def is_cuda() -> bool:
        """Check if Triton is running on CUDA backend."""
        return driver.active.get_current_target().backend == "cuda"

    @staticmethod
    def verify_tma() -> bool:
        """Check if TMA is supported on the current device."""
        return (
            CudaUtils.is_cuda()
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 9
        )

    @staticmethod
    def get_num_sms() -> int:
        """Get the number of streaming multiprocessors on the current device."""
        if not CudaUtils.is_cuda():
            raise RuntimeError("Triton is not running on CUDA backend")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return torch.cuda.get_device_properties("cuda").multi_processor_count


# ================== End of supporting functions ==================


# Define standard configurations for Hopper GPUs
HOPPER_CONFIGS = [
    # Configurations for small matrices
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
        num_stages=2,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
        num_stages=4,
        num_warps=4,
    ),
    # Configurations for medium to large matrices
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
        num_stages=4,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
        num_stages=4,
        num_warps=8,
    ),
]


# Define standard configurations - simplified for robustness
STANDARD_CONFIGS = [
    # Configurations for small matrices
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
        num_stages=2,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
        num_stages=2,
        num_warps=4,
    ),
    # Medium sizes
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
        num_stages=2,
        num_warps=8,
    ),
    # Larger sizes with more warps, stages
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
        num_stages=4,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128},
        num_stages=4,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128},
        num_stages=4,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128},
        num_stages=4,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256},
        num_stages=4,
        num_warps=8,
    ),
]


WS_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "NUM_CONSUMER_GROUPS": 2,
        },
        num_stages=2,
        num_warps=4,
        num_consumer_groups=2,
        num_buffers_warp_spec=3,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "NUM_CONSUMER_GROUPS": 1,
        },
        num_stages=3,
        num_warps=4,
        num_consumer_groups=0,  # disable warp specialization
        num_buffers_warp_spec=3,
    ),
]


_NV_WS_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
            "NUM_CONSUMER_GROUPS": max(1, num_consumer_groups),
        },
        num_stages=num_stages,
        num_warps=num_warps,
        num_ctas=1,
        num_consumer_groups=num_consumer_groups,
        num_buffers_warp_spec=num_stages,
    )
    for block_size_m in [64, 128]
    for block_size_n in [64, 128, 256]
    for block_size_k in [64, 128, 256]
    for num_stages in [2, 3, 4]
    for num_warps in [4, 8]
    for num_consumer_groups in [0, 2]
]


def early_config_prune(configs, args, **kwargs):
    """Filter out configurations that would exceed shared memory capacity."""
    k = kwargs.get("K", 0)
    valid_configs = [
        config for config in configs if config.kwargs.get("BLOCK_SIZE_K", 0) <= k
    ]
    # If all configs were filtered out, return at least one config
    if not valid_configs and configs:
        # Find the config with the smallest BLOCK_SIZE_K
        return [min(configs, key=lambda c: c.kwargs.get("BLOCK_SIZE_K", float("inf")))]

    return valid_configs
