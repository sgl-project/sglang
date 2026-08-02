# SPDX-License-Identifier: Apache-2.0
"""Decode-group state used for complete-tile VAE parallelism."""

import torch.distributed as dist

from sglang.multimodal_gen.runtime.distributed import (
    get_decode_parallel_rank,
    get_decode_parallel_world_size,
    model_parallel_is_initialized,
)


def get_tile_parallel_state():
    if not dist.is_initialized() or not model_parallel_is_initialized():
        return 0, 1
    return get_decode_parallel_rank(), get_decode_parallel_world_size()
