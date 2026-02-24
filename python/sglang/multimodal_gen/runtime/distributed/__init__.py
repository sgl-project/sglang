# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
from functools import lru_cache

from sglang.multimodal_gen.configs.models.encoders import TextEncoderConfig
from sglang.multimodal_gen.runtime.distributed.communication_op import *
from sglang.multimodal_gen.runtime.distributed.group_coordinator import (
    get_local_torch_device,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    get_dp_group,
    get_dp_rank,
    get_dp_world_size,
    get_sp_group,
    get_sp_parallel_rank,
    get_sp_world_size,
    get_tp_group,
    get_tp_rank,
    get_tp_world_size,
    get_world_group,
    get_world_rank,
    get_world_size,
    init_distributed_environment,
    initialize_model_parallel,
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.distributed.utils import *

# SPDX-License-Identifier: Apache-2.0


__all__ = [
    # Initialization
    "init_distributed_environment",
    "initialize_model_parallel",
    "cleanup_dist_env_and_memory",
    "model_parallel_is_initialized",
    "maybe_init_distributed_environment_and_model_parallel",
    # World group
    "get_world_group",
    "get_world_rank",
    "get_world_size",
    # Data parallel group
    "get_dp_group",
    "get_dp_rank",
    "get_dp_world_size",
    # Sequence parallel group
    "get_sp_group",
    "get_sp_parallel_rank",
    "get_sp_world_size",
    # Tensor parallel group
    "get_tp_group",
    "get_tp_rank",
    "get_tp_world_size",
    # Get torch device
    "get_local_torch_device",
]


def _get_folding_tp_group(
    config: TextEncoderConfig,
) -> torch.distributed.ProcessGroup | None:
    if config.parallel_folding:
        if config.parallel_folding_mode == "sp":
            return get_sp_group()
        elif config.parallel_folding_mode == "ulysses":
            return get_sp_group().ulysses_group
        elif config.parallel_folding_mode == "ring":
            return get_sp_group().ring_group
    return get_tp_group()
