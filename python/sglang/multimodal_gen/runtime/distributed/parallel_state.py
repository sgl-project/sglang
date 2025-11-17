# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/distributed/parallel_state.py
# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Adapted from
# Copyright 2024 xDiT team.
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/distributed/parallel_state.py
# Copyright 2023 The vLLM team.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""sgl-diffusion distributed state.

It takes over the control of the distributed environment from PyTorch.
The typical workflow is:

- call `init_distributed_environment` to initialize the distributed environment.
- call `initialize_model_parallel` or `ensure_model_parallel_initialized` to
 initialize the model parallel groups.

- any code dealing with the distributed stuff

- call `destroy_model_parallel` to destroy the model parallel groups.
- call `destroy_distributed_environment` to destroy the distributed environment.

If you only need to use the distributed environment without model parallelism,
 you can skip the model parallel initialization and destruction steps.
"""
import contextlib
import os
import weakref
from collections import namedtuple
from collections.abc import Callable
from contextlib import contextmanager
from multiprocessing import shared_memory
from typing import Any, List, Optional
from unittest.mock import patch

import torch
import torch.distributed
from torch.distributed import ProcessGroup

import sglang.multimodal_gen.envs as envs
from sglang.multimodal_gen.runtime.distributed.utils import StatelessProcessGroup
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from ..utils.distributed import RankGenerator
from .group_coordinator import (
    GroupCoordinator,
    PipelineGroupCoordinator,
    SequenceParallelGroupCoordinator,
    get_local_torch_device,
)

logger = init_logger(__name__)

_WORLD: Optional[GroupCoordinator] = None
_TP: Optional[GroupCoordinator] = None
_SP: Optional[SequenceParallelGroupCoordinator] = None
_PP: Optional[PipelineGroupCoordinator] = None
_CFG: Optional[GroupCoordinator] = None
_DP: Optional[GroupCoordinator] = None
_DIT: Optional[GroupCoordinator] = None
_VAE: Optional[GroupCoordinator] = None

logger = init_logger(__name__)

TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])


def _split_tensor_dict(
    tensor_dict: dict[str, torch.Tensor | Any]
) -> tuple[list[tuple[str, Any]], list[torch.Tensor]]:
    """Split the tensor dictionary into two parts:
    1. A list of (key, value) pairs. If the value is a tensor, it is replaced
         by its metadata.
    2. A list of tensors.
    """
    metadata_list: list[tuple[str, Any]] = []
    tensor_list: list[torch.Tensor] = []
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            # Note: we cannot use `value.device` here,
            # because it contains not only the device type but also the device
            # index (e.g. "cuda:0"). We only need the device type.
            # receiving side will set the device index.
            device = value.device.type
            metadata_list.append(
                (key, TensorMetadata(device, value.dtype, value.size()))
            )
            tensor_list.append(value)
        else:
            metadata_list.append((key, value))
    return metadata_list, tensor_list


_groups: dict[str, Callable[[], Optional["GroupCoordinator"]]] = {}


def _register_group(group: "GroupCoordinator") -> None:
    _groups[group.unique_name] = weakref.ref(group)


def all_reduce(tensor: torch.Tensor, group_name: str) -> torch.Tensor:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    return group._all_reduce_out_place(tensor)


def all_reduce_fake(tensor: torch.Tensor, group_name: str) -> torch.Tensor:
    return torch.empty_like(tensor)


_WORLD: GroupCoordinator | None = None
_NODE: GroupCoordinator | None = None


def get_world_group() -> GroupCoordinator:
    assert _WORLD is not None, "world group is not initialized"
    return _WORLD


def init_world_group(
    ranks: list[int], local_rank: int, backend: str
) -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=[ranks],
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_device_communicator=True,
        group_name="world",
    )


# xDiT
def init_parallel_group_coordinator(
    group_ranks: List[List[int]],
    local_rank: int,
    backend: str,
    parallel_mode: str,
    **kwargs,
) -> GroupCoordinator:
    """
    Returns a Group Coordinator for the given parallel mode
    """
    assert parallel_mode in [
        "data",
        "pipeline",
        "tensor",
        "sequence",
        "classifier_free_guidance",
    ], f"parallel_mode {parallel_mode} is not supported"
    if parallel_mode == "pipeline":
        return PipelineGroupCoordinator(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=backend,
            group_name="pp_group",
        )
    elif parallel_mode == "sequence":
        return SequenceParallelGroupCoordinator(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=backend,
            group_name="sp_group",
            **kwargs,
        )
    else:
        # fallback to GroupCoordinator
        return GroupCoordinator(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=backend,
            group_name="cfg_group",
        )


# def init_parallel_group_coordinator(
#     group_ranks: list[list[int]],
#     local_rank: int,
#     backend: str,
#     use_message_queue_broadcaster: bool = False,
#     group_name: str | None = None,
# ) -> GroupCoordinator:
#     return GroupCoordinator(
#         group_ranks=group_ranks,
#         local_rank=local_rank,
#         torch_distributed_backend=backend,
#         use_device_communicator=True,
#         use_message_queue_broadcaster=use_message_queue_broadcaster,
#         group_name=group_name,
#     )


_TP: GroupCoordinator | None = None


def get_tp_group() -> GroupCoordinator:
    assert _TP is not None, "tensor model parallel group is not initialized"
    return _TP


_ENABLE_CUSTOM_ALL_REDUCE = True


def set_custom_all_reduce(enable: bool):
    global _ENABLE_CUSTOM_ALL_REDUCE
    _ENABLE_CUSTOM_ALL_REDUCE = enable


def init_distributed_environment(
    world_size: int = 1,
    rank: int = 0,
    distributed_init_method: str = "env://",
    local_rank: int = 0,
    backend: str = "nccl",
    device_id: torch.device | None = None,
):
    # Determine the appropriate backend based on the platform
    from sglang.multimodal_gen.runtime.platforms import current_platform

    if backend == "nccl" and not current_platform.is_cuda_alike():
        # Use gloo backend for non-CUDA platforms (MPS, CPU)
        backend = "gloo"
        logger.info("Using gloo backend for %s platform", current_platform.device_name)

    logger.debug(
        "world_size=%d rank=%d local_rank=%d " "distributed_init_method=%s backend=%s",
        world_size,
        rank,
        local_rank,
        distributed_init_method,
        backend,
    )
    if not torch.distributed.is_initialized():
        assert distributed_init_method is not None, (
            "distributed_init_method must be provided when initializing "
            "distributed environment"
        )

        # For MPS, don't pass device_id as it doesn't support device indices
        extra_args = {} if current_platform.is_mps() else dict(device_id=device_id)
        torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
            **extra_args,
        )
    # set the local rank
    # local_rank is not available in torch ProcessGroup,
    # see https://github.com/pytorch/pytorch/issues/122816
    if local_rank == -1:
        # local rank not set, this usually happens in single-node
        # setting, where we can use rank as local rank
        if distributed_init_method == "env://":
            local_rank = envs.LOCAL_RANK
        else:
            local_rank = rank
    global _WORLD
    if _WORLD is None:
        ranks = list(range(torch.distributed.get_world_size()))
        _WORLD = init_world_group(ranks, local_rank, backend)
    else:
        assert (
            _WORLD.world_size == torch.distributed.get_world_size()
        ), "world group already initialized with a different world size"


_SP: GroupCoordinator | None = None


def get_sp_group() -> SequenceParallelGroupCoordinator:
    assert _SP is not None, "pipeline model parallel group is not initialized"
    return _SP


_DP: GroupCoordinator | None = None


def get_dp_group() -> GroupCoordinator:
    assert _DP is not None, "data parallel group is not initialized"
    return _DP


# xDiT
def initialize_model_parallel(
    data_parallel_size: int = 1,
    classifier_free_guidance_degree: int = 1,
    sequence_parallel_degree: Optional[int] = None,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    tensor_parallel_degree: int = 1,
    pipeline_parallel_degree: int = 1,
    vae_parallel_size: int = 0,
    backend: Optional[str] = None,
) -> None:
    """
    Initialize model parallel groups.

    Arguments:
        data_parallel_size: number of data parallelism groups.
        classifier_free_guidance_degree: number of GPUs used for Classifier Free Guidance (CFG)
        sequence_parallel_degree: number of GPUs used for sequence parallelism. sequence_parallel_degree = ulysses_degree * ring_degree
        ulysses_degree: number of GPUs used for ulysses sequence parallelism.
        ring_degree: number of GPUs used for ring sequence parallelism.
        tensor_parallel_degree: number of GPUs used for tensor parallelism.
        pipeline_parallel_degree: number of GPUs used for pipeline parallelism.
        backend: distributed backend of pytorch collective comm.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 groups to parallelize the batch dim(dp), 2 groups to parallelize
    split batch caused by CFG, and 2 GPUs to parallelize sequence.

    dp_degree (2) * cfg_degree (2) * sp_degree (2) * pp_degree (2) = 16.

    The present function will create 8 data-parallel groups,
    8 CFG group, 8 pipeline-parallel group, and
    8 sequence-parallel groups:
        8 data-parallel groups:
            [g0, g8], [g1, g9], [g2, g10], [g3, g11],
            [g4, g12], [g5, g13], [g6, g14], [g7, g15]
        8 CFG-parallel groups:
            [g0, g4], [g1, g5], [g2, g6], [g3, g7],
            [g8, g12], [g9, g13], [g10, g14], [g11, g15]
        8 sequence-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7],
            [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        8 pipeline-parallel groups:
            [g0, g2], [g4, g6], [g8, g10], [g12, g14],
            [g1, g3], [g5, g7], [g9, g11], [g13, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """

    if backend is None:
        backend = envs.get_torch_distributed_backend()
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    dit_parallel_size = (
        data_parallel_size
        * classifier_free_guidance_degree
        * sequence_parallel_degree
        * pipeline_parallel_degree
        * tensor_parallel_degree
    )

    if world_size < dit_parallel_size:
        raise RuntimeError(
            f"world_size ({world_size}) is less than "
            f"tensor_parallel_degree ({tensor_parallel_degree}) x "
            f"pipeline_parallel_degree ({pipeline_parallel_degree}) x"
            f"sequence_parallel_degree ({sequence_parallel_degree}) x"
            f"classifier_free_guidance_degree "
            f"({classifier_free_guidance_degree}) x"
            f"data_parallel_degree ({data_parallel_size})"
        )

    rank_generator: RankGenerator = RankGenerator(
        tensor_parallel_degree,
        sequence_parallel_degree,
        pipeline_parallel_degree,
        classifier_free_guidance_degree,
        data_parallel_size,
        "tp-sp-pp-cfg-dp",
    )
    global _DP
    assert _DP is None, "data parallel group is already initialized"
    _DP = init_parallel_group_coordinator(
        group_ranks=rank_generator.get_ranks("dp"),
        local_rank=get_world_group().local_rank,
        backend=backend,
        parallel_mode="data",
    )

    global _CFG
    assert _CFG is None, "classifier_free_guidance group is already initialized"
    _CFG = init_parallel_group_coordinator(
        group_ranks=rank_generator.get_ranks("cfg"),
        local_rank=get_world_group().local_rank,
        backend=backend,
        parallel_mode="classifier_free_guidance",
    )
    global _PP
    assert _PP is None, "pipeline model parallel group is already initialized"
    _PP = init_parallel_group_coordinator(
        group_ranks=rank_generator.get_ranks("pp"),
        local_rank=get_world_group().local_rank,
        backend=backend,
        parallel_mode="pipeline",
    )

    global _SP
    assert _SP is None, "sequence parallel group is already initialized"

    from yunchang import set_seq_parallel_pg
    from yunchang.globals import PROCESS_GROUP

    set_seq_parallel_pg(
        sp_ulysses_degree=ulysses_degree,
        sp_ring_degree=ring_degree,
        rank=get_world_group().rank_in_group,
        world_size=dit_parallel_size,
    )

    _SP = init_parallel_group_coordinator(
        group_ranks=rank_generator.get_ranks("sp"),
        local_rank=get_world_group().local_rank,
        backend=backend,
        parallel_mode="sequence",
        ulysses_group=PROCESS_GROUP.ULYSSES_PG,
        ring_group=PROCESS_GROUP.RING_PG,
    )

    global _TP
    assert _TP is None, "Tensor parallel group is already initialized"
    _TP = init_parallel_group_coordinator(
        group_ranks=rank_generator.get_ranks("tp"),
        local_rank=get_world_group().local_rank,
        backend=backend,
        parallel_mode="tensor",
    )

    if vae_parallel_size > 0:
        init_vae_group(dit_parallel_size, vae_parallel_size, backend)
    init_dit_group(dit_parallel_size, backend)


#


# def initialize_model_parallel(
#     tensor_model_parallel_size: int = 1,
#     sequence_model_parallel_size: int = 1,
#     data_parallel_size: int = 1,
#     backend: str | None = None,
# ) -> None:
#     """
#     Initialize model parallel groups.
#
#     Arguments:
#         tensor_model_parallel_size: number of GPUs used for tensor model
#             parallelism (used for language encoder).
#         sequence_model_parallel_size: number of GPUs used for sequence model
#             parallelism (used for DiT).
#     """
#     # Get world size and rank. Ensure some consistencies.
#     assert (
#         _WORLD is not None
#     ), "world group is not initialized, please call init_distributed_environment first"
#     world_size: int = get_world_size()
#     backend = backend or torch.distributed.get_backend(get_world_group().device_group)
#     assert (
#         world_size >= tensor_model_parallel_size
#     ), f"world_size({world_size}) must be greater than or equal to tensor_model_parallel_size({tensor_model_parallel_size})"
#     num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
#     global _TP
#     assert _TP is None, "tensor model parallel group is already initialized"
#     group_ranks = []
#     for i in range(num_tensor_model_parallel_groups):
#         ranks = list(
#             range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
#         )
#         group_ranks.append(ranks)
#
#     # message queue broadcaster is only used in tensor model parallel group
#     _TP = init_parallel_group_coordinator(
#         group_ranks,
#         get_world_group().local_rank,
#         backend,
#         use_message_queue_broadcaster=True,
#         group_name="tp",
#     )
#
#     # Build the sequence model-parallel groups.
#     num_sequence_model_parallel_groups: int = world_size // sequence_model_parallel_size
#     global _SP
#     assert _SP is None, "sequence model parallel group is already initialized"
#     group_ranks = []
#
#     # Since SP is incompatible with TP and PP, we can use a simpler group creation logic
#     for i in range(num_sequence_model_parallel_groups):
#         # Create groups of consecutive ranks
#         ranks = list(
#             range(
#                 i * sequence_model_parallel_size, (i + 1) * sequence_model_parallel_size
#             )
#         )
#         group_ranks.append(ranks)
#
#     _SP = init_parallel_group_coordinator(
#         group_ranks, get_world_group().local_rank, backend, group_name="sp"
#     )
#
#     # Build the data parallel groups.
#     num_data_parallel_groups: int = sequence_model_parallel_size
#     global _DP
#     assert _DP is None, "data parallel group is already initialized"
#     group_ranks = []
#
#     for i in range(num_data_parallel_groups):
#         ranks = list(range(i, world_size, num_data_parallel_groups))
#         group_ranks.append(ranks)
#
#     _DP = init_parallel_group_coordinator(
#         group_ranks, get_world_group().local_rank, backend, group_name="dp"
#     )
#


def get_sp_world_size() -> int:
    """Return world size for the sequence model parallel group."""
    return get_sp_group().world_size


def get_sp_parallel_rank() -> int:
    """Return my rank for the sequence model parallel group."""
    return get_sp_group().rank_in_group


def get_world_size() -> int:
    """Return world size for the world group."""
    return get_world_group().world_size


def get_world_rank() -> int:
    """Return my rank for the world group."""
    return get_world_group().rank


def get_dp_world_size() -> int:
    """Return world size for the data parallel group."""
    return get_dp_group().world_size


def get_dp_rank() -> int:
    """Return my rank for the data parallel group."""
    return get_dp_group().rank_in_group


def maybe_init_distributed_environment_and_model_parallel(
    tp_size: int,
    sp_size: int,
    enable_cfg_parallel: bool,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    dp_size: int = 1,
    distributed_init_method: str = "env://",
):
    from sglang.multimodal_gen.runtime.platforms import current_platform

    if _WORLD is not None and model_parallel_is_initialized():
        # make sure the tp and sp sizes are correct
        assert (
            get_tp_world_size() == tp_size
        ), f"You are trying to initialize model parallel groups with size {tp_size}, but they are already initialized with size {get_tp_world_size()}"
        assert (
            get_sp_world_size() == sp_size
        ), f"You are trying to initialize model parallel groups with size {sp_size}, but they are already initialized with size {get_sp_world_size()}"
        return
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    device = get_local_torch_device()
    logger.info(
        "Initializing distributed environment with world_size=%d, device=%s",
        world_size,
        device,
        main_process_only=False,
    )

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method=distributed_init_method,
        device_id=device,
    )
    initialize_model_parallel(
        data_parallel_size=dp_size,
        classifier_free_guidance_degree=2 if enable_cfg_parallel else 1,
        tensor_parallel_degree=tp_size,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        sequence_parallel_degree=sp_size,
    )

    # Only set CUDA device if we're on a CUDA platform
    if current_platform.is_cuda_alike():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)


def model_parallel_is_initialized() -> bool:
    """Check if tensor, sequence parallel groups are initialized."""
    return _TP is not None and _SP is not None and _DP is not None and _CFG is not None


_TP_STATE_PATCHED = False


@contextmanager
def patch_tensor_parallel_group(tp_group: GroupCoordinator):
    """Patch the tp group temporarily until this function ends.

    This method is for draft workers of speculative decoding to run draft model
    with different tp degree from that of target model workers.

    Args:
        tp_group (GroupCoordinator): the tp group coordinator
    """
    global _TP_STATE_PATCHED
    assert not _TP_STATE_PATCHED, "Should not call when it's already patched"

    _TP_STATE_PATCHED = True
    old_tp_group = get_tp_group()
    global _TP
    _TP = tp_group
    try:
        yield
    finally:
        # restore the original state
        _TP_STATE_PATCHED = False
        _TP = old_tp_group


def get_tp_world_size() -> int:
    """Return world size for the tensor model parallel group."""
    return get_tp_group().world_size


def get_tp_rank() -> int:
    """Return my rank for the tensor model parallel group."""
    return get_tp_group().rank_in_group


def destroy_distributed_environment() -> None:
    global _WORLD
    if _WORLD:
        _WORLD.destroy()
    _WORLD = None
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def cleanup_dist_env_and_memory(shutdown_ray: bool = False):
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    if shutdown_ray:
        import ray  # Lazy import Ray

        ray.shutdown()


def is_the_same_node_as(
    pg: ProcessGroup | StatelessProcessGroup, source_rank: int = 0
) -> list[int]:
    """
    This is a collective operation that returns if each rank is in the same node
    as the source rank. It tests if processes are attached to the same
    memory system (shared access to shared memory).
    """
    if isinstance(pg, ProcessGroup):
        assert (
            torch.distributed.get_backend(pg) != torch.distributed.Backend.NCCL
        ), "in_the_same_node_as should be tested with a non-NCCL group."
        # local rank inside the group
        rank = torch.distributed.get_rank(group=pg)
        world_size = torch.distributed.get_world_size(group=pg)

        # global ranks of the processes in the group
        ranks = torch.distributed.get_process_group_ranks(pg)
    else:
        rank = pg.rank
        world_size = pg.world_size
        ranks = list(range(world_size))

    # local tensor in each process to store the result
    is_in_the_same_node = torch.tensor([0] * world_size, dtype=torch.int32)

    magic_message = b"magic_message"
    shm = None

    try:
        with contextlib.suppress(OSError):
            if rank == source_rank:
                # create a shared memory segment
                shm = shared_memory.SharedMemory(create=True, size=128)
                shm.buf[: len(magic_message)] = magic_message
                if isinstance(pg, ProcessGroup):
                    torch.distributed.broadcast_object_list(
                        [shm.name], src=ranks[source_rank], group=pg
                    )
                else:
                    pg.broadcast_obj(shm.name, src=source_rank)
                is_in_the_same_node[rank] = 1
            else:
                # try to open the shared memory segment
                if isinstance(pg, ProcessGroup):
                    recv = [None]
                    torch.distributed.broadcast_object_list(
                        recv, src=ranks[source_rank], group=pg
                    )
                    name = recv[0]
                else:
                    name = pg.broadcast_obj(None, src=source_rank)
                # fix to https://stackoverflow.com/q/62748654/9191338
                # Python incorrectly tracks shared memory even if it is not
                # created by the process. The following patch is a workaround.
                with patch(
                    "multiprocessing.resource_tracker.register",
                    lambda *args, **kwargs: None,
                ):
                    shm = shared_memory.SharedMemory(name=name)
                if shm.buf[: len(magic_message)] == magic_message:
                    is_in_the_same_node[rank] = 1
    except Exception as e:
        logger.error("Error ignored in is_in_the_same_node: %s", e)
    finally:
        if shm:
            shm.close()

    if isinstance(pg, ProcessGroup):
        torch.distributed.barrier(group=pg)
    else:
        pg.barrier()

    # clean up the shared memory segment
    with contextlib.suppress(OSError):
        if rank == source_rank and shm:
            shm.unlink()

    if isinstance(pg, ProcessGroup):
        torch.distributed.all_reduce(is_in_the_same_node, group=pg)
        aggregated_data = is_in_the_same_node
    else:
        aggregated_data = torch.zeros_like(is_in_the_same_node)
        for i in range(world_size):
            rank_data = pg.broadcast_obj(is_in_the_same_node, src=i)
            aggregated_data += rank_data

    return [x == 1 for x in aggregated_data.tolist()]


def initialize_tensor_parallel_group(
    tensor_model_parallel_size: int = 1,
    backend: str | None = None,
    group_name_suffix: str = "",
) -> GroupCoordinator:
    """Initialize a tensor parallel group for a specific model.

    This function creates a tensor parallel group that can be used with the
    patch_tensor_parallel_group context manager. It allows different models
    to use different tensor parallelism configurations.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model parallelism.
        backend: communication backend to use.
        group_name_suffix: optional suffix to make the group name unique.

    Returns:
        A GroupCoordinator for tensor parallelism that can be used with
        the patch_tensor_parallel_group context manager.

    Example usage:
        ```python
        # Initialize tensor parallel group for model1
        tp_group_model1 = initialize_tensor_parallel_group(
            tensor_model_parallel_size=4,
            group_name_suffix="model1"
        )

        # Use tensor parallelism for model1
        with patch_tensor_parallel_group(tp_group_model1):
            # Run model1 with tensor parallelism
            output1 = model1(input1)
        ```
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    # Ensure the world size is compatible with the parallelism configuration
    assert (
        world_size % tensor_model_parallel_size == 0
    ), f"World size ({world_size}) must be divisible by tensor_model_parallel_size ({tensor_model_parallel_size})"

    # Build the tensor model-parallel groups.
    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    tp_group_ranks = []
    for i in range(num_tensor_model_parallel_groups):
        ranks = list(
            range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        )
        tp_group_ranks.append(ranks)

    # Create TP group coordinator with a unique name
    group_name = f"tp_{group_name_suffix}" if group_name_suffix else "tp"
    tp_group = init_parallel_group_coordinator(
        tp_group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name=group_name,
    )

    return tp_group


def initialize_sequence_parallel_group(
    sequence_model_parallel_size: int = 1,
    backend: str | None = None,
    group_name_suffix: str = "",
) -> GroupCoordinator:
    """Initialize a sequence parallel group for a specific model.

    This function creates a sequence parallel group that can be used with the
    patch_sequence_parallel_group context manager. It allows different models
    to use different sequence parallelism configurations.

    Arguments:
        sequence_model_parallel_size: number of GPUs used for sequence model parallelism.
        backend: communication backend to use.
        group_name_suffix: optional suffix to make the group name unique.

    Returns:
        A GroupCoordinator for sequence parallelism that can be used with
        the patch_sequence_parallel_group context manager.

    Example usage:
        ```python
        # Initialize sequence parallel group for model2
        sp_group_model2 = initialize_sequence_parallel_group(
            sequence_model_parallel_size=2,
            group_name_suffix="model2"
        )

        # Use sequence parallelism for model2
        with patch_sequence_parallel_group(sp_group_model2):
            # Run model2 with sequence parallelism
            output2 = model2(input2)
        ```
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    # Ensure the world size is compatible with the parallelism configuration
    assert (
        world_size % sequence_model_parallel_size == 0
    ), f"World size ({world_size}) must be divisible by sequence_model_parallel_size ({sequence_model_parallel_size})"

    # Build the sequence model-parallel groups.
    num_sequence_model_parallel_groups: int = world_size // sequence_model_parallel_size
    sp_group_ranks = []

    for i in range(num_sequence_model_parallel_groups):
        # Create groups of consecutive ranks
        ranks = list(
            range(
                i * sequence_model_parallel_size, (i + 1) * sequence_model_parallel_size
            )
        )
        sp_group_ranks.append(ranks)

    # Create SP group coordinator with a unique name
    group_name = f"sp_{group_name_suffix}" if group_name_suffix else "sp"
    sp_group = init_parallel_group_coordinator(
        sp_group_ranks, get_world_group().local_rank, backend, group_name=group_name
    )

    return sp_group


# * QUERY
def get_world_group() -> GroupCoordinator:
    assert _WORLD is not None, "world group is not initialized"
    return _WORLD


# TP
def get_tp_group() -> GroupCoordinator:
    assert _TP is not None, "tensor model parallel group is not initialized"
    return _TP


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return get_tp_group().world_size


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return get_tp_group().rank_in_group


def get_sequence_parallel_world_size():
    """Return world size for the sequence parallel group."""
    return get_sp_group().world_size


def get_sequence_parallel_rank():
    """Return my rank for the sequence parallel group."""
    return get_sp_group().rank_in_group


def get_ulysses_parallel_world_size():
    return get_sp_group().ulysses_world_size


def get_ulysses_parallel_rank():
    return get_sp_group().ulysses_rank


def get_ring_parallel_world_size():
    return get_sp_group().ring_world_size


def get_ring_parallel_rank():
    return get_sp_group().ring_rank


# PP
def get_pp_group() -> PipelineGroupCoordinator:
    assert _PP is not None, "pipeline model parallel group is not initialized"
    return _PP


def get_pipeline_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    return get_pp_group().world_size


def get_pipeline_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    return get_pp_group().rank_in_group


def is_pipeline_first_stage():
    """Return True if in the first pipeline model parallel stage, False otherwise."""
    return get_pipeline_parallel_rank() == 0


def is_pipeline_last_stage():
    """Return True if in the last pipeline model parallel stage, False otherwise."""
    return get_pipeline_parallel_rank() == (get_pipeline_parallel_world_size() - 1)


# CFG
def get_cfg_group() -> GroupCoordinator:
    assert (
        _CFG is not None
    ), "classifier_free_guidance parallel group is not initialized"
    return _CFG


def get_classifier_free_guidance_world_size():
    """Return world size for the classifier_free_guidance parallel group."""
    return get_cfg_group().world_size


def get_classifier_free_guidance_rank():
    """Return my rank for the classifier_free_guidance parallel group."""
    return get_cfg_group().rank_in_group


# DP
def get_dp_group() -> GroupCoordinator:
    assert _DP is not None, "pipeline model parallel group is not initialized"
    return _DP


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return get_dp_group().world_size


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return get_dp_group().rank_in_group


def is_dp_last_group():
    """Return True if in the last data parallel group, False otherwise."""
    return (
        get_sequence_parallel_rank() == (get_sequence_parallel_world_size() - 1)
        and get_classifier_free_guidance_rank()
        == (get_classifier_free_guidance_world_size() - 1)
        and get_pipeline_parallel_rank() == (get_pipeline_parallel_world_size() - 1)
    )


def get_dit_world_size():
    """Return world size for the DiT model (excluding VAE)."""
    return (
        get_data_parallel_world_size()
        * get_classifier_free_guidance_world_size()
        * get_sequence_parallel_world_size()
        * get_pipeline_parallel_world_size()
        * get_tensor_model_parallel_world_size()
    )


# Add VAE getter functions
def get_vae_parallel_group() -> GroupCoordinator:
    assert _VAE is not None, "VAE parallel group is not initialized"
    return _VAE


def get_vae_parallel_world_size():
    """Return world size for the VAE parallel group."""
    return get_vae_parallel_group().world_size


def get_vae_parallel_rank():
    """Return my rank for the VAE parallel group."""
    return get_vae_parallel_group().rank_in_group


# * SET


def init_world_group(
    ranks: List[int], local_rank: int, backend: str
) -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=[ranks],
        local_rank=local_rank,
        torch_distributed_backend=backend,
    )


def model_parallel_is_initialized():
    """Check if tensor and pipeline parallel groups are initialized."""
    return (
        _DP is not None
        and _CFG is not None
        and _SP is not None
        and _PP is not None
        and _TP is not None
    )


def init_dit_group(
    dit_parallel_size: int,
    backend: str,
):
    global _DIT
    _DIT = torch.distributed.new_group(
        ranks=list(range(dit_parallel_size)), backend=backend
    )


def get_dit_group():
    assert _DIT is not None, "DIT group is not initialized"
    return _DIT


def init_vae_group(
    dit_parallel_size: int,
    vae_parallel_size: int,
    backend: str,
):
    # Initialize VAE group first
    global _VAE
    assert _VAE is None, "VAE parallel group is already initialized"
    vae_ranks = list(range(dit_parallel_size, dit_parallel_size + vae_parallel_size))
    _VAE = torch.distributed.new_group(ranks=vae_ranks, backend=backend)


def destroy_model_parallel() -> None:
    """Set the groups to none and destroy them."""
    global _TP
    if _TP:
        _TP.destroy()
    _TP = None

    global _SP
    if _SP:
        _SP.destroy()
    _SP = None

    global _DP
    if _DP:
        _DP.destroy()
    _DP = None


# xDit
# def destroy_model_parallel():
#     """Set the groups to none and destroy them."""
#     global _DP
#     if _DP:
#         _DP.destroy()
#     _DP = None
#
#     global _CFG
#     if _CFG:
#         _CFG.destroy()
#     _CFG = None
#
#     global _SP
#     if _SP:
#         _SP.destroy()
#     _SP = None
#
#     global _TP
#     if _TP:
#         _TP.destroy()
#     _TP = None
#
#     global _PP
#     if _PP:
#         _PP.destroy()
#     _PP = None
#
#     global _VAE
#     if _VAE:
#         _VAE.destroy()
#     _VAE = None


def destroy_distributed_environment():
    global _WORLD
    if _WORLD:
        _WORLD.destroy()
    _WORLD = None
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
