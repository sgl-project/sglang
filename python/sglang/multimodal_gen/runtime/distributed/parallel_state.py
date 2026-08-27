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

"""sglang-diffusion distributed state.

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
import datetime
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

_WORLD: GroupCoordinator | None = None
_TP: GroupCoordinator | None = None
_SP: SequenceParallelGroupCoordinator | None = None
_PP: PipelineGroupCoordinator | None = None
_CFG: GroupCoordinator | None = None
_DP: GroupCoordinator | None = None
_DIT: ProcessGroup | None = None
_VAE: ProcessGroup | None = None

TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])


def _split_tensor_dict(
    tensor_dict: dict[str, torch.Tensor | Any],
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


def init_parallel_group_coordinator(
    group_ranks: List[List[int]],
    local_rank: int,
    backend: str,
    parallel_mode: str,
    **kwargs,
) -> GroupCoordinator:
    """Return a group coordinator for the given parallel mode."""
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


def get_tp_group() -> GroupCoordinator:
    assert _TP is not None, "tensor model parallel group is not initialized"
    return _TP


def init_distributed_environment(
    world_size: int = 1,
    rank: int = 0,
    distributed_init_method: str = "env://",
    local_rank: int = 0,
    backend: str | None = None,
    device_id: torch.device | None = None,
    timeout: int | None = None,
):
    # Determine the appropriate backend based on the platform
    from sglang.multimodal_gen.runtime.platforms import current_platform

    if backend is None:
        backend = current_platform.get_torch_distributed_backend_str()
        logger.info(
            "Using %s backend for %s platform", backend, current_platform.device_name
        )

    logger.debug(
        "world_size=%d rank=%d local_rank=%d "
        "distributed_init_method=%s backend=%s timeout=%s",
        world_size,
        rank,
        local_rank,
        distributed_init_method,
        backend,
        timeout,
    )
    if not torch.distributed.is_initialized():
        assert distributed_init_method is not None, (
            "distributed_init_method must be provided when initializing "
            "distributed environment"
        )

        # For MPS, MUSA, and XPU, don't pass device_id as it doesn't support device indices
        extra_args = (
            {}
            if (
                current_platform.is_mps()
                or current_platform.is_musa()
                or current_platform.is_npu()
                or current_platform.is_cpu()
                or current_platform.is_xpu()
            )
            else dict(device_id=device_id)
        )

        if timeout is not None:

            extra_args["timeout"] = datetime.timedelta(seconds=timeout)
            logger.info(f"Setting distributed timeout to {timeout} seconds")

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


def get_sp_group() -> SequenceParallelGroupCoordinator:
    assert _SP is not None, "sequence parallel group is not initialized"
    return _SP


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
        from sglang.multimodal_gen.runtime.platforms import current_platform

        backend = current_platform.get_torch_distributed_backend_str()
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

    try:
        from .parallel_groups import PROCESS_GROUP as _YC_PROCESS_GROUP
        from .parallel_groups import (
            set_seq_parallel_pg_by_sp_groups as _set_seq_parallel_pg_by_sp_groups,
        )
    except ImportError:
        _set_seq_parallel_pg_by_sp_groups = None

        class _DummyProcessGroup:
            ULYSSES_PG = torch.distributed.group.WORLD
            RING_PG = torch.distributed.group.WORLD

        PROCESS_GROUP = _DummyProcessGroup()
    else:
        # Build SGLang Diffusion SP sub-groups based on the true SP groups. This is
        # critical when TP>1, because SP groups may be strided in global ranks
        # (e.g., tp-sp order).
        sp_groups = rank_generator.get_ranks("sp")
        _set_seq_parallel_pg_by_sp_groups(
            sp_ulysses_degree=ulysses_degree,
            sp_ring_degree=ring_degree,
            rank=get_world_group().rank,
            sp_groups=sp_groups,
        )
        PROCESS_GROUP = _YC_PROCESS_GROUP

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
    cfg_degree: int = 1,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    dp_size: int = 1,
    distributed_init_method: str = "env://",
    dist_timeout: int | None = None,
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
        "Initializing distributed environment with world_size=%d, device=%s, timeout=%s",
        world_size,
        device,
        dist_timeout,
        main_process_only=False,
    )

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method=distributed_init_method,
        device_id=device,
        backend=current_platform.get_torch_distributed_backend_str(),
        timeout=dist_timeout,
    )
    initialize_model_parallel(
        data_parallel_size=dp_size,
        classifier_free_guidance_degree=cfg_degree,
        tensor_parallel_degree=tp_size,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        sequence_parallel_degree=sp_size,
    )

    # Only set CUDA device if we're on a CUDA platform
    if current_platform.is_cuda_alike():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    elif current_platform.is_npu():
        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)


def model_parallel_is_initialized() -> bool:
    """Check if model parallel groups are initialized."""
    return (
        _DP is not None
        and _CFG is not None
        and _SP is not None
        and _PP is not None
        and _TP is not None
    )


_TP_STATE_PATCHED = False


@contextmanager
def patch_tensor_parallel_group(tp_group: GroupCoordinator):
    """Patch the tp group temporarily until this function ends.

    This method is for draft workers of speculative decoding to run draft model
    with different tp degree from that of target model workers.

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


def get_tensor_model_parallel_world_size() -> int:
    """Return world size for the tensor model parallel group."""
    return get_tp_world_size()


def get_tensor_model_parallel_rank() -> int:
    """Return my rank for the tensor model parallel group."""
    return get_tp_rank()


def get_sequence_parallel_world_size() -> int:
    """Return world size for the sequence parallel group."""
    return get_sp_world_size()


def get_sequence_parallel_rank() -> int:
    """Return my rank for the sequence parallel group."""
    return get_sp_parallel_rank()


def get_ulysses_parallel_world_size() -> int:
    return get_sp_group().ulysses_world_size


def get_ulysses_parallel_rank() -> int:
    return get_sp_group().ulysses_rank


def get_ring_parallel_world_size() -> int:
    return get_sp_group().ring_world_size


def get_ring_parallel_rank() -> int:
    return get_sp_group().ring_rank


# PP
def get_pp_group() -> PipelineGroupCoordinator:
    assert _PP is not None, "pipeline model parallel group is not initialized"
    return _PP


def get_pipeline_parallel_world_size() -> int:
    """Return world size for the pipeline model parallel group."""
    return get_pp_group().world_size


def get_pipeline_parallel_rank() -> int:
    """Return my rank for the pipeline model parallel group."""
    return get_pp_group().rank_in_group


def is_pipeline_first_stage() -> bool:
    """Return True if in the first pipeline model parallel stage, False otherwise."""
    return get_pipeline_parallel_rank() == 0


def is_pipeline_last_stage() -> bool:
    """Return True if in the last pipeline model parallel stage, False otherwise."""
    return get_pipeline_parallel_rank() == (get_pipeline_parallel_world_size() - 1)


# CFG
def get_cfg_group() -> GroupCoordinator:
    assert (
        _CFG is not None
    ), "classifier_free_guidance parallel group is not initialized"
    return _CFG


def get_classifier_free_guidance_world_size() -> int:
    """Return world size for the classifier_free_guidance parallel group."""
    return get_cfg_group().world_size


def get_classifier_free_guidance_rank() -> int:
    """Return my rank for the classifier_free_guidance parallel group."""
    return get_cfg_group().rank_in_group


def get_data_parallel_world_size() -> int:
    """Return world size for the data parallel group."""
    return get_dp_world_size()


def get_data_parallel_rank() -> int:
    """Return my rank for the data parallel group."""
    return get_dp_rank()


def is_dp_last_group() -> bool:
    """Return True if in the last data parallel group, False otherwise."""
    return (
        get_sequence_parallel_rank() == (get_sequence_parallel_world_size() - 1)
        and get_classifier_free_guidance_rank()
        == (get_classifier_free_guidance_world_size() - 1)
        and get_pipeline_parallel_rank() == (get_pipeline_parallel_world_size() - 1)
    )


def get_dit_world_size() -> int:
    """Return world size for the DiT model (excluding VAE)."""
    return (
        get_data_parallel_world_size()
        * get_classifier_free_guidance_world_size()
        * get_sequence_parallel_world_size()
        * get_pipeline_parallel_world_size()
        * get_tensor_model_parallel_world_size()
    )


def get_vae_parallel_group() -> ProcessGroup:
    assert _VAE is not None, "VAE parallel group is not initialized"
    return _VAE


def get_vae_parallel_world_size() -> int:
    """Return world size for the VAE parallel group."""
    return torch.distributed.get_world_size(group=get_vae_parallel_group())


def get_vae_parallel_rank() -> int:
    """Return my rank for the VAE parallel group."""
    return torch.distributed.get_rank(group=get_vae_parallel_group())


def init_dit_group(
    dit_parallel_size: int,
    backend: str,
) -> None:
    global _DIT
    assert _DIT is None, "DIT group is already initialized"
    _DIT = torch.distributed.new_group(
        ranks=list(range(dit_parallel_size)), backend=backend
    )


def get_dit_group() -> ProcessGroup:
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
    global _TP, _SP, _DP, _CFG, _PP, _DIT, _VAE

    for group in (_TP, _SP, _DP, _CFG, _PP):
        if group is not None:
            group.destroy()

    for group in (_DIT, _VAE):
        if group is not None:
            torch.distributed.destroy_process_group(group)

    _TP, _SP, _DP, _CFG, _PP, _DIT, _VAE = (None,) * 7
