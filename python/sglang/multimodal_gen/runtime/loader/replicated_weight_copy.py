# SPDX-License-Identifier: Apache-2.0
"""Host-to-device weight copies shared by ranks that load the same checkpoint.

Under sequence parallelism (and CFG / data parallelism) every rank of a
single-node server reads the same checkpoint whole and copies it into its own
GPU: the host does the page faults and pageable transfers out of the checkpoint
mapping once per rank. When every rank of a group loads the same component at
the same time, each rank instead copies one 1/world slice of a large tensor and
an all-gather between the devices assembles the rest: the same bytes, with the
host work split across the ranks.
"""

import torch
import torch.distributed as dist

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_world_group
from sglang.multimodal_gen.runtime.distributed.utils import all_gather_single
from sglang.multimodal_gen.runtime.server_args import ServerArgs

# below this a tensor is copied by every rank itself: the collective's fixed
# cost would outweigh the host bytes it saves
REPLICATED_COPY_MIN_BYTES = 4 << 20


def replicated_load_group(
    server_args: ServerArgs, device: torch.device
) -> dist.ProcessGroup | None:
    """The device group whose ranks all load a whole component at once.

    In a monolithic single-node server every rank builds the same pipeline in
    the same order, so a component that is not FSDP-sharded is read whole by
    each of them. Callers pass only loads that every rank performs.
    """
    if (
        not envs.SGLANG_DIFFUSION_REPLICATED_WEIGHT_COPY
        or device.type != "cuda"
        or server_args.nnodes != 1
        or server_args.disagg_role != RoleType.MONOLITHIC
        or not dist.is_initialized()
        or dist.get_world_size() < 2
    ):
        return None
    return get_world_group().device_group


def uses_replicated_copy(tensor: torch.Tensor) -> bool:
    """Whether a host tensor goes through the shared copy; the same answer on
    every rank, which reads the same checkpoint."""
    return (
        tensor.device.type == "cpu"
        and tensor.is_contiguous()
        and tensor.numel() * tensor.element_size() >= REPLICATED_COPY_MIN_BYTES
    )


def replicated_copy_to_device(
    tensor: torch.Tensor, device: torch.device, group: dist.ProcessGroup
) -> torch.Tensor:
    """``tensor.to(device)`` for a contiguous host tensor whose bytes every
    rank of ``group`` holds."""
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    raw = tensor.reshape(-1).view(torch.uint8)
    nbytes = raw.numel()
    per_rank = -(-nbytes // world)
    gathered = torch.empty(per_rank * world, dtype=torch.uint8, device=device)
    begin = min(rank * per_rank, nbytes)
    end = min(begin + per_rank, nbytes)
    local = gathered[rank * per_rank : (rank + 1) * per_rank]
    local[: end - begin].copy_(raw[begin:end])
    all_gather_single(gathered, local, group=group)
    return gathered[:nbytes].view(tensor.dtype).view(tensor.shape)
