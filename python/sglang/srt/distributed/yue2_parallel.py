# SPDX-License-Identifier: Apache-2.0
"""YuE2 sequence-parallel helpers built on SGLang distributed primitives.

These helpers keep the sharding policy in one place so the AR, NAR, and VAE
paths can later opt into sequence parallelism without scattering collective
logic through model code.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _group_size(group: Optional[dist.ProcessGroup]) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    if group is None:
        return dist.get_world_size()
    return dist.get_world_size(group)


def _group_rank(group: Optional[dist.ProcessGroup]) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    if group is None:
        return dist.get_rank()
    return dist.get_rank(group)


def shard_yue2_tensor(
    tensor: torch.Tensor,
    *,
    dim: int = 1,
    group: Optional[dist.ProcessGroup] = None,
) -> tuple[torch.Tensor, int]:
    """Shard a sequence tensor across *group* with tail padding.

    Returns the local shard and the original global length so callers can trim
    padding after an all-gather.
    """
    world_size = _group_size(group)
    orig_len = tensor.shape[dim]
    if world_size <= 1:
        return tensor, orig_len

    local_len = (orig_len + world_size - 1) // world_size
    pad_len = local_len * world_size - orig_len
    if pad_len:
        pads = [0, 0] * (tensor.dim() - 1 - dim) + [0, pad_len]
        tensor = F.pad(tensor, pads)
    rank = _group_rank(group)
    return tensor.narrow(dim, rank * local_len, local_len).contiguous(), orig_len


def gather_yue2_tensor(
    tensor: torch.Tensor,
    *,
    orig_len: int,
    dim: int = 1,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """All-gather a YuE2 sequence tensor and trim tail padding."""
    world_size = _group_size(group)
    if world_size <= 1:
        return tensor

    shards = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(shards, tensor.contiguous(), group=group)
    full = torch.cat(shards, dim=dim)
    return full.narrow(dim, 0, orig_len)
