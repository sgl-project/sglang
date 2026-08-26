# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Storage helpers for replicated DCP Query projection weights."""

from __future__ import annotations

from typing import Protocol

import torch


class _AllGatherGroup(Protocol):
    world_size: int
    rank_in_group: int

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor: ...


def replicated_rank_slice(
    replicated_weight: torch.Tensor,
    *,
    local_shape: torch.Size | tuple[int, ...],
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Return the rank-local row slice of a full replicated weight."""

    if world_size <= 1:
        raise ValueError(f"world_size must be greater than one, got {world_size}.")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}.")
    if len(local_shape) == 0 or replicated_weight.ndim == 0:
        raise ValueError("DCP replicated Query weights must have at least one axis.")

    expected_shape = (local_shape[0] * world_size, *local_shape[1:])
    if tuple(replicated_weight.shape) != expected_shape:
        raise ValueError(
            "Replicated Query weight shape does not match the local row shard: "
            f"expected {expected_shape}, got {tuple(replicated_weight.shape)}."
        )
    return replicated_weight.narrow(0, rank * local_shape[0], local_shape[0])


def bind_parameter_to_replicated_rank_slice_(
    parameter: torch.nn.Parameter,
    replicated_weight: torch.Tensor,
    *,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Make a local row-sharded parameter own a view of replicated storage.

    The full replicated tensor remains the decode weight. The original
    Parameter object is retained so model loaders and parameter-name based
    update APIs continue to write the local shard in place.
    """

    if parameter.dtype != replicated_weight.dtype:
        raise TypeError(
            "Local and replicated Query weights must have the same dtype, got "
            f"{parameter.dtype} and {replicated_weight.dtype}."
        )
    if parameter.device != replicated_weight.device:
        raise ValueError(
            "Local and replicated Query weights must share a device, got "
            f"{parameter.device} and {replicated_weight.device}."
        )

    local_weight = replicated_rank_slice(
        replicated_weight,
        local_shape=parameter.shape,
        rank=rank,
        world_size=world_size,
    )
    if tuple(local_weight.stride()) != tuple(parameter.stride()):
        raise ValueError(
            "Replicated Query rank slice must preserve the local weight layout: "
            f"expected stride {tuple(parameter.stride())}, got "
            f"{tuple(local_weight.stride())}."
        )

    with torch.no_grad():
        parameter.data = local_weight
    return local_weight


def refresh_replicated_weight_(
    local_weight: torch.Tensor,
    replicated_weight: torch.Tensor,
    *,
    group: _AllGatherGroup,
) -> None:
    """Refresh graph-stable replicated storage after an online weight update."""

    gathered = group.all_gather(local_weight.contiguous(), dim=0)
    if (
        gathered.shape != replicated_weight.shape
        or gathered.dtype != replicated_weight.dtype
        or gathered.device != replicated_weight.device
    ):
        raise ValueError(
            "Refreshed replicated Query weight does not match its persistent "
            f"buffer: got shape={tuple(gathered.shape)}, dtype={gathered.dtype}, "
            f"device={gathered.device}; expected "
            f"shape={tuple(replicated_weight.shape)}, "
            f"dtype={replicated_weight.dtype}, device={replicated_weight.device}."
        )
    replicated_weight.copy_(gathered)
