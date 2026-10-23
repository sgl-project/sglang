# SPDX-License-Identifier: Apache-2.0
"""Tensor and Ulysses sequence parallelism for the Nomas DiT.

TP uses the framework's parallel linear layers. Ulysses SP is model-specific:
Nomas attention is a FlexAttention block mask over its own segment KV cache,
which USPAttention cannot express, so the head/sequence exchange around
attention is done here with the shared collectives.
"""

import torch
from msgspec import Struct

from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
    sequence_model_parallel_all_to_all_4D,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_tp_rank,
    get_tp_world_size,
    get_ulysses_ctx,
    model_parallel_is_initialized,
)


class ParallelInfo(Struct, frozen=True):
    tp_size: int = 1
    tp_rank: int = 0
    sp_size: int = 1  # Ulysses degree; ring attention is not supported
    sp_rank: int = 0

    @property
    def head_shards(self) -> int:
        return self.tp_size * self.sp_size


def current_parallel_info() -> ParallelInfo:
    if not model_parallel_is_initialized():
        return ParallelInfo()
    if get_ring_parallel_world_size() > 1:
        raise ValueError("Nomas supports Ulysses sequence parallelism only, not ring")
    sp_size, sp_rank = get_ulysses_ctx()
    return ParallelInfo(get_tp_world_size(), get_tp_rank(), sp_size, sp_rank)


def seq_to_heads(x: torch.Tensor, info: ParallelInfo) -> torch.Tensor:
    """[1, S_local, H, D] -> [1, S_global, H/U, D] (Ulysses input exchange)."""
    if info.sp_size == 1:
        return x
    return sequence_model_parallel_all_to_all_4D(x, scatter_dim=2, gather_dim=1)


def heads_to_seq(x: torch.Tensor, info: ParallelInfo) -> torch.Tensor:
    """[1, S_global, H/U, D] -> [1, S_local, H, D] (Ulysses output exchange)."""
    if info.sp_size == 1:
        return x
    return sequence_model_parallel_all_to_all_4D(x, scatter_dim=1, gather_dim=2)


def gather_heads(x: torch.Tensor, info: ParallelInfo) -> torch.Tensor:
    """[1, S, H/U, D] -> [1, S, H, D] for the replicated text stream."""
    if info.sp_size == 1:
        return x
    return sequence_model_parallel_all_gather(x, dim=2)


def gather_seq(x: torch.Tensor, info: ParallelInfo) -> torch.Tensor:
    """[1, S_local, C] -> [1, S_global, C]."""
    if info.sp_size == 1:
        return x
    return sequence_model_parallel_all_gather(x, dim=1)


def local_token_range(num_tokens: int, info: ParallelInfo) -> tuple[int, int]:
    if num_tokens % info.sp_size:
        raise ValueError(
            f"{num_tokens} video tokens are not divisible by the sequence-parallel degree {info.sp_size}"
        )
    local = num_tokens // info.sp_size
    return info.sp_rank * local, (info.sp_rank + 1) * local
