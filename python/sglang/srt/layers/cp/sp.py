# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel (SP) residual-stream helpers over a collocated prefill CP group.

Shared by hybrid linear-attention models that keep the residual stream
sequence-sharded for the whole layer stack when the CP group is the TP group
(attn_cp_size == tp_size). The functions follow the Kimi-K3 prefill-CP
implementation (branch linear-attn-prefill-cp-ulysses, SGLANG_K3_CP_SP_RESIDUAL)
so both models share one row-order contract:

- a rank's local rows are exactly ``ZigzagCPStrategy.shard_hidden_states``:
  [block r of every request] ++ [block 2C-1-r of every request], zero-padded
  to the uniform physical per-rank row count;
- ``sp_cp_all_gather`` / ``sp_cp_reduce_scatter`` move equal rank-major row
  blocks over the CP(=TP) group with no reorder (valid for token-wise work);
- ``sp_cp_reduce_scatter_rows`` first permutes FULL logical rows (global token
  order) into the rank-major zigzag layout so a reduce-scatter lands every
  rank its own local rows.
"""

from __future__ import annotations

import torch

from sglang.srt.distributed import get_pp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.runtime_context import get_parallel


def sp_cp_static_enabled() -> bool:
    """Whether the SP residual stream can be used: collocated linear-attention
    prefill CP (attn_cp_size == tp_size > 1), no attention DP, TP-sharded MoE
    (no a2a backend) and a single PP rank. Pure function of the topology, so it
    is identical on every rank; evaluate once at module construction."""
    parallel = get_parallel()
    return bool(
        parallel.enable_linear_attn_cp
        and parallel.attn_cp_size > 1
        and parallel.attn_cp_size == parallel.tp_size
        and parallel.attn_dp_size == 1
        and get_moe_a2a_backend().is_none()
        and get_pp_group().world_size == 1
    )


def sp_cp_all_gather(x: torch.Tensor) -> torch.Tensor:
    """Rank-major all-gather of equal-length local row blocks over the CP group."""
    group = get_parallel().attn_cp_group
    x = x.contiguous()
    with use_symmetric_memory(group, disabled=not is_allocation_symmetric()):
        out = torch.empty(
            (x.shape[0] * group.world_size, *x.shape[1:]),
            dtype=x.dtype,
            device=x.device,
        )
    group.all_gather_into_tensor(out, x)
    return out


def sp_cp_rank_major_blocks(x: torch.Tensor, metadata, cp_size: int) -> torch.Tensor:
    """Permute full logical rows into the concatenation of every CP rank's local
    zigzag block (rank r: block r of each request, then block 2C-1-r of each
    request), each block zero-padded to the physical per-rank length.

    One flat chunk list and one ``torch.cat`` (a single copy of ``x``), with a
    shared zero pad slab; no per-rank intermediate concatenations."""
    segments = 2 * cp_size
    bs = len(metadata.split_list) // segments
    chunks = torch.split(x[: metadata.total_seq_lens], metadata.split_list, dim=0)
    phys = max(metadata.per_rank_actual_token)
    logical = metadata.per_rank_logical_token or metadata.per_rank_actual_token
    max_pad = max(phys - n for n in logical)
    pad_slab = x.new_zeros((max_pad, *x.shape[1:])) if max_pad > 0 else None
    parts = []
    for rank in range(cp_size):
        parts.extend(chunks[i] for i in range(rank, bs * segments, segments))
        parts.extend(
            chunks[i] for i in range(segments - 1 - rank, bs * segments, segments)
        )
        pad = phys - logical[rank]
        assert pad >= 0, (phys, logical[rank])
        if pad:
            parts.append(pad_slab[:pad])
    return torch.cat(parts, dim=0)


def sp_cp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    """Sum rank-major row blocks over the CP group; return this rank's block."""
    group = get_parallel().attn_cp_group
    x = x.contiguous()
    with use_symmetric_memory(group, disabled=not is_allocation_symmetric()):
        out = torch.empty(
            (x.shape[0] // group.world_size, *x.shape[1:]),
            dtype=x.dtype,
            device=x.device,
        )
    group.reduce_scatter_tensor(out, x)
    return out


def sp_cp_reduce_scatter_rows(partial: torch.Tensor, forward_batch) -> torch.Tensor:
    """Sum TP-partial full-row outputs (global token order) over the CP(=TP)
    group and keep this rank's zigzag rows: one reduce-scatter instead of
    all-reduce + re-shard."""
    parallel = get_parallel()
    blocks = sp_cp_rank_major_blocks(
        partial, forward_batch.attn_cp_metadata, parallel.attn_cp_size
    )
    return sp_cp_reduce_scatter(blocks)


__all__ = [
    "sp_cp_all_gather",
    "sp_cp_rank_major_blocks",
    "sp_cp_reduce_scatter",
    "sp_cp_reduce_scatter_rows",
    "sp_cp_static_enabled",
]
