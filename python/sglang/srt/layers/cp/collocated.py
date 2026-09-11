# SPDX-License-Identifier: Apache-2.0
"""Collocated prefill CP helpers: the residual stream stays CP-sharded.

Shared by hybrid linear-attention models that keep the residual stream
sharded over the CP group for the whole layer stack when the CP group is the
TP group (attn_cp_size == tp_size), so every collective (MoE all-gather /
reduce-scatter, GDN gather, attention K/V gather) runs over that one group.
The functions follow the Kimi-K3 prefill-CP implementation (branch
linear-attn-prefill-cp-ulysses) so both models share one row-order contract:

- a rank's local rows are exactly ``ZigzagCPStrategy.shard_hidden_states``:
  [block r of every request] ++ [block 2C-1-r of every request], zero-padded
  to the uniform physical per-rank row count;
- ``cp_all_gather_blocks`` / ``cp_reduce_scatter_blocks`` move equal rank-major
  row blocks over the CP(=TP) group with no reorder (valid for token-wise work);
  ``cp_all_gather_blocks_multi`` gathers several such tensors (e.g. MoE rows
  plus their top-k weights and ids) in one NCCL group launch;
- ``cp_reduce_scatter_global_rows`` first permutes FULL logical rows (global
  token order) into the rank-major zigzag layout so a reduce-scatter lands
  every rank its own local rows.
"""

from __future__ import annotations

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.runtime_context import get_parallel


def cp_all_gather_blocks(x: torch.Tensor) -> torch.Tensor:
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


def cp_all_gather_blocks_multi(*tensors: torch.Tensor) -> list[torch.Tensor]:
    """Rank-major all-gather of several equal-row tensors over the CP group.

    On the torch.distributed path the collectives are coalesced into one NCCL
    group (one kernel launch for all of them); on the pynccl / symmetric-memory
    path they are issued one after another."""
    group = get_parallel().attn_cp_group
    inputs = [x.contiguous() for x in tensors]
    with use_symmetric_memory(group, disabled=not is_allocation_symmetric()):
        outputs = [
            torch.empty(
                (x.shape[0] * group.world_size, *x.shape[1:]),
                dtype=x.dtype,
                device=x.device,
            )
            for x in inputs
        ]
    pynccl = group.pynccl_comm
    torch_dist_path = pynccl is None or (
        pynccl.disabled and not group.is_symmetric_memory_enabled()
    )
    coalesce = getattr(torch.distributed, "_coalescing_manager", None)
    if torch_dist_path and coalesce is not None and len(inputs) > 1:
        with coalesce(group=group.device_group, device=inputs[0].device):
            for out, x in zip(outputs, inputs):
                torch.distributed.all_gather_into_tensor(
                    out, x, group=group.device_group
                )
    else:
        for out, x in zip(outputs, inputs):
            group.all_gather_into_tensor(out, x)
    return outputs


def cp_rank_major_blocks(x: torch.Tensor, metadata, cp_size: int) -> torch.Tensor:
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


def cp_reduce_scatter_blocks(x: torch.Tensor) -> torch.Tensor:
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


def cp_reduce_scatter_global_rows(partial: torch.Tensor, forward_batch) -> torch.Tensor:
    """Sum TP-partial full-row outputs (global token order) over the CP(=TP)
    group and keep this rank's zigzag rows: one reduce-scatter instead of
    all-reduce + re-shard."""
    parallel = get_parallel()
    blocks = cp_rank_major_blocks(
        partial, forward_batch.attn_cp_metadata, parallel.attn_cp_size
    )
    return cp_reduce_scatter_blocks(blocks)


__all__ = [
    "cp_all_gather_blocks",
    "cp_all_gather_blocks_multi",
    "cp_rank_major_blocks",
    "cp_reduce_scatter_blocks",
    "cp_reduce_scatter_global_rows",
]
