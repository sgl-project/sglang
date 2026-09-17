"""Ascend-specific metadata builders for MLA decode context parallelism."""

from __future__ import annotations

import torch

from sglang.srt.layers.dcp.layout import get_dcp_lens, maybe_dcp_kernel_indices
from sglang.srt.layers.dcp.metadata import DecodeContextParallelMetadata
from sglang.srt.runtime_context import get_parallel


def prepare_decode_context_parallel_metadata_npu(
    seq_lens: torch.Tensor,
    extend_prefix_lens_cpu,
    extend_seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens_sum: int,
    kv_cache_dtype,
    kv_cache_device,
) -> DecodeContextParallelMetadata:
    """Build the prefix indirection consumed by the Ascend extend path.

    Ascend FIA builds its decode block table separately and gathers prefix KV
    directly from the two physical MLA buffers.  The CUDA-only indptr, packed
    indices, and temporary KV buffer in the common metadata contract are
    therefore intentionally left unset.
    """
    parallel = get_parallel()
    device = req_to_token.device
    prefix_lens_cpu = [int(x) for x in extend_prefix_lens_cpu]

    prefix_parts = []
    for batch_idx, prefix_len in enumerate(prefix_lens_cpu):
        if prefix_len == 0:
            continue
        req_idx = int(req_pool_indices[batch_idx].item())
        prefix_parts.append(req_to_token[req_idx, :prefix_len])
    if prefix_parts:
        dcp_prefix_kv_indices = torch.cat(prefix_parts).to(torch.int32)
    else:
        dcp_prefix_kv_indices = torch.empty(0, dtype=torch.int32, device=device)

    dcp_local_prefix_kv_indices = maybe_dcp_kernel_indices(
        dcp_prefix_kv_indices,
        parallel.dcp_size,
        parallel.dcp_rank,
    )
    return DecodeContextParallelMetadata(
        dcp_local_prefix_kv_indices=dcp_local_prefix_kv_indices,
    )


def build_mla_dcp_local_block_tables(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    physical_page_size: int,
    dcp_size: int,
    dcp_rank: int,
    *,
    num_pages: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the rank-local FIA page table from widened virtual token ids."""
    if physical_page_size <= 0:
        raise ValueError(
            f"physical_page_size must be positive, got {physical_page_size}"
        )
    if dcp_size <= 0 or not 0 <= dcp_rank < dcp_size:
        raise ValueError(
            f"invalid DCP topology: dcp_size={dcp_size}, dcp_rank={dcp_rank}"
        )

    local_seq_lens = get_dcp_lens(seq_lens, dcp_size, dcp_rank).to(torch.int32)
    if num_pages is None:
        max_local_len = (
            int(local_seq_lens.max().item()) if local_seq_lens.numel() > 0 else 0
        )
        num_pages = max(
            1, (max_local_len + physical_page_size - 1) // physical_page_size
        )
    elif num_pages <= 0:
        raise ValueError(f"num_pages must be positive, got {num_pages}")

    local_page_offsets = torch.arange(
        num_pages, dtype=torch.long, device=req_to_token.device
    )
    global_positions = dcp_rank + local_page_offsets * physical_page_size * dcp_size
    req_rows = req_pool_indices.to(device=req_to_token.device, dtype=torch.long)
    # A graph table has a fixed maximum width and can include one rounded-up
    # page beyond the request-table width. Clamp the read and mask that page.
    positions_in_range = global_positions < req_to_token.shape[1]
    safe_global_positions = global_positions.clamp(max=req_to_token.shape[1] - 1)
    virtual_locs = req_to_token[req_rows[:, None], safe_global_positions[None, :]]
    block_tables = (virtual_locs // dcp_size // physical_page_size).to(torch.int32)
    valid_pages = (
        local_page_offsets[None, :] * physical_page_size
        < local_seq_lens.to(req_to_token.device)[:, None]
    ) & positions_in_range[None, :]
    block_tables.masked_fill_(~valid_pages, 0)
    return block_tables.contiguous(), local_seq_lens


def build_mla_dcp_mtp_mask(
    prefix_lens: torch.Tensor,
    query_lens: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    *,
    max_query_len: int | None = None,
    max_local_kv_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the rank-local causal mask consumed by FIA target verification."""
    if prefix_lens.ndim != 1 or query_lens.ndim != 1:
        raise ValueError("prefix_lens and query_lens must both be rank-1 tensors")
    if prefix_lens.numel() != query_lens.numel():
        raise ValueError(
            "prefix_lens and query_lens must have the same batch size, got "
            f"{prefix_lens.numel()} and {query_lens.numel()}"
        )
    if dcp_size <= 0 or not 0 <= dcp_rank < dcp_size:
        raise ValueError(
            f"invalid DCP topology: dcp_size={dcp_size}, dcp_rank={dcp_rank}"
        )

    prefix_lens = prefix_lens.to(torch.int64)
    query_lens = query_lens.to(device=prefix_lens.device, dtype=torch.int64)
    total_lens = prefix_lens + query_lens
    local_seq_lens = get_dcp_lens(total_lens, dcp_size, dcp_rank).to(torch.int32)

    if max_query_len is None:
        max_q_len = int(query_lens.max().item()) if query_lens.numel() else 0
    else:
        if max_query_len <= 0:
            raise ValueError(f"max_query_len must be positive, got {max_query_len}")
        max_q_len = max_query_len
    if max_local_kv_len is None:
        max_local_kv_len = (
            int(local_seq_lens.max().item()) if local_seq_lens.numel() else 0
        )
    elif max_local_kv_len <= 0:
        raise ValueError(f"max_local_kv_len must be positive, got {max_local_kv_len}")

    # FIA rejects a zero-width mask, including for an empty local shard.
    mask = torch.ones(
        (prefix_lens.numel(), max(1, max_q_len), max(1, max_local_kv_len)),
        dtype=torch.bool,
        device=prefix_lens.device,
    )
    if max_q_len == 0 or max_local_kv_len == 0:
        return mask.contiguous(), local_seq_lens

    q_idx = torch.arange(max_q_len, device=prefix_lens.device, dtype=torch.int64)
    k_idx = torch.arange(max_local_kv_len, device=prefix_lens.device, dtype=torch.int64)
    last_visible = torch.div(
        prefix_lens[:, None] + q_idx[None, :] - dcp_rank,
        dcp_size,
        rounding_mode="floor",
    )
    mask = k_idx[None, None, :] > last_visible[:, :, None]
    mask |= q_idx[None, :, None] >= query_lens[:, None, None]
    mask |= k_idx[None, None, :] >= local_seq_lens.to(torch.int64)[:, None, None]
    return mask.contiguous(), local_seq_lens
