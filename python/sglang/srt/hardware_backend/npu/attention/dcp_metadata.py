"""Ascend-specific metadata builders for MLA decode context parallelism."""

from __future__ import annotations

import torch
import triton

from sglang.kernels.ops.attention.dcp_kernels import create_mla_kv_page_table_for_dcp
from sglang.srt.layers.dcp.layout import get_dcp_lens


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
    """Allocate the FIA view and fill it with the common DCP page-table kernel."""
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
        max_len = int(local_seq_lens.max().item()) if local_seq_lens.numel() else 0
        num_pages = max(1, (max_len + physical_page_size - 1) // physical_page_size)
    elif num_pages <= 0:
        raise ValueError(f"num_pages must be positive, got {num_pages}")
    block_tables = torch.zeros(
        (req_pool_indices.numel(), num_pages),
        dtype=torch.int32,
        device=req_to_token.device,
    )
    if not req_pool_indices.numel():
        return block_tables, local_seq_lens
    create_mla_kv_page_table_for_dcp[
        (req_pool_indices.numel(), triton.cdiv(num_pages, 128))
    ](
        req_to_token,
        req_pool_indices,
        local_seq_lens,
        block_tables,
        None,
        req_to_token.stride(0),
        block_tables.stride(0),
        1,
        PHYSICAL_PAGE_SIZE=physical_page_size,
        DCP_SIZE=dcp_size,
        DCP_RANK=dcp_rank,
        PAGES_PER_BLOCK=128,
        HAS_V2P=False,
        NUM_PAGES=num_pages,
        MAX_SEQ_LEN=req_to_token.shape[1],
    )
    return block_tables, local_seq_lens


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
