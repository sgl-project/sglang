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
    """Allocate the FIA view and fill it with the common DCP page-table kernel.

    Lengths must fit req_to_token; an explicit num_pages must cover every
    local sequence. The graph caller sizes it from the CPU length upper bound.
    """
    local_seq_lens = get_dcp_lens(seq_lens, dcp_size, dcp_rank).to(torch.int32)
    if num_pages is None:
        max_len = int(local_seq_lens.max().item()) if local_seq_lens.numel() else 0
        num_pages = max(1, (max_len + physical_page_size - 1) // physical_page_size)
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
    )
    return block_tables, local_seq_lens


def build_mla_dcp_mtp_mask(
    prefix_lens: torch.Tensor,
    query_len: int,
    dcp_size: int,
    dcp_rank: int,
    *,
    max_local_kv_len: int | None = None,
) -> torch.Tensor:
    """Build FIA's rank-local causal mask for a fixed DSPARK verify window."""
    prefix_lens = prefix_lens.to(torch.int64)
    total_lens = prefix_lens + query_len
    local_seq_lens = get_dcp_lens(total_lens, dcp_size, dcp_rank).to(torch.int32)

    if max_local_kv_len is None:
        max_local_kv_len = (
            int(local_seq_lens.max().item()) if local_seq_lens.numel() else 0
        )

    # FIA rejects a zero-width mask, including for an empty local shard.
    mask = torch.ones(
        (prefix_lens.numel(), max(1, query_len), max(1, max_local_kv_len)),
        dtype=torch.bool,
        device=prefix_lens.device,
    )
    if query_len == 0 or max_local_kv_len == 0:
        return mask.contiguous()

    q_idx = torch.arange(query_len, device=prefix_lens.device, dtype=torch.int64)
    k_idx = torch.arange(max_local_kv_len, device=prefix_lens.device, dtype=torch.int64)
    last_visible = torch.div(
        prefix_lens[:, None] + q_idx[None, :] - dcp_rank,
        dcp_size,
        rounding_mode="floor",
    )
    mask = k_idx[None, None, :] > last_visible[:, :, None]
    mask |= k_idx[None, None, :] >= local_seq_lens.to(torch.int64)[:, None, None]
    return mask.contiguous()
