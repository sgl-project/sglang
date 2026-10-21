"""Ascend FIA mask adaptation for MLA decode context parallelism."""

from __future__ import annotations

import torch

from sglang.srt.layers.dcp.layout import get_dcp_chain_spec_lens, get_dcp_lens


def build_mla_dcp_mtp_mask(
    prefix_lens: torch.Tensor,
    query_len: int,
    dcp_size: int,
    dcp_rank: int,
    *,
    page_size: int = 1,
    max_local_kv_len: int | None = None,
    local_query_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Adapt common per-query DCP frontiers to FIA's fixed verify mask."""
    prefix_lens = prefix_lens.to(torch.int64)
    if max_local_kv_len is None:
        local_seq_lens = get_dcp_lens(
            prefix_lens + query_len,
            dcp_size,
            dcp_rank,
            interleave_size=page_size,
        ).to(torch.int32)
        max_local_kv_len = (
            int(local_seq_lens.max().item()) if local_seq_lens.numel() else 0
        )

    # FIA rejects a zero-width mask, including for an empty local shard.
    if query_len == 0 or max_local_kv_len == 0:
        return torch.ones(
            (prefix_lens.numel(), max(1, query_len), max(1, max_local_kv_len)),
            dtype=torch.bool,
            device=prefix_lens.device,
        ).contiguous()

    if local_query_lens is None:
        local_query_lens = get_dcp_chain_spec_lens(
            prefix_lens + query_len,
            query_len,
            dcp_size,
            dcp_rank,
            interleave_size=page_size,
        )
    k_idx = torch.arange(max_local_kv_len, device=prefix_lens.device, dtype=torch.int64)
    mask = k_idx[None, None, :] >= local_query_lens.view(
        prefix_lens.numel(), query_len, 1
    )
    return mask.contiguous()
