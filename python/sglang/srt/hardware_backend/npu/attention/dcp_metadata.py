"""FIA metadata for Ascend decode context parallelism."""

from __future__ import annotations

import torch


def build_gqa_dcp_extend_layout(
    lengths: list[int], page_size: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Map request-ordered KV rows into an ephemeral paged prefill buffer."""
    page_counts = [(length + page_size - 1) // page_size for length in lengths]
    table = torch.zeros((len(page_counts), max(page_counts)), dtype=torch.int32)
    offset = 0
    rows = []
    for request, (length, count) in enumerate(zip(lengths, page_counts)):
        table[request, :count] = torch.arange(offset, offset + count)
        rows.append(torch.arange(offset * page_size, offset * page_size + length))
        offset += count
    return table, torch.cat(rows), offset * page_size


def build_mla_dcp_mtp_mask(
    local_query_lens: torch.Tensor,
    query_len: int,
    mask_width: int,
) -> torch.Tensor:
    """Adapt common per-query DCP frontiers to FIA's fixed verify mask."""
    local_lens = local_query_lens.view(-1, query_len, 1)
    k_idx = torch.arange(mask_width, device=local_lens.device)
    return (k_idx >= local_lens).contiguous()
