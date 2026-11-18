"""Ascend FIA mask adaptation for MLA decode context parallelism."""

from __future__ import annotations

import torch


def build_mla_dcp_mtp_mask(
    local_query_lens: torch.Tensor,
    query_len: int,
    mask_width: int,
) -> torch.Tensor:
    """Adapt common per-query DCP frontiers to FIA's fixed verify mask."""
    local_lens = local_query_lens.view(-1, query_len, 1)
    k_idx = torch.arange(mask_width, device=local_lens.device)
    return (k_idx >= local_lens).contiguous()
