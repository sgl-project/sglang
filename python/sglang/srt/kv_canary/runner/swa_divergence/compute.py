from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def compute_swa_live_divergence(
    *,
    swa_allocator: "SWATokenToKVPoolAllocator",
    req_to_token_pool: "ReqToTokenPool",
    forward_batch: "ForwardBatch",
) -> torch.Tensor:
    """Count non-identity (full, swa) index pairs in the live req_to_token range."""
    full_to_swa_index_mapping = swa_allocator.full_to_swa_index_mapping
    device = full_to_swa_index_mapping.device
    req_pool_indices = forward_batch.req_pool_indices
    seq_lens = forward_batch.seq_lens

    if req_pool_indices.numel() == 0:
        return torch.zeros(1, dtype=torch.int32, device=device)

    req_to_token = req_to_token_pool.req_to_token
    rows = req_to_token[req_pool_indices]
    positions = torch.arange(rows.shape[1], device=rows.device)
    mask = positions[None, :] < seq_lens[:, None]
    swa_indices = full_to_swa_index_mapping[rows]
    return ((swa_indices != rows) & mask).sum().to(torch.int32).view(1)
