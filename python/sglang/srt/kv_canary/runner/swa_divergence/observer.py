from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.kv_canary.runner.future_tensor import FutureTensor

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class SwaLiveDivergenceObserver:
    """Counts non-identity (full, swa) index pairs in the live req_to_token range."""

    def __init__(
        self,
        *,
        swa_allocator: "SWATokenToKVPoolAllocator",
        req_to_token_pool: "ReqToTokenPool",
    ) -> None:
        self._swa_allocator = swa_allocator
        self._req_to_token_pool = req_to_token_pool

    def snapshot_nonidentity_future(
        self,
        *,
        forward_batch: "ForwardBatch",
        stream: torch.cuda.Stream,
    ) -> FutureTensor:
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        full_to_swa_index_mapping = self._swa_allocator.full_to_swa_index_mapping
        device = full_to_swa_index_mapping.device

        if req_pool_indices.numel() == 0:
            zero = torch.zeros(1, dtype=torch.int32, device=device)
            return FutureTensor.device_to_host(src_device=zero, stream=stream)

        req_to_token = self._req_to_token_pool.req_to_token

        with torch.cuda.stream(stream):
            rows = req_to_token[req_pool_indices]
            positions = torch.arange(rows.shape[1], device=rows.device)
            mask = positions[None, :] < seq_lens[:, None]
            swa_indices = full_to_swa_index_mapping[rows]
            count = ((swa_indices != rows) & mask).sum().to(torch.int32).view(1)

        return FutureTensor.device_to_host(src_device=count, stream=stream)
