from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.kv_canary.runner.future_tensor import FutureTensor

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator


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
        self._last_req_pool_indices: Optional[torch.Tensor] = None
        self._last_seq_lens: Optional[torch.Tensor] = None

    def observe_forward_batch(
        self,
        *,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        self._last_req_pool_indices = req_pool_indices
        self._last_seq_lens = seq_lens

    def snapshot_nonidentity_future(self, *, stream: torch.cuda.Stream) -> FutureTensor:
        req_pool_indices = self._last_req_pool_indices
        seq_lens = self._last_seq_lens
        mapping = self._swa_allocator.full_to_swa_index_mapping
        device = mapping.device

        if (
            req_pool_indices is None
            or seq_lens is None
            or req_pool_indices.numel() == 0
        ):
            zero = torch.zeros(1, dtype=torch.int32, device=device)
            return FutureTensor.device_to_host(src_device=zero, stream=stream)

        req_to_token = self._req_to_token_pool.req_to_token

        with torch.cuda.stream(stream):
            rows = req_to_token[req_pool_indices]
            positions = torch.arange(rows.shape[1], device=rows.device)
            mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
            swa_indices = mapping[rows]
            count = ((swa_indices != rows) & mask).sum().to(torch.int32).view(1)

        return FutureTensor.device_to_host(src_device=count, stream=stream)
