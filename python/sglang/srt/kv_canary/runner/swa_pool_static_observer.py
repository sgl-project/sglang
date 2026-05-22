from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.kv_canary.runner.future_tensor import FutureTensor

if TYPE_CHECKING:
    from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator


class SwaPoolStaticObserver:
    """Captures an install-time snapshot of the SWA allocator's
    ``full_to_swa_index_mapping`` and reports how many entries diverge from
    that baseline. Read-only: no hot-path hooks into the allocator."""

    def __init__(self, *, swa_allocator: "SWATokenToKVPoolAllocator") -> None:
        self._swa_allocator = swa_allocator
        mapping = swa_allocator.full_to_swa_index_mapping
        if mapping is None:
            raise RuntimeError(
                "SwaPoolStaticObserver requires SWA pool mapping to be registered "
                "before install; got None"
            )
        self._baseline_mapping: torch.Tensor = mapping.detach().clone()

    def snapshot_nonidentity_future(
        self, *, stream: torch.cuda.Stream
    ) -> FutureTensor:
        mapping = self._swa_allocator.full_to_swa_index_mapping
        with torch.cuda.stream(stream):
            count = (mapping != self._baseline_mapping).sum().to(torch.int32).view(1)
        return FutureTensor.device_to_host(src_device=count, stream=stream)
