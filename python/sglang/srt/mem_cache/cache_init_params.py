from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


@dataclasses.dataclass
class CacheInitParams:
    disable: bool
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    page_size: int

    is_eagle: bool = False
    tp_cache_group: int = 0
    eviction_policy: str = "lru"
    disable_finished_insert: bool = False

    enable_metrics: bool = False
    enable_kv_cache_events: bool = False
