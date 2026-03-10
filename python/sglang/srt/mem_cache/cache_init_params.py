from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.configs.mamba_utils import BaseLinearStateParams
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


@dataclasses.dataclass
class CacheInitParams:
    disable: bool
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    page_size: int

    is_eagle: bool = False
    tp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    eviction_policy: str = "lru"
    disable_finished_insert: bool = False

    enable_metrics: bool = False
    enable_kv_cache_events: bool = False

    enable_mamba_extra_buffer: bool = False

    pp_rank: int = 0
    pp_size: int = 1

    chunked_prefill_size: Optional[int] = None

    sliding_window_size: Optional[int] = None

    # Time-to-live for cache entries in seconds. If None, TTL is disabled.
    cache_ttl_seconds: Optional[float] = None

    # Marconi FLOP-aware eviction 
    marconi_eff_weight: float = 0.5
    model_config: Optional["ModelConfig"] = None
    mamba_cache_params: Optional["BaseLinearStateParams"] = None
