from __future__ import annotations

import dataclasses
from typing import Any, Optional

import torch


@dataclasses.dataclass(slots=True, kw_only=True)
class ReqCacheInfo:
    # The prefix length that is inserted into the tree cache
    cache_protected_len: int
    # TODO(ispobock): rename to last_device_node
    last_node: Any
    # The node to lock until for swa radix tree lock ref
    swa_uuid_for_lock: Optional[int]
    # Whether the prefill-time SWA tree lock has been released early
    swa_prefix_lock_released: bool


@dataclasses.dataclass(slots=True, kw_only=True)
class ReqKvInfo:
    kv_allocated_len: int
    # The length of KV that have been removed in swa cache.
    # SWA KV cache eviction behavior differs by cache type:
    # - Radix cache: KV in range [cache_protected_len, swa_evicted_seqlen) is freed manually in
    #   `ScheduleBatch.maybe_evict_swa`; KV in range [0, cache_protected_len) is freed during radix cache eviction.
    # - Chunk cache: KV in range [0, swa_evicted_seqlen) is freed manually in `ScheduleBatch.maybe_evict_swa`.
    swa_evicted_seqlen: int


@dataclasses.dataclass(slots=True, kw_only=True)
class ReqMambaInfo:
    mamba_pool_idx: Optional[torch.Tensor]  # shape (1)
    mamba_ping_pong_track_buffer: Optional[torch.Tensor]  # shape (2)
    mamba_next_track_idx: Optional[int]  # 0 or 1
    mamba_last_track_seqlen: Optional[int]  # seq len of the last cached mamba state
    # the branching point seqlen to track mamba state. If set, given by prefix match,
    # it will be the tracked seqlen in the ping pong buffer for the right prefill pass.
    mamba_branching_seqlen: Optional[int]
