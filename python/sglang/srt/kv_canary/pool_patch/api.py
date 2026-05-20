from __future__ import annotations

from typing import Callable, Dict, Optional, Type

import torch

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.pool_patch.adapters.dsv4 import attach_dsv4
from sglang.srt.kv_canary.pool_patch.adapters.mha import attach_mha
from sglang.srt.kv_canary.pool_patch.adapters.mla import attach_mla, attach_nsa
from sglang.srt.kv_canary.pool_patch.adapters.swa import attach_swa
from sglang.srt.kv_canary.pool_patch.utils import resolve_read_bytes
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    MHATokenToKVPool,
    MHATokenToKVPoolFP4,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
    NSATokenToKVPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

PoolAttacher = Callable[..., tuple[CanaryBufferGroup, ...]]

_CANARY_ATTACHED_ATTR = "_kv_canary_attached"
_CANARY_BUFFER_GROUPS_ATTR = "_kv_canary_buffer_groups"

_POOL_ATTACHERS: Dict[Type, PoolAttacher] = {
    MHATokenToKVPool: attach_mha,
    MHATokenToKVPoolFP4: attach_mha,
    MLATokenToKVPool: attach_mla,
    MLATokenToKVPoolFP4: attach_mla,
    NSATokenToKVPool: attach_nsa,
    SWAKVPool: attach_swa,
    DeepSeekV4TokenToKVPool: attach_dsv4,
}


def register_pool_attacher(pool_class: Type, attacher: PoolAttacher) -> None:
    _POOL_ATTACHERS[pool_class] = attacher


def attach_canary_buffers(
    *,
    pool: KVCache,
    config: CanaryConfig,
    device: torch.device,
    allocator: Optional[object] = None,
) -> tuple[CanaryBufferGroup, ...]:
    """Install canary buffers on a KV pool and return the resulting CanaryBufferGroup tuple
    (1 entry per pool sub-group: FULL only, or FULL + SWA). Patches the pool's
    ``get_contiguous_buf_infos`` (and ``get_state_buf_infos`` for SWA pools) to expose canary slots
    at head and tail so the rest of sglang's plumbing (PD transfer, hicache, etc.) sees them as
    first-class KV bytes.

    Per-pool dispatch is via a small ``type(pool) -> attacher`` table. New pool classes (including
    test fakes) register via :func:`register_pool_attacher`.

    Idempotent: calling twice on the same pool raises. To re-attach, detach first.

    allocator (optional): the SWA-aware token allocator wrapping this pool, when present. Required
    for SWA pools whose ``full_to_swa_index_mapping`` LUT is stored as int64: canary kernels read the
    LUT as int32, so an int32 mirror is stashed on the pool and the allocator's LUT-mutating methods
    are monkeypatched to mirror each allocation. Pools with an int32 LUT or no SWA group are left
    untouched.
    """
    if getattr(pool, _CANARY_ATTACHED_ATTR, False):
        raise RuntimeError(
            f"kv-canary: pool {type(pool).__name__} already has canary buffers attached"
        )

    attacher = _POOL_ATTACHERS.get(type(pool))
    if attacher is None:
        raise NotImplementedError(
            f"kv-canary: no attacher registered for pool class {type(pool).__name__}; "
            f"supported: {sorted(cls.__name__ for cls in _POOL_ATTACHERS)}"
        )

    read_bytes = resolve_read_bytes(config)
    groups = attacher(
        pool=pool, device=device, read_bytes=read_bytes, allocator=allocator
    )

    setattr(pool, _CANARY_ATTACHED_ATTR, True)
    setattr(pool, _CANARY_BUFFER_GROUPS_ATTR, {group.kind: group for group in groups})
    return groups


def get_canary_buffer_groups(pool: KVCache) -> Dict[PoolKind, CanaryBufferGroup]:
    groups = getattr(pool, _CANARY_BUFFER_GROUPS_ATTR, None)
    if groups is None:
        raise RuntimeError(
            f"kv-canary: pool {type(pool).__name__} has no canary buffers attached"
        )
    return groups
