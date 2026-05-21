from __future__ import annotations

import logging
from typing import Callable, Dict, Type

import torch

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup
from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.pool_patch.adapters.mha import attach_mha
from sglang.srt.kv_canary.pool_patch.adapters.swa import attach_swa
from sglang.srt.kv_canary.pool_patch.buffer_alloc import resolve_read_bytes
from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    MHATokenToKVPool,
    MHATokenToKVPoolFP4,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

logger = logging.getLogger(__name__)

PoolAttacher = Callable[..., tuple[CanaryBufferGroup, ...]]

_POOL_ATTACHERS: Dict[Type, PoolAttacher] = {
    MHATokenToKVPool: attach_mha,
    MHATokenToKVPoolFP4: attach_mha,
    SWAKVPool: attach_swa,
}


def register_pool_attacher(pool_class: Type, attacher: PoolAttacher) -> None:
    _POOL_ATTACHERS[pool_class] = attacher


def attach_canary_buffers(
    *,
    pool: KVCache,
    config: CanaryConfig,
    device: torch.device,
) -> tuple[CanaryBufferGroup, ...]:
    """Install canary buffers on a KV pool and return the resulting CanaryBufferGroup tuple
    (1 entry per pool sub-group: FULL only, or FULL + SWA). Patches the pool's
    ``get_contiguous_buf_infos`` (and ``get_state_buf_infos`` for SWA pools) to expose canary slots
    at head and tail so the rest of sglang's plumbing (PD transfer, hicache, etc.) sees them as
    first-class KV bytes.

    Per-pool dispatch is via a small ``type(pool) -> attacher`` table. New pool classes (including
    test fakes) register via :func:`register_pool_attacher`.
    """
    attacher = _POOL_ATTACHERS.get(type(pool))
    if attacher is None:
        raise NotImplementedError(
            f"kv-canary: no attacher registered for pool class {type(pool).__name__}; "
            f"supported: {sorted(cls.__name__ for cls in _POOL_ATTACHERS)}"
        )

    read_bytes = resolve_read_bytes(config)
    groups = attacher(pool=pool, device=device, read_bytes=read_bytes)
    logger.info(
        "attach_canary_buffers: pool=%s attacher=%s read_bytes=%d n_groups=%d kinds=%s",
        type(pool).__name__,
        attacher.__name__,
        read_bytes,
        len(groups),
        [g.kind.name for g in groups],
    )
    return groups
