from __future__ import annotations

import logging
from typing import Callable, Dict, Type

import torch

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup
from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.pool_patch.adapters.mha import attach_mha
from sglang.srt.kv_canary.pool_patch.adapters.swa import attach_swa
from sglang.srt.kv_canary.pool_patch.buffer_alloc import resolve_real_kv_read_bytes
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
    slot_token_offset: int = 0,
) -> tuple[CanaryBufferGroup, ...]:
    """Install canary buffers on a KV pool and return the resulting CanaryBufferGroup tuple.

    ``slot_token_offset`` is propagated into every produced :class:`CanaryBufferGroup` (0 for target
    pools; 1 for EAGLE draft pools where the rotation shifts the slot-to-token mapping by one).
    """
    attacher = _POOL_ATTACHERS.get(type(pool))
    if attacher is None:
        raise NotImplementedError(
            f"kv-canary: no attacher registered for pool class {type(pool).__name__}; "
            f"supported: {sorted(cls.__name__ for cls in _POOL_ATTACHERS)}"
        )

    read_bytes = resolve_real_kv_read_bytes(config)
    groups = attacher(
        pool=pool,
        device=device,
        read_bytes=read_bytes,
        slot_token_offset=slot_token_offset,
    )
    logger.info(
        "attach_canary_buffers: pool=%s attacher=%s read_bytes=%d n_groups=%d kinds=%s "
        "slot_token_offset=%d",
        type(pool).__name__,
        attacher.__name__,
        read_bytes,
        len(groups),
        [g.kind.name for g in groups],
        slot_token_offset,
    )
    return groups
