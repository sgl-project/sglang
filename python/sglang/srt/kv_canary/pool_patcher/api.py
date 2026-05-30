from __future__ import annotations

import logging
from typing import Callable, Dict, Type

import torch

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup
from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.pool_patcher.adapters.mha import attach_mha
from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    MHATokenToKVPool,
    MHATokenToKVPoolFP4,
)

logger = logging.getLogger(__name__)

PoolAttacher = Callable[..., tuple[CanaryBufferGroup, ...]]

_POOL_ATTACHERS: Dict[Type, PoolAttacher] = {
    MHATokenToKVPool: attach_mha,
    MHATokenToKVPoolFP4: attach_mha,
}


def register_pool_attacher(pool_class: Type, attacher: PoolAttacher) -> None:
    _POOL_ATTACHERS[pool_class] = attacher


def attach_canary_buffers(
    *,
    pool: KVCache,
    config: CanaryConfig,
    device: torch.device,
    kv_token_id_vs_position_offset: int,
) -> tuple[CanaryBufferGroup, ...]:
    """Install canary buffers on a KV pool and return the resulting CanaryBufferGroup tuple.

    ``kv_token_id_vs_position_offset`` is propagated into every produced :class:`CanaryBufferGroup` (0 for target
    pools; 1 for draft pools where the input-ids rotation shifts the slot-to-token mapping by one).
    """
    attacher = _POOL_ATTACHERS.get(type(pool))
    if attacher is None:
        raise NotImplementedError(
            f"kv-canary: no attacher registered for pool class {type(pool).__name__}; "
            f"supported: {sorted(cls.__name__ for cls in _POOL_ATTACHERS)}"
        )

    groups = attacher(
        pool=pool,
        device=device,
        kv_token_id_vs_position_offset=kv_token_id_vs_position_offset,
    )
    logger.info(
        "attach_canary_buffers: pool=%s attacher=%s n_groups=%d kinds=%s "
        "kv_token_id_vs_position_offset=%d",
        type(pool).__name__,
        attacher.__name__,
        len(groups),
        [g.kind.name for g in groups],
        kv_token_id_vs_position_offset,
    )
    return groups
