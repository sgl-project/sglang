from __future__ import annotations

from typing import (
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Type,
)

import torch

from sglang.jit_kernel.kv_canary_verify import RealKvSource
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.pool_patch.utils import (
    _CANARY_ATTACHED_ATTR,
    _CANARY_BUFFER_GROUPS_ATTR,
    _build_buffer_group,
    _ensure_swa_lut_int32,
    _resolve_read_bytes,
)
from sglang.srt.mem_cache.memory_pool import KVCache


def attach_canary_buffers(
    *,
    pool: KVCache,
    config: CanaryConfig,
    device: torch.device,
    allocator: Optional[object] = None,
) -> tuple[CanaryBufferGroup, ...]:
    """Install canary buffers on a KV pool and return the resulting CanaryBufferGroup tuple (1 entry per
    pool sub-group: FULL only, or FULL + SWA). Patches the pool's get_contiguous_buf_infos to expose
    canary slots at head and tail so the rest of sglang's plumbing (PD transfer, hicache, etc.) sees
    them as first-class KV bytes.

    Per-pool dispatch is done via a registered adapter (CanaryPoolAdapter) keyed on pool kind. Adapters
    own the layout knowledge: where to insert head/tail canary buffers in the pool's internal storage,
    how to expose RealKvSource for the canary's real-KV mixin, and which monkeypatches are needed.

    Idempotent: calling twice on the same pool is an error (raises). To re-attach, detach first.

    allocator (optional): the SWA-aware token allocator wrapping this pool, when present. Required for
    SWA pools whose ``full_to_swa_index_mapping`` LUT is stored as int64: canary kernels read the LUT as
    int32, so an int32 mirror is stashed on the pool and the allocator's LUT-mutating methods are
    monkeypatched to mirror each allocation. Pools with an int32 LUT or no SWA group are left untouched.
    """
    if getattr(pool, _CANARY_ATTACHED_ATTR, False):
        raise RuntimeError(
            f"kv-canary: pool {type(pool).__name__} already has canary buffers attached"
        )

    adapter = _resolve_adapter(pool)
    read_bytes = _resolve_read_bytes(config)
    groups: List[CanaryBufferGroup] = []

    if adapter.is_swa(pool):
        _ensure_swa_lut_int32(pool=pool, allocator=allocator)

    full_group = _build_buffer_group(
        pool=pool,
        adapter=adapter,
        kind=PoolKind.FULL,
        device=device,
        read_bytes=read_bytes,
    )
    adapter.install_full_group(pool, full_group)
    groups.append(full_group)

    if adapter.is_swa(pool):
        swa_group = _build_buffer_group(
            pool=pool,
            adapter=adapter,
            kind=PoolKind.SWA,
            device=device,
            read_bytes=read_bytes,
        )
        adapter.install_swa_group(pool, swa_group)
        groups.append(swa_group)

    setattr(pool, _CANARY_ATTACHED_ATTR, True)
    groups_tuple = tuple(groups)
    setattr(
        pool,
        _CANARY_BUFFER_GROUPS_ATTR,
        {group.kind: group for group in groups_tuple},
    )
    return groups_tuple


def get_canary_buffer_groups(pool: KVCache) -> Dict[PoolKind, CanaryBufferGroup]:
    """Return the ``{PoolKind: CanaryBufferGroup}`` mapping stashed on ``pool`` by a prior
    :func:`attach_canary_buffers` call. Raises ``RuntimeError`` if canary has not been attached.
    """
    groups = getattr(pool, _CANARY_BUFFER_GROUPS_ATTR, None)
    if groups is None:
        raise RuntimeError(
            f"kv-canary: pool {type(pool).__name__} has no canary buffers attached"
        )
    return groups


class CanaryPoolAdapter(Protocol):
    """Per-pool-kind adapter. One implementation per pool class (MHATokenToKVPool, MLATokenToKVPool,
    SWATokenToKVPool, FP4TokenToKVPool, NSATokenToKVPool, DSV4PackedPool, ...). Registered via
    @register_canary_adapter(PoolClass) decorator at module import.

    Each method is given the pool + a freshly-allocated CanaryBufferGroup (head + tail buffers already
    allocated to the right slot count) and is responsible for wiring it into the pool.
    """

    def is_swa(self, pool: KVCache) -> bool:
        """Return True iff this pool exposes a SWA sub-pool. If True, attach_canary_buffers allocates
        two CanaryBufferGroup instances (FULL + SWA) and calls install_swa_group as well.
        """

    def has_v_half(self, pool: KVCache) -> bool:
        """False for MLA-style pools (single tensor for compressed KV)."""

    def install_full_group(
        self,
        pool: KVCache,
        group: CanaryBufferGroup,
    ) -> None:
        """Wire group.k_head / k_tail (+ v_head / v_tail if has_v_half) into the pool's FULL sub-pool.
        Patches get_contiguous_buf_infos to prepend the head buffer and append the tail buffer to the
        K/V buf list (per Rule: PD layout is k0 k1 ... kN v0 v1 ... vN — the head/tail canary buffers
        sit at index 0 / N+1 within EACH half, not at the absolute ends of the combined list).
        """

    def install_swa_group(
        self,
        pool: KVCache,
        group: CanaryBufferGroup,
    ) -> None:
        """Same as install_full_group but for the SWA sub-pool. Only called when is_swa returns True."""

    def build_real_kv_sources(
        self,
        pool: KVCache,
        kind: PoolKind,  # FULL or SWA
        half: Literal["K", "V"],
        read_bytes: int,
    ) -> tuple[RealKvSource, ...]:
        """Return up to 4 RealKvSource entries (kernels.md §2.4.1) describing the real-KV bytes the
        canary should fingerprint for this (kind, half). read_bytes is the leading-byte budget per
        slot per source. Empty tuple disables the mixin for this endpoint."""


_ADAPTERS: Dict[Type, CanaryPoolAdapter] = {}


def register_canary_adapter(
    pool_class: Type,
) -> Callable[[Type], Type]:
    def _decorator(adapter_class: Type) -> Type:
        _ADAPTERS[pool_class] = adapter_class()
        return adapter_class

    return _decorator


def _resolve_adapter(pool: KVCache) -> CanaryPoolAdapter:
    adapter = _ADAPTERS.get(type(pool))
    if adapter is None:
        raise NotImplementedError(
            f"kv-canary: no CanaryPoolAdapter registered for pool class "
            f"{type(pool).__name__}; supported: "
            f"{sorted(cls.__name__ for cls in _ADAPTERS)}"
        )
    return adapter


from sglang.srt.kv_canary.pool_patch.adapters import (  # noqa: F401  # side-effect: register adapters
    dsv4,
    mha,
    mla,
    swa,
)
