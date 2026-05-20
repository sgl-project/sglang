from __future__ import annotations

from typing import (
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Type,
)

import torch

from sglang.jit_kernel.kv_canary_verify import (
    CANARY_SLOT_BYTES,
    RealKvHashMode,
    RealKvSource,
)
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.config import CanaryConfig
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

_CANARY_ATTACHED_ATTR = "_kv_canary_attached"
_CANARY_BUFFER_GROUPS_ATTR = "_kv_canary_buffer_groups"
_DEFAULT_REAL_KV_READ_BYTES = 32

_BufInfoTriple = Tuple[List[int], List[int], List[int]]


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
    int32, so the pool's and the allocator's references are replaced in lockstep with an int32 copy and
    the allocator's LUT-mutating methods are patched to keep both views coherent. Pools with an int32
    LUT or no SWA group are left untouched.
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


def _slot_count(pool: KVCache, kind: PoolKind) -> int:
    if isinstance(pool, SWAKVPool):
        sub_pool = pool.full_kv_pool if kind is PoolKind.FULL else pool.swa_kv_pool
        return int(sub_pool.k_buffer[0].shape[0])
    if isinstance(pool, DeepSeekV4TokenToKVPool):
        if kind is PoolKind.SWA:
            return (
                int(pool.swa_kv_pool.kv_buffer[0].shape[0]) * pool.swa_kv_pool.page_size
            )
        return int(pool.c4_kv_pool.kv_buffer[0].shape[0]) * pool.c4_kv_pool.page_size
    if isinstance(pool, MLATokenToKVPool):
        return int(pool.kv_buffer[0].shape[0])
    if isinstance(pool, MHATokenToKVPool):
        return int(pool.k_buffer[0].shape[0])
    raise NotImplementedError(
        f"kv-canary: cannot derive slot_count for pool class {type(pool).__name__}"
    )


def _swa_index_lut(pool: KVCache) -> Optional[torch.Tensor]:
    if isinstance(pool, SWAKVPool):
        return pool.full_to_swa_index_mapping
    if isinstance(pool, DeepSeekV4TokenToKVPool):
        return pool.full_to_swa_index_mapping
    return None


_CANARY_INT32_SWA_LUT_ATTR = "_kv_canary_int32_swa_lut_installed"


def _ensure_swa_lut_int32(
    *,
    pool: KVCache,
    allocator: Optional[object],
) -> None:
    """Force the SWA full-to-swa LUT to int32 so the canary plan/write kernels read it correctly.

    The DSV4 / SWA pools build the LUT as int64 (see swa_memory_pool.py:358) but canary_plan / canary_write
    cast the LUT pointer to int32 (canary_write.cuh:329 and Triton's element-typed loads). On little-endian
    that mismatch makes the write kernel read low-32 bits of an int64 cell — effectively `lut[i // 2]`'s low
    half — so TAIL writes land on the wrong slot and the next SWEEP launch reports stored=0 for the slot the
    plan kernel sweeps. Patch both the pool's and the SWA allocator's references to a fresh int32 view so
    the int64 storage is fully retired; in-place LUT updates after this point continue to work because
    `__setitem__` on the int32 tensor stores int32 values directly.
    """
    if getattr(pool, _CANARY_INT32_SWA_LUT_ATTR, False):
        return
    if not hasattr(pool, "full_to_swa_index_mapping"):
        return
    lut = pool.full_to_swa_index_mapping
    if lut is None:
        return
    if lut.dtype is torch.int32:
        setattr(pool, _CANARY_INT32_SWA_LUT_ATTR, True)
        return

    int32_lut = lut.to(torch.int32).contiguous()
    pool.full_to_swa_index_mapping = int32_lut
    if allocator is not None and hasattr(allocator, "full_to_swa_index_mapping"):
        allocator_lut = allocator.full_to_swa_index_mapping
        if allocator_lut is not None and allocator_lut.data_ptr() == lut.data_ptr():
            allocator.full_to_swa_index_mapping = int32_lut

    setattr(pool, _CANARY_INT32_SWA_LUT_ATTR, True)


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


def _resolve_read_bytes(config: CanaryConfig) -> int:
    if config.real_kv_hash_mode is RealKvHashMode.OFF:
        return 0
    return _DEFAULT_REAL_KV_READ_BYTES


def _build_buffer_group(
    *,
    pool: KVCache,
    adapter: CanaryPoolAdapter,
    kind: PoolKind,
    device: torch.device,
    read_bytes: int,
) -> CanaryBufferGroup:
    num_slots = _slot_count(pool, kind)
    has_v = adapter.has_v_half(pool)

    k_head = torch.zeros(num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
    k_tail = torch.zeros(num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
    v_head: Optional[torch.Tensor]
    v_tail: Optional[torch.Tensor]
    if has_v:
        v_head = torch.zeros(
            num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device
        )
        v_tail = torch.zeros(
            num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device
        )
    else:
        v_head = None
        v_tail = None

    real_kv_sources_k = adapter.build_real_kv_sources(pool, kind, "K", read_bytes)
    real_kv_sources_v = (
        adapter.build_real_kv_sources(pool, kind, "V", read_bytes) if has_v else ()
    )

    swa_index_lut = _swa_index_lut(pool) if kind is PoolKind.SWA else None

    return CanaryBufferGroup(
        kind=kind,
        k_head=k_head,
        k_tail=k_tail,
        v_head=v_head,
        v_tail=v_tail,
        real_kv_sources_k=real_kv_sources_k,
        real_kv_sources_v=real_kv_sources_v,
        swa_index_lut=swa_index_lut,
    )


def _make_row_source(
    *,
    layer_buffer: torch.Tensor,
    read_bytes: int,
) -> tuple[RealKvSource, ...]:
    contiguous = layer_buffer.contiguous()
    num_slots = int(contiguous.shape[0])
    if num_slots == 0 or read_bytes == 0:
        return ()
    flat = contiguous.view(torch.uint8).reshape(num_slots, -1)
    num_bytes_per_token = int(flat.shape[1])
    clipped = max(0, min(int(read_bytes), num_bytes_per_token))
    return (
        RealKvSource(
            tensor=flat,
            page_size=1,
            num_bytes_per_token=num_bytes_per_token,
            read_bytes=clipped,
        ),
    )


def _make_packed_source(
    *,
    page_buffer: torch.Tensor,
    page_size: int,
    bytes_per_token: int,
    read_bytes: int,
) -> tuple[RealKvSource, ...]:
    if read_bytes == 0 or page_buffer.numel() == 0:
        return ()
    flat = page_buffer.contiguous().view(torch.uint8)
    if flat.ndim == 1:
        flat = flat.reshape(1, -1)
    clipped = max(0, min(int(read_bytes), bytes_per_token))
    return (
        RealKvSource(
            tensor=flat,
            page_size=page_size,
            num_bytes_per_token=bytes_per_token,
            read_bytes=clipped,
        ),
    )


def _patch_buf_info_method(
    pool: object,
    *,
    method_name: str,
    group: CanaryBufferGroup,
    has_v_half: bool,
    page_size: int,
) -> None:
    if not hasattr(pool, method_name):
        raise AttributeError(
            f"kv-canary: pool {type(pool).__name__} missing required method {method_name!r}"
        )
    original = getattr(pool, method_name)

    def patched() -> _BufInfoTriple:
        ptrs, lens, item_lens = original()
        return _splice_canary_buf_info(
            ptrs=ptrs,
            lens=lens,
            item_lens=item_lens,
            group=group,
            has_v_half=has_v_half,
            page_size=page_size,
        )

    setattr(pool, method_name, patched)


def _splice_canary_buf_info(
    *,
    ptrs: List[int],
    lens: List[int],
    item_lens: List[int],
    group: CanaryBufferGroup,
    has_v_half: bool,
    page_size: int,
) -> _BufInfoTriple:
    k_head_entry = _entry_triple(group.k_head, page_size=page_size)
    k_tail_entry = _entry_triple(group.k_tail, page_size=page_size)

    if not has_v_half:
        return (
            [k_head_entry[0]] + list(ptrs) + [k_tail_entry[0]],
            [k_head_entry[1]] + list(lens) + [k_tail_entry[1]],
            [k_head_entry[2]] + list(item_lens) + [k_tail_entry[2]],
        )

    assert group.v_head is not None and group.v_tail is not None
    v_head_entry = _entry_triple(group.v_head, page_size=page_size)
    v_tail_entry = _entry_triple(group.v_tail, page_size=page_size)

    if len(ptrs) % 2 != 0:
        raise RuntimeError(
            f"kv-canary: K/V split adapter expects even-length buf_info list, got {len(ptrs)}"
        )
    mid = len(ptrs) // 2
    return (
        [k_head_entry[0]]
        + list(ptrs[:mid])
        + [k_tail_entry[0], v_head_entry[0]]
        + list(ptrs[mid:])
        + [v_tail_entry[0]],
        [k_head_entry[1]]
        + list(lens[:mid])
        + [k_tail_entry[1], v_head_entry[1]]
        + list(lens[mid:])
        + [v_tail_entry[1]],
        [k_head_entry[2]]
        + list(item_lens[:mid])
        + [k_tail_entry[2], v_head_entry[2]]
        + list(item_lens[mid:])
        + [v_tail_entry[2]],
    )


def _entry_triple(buf: torch.Tensor, *, page_size: int) -> Tuple[int, int, int]:
    return (
        buf.data_ptr(),
        buf.nbytes,
        buf[0].nbytes * page_size,
    )


@register_canary_adapter(MHATokenToKVPool)
class _MHAAdapter:
    def is_swa(self, pool: MHATokenToKVPool) -> bool:
        return False

    def has_v_half(self, pool: MHATokenToKVPool) -> bool:
        return True

    def build_real_kv_sources(
        self,
        pool: MHATokenToKVPool,
        kind: PoolKind,
        half: Literal["K", "V"],
        read_bytes: int,
    ) -> tuple[RealKvSource, ...]:
        buf = pool.k_buffer[0] if half == "K" else pool.v_buffer[0]
        return _make_row_source(layer_buffer=buf, read_bytes=read_bytes)

    def install_full_group(
        self, pool: MHATokenToKVPool, group: CanaryBufferGroup
    ) -> None:
        _patch_buf_info_method(
            pool,
            method_name="get_contiguous_buf_infos",
            group=group,
            has_v_half=True,
            page_size=pool.page_size,
        )

    def install_swa_group(
        self, pool: MHATokenToKVPool, group: CanaryBufferGroup
    ) -> None:
        raise NotImplementedError(
            f"kv-canary: MHA pool {type(pool).__name__} has no SWA sub-pool"
        )


@register_canary_adapter(MHATokenToKVPoolFP4)
class _MHAFp4Adapter(_MHAAdapter):
    pass


@register_canary_adapter(MLATokenToKVPool)
class _MLAAdapter:
    def is_swa(self, pool: MLATokenToKVPool) -> bool:
        return False

    def has_v_half(self, pool: MLATokenToKVPool) -> bool:
        return False

    def build_real_kv_sources(
        self,
        pool: MLATokenToKVPool,
        kind: PoolKind,
        half: Literal["K", "V"],
        read_bytes: int,
    ) -> tuple[RealKvSource, ...]:
        if half == "V":
            return ()
        return _make_row_source(layer_buffer=pool.kv_buffer[0], read_bytes=read_bytes)

    def install_full_group(
        self, pool: MLATokenToKVPool, group: CanaryBufferGroup
    ) -> None:
        _patch_buf_info_method(
            pool,
            method_name="get_contiguous_buf_infos",
            group=group,
            has_v_half=False,
            page_size=pool.page_size,
        )

    def install_swa_group(
        self, pool: MLATokenToKVPool, group: CanaryBufferGroup
    ) -> None:
        raise NotImplementedError(
            f"kv-canary: MLA pool {type(pool).__name__} has no SWA sub-pool"
        )


@register_canary_adapter(MLATokenToKVPoolFP4)
class _MLAFp4Adapter(_MLAAdapter):
    pass


@register_canary_adapter(NSATokenToKVPool)
class _NSAAdapter(_MLAAdapter):
    def build_real_kv_sources(
        self,
        pool: NSATokenToKVPool,
        kind: PoolKind,
        half: Literal["K", "V"],
        read_bytes: int,
    ) -> tuple[RealKvSource, ...]:
        if half == "V":
            return ()
        kv_sources = _make_row_source(
            layer_buffer=pool.kv_buffer[0], read_bytes=read_bytes
        )
        index_buffer = pool.index_k_with_scale_buffer[0]
        index_page_size = pool.page_size
        index_bytes_per_token = int(index_buffer.shape[1]) // index_page_size
        index_sources = _make_packed_source(
            page_buffer=index_buffer,
            page_size=index_page_size,
            bytes_per_token=index_bytes_per_token,
            read_bytes=read_bytes,
        )
        return kv_sources + index_sources


@register_canary_adapter(SWAKVPool)
class _SWAAdapter:
    def is_swa(self, pool: SWAKVPool) -> bool:
        return True

    def has_v_half(self, pool: SWAKVPool) -> bool:
        return True

    def build_real_kv_sources(
        self,
        pool: SWAKVPool,
        kind: PoolKind,
        half: Literal["K", "V"],
        read_bytes: int,
    ) -> tuple[RealKvSource, ...]:
        sub_pool = pool.full_kv_pool if kind is PoolKind.FULL else pool.swa_kv_pool
        buf = sub_pool.k_buffer[0] if half == "K" else sub_pool.v_buffer[0]
        return _make_row_source(layer_buffer=buf, read_bytes=read_bytes)

    def install_full_group(self, pool: SWAKVPool, group: CanaryBufferGroup) -> None:
        _patch_buf_info_method(
            pool,
            method_name="get_contiguous_buf_infos",
            group=group,
            has_v_half=True,
            page_size=pool.page_size,
        )

    def install_swa_group(self, pool: SWAKVPool, group: CanaryBufferGroup) -> None:
        _patch_buf_info_method(
            pool,
            method_name="get_state_buf_infos",
            group=group,
            has_v_half=True,
            page_size=pool.page_size,
        )


@register_canary_adapter(DeepSeekV4TokenToKVPool)
class _DeepSeekV4Adapter:
    def is_swa(self, pool: DeepSeekV4TokenToKVPool) -> bool:
        return True

    def has_v_half(self, pool: DeepSeekV4TokenToKVPool) -> bool:
        return False

    def build_real_kv_sources(
        self,
        pool: DeepSeekV4TokenToKVPool,
        kind: PoolKind,
        half: Literal["K", "V"],
        read_bytes: int,
    ) -> tuple[RealKvSource, ...]:
        if half == "V":
            return ()
        if kind is PoolKind.SWA:
            swa_buf = pool.swa_kv_pool.kv_buffer[0]
            return _make_row_source(layer_buffer=swa_buf, read_bytes=read_bytes)

        c4_buf = pool.c4_kv_pool.kv_buffer[0]
        c4_page_size = pool.c4_kv_pool.page_size
        c4_bytes_per_token = pool.c4_kv_pool.get_bytes_per_token()
        c4_sources = _make_packed_source(
            page_buffer=c4_buf,
            page_size=c4_page_size,
            bytes_per_token=c4_bytes_per_token,
            read_bytes=read_bytes,
        )

        indexer_buf = pool.c4_indexer_kv_pool.index_k_with_scale_buffer[0]
        indexer_page_size = pool.c4_indexer_kv_pool.page_size
        indexer_bytes_per_token = int(indexer_buf.shape[1]) // indexer_page_size
        indexer_sources = _make_packed_source(
            page_buffer=indexer_buf,
            page_size=indexer_page_size,
            bytes_per_token=indexer_bytes_per_token,
            read_bytes=read_bytes,
        )

        c128_buf = pool.c128_kv_pool.kv_buffer[0]
        c128_page_size = pool.c128_kv_pool.page_size
        c128_bytes_per_token = pool.c128_kv_pool.get_bytes_per_token()
        c128_sources = _make_packed_source(
            page_buffer=c128_buf,
            page_size=c128_page_size,
            bytes_per_token=c128_bytes_per_token,
            read_bytes=read_bytes,
        )

        return c4_sources + indexer_sources + c128_sources

    def install_full_group(
        self, pool: DeepSeekV4TokenToKVPool, group: CanaryBufferGroup
    ) -> None:
        _patch_dsv4_contiguous_buf_info(pool, group=group)

    def install_swa_group(
        self, pool: DeepSeekV4TokenToKVPool, group: CanaryBufferGroup
    ) -> None:
        _patch_dsv4_state_buf_info(pool, group=group)


def _patch_dsv4_contiguous_buf_info(
    pool: DeepSeekV4TokenToKVPool,
    *,
    group: CanaryBufferGroup,
) -> None:
    method_name = "get_contiguous_buf_infos"
    if not hasattr(pool, method_name):
        raise AttributeError(
            f"kv-canary: pool {type(pool).__name__} missing required method {method_name!r}"
        )
    original = getattr(pool, method_name)

    c4_layer_num = len(pool.c4_kv_pool.kv_buffer)
    indexer_layer_num = len(pool.c4_indexer_kv_pool.index_k_with_scale_buffer)
    c128_layer_num = len(pool.c128_kv_pool.kv_buffer)
    segment_offsets = [
        0,
        c4_layer_num,
        c4_layer_num + indexer_layer_num,
    ]
    expected_total = c4_layer_num + indexer_layer_num + c128_layer_num

    page_size = pool.page_size

    def patched() -> _BufInfoTriple:
        ptrs, lens, item_lens = original()
        if len(ptrs) != expected_total:
            raise RuntimeError(
                f"DSV4 buf_info layout drifted: got {len(ptrs)}, expected {expected_total}"
            )
        return _splice_packed_buf_info(
            ptrs=ptrs,
            lens=lens,
            item_lens=item_lens,
            segment_offsets=segment_offsets,
            group=group,
            page_size=page_size,
        )

    setattr(pool, method_name, patched)


def _patch_dsv4_state_buf_info(
    pool: DeepSeekV4TokenToKVPool,
    *,
    group: CanaryBufferGroup,
) -> None:
    method_name = "get_state_buf_infos"
    if not hasattr(pool, method_name):
        raise AttributeError(
            f"kv-canary: pool {type(pool).__name__} missing required method {method_name!r}"
        )
    original = getattr(pool, method_name)

    swa_layer_num = len(pool.swa_kv_pool.kv_buffer)
    compress_state_count = sum(1 for p in pool.compress_state_pools if p is not None)
    indexer_compress_state_count = sum(
        1 for p in pool.indexer_compress_state_pools if p is not None
    )
    if compress_state_count == 0 and indexer_compress_state_count == 0:
        raise NotImplementedError(
            "kv-canary: DSV4 SWA segmentation has empty compress_state_pools and "
            "indexer_compress_state_pools — cannot splice head/tail canary per segment"
        )
    segment_offsets = [
        0,
        swa_layer_num,
        swa_layer_num + compress_state_count,
    ]
    expected_total = swa_layer_num + compress_state_count + indexer_compress_state_count

    page_size = pool.page_size

    def patched() -> _BufInfoTriple:
        ptrs, lens, item_lens = original()
        if len(ptrs) != expected_total:
            raise RuntimeError(
                f"DSV4 state buf_info layout drifted: got {len(ptrs)}, expected {expected_total}"
            )
        return _splice_packed_buf_info(
            ptrs=ptrs,
            lens=lens,
            item_lens=item_lens,
            segment_offsets=segment_offsets,
            group=group,
            page_size=page_size,
        )

    setattr(pool, method_name, patched)


def _splice_packed_buf_info(
    *,
    ptrs: List[int],
    lens: List[int],
    item_lens: List[int],
    segment_offsets: List[int],
    group: CanaryBufferGroup,
    page_size: int,
) -> _BufInfoTriple:
    head_entry = _entry_triple(group.k_head, page_size=page_size)
    tail_entry = _entry_triple(group.k_tail, page_size=page_size)

    out_ptrs: List[int] = []
    out_lens: List[int] = []
    out_item_lens: List[int] = []

    for seg_idx in range(len(segment_offsets)):
        start = segment_offsets[seg_idx]
        stop = (
            segment_offsets[seg_idx + 1]
            if seg_idx + 1 < len(segment_offsets)
            else len(ptrs)
        )
        out_ptrs.append(head_entry[0])
        out_lens.append(head_entry[1])
        out_item_lens.append(head_entry[2])
        out_ptrs.extend(ptrs[start:stop])
        out_lens.extend(lens[start:stop])
        out_item_lens.extend(item_lens[start:stop])
        out_ptrs.append(tail_entry[0])
        out_lens.append(tail_entry[1])
        out_item_lens.append(tail_entry[2])

    return out_ptrs, out_lens, out_item_lens
