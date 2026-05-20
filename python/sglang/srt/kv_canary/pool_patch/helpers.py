from __future__ import annotations

import functools
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Tuple,
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
    MLATokenToKVPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

if TYPE_CHECKING:
    from sglang.srt.kv_canary.pool_patch.api import CanaryPoolAdapter


_CANARY_ATTACHED_ATTR = "_kv_canary_attached"
_CANARY_BUFFER_GROUPS_ATTR = "_kv_canary_buffer_groups"
_DEFAULT_REAL_KV_READ_BYTES = 32

_BufInfoTriple = Tuple[List[int], List[int], List[int]]

_WRAPPED_MARKER_ATTR = "_kv_canary_wrapped_by"


def _wrap_method(
    obj: object,
    method_name: str,
    *,
    wrapper: Callable[..., Any],
) -> None:
    """Replace ``obj.method_name`` with a closure that delegates to ``wrapper``.

    ``wrapper(original, *args, **kwargs)`` receives the original bound method as its first arg and the
    call-site args/kwargs as the rest. It decides when (and whether) to call ``original`` and what to
    return. The patched method preserves the original's metadata via :func:`functools.wraps`.

    Raises:
        AttributeError: ``obj`` has no attribute ``method_name``.
        RuntimeError: ``obj.method_name`` has already been wrapped by ``_wrap_method`` (idempotency
            guard — re-wrapping silently would stack two transforms and corrupt return values).
    """
    if not hasattr(obj, method_name):
        raise AttributeError(
            f"kv-canary: {type(obj).__name__} missing required method {method_name!r}"
        )
    original = getattr(obj, method_name)
    if getattr(original, _WRAPPED_MARKER_ATTR, None) is not None:
        raise RuntimeError(
            f"kv-canary: {type(obj).__name__}.{method_name} already wrapped by kv-canary"
        )

    @functools.wraps(original)
    def patched(*args: Any, **kwargs: Any) -> Any:
        return wrapper(original, *args, **kwargs)

    setattr(patched, _WRAPPED_MARKER_ATTR, method_name)
    setattr(obj, method_name, patched)


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
    """Return the int32 LUT view the canary kernels can index directly.

    canary_plan and canary_write read the SWA LUT as int32 (canary_write.cuh:329; Triton's element-typed
    tl.load follows the tensor's declared dtype). SWA pools build the LUT as int64 (swa_memory_pool.py:358),
    so accessing the raw pool tensor would have the kernels read low-32 bits of int64 cells — TAIL writes
    land on the wrong canary slot and the next SWEEP launch reports stored=0 for the slot it sweeps. When
    canary install detected an int64 LUT we built an int32 mirror and patched the SWA allocator to mirror
    each LUT update; that mirror is returned here. Pools whose native LUT is already int32 (or pools without
    an SWA group) fall back to the raw attribute.
    """
    int32_mirror = getattr(pool, _CANARY_INT32_LUT_MIRROR_ATTR, None)
    if int32_mirror is not None:
        return int32_mirror
    if isinstance(pool, SWAKVPool):
        return pool.full_to_swa_index_mapping
    if isinstance(pool, DeepSeekV4TokenToKVPool):
        return pool.full_to_swa_index_mapping
    return None


_CANARY_INT32_LUT_MIRROR_ATTR = "_kv_canary_int32_swa_lut_mirror"
_CANARY_LUT_MIRROR_INSTALLED_ATTR = "_kv_canary_int32_swa_lut_mirror_installed"
_LUT_ALLOC_METHOD_NAMES: Tuple[str, ...] = (
    "alloc",
    "alloc_extend",
    "alloc_extend_swa_tail",
)


def _ensure_swa_lut_int32(
    *,
    pool: KVCache,
    allocator: Optional[object],
) -> None:
    """Install an int32 mirror of the SWA full-to-swa LUT for canary kernels to consume.

    The canary plan/write kernels cast the LUT pointer to int32 (canary_write.cuh:329 and Triton's element-
    typed loads). SWA pools build the LUT as int64 (swa_memory_pool.py:358); on little-endian the int32 cast
    reads the low 32 bits of every other int64 cell, so TAIL writes land on the wrong canary slot and the
    next SWEEP launch reports stored=0 for a slot it sweeps. Allocate an int32 mirror once, stash it on the
    pool, and monkeypatch the allocator's LUT-mutating methods (alloc / alloc_extend / alloc_extend_swa_tail)
    to scatter the int32 view of each new allocation in lockstep with the int64 update. The DSV4 attention
    backend continues to read the int64 LUT directly (we never replace the pool's tensor), so this is a
    canary-only mirror.
    """
    if getattr(pool, _CANARY_LUT_MIRROR_INSTALLED_ATTR, False):
        return
    if not hasattr(pool, "full_to_swa_index_mapping"):
        return
    lut = pool.full_to_swa_index_mapping
    if lut is None:
        return
    if lut.dtype is torch.int32:
        setattr(pool, _CANARY_INT32_LUT_MIRROR_ATTR, lut)
        setattr(pool, _CANARY_LUT_MIRROR_INSTALLED_ATTR, True)
        return

    mirror = lut.to(torch.int32).contiguous()
    setattr(pool, _CANARY_INT32_LUT_MIRROR_ATTR, mirror)

    if allocator is not None:
        _patch_allocator_lut_mirror(
            allocator=allocator,
            int64_lut=lut,
            int32_mirror=mirror,
        )

    setattr(pool, _CANARY_LUT_MIRROR_INSTALLED_ATTR, True)


def _patch_allocator_lut_mirror(
    *,
    allocator: object,
    int64_lut: torch.Tensor,
    int32_mirror: torch.Tensor,
) -> None:
    """Wrap each LUT-mutating method on the allocator with a post-write mirror copy.

    SWATokenToKVPoolAllocator (and the DSV4 wrappers) updates the int64 LUT inside alloc / alloc_extend /
    alloc_extend_swa_tail via tensor __setitem__ on the freshly allocated full-pool indices. Each wrapper
    re-scatters the same range into the int32 mirror after the original method runs so the canary kernels
    always see a coherent value. The wrapped method's return value (the full-pool indices) is preserved.
    """
    for name in _LUT_ALLOC_METHOD_NAMES:
        original = getattr(allocator, name, None)
        if original is None:
            continue
        wrapped = _make_lut_mirror_wrapper(
            original=original,
            int64_lut=int64_lut,
            int32_mirror=int32_mirror,
        )
        setattr(allocator, name, wrapped)


def _make_lut_mirror_wrapper(
    *,
    original: Callable,
    int64_lut: torch.Tensor,
    int32_mirror: torch.Tensor,
) -> Callable:
    def wrapped(*args, **kwargs):
        alloc_full_indices = original(*args, **kwargs)
        if not isinstance(alloc_full_indices, torch.Tensor):
            return alloc_full_indices
        if alloc_full_indices.numel() == 0:
            return alloc_full_indices
        idx_long = (
            alloc_full_indices
            if alloc_full_indices.dtype is torch.int64
            else alloc_full_indices.to(torch.int64)
        )
        int32_mirror[idx_long] = int64_lut[idx_long].to(torch.int32)
        return alloc_full_indices

    return wrapped


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
