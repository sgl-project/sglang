from __future__ import annotations

import functools
from typing import Any, Callable, List, Optional, Tuple

import torch

from sglang.jit_kernel.kv_canary.verify import (
    CANARY_SLOT_BYTES,
    RealKvHashMode,
    RealKvSource,
)
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup
from sglang.srt.kv_canary.config import CanaryConfig

BufInfoTriple = Tuple[List[int], List[int], List[int]]

_DEFAULT_REAL_KV_READ_BYTES = 32
_WRAPPED_MARKER_ATTR = "_kv_canary_wrapped_by"
_CANARY_INT32_LUT_MIRROR_ATTR = "_kv_canary_int32_swa_lut_mirror"
_CANARY_LUT_MIRROR_INSTALLED_ATTR = "_kv_canary_int32_swa_lut_mirror_installed"
_LUT_ALLOC_METHOD_NAMES: Tuple[str, ...] = (
    "alloc",
    "alloc_extend",
    "alloc_extend_swa_tail",
)


def wrap_method(
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
        RuntimeError: ``obj.method_name`` has already been wrapped by ``wrap_method`` (idempotency
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


def resolve_read_bytes(config: CanaryConfig) -> int:
    if config.real_kv_hash_mode is RealKvHashMode.OFF:
        return 0
    return _DEFAULT_REAL_KV_READ_BYTES


def alloc_canary_buf(
    *,
    num_slots: int,
    device: torch.device,
) -> torch.Tensor:
    """Allocate one canary buffer (head or tail) of shape ``[num_slots, CANARY_SLOT_BYTES]``."""
    return torch.zeros(num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)


def make_row_source(
    *,
    layer_buffer: torch.Tensor,
    read_bytes: int,
) -> Tuple[RealKvSource, ...]:
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


def make_packed_source(
    *,
    page_buffer: torch.Tensor,
    page_size: int,
    bytes_per_token: int,
    read_bytes: int,
) -> Tuple[RealKvSource, ...]:
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


def patch_buf_info_method(
    pool: object,
    *,
    method_name: str,
    group: CanaryBufferGroup,
    has_v_half: bool,
    page_size: int,
) -> None:
    """Wrap ``pool.<method_name>()`` so its (ptrs, lens, item_lens) triple is spliced with K/V
    head and tail entries from ``group``.

    Per Rule: PD layout is ``k0 k1 ... kN v0 v1 ... vN`` — head/tail sit at index 0 / N+1 within
    EACH half, not at the absolute ends of the combined list.
    """

    def _with_splice(original: Callable, *args: Any, **kwargs: Any) -> BufInfoTriple:
        ptrs, lens, item_lens = original(*args, **kwargs)
        return _splice_kv_buf_info(
            ptrs=ptrs,
            lens=lens,
            item_lens=item_lens,
            group=group,
            has_v_half=has_v_half,
            page_size=page_size,
        )

    wrap_method(pool, method_name, wrapper=_with_splice)


def splice_segmented_buf_info(
    *,
    ptrs: List[int],
    lens: List[int],
    item_lens: List[int],
    segment_offsets: List[int],
    group: CanaryBufferGroup,
    page_size: int,
) -> BufInfoTriple:
    """For multi-segment packed pools (DSV4): prepend/append head + tail canary entries around EACH
    segment, using only ``group.k_head`` / ``group.k_tail`` (these pools have no V half).
    """
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


def ensure_swa_lut_int32(*, pool: object, allocator: Optional[object]) -> None:
    """Install an int32 mirror of the SWA full-to-swa LUT for canary kernels to consume.

    The canary plan/write kernels cast the LUT pointer to int32 (canary_write.cuh:329 and Triton's
    element-typed loads). SWA pools build the LUT as int64 (swa_memory_pool.py:358); on little-endian
    the int32 cast reads the low 32 bits of every other int64 cell, so TAIL writes land on the wrong
    canary slot and the next SWEEP launch reports stored=0 for a slot it sweeps. Allocate an int32
    mirror once, stash it on the pool, and monkeypatch the allocator's LUT-mutating methods (alloc /
    alloc_extend / alloc_extend_swa_tail) to scatter the int32 view of each new allocation in lockstep
    with the int64 update. The DSV4 attention backend continues to read the int64 LUT directly (we
    never replace the pool's tensor), so this is a canary-only mirror.

    Idempotent; no-op when ``pool.full_to_swa_index_mapping`` is missing or None. When the native LUT
    is already int32, the same tensor is recorded as the mirror (no allocator patch needed).
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


def swa_index_lut(pool: object) -> Optional[torch.Tensor]:
    """Return the int32 LUT view installed by :func:`ensure_swa_lut_int32`, or None if no SWA LUT
    has been set up on ``pool``.
    """
    return getattr(pool, _CANARY_INT32_LUT_MIRROR_ATTR, None)


def _splice_kv_buf_info(
    *,
    ptrs: List[int],
    lens: List[int],
    item_lens: List[int],
    group: CanaryBufferGroup,
    has_v_half: bool,
    page_size: int,
) -> BufInfoTriple:
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


def _patch_allocator_lut_mirror(
    *,
    allocator: object,
    int64_lut: torch.Tensor,
    int32_mirror: torch.Tensor,
) -> None:
    """Wrap each LUT-mutating method on the allocator with a post-write mirror copy.

    SWATokenToKVPoolAllocator (and the DSV4 wrappers) updates the int64 LUT inside alloc /
    alloc_extend / alloc_extend_swa_tail via tensor __setitem__ on the freshly allocated full-pool
    indices. Each wrapper re-scatters the same range into the int32 mirror after the original method
    runs so the canary kernels always see a coherent value. The wrapped method's return value (the
    full-pool indices) is preserved.
    """
    for name in _LUT_ALLOC_METHOD_NAMES:
        if not hasattr(allocator, name):
            continue

        def _with_mirror(original: Callable, *args: Any, **kwargs: Any) -> Any:
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

        wrap_method(allocator, name, wrapper=_with_mirror)
