from __future__ import annotations

import enum
import logging
from typing import TYPE_CHECKING, Callable, List, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
    from sglang.srt.mem_cache.memory_pool import (
        KVCache,
        MHATokenToKVPool,
        MLATokenToKVPool,
    )

    PoolLike = KVCache

logger = logging.getLogger(__name__)

_CANARY_POOL_ATTR = "_kv_cache_canary_attached"
_CANARY_POOL_KIND_ATTR = "_kv_cache_canary_pool_kind"


class PoolKind(str, enum.Enum):
    """Identifier for which pool flavor a canary runner is attached to.

    Multiple canary runners may coexist on a single ModelRunner (e.g. SWA full
    + SWA sub-pool, or spec decoding draft + target). Each gets a distinct
    ``PoolKind`` so host state lookups never collide.
    """

    FULL = "full"
    SWA = "swa"
    MLA = "mla"
    DRAFT = "draft"
    TARGET = "target"


def attach_shadow_buffers(
    pool: "KVCache",
    *,
    pool_kind: PoolKind = PoolKind.FULL,
) -> None:
    """Attach canary shadow buffers + monkey-patch buf_info methods on the pool.

    Dispatches on the pool type:

    - ``MHATokenToKVPool`` (full attention) — K+V halves, both layer-shaped.
    - ``MLATokenToKVPool`` / ``MLATokenToKVPoolFP4`` / ``NSATokenToKVPool`` —
      single ``kv_buffer`` (latent rep), only K-half shadow.
    - ``BaseSWAKVPool`` (incl. ``SWAKVPool`` and ``DeepSeekV4TokenToKVPool``,
      page_size=128 case) — attach on the SWA sub-pool ``swa_kv_pool`` so the
      shadow lives in the SWA slot index space (independent from full pool).
      Patches BOTH ``get_contiguous_buf_infos`` and ``get_state_buf_infos``
      because PD takes SWA via the latter.

    Idempotent: a second call on the same pool is a no-op.
    """
    if getattr(pool, _CANARY_POOL_ATTR, False):
        return

    # Dispatch by structural attribute, not just isinstance, so duck-typed
    # fake pools (used by host-side unit tests) also work without depending
    # on the real pool class hierarchy.
    if hasattr(pool, "swa_kv_pool"):
        _attach_swa(pool)
    elif hasattr(pool, "kv_buffer") and not hasattr(pool, "k_buffer"):
        _attach_mla(pool)
    elif hasattr(pool, "k_buffer") and hasattr(pool, "v_buffer"):
        _attach_mha(pool)
    else:
        raise RuntimeError(
            f"kv-canary: unsupported pool type {type(pool).__name__}; "
            "extend pool_patch.py with a dispatch branch"
        )

    setattr(pool, _CANARY_POOL_ATTR, True)
    setattr(pool, _CANARY_POOL_KIND_ATTR, pool_kind)


def _attach_mha(pool: "MHATokenToKVPool") -> None:
    layer_num = pool.layer_num
    if layer_num <= 0:
        raise RuntimeError(f"kv-canary: pool has invalid layer_num={layer_num}")

    k_template = pool.k_buffer[0]
    v_template = pool.v_buffer[0]

    k_shape = tuple(k_template.shape)
    v_shape = tuple(v_template.shape)
    dtype = k_template.dtype
    device = k_template.device

    pool.canary_k_head = torch.zeros(k_shape, dtype=dtype, device=device)
    pool.canary_k_tail = torch.zeros(k_shape, dtype=dtype, device=device)
    pool.canary_v_head = torch.zeros(v_shape, dtype=dtype, device=device)
    pool.canary_v_tail = torch.zeros(v_shape, dtype=dtype, device=device)

    pool.canary_slot_stride_bytes = int(k_template[0].nbytes)
    pool.canary_has_v_half = True

    _patch_get_contiguous_buf_infos_mha(pool)

    logger.info(
        "kv-canary: attached shadow tensors to MHATokenToKVPool "
        "(k_shape=%s, v_shape=%s, dtype=%s, slot_stride_bytes=%d)",
        k_shape,
        v_shape,
        dtype,
        pool.canary_slot_stride_bytes,
    )


def _attach_mla(pool: "MLATokenToKVPool") -> None:
    """Attach to MLA / NSA / FP4 — single latent ``kv_buffer`` (no V half).

    MLA stores a single latent representation per slot; ``kv_buffer[i]`` is
    the only layer-shaped buffer. We only allocate a K-half shadow pair
    (``canary_k_head`` / ``canary_k_tail``); ``canary_has_v_half=False`` tells
    the runner to skip the V-half kernel launch.
    """
    layer_num = pool.layer_num
    if layer_num <= 0:
        raise RuntimeError(f"kv-canary: MLA pool has invalid layer_num={layer_num}")

    kv_template = pool.kv_buffer[0]
    kv_shape = tuple(kv_template.shape)
    dtype = kv_template.dtype
    device = kv_template.device

    pool.canary_k_head = torch.zeros(kv_shape, dtype=dtype, device=device)
    pool.canary_k_tail = torch.zeros(kv_shape, dtype=dtype, device=device)
    pool.canary_v_head = None
    pool.canary_v_tail = None

    pool.canary_slot_stride_bytes = int(kv_template[0].nbytes)
    pool.canary_has_v_half = False

    _patch_get_contiguous_buf_infos_mla(pool)

    logger.info(
        "kv-canary: attached shadow tensors to %s "
        "(kv_shape=%s, dtype=%s, slot_stride_bytes=%d)",
        type(pool).__name__,
        kv_shape,
        dtype,
        pool.canary_slot_stride_bytes,
    )


def _attach_swa(pool: "BaseSWAKVPool") -> None:
    """Attach shadows to the SWA sub-pool using its own slot index space.

    Critical: SWA's slot index space is independent from the full pool's. We
    attach the shadows on ``pool.swa_kv_pool`` (an ``MHATokenToKVPool`` for
    SWAKVPool, a ``DeepSeekV4SingleKVPool`` for DSV4). This way
    ``shadow[swa_slot_idx]`` corresponds 1:1 with ``swa_kv_pool[swa_slot_idx]``
    and never aliases full-pool slots.

    We additionally hook the parent ``BaseSWAKVPool`` ``get_state_buf_infos``
    (the PD-main path for SWA) so the canary shadow rides PD KV transfer.
    """
    swa_sub_pool = pool.swa_kv_pool
    if not hasattr(swa_sub_pool, "k_buffer") or not hasattr(swa_sub_pool, "v_buffer"):
        _attach_swa_single_buffer(pool, swa_sub_pool)
    else:
        _attach_swa_mha_style(pool, swa_sub_pool)

    _patch_get_state_buf_infos_swa(pool)


def _attach_swa_mha_style(pool: "BaseSWAKVPool", swa_sub_pool: "KVCache") -> None:
    k_template = swa_sub_pool.k_buffer[0]
    v_template = swa_sub_pool.v_buffer[0]

    k_shape = tuple(k_template.shape)
    v_shape = tuple(v_template.shape)
    dtype = k_template.dtype
    device = k_template.device

    pool.canary_k_head = torch.zeros(k_shape, dtype=dtype, device=device)
    pool.canary_k_tail = torch.zeros(k_shape, dtype=dtype, device=device)
    pool.canary_v_head = torch.zeros(v_shape, dtype=dtype, device=device)
    pool.canary_v_tail = torch.zeros(v_shape, dtype=dtype, device=device)

    pool.canary_slot_stride_bytes = int(k_template[0].nbytes)
    pool.canary_has_v_half = True

    logger.info(
        "kv-canary: attached shadow tensors to SWA pool sub-pool "
        "(k_shape=%s, v_shape=%s, dtype=%s, slot_stride_bytes=%d)",
        k_shape,
        v_shape,
        dtype,
        pool.canary_slot_stride_bytes,
    )


def _attach_swa_single_buffer(pool: "BaseSWAKVPool", swa_sub_pool: "KVCache") -> None:
    """DSV4 case: sub-pool has a single ``kv_buffer`` (no K/V split)."""
    if not hasattr(swa_sub_pool, "kv_buffer"):
        raise RuntimeError(
            f"kv-canary: SWA sub-pool {type(swa_sub_pool).__name__} has neither "
            "k_buffer/v_buffer nor kv_buffer; cannot attach shadow"
        )
    kv_template = swa_sub_pool.kv_buffer[0]
    kv_shape = tuple(kv_template.shape)
    dtype = kv_template.dtype
    device = kv_template.device

    pool.canary_k_head = torch.zeros(kv_shape, dtype=dtype, device=device)
    pool.canary_k_tail = torch.zeros(kv_shape, dtype=dtype, device=device)
    pool.canary_v_head = None
    pool.canary_v_tail = None

    pool.canary_slot_stride_bytes = int(kv_template[0].nbytes)
    pool.canary_has_v_half = False

    logger.info(
        "kv-canary: attached shadow tensors to SWA pool (single kv_buffer) "
        "(kv_shape=%s, dtype=%s, slot_stride_bytes=%d)",
        kv_shape,
        dtype,
        pool.canary_slot_stride_bytes,
    )


def _patch_get_contiguous_buf_infos_mha(pool: "MHATokenToKVPool") -> None:
    original = pool.get_contiguous_buf_infos

    def patched_get_contiguous_buf_infos() -> Tuple[List[int], List[int], List[int]]:
        kv_data_ptrs, kv_data_lens, kv_item_lens = original()
        num = len(kv_data_ptrs) // 2
        k_ptrs = kv_data_ptrs[:num]
        v_ptrs = kv_data_ptrs[num:]
        k_lens = kv_data_lens[:num]
        v_lens = kv_data_lens[num:]
        k_item_lens = kv_item_lens[:num]
        v_item_lens = kv_item_lens[num:]

        k_head = pool.canary_k_head
        k_tail = pool.canary_k_tail
        v_head = pool.canary_v_head
        v_tail = pool.canary_v_tail

        new_ptrs = (
            k_ptrs
            + [k_head.data_ptr(), k_tail.data_ptr()]
            + v_ptrs
            + [v_head.data_ptr(), v_tail.data_ptr()]
        )
        new_lens = (
            k_lens
            + [k_head.nbytes, k_tail.nbytes]
            + v_lens
            + [v_head.nbytes, v_tail.nbytes]
        )
        new_item_lens = (
            k_item_lens
            + [k_head[0].nbytes * pool.page_size, k_tail[0].nbytes * pool.page_size]
            + v_item_lens
            + [v_head[0].nbytes * pool.page_size, v_tail[0].nbytes * pool.page_size]
        )
        return new_ptrs, new_lens, new_item_lens

    pool.get_contiguous_buf_infos = patched_get_contiguous_buf_infos


def _patch_get_contiguous_buf_infos_mla(pool: "MLATokenToKVPool") -> None:
    """MLA only has a single kv_buffer list; append two canary entries to the tail.

    Downstream PD consumers that split with ``len/2`` are MHA-specific paths;
    MLA paths read the list flat. Appending head+tail at the end preserves
    per-layer iteration order.
    """
    original = pool.get_contiguous_buf_infos

    def patched_get_contiguous_buf_infos() -> Tuple[List[int], List[int], List[int]]:
        data_ptrs, data_lens, item_lens = original()
        k_head = pool.canary_k_head
        k_tail = pool.canary_k_tail
        new_ptrs = data_ptrs + [k_head.data_ptr(), k_tail.data_ptr()]
        new_lens = data_lens + [k_head.nbytes, k_tail.nbytes]
        new_item_lens = item_lens + [
            k_head[0].nbytes * pool.page_size,
            k_tail[0].nbytes * pool.page_size,
        ]
        return new_ptrs, new_lens, new_item_lens

    pool.get_contiguous_buf_infos = patched_get_contiguous_buf_infos


def _patch_get_state_buf_infos_swa(pool: "BaseSWAKVPool") -> None:
    """SWA + PD: ``get_state_buf_infos`` is the main path.

    When the SWA sub-pool has an MHA-style K|V split (``canary_has_v_half``
    True), we insert the K-shadow at the K block tail and the V-shadow at the
    V block tail so downstream PD code that bisects with ``len/2`` keeps
    splitting K vs V correctly. When the sub-pool has a single ``kv_buffer``
    (e.g. DSV4), we just append the two K-shadow entries at the end.
    """
    original = pool.get_state_buf_infos

    def patched_get_state_buf_infos() -> Tuple[List[int], List[int], List[int]]:
        data_ptrs, data_lens, item_lens = original()
        k_head = pool.canary_k_head
        k_tail = pool.canary_k_tail
        page_size = getattr(pool, "swa_page_size", None) or getattr(
            pool, "page_size", 1
        )

        if pool.canary_v_head is not None and pool.canary_v_tail is not None:
            # K|V midpoint preserving insert.
            num = len(data_ptrs) // 2
            k_ptrs = list(data_ptrs[:num])
            v_ptrs = list(data_ptrs[num:])
            k_lens = list(data_lens[:num])
            v_lens = list(data_lens[num:])
            k_item_lens = list(item_lens[:num])
            v_item_lens = list(item_lens[num:])
            v_head = pool.canary_v_head
            v_tail = pool.canary_v_tail
            new_ptrs = (
                k_ptrs
                + [k_head.data_ptr(), k_tail.data_ptr()]
                + v_ptrs
                + [v_head.data_ptr(), v_tail.data_ptr()]
            )
            new_lens = (
                k_lens
                + [k_head.nbytes, k_tail.nbytes]
                + v_lens
                + [v_head.nbytes, v_tail.nbytes]
            )
            new_item_lens = (
                k_item_lens
                + [k_head[0].nbytes * page_size, k_tail[0].nbytes * page_size]
                + v_item_lens
                + [v_head[0].nbytes * page_size, v_tail[0].nbytes * page_size]
            )
            return new_ptrs, new_lens, new_item_lens

        new_ptrs = list(data_ptrs) + [k_head.data_ptr(), k_tail.data_ptr()]
        new_lens = list(data_lens) + [k_head.nbytes, k_tail.nbytes]
        new_item_lens = list(item_lens) + [
            k_head[0].nbytes * page_size,
            k_tail[0].nbytes * page_size,
        ]
        return new_ptrs, new_lens, new_item_lens

    pool.get_state_buf_infos = patched_get_state_buf_infos


def get_shadow_buffers(
    pool: "KVCache",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Return ``(k_head, k_tail, v_head, v_tail)``. v halves may be ``None``."""
    return (
        pool.canary_k_head,
        pool.canary_k_tail,
        pool.canary_v_head,
        pool.canary_v_tail,
    )


def get_pool_kind(pool: "KVCache") -> PoolKind:
    kind = getattr(pool, _CANARY_POOL_KIND_ATTR, None)
    if kind is None:
        raise RuntimeError(
            "kv-canary: pool has no canary kind set; call attach_shadow_buffers() first"
        )
    return kind


def install_swa_free_hook(
    *, pool: "BaseSWAKVPool", on_free: Callable[[], None]
) -> None:
    """Wrap SWA allocator/pool ``free`` so window-slide evictions trigger reset.

    SWA evicts slots that fall outside the sliding window. When that happens,
    any host-side state that thinks those slots are 'still valid' (high water
    mark, prev_hash_tail) must be cleared so the canary doesn't try to verify
    against an evicted slot. ``on_free`` is called with no args after every
    eviction batch.
    """
    if getattr(pool, "_kv_canary_swa_free_patched", False):
        return
    if not hasattr(pool, "free_swa"):
        return
    original_free_swa = pool.free_swa

    def patched_free_swa(free_index: torch.Tensor) -> None:
        original_free_swa(free_index)
        try:
            on_free()
        except Exception:
            logger.exception("kv-canary: SWA free_swa eviction hook failed")

    pool.free_swa = patched_free_swa
    setattr(pool, "_kv_canary_swa_free_patched", True)
