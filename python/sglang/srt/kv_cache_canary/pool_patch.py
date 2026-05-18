from __future__ import annotations

import enum
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

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


def _set_canary_shadow(
    target: object,
    *,
    k_template: torch.Tensor,
    v_template: Optional[torch.Tensor],
) -> None:
    """Allocate canary shadow tensors on ``target`` from K (and optional V).

    ``target`` is the pool object the runner pulls shadows off (the pool
    itself for MHA/MLA, or the parent SWA pool whose ``swa_kv_pool``
    sub-pool supplied the templates). ``v_template=None`` means the pool
    has no V half (MLA latent rep, DSV4 single kv_buffer); only the K-half
    shadow gets allocated and ``canary_has_v_half`` is False.

    K and V halves get INDEPENDENT per-slot strides because some pool
    flavours (e.g. DeepSeek MLA, certain SWA layouts) use different
    ``head_dim`` for K vs V — sharing the K stride to index V causes
    cross-slot byte aliasing.
    """
    dtype = k_template.dtype
    device = k_template.device
    k_shape = tuple(k_template.shape)

    target.canary_k_head = torch.zeros(k_shape, dtype=dtype, device=device)
    target.canary_k_tail = torch.zeros(k_shape, dtype=dtype, device=device)
    target.canary_k_slot_stride_bytes = int(k_template[0].nbytes)
    if v_template is None:
        target.canary_v_head = None
        target.canary_v_tail = None
        target.canary_has_v_half = False
        target.canary_v_slot_stride_bytes = 0
    else:
        v_shape = tuple(v_template.shape)
        target.canary_v_head = torch.zeros(v_shape, dtype=dtype, device=device)
        target.canary_v_tail = torch.zeros(v_shape, dtype=dtype, device=device)
        target.canary_has_v_half = True
        target.canary_v_slot_stride_bytes = int(v_template[0].nbytes)

    # Legacy alias: callers can still read ``canary_slot_stride_bytes`` for
    # the K-half stride. Prefer the explicit K/V fields above.
    target.canary_slot_stride_bytes = target.canary_k_slot_stride_bytes


def _attach_mha(pool: "MHATokenToKVPool") -> None:
    if pool.layer_num <= 0:
        raise RuntimeError(f"kv-canary: pool has invalid layer_num={pool.layer_num}")
    _set_canary_shadow(pool, k_template=pool.k_buffer[0], v_template=pool.v_buffer[0])
    _patch_get_contiguous_buf_infos_mha(pool)
    logger.info(
        "kv-canary: attached MHA shadow (dtype=%s, slot_stride_bytes=%d)",
        pool.canary_k_head.dtype,
        pool.canary_slot_stride_bytes,
    )


def _attach_mla(pool: "MLATokenToKVPool") -> None:
    """Attach to MLA / NSA / FP4 — single latent ``kv_buffer`` (no V half).

    MLA stores a single latent representation per slot; ``kv_buffer[i]`` is
    the only layer-shaped buffer. Only allocate a K-half shadow pair;
    ``canary_has_v_half=False`` tells the runner to skip the V-half kernel
    launch.
    """
    if pool.layer_num <= 0:
        raise RuntimeError(
            f"kv-canary: MLA pool has invalid layer_num={pool.layer_num}"
        )
    _set_canary_shadow(pool, k_template=pool.kv_buffer[0], v_template=None)
    _patch_get_contiguous_buf_infos_mla(pool)
    logger.info(
        "kv-canary: attached MLA-style shadow on %s (dtype=%s, slot_stride_bytes=%d)",
        type(pool).__name__,
        pool.canary_k_head.dtype,
        pool.canary_slot_stride_bytes,
    )


def _attach_swa(pool: "BaseSWAKVPool") -> None:
    """Attach shadows to the SWA sub-pool using its own slot index space.

    Critical: SWA's slot index space is independent from the full pool's.
    We size the shadows off ``pool.swa_kv_pool`` (an MHA-style pool for
    SWAKVPool, a single-kv_buffer pool for DSV4) so
    ``shadow[swa_slot_idx]`` corresponds 1:1 with ``swa_kv_pool[swa_slot_idx]``
    and never aliases full-pool slots.

    We additionally hook the parent ``BaseSWAKVPool`` ``get_state_buf_infos``
    (the PD-main path for SWA) so the canary shadow rides PD KV transfer.
    """
    swa_sub_pool = pool.swa_kv_pool
    if hasattr(swa_sub_pool, "k_buffer") and hasattr(swa_sub_pool, "v_buffer"):
        _set_canary_shadow(
            pool,
            k_template=swa_sub_pool.k_buffer[0],
            v_template=swa_sub_pool.v_buffer[0],
        )
    elif hasattr(swa_sub_pool, "kv_buffer"):
        _set_canary_shadow(pool, k_template=swa_sub_pool.kv_buffer[0], v_template=None)
    else:
        raise RuntimeError(
            f"kv-canary: SWA sub-pool {type(swa_sub_pool).__name__} has neither "
            "k_buffer/v_buffer nor kv_buffer; cannot attach shadow"
        )
    _patch_get_state_buf_infos_swa(pool)
    logger.info(
        "kv-canary: attached SWA shadow on %s (v_half=%s, dtype=%s, slot_stride_bytes=%d)",
        type(pool).__name__,
        pool.canary_has_v_half,
        pool.canary_k_head.dtype,
        pool.canary_slot_stride_bytes,
    )


def _compose_buf_infos_with_canaries(
    *,
    data_ptrs: List[int],
    data_lens: List[int],
    item_lens: List[int],
    pool: object,
    page_size: int,
    has_v_half: bool,
) -> Tuple[List[int], List[int], List[int]]:
    """Splice canary entries into a `(ptrs, lens, item_lens)` buf-info triple.

    Single buffer (``has_v_half=False``): append ``[k_head, k_tail]`` at the
    tail.

    K|V split (``has_v_half=True``): insert ``[k_head, k_tail]`` at the K
    block tail and ``[v_head, v_tail]`` at the V block tail so ``len // 2``
    still bisects K vs V — what downstream PD code relies on.
    """
    k_head = pool.canary_k_head
    k_tail = pool.canary_k_tail
    k_extra_ptrs = [k_head.data_ptr(), k_tail.data_ptr()]
    k_extra_lens = [k_head.nbytes, k_tail.nbytes]
    k_extra_item_lens = [k_head[0].nbytes * page_size, k_tail[0].nbytes * page_size]

    if not has_v_half:
        return (
            list(data_ptrs) + k_extra_ptrs,
            list(data_lens) + k_extra_lens,
            list(item_lens) + k_extra_item_lens,
        )

    v_head = pool.canary_v_head
    v_tail = pool.canary_v_tail
    v_extra_ptrs = [v_head.data_ptr(), v_tail.data_ptr()]
    v_extra_lens = [v_head.nbytes, v_tail.nbytes]
    v_extra_item_lens = [v_head[0].nbytes * page_size, v_tail[0].nbytes * page_size]

    mid = len(data_ptrs) // 2
    return (
        list(data_ptrs[:mid]) + k_extra_ptrs + list(data_ptrs[mid:]) + v_extra_ptrs,
        list(data_lens[:mid]) + k_extra_lens + list(data_lens[mid:]) + v_extra_lens,
        list(item_lens[:mid])
        + k_extra_item_lens
        + list(item_lens[mid:])
        + v_extra_item_lens,
    )


def _patch_get_contiguous_buf_infos_mha(pool: "MHATokenToKVPool") -> None:
    original = pool.get_contiguous_buf_infos

    def patched() -> Tuple[List[int], List[int], List[int]]:
        ptrs, lens, item_lens = original()
        return _compose_buf_infos_with_canaries(
            data_ptrs=ptrs,
            data_lens=lens,
            item_lens=item_lens,
            pool=pool,
            page_size=pool.page_size,
            has_v_half=True,
        )

    pool.get_contiguous_buf_infos = patched


def _patch_get_contiguous_buf_infos_mla(pool: "MLATokenToKVPool") -> None:
    """MLA only has a single kv_buffer list; append two canary entries to the tail.

    Downstream PD consumers that split with ``len/2`` are MHA-specific paths;
    MLA paths read the list flat. Appending head+tail at the end preserves
    per-layer iteration order.
    """
    original = pool.get_contiguous_buf_infos

    def patched() -> Tuple[List[int], List[int], List[int]]:
        ptrs, lens, item_lens = original()
        return _compose_buf_infos_with_canaries(
            data_ptrs=ptrs,
            data_lens=lens,
            item_lens=item_lens,
            pool=pool,
            page_size=pool.page_size,
            has_v_half=False,
        )

    pool.get_contiguous_buf_infos = patched


def _patch_get_state_buf_infos_swa(pool: "BaseSWAKVPool") -> None:
    """SWA + PD: ``get_state_buf_infos`` is the main path.

    When the SWA sub-pool has an MHA-style K|V split, we insert K-shadow at
    the K block tail and V-shadow at the V block tail so downstream PD code
    that bisects with ``len/2`` keeps splitting K vs V correctly. Single
    ``kv_buffer`` (DSV4) just appends the two K-shadow entries at the end.
    """
    original = pool.get_state_buf_infos

    def patched() -> Tuple[List[int], List[int], List[int]]:
        ptrs, lens, item_lens = original()
        return _compose_buf_infos_with_canaries(
            data_ptrs=ptrs,
            data_lens=lens,
            item_lens=item_lens,
            pool=pool,
            page_size=pool.page_size,
            has_v_half=pool.canary_has_v_half,
        )

    pool.get_state_buf_infos = patched


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


