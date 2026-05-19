from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.jit_kernel.kv_cache_canary import CANARY_SLOT_BYTES

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
_CANARY_BUFFER_GROUPS_ATTR = "_kv_cache_canary_buffer_groups"


class PoolKind(str, enum.Enum):
    """Which attention regime a canary group belongs to.

    - ``FULL`` covers ``[0, K_req)``. Attached to plain MHA/MLA pools and
      as one of the two canaries on every SWA system.
    - ``SWA`` covers ``[max(0, K_req - window), K_req)``. Attached as the
      second canary on every ``BaseSWAKVPool``.
    """

    FULL = "full"
    SWA = "swa"


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryBufferGroup:
    """One canary's worth of canary tensors on a pool.

    K/V split lives here; head/tail symmetry lives in :class:`CanaryEndpoint`.
    ``v_head`` / ``v_tail`` are ``None`` for single-buffer pools (MLA / DSV4).
    ``real_kv_source`` is the underlying real KV layer-0 K buffer used by
    the canary-with-real-data fingerprint; ``None`` disables that feature
    for the group.
    """

    kind: PoolKind
    k_head: torch.Tensor
    k_tail: torch.Tensor
    v_head: Optional[torch.Tensor]
    v_tail: Optional[torch.Tensor]
    k_slot_stride_bytes: int
    v_slot_stride_bytes: int
    real_kv_source: Optional[torch.Tensor] = None
    real_kv_slot_stride_bytes: int = 0

    @property
    def has_v_half(self) -> bool:
        return self.v_head is not None


def attach_canary_buffers(pool: "KVCache") -> None:
    """Attach canary canary buffers + monkey-patch buf_info methods on the pool.

    Dispatches on the pool type:

    - ``MHATokenToKVPool`` (full attention) — attach 1 ``FULL`` group with
      K+V halves.
    - ``MLATokenToKVPool`` / ``MLATokenToKVPoolFP4`` / ``NSATokenToKVPool`` —
      attach 1 ``FULL`` group with K-half only (single ``kv_buffer``).
    - ``BaseSWAKVPool`` (incl. ``SWAKVPool`` and ``DeepSeekV4TokenToKVPool``)
      — attach **two** groups: 1 ``FULL`` + 1 ``SWA``, each with its own
      independent canary tensors. The ``FULL`` group's canary buffer sits on the
      full sub-pool's slot index space (or the swa sub-pool's slot index
      space when no full sub-pool exists — DSV4 case); the ``SWA`` group's
      canary buffer always sits on the swa sub-pool's slot index space. Patches
      BOTH ``get_contiguous_buf_infos`` and ``get_state_buf_infos`` because
      PD takes SWA via the latter, and each patch splices in all attached
      groups' canary buffers.

    Idempotent: a second call on the same pool is a no-op.
    """
    if getattr(pool, _CANARY_POOL_ATTR, False):
        return

    buffer_groups: Dict[PoolKind, CanaryBufferGroup] = {}

    # Dispatch by structural attribute, not just isinstance, so duck-typed
    # fake pools (used by host-side unit tests) also work without depending
    # on the real pool class hierarchy.
    if hasattr(pool, "swa_kv_pool"):
        _attach_swa(pool, buffer_groups=buffer_groups)
    elif hasattr(pool, "kv_buffer") and not hasattr(pool, "k_buffer"):
        _attach_mla(pool, buffer_groups=buffer_groups)
    elif hasattr(pool, "k_buffer") and hasattr(pool, "v_buffer"):
        _attach_mha(pool, buffer_groups=buffer_groups)
    else:
        raise RuntimeError(
            f"kv-canary: unsupported pool type {type(pool).__name__}; "
            "extend pool_patch.py with a dispatch branch"
        )

    setattr(pool, _CANARY_BUFFER_GROUPS_ATTR, buffer_groups)
    setattr(pool, _CANARY_POOL_ATTR, True)

    # Legacy single-canary attribute aliases (used by host-side unit tests
    # and a few inspection helpers). They alias to the FIRST attached
    # group — i.e., the FULL group on every pool. SWA-aware tests should
    # query ``get_buffer_group(pool, PoolKind.SWA)`` instead.
    first_group = next(iter(buffer_groups.values()))
    pool.canary_k_head = first_group.k_head
    pool.canary_k_tail = first_group.k_tail
    pool.canary_v_head = first_group.v_head
    pool.canary_v_tail = first_group.v_tail
    pool.canary_has_v_half = first_group.has_v_half
    pool.canary_k_slot_stride_bytes = first_group.k_slot_stride_bytes
    pool.canary_v_slot_stride_bytes = first_group.v_slot_stride_bytes
    pool.canary_slot_stride_bytes = first_group.k_slot_stride_bytes


def _allocate_buffer_group(
    *,
    kind: PoolKind,
    k_template: torch.Tensor,
    v_template: Optional[torch.Tensor],
    real_kv_source: Optional[torch.Tensor] = None,
) -> CanaryBufferGroup:
    """Allocate a fresh canary buffer group sized off the provided slot templates.

    Each canary buffer has shape ``[num_slots, CANARY_SLOT_BYTES]`` uint8 — only
    the slot count is borrowed from the template; the per-slot footprint
    is the canary's tiny fingerprint rather than the real KV's per-slot
    size, so canary memory and PD transfer stay bounded.

    ``real_kv_source`` (optional) is the underlying real KV pool's layer
    template (same tensor used for ``k_template`` for K-half pools); we
    derive the per-slot real-stride from its layout so the kernel can
    address slot N at ``real_kv_source.flatten().view(uint8)[N*stride : ...]``.
    """
    device = k_template.device
    num_slots = int(k_template.shape[0])
    k_head = torch.zeros(num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
    k_tail = torch.zeros(num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)

    v_head: Optional[torch.Tensor]
    v_tail: Optional[torch.Tensor]
    v_slot_stride_bytes: int
    if v_template is None:
        v_head = None
        v_tail = None
        v_slot_stride_bytes = 0
    else:
        v_num_slots = int(v_template.shape[0])
        v_head = torch.zeros(
            v_num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device
        )
        v_tail = torch.zeros(
            v_num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device
        )
        v_slot_stride_bytes = CANARY_SLOT_BYTES

    real_kv_slot_stride_bytes: int = 0
    if real_kv_source is not None and num_slots > 0:
        real_kv_slot_stride_bytes = int(real_kv_source[0].nbytes)

    return CanaryBufferGroup(
        kind=kind,
        k_head=k_head,
        k_tail=k_tail,
        v_head=v_head,
        v_tail=v_tail,
        k_slot_stride_bytes=CANARY_SLOT_BYTES,
        v_slot_stride_bytes=v_slot_stride_bytes,
        real_kv_source=real_kv_source,
        real_kv_slot_stride_bytes=real_kv_slot_stride_bytes,
    )


def _attach_mha(
    pool: "MHATokenToKVPool", *, buffer_groups: Dict[PoolKind, CanaryBufferGroup]
) -> None:
    if pool.layer_num <= 0:
        raise RuntimeError(f"kv-canary: pool has invalid layer_num={pool.layer_num}")
    group = _allocate_buffer_group(
        kind=PoolKind.FULL,
        k_template=pool.k_buffer[0],
        v_template=pool.v_buffer[0],
        real_kv_source=pool.k_buffer[0],
    )
    buffer_groups[PoolKind.FULL] = group
    _patch_get_contiguous_buf_infos(pool, buffer_groups=buffer_groups, has_v_half=True)
    logger.info(
        "kv-canary: attached MHA canary buffer (FULL, dtype=%s, slot_stride_bytes=%d)",
        group.k_head.dtype,
        group.k_slot_stride_bytes,
    )


def _attach_mla(
    pool: "MLATokenToKVPool", *, buffer_groups: Dict[PoolKind, CanaryBufferGroup]
) -> None:
    """Attach to MLA / NSA / FP4 — single latent ``kv_buffer`` (no V half)."""
    if pool.layer_num <= 0:
        raise RuntimeError(
            f"kv-canary: MLA pool has invalid layer_num={pool.layer_num}"
        )
    group = _allocate_buffer_group(
        kind=PoolKind.FULL,
        k_template=pool.kv_buffer[0],
        v_template=None,
        real_kv_source=pool.kv_buffer[0],
    )
    buffer_groups[PoolKind.FULL] = group
    _patch_get_contiguous_buf_infos(pool, buffer_groups=buffer_groups, has_v_half=False)
    logger.info(
        "kv-canary: attached MLA-style canary buffer on %s (FULL, dtype=%s, slot_stride_bytes=%d)",
        type(pool).__name__,
        group.k_head.dtype,
        group.k_slot_stride_bytes,
    )


def _attach_swa(
    pool: "BaseSWAKVPool", *, buffer_groups: Dict[PoolKind, CanaryBufferGroup]
) -> None:
    """Attach BOTH a FULL and a SWA canary to an SWA system.

    A normal SWA system always runs both — a FULL canary that covers the
    entire prefix and a SWA canary that only covers the window. Two
    independent canary buffer groups live on the pool:

    - ``FULL``: sized off the full sub-pool (``pool.full_kv_pool``) when
      it exists (sglang ``SWAKVPool``), otherwise off the SWA sub-pool
      (DSV4 case — no separate full pool). Slot indices for this group
      come straight from ``req_to_token`` without translation.
    - ``SWA``: sized off ``pool.swa_kv_pool``. Slot indices for this group
      come from ``req_to_token`` translated through
      ``pool.full_to_swa_index_mapping`` if available; pools without that
      mapping (DSV4) use the indices as-is — both groups happen to share
      the same slot index space there.

    Both ``get_contiguous_buf_infos`` and ``get_state_buf_infos`` (PD's
    main route for SWA) get patched to splice in each attached group's
    K-canary entries at the K-block tail and V-canary entries at the
    V-block tail.
    """
    swa_sub_pool = pool.swa_kv_pool
    swa_k_template, swa_v_template = _pull_kv_templates(
        sub_pool=swa_sub_pool, label="SWA sub-pool"
    )

    full_sub_pool = getattr(pool, "full_kv_pool", None)
    if full_sub_pool is not None:
        full_k_template, full_v_template = _pull_kv_templates(
            sub_pool=full_sub_pool, label="SWA full sub-pool"
        )
    else:
        # DSV4 case: no separate full_kv_pool. Fall back to swa templates;
        # the resulting FULL group's canary buffer lives in the swa-sub-pool slot
        # index space (verify range covers the entire prefix).
        full_k_template = swa_k_template
        full_v_template = swa_v_template

    full_group = _allocate_buffer_group(
        kind=PoolKind.FULL,
        k_template=full_k_template,
        v_template=full_v_template,
        real_kv_source=full_k_template,
    )
    swa_group = _allocate_buffer_group(
        kind=PoolKind.SWA,
        k_template=swa_k_template,
        v_template=swa_v_template,
        real_kv_source=swa_k_template,
    )
    buffer_groups[PoolKind.FULL] = full_group
    buffer_groups[PoolKind.SWA] = swa_group

    _patch_get_state_buf_infos(
        pool,
        buffer_groups=buffer_groups,
        has_v_half=swa_group.has_v_half,
    )
    logger.info(
        "kv-canary: attached dual SWA canary buffers on %s (kinds=%s, v_half=%s, "
        "full_slots=%d, swa_slots=%d)",
        type(pool).__name__,
        [k.value for k in buffer_groups.keys()],
        swa_group.has_v_half,
        int(full_group.k_head.shape[0]),
        int(swa_group.k_head.shape[0]),
    )


def _pull_kv_templates(
    *, sub_pool: object, label: str
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return (k_template, v_template) for an MHA-style or MLA-style sub-pool.

    MHA-style sub-pools expose ``k_buffer`` + ``v_buffer``; MLA-style
    sub-pools expose a single ``kv_buffer`` (V-half is ``None``).
    """
    if hasattr(sub_pool, "k_buffer") and hasattr(sub_pool, "v_buffer"):
        return sub_pool.k_buffer[0], sub_pool.v_buffer[0]
    if hasattr(sub_pool, "kv_buffer"):
        return sub_pool.kv_buffer[0], None
    raise RuntimeError(
        f"kv-canary: {label} {type(sub_pool).__name__} has neither "
        "k_buffer/v_buffer nor kv_buffer; cannot attach canary buffer"
    )


def _compose_buf_infos_with_canaries(
    *,
    data_ptrs: List[int],
    data_lens: List[int],
    item_lens: List[int],
    buffer_groups: Dict[PoolKind, CanaryBufferGroup],
    page_size: int,
    has_v_half: bool,
) -> Tuple[List[int], List[int], List[int]]:
    """Splice every attached group's canary entries into the buf-info triple.

    For each ``CanaryBufferGroup`` in ``buffer_groups`` we contribute
    ``[k_head, k_tail]`` at the K-block tail and (when ``has_v_half`` is
    True for the pool) ``[v_head, v_tail]`` at the V-block tail. So a
    single-attach pool yields 2 entries (or 4 with V); a SWA dual-attach
    pool yields 4 entries (or 8 with V).

    ``len // 2`` still bisects K vs V (each group contributes the same
    number of entries to each half), preserving downstream PD code's
    midpoint convention.
    """
    extra_k_ptrs: List[int] = []
    extra_k_lens: List[int] = []
    extra_k_item_lens: List[int] = []
    extra_v_ptrs: List[int] = []
    extra_v_lens: List[int] = []
    extra_v_item_lens: List[int] = []

    for group in buffer_groups.values():
        extra_k_ptrs.extend([group.k_head.data_ptr(), group.k_tail.data_ptr()])
        extra_k_lens.extend([group.k_head.nbytes, group.k_tail.nbytes])
        extra_k_item_lens.extend(
            [
                group.k_head[0].nbytes * page_size,
                group.k_tail[0].nbytes * page_size,
            ]
        )
        if has_v_half and group.has_v_half:
            assert group.v_head is not None and group.v_tail is not None
            extra_v_ptrs.extend([group.v_head.data_ptr(), group.v_tail.data_ptr()])
            extra_v_lens.extend([group.v_head.nbytes, group.v_tail.nbytes])
            extra_v_item_lens.extend(
                [
                    group.v_head[0].nbytes * page_size,
                    group.v_tail[0].nbytes * page_size,
                ]
            )

    if not has_v_half:
        return (
            list(data_ptrs) + extra_k_ptrs,
            list(data_lens) + extra_k_lens,
            list(item_lens) + extra_k_item_lens,
        )

    mid = len(data_ptrs) // 2
    return (
        list(data_ptrs[:mid]) + extra_k_ptrs + list(data_ptrs[mid:]) + extra_v_ptrs,
        list(data_lens[:mid]) + extra_k_lens + list(data_lens[mid:]) + extra_v_lens,
        list(item_lens[:mid])
        + extra_k_item_lens
        + list(item_lens[mid:])
        + extra_v_item_lens,
    )


def _patch_get_contiguous_buf_infos(
    pool: "KVCache",
    *,
    buffer_groups: Dict[PoolKind, CanaryBufferGroup],
    has_v_half: bool,
) -> None:
    original = pool.get_contiguous_buf_infos

    def patched() -> Tuple[List[int], List[int], List[int]]:
        ptrs, lens, item_lens = original()
        return _compose_buf_infos_with_canaries(
            data_ptrs=ptrs,
            data_lens=lens,
            item_lens=item_lens,
            buffer_groups=buffer_groups,
            page_size=pool.page_size,
            has_v_half=has_v_half,
        )

    pool.get_contiguous_buf_infos = patched


def _patch_get_state_buf_infos(
    pool: "BaseSWAKVPool",
    *,
    buffer_groups: Dict[PoolKind, CanaryBufferGroup],
    has_v_half: bool,
) -> None:
    """SWA + PD: ``get_state_buf_infos`` is the main path."""
    original = pool.get_state_buf_infos

    def patched() -> Tuple[List[int], List[int], List[int]]:
        ptrs, lens, item_lens = original()
        return _compose_buf_infos_with_canaries(
            data_ptrs=ptrs,
            data_lens=lens,
            item_lens=item_lens,
            buffer_groups=buffer_groups,
            page_size=pool.page_size,
            has_v_half=has_v_half,
        )

    pool.get_state_buf_infos = patched


def get_canary_buffer_groups(pool: "KVCache") -> Dict[PoolKind, CanaryBufferGroup]:
    """Return the dict of attached canary buffer groups keyed by :class:`PoolKind`.

    Empty dict if the pool has not been attached yet (caller responsibility
    to invoke :func:`attach_canary_buffers` first).
    """
    groups = getattr(pool, _CANARY_BUFFER_GROUPS_ATTR, None)
    if groups is None:
        return {}
    return groups
