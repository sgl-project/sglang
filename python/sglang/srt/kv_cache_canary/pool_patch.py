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
_CANARY_SHADOW_GROUPS_ATTR = "_kv_cache_canary_shadow_groups"


class PoolKind(str, enum.Enum):
    """Identifier for which logical canary attaches to a pool.

    For sglang's logical canary model there are only two attention regimes:

    - ``FULL`` — standard full-prefix attention. Attached to any
      ``MHATokenToKVPool`` / ``MLATokenToKVPool``-style pool, AND as one
      of the two canaries on every SWA system (rule from the spec:
      "正常 SWA 系统其实就是 2 条都走"). The verify range covers
      ``[0, K_req)``.
    - ``SWA`` — sliding-window attention. Attached as the second canary
      on every ``BaseSWAKVPool`` (sglang ``SWAKVPool``, DSV4
      ``DeepSeekV4TokenToKVPool``). The verify range covers
      ``[max(0, K_req - window), K_req)``.

    Spec / draft / target pools are treated as ``FULL`` (no sliding-window
    cap); the runner distinguishes them only at install time by which pool
    the canary attaches to, not by a different ``PoolKind``.
    """

    FULL = "full"
    SWA = "swa"


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryShadowGroup:
    """One canary's worth of shadow tensors on a pool.

    Each entry corresponds to one ``(kernel_kind, half)`` of the
    head/tail × K/V product:

    - ``k_head`` / ``k_tail``: K-half shadow buffers (always present).
    - ``v_head`` / ``v_tail``: V-half shadow buffers (``None`` for
      single-``kv_buffer`` pools like MLA / DSV4-style).

    The same group is wrapped in two :class:`~CanaryEndpoint` instances
    inside :class:`~CanaryRunner` (head + tail), so the K/V split lives
    here as data while the head/tail symmetry lives in the endpoint.

    The optional ``real_kv_source`` references the underlying **real** KV
    pool's layer-0 K buffer (uint8-view, flat across slots). When
    ``--kv-cache-canary-real-data`` is on, the canary kernel reads a
    portion of this tensor at the same slot index it is writing the
    canary fingerprint for, hashes the bytes through splitmix64, and
    stores / verifies the result against the ``real_kv_hash`` slot
    field. ``None`` means the feature is disabled for this group.
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


def attach_shadow_buffers(
    pool: "KVCache",
    *,
    pool_kind: PoolKind = PoolKind.FULL,
) -> None:
    """Attach canary shadow buffers + monkey-patch buf_info methods on the pool.

    Dispatches on the pool type:

    - ``MHATokenToKVPool`` (full attention) — attach 1 ``FULL`` group with
      K+V halves.
    - ``MLATokenToKVPool`` / ``MLATokenToKVPoolFP4`` / ``NSATokenToKVPool`` —
      attach 1 ``FULL`` group with K-half only (single ``kv_buffer``).
    - ``BaseSWAKVPool`` (incl. ``SWAKVPool`` and ``DeepSeekV4TokenToKVPool``)
      — attach **two** groups: 1 ``FULL`` + 1 ``SWA``, each with its own
      independent shadow tensors. The ``FULL`` group's shadow sits on the
      full sub-pool's slot index space (or the swa sub-pool's slot index
      space when no full sub-pool exists — DSV4 case); the ``SWA`` group's
      shadow always sits on the swa sub-pool's slot index space. Patches
      BOTH ``get_contiguous_buf_infos`` and ``get_state_buf_infos`` because
      PD takes SWA via the latter, and each patch splices in all attached
      groups' shadows.

    The ``pool_kind`` argument is accepted only for backward compatibility
    with single-attach callers (mostly host-side unit tests); the dispatch
    decides the actual set of attached groups regardless. Idempotent: a
    second call on the same pool is a no-op.
    """
    if getattr(pool, _CANARY_POOL_ATTR, False):
        return

    _ = pool_kind  # see docstring — dispatch is structural, not parameter-driven.

    shadow_groups: Dict[PoolKind, CanaryShadowGroup] = {}

    # Dispatch by structural attribute, not just isinstance, so duck-typed
    # fake pools (used by host-side unit tests) also work without depending
    # on the real pool class hierarchy.
    if hasattr(pool, "swa_kv_pool"):
        _attach_swa(pool, shadow_groups=shadow_groups)
    elif hasattr(pool, "kv_buffer") and not hasattr(pool, "k_buffer"):
        _attach_mla(pool, shadow_groups=shadow_groups)
    elif hasattr(pool, "k_buffer") and hasattr(pool, "v_buffer"):
        _attach_mha(pool, shadow_groups=shadow_groups)
    else:
        raise RuntimeError(
            f"kv-canary: unsupported pool type {type(pool).__name__}; "
            "extend pool_patch.py with a dispatch branch"
        )

    setattr(pool, _CANARY_SHADOW_GROUPS_ATTR, shadow_groups)
    setattr(pool, _CANARY_POOL_ATTR, True)

    # Legacy single-canary attribute aliases (used by host-side unit tests
    # and a few inspection helpers). They alias to the FIRST attached
    # group — i.e., the FULL group on every pool. SWA-aware tests should
    # query ``get_shadow_group(pool, PoolKind.SWA)`` instead.
    first_group = next(iter(shadow_groups.values()))
    pool.canary_k_head = first_group.k_head
    pool.canary_k_tail = first_group.k_tail
    pool.canary_v_head = first_group.v_head
    pool.canary_v_tail = first_group.v_tail
    pool.canary_has_v_half = first_group.has_v_half
    pool.canary_k_slot_stride_bytes = first_group.k_slot_stride_bytes
    pool.canary_v_slot_stride_bytes = first_group.v_slot_stride_bytes
    pool.canary_slot_stride_bytes = first_group.k_slot_stride_bytes


def _allocate_shadow_group(
    *,
    kind: PoolKind,
    k_template: torch.Tensor,
    v_template: Optional[torch.Tensor],
    real_kv_source: Optional[torch.Tensor] = None,
) -> CanaryShadowGroup:
    """Allocate a fresh shadow group sized off the provided slot templates.

    Each shadow has shape ``[num_slots, CANARY_SLOT_BYTES]`` uint8 — only
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

    return CanaryShadowGroup(
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
    pool: "MHATokenToKVPool", *, shadow_groups: Dict[PoolKind, CanaryShadowGroup]
) -> None:
    if pool.layer_num <= 0:
        raise RuntimeError(f"kv-canary: pool has invalid layer_num={pool.layer_num}")
    group = _allocate_shadow_group(
        kind=PoolKind.FULL,
        k_template=pool.k_buffer[0],
        v_template=pool.v_buffer[0],
        real_kv_source=pool.k_buffer[0],
    )
    shadow_groups[PoolKind.FULL] = group
    _patch_get_contiguous_buf_infos(pool, shadow_groups=shadow_groups, has_v_half=True)
    logger.info(
        "kv-canary: attached MHA shadow (FULL, dtype=%s, slot_stride_bytes=%d)",
        group.k_head.dtype,
        group.k_slot_stride_bytes,
    )


def _attach_mla(
    pool: "MLATokenToKVPool", *, shadow_groups: Dict[PoolKind, CanaryShadowGroup]
) -> None:
    """Attach to MLA / NSA / FP4 — single latent ``kv_buffer`` (no V half)."""
    if pool.layer_num <= 0:
        raise RuntimeError(
            f"kv-canary: MLA pool has invalid layer_num={pool.layer_num}"
        )
    group = _allocate_shadow_group(
        kind=PoolKind.FULL,
        k_template=pool.kv_buffer[0],
        v_template=None,
        real_kv_source=pool.kv_buffer[0],
    )
    shadow_groups[PoolKind.FULL] = group
    _patch_get_contiguous_buf_infos(pool, shadow_groups=shadow_groups, has_v_half=False)
    logger.info(
        "kv-canary: attached MLA-style shadow on %s (FULL, dtype=%s, slot_stride_bytes=%d)",
        type(pool).__name__,
        group.k_head.dtype,
        group.k_slot_stride_bytes,
    )


def _attach_swa(
    pool: "BaseSWAKVPool", *, shadow_groups: Dict[PoolKind, CanaryShadowGroup]
) -> None:
    """Attach BOTH a FULL and a SWA canary to an SWA system.

    Rule from the spec / user instruction: "正常 SWA 系统其实就是 2 条都走".
    Two independent shadow groups live on the pool:

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
    K-shadow entries at the K-block tail and V-shadow entries at the
    V-block tail.
    """
    swa_sub_pool = pool.swa_kv_pool
    if hasattr(swa_sub_pool, "k_buffer") and hasattr(swa_sub_pool, "v_buffer"):
        swa_k_template = swa_sub_pool.k_buffer[0]
        swa_v_template = swa_sub_pool.v_buffer[0]
    elif hasattr(swa_sub_pool, "kv_buffer"):
        swa_k_template = swa_sub_pool.kv_buffer[0]
        swa_v_template = None
    else:
        raise RuntimeError(
            f"kv-canary: SWA sub-pool {type(swa_sub_pool).__name__} has neither "
            "k_buffer/v_buffer nor kv_buffer; cannot attach shadow"
        )

    full_sub_pool = getattr(pool, "full_kv_pool", None)
    if full_sub_pool is not None:
        if hasattr(full_sub_pool, "k_buffer") and hasattr(full_sub_pool, "v_buffer"):
            full_k_template = full_sub_pool.k_buffer[0]
            full_v_template = full_sub_pool.v_buffer[0]
        elif hasattr(full_sub_pool, "kv_buffer"):
            full_k_template = full_sub_pool.kv_buffer[0]
            full_v_template = None
        else:
            raise RuntimeError(
                f"kv-canary: SWA full sub-pool {type(full_sub_pool).__name__} has "
                "neither k_buffer/v_buffer nor kv_buffer; cannot attach shadow"
            )
    else:
        # DSV4 case: no separate full_kv_pool. Fall back to swa templates;
        # the resulting FULL group's shadow lives in the swa-sub-pool slot
        # index space (verify range covers the entire prefix).
        full_k_template = swa_k_template
        full_v_template = swa_v_template

    full_group = _allocate_shadow_group(
        kind=PoolKind.FULL,
        k_template=full_k_template,
        v_template=full_v_template,
        real_kv_source=full_k_template,
    )
    swa_group = _allocate_shadow_group(
        kind=PoolKind.SWA,
        k_template=swa_k_template,
        v_template=swa_v_template,
        real_kv_source=swa_k_template,
    )
    shadow_groups[PoolKind.FULL] = full_group
    shadow_groups[PoolKind.SWA] = swa_group

    _patch_get_state_buf_infos(
        pool,
        shadow_groups=shadow_groups,
        has_v_half=swa_group.has_v_half,
    )
    logger.info(
        "kv-canary: attached dual SWA shadows on %s (kinds=%s, v_half=%s, "
        "full_slots=%d, swa_slots=%d)",
        type(pool).__name__,
        [k.value for k in shadow_groups.keys()],
        swa_group.has_v_half,
        int(full_group.k_head.shape[0]),
        int(swa_group.k_head.shape[0]),
    )


def _compose_buf_infos_with_canaries(
    *,
    data_ptrs: List[int],
    data_lens: List[int],
    item_lens: List[int],
    shadow_groups: Dict[PoolKind, CanaryShadowGroup],
    page_size: int,
    has_v_half: bool,
) -> Tuple[List[int], List[int], List[int]]:
    """Splice every attached group's canary entries into the buf-info triple.

    For each ``CanaryShadowGroup`` in ``shadow_groups`` we contribute
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

    for group in shadow_groups.values():
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
    shadow_groups: Dict[PoolKind, CanaryShadowGroup],
    has_v_half: bool,
) -> None:
    original = pool.get_contiguous_buf_infos

    def patched() -> Tuple[List[int], List[int], List[int]]:
        ptrs, lens, item_lens = original()
        return _compose_buf_infos_with_canaries(
            data_ptrs=ptrs,
            data_lens=lens,
            item_lens=item_lens,
            shadow_groups=shadow_groups,
            page_size=pool.page_size,
            has_v_half=has_v_half,
        )

    pool.get_contiguous_buf_infos = patched


def _patch_get_state_buf_infos(
    pool: "BaseSWAKVPool",
    *,
    shadow_groups: Dict[PoolKind, CanaryShadowGroup],
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
            shadow_groups=shadow_groups,
            page_size=pool.page_size,
            has_v_half=has_v_half,
        )

    pool.get_state_buf_infos = patched


def get_shadow_groups(pool: "KVCache") -> Dict[PoolKind, CanaryShadowGroup]:
    """Return the dict of attached shadow groups keyed by :class:`PoolKind`.

    Empty dict if the pool has not been attached yet (caller responsibility
    to invoke :func:`attach_shadow_buffers` first).
    """
    groups = getattr(pool, _CANARY_SHADOW_GROUPS_ATTR, None)
    if groups is None:
        return {}
    return groups
