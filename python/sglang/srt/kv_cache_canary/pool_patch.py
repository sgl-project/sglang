"""Pool attach + buf_info patch for the canary.

Per kernels.md §3.2 / §2.4, this module:

- Allocates one or two :class:`CanaryBufferGroup` instances per pool (FULL alone for MHA / MLA; FULL + SWA
  for SWA / DSV4).
- Constructs each group's ``real_kv_sources_k`` / ``real_kv_sources_v`` tuples using the new
  :class:`RealKvSource` ABI (kernels.md §2.4 + §2.4.1).
- Monkey-patches ``get_contiguous_buf_infos`` (and ``get_state_buf_infos`` on SWA) so PD transfers and
  state-buf consumers see the canary tensors alongside the real KV ones.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.jit_kernel.kv_cache_canary_verify import CANARY_SLOT_BYTES, RealKvSource
from sglang.srt.kv_cache_canary.buffer_group import CanaryBufferGroup, PoolKind

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
    from sglang.srt.mem_cache.memory_pool import (
        KVCache,
        MHATokenToKVPool,
        MLATokenToKVPool,
    )

logger = logging.getLogger(__name__)

_CANARY_POOL_ATTR = "_kv_cache_canary_attached"
_CANARY_BUFFER_GROUPS_ATTR = "_kv_cache_canary_buffer_groups"


def attach_canary_buffers(
    pool: "KVCache",
    *,
    real_kv_read_bytes: int,
) -> None:
    """Attach canary buffers and patch ``buf_info`` methods on ``pool``.

    Dispatches on the pool type (kernels.md §2.4 + §3.2):

    - ``MHATokenToKVPool`` — one ``FULL`` group with K+V halves.
    - ``MLATokenToKVPool`` / ``MLATokenToKVPoolFP4`` / ``NSATokenToKVPool`` — one ``FULL`` group with K only
      (single ``kv_buffer``).
    - ``BaseSWAKVPool`` (incl. ``SWAKVPool`` and ``DeepSeekV4TokenToKVPool``) — one ``FULL`` + one ``SWA``
      group, each with its own canary tensors and ``RealKvSource`` lists.

    ``real_kv_read_bytes`` controls how many leading bytes of each per-token slice each ``RealKvSource``
    contributes to the canary fingerprint. Caller passes ``0`` to disable the real-KV mixin entirely
    (kernel takes the OFF early-out and ignores the sources).
    """
    if getattr(pool, _CANARY_POOL_ATTR, False):
        return

    buffer_groups: Dict[PoolKind, CanaryBufferGroup] = {}

    if hasattr(pool, "swa_kv_pool"):
        _attach_swa(
            pool, buffer_groups=buffer_groups, real_kv_read_bytes=real_kv_read_bytes
        )
    elif hasattr(pool, "kv_buffer") and not hasattr(pool, "k_buffer"):
        _attach_mla(
            pool, buffer_groups=buffer_groups, real_kv_read_bytes=real_kv_read_bytes
        )
    elif hasattr(pool, "k_buffer") and hasattr(pool, "v_buffer"):
        _attach_mha(
            pool, buffer_groups=buffer_groups, real_kv_read_bytes=real_kv_read_bytes
        )
    else:
        raise RuntimeError(
            f"kv-canary: unsupported pool type {type(pool).__name__}; extend pool_patch.py with a "
            "dispatch branch"
        )

    setattr(pool, _CANARY_BUFFER_GROUPS_ATTR, buffer_groups)
    setattr(pool, _CANARY_POOL_ATTR, True)

    first_group = next(iter(buffer_groups.values()))
    pool.canary_k_head = first_group.k_head
    pool.canary_k_tail = first_group.k_tail
    pool.canary_v_head = first_group.v_head
    pool.canary_v_tail = first_group.v_tail
    pool.canary_has_v_half = first_group.has_v_half


def get_canary_buffer_groups(pool: "KVCache") -> Dict[PoolKind, CanaryBufferGroup]:
    """Return the dict of attached canary buffer groups keyed by :class:`PoolKind`.

    Empty dict if the pool has not been attached yet (caller responsibility to invoke
    :func:`attach_canary_buffers` first).
    """
    groups = getattr(pool, _CANARY_BUFFER_GROUPS_ATTR, None)
    if groups is None:
        return {}
    return groups


def _attach_mha(
    pool: "MHATokenToKVPool",
    *,
    buffer_groups: Dict[PoolKind, CanaryBufferGroup],
    real_kv_read_bytes: int,
) -> None:
    if pool.layer_num <= 0:
        raise RuntimeError(f"kv-canary: pool has invalid layer_num={pool.layer_num}")

    k_template = pool.k_buffer[0]
    v_template = pool.v_buffer[0]

    real_kv_sources_k = _build_real_kv_sources_simple(
        layer_buffer=k_template, read_bytes=real_kv_read_bytes
    )
    real_kv_sources_v = _build_real_kv_sources_simple(
        layer_buffer=v_template, read_bytes=real_kv_read_bytes
    )

    group = _allocate_buffer_group(
        kind=PoolKind.FULL,
        k_template=k_template,
        v_template=v_template,
        real_kv_sources_k=real_kv_sources_k,
        real_kv_sources_v=real_kv_sources_v,
    )
    buffer_groups[PoolKind.FULL] = group
    _patch_get_contiguous_buf_infos(pool, buffer_groups=buffer_groups, has_v_half=True)
    logger.info(
        "kv-canary: attached MHA canary buffer (FULL, dtype=%s, num_slots=%d)",
        group.k_head.dtype,
        int(group.k_head.shape[0]),
    )


def _attach_mla(
    pool: "MLATokenToKVPool",
    *,
    buffer_groups: Dict[PoolKind, CanaryBufferGroup],
    real_kv_read_bytes: int,
) -> None:
    """Attach to MLA / NSA / FP4 — single latent ``kv_buffer`` (no V half)."""
    if pool.layer_num <= 0:
        raise RuntimeError(
            f"kv-canary: MLA pool has invalid layer_num={pool.layer_num}"
        )
    k_template = pool.kv_buffer[0]
    real_kv_sources_k = _build_real_kv_sources_simple(
        layer_buffer=k_template, read_bytes=real_kv_read_bytes
    )

    group = _allocate_buffer_group(
        kind=PoolKind.FULL,
        k_template=k_template,
        v_template=None,
        real_kv_sources_k=real_kv_sources_k,
        real_kv_sources_v=(),
    )
    buffer_groups[PoolKind.FULL] = group
    _patch_get_contiguous_buf_infos(pool, buffer_groups=buffer_groups, has_v_half=False)
    logger.info(
        "kv-canary: attached MLA-style canary buffer on %s (FULL, dtype=%s, num_slots=%d)",
        type(pool).__name__,
        group.k_head.dtype,
        int(group.k_head.shape[0]),
    )


def _attach_swa(
    pool: "BaseSWAKVPool",
    *,
    buffer_groups: Dict[PoolKind, CanaryBufferGroup],
    real_kv_read_bytes: int,
) -> None:
    """Attach BOTH a FULL and a SWA canary to an SWA system.

    Mirrors the legacy layout but builds :class:`RealKvSource` tuples instead of carrying a single
    ``real_kv_source`` tensor — DSV4 / multi-layer pools can be extended later by overriding
    ``_build_real_kv_sources_simple`` to emit more than one source.
    """
    swa_sub_pool = pool.swa_kv_pool
    swa_k_template, swa_v_template = _pull_kv_templates(
        sub_pool=swa_sub_pool, label="SWA sub-pool"
    )

    full_sub_pool = getattr(pool, "full_kv_pool", None)
    swa_lut: Optional[torch.Tensor]
    if full_sub_pool is not None:
        full_k_template, full_v_template = _pull_kv_templates(
            sub_pool=full_sub_pool, label="SWA full sub-pool"
        )
        if hasattr(pool, "full_to_swa_index_mapping"):
            swa_lut = pool.full_to_swa_index_mapping
        else:
            swa_lut = None
    else:
        full_k_template = swa_k_template
        full_v_template = swa_v_template
        swa_lut = None

    full_real_kv_sources_k = _build_real_kv_sources_simple(
        layer_buffer=full_k_template, read_bytes=real_kv_read_bytes
    )
    full_real_kv_sources_v = (
        _build_real_kv_sources_simple(
            layer_buffer=full_v_template, read_bytes=real_kv_read_bytes
        )
        if full_v_template is not None
        else ()
    )
    swa_real_kv_sources_k = _build_real_kv_sources_simple(
        layer_buffer=swa_k_template, read_bytes=real_kv_read_bytes
    )
    swa_real_kv_sources_v = (
        _build_real_kv_sources_simple(
            layer_buffer=swa_v_template, read_bytes=real_kv_read_bytes
        )
        if swa_v_template is not None
        else ()
    )

    full_group = _allocate_buffer_group(
        kind=PoolKind.FULL,
        k_template=full_k_template,
        v_template=full_v_template,
        real_kv_sources_k=full_real_kv_sources_k,
        real_kv_sources_v=full_real_kv_sources_v,
    )
    swa_group = _allocate_buffer_group(
        kind=PoolKind.SWA,
        k_template=swa_k_template,
        v_template=swa_v_template,
        real_kv_sources_k=swa_real_kv_sources_k,
        real_kv_sources_v=swa_real_kv_sources_v,
        swa_index_lut=swa_lut,
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


def _allocate_buffer_group(
    *,
    kind: PoolKind,
    k_template: torch.Tensor,
    v_template: Optional[torch.Tensor],
    real_kv_sources_k: tuple[RealKvSource, ...],
    real_kv_sources_v: tuple[RealKvSource, ...],
    swa_index_lut: Optional[torch.Tensor] = None,
) -> CanaryBufferGroup:
    """Allocate a fresh canary buffer group sized off the provided slot templates."""
    device = k_template.device
    num_slots = int(k_template.shape[0])
    k_head = torch.zeros(num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
    k_tail = torch.zeros(num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)

    v_head: Optional[torch.Tensor]
    v_tail: Optional[torch.Tensor]
    if v_template is None:
        v_head = None
        v_tail = None
    else:
        v_num_slots = int(v_template.shape[0])
        v_head = torch.zeros(
            v_num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device
        )
        v_tail = torch.zeros(
            v_num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device
        )

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


def _build_real_kv_sources_simple(
    *, layer_buffer: torch.Tensor, read_bytes: int
) -> tuple[RealKvSource, ...]:
    """Build a single :class:`RealKvSource` covering one KV layer-0 tensor.

    Page-size is taken to be 1 (one slot per row of dim 0) because every supported pool's layer buffer is
    laid out as ``[num_slots, ...]`` row-major contiguous. ``num_bytes_per_token`` is the flattened per-slot
    byte count from the tensor's row stride; ``read_bytes`` is clipped into ``[0, num_bytes_per_token]``.
    """
    contiguous = layer_buffer.contiguous()
    num_slots = int(contiguous.shape[0])
    if num_slots == 0:
        return ()
    flat = contiguous.view(torch.uint8).reshape(num_slots, -1)
    num_bytes_per_token = int(flat.shape[1])
    clipped_read_bytes = max(0, min(int(read_bytes), num_bytes_per_token))
    return (
        RealKvSource(
            tensor=flat,
            page_size=1,
            num_bytes_per_token=num_bytes_per_token,
            read_bytes=clipped_read_bytes,
        ),
    )


def _pull_kv_templates(
    *, sub_pool: object, label: str
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return ``(k_template, v_template)`` for an MHA-style or MLA-style sub-pool.

    MHA-style sub-pools expose ``k_buffer`` + ``v_buffer``; MLA-style sub-pools expose a single
    ``kv_buffer`` (V-half is ``None``).
    """
    if hasattr(sub_pool, "k_buffer") and hasattr(sub_pool, "v_buffer"):
        return sub_pool.k_buffer[0], sub_pool.v_buffer[0]
    if hasattr(sub_pool, "kv_buffer"):
        return sub_pool.kv_buffer[0], None
    raise RuntimeError(
        f"kv-canary: {label} {type(sub_pool).__name__} has neither k_buffer/v_buffer nor kv_buffer; "
        "cannot attach canary buffer"
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
    """Splice every attached group's canary entries into the buf-info triple."""
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
