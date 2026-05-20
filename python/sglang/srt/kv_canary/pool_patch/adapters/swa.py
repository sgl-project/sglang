from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.pool_patch.utils import (
    alloc_canary_buf,
    ensure_swa_lut_int32,
    make_row_source,
    patch_buf_info_method,
    swa_index_lut,
)


def attach_swa(
    *,
    pool: object,
    device: torch.device,
    read_bytes: int,
    allocator: Optional[object] = None,
) -> tuple[CanaryBufferGroup, ...]:
    """Attach canary buffers to an SWA pool with both FULL and SWA sub-pools.

    FULL group splices into ``get_contiguous_buf_infos``; SWA group splices into ``get_state_buf_infos``.
    """
    ensure_swa_lut_int32(pool=pool, allocator=allocator)

    full_group = _build_subpool_group(
        sub_pool=pool.full_kv_pool,
        kind=PoolKind.FULL,
        device=device,
        read_bytes=read_bytes,
        swa_lut=None,
    )
    swa_group = _build_subpool_group(
        sub_pool=pool.swa_kv_pool,
        kind=PoolKind.SWA,
        device=device,
        read_bytes=read_bytes,
        swa_lut=swa_index_lut(pool),
    )

    patch_buf_info_method(
        pool,
        method_name="get_contiguous_buf_infos",
        group=full_group,
        has_v_half=True,
        page_size=pool.page_size,
    )
    patch_buf_info_method(
        pool,
        method_name="get_state_buf_infos",
        group=swa_group,
        has_v_half=True,
        page_size=pool.page_size,
    )
    return (full_group, swa_group)


def _build_subpool_group(
    *,
    sub_pool: object,
    kind: PoolKind,
    device: torch.device,
    read_bytes: int,
    swa_lut: Optional[torch.Tensor],
) -> CanaryBufferGroup:
    num_slots = int(sub_pool.k_buffer[0].shape[0])
    k_head = alloc_canary_buf(num_slots=num_slots, device=device)
    k_tail = alloc_canary_buf(num_slots=num_slots, device=device)
    v_head = alloc_canary_buf(num_slots=num_slots, device=device)
    v_tail = alloc_canary_buf(num_slots=num_slots, device=device)
    return CanaryBufferGroup(
        kind=kind,
        k_head=k_head,
        k_tail=k_tail,
        v_head=v_head,
        v_tail=v_tail,
        real_kv_sources_k=make_row_source(
            layer_buffer=sub_pool.k_buffer[0], read_bytes=read_bytes
        ),
        real_kv_sources_v=make_row_source(
            layer_buffer=sub_pool.v_buffer[0], read_bytes=read_bytes
        ),
        swa_index_lut=swa_lut,
    )
