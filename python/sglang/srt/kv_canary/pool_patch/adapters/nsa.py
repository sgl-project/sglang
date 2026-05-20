from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.pool_patch.utils import (
    alloc_canary_buf,
    make_packed_source,
    make_row_source,
    patch_buf_info_method,
)


def attach_nsa(
    *,
    pool: object,
    device: torch.device,
    read_bytes: int,
    allocator: Optional[object] = None,
) -> tuple[CanaryBufferGroup, ...]:
    """Attach canary buffers to an NSA pool (MLA-style ``kv_buffer`` + a packed index buffer)."""
    num_slots = int(pool.kv_buffer[0].shape[0])
    k_head = alloc_canary_buf(num_slots=num_slots, device=device)
    k_tail = alloc_canary_buf(num_slots=num_slots, device=device)

    index_buffer = pool.index_k_with_scale_buffer[0]
    index_page_size = pool.page_size
    index_bytes_per_token = int(index_buffer.shape[1]) // index_page_size
    real_kv_sources_k = make_row_source(
        layer_buffer=pool.kv_buffer[0], read_bytes=read_bytes
    ) + make_packed_source(
        page_buffer=index_buffer,
        page_size=index_page_size,
        bytes_per_token=index_bytes_per_token,
        read_bytes=read_bytes,
    )

    group = CanaryBufferGroup(
        kind=PoolKind.FULL,
        k_head=k_head,
        k_tail=k_tail,
        v_head=None,
        v_tail=None,
        real_kv_sources_k=real_kv_sources_k,
        real_kv_sources_v=(),
        swa_index_lut=None,
    )
    patch_buf_info_method(
        pool,
        method_name="get_contiguous_buf_infos",
        group=group,
        has_v_half=False,
        page_size=pool.page_size,
    )
    return (group,)
