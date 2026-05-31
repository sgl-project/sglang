from __future__ import annotations

import torch

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.pool_patcher.buf_info_splice import patch_buf_info_method
from sglang.srt.kv_canary.pool_patcher.buffer_alloc import alloc_canary_buf


def attach_dsv4(
    *,
    pool: object,
    device: torch.device,
    kv_token_id_vs_position_offset: int,
) -> tuple[CanaryBufferGroup, ...]:
    """Attach canary buffers to a DeepSeekV4TokenToKVPool.

    TODO: only the swa_kv_pool sub-pool is wired; c4_kv_pool / c128_kv_pool /
    c4_indexer_kv_pool / compress state pools are left uncovered.
    """
    sub_pool = pool.swa_kv_pool
    num_slots = int(sub_pool.size)

    k_head = alloc_canary_buf(num_slots=num_slots, device=device)
    k_tail = alloc_canary_buf(num_slots=num_slots, device=device)

    group = CanaryBufferGroup(
        kind=PoolKind.SWA,
        k_head=k_head,
        k_tail=k_tail,
        v_head=None,
        v_tail=None,
        swa_index_lut=pool.full_to_swa_index_mapping,
        kv_token_id_vs_position_offset=kv_token_id_vs_position_offset,
    )

    patch_buf_info_method(
        pool,
        method_name="get_state_buf_infos",
        group=group,
        has_v_half=False,
        page_size=sub_pool.page_size,
    )

    return (group,)
