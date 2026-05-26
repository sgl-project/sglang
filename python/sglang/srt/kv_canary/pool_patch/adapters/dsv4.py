from __future__ import annotations

import torch

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.pool_patch.buf_info_splice import patch_buf_info_method
from sglang.srt.kv_canary.pool_patch.buffer_alloc import alloc_canary_buf


def attach_dsv4(
    *,
    pool: object,
    device: torch.device,
    read_bytes: int,
    kv_token_id_vs_position_offset: int,
) -> tuple[CanaryBufferGroup, ...]:
    # DSV4 stores 584 B/token (not 16-aligned); RealKvSource requires
    # num_bytes_per_token % 16 == 0, so real-KV fingerprint is disabled.
    del read_bytes

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
        real_kv_sources_k=(),
        real_kv_sources_v=(),
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
