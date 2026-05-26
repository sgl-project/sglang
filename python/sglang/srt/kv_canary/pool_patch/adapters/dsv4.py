"""kv-canary adapter for DSV4 SWA sub-pool.

Scope is intentionally narrow: this adapter only attaches canary buffers to the
``swa_kv_pool`` sub-pool of :class:`DeepSeekV4TokenToKVPool`. The other
sub-pools (``c4_kv_pool``, ``c128_kv_pool``, ``c4_indexer_kv_pool``, compress
state pools) are left untouched in this PR.

Real-KV fingerprint is disabled for DSV4. ``DeepSeekV4SingleKVPool`` stores
584 bytes per token (qk_nope FP8 + qk_rope BF16 + nope FP8 scales + pad),
which is not a multiple of 16. ``RealKvSource`` requires ``num_bytes_per_token``
to be a positive multiple of 16 (the slot stride in the verify kernel equals
``num_bytes_per_token``), so we cannot fingerprint a prefix of each slot
without first introducing an independent ``slot_stride_bytes`` dimension. Until
that lands, this adapter degrades to slot-lifecycle-only canary: head/tail
write-position asserts and verify slot read-back are exercised, but real KV
bytes are not folded into the canary hash. ``read_bytes`` is accepted to keep
the registry signature uniform but is deliberately ignored here.
"""

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
    del read_bytes  # DSV4 584B/token is not 16-aligned; real-KV fingerprint disabled.

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
