"""Perturb point (c): flip the first byte of a radix-cached but currently-unused
(orphan) slot.

Detection should come from sweep (per-forward verify won't even look at this
slot). Designed to surface bugs where cached KV is silently corrupted and
sleeps until much later when a prefix happens to match.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup
from sglang.srt.kv_canary.runner.perturb.config import PerturbConfig
from sglang.srt.kv_canary.runner.perturb.utils import (
    WarmupGate,
    flip_first_byte_in_source,
    pick_orphan_slot,
    pick_target_group,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


def run(
    *,
    forward_batch: Optional["ForwardBatch"],
    config: PerturbConfig,
    buffer_groups: tuple[CanaryBufferGroup, ...],
    radix_cache: Optional["BasePrefixCache"],
    warmup_gate: WarmupGate,
) -> None:
    del forward_batch  # unused — orphan picks from radix tree, not forward batch
    if config.real_kv_unused_cache_prob <= 0.0:
        return
    if warmup_gate.is_in_warmup():
        return
    if torch.rand((), device="cpu").item() >= config.real_kv_unused_cache_prob:
        return

    slot = pick_orphan_slot(radix_cache=radix_cache)
    if slot is None:
        return
    group = pick_target_group(
        buffer_groups=buffer_groups,
        target_kind=config.target_group_kind,
    )
    if group is None or not group.real_kv_sources_k:
        return
    source_pick = int(torch.randint(0, len(group.real_kv_sources_k), (1,)).item())
    source = group.real_kv_sources_k[source_pick]
    flip_result = flip_first_byte_in_source(
        group=group, source=source, slot_idx=slot
    )
    if flip_result is None:
        return
    row, col, original_byte = flip_result
    logger.info(
        "kv_canary perturb real_kv_unused_cache: group=%s source_idx=%d slot=%d row=%d col=%d "
        "original_byte=0x%02X new_byte=0x%02X",
        group.kind.name,
        source_pick,
        slot,
        row,
        col,
        original_byte,
        original_byte ^ 0xFF,
    )
