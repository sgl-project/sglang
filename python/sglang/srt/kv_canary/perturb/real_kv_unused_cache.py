"""Perturb point (c): flip the first byte of a radix-cached but currently-unused
(orphan) slot.

Detection should come from sweep (per-forward verify won't even look at this
slot). Designed to surface bugs where cached KV is silently corrupted and
sleeps until much later when a prefix happens to match.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Optional

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup
from sglang.srt.kv_canary.perturb.config import PerturbConfig
from sglang.srt.kv_canary.perturb.slot_picker import pick_orphan_slot
from sglang.srt.kv_canary.perturb.utils import (
    WarmupGate,
    flip_first_byte_in_source,
    pick_target_group,
    should_run_perturbation,
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
    if not should_run_perturbation(
        perturb_name="real_kv_unused_cache",
        probability=config.real_kv_unused_cache_prob,
        warmup_gate=warmup_gate,
        forward_batch=forward_batch,
        require_forward_batch=False,
    ):
        return

    slot = pick_orphan_slot(radix_cache=radix_cache)
    if slot is None:
        logger.info(
            "kv_canary perturb real_kv_unused_cache: skipped because no orphan radix-cache slot "
            "was found"
        )
        return
    group = pick_target_group(
        buffer_groups=buffer_groups,
        target_kind=config.target_group_kind,
    )
    if group is None or not group.real_kv_sources_k:
        logger.info(
            "kv_canary perturb real_kv_unused_cache: skipped because no target group with "
            "real_kv_sources_k matched target_group_kind=%s slot=%d",
            config.target_group_kind,
            slot,
        )
        return
    source_pick = random.randrange(len(group.real_kv_sources_k))
    source = group.real_kv_sources_k[source_pick]
    flip_result = flip_first_byte_in_source(group=group, source=source, slot_idx=slot)
    if flip_result is None:
        logger.info(
            "kv_canary perturb real_kv_unused_cache: skipped because slot=%d could not be mapped "
            "into group=%s source_idx=%d",
            slot,
            group.kind.name,
            source_pick,
        )
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
