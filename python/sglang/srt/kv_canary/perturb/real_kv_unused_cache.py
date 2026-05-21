"""Perturb point (c): flip the first byte of a radix-cached but currently-unused
(orphan) slot.

Detection should come from sweep (per-forward verify won't even look at this
slot). Designed to surface bugs where cached KV is silently corrupted and
sleeps until much later when a prefix happens to match.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup
from sglang.srt.kv_canary.perturb.config import PerturbConfig
from sglang.srt.kv_canary.perturb.slot_picker import pick_orphan_slot
from sglang.srt.kv_canary.perturb.utils import (
    WarmupGate,
    flip_random_source_byte_and_log,
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
    if group is None:
        logger.info(
            "kv_canary perturb real_kv_unused_cache: skipped because no target group matched "
            "target_group_kind=%s slot=%d",
            config.target_group_kind,
            slot,
        )
        return
    flip_random_source_byte_and_log(
        perturb_name="real_kv_unused_cache",
        group=group,
        slot_idx=slot,
    )
