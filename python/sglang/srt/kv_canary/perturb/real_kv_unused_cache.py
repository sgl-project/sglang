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

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.perturb.config import (
    PerturbConfig,
    require_target_group_kind,
)
from sglang.srt.kv_canary.perturb.utils import (
    WarmupGate,
    flip_first_byte_in_source,
    pick_target_group,
    should_run_perturbation,
)
from sglang.srt.kv_canary.sweep_plan_builder import build_verify_plan_radix_sweep

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
    swa_window_size: int,
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

    group = pick_target_group(
        buffer_groups=buffer_groups,
        target_kind=require_target_group_kind(
            target_group_kind=config.target_group_kind,
            perturb_name="real_kv_unused_cache",
        ),
    )
    if group is None:
        logger.info(
            "kv_canary perturb real_kv_unused_cache: skipped because no target group with "
            "real_kv_sources_k matched target_group_kind=%s",
            config.target_group_kind,
        )
        return
    slot = _pick_sweep_slot_for_group(
        radix_cache=radix_cache,
        group=group,
        swa_window_size=swa_window_size,
    )
    if slot is None:
        logger.info(
            "kv_canary perturb real_kv_unused_cache: skipped because no orphan sweep slot "
            "was found for group=%s",
            group.kind.name,
        )
        return
    source_pick = int(torch.randint(0, len(group.real_kv_sources_k), (1,)).item())
    source = group.real_kv_sources_k[source_pick]
    flip_result = flip_first_byte_in_source(
        group=group,
        source=source,
        slot_idx=slot,
        slot_is_physical=True,
    )
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


def _pick_sweep_slot_for_group(
    *,
    radix_cache: Optional["BasePrefixCache"],
    group: CanaryBufferGroup,
    swa_window_size: int,
) -> Optional[int]:
    if radix_cache is None:
        return None

    window = swa_window_size if group.kind is PoolKind.SWA else 0
    verify_plan = build_verify_plan_radix_sweep(
        radix_cache=radix_cache,
        swa_window_size=window,
        full_to_swa_index_mapping=group.swa_index_lut,
        unlocked_only=True,
    )
    slots = [
        int(raw_slot)
        for raw_slot in verify_plan.verify_slot_indices.detach().to("cpu").tolist()
        if int(raw_slot) >= 0
    ]
    if not slots:
        return None

    pick = int(torch.randint(0, len(slots), (1,)).item())
    return slots[pick]
