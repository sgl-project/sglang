"""Flip the first byte of a radix-cached but currently-unused (orphan) slot.

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
from sglang.srt.kv_canary.radix_cache_walker import walk_radix_cache_for_canary

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


def run(
    *,
    maybe_inaccurate_forward_batch: Optional[ForwardBatch],
    config: PerturbConfig,
    buffer_groups: tuple[CanaryBufferGroup, ...],
    radix_cache: Optional[BasePrefixCache],
    swa_window_size: int,
    sweep_interval: int,
    outer_step_counter: int,
    warmup_gate: WarmupGate,
) -> None:
    if sweep_interval <= 0 or outer_step_counter % sweep_interval != 0:
        return

    if not should_run_perturbation(
        perturb_name="real_kv_unused_cache",
        probability=config.real_kv_unused_cache_prob,
        warmup_gate=warmup_gate,
        maybe_inaccurate_forward_batch=maybe_inaccurate_forward_batch,
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
    radix_cache: Optional[BasePrefixCache],
    group: CanaryBufferGroup,
    swa_window_size: int,
) -> Optional[int]:
    if radix_cache is None:
        return None

    walk_result = walk_radix_cache_for_canary(
        radix_cache=radix_cache,
        unlocked_only=True,
        swa_resident_only=group.kind is PoolKind.SWA,
    )
    slots = [
        int(raw_slot)
        for raw_slot in walk_result.slot_indices.detach().to("cpu").tolist()
        if int(raw_slot) >= 0
    ]
    if group.kind is PoolKind.SWA:
        slots = _translate_full_slots_to_swa_slots(
            slots=slots,
            full_to_swa_index_mapping=group.swa_index_lut,
        )
    if not slots:
        return None

    pick = int(torch.randint(0, len(slots), (1,)).item())
    return slots[pick]


def _translate_full_slots_to_swa_slots(
    *,
    slots: list[int],
    full_to_swa_index_mapping: Optional[torch.Tensor],
) -> list[int]:
    if full_to_swa_index_mapping is None:
        return []

    lut = full_to_swa_index_mapping.detach().to("cpu").to(torch.int64)
    translated: list[int] = []
    for slot in slots:
        if slot >= int(lut.shape[0]):
            continue
        physical_slot = int(lut[slot].item())
        if physical_slot >= 0:
            translated.append(physical_slot)
    return translated
