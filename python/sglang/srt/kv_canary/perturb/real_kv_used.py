"""Flip the first byte of a slot currently being used by an active req.

Detection should come from per-forward verify (HEAD/TAIL kernel), NOT from
sweep. Designed to surface CUDA-graph-idle-class bugs where production reads
a slot whose KV byte was silently overwritten.
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
from sglang.srt.kv_canary.perturb.slot_picker import (
    ReqToTokenEntry,
    collect_active_slots,
)
from sglang.srt.kv_canary.perturb.utils import (
    WarmupGate,
    flip_first_byte_in_source,
    pick_target_group,
    should_run_perturbation,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


def run(
    *,
    maybe_inaccurate_forward_batch: Optional[ForwardBatch],
    config: PerturbConfig,
    req_to_token_pool: ReqToTokenPool,
    buffer_groups: tuple[CanaryBufferGroup, ...],
    swa_window_size: int,
    warmup_gate: WarmupGate,
) -> None:
    if not should_run_perturbation(
        perturb_name="real_kv_used",
        probability=config.real_kv_used_prob,
        warmup_gate=warmup_gate,
        maybe_inaccurate_forward_batch=maybe_inaccurate_forward_batch,
    ):
        return

    group = pick_target_group(
        buffer_groups=buffer_groups,
        target_kind=require_target_group_kind(
            target_group_kind=config.target_group_kind,
            perturb_name="real_kv_used",
        ),
    )
    if group is None:
        logger.info(
            "kv_canary perturb real_kv_used: skipped because no target group with real_kv_sources_k "
            "matched target_group_kind=%s",
            config.target_group_kind,
        )
        return

    target = _pick_active_slot_for_group(
        maybe_inaccurate_forward_batch=maybe_inaccurate_forward_batch,
        req_to_token_pool=req_to_token_pool,
        group=group,
        swa_window_size=swa_window_size,
    )
    if target is None:
        logger.info(
            "kv_canary perturb real_kv_used: skipped because no active slot was found "
            "for group=%s",
            group.kind.name,
        )
        return

    source_pick = int(torch.randint(0, len(group.real_kv_sources_k), (1,)).item())
    source = group.real_kv_sources_k[source_pick]
    flip_result = flip_first_byte_in_source(
        group=group, source=source, slot_idx=target.value
    )
    if flip_result is None:
        logger.info(
            "kv_canary perturb real_kv_used: skipped because slot=%d could not be mapped into "
            "group=%s source_idx=%d",
            target.value,
            group.kind.name,
            source_pick,
        )
        return
    row, col, original_byte = flip_result
    logger.info(
        "kv_canary perturb real_kv_used: group=%s source_idx=%d slot=%d row=%d col=%d "
        "original_byte=0x%02X new_byte=0x%02X",
        group.kind.name,
        source_pick,
        target.value,
        row,
        col,
        original_byte,
        original_byte ^ 0xFF,
    )


def _pick_active_slot_for_group(
    *,
    maybe_inaccurate_forward_batch: ForwardBatch,
    req_to_token_pool: ReqToTokenPool,
    group: CanaryBufferGroup,
    swa_window_size: int,
) -> Optional[ReqToTokenEntry]:
    candidates = collect_active_slots(
        maybe_inaccurate_forward_batch=maybe_inaccurate_forward_batch,
        req_to_token_pool=req_to_token_pool,
        exclude_out_cache_loc=True,
    )
    if group.kind is PoolKind.SWA:
        candidates = [
            entry
            for entry in candidates
            if entry.position >= max(0, entry.seq_len - swa_window_size)
        ]
    if not candidates:
        return None

    pick = int(torch.randint(0, len(candidates), (1,)).item())
    return candidates[pick]
