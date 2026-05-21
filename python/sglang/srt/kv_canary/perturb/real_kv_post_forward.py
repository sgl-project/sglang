"""Perturb point (d): flip the first byte of a slot in forward_batch.out_cache_loc
AFTER the TAIL kernel has captured its canary hash.

The flip is a PyTorch indexed write on the current CUDA stream; because TAIL is
launched on the same stream, stream ordering guarantees it happens-after TAIL's
canary write.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup
from sglang.srt.kv_canary.perturb.config import PerturbConfig
from sglang.srt.kv_canary.perturb.utils import (
    WarmupGate,
    flip_first_byte_in_source,
    pick_target_group,
    should_run_perturbation,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


def run(
    *,
    forward_batch: Optional["ForwardBatch"],
    config: PerturbConfig,
    buffer_groups: tuple[CanaryBufferGroup, ...],
    warmup_gate: WarmupGate,
) -> None:
    if not should_run_perturbation(
        perturb_name="real_kv_post_forward",
        probability=config.real_kv_post_forward_prob,
        warmup_gate=warmup_gate,
        forward_batch=forward_batch,
    ):
        return

    slot = _pick_out_cache_slot(forward_batch=forward_batch)
    if slot is None:
        logger.info(
            "kv_canary perturb real_kv_post_forward: skipped because forward_batch.out_cache_loc "
            "had no valid slot"
        )
        return
    group = pick_target_group(
        buffer_groups=buffer_groups,
        target_kind=config.target_group_kind,
    )
    if group is None or not group.real_kv_sources_k:
        logger.info(
            "kv_canary perturb real_kv_post_forward: skipped because no target group with "
            "real_kv_sources_k matched target_group_kind=%s slot=%d",
            config.target_group_kind,
            slot,
        )
        return
    source_pick = int(torch.randint(0, len(group.real_kv_sources_k), (1,)).item())
    source = group.real_kv_sources_k[source_pick]
    # Runs after TAIL launch; relies on same-stream ordering vs the TAIL write.
    flip_result = flip_first_byte_in_source(group=group, source=source, slot_idx=slot)
    if flip_result is None:
        logger.info(
            "kv_canary perturb real_kv_post_forward: skipped because slot=%d could not be mapped "
            "into group=%s source_idx=%d",
            slot,
            group.kind.name,
            source_pick,
        )
        return
    row, col, original_byte = flip_result
    logger.info(
        "kv_canary perturb real_kv_post_forward: group=%s source_idx=%d slot=%d row=%d col=%d "
        "original_byte=0x%02X new_byte=0x%02X",
        group.kind.name,
        source_pick,
        slot,
        row,
        col,
        original_byte,
        original_byte ^ 0xFF,
    )


def _pick_out_cache_slot(*, forward_batch: "ForwardBatch") -> Optional[int]:
    out_cache_loc = forward_batch.out_cache_loc
    if out_cache_loc is None:
        return None
    total = int(out_cache_loc.shape[0])
    if total <= 0:
        return None
    valid_num_tokens = forward_batch.num_token_non_padded_cpu
    if valid_num_tokens is None:
        valid_num_tokens = total
    valid_num_tokens = int(valid_num_tokens)
    if valid_num_tokens <= 0:
        return None
    pick = int(torch.randint(0, valid_num_tokens, (1,)).item())
    slot = int(out_cache_loc[pick].item())
    if slot < 0:
        return None
    return slot
