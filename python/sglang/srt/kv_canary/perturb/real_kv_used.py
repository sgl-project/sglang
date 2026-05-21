"""Perturb point (b): flip the first byte of a slot currently being used by an
active req.

Detection should come from per-forward verify (HEAD/TAIL kernel), NOT from
sweep. Designed to surface CUDA-graph-idle-class bugs where production reads
a slot whose KV byte was silently overwritten.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup
from sglang.srt.kv_canary.perturb.config import PerturbConfig
from sglang.srt.kv_canary.perturb.slot_picker import pick_active_slot
from sglang.srt.kv_canary.perturb.utils import (
    WarmupGate,
    flip_random_source_byte_and_log,
    pick_target_group,
    should_run_perturbation,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


def run(
    *,
    forward_batch: Optional["ForwardBatch"],
    config: PerturbConfig,
    req_to_token_pool: "ReqToTokenPool",
    buffer_groups: tuple[CanaryBufferGroup, ...],
    warmup_gate: WarmupGate,
) -> None:
    if not should_run_perturbation(
        perturb_name="real_kv_used",
        probability=config.real_kv_used_prob,
        warmup_gate=warmup_gate,
        forward_batch=forward_batch,
    ):
        return

    target = pick_active_slot(
        forward_batch=forward_batch,
        req_to_token_pool=req_to_token_pool,
        exclude_out_cache_loc=True,
    )
    if target is None:
        logger.info(
            "kv_canary perturb real_kv_used: skipped because no active slot was found"
        )
        return
    group = pick_target_group(
        buffer_groups=buffer_groups,
        target_kind=config.target_group_kind,
    )
    if group is None:
        logger.info(
            "kv_canary perturb real_kv_used: skipped because no target group matched "
            "target_group_kind=%s slot=%d",
            config.target_group_kind,
            target.value,
        )
        return
    flip_random_source_byte_and_log(
        perturb_name="real_kv_used",
        group=group,
        slot_idx=target.value,
    )
