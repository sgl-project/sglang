"""Perturb point (d): flip the first byte of a slot in forward_batch.out_cache_loc
AFTER the TAIL kernel has captured its canary hash.

The flip is a PyTorch indexed write on the current CUDA stream; because TAIL is
launched on the same stream, stream ordering guarantees it happens-after TAIL's
canary write.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup
from sglang.srt.kv_canary.perturb.config import (
    PerturbConfig,
    require_target_group_kind,
)
from sglang.srt.kv_canary.perturb.slot_picker import (
    pick_out_cache_loc_slot,
    pick_out_cache_loc_slot_from_tensor,
)
from sglang.srt.kv_canary.perturb.utils import (
    WarmupGate,
    flip_random_source_byte_and_log,
    pick_target_group,
    should_run_perturbation,
)

if TYPE_CHECKING:
    from sglang.srt.kv_canary.runner.single_forward_manager import (
        PostOpsInsideGraphOutputSnapshot,
    )
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

    slot = pick_out_cache_loc_slot(forward_batch=forward_batch)
    if slot is None:
        logger.info(
            "kv_canary perturb real_kv_post_forward: skipped because forward_batch.out_cache_loc "
            "had no valid slot"
        )
        return
    group = pick_target_group(
        buffer_groups=buffer_groups,
        target_kind=require_target_group_kind(
            target_group_kind=config.target_group_kind,
            perturb_name="real_kv_post_forward",
        ),
    )
    if group is None:
        logger.info(
            "kv_canary perturb real_kv_post_forward: skipped because no target group matched "
            "target_group_kind=%s slot=%d",
            config.target_group_kind,
            slot,
        )
        return
    flip_random_source_byte_and_log(
        perturb_name="real_kv_post_forward",
        group=group,
        slot_idx=slot,
    )


def run_from_snapshot(
    *,
    snapshot: "PostOpsInsideGraphOutputSnapshot",
    config: PerturbConfig,
    buffer_groups: tuple[CanaryBufferGroup, ...],
    warmup_gate: WarmupGate,
) -> None:
    """Snapshot-driven variant of :func:`run`. Reads only the per-SFM
    snapshot cloned at phase 3; never touches the live ForwardBatch."""
    if not should_run_perturbation(
        perturb_name="real_kv_post_forward",
        probability=config.real_kv_post_forward_prob,
        warmup_gate=warmup_gate,
        forward_batch=None,
        require_forward_batch=False,
    ):
        return

    slot = pick_out_cache_loc_slot_from_tensor(
        out_cache_loc=snapshot.out_cache_loc
    )
    if slot is None:
        logger.info(
            "kv_canary perturb real_kv_post_forward: skipped because snapshot.out_cache_loc "
            "had no valid slot"
        )
        return
    group = pick_target_group(
        buffer_groups=buffer_groups,
        target_kind=require_target_group_kind(
            target_group_kind=config.target_group_kind,
            perturb_name="real_kv_post_forward",
        ),
    )
    if group is None:
        logger.info(
            "kv_canary perturb real_kv_post_forward: skipped because no target group matched "
            "target_group_kind=%s slot=%d",
            config.target_group_kind,
            slot,
        )
        return
    flip_random_source_byte_and_log(
        perturb_name="real_kv_post_forward",
        group=group,
        slot_idx=slot,
    )
