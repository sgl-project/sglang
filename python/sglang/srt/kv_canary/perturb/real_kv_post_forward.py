"""Pick a random real-KV source byte derived from out_cache_loc and flip it
in-place AFTER the TAIL kernel has captured its canary hash.

The slot id is taken from maybe_inaccurate_forward_batch.out_cache_loc and used
only as a lookup into the target group's real-KV source buffer; the actual flip
happens inside that buffer. The flip is a PyTorch indexed write on the current
CUDA stream; because TAIL is launched on the same stream, stream ordering
guarantees it happens-after TAIL's canary write.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup
from sglang.srt.kv_canary.perturb.config import (
    PerturbConfig,
    require_target_group_kind,
)
from sglang.srt.kv_canary.perturb.slot_picker import pick_out_cache_loc_slot
from sglang.srt.kv_canary.perturb.utils import (
    WarmupGate,
    flip_random_source_byte_and_log,
    pick_target_group,
    should_run_perturbation,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


def run(
    *,
    maybe_inaccurate_forward_batch: Optional[ForwardBatch],
    config: PerturbConfig,
    buffer_groups: tuple[CanaryBufferGroup, ...],
    warmup_gate: WarmupGate,
) -> None:
    if not should_run_perturbation(
        perturb_name="real_kv_post_forward",
        probability=config.real_kv_post_forward_prob,
        warmup_gate=warmup_gate,
        maybe_inaccurate_forward_batch=maybe_inaccurate_forward_batch,
    ):
        return

    slot = pick_out_cache_loc_slot(
        maybe_inaccurate_forward_batch=maybe_inaccurate_forward_batch
    )
    if slot is None:
        logger.info(
            "kv_canary perturb real_kv_post_forward: skipped because maybe_inaccurate_forward_batch.out_cache_loc "
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
