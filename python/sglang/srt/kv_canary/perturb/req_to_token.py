"""Flip the req_to_token pointer of a currently-active req.

The hook picks a random (req_pool_idx, position, value) from active reqs,
filtering out entries whose value is 0 (so slot 0 is excluded), and overwrites
req_to_token[req_pool_idx, position] with another active req's slot id.
KV bytes are not touched.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.kv_canary.perturb.config import PerturbConfig
from sglang.srt.kv_canary.perturb.slot_picker import collect_active_slots
from sglang.srt.kv_canary.perturb.utils import WarmupGate, should_run_perturbation

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


def run(
    *,
    maybe_inaccurate_forward_batch: Optional["ForwardBatch"],
    config: PerturbConfig,
    req_to_token_pool: "ReqToTokenPool",
    warmup_gate: WarmupGate,
) -> None:
    if not should_run_perturbation(
        perturb_name="req_to_token",
        probability=config.req_to_token_prob,
        warmup_gate=warmup_gate,
        maybe_inaccurate_forward_batch=maybe_inaccurate_forward_batch,
    ):
        return

    entries = collect_active_slots(
        maybe_inaccurate_forward_batch=maybe_inaccurate_forward_batch,
        req_to_token_pool=req_to_token_pool,
        exclude_out_cache_loc=True,
    )
    entries = [entry for entry in entries if entry.value >= 1]
    if not entries:
        logger.info(
            "kv_canary perturb req_to_token: skipped because no active nonzero slots were found"
        )
        return

    pick = int(torch.randint(0, len(entries), (1,)).item())
    target = entries[pick]
    replacement_values = [item.value for item in entries if item.value != target.value]
    if not replacement_values:
        logger.info(
            "kv_canary perturb req_to_token: skipped because no replacement slot differs from "
            "original_slot=%d",
            target.value,
        )
        return
    replacement_pick = int(torch.randint(0, len(replacement_values), (1,)).item())
    new_value = replacement_values[replacement_pick]

    req_to_token = req_to_token_pool.req_to_token
    logger.info(
        "kv_canary perturb req_to_token: req_pool_idx=%d position=%d original_slot=%d new_slot=%d",
        target.req_pool_idx,
        target.position,
        target.value,
        new_value,
    )
    req_to_token[target.req_pool_idx, target.position] = new_value
