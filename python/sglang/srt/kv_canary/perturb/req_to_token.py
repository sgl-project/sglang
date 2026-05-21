"""Perturb point (a): flip the req_to_token pointer of a currently-active req.

The hook picks a random (req_pool_idx, position, slot) from active reqs and
overwrites table[req_pool_idx, position] with another active req's slot id.
KV bytes are not touched.

Modeled after the bug class: req_to_token bookkeeping (out_cache_loc updates,
sequence batching, prefix accounting) silently writes a wrong slot id. The
canary catches it via per-forward chain_hash violation because stored
prev_hash was computed against the correct slot.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.kv_canary.perturb.config import PerturbConfig
from sglang.srt.kv_canary.perturb.slot_picker import collect_active_slots
from sglang.srt.kv_canary.perturb.utils import WarmupGate

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


def run(
    *,
    forward_batch: Optional["ForwardBatch"],
    config: PerturbConfig,
    req_to_token_pool: "ReqToTokenPool",
    warmup_gate: WarmupGate,
) -> None:
    if config.req_to_token_prob <= 0.0:
        return
    if warmup_gate.is_in_warmup():
        return
    if forward_batch is None:
        return
    if torch.rand((), device="cpu").item() >= config.req_to_token_prob:
        return

    active_targets = collect_active_slots(
        forward_batch=forward_batch,
        req_to_token_pool=req_to_token_pool,
        exclude_out_cache_loc=False,
    )
    # Old req_to_token hook required slot >= 1 (treating 0 as padding sentinel) — preserve.
    active_targets = [t for t in active_targets if t.slot >= 1]
    if not active_targets:
        return

    pick = int(torch.randint(0, len(active_targets), (1,)).item())
    target = active_targets[pick]
    replacement_slots = [
        item.slot for item in active_targets if item.slot != target.slot
    ]
    if not replacement_slots:
        return
    replacement_pick = int(torch.randint(0, len(replacement_slots), (1,)).item())
    new_value = replacement_slots[replacement_pick]

    table = req_to_token_pool.req_to_token
    logger.info(
        "kv_canary perturb req_to_token: req_pool_idx=%d position=%d original_slot=%d new_slot=%d",
        target.req_pool_idx,
        target.position,
        target.slot,
        new_value,
    )
    table[target.req_pool_idx, target.position] = new_value
