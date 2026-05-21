"""Perturb point (b): flip the first byte of a slot currently being used by an
active req.

Detection should come from per-forward verify (HEAD/TAIL kernel), NOT from
sweep. Designed to surface CUDA-graph-idle-class bugs where production reads
a slot whose KV byte was silently overwritten.
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
    pick_active_slot,
    pick_target_group,
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
    if config.real_kv_used_prob <= 0.0:
        return
    if warmup_gate.is_in_warmup():
        return
    if forward_batch is None:
        return
    if torch.rand((), device="cpu").item() >= config.real_kv_used_prob:
        return

    target = pick_active_slot(
        forward_batch=forward_batch,
        req_to_token_pool=req_to_token_pool,
        exclude_out_cache_loc=True,
    )
    if target is None:
        return
    group = pick_target_group(
        buffer_groups=buffer_groups,
        target_kind=config.target_group_kind,
    )
    if group is None or not group.real_kv_sources_k:
        return
    source_pick = int(torch.randint(0, len(group.real_kv_sources_k), (1,)).item())
    source = group.real_kv_sources_k[source_pick]
    flip_result = flip_first_byte_in_source(
        group=group, source=source, slot_idx=target.slot
    )
    if flip_result is None:
        return
    row, col, original_byte = flip_result
    logger.info(
        "kv_canary perturb real_kv_used: group=%s source_idx=%d slot=%d row=%d col=%d "
        "original_byte=0x%02X new_byte=0x%02X",
        group.kind.name,
        source_pick,
        target.slot,
        row,
        col,
        original_byte,
        original_byte ^ 0xFF,
    )
