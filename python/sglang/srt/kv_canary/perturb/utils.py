from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.kv_canary.verify import RealKvSource
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.plan_input import walk_radix_cache_for_canary
from sglang.srt.kv_canary.perturb.config import PerturbConfig
from sglang.srt.kv_canary.runner.pump import PumpAndAllreduce

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class ActiveSlotTarget:
    req_pool_idx: int
    position: int
    slot: int


class WarmupGate:
    """Per-hook warmup window check + once-per-lifetime disable/enable log emission.

    Shared across the three perturb-point hooks so warmup state is decided in one place
    rather than duplicated per hook.
    """

    def __init__(
        self,
        *,
        config: PerturbConfig,
        pump_and_allreduce: PumpAndAllreduce,
    ) -> None:
        self._config = config
        self._pump_and_allreduce = pump_and_allreduce
        self._warmup_disable_logged: bool = False
        self._warmup_enable_logged: bool = False

    def is_in_warmup(self) -> bool:
        step = self._pump_and_allreduce.step_counter
        warmup_steps = self._config.warmup_steps
        if step < warmup_steps:
            if not self._warmup_disable_logged:
                logger.info(
                    "kv_canary perturb: disabled during warmup window "
                    "(first %d forward steps)",
                    warmup_steps,
                )
                self._warmup_disable_logged = True
            return True
        if not self._warmup_enable_logged:
            logger.info(
                "kv_canary perturb: enabled after warmup window at step=%d", step
            )
            self._warmup_enable_logged = True
        return False


def collect_active_slots(
    *,
    forward_batch: "ForwardBatch",
    req_to_token_pool: "ReqToTokenPool",
    exclude_out_cache_loc: bool = True,
) -> list[ActiveSlotTarget]:
    """Collect every (req_pool_idx, position, slot) triple for currently-active reqs.

    Excludes slots in ``forward_batch.out_cache_loc`` when ``exclude_out_cache_loc=True``
    so a slot the current forward is about to write isn't picked (write race).
    """
    req_pool_indices = forward_batch.req_pool_indices
    seq_lens = forward_batch.seq_lens
    if req_pool_indices is None or seq_lens is None:
        return []

    table = req_to_token_pool.req_to_token
    if not isinstance(table, torch.Tensor) or table.numel() == 0:
        return []

    excluded: set[int] = set()
    if exclude_out_cache_loc:
        out_cache_loc = forward_batch.out_cache_loc
        if out_cache_loc is not None:
            excluded = set(int(x) for x in out_cache_loc.detach().to("cpu").tolist())

    req_pool_indices_cpu = req_pool_indices.detach().to("cpu").tolist()
    seq_lens_cpu = seq_lens.detach().to("cpu").tolist()
    rows, cols = int(table.shape[0]), int(table.shape[1])

    candidates: list[ActiveSlotTarget] = []
    for req_pool_idx, seq_len in zip(req_pool_indices_cpu, seq_lens_cpu):
        req_pool_idx_int = int(req_pool_idx)
        seq_len_int = int(seq_len)
        if req_pool_idx_int < 0 or req_pool_idx_int >= rows:
            continue
        upper = min(seq_len_int, cols)
        if upper <= 0:
            continue
        row_slots = table[req_pool_idx_int, :upper].detach().to("cpu").tolist()
        for pos, raw_slot in enumerate(row_slots):
            slot = int(raw_slot)
            if slot < 0:
                continue
            if slot in excluded:
                continue
            candidates.append(
                ActiveSlotTarget(
                    req_pool_idx=req_pool_idx_int,
                    position=pos,
                    slot=slot,
                )
            )
    return candidates


def pick_active_slot(
    *,
    forward_batch: "ForwardBatch",
    req_to_token_pool: "ReqToTokenPool",
    exclude_out_cache_loc: bool = True,
) -> Optional[ActiveSlotTarget]:
    """Random pick from ``collect_active_slots`` output. Returns None if no candidate."""
    candidates = collect_active_slots(
        forward_batch=forward_batch,
        req_to_token_pool=req_to_token_pool,
        exclude_out_cache_loc=exclude_out_cache_loc,
    )
    if not candidates:
        return None
    pick = int(torch.randint(0, len(candidates), (1,)).item())
    return candidates[pick]


def pick_orphan_slot(*, radix_cache: Optional["BasePrefixCache"]) -> Optional[int]:
    """Pick one random orphan slot (radix-cached but not currently locked by any active req).
    Returns None if radix_cache is None or no orphan slots exist."""
    if radix_cache is None:
        return None
    slot_tensor, _, _ = walk_radix_cache_for_canary(
        radix_cache=radix_cache,
        unlocked_only=True,
    )
    if slot_tensor.numel() == 0:
        return None
    valid: list[int] = []
    for raw_slot in slot_tensor.tolist():
        slot = int(raw_slot)
        if slot < 0:
            continue
        valid.append(slot)
    if not valid:
        return None
    pick = int(torch.randint(0, len(valid), (1,)).item())
    return valid[pick]


def pick_target_group(
    *,
    buffer_groups: tuple[CanaryBufferGroup, ...],
    target_kind: str,
) -> Optional[CanaryBufferGroup]:
    """Filter buffer_groups by target_kind ('full' / 'swa' exact, 'any' random) restricted to
    groups with non-empty real_kv_sources_k. Returns None if no group matches."""
    eligible = [group for group in buffer_groups if group.real_kv_sources_k]
    if not eligible:
        return None
    if target_kind == "any":
        pick = int(torch.randint(0, len(eligible), (1,)).item())
        return eligible[pick]
    want = PoolKind.FULL if target_kind == "full" else PoolKind.SWA
    filtered = [group for group in eligible if group.kind == want]
    if not filtered:
        return None
    pick = int(torch.randint(0, len(filtered), (1,)).item())
    return filtered[pick]


def flip_first_byte_in_source(
    *,
    group: CanaryBufferGroup,
    source: RealKvSource,
    slot_idx: int,
) -> Optional[tuple[int, int, int]]:
    """XOR 0xFF on byte_offset=0 of slot_idx in source.tensor.

    For SWA groups, slot_idx is translated through group.swa_index_lut before computing
    (row, col). Returns (row, col, original_byte) for logging, or None if the slot is
    out-of-range / source is degenerate.
    """
    if source.num_bytes_per_token <= 0 or source.read_bytes <= 0:
        return None

    physical_slot = slot_idx
    if group.kind == PoolKind.SWA and group.swa_index_lut is not None:
        lut = group.swa_index_lut
        if slot_idx < 0 or slot_idx >= int(lut.shape[0]):
            return None
        physical_slot = int(lut[slot_idx].detach().to("cpu").item())
        if physical_slot < 0:
            return None

    page_size = max(1, source.page_size)
    row = physical_slot // page_size
    col = (physical_slot % page_size) * source.num_bytes_per_token
    if row < 0 or row >= int(source.tensor.shape[0]):
        return None
    if col < 0 or col >= int(source.tensor.shape[1]):
        return None

    flat = source.tensor
    original_byte = int(flat[row, col].item())
    flat[row, col] = original_byte ^ 0xFF
    return row, col, original_byte
