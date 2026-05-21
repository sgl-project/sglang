from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.kv_canary.verify import RealKvSource
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.plan_input import walk_radix_cache_for_canary
from sglang.srt.kv_canary.runner.perturb_config import PerturbConfig
from sglang.srt.kv_canary.runner.pump import PumpAndAllreduce

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class _ReqToTokenPerturbTarget:
    req_pool_idx: int
    position: int
    slot: int


@dataclass(frozen=True, slots=True, kw_only=True)
class _ActiveSlotTarget:
    req_pool_idx: int
    position: int
    slot: int


class PerturbHook:
    def __init__(
        self,
        *,
        config: PerturbConfig,
        req_to_token_pool: "ReqToTokenPool",
        buffer_groups: tuple[CanaryBufferGroup, ...],
        pump_and_allreduce: PumpAndAllreduce,
    ) -> None:
        self._config = config
        self._req_to_token_pool = req_to_token_pool
        self._buffer_groups = buffer_groups
        self._pump_and_allreduce = pump_and_allreduce
        self._radix_cache: Optional["BasePrefixCache"] = None
        self._warmup_disable_logged: bool = False
        self._warmup_enable_logged: bool = False

    def attach_radix_cache(self, radix_cache: "BasePrefixCache") -> None:
        self._radix_cache = radix_cache

    def _is_in_warmup(self) -> bool:
        """Return True if the perturb hook should skip this step because the canary is still
        inside the configured warmup window. Emits the disable/enable transition logs exactly
        once across the lifetime of the hook (idempotent under multiple per-step calls).
        """
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

    def perturb_req_to_token_hook(self, forward_batch: Optional["ForwardBatch"]) -> None:
        if self._config.req_to_token_prob <= 0.0:
            return
        if self._is_in_warmup():
            return
        table = self._req_to_token_pool.req_to_token
        if not isinstance(table, torch.Tensor) or table.numel() == 0:
            return
        if forward_batch is None:
            return
        if torch.rand((), device="cpu").item() >= self._config.req_to_token_prob:
            return

        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        if req_pool_indices is None or seq_lens is None:
            return

        req_pool_indices_cpu = req_pool_indices.detach().to("cpu").tolist()
        seq_lens_cpu = seq_lens.detach().to("cpu").tolist()
        rows, cols = int(table.shape[0]), int(table.shape[1])
        active_targets: list[_ReqToTokenPerturbTarget] = []
        for req_pool_idx, seq_len in zip(req_pool_indices_cpu, seq_lens_cpu):
            req_pool_idx_int = int(req_pool_idx)
            seq_len_int = int(seq_len)
            if seq_len_int <= 0:
                continue
            if req_pool_idx_int < 0 or req_pool_idx_int >= rows:
                continue
            upper = min(seq_len_int, cols)
            row_slots = table[req_pool_idx_int, :upper].detach().to("cpu").tolist()
            for pos, raw_slot in enumerate(row_slots):
                slot = int(raw_slot)
                if slot < 1:
                    continue
                active_targets.append(
                    _ReqToTokenPerturbTarget(
                        req_pool_idx=req_pool_idx_int,
                        position=pos,
                        slot=slot,
                    )
                )
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

        logger.info(
            "kv_canary perturb req_to_token: req_pool_idx=%d position=%d original_slot=%d new_slot=%d",
            target.req_pool_idx,
            target.position,
            target.slot,
            new_value,
        )
        table[target.req_pool_idx, target.position] = new_value

    def perturb_real_kv_used_hook(self, forward_batch: Optional["ForwardBatch"]) -> None:
        """Inject corruption into an active req's currently-used slot. Detection should come
        from per-forward verify (HEAD/TAIL kernel), NOT from sweep. Designed to surface
        CUDA-graph-idle-class bugs where production reads a slot whose KV byte was silently
        overwritten."""
        if self._config.real_kv_used_prob <= 0.0:
            return
        if self._is_in_warmup():
            return
        if forward_batch is None:
            return
        if torch.rand((), device="cpu").item() >= self._config.real_kv_used_prob:
            return

        target = _pick_active_slot(
            forward_batch=forward_batch,
            req_to_token_pool=self._req_to_token_pool,
            exclude_out_cache_loc=True,
        )
        if target is None:
            return
        group = _pick_target_group(
            buffer_groups=self._buffer_groups,
            target_kind=self._config.target_group_kind,
        )
        if group is None or not group.real_kv_sources_k:
            return
        source_pick = int(torch.randint(0, len(group.real_kv_sources_k), (1,)).item())
        source = group.real_kv_sources_k[source_pick]
        flip_result = _flip_first_byte_in_source(
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

    def perturb_real_kv_unused_cache_hook(
        self, forward_batch: Optional["ForwardBatch"]
    ) -> None:
        """Inject corruption into a radix-cached but currently-unused (orphan) slot. Detection
        should come from sweep (per-forward verify won't even look at this slot). Designed to
        surface bugs where cached KV is silently corrupted and sleeps until much later when a
        prefix happens to match."""
        if self._config.real_kv_unused_cache_prob <= 0.0:
            return
        if self._is_in_warmup():
            return
        if torch.rand((), device="cpu").item() >= self._config.real_kv_unused_cache_prob:
            return

        slot = _pick_orphan_slot(radix_cache=self._radix_cache)
        if slot is None:
            return
        group = _pick_target_group(
            buffer_groups=self._buffer_groups,
            target_kind=self._config.target_group_kind,
        )
        if group is None or not group.real_kv_sources_k:
            return
        source_pick = int(torch.randint(0, len(group.real_kv_sources_k), (1,)).item())
        source = group.real_kv_sources_k[source_pick]
        flip_result = _flip_first_byte_in_source(
            group=group, source=source, slot_idx=slot
        )
        if flip_result is None:
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


def _pick_active_slot(
    *,
    forward_batch: "ForwardBatch",
    req_to_token_pool: "ReqToTokenPool",
    exclude_out_cache_loc: bool = True,
) -> Optional[_ActiveSlotTarget]:
    """Pick one random (req_pool_idx, position, slot) triple from currently-active reqs.

    Mirrors the old _collect_running_req_slots walk but emits a single _ActiveSlotTarget
    so the caller doesn't need to know the position/req for logging. Returns None if no
    candidate exists.
    """
    req_pool_indices = forward_batch.req_pool_indices
    seq_lens = forward_batch.seq_lens
    if req_pool_indices is None or seq_lens is None:
        return None

    table = req_to_token_pool.req_to_token
    if not isinstance(table, torch.Tensor) or table.numel() == 0:
        return None

    excluded: set[int] = set()
    if exclude_out_cache_loc:
        out_cache_loc = forward_batch.out_cache_loc
        if out_cache_loc is not None:
            excluded = set(int(x) for x in out_cache_loc.detach().to("cpu").tolist())

    req_pool_indices_cpu = req_pool_indices.detach().to("cpu").tolist()
    seq_lens_cpu = seq_lens.detach().to("cpu").tolist()
    rows, cols = int(table.shape[0]), int(table.shape[1])

    candidates: list[_ActiveSlotTarget] = []
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
                _ActiveSlotTarget(
                    req_pool_idx=req_pool_idx_int,
                    position=pos,
                    slot=slot,
                )
            )
    if not candidates:
        return None

    pick = int(torch.randint(0, len(candidates), (1,)).item())
    return candidates[pick]


def _pick_orphan_slot(*, radix_cache: Optional["BasePrefixCache"]) -> Optional[int]:
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


def _pick_target_group(
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


def _flip_first_byte_in_source(
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
