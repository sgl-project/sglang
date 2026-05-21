from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup
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
        self._perturb_undo: Optional[tuple[int, int, int]] = None
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

    def perturb_hook(self, forward_batch: Optional["ForwardBatch"]) -> None:
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
        self._perturb_undo = (target.req_pool_idx, target.position, target.slot)

    def perturb_real_kv_hook(self, forward_batch: Optional["ForwardBatch"]) -> None:
        if self._config.real_kv_prob <= 0.0:
            return
        if self._is_in_warmup():
            return
        if forward_batch is None:
            return
        if torch.rand((), device="cpu").item() >= self._config.real_kv_prob:
            return

        candidate_slots = self._collect_real_kv_perturb_candidates(forward_batch)
        if not candidate_slots:
            return

        groups_with_real_kv: list[CanaryBufferGroup] = [
            group for group in self._buffer_groups if group.real_kv_sources_k
        ]
        if not groups_with_real_kv:
            return

        group_pick = int(torch.randint(0, len(groups_with_real_kv), (1,)).item())
        group = groups_with_real_kv[group_pick]
        sources = group.real_kv_sources_k
        source_pick = int(torch.randint(0, len(sources), (1,)).item())
        source = sources[source_pick]
        if source.read_bytes <= 0 or source.num_bytes_per_token <= 0:
            return

        slot_pick = int(torch.randint(0, len(candidate_slots), (1,)).item())
        slot_idx = int(candidate_slots[slot_pick])
        row = slot_idx // max(1, source.page_size)
        col_base = (slot_idx % max(1, source.page_size)) * source.num_bytes_per_token
        max_offset = min(int(source.read_bytes), int(source.num_bytes_per_token))
        if max_offset <= 0:
            return
        if row < 0 or row >= int(source.tensor.shape[0]):
            return
        byte_offset = int(torch.randint(0, max_offset, (1,)).item())
        col = col_base + byte_offset
        if col < 0 or col >= int(source.tensor.shape[1]):
            return

        flat = source.tensor
        original_byte = int(flat[row, col].item())
        new_byte = original_byte ^ 0xFF
        logger.info(
            "kv_canary perturb real_kv: group_kind=%s source_idx=%d slot=%d row=%d col=%d "
            "byte_offset=%d original_byte=0x%02X new_byte=0x%02X",
            group.kind.name,
            source_pick,
            slot_idx,
            row,
            col,
            byte_offset,
            original_byte,
            new_byte,
        )
        flat[row, col] = new_byte

    def undo_after_step(self) -> None:
        if self._perturb_undo is not None:
            row, col, original = self._perturb_undo
            self._req_to_token_pool.req_to_token[row, col] = original
            logger.info(
                "kv_canary perturb undo req_to_token: req_pool_idx=%d position=%d restored_slot=%d",
                row,
                col,
                original,
            )
            self._perturb_undo = None

    def _collect_real_kv_perturb_candidates(
        self, forward_batch: "ForwardBatch"
    ) -> list[int]:
        out_cache_loc = forward_batch.out_cache_loc
        excluded: set[int] = set()
        if out_cache_loc is not None:
            excluded = set(int(x) for x in out_cache_loc.detach().to("cpu").tolist())

        orphan_slots = self._collect_radix_orphan_slots(excluded=excluded)
        if orphan_slots:
            return orphan_slots
        if self._config.real_kv_require_orphan:
            return []
        return self._collect_running_req_slots(
            forward_batch=forward_batch, excluded=excluded
        )

    def _collect_radix_orphan_slots(self, *, excluded: set[int]) -> list[int]:
        if self._radix_cache is None:
            return []
        slot_tensor, _, _ = walk_radix_cache_for_canary(
            radix_cache=self._radix_cache,
            unlocked_only=True,
        )
        if slot_tensor.numel() == 0:
            return []
        candidates: list[int] = []
        for raw_slot in slot_tensor.tolist():
            slot = int(raw_slot)
            if slot < 0:
                continue
            if slot in excluded:
                continue
            candidates.append(slot)
        return candidates

    def _collect_running_req_slots(
        self,
        *,
        forward_batch: "ForwardBatch",
        excluded: set[int],
    ) -> list[int]:
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        if req_pool_indices is None or seq_lens is None:
            return []

        table = self._req_to_token_pool.req_to_token
        if not isinstance(table, torch.Tensor) or table.numel() == 0:
            return []

        req_pool_indices_cpu = req_pool_indices.detach().to("cpu").tolist()
        seq_lens_cpu = seq_lens.detach().to("cpu").tolist()
        rows, cols = int(table.shape[0]), int(table.shape[1])
        candidate_slots: list[int] = []
        for req_pool_idx, seq_len in zip(req_pool_indices_cpu, seq_lens_cpu):
            req_pool_idx_int = int(req_pool_idx)
            seq_len_int = int(seq_len)
            if req_pool_idx_int < 0 or req_pool_idx_int >= rows:
                continue
            upper = min(seq_len_int, cols)
            if upper <= 0:
                continue
            row_slots = table[req_pool_idx_int, :upper].detach().to("cpu").tolist()
            for raw_slot in row_slots:
                slot = int(raw_slot)
                if slot < 0:
                    continue
                if slot in excluded:
                    continue
                candidate_slots.append(slot)
        return candidate_slots
