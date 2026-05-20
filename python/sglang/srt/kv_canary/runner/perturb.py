from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup
from sglang.srt.kv_canary.plan_input import walk_radix_cache_for_canary

if TYPE_CHECKING:
    from sglang.srt.kv_canary.runner.canary_runner import CanaryRunner
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class PerturbHook:
    def __init__(self, *, owner: "CanaryRunner") -> None:
        self._owner = owner
        self._perturb_undo: Optional[tuple[int, int, int]] = None

    def perturb_hook(self, forward_batch: Optional["ForwardBatch"]) -> None:
        owner = self._owner
        if owner.config.perturb_req_to_token_prob <= 0.0:
            return
        table = owner._req_to_token_pool.req_to_token
        if not isinstance(table, torch.Tensor) or table.numel() == 0:
            return
        if forward_batch is None:
            return
        if (
            torch.rand((), device="cpu").item()
            >= owner.config.perturb_req_to_token_prob
        ):
            return

        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        if req_pool_indices is None or seq_lens is None:
            return

        req_pool_indices_cpu = req_pool_indices.detach().to("cpu").tolist()
        seq_lens_cpu = seq_lens.detach().to("cpu").tolist()
        active_pairs: list[tuple[int, int]] = []
        for req_pool_idx, seq_len in zip(req_pool_indices_cpu, seq_lens_cpu):
            req_pool_idx_int = int(req_pool_idx)
            seq_len_int = int(seq_len)
            if seq_len_int <= 0:
                continue
            for pos in range(seq_len_int):
                active_pairs.append((req_pool_idx_int, pos))
        if not active_pairs:
            return

        pick = int(torch.randint(0, len(active_pairs), (1,)).item())
        req_pool_idx, position = active_pairs[pick]
        rows, cols = int(table.shape[0]), int(table.shape[1])
        if req_pool_idx < 0 or req_pool_idx >= rows or position < 0 or position >= cols:
            return

        original = int(table[req_pool_idx, position].item())
        slot_upper = rows * cols
        if slot_upper <= 1:
            return
        new_value = int(torch.randint(0, slot_upper, (1,)).item())
        if new_value == original:
            new_value = (original + 1) % slot_upper
        table[req_pool_idx, position] = new_value
        self._perturb_undo = (req_pool_idx, position, original)

    def perturb_real_kv_hook(self, forward_batch: Optional["ForwardBatch"]) -> None:
        owner = self._owner
        if owner.config.perturb_real_kv_prob <= 0.0:
            return
        if forward_batch is None:
            return
        if torch.rand((), device="cpu").item() >= owner.config.perturb_real_kv_prob:
            return

        candidate_slots = self._collect_real_kv_perturb_candidates(forward_batch)
        if not candidate_slots:
            return

        groups_with_real_kv: list[CanaryBufferGroup] = [
            group for group in owner._groups if group.real_kv_sources_k
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
        flat[row, col] = original_byte ^ 0xFF

    def undo_after_step(self) -> None:
        owner = self._owner
        if self._perturb_undo is not None:
            row, col, original = self._perturb_undo
            owner._req_to_token_pool.req_to_token[row, col] = original
            self._perturb_undo = None

    def _collect_real_kv_perturb_candidates(
        self, forward_batch: "ForwardBatch"
    ) -> list[int]:
        owner = self._owner
        out_cache_loc = forward_batch.out_cache_loc
        excluded: set[int] = set()
        if out_cache_loc is not None:
            excluded = set(int(x) for x in out_cache_loc.detach().to("cpu").tolist())

        orphan_slots = self._collect_radix_orphan_slots(excluded=excluded)
        if orphan_slots:
            return orphan_slots
        if owner.config.perturb_real_kv_require_orphan:
            return []
        return self._collect_running_req_slots(
            forward_batch=forward_batch, excluded=excluded
        )

    def _collect_radix_orphan_slots(self, *, excluded: set[int]) -> list[int]:
        owner = self._owner
        if owner._radix_cache is None:
            return []
        slot_tensor, _, _ = walk_radix_cache_for_canary(
            radix_cache=owner._radix_cache,
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
        owner = self._owner
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        if req_pool_indices is None or seq_lens is None:
            return []

        table = owner._req_to_token_pool.req_to_token
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
