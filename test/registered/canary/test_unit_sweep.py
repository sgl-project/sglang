"""Host-side unit tests for the periodic full-pool sweep path.

The sweep runs every ``real_data_sweep_every_n_steps`` forwards and re-verifies
every slot currently owned by an alive req in the live ``ForwardBatch``. It
reuses the per-step ``kernel_run_pair`` kernel via ``KERNEL_KIND_SWEEP`` and
drives ``verify_prev_slot_indices`` to ``SKIP_CHAIN_SENTINEL`` so the kernel
skips the chain hash check (sweep can only verify ``real_kv_hash``).
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.jit_kernel.kv_cache_canary import (
    CANARY_SLOT_BYTES,
    SKIP_CHAIN_SENTINEL,
)
from sglang.jit_kernel.kv_cache_canary_plan_ref import BatchPlan
from sglang.srt.kv_cache_canary.sweep import (
    build_sweep_plan,
    compute_alive_owned_slots,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="extra-a", runner_config="1-gpu-small")


@dataclass
class _FakeReqToTokenPool:
    """Duck-typed stand-in exposing the bits ``compute_alive_owned_slots`` reads."""

    req_to_token: torch.Tensor


@dataclass
class _FakeForwardBatch:
    """Duck-typed stand-in for ``ForwardBatch`` carrying only the fields sweep needs."""

    req_pool_indices: Optional[torch.Tensor]
    seq_lens: Optional[torch.Tensor]


class TestSweepValiditySet(unittest.TestCase):
    """``compute_alive_owned_slots`` returns the union of owned ranges.

    Uniqueness follows from the allocator invariant — each slot is owned by
    at most one alive req — so the function intentionally does not dedupe;
    callers are responsible for not feeding the same req twice in one batch.
    """

    def _build_pool(self) -> _FakeReqToTokenPool:
        # Row 0 padding, rows 1..3 hold three reqs with seq_lens 5/3/7 below.
        table = torch.zeros((4, 16), dtype=torch.int32)
        table[1, :5] = torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32)
        table[2, :3] = torch.tensor([20, 21, 22], dtype=torch.int32)
        table[3, :7] = torch.tensor([30, 31, 32, 33, 34, 35, 36], dtype=torch.int32)
        return _FakeReqToTokenPool(req_to_token=table)

    def test_compute_alive_owned_slots_returns_union_of_history(self) -> None:
        pool = self._build_pool()
        forward_batch = _FakeForwardBatch(
            req_pool_indices=torch.tensor([1, 2, 3], dtype=torch.int32),
            seq_lens=torch.tensor([5, 3, 7], dtype=torch.int32),
        )
        slots = compute_alive_owned_slots(
            req_to_token_pool=pool, forward_batch=forward_batch
        )
        expected = sorted({10, 11, 12, 13, 14, 20, 21, 22, 30, 31, 32, 33, 34, 35, 36})
        self.assertEqual(sorted(slots.tolist()), expected)

    def test_compute_alive_owned_slots_skips_padding_row(self) -> None:
        pool = self._build_pool()
        # Row 0 (padding) and a real row 2 with seq_len 3.
        forward_batch = _FakeForwardBatch(
            req_pool_indices=torch.tensor([0, 2], dtype=torch.int32),
            seq_lens=torch.tensor([0, 3], dtype=torch.int32),
        )
        slots = compute_alive_owned_slots(
            req_to_token_pool=pool, forward_batch=forward_batch
        )
        self.assertEqual(sorted(slots.tolist()), [20, 21, 22])

    def test_compute_alive_owned_slots_handles_zero_seq_len(self) -> None:
        pool = self._build_pool()
        forward_batch = _FakeForwardBatch(
            req_pool_indices=torch.tensor([1, 2], dtype=torch.int32),
            seq_lens=torch.tensor([0, 3], dtype=torch.int32),
        )
        slots = compute_alive_owned_slots(
            req_to_token_pool=pool, forward_batch=forward_batch
        )
        self.assertEqual(sorted(slots.tolist()), [20, 21, 22])

    def test_compute_alive_owned_slots_returns_empty_for_empty_batch(self) -> None:
        pool = self._build_pool()
        forward_batch = _FakeForwardBatch(
            req_pool_indices=torch.tensor([], dtype=torch.int32),
            seq_lens=torch.tensor([], dtype=torch.int32),
        )
        slots = compute_alive_owned_slots(
            req_to_token_pool=pool, forward_batch=forward_batch
        )
        self.assertEqual(slots.numel(), 0)

    def test_compute_alive_owned_slots_returns_empty_for_none_fields(self) -> None:
        pool = self._build_pool()
        forward_batch = _FakeForwardBatch(req_pool_indices=None, seq_lens=None)
        slots = compute_alive_owned_slots(
            req_to_token_pool=pool, forward_batch=forward_batch
        )
        self.assertEqual(slots.numel(), 0)


class TestBuildSweepPlan(unittest.TestCase):
    """The verify-only plan must skip chain check and feed back stored positions."""

    def _build_canary_buf(self, *, num_slots: int) -> torch.Tensor:
        # 4 int64 fields per slot: [token_id, position, prev_hash, real_kv_hash].
        # Populate position with slot_idx * 7 so we can assert the plan
        # carries that value back from the canary buffer.
        buf = torch.zeros((num_slots, CANARY_SLOT_BYTES), dtype=torch.uint8)
        view = buf.view(torch.int64).reshape(num_slots, -1)
        for slot in range(num_slots):
            view[slot, 1] = slot * 7
        return buf

    def test_build_sweep_plan_empty_input_returns_empty_plan(self) -> None:
        canary_buf = self._build_canary_buf(num_slots=8)
        plan = build_sweep_plan(
            canary_buf=canary_buf,
            alive_slot_indices=torch.empty(0, dtype=torch.int64),
        )
        self.assertEqual(plan.num_verify, 0)
        self.assertEqual(plan.num_write, 0)
        self.assertEqual(plan.num_write_reqs, 0)

    def test_build_sweep_plan_fills_skip_sentinel_and_positions(self) -> None:
        canary_buf = self._build_canary_buf(num_slots=8)
        alive = torch.tensor([2, 5, 6], dtype=torch.int64)
        plan = build_sweep_plan(
            canary_buf=canary_buf,
            alive_slot_indices=alive,
        )
        self.assertEqual(plan.num_verify, 3)
        self.assertEqual(plan.verify_slot_indices, [2, 5, 6])
        # Position field was seeded as slot_idx * 7 above.
        self.assertEqual(plan.verify_positions, [14, 35, 42])
        self.assertEqual(
            plan.verify_prev_slot_indices,
            [SKIP_CHAIN_SENTINEL, SKIP_CHAIN_SENTINEL, SKIP_CHAIN_SENTINEL],
        )
        # Sweep writes nothing.
        self.assertEqual(plan.write_slot_indices, [])
        self.assertEqual(plan.write_token_ids, [])
        self.assertEqual(plan.write_req_seed_slot_indices, [])


class TestBatchPlanEmpty(unittest.TestCase):
    """``BatchPlan.empty()`` is the canonical zero-row plan used by sweep."""

    def test_empty_plan_has_all_zero_counts_and_lists(self) -> None:
        plan = BatchPlan.empty()
        self.assertEqual(plan.num_verify, 0)
        self.assertEqual(plan.num_write, 0)
        self.assertEqual(plan.num_write_reqs, 0)
        self.assertEqual(plan.verify_slot_indices, [])
        self.assertEqual(plan.write_slot_indices, [])
        self.assertEqual(plan.write_req_pool_indices, [])


if __name__ == "__main__":
    unittest.main()
