from __future__ import annotations

import unittest

import torch

from sglang.jit_kernel.kv_canary import consts
from sglang.jit_kernel.kv_canary.consts import RealKvHashMode
from sglang.jit_kernel.kv_canary.plan_ref import run_canary_plan_torch_reference
from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag, VerifyPlan
from sglang.jit_kernel.kv_canary.verify_ref import run_canary_verify_torch_reference
from sglang.jit_kernel.kv_canary.write import WritePlan
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="extra-a", runner_config="1-gpu-large")


class TestPlanRefOverflowGate(CustomTestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")

    @staticmethod
    def _empty_extras(device: torch.device) -> tuple[torch.Tensor, ...]:
        return (
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )

    def _run_plan_ref(
        self, *, verify_capacity: int, bs: int, prefix_lens: list[int]
    ) -> VerifyPlan:
        max_seq_len = max(prefix_lens) + 1
        verify_plan = VerifyPlan.allocate(
            verify_capacity=verify_capacity, device=self.device
        )
        write_plan = WritePlan.allocate(write_req_capacity=bs, device=self.device)
        req_pool_indices = torch.tensor(
            list(range(1, bs + 1)), dtype=torch.int64, device=self.device
        )
        prefix_lens = torch.tensor(
            prefix_lens, dtype=torch.int64, device=self.device
        )
        extend_seq_lens = torch.zeros(bs, dtype=torch.int64, device=self.device)
        req_to_token = torch.arange(
            (bs + 1) * max_seq_len, dtype=torch.int32, device=self.device
        ).reshape(bs + 1, max_seq_len)
        extras = self._empty_extras(self.device)
        run_canary_plan_torch_reference(
            verify_plan_out=verify_plan,
            write_plan_out=write_plan,
            req_pool_indices=req_pool_indices,
            prefix_lens=prefix_lens,
            extend_seq_lens=extend_seq_lens,
            req_to_token=req_to_token,
            extra_verify_slot_indices=extras[0],
            extra_verify_positions=extras[1],
            extra_verify_prev_slot_indices=extras[2],
            extra_verify_num_valid=extras[3],
            swa_window_size=0,
            full_to_swa_index_mapping=None,
            verify_capacity=verify_capacity,
        )
        return verify_plan

    def test_plan_ref_sets_enable_zero_and_clamps_when_overflow(self) -> None:
        """Verify plan reference disables verification and clamps overflow output."""
        # requested = sum(prefix_lens) = 8 > capacity = 4.
        plan = self._run_plan_ref(verify_capacity=4, bs=2, prefix_lens=[5, 5])
        self.assertEqual(int(plan.enable[0].item()), 0)
        self.assertEqual(int(plan.verify_num_valid[0].item()), 4)

    def test_plan_ref_sets_enable_one_when_within_capacity(self) -> None:
        """Verify plan reference enables verification within capacity."""
        # requested = 4 <= capacity = 16.
        plan = self._run_plan_ref(verify_capacity=16, bs=2, prefix_lens=[2, 2])
        self.assertEqual(int(plan.enable[0].item()), 1)
        self.assertEqual(int(plan.verify_num_valid[0].item()), 4)

    def test_verify_ref_skips_when_enable_zero(self) -> None:
        """Verify verify reference skips work when the plan is disabled."""
        plan = self._run_plan_ref(verify_capacity=4, bs=2, prefix_lens=[5, 5])
        self.assertEqual(int(plan.enable[0].item()), 0)

        canary_buf = torch.zeros(64, 32, dtype=torch.uint8, device=self.device)
        violation_ring = torch.zeros(
            4, consts.VIOLATION_FIELDS, dtype=torch.int64, device=self.device
        )
        violation_write_index = torch.zeros(1, dtype=torch.int32, device=self.device)
        slot_run_counter = torch.zeros(1, dtype=torch.int64, device=self.device)
        kernel_run_counter = torch.zeros(1, dtype=torch.int64, device=self.device)
        ring_before = violation_ring.clone()

        run_canary_verify_torch_reference(
            canary_buf=canary_buf,
            plan=plan,
            kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
            violation_ring=violation_ring,
            violation_write_index=violation_write_index,
            slot_run_counter=slot_run_counter,
            kernel_run_counter=kernel_run_counter,
            real_kv_sources=(),
            real_kv_hash_mode=RealKvHashMode.OFF,
        )
        # kernel_run_counter bumps regardless of enable; everything else must be untouched.
        self.assertEqual(int(kernel_run_counter[0].item()), 1)
        self.assertEqual(int(violation_write_index[0].item()), 0)
        self.assertEqual(int(slot_run_counter[0].item()), 0)
        self.assertTrue(torch.equal(violation_ring, ring_before))


if __name__ == "__main__":
    unittest.main()
