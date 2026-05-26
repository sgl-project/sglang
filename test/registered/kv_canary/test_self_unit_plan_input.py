from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.kv_canary.plan_input import PlanInput
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.fixtures import (
    DEFAULT_DEVICE,
    make_forward_batch,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="extra-a", runner_config="1-gpu-small")


def _make_static_plan_input(*, bs_capacity: int, device) -> PlanInput:
    return PlanInput(
        req_pool_indices=torch.zeros(bs_capacity, dtype=torch.int64, device=device),
        prefix_lens=torch.zeros(bs_capacity, dtype=torch.int64, device=device),
        extend_seq_lens=torch.zeros(bs_capacity, dtype=torch.int64, device=device),
        req_to_verify_expected_tokens_valid_lens=torch.zeros(
            bs_capacity, dtype=torch.int64, device=device
        ),
    )


class TestSelfUnitPlanInput(CustomTestCase):
    def setUp(self):
        self.device = DEFAULT_DEVICE

    def test_plan_input_fill_from_forward_batch_extend(self):
        """Verify extend batches populate per-forward plan inputs."""
        fb = make_forward_batch(
            self.device,
            req_pool_indices=torch.tensor(
                [1, 2], dtype=torch.int64, device=self.device
            ),
            seq_lens=torch.tensor([10, 12], dtype=torch.int32, device=self.device),
            extend_prefix_lens=torch.tensor(
                [3, 5], dtype=torch.int32, device=self.device
            ),
            extend_seq_lens=torch.tensor([7, 7], dtype=torch.int32, device=self.device),
            is_extend=True,
        )
        plan = _make_static_plan_input(bs_capacity=4, device=self.device)
        plan.fill_from_forward_batch(forward_batch=fb)
        self.assertEqual(plan.req_pool_indices[:2].tolist(), [1, 2])
        self.assertEqual(plan.req_pool_indices[2:].tolist(), [0, 0])
        self.assertEqual(plan.prefix_lens[:2].tolist(), [3, 5])
        self.assertEqual(plan.extend_seq_lens[:2].tolist(), [7, 7])
        self.assertEqual(plan.prefix_lens.dtype, torch.int64)
        self.assertEqual(plan.extend_seq_lens.dtype, torch.int64)

    def test_plan_input_fill_from_forward_batch_target_verify(self):
        """Verify target-verify batches derive draft verification spans."""
        # TARGET_VERIFY writes spec_info.draft_token_num positions per req starting at the
        # current (un-bumped) seq_lens; extend_prefix_lens / extend_seq_lens are deliberately
        # NOT supplied because init_new does not populate them for this mode.
        spec_info = SimpleNamespace(draft_token_num=4)
        fb = make_forward_batch(
            self.device,
            req_pool_indices=torch.tensor(
                [6, 7], dtype=torch.int64, device=self.device
            ),
            seq_lens=torch.tensor([10, 14], dtype=torch.int32, device=self.device),
            is_target_verify=True,
            spec_info=spec_info,
        )
        plan = _make_static_plan_input(bs_capacity=4, device=self.device)
        plan.fill_from_forward_batch(forward_batch=fb)
        self.assertEqual(plan.prefix_lens[:2].tolist(), [10, 14])
        self.assertEqual(plan.extend_seq_lens[:2].tolist(), [4, 4])

    def test_plan_input_fill_from_forward_batch_draft_extend_v2(self):
        """Verify draft-extend-v2 batches derive prefix lengths from sequence lengths."""
        # DRAFT_EXTEND_V2 has seq_lens already bumped by the per-req draft refill length;
        # extend_prefix_lens is intentionally absent (cuda-graph replay does not set it). The
        # builder must derive prefix as seq_lens - extend_seq_lens.
        fb = make_forward_batch(
            self.device,
            req_pool_indices=torch.tensor(
                [4, 5], dtype=torch.int64, device=self.device
            ),
            seq_lens=torch.tensor([14, 18], dtype=torch.int32, device=self.device),
            extend_seq_lens=torch.tensor([4, 4], dtype=torch.int32, device=self.device),
            is_draft_extend_v2=True,
        )
        plan = _make_static_plan_input(bs_capacity=4, device=self.device)
        plan.fill_from_forward_batch(forward_batch=fb)
        self.assertEqual(plan.prefix_lens[:2].tolist(), [10, 14])
        self.assertEqual(plan.extend_seq_lens[:2].tolist(), [4, 4])

    def test_plan_input_fill_from_forward_batch_decode(self):
        """Verify decode batches populate one-token verification spans."""
        fb = make_forward_batch(
            self.device,
            req_pool_indices=torch.tensor(
                [1, 2, 3], dtype=torch.int64, device=self.device
            ),
            seq_lens=torch.tensor([4, 7, 1], dtype=torch.int32, device=self.device),
            is_extend=False,
        )
        plan = _make_static_plan_input(bs_capacity=4, device=self.device)
        plan.fill_from_forward_batch(forward_batch=fb)
        self.assertEqual(plan.prefix_lens[:3].tolist(), [3, 6, 0])
        self.assertEqual(plan.extend_seq_lens[:3].tolist(), [1, 1, 1])

    def test_plan_input_mirrors_req_all_ids_lens(self):
        """req_to_verify_expected_tokens_valid_lens copies forward_batch.req_all_ids_lens for active rows."""
        fb = make_forward_batch(
            self.device,
            req_pool_indices=torch.tensor(
                [1, 2], dtype=torch.int64, device=self.device
            ),
            seq_lens=torch.tensor([10, 12], dtype=torch.int32, device=self.device),
            is_extend=False,
        )
        fb.req_all_ids_lens = torch.tensor([7, 9], dtype=torch.int64, pin_memory=True)
        plan = _make_static_plan_input(bs_capacity=4, device=self.device)
        plan.fill_from_forward_batch(forward_batch=fb)
        torch.cuda.synchronize()
        self.assertEqual(
            plan.req_to_verify_expected_tokens_valid_lens[:2].tolist(), [7, 9]
        )
        # Padding tail stays at zero so the plan kernel reads "no in-range positions" for it.
        self.assertEqual(
            plan.req_to_verify_expected_tokens_valid_lens[2:].tolist(), [0, 0]
        )

    def test_plan_input_padding_dummy_sentinel(self):
        """Verify padding sentinel rows remain valid plan input entries."""
        fb = make_forward_batch(
            self.device,
            req_pool_indices=torch.tensor(
                [0, 5, 0], dtype=torch.int64, device=self.device
            ),
            seq_lens=torch.tensor([0, 3, 0], dtype=torch.int32, device=self.device),
            is_extend=False,
        )
        plan = _make_static_plan_input(bs_capacity=4, device=self.device)
        plan.fill_from_forward_batch(forward_batch=fb)
        self.assertEqual(plan.req_pool_indices[:3].tolist(), [0, 5, 0])
        self.assertEqual(plan.req_pool_indices.dtype, torch.int64)


if __name__ == "__main__":
    unittest.main()
