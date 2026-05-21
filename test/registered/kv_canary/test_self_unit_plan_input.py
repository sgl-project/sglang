from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.kv_canary.plan_input_builder import (
    PlanInput,
    fill_plan_input_per_forward,
)
from sglang.srt.kv_canary.sweep_plan_builder import build_verify_plan_radix_sweep
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.fixtures import (
    DEFAULT_DEVICE,
    make_forward_batch,
    make_radix_cache,
    make_req_to_token_pool,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="extra-a", runner_config="1-gpu-large")


def _make_static_plan_input(*, bs_capacity: int, device) -> PlanInput:
    return PlanInput(
        fb_req_pool_indices=torch.zeros(bs_capacity, dtype=torch.int64, device=device),
        fb_prefix_lens=torch.zeros(bs_capacity, dtype=torch.int64, device=device),
        fb_extend_seq_lens=torch.zeros(bs_capacity, dtype=torch.int64, device=device),
    )


class TestSelfUnitPlanInput(CustomTestCase):
    def setUp(self):
        self.device = DEFAULT_DEVICE

    def test_fill_plan_input_per_forward_extend(self):
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
        bs = fill_plan_input_per_forward(forward_batch=fb, plan_input_out=plan)
        self.assertEqual(bs, 2)
        self.assertEqual(plan.fb_req_pool_indices[:2].tolist(), [1, 2])
        self.assertEqual(plan.fb_req_pool_indices[2:].tolist(), [0, 0])
        self.assertEqual(plan.fb_prefix_lens[:2].tolist(), [3, 5])
        self.assertEqual(plan.fb_extend_seq_lens[:2].tolist(), [7, 7])
        self.assertEqual(plan.fb_prefix_lens.dtype, torch.int64)
        self.assertEqual(plan.fb_extend_seq_lens.dtype, torch.int64)

    def test_fill_plan_input_per_forward_target_verify(self):
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
        bs = fill_plan_input_per_forward(forward_batch=fb, plan_input_out=plan)
        self.assertEqual(bs, 2)
        self.assertEqual(plan.fb_prefix_lens[:2].tolist(), [10, 14])
        self.assertEqual(plan.fb_extend_seq_lens[:2].tolist(), [4, 4])

    def test_fill_plan_input_per_forward_draft_extend_v2(self):
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
        bs = fill_plan_input_per_forward(forward_batch=fb, plan_input_out=plan)
        self.assertEqual(bs, 2)
        self.assertEqual(plan.fb_prefix_lens[:2].tolist(), [10, 14])
        self.assertEqual(plan.fb_extend_seq_lens[:2].tolist(), [4, 4])

    def test_fill_plan_input_per_forward_decode(self):
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
        bs = fill_plan_input_per_forward(forward_batch=fb, plan_input_out=plan)
        self.assertEqual(bs, 3)
        self.assertEqual(plan.fb_prefix_lens[:3].tolist(), [3, 6, 0])
        self.assertEqual(plan.fb_extend_seq_lens[:3].tolist(), [1, 1, 1])

    def test_build_verify_plan_radix_sweep(self):
        """Verify radix sweep verify plans include cached slot chains."""
        empty_cache = make_radix_cache([[]], device=self.device)
        empty_cache.req_to_token_pool = make_req_to_token_pool(self.device)
        empty_out = build_verify_plan_radix_sweep(
            radix_cache=empty_cache,
            swa_window_size=0,
            full_to_swa_index_mapping=None,
        )
        self.assertEqual(int(empty_out.verify_num_valid.item()), 0)

        cache = make_radix_cache([[], [100, 101, 102]], device=self.device)
        cache.req_to_token_pool = make_req_to_token_pool(self.device)
        out = build_verify_plan_radix_sweep(
            radix_cache=cache,
            swa_window_size=0,
            full_to_swa_index_mapping=None,
        )
        self.assertEqual(int(out.verify_num_valid.item()), 3)
        self.assertEqual(out.verify_slot_indices.dtype, torch.int64)
        self.assertEqual(out.verify_positions.dtype, torch.int64)
        self.assertEqual(out.verify_prev_slot_indices.dtype, torch.int64)
        self.assertEqual(out.verify_slot_indices[:3].tolist(), [100, 101, 102])
        self.assertEqual(out.verify_positions[:3].tolist(), [0, 1, 2])
        self.assertEqual(out.verify_prev_slot_indices[:3].tolist(), [-1, 100, 101])

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
        fill_plan_input_per_forward(forward_batch=fb, plan_input_out=plan)
        self.assertEqual(plan.fb_req_pool_indices[:3].tolist(), [0, 5, 0])
        self.assertEqual(plan.fb_req_pool_indices.dtype, torch.int64)

    def test_radix_held_slot_still_swept(self):
        """Verify held radix slots are still included in sweep plans."""
        cache = make_radix_cache([[], [42, 43, 44]], device=self.device)
        cache.req_to_token_pool = make_req_to_token_pool(self.device)
        out = build_verify_plan_radix_sweep(
            radix_cache=cache,
            swa_window_size=0,
            full_to_swa_index_mapping=None,
        )
        n = int(out.verify_num_valid.item())
        self.assertEqual(n, 3)
        self.assertEqual(set(out.verify_slot_indices[:n].tolist()), {42, 43, 44})

    def test_truly_free_slot_not_swept(self):
        """Verify free radix slots are excluded from sweep plans."""
        empty_cache = make_radix_cache([[]], device=self.device)
        empty_cache.req_to_token_pool = make_req_to_token_pool(self.device)
        out = build_verify_plan_radix_sweep(
            radix_cache=empty_cache,
            swa_window_size=0,
            full_to_swa_index_mapping=None,
        )
        self.assertEqual(int(out.verify_num_valid.item()), 0)


if __name__ == "__main__":
    unittest.main()
