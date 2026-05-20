from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from _fixtures import (  # noqa: E402
    CPU_DEVICE,
    make_forward_batch,
    make_radix_cache,
    make_req_to_token_pool,
)

from sglang.srt.kv_canary.plan_input import (  # noqa: E402
    PlanInput,
    build_plan_input_radix_sweep,
    fill_plan_input_per_forward,
)
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cuda_ci(est_time=30, stage="extra-a", runner_config="1-gpu-large")


def _make_static_plan_input(*, bs_capacity: int, device) -> PlanInput:
    return PlanInput(
        fb_req_pool_indices=torch.zeros(bs_capacity, dtype=torch.int64, device=device),
        fb_prefix_lens=torch.zeros(bs_capacity, dtype=torch.int32, device=device),
        fb_extend_seq_lens=torch.zeros(bs_capacity, dtype=torch.int32, device=device),
        extra_verify_slot_indices=torch.zeros(0, dtype=torch.int32, device=device),
        extra_verify_positions=torch.zeros(0, dtype=torch.int32, device=device),
        extra_verify_prev_slot_indices=torch.zeros(0, dtype=torch.int32, device=device),
        extra_verify_num_valid=torch.zeros(1, dtype=torch.int32, device=device),
    )


class TestSelfUnitPlanInput(CustomTestCase):
    def setUp(self):
        self.device = CPU_DEVICE

    def test_fill_plan_input_per_forward_extend(self):
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

    def test_fill_plan_input_per_forward_decode(self):
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

    def test_build_plan_input_radix_sweep(self):
        empty_cache = make_radix_cache([[]], device=self.device)
        empty_cache.req_to_token_pool = make_req_to_token_pool(self.device)
        empty_out = build_plan_input_radix_sweep(
            radix_cache=empty_cache,
            swa_window_size=0,
            full_to_swa_index_mapping=None,
        )
        self.assertEqual(int(empty_out.extra_verify_num_valid.item()), 0)

        cache = make_radix_cache([[], [100, 101, 102]], device=self.device)
        cache.req_to_token_pool = make_req_to_token_pool(self.device)
        out = build_plan_input_radix_sweep(
            radix_cache=cache,
            swa_window_size=0,
            full_to_swa_index_mapping=None,
        )
        self.assertEqual(int(out.extra_verify_num_valid.item()), 3)
        self.assertEqual(out.extra_verify_slot_indices[:3].tolist(), [100, 101, 102])
        self.assertEqual(out.extra_verify_positions[:3].tolist(), [0, 1, 2])
        self.assertEqual(
            out.extra_verify_prev_slot_indices[:3].tolist(), [-1, 100, 101]
        )

    def test_plan_input_padding_dummy_sentinel(self):
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
        cache = make_radix_cache([[], [42, 43, 44]], device=self.device)
        cache.req_to_token_pool = make_req_to_token_pool(self.device)
        out = build_plan_input_radix_sweep(
            radix_cache=cache,
            swa_window_size=0,
            full_to_swa_index_mapping=None,
        )
        n = int(out.extra_verify_num_valid.item())
        self.assertEqual(n, 3)
        self.assertEqual(set(out.extra_verify_slot_indices[:n].tolist()), {42, 43, 44})

    def test_truly_free_slot_not_swept(self):
        empty_cache = make_radix_cache([[]], device=self.device)
        empty_cache.req_to_token_pool = make_req_to_token_pool(self.device)
        out = build_plan_input_radix_sweep(
            radix_cache=empty_cache,
            swa_window_size=0,
            full_to_swa_index_mapping=None,
        )
        self.assertEqual(int(out.extra_verify_num_valid.item()), 0)


if __name__ == "__main__":
    unittest.main()
