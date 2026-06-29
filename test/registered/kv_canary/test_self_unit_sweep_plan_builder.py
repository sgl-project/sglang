from __future__ import annotations

import unittest

import torch

from sglang.srt.kv_canary.sweep_plan_builder import build_verify_plan_radix_sweep
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kv_canary.fixtures import (
    DEFAULT_DEVICE,
    make_radix_cache,
    make_req_to_token_pool,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=30, suite="extra-a-test-1-gpu-small-amd")


class TestSelfUnitSweepPlanBuilder(CustomTestCase):
    def setUp(self) -> None:
        self.device = DEFAULT_DEVICE

    def test_build_verify_plan_radix_sweep(self) -> None:
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
        self.assertEqual(out.verify_expected_positions.dtype, torch.int64)
        self.assertEqual(out.verify_prev_slot_indices.dtype, torch.int64)
        self.assertEqual(out.verify_slot_indices[:3].tolist(), [100, 101, 102])
        self.assertEqual(out.verify_expected_positions[:3].tolist(), [0, 1, 2])
        self.assertEqual(out.verify_prev_slot_indices[:3].tolist(), [-1, 100, 101])

    def test_radix_held_slot_still_swept(self) -> None:
        """Verify held radix slots are still included in sweep plans."""
        cache = make_radix_cache([[], [42, 43, 44]], device=self.device)
        cache.req_to_token_pool = make_req_to_token_pool(self.device)
        out = build_verify_plan_radix_sweep(
            radix_cache=cache,
            swa_window_size=0,
            full_to_swa_index_mapping=None,
        )
        num_valid = int(out.verify_num_valid.item())
        self.assertEqual(num_valid, 3)
        self.assertEqual(
            set(out.verify_slot_indices[:num_valid].tolist()), {42, 43, 44}
        )

    def test_truly_free_slot_not_swept(self) -> None:
        """Verify free radix slots are excluded from sweep plans."""
        empty_cache = make_radix_cache([[]], device=self.device)
        empty_cache.req_to_token_pool = make_req_to_token_pool(self.device)
        out = build_verify_plan_radix_sweep(
            radix_cache=empty_cache,
            swa_window_size=0,
            full_to_swa_index_mapping=None,
        )
        self.assertEqual(int(out.verify_num_valid.item()), 0)

    def test_swa_translate_preserves_evicted_as_padding_sentinel(self) -> None:
        """Evicted (LUT=0) slots stay in the plan as the padding sentinel; the kernel does the skipping."""
        cache = make_radix_cache([[], [100, 101, 102]], device=self.device)
        cache.req_to_token_pool = make_req_to_token_pool(self.device)

        lut = torch.zeros(200, dtype=torch.int64, device=self.device)
        lut[100] = 500
        lut[101] = 0
        lut[102] = 502

        out = build_verify_plan_radix_sweep(
            radix_cache=cache,
            swa_window_size=128,
            full_to_swa_index_mapping=lut,
        )
        num_valid = int(out.verify_num_valid.item())
        self.assertEqual(num_valid, 3)
        self.assertEqual(out.verify_slot_indices[:num_valid].tolist(), [500, 0, 502])
        self.assertEqual(
            out.verify_prev_slot_indices[:num_valid].tolist(), [-1, 500, 0]
        )
        self.assertEqual(out.verify_expected_positions[:num_valid].tolist(), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
