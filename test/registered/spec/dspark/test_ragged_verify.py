import os
import unittest
from unittest import mock

import torch

from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    build_ragged_target_verify_geometry,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_DEVICE = torch.device("cpu")
_GRID = [8, 16, 24, 32, 64]

# The backend capability checks (supports_ragged_verify_graph) live in
# test_ragged_verify_backend_capability.py: importing the backend modules
# pulls GPU-only wheels, which fail to import on the CPU runners.


class TestRaggedTargetVerifyGeometry(CustomTestCase):
    def test_mixed_verify_lens_geometry(self):
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 1, 3], device=_DEVICE, grid=_GRID
        )
        seq_lens = torch.tensor([10, 20, 30], dtype=torch.int32)
        geometry = build_ragged_target_verify_geometry(seq_lens=seq_lens, layout=layout)
        self.assertEqual(geometry.cache_seqlens_int32.tolist(), [18, 21, 33])
        self.assertEqual(geometry.cu_seqlens_q.tolist(), [0, 8, 9, 12])
        self.assertEqual(geometry.cu_seqlens_k.tolist(), [0, 18, 39, 72])
        self.assertEqual(geometry.max_seq_len_q, 8)

    def test_geometry_dtypes_are_int32(self):
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 1, 3], device=_DEVICE, grid=_GRID
        )
        seq_lens = torch.tensor([10, 20, 30], dtype=torch.int64)
        geometry = build_ragged_target_verify_geometry(seq_lens=seq_lens, layout=layout)
        self.assertEqual(geometry.cache_seqlens_int32.dtype, torch.int32)
        self.assertEqual(geometry.cu_seqlens_q.dtype, torch.int32)
        self.assertEqual(geometry.cu_seqlens_k.dtype, torch.int32)


class TestPaddedRaggedVerifyGeometry(CustomTestCase):
    def test_padded_layout_grows_bs_and_fills_bucket(self):
        raw = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 1, 3],
            device=_DEVICE,
            grid=[8, 16, 32, 64],
            graph_num_tokens_floor=24,
        )
        self.assertEqual(raw.graph_num_tokens, 32)
        padded = raw.padded_to_bucket(padded_bs=4)
        self.assertEqual(padded.bs, 4)
        self.assertEqual(padded.verify_lens.tolist(), [8, 1, 3, 20])
        self.assertEqual(padded.qo_indptr_device.tolist(), [0, 8, 9, 12, 32])
        seq_lens = torch.tensor([10, 20, 30, 1], dtype=torch.int32)
        geometry = build_ragged_target_verify_geometry(seq_lens=seq_lens, layout=padded)
        self.assertEqual(geometry.cu_seqlens_q.tolist(), [0, 8, 9, 12, 32])
        self.assertEqual(geometry.cache_seqlens_int32.tolist(), [18, 21, 33, 21])
        self.assertEqual(int(geometry.cu_seqlens_k[-1]), 18 + 21 + 33 + 21)

    def test_padded_layout_decoupled_slots_spread_slack(self):
        raw = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 1, 3],
            device=_DEVICE,
            grid=[8, 16, 32, 64],
            graph_num_tokens_floor=24,
        )
        padded = raw.padded_to_bucket(padded_bs=6)
        self.assertEqual(padded.bs, 6)
        self.assertEqual(padded.verify_lens.tolist(), [8, 1, 3, 7, 7, 6])
        self.assertEqual(int(padded.qo_indptr_device[-1]), 32)

    def test_padded_layout_budget_tier_below_uniform(self):
        raw = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 1, 3],
            device=_DEVICE,
            grid=[8, 16, 32, 64],
        )
        self.assertEqual(raw.graph_num_tokens, 16)
        padded = raw.padded_to_bucket(padded_bs=3)
        self.assertEqual(padded.verify_lens.tolist(), [8, 1, 7])
        self.assertEqual(int(padded.qo_indptr_device[-1]), 16)

    def test_padded_layout_zero_len_pad_rows(self):
        raw = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 8],
            device=_DEVICE,
            grid=[8, 16, 32, 64],
        )
        self.assertEqual(raw.graph_num_tokens, 16)
        padded = raw.padded_to_bucket(padded_bs=8)
        self.assertEqual(padded.verify_lens.tolist(), [8, 8, 0, 0, 0, 0, 0, 0])
        self.assertEqual(int(padded.qo_indptr_device[-1]), 16)


class TestCaptureVerifyLens(CustomTestCase):
    def test_small_tier_one_token_rows(self):
        from sglang.srt.speculative.ragged_verify import build_capture_verify_lens

        lens = build_capture_verify_lens(num_tokens=8, num_slots=8, num_draft_tokens=8)
        self.assertEqual(lens, [1] * 8)

    def test_large_tier_spreads_within_window(self):
        from sglang.srt.speculative.ragged_verify import build_capture_verify_lens

        lens = build_capture_verify_lens(
            num_tokens=1024, num_slots=128, num_draft_tokens=8
        )
        self.assertEqual(sum(lens), 1024)
        self.assertEqual(lens, [8] * 128)

    def test_uneven_tier_rows_stay_legal(self):
        from sglang.srt.speculative.ragged_verify import build_capture_verify_lens

        lens = build_capture_verify_lens(num_tokens=24, num_slots=5, num_draft_tokens=8)
        self.assertEqual(sum(lens), 24)
        self.assertTrue(all(1 <= v <= 8 for v in lens))

    def test_rejects_overpacked_tier(self):
        from sglang.srt.speculative.ragged_verify import build_capture_verify_lens

        with self.assertRaises(ValueError):
            build_capture_verify_lens(num_tokens=64, num_slots=4, num_draft_tokens=8)
        with self.assertRaises(ValueError):
            build_capture_verify_lens(num_tokens=4, num_slots=8, num_draft_tokens=8)


class TestForcedUniformCapture(CustomTestCase):
    WIDTH = 6
    TIERS = [6, 12, 24, 36]

    def _runner(self):
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.ragged_verify_mode = True
        runner.max_bs = 8
        runner.captured_req_width = self.WIDTH
        runner.capture_num_tokens = self.TIERS
        runner.device = _DEVICE
        runner._captured_ragged_layouts = {}
        return runner

    def test_capture_carries_full_width_layout(self):
        from sglang.srt.speculative.ragged_verify import (
            compute_target_verify_graph_key,
        )

        width, tiers = self.WIDTH, self.TIERS
        runner = self._runner()
        env = {
            "SGLANG_RAGGED_VERIFY_MODE": "compact",
            "SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE": "1",
        }
        with mock.patch.dict(os.environ, env):
            # Each tier captures tier // width full-width requests. The capture
            # carries their layout, so the attention backend records the packed
            # (ragged) geometry that replay refreshes, keyed by token count.
            captured = {}
            for tier in tiers:
                layout = runner._capture_ragged_verify_layout(tier)
                self.assertIsNotNone(layout)
                self.assertEqual(layout.verify_lens.tolist(), [width] * (tier // width))
                self.assertIs(runner._captured_ragged_layouts[tier], layout)
                key = compute_target_verify_graph_key(
                    bs=tier // width, num_draft_tokens=width, ragged_layout=layout
                )
                captured[key[0]] = tier
            for tier in tiers:
                # Replay: one live request padded to the tier's capture slots.
                live = RaggedVerifyLayout.from_verify_lens(
                    verify_lens_cpu=[1],
                    device=_DEVICE,
                    grid=tiers,
                    graph_num_tokens_floor=tier,
                ).padded_to_bucket(padded_bs=tier // width)
                key = compute_target_verify_graph_key(
                    bs=tier // width, num_draft_tokens=width, ragged_layout=live
                )
                self.assertEqual(captured.get(key[0]), tier)

            # Two requests of unequal length in the 12-token tier: replay stages
            # their boundaries into the captured layout, so request 1 starts at
            # token 3 instead of the captured 6.
            live = RaggedVerifyLayout.from_verify_lens_device(
                verify_lens=torch.tensor([3, 4], dtype=torch.int32),
                graph_num_tokens=12,
            )
            expected = live.padded_to_bucket(padded_bs=2, cap=width)
            runner._stage_ragged_verify_layout(live, 12)
            layout = runner._captured_ragged_layouts[12]
            torch.testing.assert_close(layout.verify_lens, expected.verify_lens)
            torch.testing.assert_close(
                layout.qo_indptr_device, expected.qo_indptr_device
            )
            self.assertEqual(int(layout.qo_indptr_device[1]), 3)


if __name__ == "__main__":
    unittest.main()
