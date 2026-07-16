import unittest

import torch

from sglang.srt.speculative.ragged_verify import (
    DSA_TARGET_VERIFY_POST_TOPK_GRAPH,
    DSA_TARGET_VERIFY_PRE_TOPK_GRAPH,
    RaggedVerifyLayout,
    build_ragged_target_verify_geometry,
    classify_dsa_target_verify_graph_regime,
    is_static_full_verify_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_DEVICE = torch.device("cpu")
_GRID = [8, 16, 24, 32, 64]

# The backend capability checks (supports_ragged_verify_graph) live in
# test_ragged_verify_backend_capability.py: importing the backend modules
# pulls GPU-only wheels, which fail to import on the CPU runners.


class TestRaggedTargetVerifyGeometry(unittest.TestCase):
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


class TestDsaTargetVerifyGraphRegime(unittest.TestCase):
    def test_pre_topk_window_uses_default_regime(self):
        regime = classify_dsa_target_verify_graph_regime(
            seq_lens_cpu=[1000, 1200],
            verify_lens_cpu=[8, 1],
            dsa_index_topk=2048,
        )
        self.assertEqual(regime, DSA_TARGET_VERIFY_PRE_TOPK_GRAPH)

    def test_post_topk_window_uses_post_regime(self):
        regime = classify_dsa_target_verify_graph_regime(
            seq_lens_cpu=[2048, 3000],
            verify_lens_cpu=[1, 8],
            dsa_index_topk=2048,
        )
        self.assertEqual(regime, DSA_TARGET_VERIFY_POST_TOPK_GRAPH)

    def test_boundary_token_is_post_topk(self):
        regime = classify_dsa_target_verify_graph_regime(
            seq_lens_cpu=[2047],
            verify_lens_cpu=[1],
            dsa_index_topk=2048,
        )
        self.assertEqual(regime, DSA_TARGET_VERIFY_POST_TOPK_GRAPH)

    def test_mixed_transition_window_stays_eager(self):
        regime = classify_dsa_target_verify_graph_regime(
            seq_lens_cpu=[2044],
            verify_lens_cpu=[8],
            dsa_index_topk=2048,
        )
        self.assertIsNone(regime)

    def test_zero_verify_padding_does_not_force_mixed(self):
        regime = classify_dsa_target_verify_graph_regime(
            seq_lens_cpu=[2050, 1],
            verify_lens_cpu=[4, 0],
            dsa_index_topk=2048,
        )
        self.assertEqual(regime, DSA_TARGET_VERIFY_POST_TOPK_GRAPH)


class TestPaddedRaggedVerifyGeometry(unittest.TestCase):
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


class TestStaticFullVerifyLayout(unittest.TestCase):
    def test_full_width_layout_is_static_even_when_graph_tier_rounds_up(self):
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 8, 8],
            device=_DEVICE,
            grid=[8, 16, 32, 64],
        )

        self.assertEqual(layout.graph_num_tokens, 32)
        self.assertTrue(
            is_static_full_verify_layout(layout, num_tokens_per_req=8)
        )

    def test_single_full_width_layout_is_static(self):
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8],
            device=_DEVICE,
            grid=[8, 16, 32, 64],
        )

        self.assertTrue(
            is_static_full_verify_layout(layout, num_tokens_per_req=8)
        )

    def test_non_full_width_layout_stays_ragged(self):
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[8, 7],
            device=_DEVICE,
            grid=[8, 16, 32, 64],
        )

        self.assertFalse(
            is_static_full_verify_layout(layout, num_tokens_per_req=8)
        )


class TestCaptureVerifyLens(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
