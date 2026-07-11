import os
import types
import unittest

# This suite runs on CPU-only CI; pin the layout kernels to their torch
# implementations (the triton defaults require CUDA tensors). Must be set
# before the kernel modules are imported.
os.environ.setdefault("SGLANG_DSPARK_KERNEL_QO_INDPTR", "torch")
os.environ.setdefault("SGLANG_DSPARK_KERNEL_PADDED_TO_BUCKET", "torch")

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.trtllm_mha_backend import (
    TRTLLMHAAttnBackend,
    _resolve_ragged_verify_layout,
    build_ragged_target_verify_geometry,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_DEVICE = torch.device("cpu")
_GRID = [8, 16, 24, 32, 64]


class TestRaggedVerifyGraphCapability(CustomTestCase):
    def test_base_backend_defaults_false(self):
        self.assertFalse(AttentionBackend.supports_ragged_verify_graph)

    def test_trtllm_mha_supports_ragged_verify_graph(self):
        self.assertTrue(TRTLLMHAAttnBackend.supports_ragged_verify_graph)

    def test_dsv4_supports_ragged_verify_graph(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        self.assertTrue(DeepseekV4AttnBackend.supports_ragged_verify_graph)


class TestResolveRaggedVerifyLayout(CustomTestCase):
    def test_none_without_spec_info(self):
        fb = types.SimpleNamespace(spec_info=None)
        self.assertIsNone(_resolve_ragged_verify_layout(fb))

    def test_none_without_layout_attr(self):
        fb = types.SimpleNamespace(spec_info=types.SimpleNamespace())
        self.assertIsNone(_resolve_ragged_verify_layout(fb))

    def test_returns_attached_layout(self):
        layout = RaggedVerifyLayout.uniform(
            bs=2, num_draft_tokens=8, device=_DEVICE, grid=_GRID
        )
        fb = types.SimpleNamespace(
            spec_info=types.SimpleNamespace(ragged_verify_layout=layout)
        )
        self.assertIs(_resolve_ragged_verify_layout(fb), layout)


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


class TestNegativeSeamGeometry(CustomTestCase):
    def test_uniform_layout_geometry_matches_legacy_arange(self):
        seq_lens = torch.tensor([5, 7, 9], dtype=torch.int32)
        uniform = RaggedVerifyLayout.uniform(
            bs=3, num_draft_tokens=8, device=_DEVICE, grid=_GRID
        )
        geometry = build_ragged_target_verify_geometry(
            seq_lens=seq_lens, layout=uniform
        )
        self.assertEqual(geometry.cu_seqlens_q.tolist(), [0, 8, 16, 24])
        self.assertEqual(geometry.max_seq_len_q, 8)


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


class _FakeRaggedRunner(types.SimpleNamespace):
    pass


def _fake_model_runner(capture_num_tokens, max_bs):
    runner = _FakeRaggedRunner(
        ragged_verify_mode=True,
        capture_num_tokens=capture_num_tokens,
        max_bs=max_bs,
    )
    return types.SimpleNamespace(decode_cuda_graph_runner=runner)


class TestBudgetTierSelection(CustomTestCase):
    def test_floor_uses_tier_hint_capped_at_uniform_window(self):
        from sglang.srt.speculative.dspark_components.dspark_planner import (
            verify_layout_graph_num_tokens_floor,
        )
        from sglang.srt.speculative.ragged_verify import RaggedVerifyMode

        model_runner = _fake_model_runner([8, 16, 1024], max_bs=128)
        floor = verify_layout_graph_num_tokens_floor(
            num_reqs=100,
            ragged_verify_mode=RaggedVerifyMode.COMPACT,
            verify_num_draft_tokens=8,
            model_runner=model_runner,
            tier_num_tokens=150,
        )
        self.assertEqual(floor, 150)
        capped = verify_layout_graph_num_tokens_floor(
            num_reqs=10,
            ragged_verify_mode=RaggedVerifyMode.COMPACT,
            verify_num_draft_tokens=8,
            model_runner=model_runner,
            tier_num_tokens=150,
        )
        self.assertEqual(capped, 80)
        pinned = verify_layout_graph_num_tokens_floor(
            num_reqs=100,
            ragged_verify_mode=RaggedVerifyMode.COMPACT,
            verify_num_draft_tokens=8,
            model_runner=model_runner,
        )
        self.assertEqual(pinned, 800)

    def test_exceeds_gate_checks_slots_and_tier(self):
        from sglang.srt.speculative.dspark_components.dspark_planner import (
            ragged_layout_exceeds_captured_grid,
        )

        model_runner = _fake_model_runner([8, 16, 1024], max_bs=128)
        self.assertTrue(
            ragged_layout_exceeds_captured_grid(
                num_reqs=129,
                verify_num_draft_tokens=8,
                model_runner=model_runner,
                tier_tokens_hint=200,
            )
        )
        self.assertFalse(
            ragged_layout_exceeds_captured_grid(
                num_reqs=128,
                verify_num_draft_tokens=8,
                model_runner=model_runner,
                tier_tokens_hint=512,
            )
        )
        self.assertFalse(
            ragged_layout_exceeds_captured_grid(
                num_reqs=128,
                verify_num_draft_tokens=8,
                model_runner=model_runner,
            )
        )
        self.assertTrue(
            ragged_layout_exceeds_captured_grid(
                num_reqs=128,
                verify_num_draft_tokens=9,
                model_runner=model_runner,
            )
        )


if __name__ == "__main__":
    unittest.main()
