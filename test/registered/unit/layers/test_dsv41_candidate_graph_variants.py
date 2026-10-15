"""DeepSeek-V4.1 candidate graph variants are admitted on ROCm and bound every HIP verify position."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.graph_variants import (
    SPLIT_KV_LONG,
    SPLIT_KV_MIN_LENGTH,
    SPLIT_KV_STOCK,
    SplitKVGraphVariants,
    create_dsv41_candidate_graph_variants,
    create_split_kv_graph_variants,
    use_split_kv_attention,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestVerifyCandidateGraph(CustomTestCase):
    def test_hip_verify_shortcut_bounds_all_draft_positions(self):
        """A verify row whose last draft position passes the candidate span must not take the shortcut."""
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            low_ratio_decode_rows_fit_candidate_span,
        )

        backend = SimpleNamespace(
            low_ratio_candidate_span=16384, speculative_num_draft_tokens=6
        )
        for length, expected in ((16378, True), (16379, False)):
            batch = SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY,
                seq_lens_cpu=torch.tensor([length]),
            )
            self.assertEqual(
                low_ratio_decode_rows_fit_candidate_span(backend, batch), expected
            )
        batch.seq_lens_cpu = None
        self.assertFalse(low_ratio_decode_rows_fit_candidate_span(backend, batch))

    def test_factory_keeps_causal_indexer_for_verify(self):
        """gfx950 must be admitted without a Blackwell capability; other ROCm GPUs and
        pre-Blackwell CUDA stay rejected."""
        config = SimpleNamespace(
            model_type="deepseek_v41",
            candidate_source_layer_id=1,
            candidate_topk_blocks=128,
            candidate_block_size=128,
            compress_ratios=[1, 2],
            index_topk=2048,
        )
        runner = SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=config),
            device="cuda",
            gpu_id=0,
            spec_algorithm=SimpleNamespace(is_dspark=lambda: True),
            is_draft_worker=False,
        )
        with (
            patch("torch.cuda.get_device_capability", return_value=(9, 5)),
            patch("sglang.srt.utils.is_hip", return_value=True),
            patch("sglang.srt.utils.is_gfx95_supported", return_value=True),
        ):
            self.assertIsNotNone(
                create_dsv41_candidate_graph_variants(
                    runner, ForwardMode.TARGET_VERIFY, 6
                )
            )
        with (
            patch("torch.cuda.get_device_capability", return_value=(9, 4)),
            patch("sglang.srt.utils.is_hip", return_value=True),
            patch("sglang.srt.utils.is_gfx95_supported", return_value=False),
        ):
            self.assertIsNone(
                create_dsv41_candidate_graph_variants(
                    runner, ForwardMode.TARGET_VERIFY, 6
                )
            )
        with (
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            patch("sglang.srt.utils.is_hip", return_value=False),
            patch("sglang.srt.utils.is_gfx95_supported", return_value=False),
        ):
            self.assertIsNone(
                create_dsv41_candidate_graph_variants(
                    runner, ForwardMode.TARGET_VERIFY, 6
                )
            )


class TestSplitKVGraphRouting(CustomTestCase):
    def test_draft_extend_captures_do_not_overwrite_each_other(self):
        """The draft-extend override must preserve both algorithms in its graph key."""
        from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
            EAGLEDraftExtendCudaGraphRunner,
        )

        make_key = EAGLEDraftExtendCudaGraphRunner._make_graph_key
        stock = make_key(None, 1, attention_variant=SPLIT_KV_STOCK)
        split = make_key(None, 1, attention_variant=SPLIT_KV_LONG)
        self.assertNotEqual(stock, split)
        self.assertEqual(stock.attention_variant, SPLIT_KV_STOCK)
        self.assertEqual(split.attention_variant, SPLIT_KV_LONG)
        self.assertEqual(make_key(None, 1).attention_variant, None)

    def test_route_tracks_host_lower_bound_without_reading_device_lengths(self):
        """A growing or replaced request must select its own graph, without a D2H read."""
        variants = SplitKVGraphVariants()
        batch = SimpleNamespace(
            batch_size=1,
            attention_kv_len_lower_bound=SPLIT_KV_MIN_LENGTH - 1,
            seq_lens_cpu=None,
            seq_lens=torch.empty(1, device="meta"),
        )
        self.assertEqual(variants.select(batch), SPLIT_KV_STOCK)
        batch.attention_kv_len_lower_bound += 1
        self.assertEqual(variants.select(batch), SPLIT_KV_LONG)
        batch.attention_kv_len_lower_bound = None
        self.assertEqual(variants.select(batch), SPLIT_KV_STOCK)
        batch.seq_lens_cpu = torch.tensor([SPLIT_KV_MIN_LENGTH])
        self.assertEqual(variants.select(batch), SPLIT_KV_LONG)
        batch.seq_lens_cpu = torch.empty(1, device="meta")
        self.assertEqual(variants.select(batch), SPLIT_KV_STOCK)
        batch.attention_kv_len_lower_bound = SPLIT_KV_MIN_LENGTH
        batch.batch_size = 2
        self.assertEqual(variants.select(batch), SPLIT_KV_STOCK)

    def test_capture_label_controls_the_recorded_algorithm(self):
        """Long graphs must record split attention even with short dummy capture inputs."""
        batch = SimpleNamespace(
            batch_size=1, attention_kv_len_lower_bound=0, seq_lens_cpu=None
        )
        capture_module = "sglang.srt.model_executor.runner_utils.capture_mode"
        with (
            patch(
                f"{capture_module}.get_capture_attention_variant",
                return_value=SPLIT_KV_LONG,
            ),
            patch(f"{capture_module}.get_is_capture_mode", return_value=True),
        ):
            self.assertTrue(use_split_kv_attention(batch))
        batch.attention_kv_len_lower_bound = SPLIT_KV_MIN_LENGTH
        for label in (SPLIT_KV_STOCK, None):
            with (
                patch(
                    f"{capture_module}.get_capture_attention_variant",
                    return_value=label,
                ),
                patch(f"{capture_module}.get_is_capture_mode", return_value=True),
            ):
                self.assertFalse(use_split_kv_attention(batch))

    def test_flag_off_and_single_token_keep_existing_graph_keys(self):
        """The optional graph pair must not affect the default or single-token path."""
        from sglang.srt.environ import envs

        with envs.SGLANG_Q35MF_ATTN_SPLIT_KV.override(False):
            self.assertIsNone(
                create_split_kv_graph_variants(ForwardMode.TARGET_VERIFY, 7)
            )
        with envs.SGLANG_Q35MF_ATTN_SPLIT_KV.override(True):
            self.assertIsNone(create_split_kv_graph_variants(ForwardMode.DECODE, 1))
            for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2):
                self.assertIsInstance(
                    create_split_kv_graph_variants(mode, 7), SplitKVGraphVariants
                )


if __name__ == "__main__":
    unittest.main()
