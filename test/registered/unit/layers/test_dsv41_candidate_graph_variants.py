"""DeepSeek-V4.1 candidate graph variants are admitted on ROCm and bound every HIP verify position."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.graph_variants import (
    create_dsv41_candidate_graph_variants,
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


class TestPrefillGraphCandidateWindow(CustomTestCase):
    def test_wide_prefill_graph_keeps_low_ratio_sources_eager(self):
        """The captured ratio-1/2 indexer is exact only inside the candidate window
        (2048 blocks x 8 on V4.1-Flash); a wider prefill graph must not capture it."""
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            _low_ratio_graph_fits_candidates,
        )

        def fits(low_ratios, max_seq_len, window=2048 * 8):
            return _low_ratio_graph_fits_candidates(
                low_ratios=low_ratios, candidate_window=window, max_seq_len=max_seq_len
            )

        self.assertTrue(fits((1, 2), 16384))
        self.assertFalse(fits((1, 2), 16384 + 256))
        # The smallest ratio sets the bound.
        self.assertTrue(fits((2,), 32768))
        self.assertFalse(fits((1, 2), None))
        # Without candidate blocks the captured top-k is exact at any width.
        self.assertTrue(fits((1, 2), 262144, window=0))
        self.assertTrue(fits((), None))


if __name__ == "__main__":
    unittest.main()
