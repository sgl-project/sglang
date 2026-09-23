"""DeepSeek-V4.1 candidate graph selection must bound every speculative query position and be admitted on ROCm."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.graph_variants import (
    Dsv41CandidateGraphVariants,
    create_dsv41_candidate_graph_variants,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.dspark_components.dspark_verify import (
    candidate_request_length_bound,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestVerifyCandidateGraph(CustomTestCase):
    def make_policy(self, width=6):
        return Dsv41CandidateGraphVariants(
            graph_limits=(("candidate_unfiltered", 16384),),
            capture_labels=("candidate_unfiltered", "candidate_filtered"),
            verify_extra_tokens=width,
        )

    def test_longest_query_controls_selection(self):
        policy = self.make_policy()
        for lengths, expected in (
            ([4096] * 64, "candidate_unfiltered"),
            ([4096, 16378], "candidate_unfiltered"),
            ([4096, 16379], "candidate_filtered"),
            ([16384], "candidate_filtered"),
            ([1000000], "candidate_filtered"),
        ):
            with self.subTest(lengths=lengths):
                batch = SimpleNamespace(seq_lens_cpu=torch.tensor(lengths))
                self.assertEqual(policy.select(batch), expected)
        # Plain decode (no verify width) keeps the existing <= boundary.
        batch = SimpleNamespace(seq_lens_cpu=torch.tensor([16384]))
        self.assertEqual(
            self.make_policy(width=0).select(batch), "candidate_unfiltered"
        )

    def test_missing_or_non_cpu_lengths_use_full_graph(self):
        policy = self.make_policy()
        for lengths in (
            None,
            torch.empty(0, dtype=torch.int64),
            torch.empty(1, device="meta"),
        ):
            with self.subTest(lengths=lengths):
                self.assertEqual(
                    policy.select(SimpleNamespace(seq_lens_cpu=lengths)),
                    "candidate_filtered",
                )

    def test_gpu_only_lengths_use_request_budget(self):
        policy = self.make_policy()
        for bound, expected in (
            (5120, "candidate_unfiltered"),
            (16378, "candidate_unfiltered"),
            (16379, "candidate_filtered"),
            (None, "candidate_filtered"),
        ):
            with self.subTest(bound=bound):
                batch = SimpleNamespace(
                    seq_lens_cpu=None,
                    spec_info=SimpleNamespace(candidate_max_seq_len_upper_bound=bound),
                )
                self.assertEqual(policy.select(batch), expected)

    def test_hip_verify_shortcut_bounds_all_draft_positions(self):
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
        # ROCm is admitted without a Blackwell capability; a pre-Blackwell CUDA
        # device is not.
        with (
            patch("torch.cuda.get_device_capability", return_value=(9, 4)),
            patch("sglang.srt.utils.is_hip", return_value=True),
        ):
            self.assertIsNotNone(
                create_dsv41_candidate_graph_variants(
                    runner, ForwardMode.TARGET_VERIFY, 6
                )
            )
        with (
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            patch("sglang.srt.utils.is_hip", return_value=False),
        ):
            self.assertIsNone(
                create_dsv41_candidate_graph_variants(
                    runner, ForwardMode.TARGET_VERIFY, 6
                )
            )
        with (
            patch("torch.cuda.get_device_capability", return_value=(10, 0)),
            patch("sglang.srt.utils.is_hip", return_value=False),
        ):
            verify = create_dsv41_candidate_graph_variants(
                runner, ForwardMode.TARGET_VERIFY, 6
            )
            self.assertEqual(
                verify.capture_labels, ("candidate_unfiltered", "candidate_filtered")
            )
            self.assertEqual(verify.verify_extra_tokens, 6)
            decode = create_dsv41_candidate_graph_variants(
                runner, ForwardMode.DECODE, 1
            )
            self.assertEqual(
                decode.capture_labels,
                (
                    "candidate_all",
                    "candidate_c2_all",
                    "candidate_unfiltered",
                    "candidate_filtered",
                ),
            )
            self.assertEqual(decode.verify_extra_tokens, 0)
            # Never enable the shortcut for a draft worker or another algorithm.
            runner.is_draft_worker = True
            self.assertIsNone(
                create_dsv41_candidate_graph_variants(
                    runner, ForwardMode.TARGET_VERIFY, 6
                )
            )
            runner.is_draft_worker = False
            runner.spec_algorithm.is_dspark = lambda: False
            self.assertIsNone(
                create_dsv41_candidate_graph_variants(
                    runner, ForwardMode.TARGET_VERIFY, 6
                )
            )
            runner.spec_algorithm.is_dspark = lambda: True
            self.assertIsNone(
                create_dsv41_candidate_graph_variants(
                    runner, ForwardMode.TARGET_VERIFY, 0
                )
            )

    def test_decode_ignores_verify_request_budget(self):
        policy = self.make_policy(width=0)
        batch = SimpleNamespace(
            seq_lens_cpu=None,
            spec_info=SimpleNamespace(candidate_max_seq_len_upper_bound=5120),
        )
        self.assertEqual(policy.select(batch), "candidate_filtered")

    def test_host_lengths_take_precedence_over_request_budget(self):
        batch = SimpleNamespace(
            seq_lens_cpu=torch.tensor([16379]),
            spec_info=SimpleNamespace(candidate_max_seq_len_upper_bound=5120),
        )
        self.assertEqual(self.make_policy().select(batch), "candidate_filtered")

    def test_request_bound_uses_full_output_budget(self):
        req = SimpleNamespace(
            origin_input_ids=[0] * 4096,
            sampling_params=SimpleNamespace(max_new_tokens=1024),
        )
        self.assertEqual(candidate_request_length_bound([req] * 64), 5120)
        self.assertEqual(candidate_request_length_bound([req] * 64, 6), 5126)
        req.output_ids = []  # Accepted tokens can still be in flight.
        self.assertEqual(candidate_request_length_bound([req]), 5120)
        for attr, value in (
            ("to_finish", object()),
            ("input_embeds", object()),
            ("multimodal_inputs", object()),
        ):
            with self.subTest(attr=attr):
                setattr(req, attr, value)
                self.assertIsNone(candidate_request_length_bound([req]))
                setattr(req, attr, None)
        for budget in (None, -1, 1.5):
            with self.subTest(budget=budget):
                req.sampling_params.max_new_tokens = budget
                self.assertIsNone(candidate_request_length_bound([req]))
        self.assertIsNone(candidate_request_length_bound([]))


if __name__ == "__main__":
    unittest.main()
