"""Verify graph selection must bound all speculative query positions."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    candidate_request_length_bound,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestVerifyCandidateGraph(CustomTestCase):
    def make_runner(self, width=6):
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.candidate_filter_span = 16384
        runner.candidate_graph_limits = [("candidate_unfiltered", 16384)]
        runner.candidate_verify_extra_tokens = width
        return runner

    def test_longest_query_controls_selection(self):
        runner = self.make_runner()
        for lengths, expected in (
            ([4096] * 64, "candidate_unfiltered"),
            ([4096, 16378], "candidate_unfiltered"),
            ([4096, 16379], "candidate_filtered"),
            ([16384], "candidate_filtered"),
            ([1000000], "candidate_filtered"),
        ):
            with self.subTest(lengths=lengths):
                batch = SimpleNamespace(seq_lens_cpu=torch.tensor(lengths))
                self.assertEqual(runner._resolve_dsa_variant(batch), expected)

    def test_missing_or_non_cpu_lengths_use_full_graph(self):
        runner = self.make_runner()
        for lengths in (
            None,
            torch.empty(0, dtype=torch.int64),
            torch.empty(1, device="meta"),
        ):
            with self.subTest(lengths=lengths):
                self.assertEqual(
                    runner._resolve_dsa_variant(SimpleNamespace(seq_lens_cpu=lengths)),
                    "candidate_filtered",
                )

    def test_plain_decode_keeps_existing_boundary(self):
        runner = self.make_runner(width=0)
        batch = SimpleNamespace(seq_lens_cpu=torch.tensor([16384]))
        self.assertEqual(runner._resolve_dsa_variant(batch), "candidate_unfiltered")

    def test_gpu_only_lengths_use_request_budget(self):
        runner = self.make_runner()
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
                self.assertEqual(runner._resolve_dsa_variant(batch), expected)

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
        req.sampling_params.max_new_tokens = None
        self.assertIsNone(candidate_request_length_bound([req]))
        self.assertIsNone(candidate_request_length_bound([]))


if __name__ == "__main__":
    unittest.main()
