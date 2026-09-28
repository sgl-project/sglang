import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.graph_variants import (
    DSV41_CANDIDATE_FILTERED,
    create_dsv41_candidate_graph_variants,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCandidateGraphVariants(unittest.TestCase):
    def make_variants(self, sm=9, verify=False, width=6, **overrides):
        config = SimpleNamespace(
            model_type="deepseek_v41",
            candidate_source_layer_id=20,
            candidate_topk_blocks=2048,
            candidate_block_size=8,
            compress_ratios=[1, 2, 128],
            index_topk=512,
        )
        runner = SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=config),
            device="cuda",
            gpu_id=0,
            spec_algorithm=SimpleNamespace(is_dspark=lambda: True),
            is_draft_worker=False,
        )
        for key, value in overrides.items():
            setattr(runner, key, value)
        with (
            patch("torch.cuda.get_device_capability", return_value=(sm, 0)),
            patch("sglang.srt.utils.is_hip", return_value=False),
        ):
            return create_dsv41_candidate_graph_variants(
                runner,
                ForwardMode.TARGET_VERIFY if verify else ForwardMode.DECODE,
                width if verify else 0,
            )

    def select(self, variants, lengths, upper_bound=None):
        return variants.select(
            SimpleNamespace(
                seq_lens_cpu=None if lengths is None else torch.tensor(lengths),
                spec_info=SimpleNamespace(
                    candidate_max_seq_len_upper_bound=upper_bound
                ),
            )
        )

    def test_hopper_and_blackwell_share_decode_limits(self):
        for sm in (9, 10):
            variants = self.make_variants(sm=sm)
            self.assertEqual(
                variants.graph_limits,
                (
                    ("candidate_all", 512),
                    ("candidate_c2_all", 1024),
                    ("candidate_unfiltered", 16384),
                ),
            )
            self.assertEqual(variants.capture_labels[-1], DSV41_CANDIDATE_FILTERED)
            for length, label in (
                (0, "candidate_all"),
                (512, "candidate_all"),
                (513, "candidate_c2_all"),
                (1024, "candidate_c2_all"),
                (1025, "candidate_unfiltered"),
                (16384, "candidate_unfiltered"),
                (16385, DSV41_CANDIDATE_FILTERED),
            ):
                with self.subTest(sm=sm, length=length):
                    self.assertEqual(self.select(variants, [length]), label)
            self.assertEqual(
                self.select(variants, [300, 900, 20000]), DSV41_CANDIDATE_FILTERED
            )
            self.assertEqual(self.select(variants, []), DSV41_CANDIDATE_FILTERED)
            self.assertEqual(self.select(variants, None), DSV41_CANDIDATE_FILTERED)

    def test_verify_retains_causal_scoring_and_reserves_full_width(self):
        variants = self.make_variants(verify=True)
        self.assertEqual(
            variants.capture_labels, ("candidate_unfiltered", DSV41_CANDIDATE_FILTERED)
        )
        self.assertEqual(variants.verify_extra_tokens, 6)
        for lengths, upper_bound, expected in (
            ([1], None, "candidate_unfiltered"),
            ([16378], None, "candidate_unfiltered"),
            ([16379], None, DSV41_CANDIDATE_FILTERED),
            (None, 16378, "candidate_unfiltered"),
            (None, 16379, DSV41_CANDIDATE_FILTERED),
            (None, None, DSV41_CANDIDATE_FILTERED),
        ):
            self.assertEqual(self.select(variants, lengths, upper_bound), expected)

    def test_unsupported_architecture_and_verify_modes_stay_disabled(self):
        self.assertIsNone(self.make_variants(sm=8))
        self.assertIsNone(self.make_variants(device="cpu"))
        self.assertIsNone(self.make_variants(verify=True, is_draft_worker=True))
        self.assertIsNone(self.make_variants(verify=True, width=0))
        self.assertIsNone(
            self.make_variants(
                verify=True, spec_algorithm=SimpleNamespace(is_dspark=lambda: False)
            )
        )


if __name__ == "__main__":
    unittest.main()
