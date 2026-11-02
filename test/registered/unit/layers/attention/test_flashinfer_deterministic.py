import unittest

from sglang.srt.layers.attention.flashinfer_deterministic import (
    build_prefill_pad_indices,
    deterministic_prefill_min_q,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestDeterministicPrefillMinQ(CustomTestCase):
    def test_head_dim_256_threshold(self):
        self.assertEqual(deterministic_prefill_min_q(24, 4, 256), 3)
        self.assertEqual(deterministic_prefill_min_q(16, 4, 256), 5)

    def test_head_dim_below_256_threshold(self):
        self.assertEqual(deterministic_prefill_min_q(32, 8, 128), 17)
        self.assertEqual(deterministic_prefill_min_q(4, 4, 64), 65)

    def test_head_dim_at_least_512_threshold(self):
        self.assertEqual(deterministic_prefill_min_q(8, 1, 512), 5)

    def test_group_larger_than_threshold_needs_no_padding(self):
        self.assertEqual(deterministic_prefill_min_q(32, 1, 256), 1)


class TestBuildPrefillPadIndices(CustomTestCase):
    def test_pads_front_of_each_request(self):
        gather, out = build_prefill_pad_indices([1, 3], 5, "cpu")
        self.assertEqual(gather.tolist(), [4, 4, 4, 4, 0, 4, 4, 1, 2, 3])
        self.assertEqual(out.tolist(), [4, 7, 8, 9])

    def test_no_padding_returns_none(self):
        gather, out = build_prefill_pad_indices([5, 7], 5, "cpu")
        self.assertIsNone(gather)
        self.assertIsNone(out)

    def test_zero_length_request_is_not_padded(self):
        gather, out = build_prefill_pad_indices([0, 2], 5, "cpu")
        self.assertEqual(gather.tolist(), [2, 2, 2, 0, 1])
        self.assertEqual(out.tolist(), [3, 4])


if __name__ == "__main__":
    unittest.main()
