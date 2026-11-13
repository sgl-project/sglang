import unittest
from unittest import TestCase

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSV4DraftExtendCudaGraph(TestCase):
    def test_padding_causal_lengths_stay_positive(self):
        """A graph-padded request (seq_len=1) wider than one token must not
        expand into nonpositive causal lengths or negative positions."""
        backend = object.__new__(DeepseekV4AttnBackend)
        backend.cuda_int32_kwargs = {"dtype": torch.int32, "device": "cpu"}

        seq_lens_casual, req_pool_indices = backend.expand_extend_with_same_length(
            bs=2,
            qo_len=4,
            seq_lens=torch.tensor([7, 1], dtype=torch.int32),
            req_pool_indices=torch.tensor([3, 0], dtype=torch.int32),
        )

        self.assertEqual(seq_lens_casual.tolist(), [4, 5, 6, 7, 1, 1, 1, 1])
        self.assertEqual(req_pool_indices.tolist(), [3, 3, 3, 3, 0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
