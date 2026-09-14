import unittest

import torch

from sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe import (
    _preprocess_humming_in_expert_chunks,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHummingChunkedPreprocess(unittest.TestCase):
    def test_preserves_expert_order_and_outputs(self):
        weight = torch.arange(30, dtype=torch.int32).reshape(5, 2, 3)
        scale = torch.arange(20, dtype=torch.int32).reshape(5, 2, 2)
        chunk_sizes = []

        def preprocess(weight_chunk, scale_chunk):
            chunk_sizes.append(weight_chunk.shape[0])
            residual = weight_chunk[:, 0, 0].to(torch.float32)
            return weight_chunk + 1, scale_chunk + 2, residual

        weight_out, scale_out, residual_out = _preprocess_humming_in_expert_chunks(
            weight,
            scale,
            preprocess,
            chunk_size=2,
        )

        self.assertEqual(chunk_sizes, [2, 2, 1])
        torch.testing.assert_close(weight_out, weight + 1)
        torch.testing.assert_close(scale_out, scale + 2)
        torch.testing.assert_close(residual_out, weight[:, 0, 0].to(torch.float32))

    def test_rejects_invalid_chunk_size(self):
        with self.assertRaisesRegex(ValueError, "chunk size must be positive"):
            _preprocess_humming_in_expert_chunks(
                torch.empty(1, 1, 1),
                torch.empty(1, 1, 1),
                lambda weight, scale: (weight, scale, torch.ones(1)),
                chunk_size=0,
            )

    def test_rejects_mismatched_expert_count(self):
        with self.assertRaisesRegex(ValueError, "expert counts must match"):
            _preprocess_humming_in_expert_chunks(
                torch.empty(2, 1, 1),
                torch.empty(1, 1, 1),
                lambda weight, scale: (weight, scale, torch.ones(1)),
                chunk_size=1,
            )


if __name__ == "__main__":
    unittest.main()
