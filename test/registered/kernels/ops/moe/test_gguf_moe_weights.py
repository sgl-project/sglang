"""GGUF route weighting preserves per-slot rounding and skips inactive data."""

import unittest

import torch

from sglang.kernels.ops.moe.gguf_moe_weights import apply_gguf_moe_weights
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=4, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestGgufMoeWeights(CustomTestCase):
    def test_rounding_and_poisoned_routes(self):
        """Weights round per slot before summation; dropped NaNs contribute zero."""
        generator = torch.Generator(device="cuda").manual_seed(836)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for strided in (False, True):
                with self.subTest(dtype=dtype, strided=strided):
                    outputs = torch.randn(
                        12, 257, generator=generator, device="cuda", dtype=dtype
                    )
                    weights = torch.randn(4, 3, generator=generator, device="cuda")
                    ids = torch.tensor(
                        [[0, 1, 2], [0, -1, 2], [-1, -1, -1], [1, 0, -1]],
                        device="cuda",
                        dtype=torch.int64 if strided else torch.int32,
                    )
                    if strided:
                        weights = weights.t().contiguous().t()
                        ids = ids.t().contiguous().t()
                    inactive = ids < 0
                    outputs[inactive.flatten()] = float("nan")
                    weights[inactive] = float("nan")
                    expected = outputs.clone()
                    expected[inactive.flatten()] = 0
                    expected.view(4, 3, 257).mul_(
                        weights.masked_fill(inactive, 0).unsqueeze(-1)
                    )
                    apply_gguf_moe_weights(outputs, weights, ids)
                    torch.testing.assert_close(outputs, expected, rtol=0, atol=0)
                    self.assertTrue(bool(torch.isfinite(outputs).all()))
                    ids.fill_(-1)
                    weights.fill_(float("nan"))
                    outputs.fill_(float("nan"))
                    apply_gguf_moe_weights(outputs, weights, ids)
                    self.assertEqual(int(torch.count_nonzero(outputs)), 0)


if __name__ == "__main__":
    unittest.main()
