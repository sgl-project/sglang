"""Exact BF16 rounding and graph replay for DiffusionGemma's layer scalar."""

import unittest

import torch

from sglang.kernels.ops.layernorm.gemma4_fused_ops import gemma_residual_scalar
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestGemmaResidualScalar(CustomTestCase):
    def test_rounding_and_changed_graph_inputs(self):
        for shape in [(33, 2817), (256, 2816), (256, 4096)]:
            with self.subTest(shape=shape):
                x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                residual = torch.randn_like(x)
                scalar = torch.tensor([0.8125], device="cuda", dtype=torch.bfloat16)
                gemma_residual_scalar(x, residual, scalar)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = gemma_residual_scalar(x, residual, scalar)
                for value in [1.0, -2.0, 0.0, 0.001]:
                    scalar.fill_(value)
                    x.neg_()
                    residual.mul_(-0.5)
                    graph.replay()
                    torch.testing.assert_close(
                        actual, (residual + x) * scalar, rtol=0, atol=0
                    )

    def test_special_values_and_empty(self):
        x = torch.tensor(
            [
                0.0,
                -0.0,
                1e-40,
                -1e-40,
                3.4e38,
                -3.4e38,
                float("inf"),
                -float("inf"),
                float("nan"),
                1.00390625,
            ],
            device="cuda",
            dtype=torch.bfloat16,
        ).repeat(33, 1)
        for value in [0.0, 1.0, -1.0, 0.8125]:
            scalar = torch.tensor([value], device="cuda", dtype=torch.bfloat16)
            for residual in [torch.ones_like(x), -x]:
                torch.testing.assert_close(
                    gemma_residual_scalar(x, residual, scalar),
                    (residual + x) * scalar,
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )
        empty = x[:0]
        self.assertEqual(gemma_residual_scalar(empty, empty, scalar).shape, empty.shape)


if __name__ == "__main__":
    unittest.main()
