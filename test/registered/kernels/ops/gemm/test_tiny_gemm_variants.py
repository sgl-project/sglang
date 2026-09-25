"""Representative parity coverage for the lightweight Kimi-K3 prerequisites."""

import unittest

import torch

from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")

NUM_EXPERTS = 896

TOPK = 16

NOPE_DIM = 512

ROPE_DIM = 64

MLA_DIM = NOPE_DIM + ROPE_DIM

MLA_PAGES = 256


class TestKimiK3PrerequisiteOps(CustomTestCase):
    def test_tiny_gemm_variants(self):
        """Both K3 gate-projection shapes, one per kernel variant: N=144 is the
        tiny dimension for the first, K=128 for the second."""
        torch.manual_seed(2)
        x = torch.randn(2, 7168, device="cuda", dtype=torch.bfloat16) / 8
        weight = torch.randn(144, 7168, device="cuda", dtype=torch.bfloat16) / 8
        actual = tiny_gemm_bf16(x, weight, out_dtype=torch.float32)
        torch.testing.assert_close(
            actual.double(), x.double() @ weight.double().t(), rtol=1e-3, atol=1e-3
        )

        x = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16) / 4
        weight = torch.randn(1536, 128, device="cuda", dtype=torch.bfloat16) / 4
        actual = tiny_gemm_bf16(x, weight)
        torch.testing.assert_close(
            actual.double(), x.double() @ weight.double().t(), rtol=2e-2, atol=2e-2
        )


if __name__ == "__main__":
    unittest.main()
