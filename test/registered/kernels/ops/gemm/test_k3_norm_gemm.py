"""``kda_onorm_oproj_hip`` (gated per-head RMSNorm fused into the o_proj GEMM)
must match the unfused norm followed by a bf16 GEMM."""

import unittest

import torch

from sglang.kernels.ops.gemm.k3_norm_gemm import (
    hip_onorm_supported,
    kda_onorm_oproj_hip,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")

HEADS, D, EPS = 12, 128, 1e-6


def _reference(x, gate, norm_weight, w):
    x32 = x.float().view(x.shape[0], HEADS, D)
    rstd = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)
    o = x32 * rstd * norm_weight.float() * torch.sigmoid(gate.float().view_as(x32))
    return o.to(torch.bfloat16).view(x.shape[0], -1) @ w.t()


class TestK3NormGemm(CustomTestCase):
    def test_matches_unfused(self):
        K = HEADS * D
        for M in (1, 3, 8, 16):
            for N in (1024, 7168):
                with self.subTest(M=M, N=N):
                    self.assertTrue(hip_onorm_supported(M, K, N, D))
                    torch.manual_seed(M * N)
                    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                    # gate is a column slice of a wider projection output
                    gate = torch.randn(M, 2 * K, device="cuda", dtype=torch.bfloat16)[
                        :, K:
                    ]
                    norm_weight = torch.rand(D, device="cuda") + 0.5
                    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / 32

                    out = kda_onorm_oproj_hip(x, gate, norm_weight, EPS, w)

                    torch.testing.assert_close(
                        out, _reference(x, gate, norm_weight, w), rtol=2e-2, atol=2e-2
                    )

    def test_rejects_unsupported_shapes(self):
        self.assertFalse(hip_onorm_supported(17, HEADS * D, 1024, D))
        self.assertFalse(hip_onorm_supported(4, HEADS * D, 1000, D))
        self.assertFalse(hip_onorm_supported(4, 8 * D, 1024, D))


if __name__ == "__main__":
    unittest.main()
