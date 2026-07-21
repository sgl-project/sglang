"""Numerical tests for the HPC-Ops bf16xfp32 router GEMM path.

Validates sglang.jit_kernel.dsv4.linear_bf16_fp32's HPC-Ops branch against
the fp32 reference on the LongCat-Flash router shapes. Skipped when HPC-Ops
(https://github.com/Tencent/hpc-ops) is not installed or the GPU is not
Hopper (the kernels ship sm90a only).
"""

import unittest

import torch

from sglang.jit_kernel.dsv4.gemm import (
    _hpc_gemm_bf16xfp32_available,
    _linear_bf16_fp32_hpc,
    linear_bf16_fp32,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

# (hidden_size, n_routed_experts + zero experts) for LongCat-Flash Chat / Lite.
_ROUTER_SHAPES = ((6144, 768), (3072, 384))


@unittest.skipUnless(
    _hpc_gemm_bf16xfp32_available(),
    "requires HPC-Ops (https://github.com/Tencent/hpc-ops) and a Hopper GPU",
)
class TestLinearBf16Fp32Hpc(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)

    def test_matches_fp32_reference(self):
        for k, n in _ROUTER_SHAPES:
            for m in (8, 64, 512):
                with self.subTest(m=m, k=k, n=n):
                    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
                    w = torch.randn(n, k, dtype=torch.float32, device="cuda")
                    out = _linear_bf16_fp32_hpc(x, w)
                    self.assertIsNotNone(out)
                    self.assertEqual(out.dtype, torch.float32)
                    ref = torch.mm(x.float(), w.t())
                    torch.testing.assert_close(out, ref, rtol=0.08, atol=0.01)

    def test_min_m_dispatch(self):
        k, n = _ROUTER_SHAPES[0]
        w = torch.randn(n, k, dtype=torch.float32, device="cuda")
        below = torch.randn(4, k, dtype=torch.bfloat16, device="cuda")
        self.assertIsNone(_linear_bf16_fp32_hpc(below, w, min_m=8))
        # The public entry falls back to cublas below min_m and still
        # returns the correct fp32 result.
        out = linear_bf16_fp32(below, w, hpc_kernel_min_m=8)
        torch.testing.assert_close(
            out, torch.mm(below.float(), w.t()), rtol=0.08, atol=0.01
        )

    def test_weight_split_cache_reused(self):
        k, n = _ROUTER_SHAPES[1]
        x = torch.randn(16, k, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(n, k, dtype=torch.float32, device="cuda")
        out1 = _linear_bf16_fp32_hpc(x, w)
        cache = getattr(w, "_sglang_bf16xfp32_weight_cache")
        out2 = _linear_bf16_fp32_hpc(x, w)
        self.assertIs(getattr(w, "_sglang_bf16xfp32_weight_cache"), cache)
        torch.testing.assert_close(out1, out2)
        # The kernel leaves the cached split-K workspace zeroed.
        self.assertTrue((cache[3] == 0).all().item())


if __name__ == "__main__":
    unittest.main()
