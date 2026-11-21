"""sm_121 skinny GEMM vs torch.matmul on the planned decode shapes (GB10 only)."""

import unittest

import torch

from sglang.kernels.ops.gemm.sm121_skinny_gemm import (
    SM121_GEMM_PLANS,
    maybe_skinny_gemm,
    skinny_gemm,
    sm121_skinny_enabled,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_ON_SM121 = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (12, 1)
    and sm121_skinny_enabled()
    and skinny_gemm.is_available()
)

# A representative subset; the full table is covered structurally on CPU.
_SHAPES = [(48, 2560), (640, 2560), (2560, 3072), (8240, 2560), (336, 10240)]


@unittest.skipUnless(_ON_SM121, "needs an sm_121 GPU with cutlass-dsl and quack")
class TestSm121SkinnyGemm(unittest.TestCase):
    def test_matches_torch(self):
        torch.manual_seed(0)
        for n, k in _SHAPES:
            w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") / k**0.5
            for m in SM121_GEMM_PLANS[(n, k)]:
                with self.subTest(n=n, k=k, m=m):
                    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
                    out = maybe_skinny_gemm(x, w)
                    self.assertIsNotNone(out)
                    ref = (x.float() @ w.float().T).to(torch.bfloat16)
                    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

    def test_residual_and_out(self):
        n, k, m = 2560, 3072, 4
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") / k**0.5
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        r = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
        ref = (x.float() @ w.float().T + r.float()).to(torch.bfloat16)
        torch.testing.assert_close(
            maybe_skinny_gemm(x, w, residual=r), ref, atol=2e-2, rtol=2e-2
        )
        out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        res = maybe_skinny_gemm(x, w, out=out)
        self.assertEqual(res.data_ptr(), out.data_ptr())

    def test_unplanned_shape_falls_back(self):
        w = torch.randn(1000, 2560, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(4, 2560, dtype=torch.bfloat16, device="cuda")
        self.assertIsNone(maybe_skinny_gemm(x, w))
        x = torch.randn(32, 2560, dtype=torch.bfloat16, device="cuda")
        self.assertIsNone(maybe_skinny_gemm(x, w))


if __name__ == "__main__":
    unittest.main()
