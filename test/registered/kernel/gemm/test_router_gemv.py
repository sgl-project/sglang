"""Parity test for the gfx950 skinny router GEMV (bf16 x bf16 -> fp32 logits).

Covers every M bucket, including the 65..128-row bucket that EAGLE
target-verify batches use (bs x num_draft_tokens rows), against a fp32
torch.mm reference.
"""

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")

import unittest

import torch

from sglang.kernels.ops.gemm import router_gemv as rg
from sglang.test.test_utils import CustomTestCase


class TestRouterGemv(CustomTestCase):
    N = 128
    K = 6144

    @unittest.skipUnless(
        torch.cuda.is_available() and rg._is_gfx95_supported, "gfx950 only"
    )
    def test_all_row_buckets_match_fp32_reference(self):
        torch.manual_seed(0)
        w = (torch.randn(self.N, self.K, device="cuda") * 0.02).to(torch.bfloat16)
        for m in (1, 4, 8, 9, 16, 17, 32, 33, 64, 65, 96, 128):
            x = torch.randn(m, self.K, device="cuda").to(torch.bfloat16)
            self.assertTrue(rg.router_gemv_supported(x, w), msg=f"m={m}")
            ref = torch.mm(x.float(), w.float().t())
            out = rg.router_gemv(x, w)
            self.assertEqual(out.dtype, torch.float32)
            torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4, msg=f"m={m}")
            # Second call reuses the self-cleaning split-K counter.
            torch.testing.assert_close(rg.router_gemv(x, w), ref, rtol=1e-4, atol=1e-4)
        x = torch.randn(rg._MAX_M + 1, self.K, device="cuda").to(torch.bfloat16)
        self.assertFalse(rg.router_gemv_supported(x, w))


if __name__ == "__main__":
    unittest.main()
