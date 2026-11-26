"""Batch-invariant log_softmax and RMSNorm (no residual).

These one-program-per-row kernels pick BLOCK_SIZE and num_warps from the row
width only. A row's result must not depend on how many other rows share the
batch, and must stay close to the eager result.
"""

import unittest

import torch

from sglang.srt.batch_invariant_ops import log_softmax, rms_norm_batch_invariant
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=15, suite="base-b-test-1-gpu-small-amd-mi35x")

EPS = 1e-6


def _check_batch_invariance(test, fn, x, m):
    out = fn(x)
    for bs in (1, 3, 17):
        for start in range(0, m, bs):
            rows = slice(start, min(start + bs, m))
            test.assertTrue(torch.equal(fn(x[rows]), out[rows]), f"bs={bs}")
    return out


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestRowKernelsBatchInvariant(CustomTestCase):
    def test_log_softmax(self):
        # short rows, non-powers of two, and vocabulary widths
        for n_cols in (100, 1024, 5000, 32000, 128256, 151936):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(n_cols=n_cols, dtype=dtype):
                    torch.manual_seed(0)
                    m = 37
                    x = (torch.randn(m, n_cols, device="cuda") * 3).to(dtype)
                    out = _check_batch_invariance(self, log_softmax, x, m)
                    ref = torch.log_softmax(x.double(), dim=-1).to(dtype)
                    tol = 1e-5 if dtype == torch.float32 else 3e-2
                    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)

    def test_log_softmax_extreme_values(self):
        torch.manual_seed(0)
        x = torch.randn(8, 151936, device="cuda")
        x[0, 7] = 1e4  # one dominant logit
        x[1, :] = -1e4  # all very negative
        x[2, 100:] = float("-inf")  # masked tail
        out = log_softmax(x)
        ref = torch.log_softmax(x.double(), dim=-1).float()
        finite = torch.isfinite(ref)
        self.assertTrue(torch.equal(torch.isfinite(out), finite))
        torch.testing.assert_close(out[finite], ref[finite], atol=1e-5, rtol=1e-5)

    def test_rms_norm(self):
        # 128 is the q/k-norm head size; the rest are hidden sizes
        for hidden in (128, 2048, 4096, 5120, 2880):
            for dtype in (torch.bfloat16, torch.float16):
                with self.subTest(hidden=hidden, dtype=dtype):
                    torch.manual_seed(0)
                    m = 64
                    x = torch.randn(m, hidden, device="cuda", dtype=dtype)
                    weight = torch.randn(hidden, device="cuda", dtype=dtype)

                    def fn(t):
                        return rms_norm_batch_invariant(t, weight, EPS)

                    out = _check_batch_invariance(self, fn, x, m)
                    xf = x.float()
                    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
                    ref = (ref * weight.float()).to(dtype)
                    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
