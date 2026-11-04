"""Batch-invariant fused residual add + RMSNorm.

In batch-invariant mode, RMSNorm calls with a residual run this kernel instead of
the eager float32 path. A row's output and new residual must not depend on how
many other rows share the batch, and must match the eager float32 math.
"""

import unittest

import torch

from sglang.srt.batch_invariant_ops import fused_add_rms_norm_batch_invariant
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd-mi35x")

EPS = 1e-6


def _reference(x, residual, weight, addend, fp32_residual):
    """The float32 math of RMSNorm.forward_native with a residual."""
    s = x.float() + residual.float()
    if addend is not None:
        s = s + addend.float()
    new_residual = s.clone() if fp32_residual else s.to(x.dtype)
    out = s * torch.rsqrt(s.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return (out * weight).to(x.dtype), new_residual


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestFusedAddRMSNormBatchInvariant(CustomTestCase):
    def _check(self, hidden, dtype, with_addend, fp32_residual):
        torch.manual_seed(0)
        m = 64
        x = torch.randn(m, hidden, device="cuda", dtype=dtype)
        residual = torch.randn(m, hidden, device="cuda", dtype=dtype)
        addend = (
            torch.randn(m, hidden, device="cuda", dtype=dtype) if with_addend else None
        )
        weight = torch.randn(hidden, device="cuda", dtype=dtype)
        residual_dtype = torch.float32 if fp32_residual else dtype

        def run(rows):
            return fused_add_rms_norm_batch_invariant(
                x[rows],
                residual[rows],
                weight,
                EPS,
                addend=addend[rows] if addend is not None else None,
                residual_dtype=residual_dtype,
            )

        out, new_residual = run(slice(0, m))
        self.assertEqual(new_residual.dtype, residual_dtype)

        # Batch invariance: every row equals the same row computed in smaller batches.
        for bs in (1, 3, 17):
            for start in range(0, m, bs):
                rows = slice(start, min(start + bs, m))
                part_out, part_residual = run(rows)
                self.assertTrue(torch.equal(part_out, out[rows]), f"bs={bs}")
                self.assertTrue(torch.equal(part_residual, new_residual[rows]))

        # Same math as the eager path.
        ref_out, ref_residual = _reference(x, residual, weight, addend, fp32_residual)
        self.assertTrue(torch.equal(new_residual, ref_residual))
        torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=2e-2)

    def test_fused_add_rms_norm(self):
        for hidden in (2048, 4096, 5120, 2880):
            for dtype in (torch.bfloat16, torch.float16):
                for with_addend in (False, True):
                    for fp32_residual in (False, True):
                        with self.subTest(
                            hidden=hidden,
                            dtype=dtype,
                            with_addend=with_addend,
                            fp32_residual=fp32_residual,
                        ):
                            self._check(hidden, dtype, with_addend, fp32_residual)


if __name__ == "__main__":
    unittest.main()
