"""The fused gate retains FP32 normalization and casts only the final result."""

import itertools
import unittest

import torch

from sglang.kernels.ops.embeddings.engram_gate import fused_engram_gate
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")


def reference(x, kv, qw, kw, eps, clamp):
    hc, dim = x.shape[-2:]
    key, value = kv.split([hc * dim, dim], dim=-1)
    h, key, value = x.float(), key.float().reshape_as(x), value.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (h * (qw.float() * kw.float()) * key).sum(-1) * rstd * dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp).sqrt(), dot))
    return (h + gate.unsqueeze(-1) * value.unsqueeze(-2)).to(x.dtype)


class TestEngramGate(CustomTestCase):
    def test_numerics_and_ownership(self):
        torch.manual_seed(23)
        # Token counts span decode (1), a verify block (8) and a prefill chunk (4096).
        for dtype, weight_dtype, batch, dim in itertools.product(
            (torch.bfloat16, torch.float32),
            (torch.bfloat16, torch.float32),
            (0, 1, 8, 4096),
            (128, 5120),
        ):
            with self.subTest(dtype=dtype, weights=weight_dtype, batch=batch, dim=dim):
                x = torch.randn(batch, 4, dim, device="cuda", dtype=dtype)
                kv = torch.randn(batch, 5 * dim, device="cuda", dtype=dtype)
                qw = torch.randn(4, dim, device="cuda", dtype=weight_dtype)
                kw = torch.randn_like(qw)
                originals = [v.clone() for v in (x, kv, qw, kw)]
                result = fused_engram_gate(x, kv, qw, kw, 1e-6, 1e-6)
                # fp32: the 5120-wide reductions run in a different order than
                # torch's, worth ~1e-5 absolute on a handful of elements at 4096 tokens.
                torch.testing.assert_close(
                    result,
                    reference(x, kv, qw, kw, 1e-6, 1e-6),
                    atol=1e-5 if dtype == torch.bfloat16 else 2e-5,
                    rtol=8e-3 if dtype == torch.bfloat16 else 3e-5,
                )
                self.assertEqual(result.dtype, dtype)
                self.assertTrue(result.is_contiguous())
                for value, original in zip((x, kv, qw, kw), originals):
                    self.assertTrue(torch.equal(value, original))

    def test_image_select_keeps_the_input(self):
        """``image_select`` folds the model's ``where(input_ids == image_id, x, gated)`` into
        the launch, bitwise."""
        torch.manual_seed(31)
        for batch, dim in ((1, 5120), (64, 5120)):
            with self.subTest(batch=batch, dim=dim):
                x = torch.randn(batch, 4, dim, device="cuda", dtype=torch.bfloat16)
                kv = torch.randn(batch, 5 * dim, device="cuda", dtype=torch.bfloat16)
                qw = torch.rand(4, dim, device="cuda", dtype=torch.bfloat16)
                kw = torch.rand_like(qw)
                ids = torch.randint(0, 3, (batch,), device="cuda")
                ids[0] = 2
                gated = fused_engram_gate(x, kv, qw, kw, 1e-6, 1e-6)
                ref = torch.where((ids == 2)[:, None, None], x, gated)
                got = fused_engram_gate(
                    x, kv, qw, kw, 1e-6, 1e-6, image_select=(ids, 2)
                )
                self.assertTrue(torch.equal(got, ref))
                self.assertTrue(torch.equal(got[0], x[0]))
                # no image token in the batch: the plain gate
                got = fused_engram_gate(
                    x, kv, qw, kw, 1e-6, 1e-6, image_select=(ids, 99)
                )
                self.assertTrue(torch.equal(got, gated))


if __name__ == "__main__":
    unittest.main()
