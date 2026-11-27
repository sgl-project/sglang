"""Fused 1x128 FP8 input quant (RMSNorm and SiLU-and-mul) vs the unfused path
that feeds the gfx942 block-FP8 GEMM: bf16 norm / activation, then aiter's
per-1x128 quant to fnuz fp8 with a row-major scale."""

import itertools
import os
import unittest

os.environ.setdefault("SGLANG_USE_AITER", "1")

import torch

from sglang.srt.utils import is_gfx942_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="base-b-test-1-gpu-small-amd")

M_VALUES = (1, 8, 32, 64, 256, 512)
# Qwen3-14B and Qwen3-1.7B hidden / intermediate sizes, and their per-rank
# intermediate sizes at TP8 / TP4 (17 / 12 groups: a partial last program).
HIDDEN_SIZES = (5120, 2048)
INTERMEDIATE_SIZES = (17408, 6144, 2176, 1536)

# The fused norm runs a different RMSNorm kernel than the unfused one, so the
# normalized value can land on the other side of a rounding boundary: allow
# one fp8 step on a small fraction of elements, and a scale within the bf16
# rounding of the group amax. The fused SiLU-and-mul rounds like SiluAndMul,
# so its fp8 output and scales must match exactly.
NORM_MAX_MISMATCH = 0.05
SILU_MAX_MISMATCH = 0.0
SCALE_RTOL = 2**-7


def _fp8_ulp(x: torch.Tensor) -> torch.Tensor:
    _, exp = torch.frexp(x.abs())
    return torch.ldexp(torch.ones_like(x), exp - 4).clamp_min(2.0**-10)


def _ref_quant(x: torch.Tensor):
    import aiter

    from sglang.srt.layers.quantization.fp8_utils import aiter_per1x128_quant

    return aiter_per1x128_quant(x, quant_dtype=aiter.dtypes.fp8, transpose_scale=False)


@unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_gfx942_supported(),
    "gfx942 + aiter only",
)
class TestFusedNormActFp8GroupQuantGfx942(CustomTestCase):
    def _compare(self, name, q, s, q_ref, s_ref, max_mismatch):
        self.assertEqual(q.dtype, q_ref.dtype)
        self.assertEqual(q.shape, q_ref.shape)
        self.assertEqual(s.shape, s_ref.shape)
        self.assertEqual(s.stride(), s_ref.stride())
        qf, qr = q.float(), q_ref.float()
        q_mismatch = (q.view(torch.uint8) != q_ref.view(torch.uint8)).float().mean()
        s_mismatch = (s != s_ref).float().mean()
        step = (qf - qr).abs() / _fp8_ulp(torch.maximum(qf.abs(), qr.abs()))
        print(
            f"{name}: fp8 mismatch {q_mismatch.item():.5f}, max steps "
            f"{step.max().item():.2f}, scale mismatch {s_mismatch.item():.5f}, "
            f"max scale rel diff {((s - s_ref).abs() / s_ref).max().item():.3e}"
        )
        torch.testing.assert_close(s, s_ref, rtol=SCALE_RTOL, atol=0)
        self.assertLessEqual(q_mismatch.item(), max_mismatch)
        self.assertLessEqual(step.max().item(), 1.0)

    def test_rmsnorm_fp8_group_quant(self):
        from sglang.srt.layers.layer_boundary.residual.add_norm import (
            _update_and_read_residual_aiter_fp8_group_gfx942,
        )
        from sglang.srt.layers.layernorm import RMSNorm

        for hidden in HIDDEN_SIZES:
            norm = RMSNorm(hidden, eps=1e-6, weight_dtype=torch.bfloat16).cuda()
            with torch.no_grad():
                norm.weight.normal_(1.0, 0.1)
            for m, with_residual in itertools.product(M_VALUES, (True, False)):
                with self.subTest(hidden=hidden, m=m, residual=with_residual):
                    torch.manual_seed(m)
                    x = torch.randn(m, hidden, dtype=torch.bfloat16, device="cuda")
                    res = torch.randn_like(x) * 4 if with_residual else None
                    if with_residual:
                        y, res_ref = norm(x.clone(), res.clone())
                    else:
                        y, res_ref = norm(x.clone()), x
                    q_ref, s_ref = _ref_quant(y)

                    (q, s), res_out = _update_and_read_residual_aiter_fp8_group_gfx942(
                        norm, x.clone(), None if res is None else res.clone(), None
                    )

                    self.assertTrue(torch.equal(res_out, res_ref))
                    self._compare(
                        f"norm h={hidden} m={m} res={with_residual}",
                        q,
                        s,
                        q_ref,
                        s_ref,
                        NORM_MAX_MISMATCH,
                    )

    def test_silu_mul_fp8_group_quant(self):
        import aiter

        from sglang.kernels.ops.activation.silu_mul_fp8_group_quant import (
            silu_mul_fp8_group_quant,
        )
        from sglang.srt.layers.activation import SiluAndMul

        act = SiluAndMul()
        for d in INTERMEDIATE_SIZES:
            for m in M_VALUES:
                with self.subTest(d=d, m=m):
                    torch.manual_seed(m)
                    x = 2 * torch.randn(m, 2 * d, dtype=torch.bfloat16, device="cuda")
                    q_ref, s_ref = _ref_quant(act(x))

                    q, s = silu_mul_fp8_group_quant(
                        x, transpose_scale=False, dtype=aiter.dtypes.fp8
                    )
                    self._compare(
                        f"silu d={d} m={m}", q, s, q_ref, s_ref, SILU_MAX_MISMATCH
                    )
                    self.assertTrue(torch.equal(s, s_ref))


if __name__ == "__main__":
    unittest.main(verbosity=2)
