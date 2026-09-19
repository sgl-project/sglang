"""MXFP4 activation quant on gfx950: match aiter bitwise, and cover the part.

The SGLang kernel exists to re-pick the tile when a decode-sized activation
underfills the launch; the quantization itself is per-32-block and independent
of the tiling, so the output must stay bit-identical to aiter's at every shape,
decode-sized or not. This pins that, the transposed scale layout the MXFP4
GEMMs read, and the fact that the decode shapes actually take the re-pick.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

# GLM-5.2 TP4/EP4 decode issues M = 6 * concurrency into the MoE; the prefill
# rows stand in for the regime the published ladder was sized for.
DECODE_SHAPES = [(6, 6144), (24, 6144), (48, 6144), (60, 4096), (84, 2048)]
PREFILL_SHAPES = [(4096, 6144), (2048, 16384), (1024, 512)]
# N <= 1024 and M <= 32 are the ladder's other two branches.
EDGE_SHAPES = [(1, 512), (32, 1024), (33, 64), (7, 2048)]


def _gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    return (
        str(torch.cuda.get_device_properties(0).gcnArchName).split(":")[0] == "gfx950"
    )


@unittest.skipUnless(_gfx950(), "gfx950 only")
class TestMxfp4QuantGfx950(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from aiter.ops.triton.quant import dynamic_mxfp4_quant as aiter_quant

        from sglang.kernels.ops.quantization.mxfp4_quant import dynamic_mxfp4_quant

        cls.aiter_quant = staticmethod(aiter_quant)
        cls.quant = staticmethod(dynamic_mxfp4_quant)
        cls.num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    def _inputs(self, m, n):
        torch.manual_seed(m * 1000 + n)
        # Spread the magnitudes so every E2M1 branch is exercised: normals,
        # denormals (|x| below one scaled unit) and saturation (above 6).
        x = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
        x[:, ::7] *= 1e-3
        x[:, ::11] *= 1e3
        return x

    def test_matches_aiter_bitwise(self):
        for m, n in DECODE_SHAPES + PREFILL_SHAPES + EDGE_SHAPES:
            with self.subTest(m=m, n=n):
                x = self._inputs(m, n)
                got_v, got_s = self.quant(x)
                want_v, want_s = self.aiter_quant(x)
                self.assertEqual(got_v.shape, want_v.shape)
                self.assertEqual(got_s.shape, want_s.shape)
                self.assertTrue(torch.equal(got_v, want_v), f"values differ at {m}x{n}")
                self.assertTrue(torch.equal(got_s, want_s), f"scales differ at {m}x{n}")

    def test_scale_row_stride_is_one(self):
        # gemm_afp4wfp4 reads a column of block scales per row-tile.
        for m, n in DECODE_SHAPES:
            with self.subTest(m=m, n=n):
                _, s = self.quant(self._inputs(m, n))
                self.assertEqual(s.stride(0), 1)
                self.assertEqual(s.stride(1), m)

    def test_refill_never_reduces_coverage(self):
        # The 512-element tile floor can exclude every wider-grid candidate, so
        # the re-pick is not monotone on its own; the launcher has to guard it.
        from sglang.kernels.ops.quantization.mxfp4_quant import (
            _ladder_tile,
            _refill_tile,
            _select_tile,
        )

        for m, n in DECODE_SHAPES + PREFILL_SHAPES + EDGE_SHAPES:
            with self.subTest(m=m, n=n):
                lbm, lbn, lni, _, _ = _ladder_tile(m, n)
                ladder_grid = (-(-m // lbm)) * (-(-n // (lbn * lni)))
                bm, bn, ni, _, _ = _select_tile(m, n, self.num_sms)
                grid = (-(-m // bm)) * (-(-n // (bn * ni)))
                self.assertGreaterEqual(grid, ladder_grid)

        # M = 6, N = 6144 is the shape where the bare re-pick regresses.
        rbm, rbn = _refill_tile(6, 6144, 256)
        self.assertLess((-(-6 // rbm)) * (-(-6144 // rbn)), 192)
        sbm, sbn, sni, _, _ = _select_tile(6, 6144, 256)
        self.assertEqual((-(-6 // sbm)) * (-(-6144 // (sbn * sni))), 192)

    def test_decode_verify_shape_gains_coverage(self):
        from sglang.kernels.ops.quantization.mxfp4_quant import (
            _ladder_tile,
            _select_tile,
        )

        # 6 * concurrency 8, the deployed GLM-5.2 TP4/EP4 verify row count.
        lbm, lbn, lni, _, _ = _ladder_tile(48, 6144)
        bm, bn, ni, _, _ = _select_tile(48, 6144, self.num_sms)
        self.assertGreater(
            (-(-48 // bm)) * (-(-6144 // (bn * ni))),
            4 * ((-(-48 // lbm)) * (-(-6144 // (lbn * lni)))),
        )

    def test_zero_and_constant_rows(self):
        # An all-zero block has amax 0, so log2 underflows; the clamp has to
        # hold it at the E8M0 floor rather than emit a NaN scale.
        for m, n in [(8, 2048), (48, 6144)]:
            with self.subTest(m=m, n=n):
                for x in (
                    torch.zeros(m, n, dtype=torch.bfloat16, device="cuda"),
                    torch.full((m, n), 1.5, dtype=torch.bfloat16, device="cuda"),
                ):
                    got_v, got_s = self.quant(x)
                    want_v, want_s = self.aiter_quant(x)
                    self.assertTrue(torch.equal(got_v, want_v))
                    self.assertTrue(torch.equal(got_s, want_s))


if __name__ == "__main__":
    unittest.main()
