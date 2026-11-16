"""gfx950 BF16 small-M GEMMs of bf16_tiled_small_m_gemm: the row-major split-K path and the tiled path.

Every ROW_MAJOR bucket at its smallest, a middle (odd) and its largest row count: coverage, BF16 output and the
split-K partials (with the rowmajor_gemm_partials_split contract) against an FP32 reference, the K-start rotation
(ROW_MAJOR_ROT) against the in-order loop, and the tiled path, which shares _sum_partials.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=90, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _rows(bm):
    lo = 1 if bm == 16 else bm // 2 + 1
    return sorted({lo, (lo + bm) // 2 | 1, bm})


def _rel_err(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


@unittest.skipUnless(is_gfx95_supported(), "gfx950 (MI35x) only")
class TestBf16RowmajorSmallMGemmGfx950(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.gemm import bf16_tiled_small_m_gemm as g

        cls.g = g
        cls.gen = torch.Generator(device="cuda").manual_seed(0)

    def weight(self, n, k):
        return (torch.randn(n, k, device="cuda", generator=self.gen) * 0.02).to(
            torch.bfloat16
        )

    def x(self, m, k):
        return torch.randn(m, k, device="cuda", generator=self.gen).to(torch.bfloat16)

    def test_rowmajor_every_bucket(self):
        g = self.g
        for n, k in sorted({(n, k) for n, k, _ in g.ROW_MAJOR}):
            w = self.weight(n, k)
            for bm in sorted(b for nn, kk, b in g.ROW_MAJOR if (nn, kk) == (n, k)):
                for m in _rows(bm):
                    with self.subTest(n=n, k=k, m=m):
                        x = self.x(m, k)
                        self.assertTrue(g.rowmajor_gemm_applies(x, w))
                        ref = x.float() @ w.float().T
                        y = g.rowmajor_gemm(x, w)
                        self.assertEqual(
                            (y.dtype, tuple(y.shape)), (torch.bfloat16, (m, n))
                        )
                        self.assertLess(_rel_err(y, ref), 1e-2)
                        p = g.rowmajor_gemm_partials(x, w)
                        sk = g.rowmajor_gemm_partials_split(m, n, k)
                        if sk == 1:
                            self.assertEqual(
                                (p.dtype, tuple(p.shape)), (torch.bfloat16, (m, n))
                            )
                            total = p.float()
                        else:
                            self.assertEqual(
                                (p.dtype, tuple(p.shape)), (torch.float32, (sk, m, n))
                            )
                            total = p.sum(0)
                        self.assertLess(_rel_err(total, ref), 1e-2)

    def test_rotation_matches_in_order_loop(self):
        g = self.g
        for (n, k, bm), rot in sorted(g.ROW_MAJOR_ROT.items()):
            if not rot:
                continue
            w = self.weight(n, k)
            for m in _rows(bm):
                with self.subTest(n=n, k=k, m=m):
                    x = self.x(m, k)
                    rotated = g.rowmajor_gemm_partials(x, w)
                    with patch.dict(g.ROW_MAJOR_ROT, {(n, k, bm): 0}):
                        in_order = g.rowmajor_gemm_partials(x, w)
                    # Same products, K summed in a different FP32 order; compared in norm, since single elements
                    # near zero can differ by cancellation. SK == 1 launches write BF16, where the order can move an
                    # element by one BF16 rounding step, hence the looser bound there.
                    tol = 1e-5 if rotated.dtype == torch.float32 else 1e-3
                    self.assertLess(_rel_err(rotated, in_order.float()), tol)

    def test_tiled_path(self):
        g = self.g
        for n, k, bm in sorted(g.LAUNCH):
            with self.subTest(n=n, k=k, bm=bm):
                w = self.weight(n, k)
                tiled = g.prepare_tiled_weight(w)
                m = bm // 2 + 3
                x = self.x(m, k)
                self.assertTrue(g.tiled_gemm_applies(x, tiled))
                y = g.tiled_gemm(x, tiled, n, k)
                self.assertLess(_rel_err(y, x.float() @ w.float().T), 1e-2)


if __name__ == "__main__":
    unittest.main()
