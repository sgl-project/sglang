"""Both SAIL MXFP4 scale storage layouts must keep the quantized values identical."""

import unittest

import torch

from sglang.srt.utils import is_sail
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=20, suite="stage-a-test-1-gpu-ppu")


class TestSailMxfp4ScaleLayout(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not is_sail() or not torch.cuda.is_available():
            raise unittest.SkipTest(
                "The MXFP4 quantization kernels only run on SAIL devices."
            )

    def test_fused_layout_preserves_quantized_values(self):
        from sglang.kernels.ops.quantization.sail_mxfp4 import downcast_to_mxfp4

        for shape, axis in (
            ((17, 192), -1),
            ((2, 96, 3), 1),
            ((1, 128), 1),
            ((0, 128), 1),
        ):
            for round_up in (False, True):
                with self.subTest(shape=shape, axis=axis, round_up=round_up):
                    torch.manual_seed(42)
                    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                    ordinary_q, ordinary_s = downcast_to_mxfp4(x, axis, round_up)
                    fused_q, fused_s = downcast_to_mxfp4(
                        x, axis, round_up, enable_fused_moe=True
                    )
                    self.assertEqual(fused_s.dtype, torch.uint16)
                    self.assertTrue(torch.equal(ordinary_q, fused_q))
                    self.assertTrue(
                        torch.equal(ordinary_s.to(torch.int32), fused_s.to(torch.int32))
                    )
                    k_major = fused_s.movedim(axis, -1)
                    self.assertTrue(k_major.is_contiguous())
                    if shape == (17, 192):
                        self.assertEqual(ordinary_s.stride(), (1, 17))
                        self.assertEqual(fused_s.stride(), (3, 1))


if __name__ == "__main__":
    unittest.main()
