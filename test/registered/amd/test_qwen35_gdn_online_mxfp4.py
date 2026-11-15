"""ROCm coverage for the online MXFP4 Qwen3.5/3.8 GDN projections."""

import itertools
import unittest

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

# Qwen3.8-2.4T TP8: in_proj_qkvz and out_proj.
SHAPES = ((4608, 8192), (8192, 2048))
# Rows on both sides of the 32-row scale-layout split and of the 64-row kernel cut.
ROWS = (1, 4, 16, 31, 32, 64, 65, 256)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip and is_gfx95_supported(),
    "requires AMD gfx95",
)
class TestQwen35GdnOnlineMxfp4(CustomTestCase):
    def test_matches_dequantized_reference(self):
        from aiter import per_1x32_f4_quant_hip

        from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
            AiterMxfp4OnlineLinearMethod,
            _dequant_mxfp4_to_bf16,
            _dynamic_mxfp4_quant_orig,
        )

        for (n, k), m in itertools.product(SHAPES, ROWS):
            with self.subTest(n=n, k=k, m=m):
                torch.manual_seed(m)
                w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                w_q, w_s = _dynamic_mxfp4_quant_orig(w)
                x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=False)
                ref = torch.nn.functional.linear(
                    _dequant_mxfp4_to_bf16(
                        x_q.view(torch.uint8), x_s.view(torch.uint8)[:m]
                    ).float(),
                    _dequant_mxfp4_to_bf16(w_q, w_s.contiguous()).float(),
                )

                layer = torch.nn.Module()
                layer.weight = torch.nn.Parameter(w, requires_grad=False)
                layer.quant_method = AiterMxfp4OnlineLinearMethod()
                # The second call must not quantize the MXFP4 weight again.
                layer.quant_method.process_weights_after_loading(layer)
                layer.quant_method.process_weights_after_loading(layer)
                y = layer.quant_method.apply(layer, x)

                self.assertEqual(y.shape, (m, n))
                rel = ((y.float() - ref).norm() / ref.norm()).item()
                self.assertLess(rel, 1e-2)


if __name__ == "__main__":
    unittest.main()
