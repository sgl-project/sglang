"""ROCm coverage for the online MXFP4 Qwen3.5/3.8 GDN projections."""

import itertools
import sys
import types
import unittest
from unittest import mock

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

# Qwen3.8-2.4T TP8: in_proj_qkvz and out_proj.
SHAPES = ((4608, 8192), (8192, 2048))
# Rows on both sides of the 32-row scale-layout split and of the 64-row kernel cut.
ROWS = (1, 4, 16, 31, 32, 64, 65, 256)


class TestBlockHadamard32(CustomTestCase):
    def test_hadamard_keeps_products(self):
        from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
            _block_hadamard32,
        )

        x, w = torch.randn(4, 256), torch.randn(8, 256)
        default = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)  # as in the model loader
        try:
            xh, wh = _block_hadamard32(x), _block_hadamard32(w)
        finally:
            torch.set_default_dtype(default)
        torch.testing.assert_close(xh @ wh.T, x @ w.T, rtol=1e-4, atol=1e-4)

    def test_falls_back_without_aiter_hadamard(self):
        from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
            AiterMxfp4OnlineLinearMethod,
        )

        old_aiter = types.SimpleNamespace(per_1x32_f4_quant_hip=lambda x, shuffle: x)
        with mock.patch.dict(sys.modules, {"aiter": old_aiter}):
            method = AiterMxfp4OnlineLinearMethod(hadamard=True)
        self.assertFalse(method.hadamard)
        self.assertEqual(method.quant_kwargs, {})


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip and is_gfx95_supported(),
    "requires AMD gfx95",
)
class TestQwen35GdnOnlineMxfp4(CustomTestCase):
    def test_matches_dequantized_reference(self):
        import inspect

        from aiter import per_1x32_f4_quant_hip

        from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
            AiterMxfp4OnlineLinearMethod,
            _block_hadamard32,
            _dequant_mxfp4_to_bf16,
            _dynamic_mxfp4_quant_orig,
        )

        rotations = [False]
        if "hadamard_rotate" in inspect.signature(per_1x32_f4_quant_hip).parameters:
            rotations.append(True)
        for (n, k), m, had in itertools.product(SHAPES, ROWS, rotations):
            with self.subTest(n=n, k=k, m=m, hadamard=had):
                torch.manual_seed(m)
                w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                rot = _block_hadamard32 if had else (lambda t: t)
                w_q, w_s = _dynamic_mxfp4_quant_orig(rot(w))
                x_q, x_s = per_1x32_f4_quant_hip(rot(x), shuffle=False)
                ref = torch.nn.functional.linear(
                    _dequant_mxfp4_to_bf16(
                        x_q.view(torch.uint8), x_s.view(torch.uint8)[:m]
                    ).float(),
                    _dequant_mxfp4_to_bf16(w_q, w_s.contiguous()).float(),
                )

                layer = torch.nn.Module()
                layer.weight = torch.nn.Parameter(w, requires_grad=False)
                layer.quant_method = AiterMxfp4OnlineLinearMethod(had)
                # The second call must not quantize the MXFP4 weight again.
                layer.quant_method.process_weights_after_loading(layer)
                layer.quant_method.process_weights_after_loading(layer)
                y = layer.quant_method.apply(layer, x)

                self.assertEqual(y.shape, (m, n))
                rel = ((y.float() - ref).norm() / ref.norm()).item()
                self.assertLess(rel, 1e-2)


if __name__ == "__main__":
    unittest.main()
