"""Real-kernel validation on gfx1200/1201; no model checkpoint is required.

Run with SGLANG_USE_AITER=1 on a ROCm build with rowwise torch._scaled_mm.
"""

import unittest
from unittest.mock import patch

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy

from sglang.kernels.ops.quantization import fp8_kernel
from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a8_fp8 as ct_fp8,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.quantization.quark.schemes.quark_w8a8_fp8 import QuarkW8A8Fp8
from sglang.srt.utils import is_gfx120x_supported


@unittest.skipUnless(
    is_gfx120x_supported() and fp8_kernel._use_aiter,
    "requires gfx1200/1201 and SGLANG_USE_AITER=1",
)
class TestRdna4AiterFp8(unittest.TestCase):
    def test_channel_linear_with_aiter_enabled(self):
        torch.manual_seed(17)
        self.assertTrue(fp8_utils._use_aiter)
        self.assertFalse(fp8_utils.use_aiter_bpreshuffle_gemm(8192))

        # Include a previously failing shape, unrelated dense-layer dimensions,
        # and a narrow output partition. Include serialized and online FP8.
        for scheme_name in ("compressed-tensors", "quark", "online-fp8"):
            for n, k in ((8192, 2048), (1536, 1024), (32, 512)):
                dense = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
                sw = dense.float().abs().amax(dim=1, keepdim=True) / 448.0
                weight = (dense.float() / sw).to(torch.float8_e4m3fn)
                layer = torch.nn.Module()
                layer.weight = torch.nn.Parameter(
                    dense if scheme_name == "online-fp8" else weight,
                    requires_grad=False,
                )
                layer.weight_scale = torch.nn.Parameter(sw, requires_grad=False)
                layer.logical_widths = [n]
                if scheme_name == "compressed-tensors":
                    scheme = ct_fp8.CompressedTensorsW8A8Fp8(
                        QuantizationArgs(
                            num_bits=8,
                            type="float",
                            strategy=QuantizationStrategy.CHANNEL,
                        ),
                        is_static_input_scheme=False,
                    )
                elif scheme_name == "quark":
                    scheme = QuarkW8A8Fp8(
                        weight_config={"qscheme": "per_channel"},
                        input_config={"is_dynamic": True, "qscheme": "per_channel"},
                    )
                else:
                    scheme = Fp8LinearMethod(Fp8Config())
                    scheme.use_aiter_fp8_per_token = True
                    weight, sw = fp8_utils.per_token_group_quant_fp8(
                        dense, group_size=k
                    )
                scheme.process_weights_after_loading(layer)
                torch.testing.assert_close(
                    layer.weight.view(torch.uint8), weight.t().view(torch.uint8)
                )

                for m in (1, 4, 8, 16, 32):
                    with self.subTest(scheme=scheme_name, m=m, n=n, k=k):
                        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                        bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
                        if scheme_name == "compressed-tensors":
                            qx, sx = fp8_kernel.scaled_fp8_quant(
                                x, use_per_token_if_dynamic=True
                            )
                        else:
                            # Quark keeps its existing Triton quantizer; compare
                            # the GEMM against that quantizer's actual operands.
                            qx, sx = fp8_utils.per_token_group_quant_fp8(
                                x, group_size=k
                            )
                        reference = (
                            (qx.float() @ weight.float().t()) * sx * sw.t() + bias
                        ).bfloat16()
                        with patch.object(
                            fp8_utils,
                            "gemm_a8w8_bpreshuffle",
                            side_effect=AssertionError("unsupported GEMM was selected"),
                        ):
                            if scheme_name == "online-fp8":
                                output = scheme.apply(layer, x, bias)
                            else:
                                output = scheme.apply_weights(layer, x, bias)
                        self.assertTrue(torch.isfinite(output).all())
                        torch.testing.assert_close(
                            output, reference, rtol=0.02, atol=0.01
                        )
                        if scheme_name == "compressed-tensors":
                            output = scheme.apply_weights(
                                layer, (qx, sx, torch.bfloat16), bias
                            )
                            torch.testing.assert_close(
                                output, reference, rtol=0.02, atol=0.01
                            )


if __name__ == "__main__":
    unittest.main()
