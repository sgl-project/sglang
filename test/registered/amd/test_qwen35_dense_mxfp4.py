import unittest

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

_RUNNABLE = is_hip() and is_gfx95_supported()
if _RUNNABLE:
    try:
        from sglang.kernels.ops.gemm import mxfp4_dense_aiter_hip as mxfp4
        from sglang.srt.layers.quantization.mxfp4_dense import (
            Mxfp4DenseLinearMethod,
            enable_mxfp4_dense,
        )
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        _RUNNABLE = mxfp4.supported()
    except Exception:
        _RUNNABLE = False

# (name, out_features, in_features) per rank at TP2 for Qwen3.5-397B-A17B-MXFP4.
SHAPES = [
    ("gdn.in_proj_qkvz", 10240, 4096),
    ("gdn.out_proj", 4096, 4096),
    ("attn.qkv_proj", 8704, 4096),
    ("attn.o_proj", 4096, 4096),
]

# MXFP4 is e2m1 with one E8M0 scale per 32 elements, so a normalized product keeps
# only a few bits. On zero-mean random inputs the accumulation does not average the
# error away, and ~20% is the level the routed experts of an MXFP4 checkpoint
# already run at; this bound is here to catch a broken layout, not to bless the
# precision. Real-model accuracy is a gsm8k question.
_MAX_REL_ERR = 0.30


class _Layer(torch.nn.Module):
    """The surface `UnquantizedLinearMethod.apply` reads off a LinearBase."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.quant_method = UnquantizedLinearMethod()

    def load(self) -> None:
        self.quant_method.process_weights_after_loading(self)


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    got, ref = got.float(), ref.float()
    return (torch.linalg.vector_norm(got - ref) / torch.linalg.vector_norm(ref)).item()


@unittest.skipUnless(_RUNNABLE, "requires HIP gfx950 with aiter")
class TestQwen35DenseMxfp4(CustomTestCase):
    def _layer(self, n: int, k: int, min_tokens: int) -> _Layer:
        torch.manual_seed(0)
        weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") / (k**0.5)
        layer = _Layer(weight)
        self.assertTrue(enable_mxfp4_dense(layer, min_tokens))
        self.assertIsInstance(layer.quant_method, Mxfp4DenseLinearMethod)
        layer.load()
        return layer

    def test_matches_bf16_within_mxfp4_error(self):
        for name, n, k in SHAPES:
            with self.subTest(name):
                layer = self._layer(n, k, min_tokens=1024)
                x = torch.randn(2048, k, dtype=torch.bfloat16, device="cuda")
                ref = torch.nn.functional.linear(x.float(), layer.weight.float())
                got = layer.quant_method.apply(layer, x)
                self.assertEqual(got.shape, (x.shape[0], n))
                self.assertEqual(got.dtype, torch.bfloat16)
                self.assertLess(_rel_err(got, ref), _MAX_REL_ERR)

    def test_below_threshold_is_untouched_bf16(self):
        """Decode must be bit-identical to the BF16 path it would have taken."""
        name, n, k = SHAPES[0]
        layer = self._layer(n, k, min_tokens=1024)
        for tokens in (1, 128, 1023):
            with self.subTest(tokens=tokens):
                x = torch.randn(tokens, k, dtype=torch.bfloat16, device="cuda")
                expect = UnquantizedLinearMethod().apply(layer, x)
                torch.testing.assert_close(
                    layer.quant_method.apply(layer, x), expect, rtol=0, atol=0
                )

    def test_threshold_is_inclusive(self):
        name, n, k = SHAPES[1]
        layer = self._layer(n, k, min_tokens=512)
        x = torch.randn(512, k, dtype=torch.bfloat16, device="cuda")
        bf16 = UnquantizedLinearMethod().apply(layer, x)
        got = layer.quant_method.apply(layer, x)
        self.assertFalse(torch.equal(got, bf16), "512 tokens should have used MXFP4")

    def test_bias_is_applied(self):
        name, n, k = SHAPES[1]
        layer = self._layer(n, k, min_tokens=1024)
        x = torch.randn(2048, k, dtype=torch.bfloat16, device="cuda")
        bias = torch.randn(n, dtype=torch.bfloat16, device="cuda")
        torch.testing.assert_close(
            layer.quant_method.apply(layer, x, bias),
            layer.quant_method.apply(layer, x) + bias,
            rtol=2e-2,
            atol=2e-2,
        )

    def test_unpackable_weight_stays_bf16(self):
        """A shape MXFP4 cannot take must degrade to BF16, not fail to load."""
        # 40 output rows is not a multiple of the 16-row preshuffle.
        weight = torch.randn(40, 4096, dtype=torch.bfloat16, device="cuda")
        layer = _Layer(weight)
        self.assertTrue(enable_mxfp4_dense(layer, min_tokens=1))
        layer.load()
        x = torch.randn(2048, 4096, dtype=torch.bfloat16, device="cuda")
        torch.testing.assert_close(
            layer.quant_method.apply(layer, x),
            UnquantizedLinearMethod().apply(layer, x),
            rtol=0,
            atol=0,
        )

    def test_declines_an_already_quantized_layer(self):
        weight = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
        layer = _Layer(weight)
        layer.quant_method = object()
        self.assertFalse(enable_mxfp4_dense(layer, min_tokens=1))

    def test_non_2d_input_falls_back(self):
        name, n, k = SHAPES[1]
        layer = self._layer(n, k, min_tokens=1)
        x = torch.randn(4, 512, k, dtype=torch.bfloat16, device="cuda")
        torch.testing.assert_close(
            layer.quant_method.apply(layer, x),
            UnquantizedLinearMethod().apply(layer, x),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
