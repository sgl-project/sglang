import unittest

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

_RUNNABLE = is_hip() and is_gfx95_supported()
if _RUNNABLE:
    try:
        from sglang.kernels.ops.gemm import mxfp4_dense_aiter_hip as _mxfp4
        from sglang.kernels.ops.gemm import mxfp6_dense_aiter_hip as _mxfp6
        from sglang.srt.layers.quantization.mx_dense import (
            MxDenseLinearMethod,
            mx_dense_supported,
        )
        from sglang.srt.layers.quantization.quark import dense_mx
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        from sglang.srt.models import qwen3_5_dense_mx

        _RUNNABLE = _mxfp4.supported() and _mxfp6.supported()
    except Exception:
        _RUNNABLE = False

# (name, out_features, in_features) per rank at TP2 for Qwen3.5-397B-A17B.
SHAPES = [
    ("gdn.in_proj_qkvz", 10240, 4096),
    ("gdn.out_proj", 4096, 4096),
    ("attn.qkv_proj", 8704, 4096),
    ("attn.o_proj", 4096, 4096),
]

# Per-format error ceilings on zero-mean random inputs, where accumulation does
# not average the error away. These are loose bounds meant to catch a broken
# layout, not to bless the precision -- real-model accuracy is a gsm8k question.
# The measured values on these shapes are ~16% for MXFP4 and ~4% for MXFP6, and
# the gap between them is the reason MXFP6 is the default, so it is asserted
# separately in test_mxfp6_is_markedly_more_accurate_than_mxfp4.
_MAX_REL_ERR = {"mxfp4": 0.30, "mxfp6": 0.10}

FORMATS = tuple(_MAX_REL_ERR)


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
class TestQwen35DenseMx(CustomTestCase):
    def _layer(self, n: int, k: int, min_tokens: int, fmt: str) -> _Layer:
        torch.manual_seed(0)
        weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") / (k**0.5)
        layer = _Layer(weight)
        layer.quant_method = MxDenseLinearMethod(fmt, min_tokens)
        layer.load()
        return layer

    def test_matches_bf16_within_format_error(self):
        for fmt in FORMATS:
            for name, n, k in SHAPES:
                with self.subTest(fmt=fmt, shape=name):
                    layer = self._layer(n, k, 1024, fmt)
                    x = torch.randn(2048, k, dtype=torch.bfloat16, device="cuda")
                    ref = torch.nn.functional.linear(x.float(), layer.weight.float())
                    got = layer.quant_method.apply(layer, x)
                    self.assertEqual(got.shape, (x.shape[0], n))
                    self.assertEqual(got.dtype, torch.bfloat16)
                    self.assertLess(_rel_err(got, ref), _MAX_REL_ERR[fmt])

    def test_mxfp6_is_markedly_more_accurate_than_mxfp4(self):
        """The premise of defaulting to MXFP6: much less error at the same rate.

        MXFP6 is E2M3 against MXFP4's E2M1 on an identical per-32 E8M0 block
        scale, so two extra mantissa bits should show up as a large, not
        marginal, error reduction. Anything under 2x here means a layout or
        rotation problem rather than a precision difference.
        """
        for name, n, k in SHAPES:
            with self.subTest(name):
                x = torch.randn(2048, k, dtype=torch.bfloat16, device="cuda")
                errs = {}
                for fmt in ("mxfp4", "mxfp6"):
                    layer = self._layer(n, k, 1024, fmt)
                    ref = torch.nn.functional.linear(x.float(), layer.weight.float())
                    errs[fmt] = _rel_err(layer.quant_method.apply(layer, x), ref)
                self.assertLess(errs["mxfp6"] * 2, errs["mxfp4"], errs)

    def test_below_threshold_is_untouched_bf16(self):
        """Decode must be bit-identical to the BF16 path it would have taken."""
        _, n, k = SHAPES[0]
        for fmt in FORMATS:
            layer = self._layer(n, k, 1024, fmt)
            for tokens in (1, 128, 1023):
                with self.subTest(fmt=fmt, tokens=tokens):
                    x = torch.randn(tokens, k, dtype=torch.bfloat16, device="cuda")
                    expect = UnquantizedLinearMethod().apply(layer, x)
                    torch.testing.assert_close(
                        layer.quant_method.apply(layer, x), expect, rtol=0, atol=0
                    )

    def test_threshold_is_inclusive(self):
        _, n, k = SHAPES[1]
        for fmt in FORMATS:
            with self.subTest(fmt=fmt):
                layer = self._layer(n, k, 512, fmt)
                x = torch.randn(512, k, dtype=torch.bfloat16, device="cuda")
                bf16 = UnquantizedLinearMethod().apply(layer, x)
                got = layer.quant_method.apply(layer, x)
                self.assertFalse(torch.equal(got, bf16), "512 tokens should quantize")

    def test_bias_is_applied(self):
        _, n, k = SHAPES[1]
        for fmt in FORMATS:
            with self.subTest(fmt=fmt):
                layer = self._layer(n, k, 1024, fmt)
                x = torch.randn(2048, k, dtype=torch.bfloat16, device="cuda")
                bias = torch.randn(n, dtype=torch.bfloat16, device="cuda")
                torch.testing.assert_close(
                    layer.quant_method.apply(layer, x, bias),
                    layer.quant_method.apply(layer, x) + bias,
                    rtol=2e-2,
                    atol=2e-2,
                )

    def test_unpackable_weight_stays_bf16(self):
        """A shape the format cannot take must degrade to BF16, not fail to load."""
        # 40 output rows is neither a multiple of MXFP4's 16-row preshuffle nor
        # of MXFP6's 256-row tile.
        for fmt in FORMATS:
            with self.subTest(fmt=fmt):
                weight = torch.randn(40, 4096, dtype=torch.bfloat16, device="cuda")
                layer = _Layer(weight)
                layer.quant_method = MxDenseLinearMethod(fmt, 1)
                layer.load()
                x = torch.randn(2048, 4096, dtype=torch.bfloat16, device="cuda")
                torch.testing.assert_close(
                    layer.quant_method.apply(layer, x),
                    UnquantizedLinearMethod().apply(layer, x),
                    rtol=0,
                    atol=0,
                )

    def test_non_2d_input_falls_back(self):
        _, n, k = SHAPES[1]
        for fmt in FORMATS:
            with self.subTest(fmt=fmt):
                layer = self._layer(n, k, 1, fmt)
                x = torch.randn(4, 512, k, dtype=torch.bfloat16, device="cuda")
                torch.testing.assert_close(
                    layer.quant_method.apply(layer, x),
                    UnquantizedLinearMethod().apply(layer, x),
                    rtol=0,
                    atol=0,
                )

    def test_unknown_format_is_rejected(self):
        with self.assertRaises(ValueError):
            mx_dense_supported("mxfp3")


class _FakeQuantConfig:
    """Stands in for QuarkConfig, which is all dense_mx.register() needs."""


@unittest.skipUnless(_RUNNABLE, "requires HIP gfx950 with aiter")
class TestQwen35DenseMxPolicy(CustomTestCase):
    """The Qwen3.5 policy decides by module-name substring, so collisions matter.

    `o_proj` must not match `out_proj` or `in_proj_qkvz`, and the layers whose
    measurements said "leave it alone" must stay bf16 -- in particular
    `shared_expert.down_proj`, which is claimed by the separate dense-FP8 path,
    and the 64-column `in_proj_ba`.
    """

    CONVERT = (
        "model.layers.0.linear_attn.in_proj_qkvz",
        "model.layers.0.linear_attn.out_proj",
        "model.layers.3.self_attn.qkv_proj",
        "model.layers.3.self_attn.o_proj",
    )
    LEAVE_BF16 = (
        "model.layers.0.linear_attn.in_proj_ba",
        "model.layers.0.linear_attn.in_proj_b",
        "model.layers.0.linear_attn.in_proj_a",
        "model.layers.0.linear_attn.conv1d",
        "model.layers.1.mlp.gate",
        "model.layers.1.mlp.shared_expert.down_proj",
        "model.layers.1.mlp.shared_expert.gate_up_proj",
        "model.layers.1.mlp.shared_expert_gate",
        "model.layers.1.mlp.experts.0.down_proj",
        "lm_head",
        "model.embed_tokens",
    )

    def _policy_config(self):
        cfg = _FakeQuantConfig()
        dense_mx.register(
            cfg,
            include=qwen3_5_dense_mx._INCLUDE,
            exclude=qwen3_5_dense_mx._EXCLUDE,
            min_output_size=qwen3_5_dense_mx._MIN_OUTPUT_SIZE,
            min_tokens=qwen3_5_dense_mx._MIN_TOKENS,
        )
        return cfg

    def _wide_layer(self):
        layer = torch.nn.Module()
        layer.output_size_per_partition = 4096
        return layer

    def _routed(self, cfg, prefix, layer):
        """linear_method_for with the server flag forced on."""
        real = dense_mx._flag_enabled
        dense_mx._flag_enabled = lambda: True
        try:
            return dense_mx.linear_method_for(cfg, prefix, layer)
        finally:
            dense_mx._flag_enabled = real

    def test_converts_the_four_measured_projections(self):
        cfg = self._policy_config()
        for prefix in self.CONVERT:
            with self.subTest(prefix):
                method = self._routed(cfg, prefix, self._wide_layer())
                self.assertIsInstance(method, MxDenseLinearMethod)
                self.assertEqual(method.min_tokens, qwen3_5_dense_mx._MIN_TOKENS)

    def test_reports_which_projections_converted(self):
        """A server log must name them, so an inert arm cannot read as a null result."""
        dense_mx._converted.clear()
        cfg = self._policy_config()
        for prefix in self.CONVERT:
            self._routed(cfg, prefix, self._wide_layer())
        self.assertEqual(
            dense_mx._converted, set(qwen3_5_dense_mx._INCLUDE), dense_mx._converted
        )
        # A declined layer must not appear.
        self._routed(cfg, "model.layers.1.mlp.gate", self._wide_layer())
        self.assertNotIn("mlp.gate", dense_mx._converted)

    def test_leaves_everything_else_bf16(self):
        cfg = self._policy_config()
        for prefix in self.LEAVE_BF16:
            with self.subTest(prefix):
                self.assertIsNone(self._routed(cfg, prefix, self._wide_layer()))

    def test_narrow_layer_is_declined_even_if_the_name_matches(self):
        cfg = self._policy_config()
        narrow = torch.nn.Module()
        narrow.output_size_per_partition = 64
        self.assertIsNone(self._routed(cfg, "model.layers.0.self_attn.o_proj", narrow))

    def test_flag_off_leaves_every_layer_bf16(self):
        cfg = self._policy_config()
        for prefix in self.CONVERT:
            with self.subTest(prefix):
                # No forced flag: an uninitialized runtime context reads as off.
                self.assertIsNone(
                    dense_mx.linear_method_for(cfg, prefix, self._wide_layer())
                )

    def test_unregistered_config_leaves_every_layer_bf16(self):
        for prefix in self.CONVERT:
            with self.subTest(prefix):
                self.assertIsNone(
                    self._routed(_FakeQuantConfig(), prefix, self._wide_layer())
                )


if __name__ == "__main__":
    unittest.main()
