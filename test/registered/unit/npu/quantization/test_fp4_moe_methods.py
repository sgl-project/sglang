import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")

# Load the quantization package first so `base_config`, `moe_methods`, and
# `linear_method_npu` initialize in dependency order. Importing `fp4_moe_methods`
# (or `linear_method_npu`) directly from a cold process triggers a circular
# import: linear_method_npu -> base_config -> quantization/__init__ ->
# gguf/unquant/gptq_moe -> moe_methods -> linear_method_npu (partially
# initialized, `_get_float8_e8m0fnu_dtype` not yet defined). Initializing the
# package first mirrors how the engine loads quantization at model-config time.
import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.quantization import fp4_moe_methods
from sglang.srt.hardware_backend.npu.quantization.fp4_moe_methods import (
    NPUW4A4Fp4MoEMethod,
    _apply_swiglu_limit_npu,
    _pair_pack_mxfp_act_scale,
    _reshape_mxfp4_scale_for_npu,
    _w4a8_mxfp_gmm,
    npu_apply_without_routing_weights_w4a4_mxfp,
)


class TestApplySwiGLULimitNpu(unittest.TestCase):
    def test_clamps_gate_and_up_asymmetrically(self):
        # DeepSeek-V4 clamps gate (first half) to <= limit but only the upper
        # bound, while up (second half) is clamped symmetrically to [-limit, limit].
        # A regression that swapped these would silently change expert activations.
        gate_up = torch.tensor([[8.0, -9.0, 9.0, -9.0]])
        _apply_swiglu_limit_npu(gate_up, 7.0)
        self.assertTrue(torch.equal(gate_up, torch.tensor([[7.0, -9.0, 7.0, -7.0]])))

    def test_noop_when_limit_none(self):
        gate_up = torch.tensor([[8.0, -9.0]])
        _apply_swiglu_limit_npu(gate_up, None)
        self.assertTrue(torch.equal(gate_up, torch.tensor([[8.0, -9.0]])))

    def test_noop_when_limit_nonpositive(self):
        gate_up = torch.tensor([[8.0, -9.0]])
        _apply_swiglu_limit_npu(gate_up, 0.0)
        self.assertTrue(torch.equal(gate_up, torch.tensor([[8.0, -9.0]])))


class TestReshapeMxfp4ScaleForNpu(unittest.TestCase):
    def test_packs_scale_to_gmm_layout(self):
        # [E, N, K/32] -> [E, K/64, N, 2] is the packed-pair layout the GMM reads;
        # getting the transpose axis wrong silently dequantizes with the wrong scale.
        scale = torch.arange(8, dtype=torch.uint8).view(1, 2, 4)
        out = _reshape_mxfp4_scale_for_npu(scale)
        self.assertEqual(tuple(out.shape), (1, 2, 2, 2))
        self.assertTrue(torch.equal(out, scale.view(1, 2, 2, 2).transpose(1, 2)))

    def test_rejects_odd_k_dim(self):
        with self.assertRaises(ValueError):
            _reshape_mxfp4_scale_for_npu(torch.zeros(1, 2, 3, dtype=torch.uint8))


class TestPairPackMxfpActScale(unittest.TestCase):
    def test_packs_as_view(self):
        # The GMM expects a pair-packed *view* of the per-token scale, not a copy;
        # materializing a copy here would break the kernel's aliasing contract.
        flat = torch.arange(8).view(2, 4)
        packed = _pair_pack_mxfp_act_scale(flat)
        self.assertEqual(tuple(packed.shape), (2, 2, 2))
        self.assertEqual(packed.data_ptr(), flat.data_ptr())

    def test_rejects_odd_scale_dim(self):
        with self.assertRaises(ValueError):
            _pair_pack_mxfp_act_scale(torch.zeros(2, 3))


class TestW4A8MxfpGmmInputScale(unittest.TestCase):
    def setUp(self):
        self.input = torch.randn(2, 64)
        self.input_scale = torch.ones(2, 1, 2)
        self.weight = torch.empty(2, 64, 32, dtype=torch.uint8)
        self.weight_scale = torch.ones(2, 1, 32, 2, dtype=torch.uint8)
        self.group_list = torch.tensor([1, 1], dtype=torch.int64)

    def _call_gmm(self, input_scale):
        return _w4a8_mxfp_gmm(
            input=self.input,
            input_scale=input_scale,
            weight=self.weight,
            weight_scale=self.weight_scale,
            group_list_type=1,
            group_list=self.group_list,
            output_dtype=torch.bfloat16,
        )

    def test_supplied_scale_skips_dynamic_quant(self):
        expected = torch.randn(2, 32)
        with (
            patch.object(
                torch.ops.npu, "npu_dynamic_mx_quant", create=True
            ) as dynamic_quant,
            patch.object(
                torch.ops.npu,
                "npu_grouped_matmul",
                return_value=[expected],
                create=True,
            ) as grouped_matmul,
        ):
            output = self._call_gmm(self.input_scale)

        dynamic_quant.assert_not_called()
        self.assertIs(output, expected)
        call_kwargs = grouped_matmul.call_args.kwargs
        self.assertIs(call_kwargs["per_token_scale"][0], self.input_scale)
        self.assertIs(call_kwargs["group_list"], self.group_list)

    def test_missing_scale_uses_dynamic_quant(self):
        quantized = torch.empty(2, 64, dtype=torch.float8_e4m3fn)
        quantized_scale = torch.ones(2, 1, 2)
        expected = torch.randn(2, 32)
        with (
            patch.object(
                torch.ops.npu,
                "npu_dynamic_mx_quant",
                return_value=(quantized, quantized_scale),
                create=True,
            ) as dynamic_quant,
            patch.object(
                torch.ops.npu,
                "npu_grouped_matmul",
                return_value=[expected],
                create=True,
            ) as grouped_matmul,
        ):
            output = self._call_gmm(None)

        dynamic_quant.assert_called_once()
        self.assertIs(output, expected)
        self.assertIs(
            grouped_matmul.call_args.kwargs["per_token_scale"][0], quantized_scale
        )


class TestW4A8MxfpGmmChain(unittest.TestCase):
    def test_applies_swiglu_limit_before_swiglu(self):
        # The clamp must be wired in *before* npu_swiglu using the layer's
        # configured swiglu_limit; dropping the clamp (or applying it after)
        # silently changes routed-expert output on near-limit activations.
        gate_up = torch.tensor([[8.0, -9.0, 9.0, -9.0]])
        activated = torch.randn(1, 2)
        expected = torch.randn(1, 2)
        layer = SimpleNamespace(
            w13_weight=MagicMock(),
            w13_weight_scale_inv=MagicMock(),
            w2_weight=MagicMock(),
            w2_weight_scale_inv=MagicMock(),
            moe_runner_config=SimpleNamespace(swiglu_limit=7.0),
        )

        with (
            patch.object(
                fp4_moe_methods, "w4a4_mxfp_gmm_npu", side_effect=[gate_up, expected]
            ) as gmm,
            patch.object(
                torch.ops.npu, "npu_swiglu", return_value=activated, create=True
            ) as swiglu,
        ):
            output = npu_apply_without_routing_weights_w4a4_mxfp(
                layer,
                torch.randn(1, 4),
                torch.ones(1, 1, 2),
                group_list_type=1,
                group_list=torch.tensor([1], dtype=torch.int64),
                output_dtype=torch.bfloat16,
            )

        self.assertIs(output, expected)
        self.assertTrue(
            torch.equal(
                swiglu.call_args.args[0], torch.tensor([[7.0, -9.0, 7.0, -7.0]])
            )
        )
        self.assertIs(gmm.call_args_list[1].kwargs["input"], activated)


class TestProcessWeightsAfterLoadingZeroScale(unittest.TestCase):
    @staticmethod
    def _method():
        return NPUW4A4Fp4MoEMethod(fp8_method=MagicMock(), prefix="test")

    def test_raises_when_w13_scales_never_loaded(self):
        # An all-zero scale is the signature of a checkpoint whose scale names
        # never matched; without this guard every routed expert computes silently
        # as zero instead of failing loudly.
        layer = SimpleNamespace(
            w13_weight_scale_inv=torch.nn.Parameter(
                torch.zeros(2, 2, 4, dtype=torch.uint8)
            ),
            w2_weight_scale_inv=torch.nn.Parameter(
                torch.zeros(2, 2, 4, dtype=torch.uint8)
            ),
        )
        with self.assertRaises(RuntimeError):
            self._method().process_weights_after_loading(layer)

    def test_raises_when_w2_scales_never_loaded(self):
        layer = SimpleNamespace(
            w13_weight_scale_inv=torch.nn.Parameter(
                torch.ones(2, 2, 4, dtype=torch.uint8)
            ),
            w2_weight_scale_inv=torch.nn.Parameter(
                torch.zeros(2, 2, 4, dtype=torch.uint8)
            ),
        )
        with self.assertRaises(RuntimeError):
            self._method().process_weights_after_loading(layer)


if __name__ == "__main__":
    unittest.main()
