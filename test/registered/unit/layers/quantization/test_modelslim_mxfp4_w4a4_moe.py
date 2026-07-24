"""CPU regression tests for ModelSlim / online MXFP4 W4A4 MoE support."""

import unittest

import torch

from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUMXFP4W4A4MoEMethod,
)
from sglang.srt.hardware_backend.npu.quantization.online_moe_methods import (
    NPUMXFP4W4A4FusedMoEMethod,
)
from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
from sglang.srt.layers.quantization.modelslim.schemes import (
    ModelSlimMXFP4W4A4MoEScheme,
)
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestModelSlimMXFP4W4A4MoE(CustomTestCase):
    def test_online_method_reuses_generic_fused_moe_plumbing(self):
        # The online entry must stay a subclass of UnquantizedFusedMoEMethod:
        # that membership is what puts it on FusedMoE's flashinfer shard-swap
        # list and keeps the Ascend runner path. Dropping it silently reroutes
        # weight loading and degrades output.
        self.assertTrue(
            issubclass(NPUMXFP4W4A4FusedMoEMethod, UnquantizedFusedMoEMethod)
        )

    def test_config_resolves_w4a4_mxfp4_moe_scheme(self):
        # Guards the moe_quant_schemes registration: a W4A4_MXFP4 checkpoint must
        # map both projection groups to the W4A4 MoE scheme, not fall through.
        prefix = "model.layers.0.mlp.experts"
        quant_config = ModelSlimConfig(
            {
                f"{prefix}.0.gate_proj.weight": "W4A4_MXFP4",
                f"{prefix}.0.up_proj.weight": "W4A4_MXFP4",
                f"{prefix}.0.down_proj.weight": "W4A4_MXFP4",
            }
        )

        w13_scheme, w2_scheme = quant_config.get_moe_scheme(torch.nn.Module(), prefix)

        self.assertIsInstance(w13_scheme, ModelSlimMXFP4W4A4MoEScheme)
        self.assertIsInstance(w2_scheme, ModelSlimMXFP4W4A4MoEScheme)
        self.assertEqual(w13_scheme.weight_prefix, "w13")
        self.assertEqual(w2_scheme.weight_prefix, "w2")

    def test_creates_packed_weights_scales_and_empty_offsets(self):
        # Packed fp4 halves the K dim (uint8 [E, N, K//2]) and the E8M0 scale is
        # one uint8 per 32-element block; the offset must be registered as None
        # (W4A4_MXFP4 has no zero point) or ModelSlimFusedMoEMethod.apply raises
        # AttributeError building AscendQuantInfo.
        layer = torch.nn.Module()
        w13_scheme = ModelSlimMXFP4W4A4MoEScheme({}, "w13")
        w2_scheme = ModelSlimMXFP4W4A4MoEScheme({}, "w2")

        for scheme in (w13_scheme, w2_scheme):
            scheme.create_weights(
                layer=layer,
                num_experts=2,
                hidden_size=64,
                intermediate_size_per_partition=96,
            )

        self.assertEqual(layer.w13_weight.shape, (2, 192, 32))
        self.assertEqual(layer.w13_weight_scale.shape, (2, 192, 2))
        self.assertIsNone(layer.w13_weight_offset)
        self.assertEqual(layer.w2_weight.shape, (2, 64, 48))
        self.assertEqual(layer.w2_weight_scale.shape, (2, 64, 3))
        self.assertIsNone(layer.w2_weight_offset)

    def test_w13_kernel_constructs_without_touching_npu_ops(self):
        # The fp4 activation quantiser must be built lazily: resolving it (and the
        # torch.ops.npu op it holds) at construction would raise on CPU, breaking
        # this very test and any CPU import of the offline scheme. Constructing
        # the w13 kernel here is the guard against a regression to eager build.
        kernel = NPUMXFP4W4A4MoEMethod("w13")
        self.assertIsNone(kernel._hidden_states_quantizer)
        self.assertEqual(kernel.weight_prefix, "w13")

        with self.assertRaises(ValueError):
            NPUMXFP4W4A4MoEMethod("w1")  # only w13 / w2 are valid groups

    def test_w2_apply_rejects_non_w2_prefix(self):
        # gmm1 is fused (apply_fused_gmm1_swiglu); routing a w13 group through
        # apply() would silently drop the returned activation scale, so apply()
        # must reject anything but w2.
        kernel = NPUMXFP4W4A4MoEMethod("w2")
        with self.assertRaises(ValueError):
            kernel.apply(
                quant_info=None,
                hidden_states=None,
                expert_tokens=None,
                pertoken_scale=None,
                output_dtype=torch.bfloat16,
                weight_prefix="w13",
                group_list_type=1,
            )


if __name__ == "__main__":
    unittest.main()
