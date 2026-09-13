"""CPU tests for MXFP4 MoE checkpoint and runtime parameter names."""

import unittest
from unittest import mock

import torch

from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMxfp4CheckpointWeightNames(CustomTestCase):
    @staticmethod
    def _make_method(checkpoint_weight_suffix: str) -> Mxfp4MoEMethod:
        method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
        method.checkpoint_weight_suffix = checkpoint_weight_suffix
        return method

    def test_packed_mode_registers_checkpoint_names(self):
        method = self._make_method("weight_packed")
        method.use_marlin = False
        method.use_mega_moe = False
        method.use_deep_gemm = False
        method.use_flashinfer = False
        method._fi_kernel = None
        layer = torch.nn.Module()
        layer.num_local_experts = 2

        with (
            mock.patch(
                "sglang.srt.layers.quantization.mxfp4.is_sm100_supported",
                return_value=False,
            ),
            mock.patch("sglang.srt.layers.quantization.mxfp4._use_aiter", False),
            mock.patch(
                "sglang.srt.layers.quantization.mxfp4.has_triton_kernels", False
            ),
        ):
            method.create_weights(
                layer=layer,
                num_experts=2,
                hidden_size=64,
                intermediate_size_per_partition=64,
                params_dtype=torch.bfloat16,
            )

        self.assertIn("w13_weight_packed", layer._parameters)
        self.assertIn("w2_weight_packed", layer._parameters)
        self.assertNotIn("w13_weight", layer._parameters)
        self.assertNotIn("w2_weight", layer._parameters)

    def test_compressed_tensors_selects_packed_checkpoint_names(self):
        config = CompressedTensorsConfig(
            target_scheme_map={},
            ignore=[],
            quant_format="mxfp4-quantized",
            sparsity_scheme_map={},
            sparsity_ignore_list=[],
        )
        layer = FusedMoE.__new__(FusedMoE)
        torch.nn.Module.__init__(layer)

        with mock.patch(
            "sglang.srt.layers.quantization.mxfp4.Mxfp4MoEMethod"
        ) as method_cls:
            config.get_quant_method(layer, "model.layers.0.mlp.experts")

        method_cls.assert_called_once_with(
            prefix="model.layers.0.mlp.experts",
            checkpoint_weight_suffix="weight_packed",
        )

    def test_packed_parameters_are_rebound_without_copying(self):
        method = self._make_method("weight_packed")
        layer = torch.nn.Module()
        w13 = torch.nn.Parameter(torch.tensor([1], dtype=torch.uint8), False)
        w2 = torch.nn.Parameter(torch.tensor([2], dtype=torch.uint8), False)
        loader = object()
        w13.weight_loader = loader
        layer.register_parameter("w13_weight_packed", w13)
        layer.register_parameter("w2_weight_packed", w2)

        method.prepare_weights_for_post_load(layer)

        self.assertIs(layer.w13_weight, w13)
        self.assertIs(layer.w2_weight, w2)
        self.assertIs(layer.w13_weight.weight_loader, loader)
        self.assertFalse(hasattr(layer, "w13_weight_packed"))
        self.assertFalse(hasattr(layer, "w2_weight_packed"))

        # Finalization can be called again without changing the parameters.
        method.prepare_weights_for_post_load(layer)
        self.assertIs(layer.w13_weight, w13)
        self.assertIs(layer.w2_weight, w2)

    def test_packed_main_weights_are_not_treated_as_side_tensors(self):
        layer = FusedMoE.__new__(FusedMoE)
        torch.nn.Module.__init__(layer)
        layer.register_parameter(
            "w13_weight_packed",
            torch.nn.Parameter(torch.zeros(2, 4, 2, dtype=torch.uint8), False),
        )
        layer.register_parameter(
            "w2_weight_packed",
            torch.nn.Parameter(torch.zeros(2, 2, 2, dtype=torch.uint8), False),
        )
        layer.register_parameter(
            "w13_weight_scale",
            torch.nn.Parameter(torch.zeros(2, 4, 1, dtype=torch.uint8), False),
        )

        names = {name for name, _ in layer.named_per_expert_tensors(2)}

        self.assertNotIn("w13_weight_packed", names)
        self.assertNotIn("w2_weight_packed", names)
        self.assertIn("w13_weight_scale", names)

    def test_native_mxfp4_keeps_runtime_names(self):
        method = self._make_method("weight")
        layer = torch.nn.Module()
        w13 = torch.nn.Parameter(torch.tensor([1], dtype=torch.uint8), False)
        w2 = torch.nn.Parameter(torch.tensor([2], dtype=torch.uint8), False)
        layer.register_parameter("w13_weight", w13)
        layer.register_parameter("w2_weight", w2)

        method.prepare_weights_for_post_load(layer)

        self.assertIs(layer.w13_weight, w13)
        self.assertIs(layer.w2_weight, w2)


if __name__ == "__main__":
    unittest.main()
