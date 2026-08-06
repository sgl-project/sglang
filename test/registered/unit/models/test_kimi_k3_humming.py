"""Minimal unit coverage for the Kimi-K3 Humming integration."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig  # noqa: E402
from sglang.srt.layers.moe.moe_runner.humming import (  # noqa: E402
    HummingRunnerCore,
)
from sglang.srt.layers.quantization.humming import HummingConfig  # noqa: E402
from sglang.srt.models.kimi_k3 import KimiK3LinearForCausalLM  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestKimiK3Humming(CustomTestCase):
    @staticmethod
    def _situ_config(**overrides):
        values = {
            "num_experts": 896,
            "num_local_experts": 56,
            "activation": "situ",
            "is_gated": True,
            "gemm1_alpha": 4.0,
            "gemm1_clamp_limit": 25.0,
            "gate_up_interleaved": False,
        }
        values.update(overrides)
        return MoeRunnerConfig(**values)

    def test_situ_config_validation(self):
        runner = HummingRunnerCore(self._situ_config())
        self.assertEqual((runner.gemm1_alpha, runner.gemm1_clamp_limit), (4.0, 25.0))

        invalid_configs = (
            ({"gemm1_alpha": None}, "requires gemm1_alpha"),
            ({"gemm1_clamp_limit": None}, "gemm1_clamp_limit"),
            ({"gemm1_alpha": 0.0}, "finite and positive"),
            ({"gemm1_alpha": float("nan")}, "finite and positive"),
            ({"gemm1_clamp_limit": 0.0}, "finite and positive"),
            ({"gemm1_clamp_limit": float("inf")}, "finite and positive"),
            ({"gate_up_interleaved": True}, "non-interleaved"),
            ({"is_gated": False}, "gated MoE"),
        )
        for overrides, message in invalid_configs:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    HummingRunnerCore(self._situ_config(**overrides))

    def test_explicit_humming_quantization_selection(self):
        self.assertEqual(
            HummingConfig.override_quantization_method(
                {"quant_method": "compressed-tensors"}, "humming"
            ),
            "humming",
        )
        self.assertIsNone(
            HummingConfig.override_quantization_method(
                {"quant_method": "mxfp4"}, "humming"
            )
        )

    @staticmethod
    def _make_model():
        model = KimiK3LinearForCausalLM.__new__(KimiK3LinearForCausalLM)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(
            linear_attn_config={},
            is_moe=True,
            num_experts=1,
            num_hidden_layers=1,
            is_linear_attn=False,
            is_kda_layer=lambda _layer_id: False,
        )
        model.model = SimpleNamespace(start_layer=0, end_layer=1)
        return model

    @staticmethod
    def _checkpoint_weights():
        prefix = "model.layers.0.mlp.experts.0"
        return [
            (f"{prefix}.w1.weight_packed", torch.empty(1)),
            (f"{prefix}.w3.weight_packed", torch.empty(1)),
            (f"{prefix}.w2.weight_packed", torch.empty(1)),
            (f"{prefix}.w1.weight_scale", torch.empty(1)),
            (f"{prefix}.w3.weight_scale", torch.empty(1)),
            (f"{prefix}.w2.weight_scale", torch.empty(1)),
        ]

    def test_dual_expert_schema_resolution(self):
        prefix = "model.layers.0.mlp.experts"
        schemas = {
            "humming-packed": (
                f"{prefix}.w13_weight_packed",
                f"{prefix}.w13_weight_scale",
                f"{prefix}.w2_weight_packed",
                f"{prefix}.w2_weight_scale",
            ),
            "native-mxfp4": (
                f"{prefix}.w13_weight",
                f"{prefix}.w13_weight_scale",
                f"{prefix}.w2_weight",
                f"{prefix}.w2_weight_scale",
            ),
        }
        expected_call_counts = (2, 2, 1, 1)

        for schema, parameter_names in schemas.items():
            with self.subTest(schema=schema):
                model = self._make_model()
                parameters = {
                    name: SimpleNamespace(weight_loader=MagicMock())
                    for name in parameter_names
                }
                with (
                    patch.object(
                        model, "named_parameters", return_value=parameters.items()
                    ),
                    patch.object(model, "post_load_weights") as post_load,
                ):
                    model.load_weights(self._checkpoint_weights())

                post_load.assert_called_once_with()
                for name, expected_count in zip(parameter_names, expected_call_counts):
                    loader = parameters[name].weight_loader
                    self.assertEqual(loader.call_count, expected_count)
                    for call in loader.call_args_list:
                        self.assertEqual(call.args[2], name)


if __name__ == "__main__":
    unittest.main()
