"""Native MXFP4 expert parameter routing for Kimi-K3."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.models.kimi_k3 import KimiK3LinearForCausalLM  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestKimiK3Humming(CustomTestCase):
    def test_humming_rejects_megamoe_a2a_backend(self):
        from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
        from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

        with (
            patch(
                "sglang.srt.layers.quantization.mxfp4.get_moe_runner_backend",
                return_value=MoeRunnerBackend.HUMMING,
            ),
            patch(
                "sglang.srt.layers.quantization.mxfp4.get_moe_a2a_backend",
                return_value=MoeA2ABackend.MEGAMOE,
            ),
            patch(
                "sglang.srt.layers.quantization.mxfp4.get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(flashinfer_mxfp4_moe_precision="default")
                ),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "humming.*megamoe"):
                Mxfp4MoEMethod(prefix="layers.0.mlp.experts")

    @staticmethod
    def _make_model():
        model = KimiK3LinearForCausalLM.__new__(KimiK3LinearForCausalLM)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(
            linear_attn_config={},
            dsa_config=None,
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
                    patch.object(model, "post_load_weights"),
                ):
                    model.load_weights(self._checkpoint_weights())

                for name, expected_count in zip(parameter_names, expected_call_counts):
                    loader = parameters[name].weight_loader
                    self.assertEqual(loader.call_count, expected_count)
                    for call in loader.call_args_list:
                        self.assertEqual(call.args[2], name)


if __name__ == "__main__":
    unittest.main()
