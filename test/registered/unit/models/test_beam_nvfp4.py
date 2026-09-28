"""Compressed-tensors NVFP4 contracts for Beam."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.configs.beam import BeamConfig
from sglang.srt.layers.moe.topk import TopKOutputFormat, biased_grouped_topk_impl
from sglang.srt.layers.moe.utils import (
    MoeA2ABackend,
    MoeRunnerBackend,
    RoutingMethodType,
)
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.models.beam import BeamForCausalLM, BeamMoE
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def nvfp4_quant_config():
    return SimpleNamespace(
        get_name=lambda: "compressed_tensors", quant_format="nvfp4-pack-quantized"
    )


def modelopt_quant_config(quant_algo="NVFP4"):
    # Mirrors the `ignore` globs of a ModelOpt experts-only Beam export.
    return ModelOptFp4Config.from_config(
        {
            "quant_method": "modelopt",
            "quant_algo": quant_algo,
            "group_size": 16,
            "ignore": [
                "lm_head",
                "model.embed_tokens",
                "model.layers.0*",
                "model.layers.1.*",
                "model.layers.3.mlp.gate",
                "model.layers.3.self_attn*",
            ],
        }
    )


class TestBeamNvfp4(CustomTestCase):
    def test_nvfp4_config_reaches_model_allocation(self):
        with (
            get_context().override_server_args(attention_backend="trtllm_mha"),
            patch(
                "sglang.srt.models.beam.BeamModel",
                side_effect=RuntimeError("allocation reached"),
            ),
            self.assertRaisesRegex(RuntimeError, "allocation reached"),
        ):
            BeamForCausalLM(BeamConfig(), quant_config=nvfp4_quant_config())

    def test_modelopt_nvfp4_config_reaches_model_allocation(self):
        with (
            get_context().override_server_args(attention_backend="trtllm_mha"),
            patch(
                "sglang.srt.models.beam.BeamModel",
                side_effect=RuntimeError("allocation reached"),
            ),
            self.assertRaisesRegex(RuntimeError, "allocation reached"),
        ):
            BeamForCausalLM(BeamConfig(), quant_config=modelopt_quant_config())

    def test_other_quantization_is_rejected(self):
        quant = SimpleNamespace(get_name=lambda: "awq", quant_format=None)
        with (
            get_context().override_server_args(attention_backend="trtllm_mha"),
            self.assertRaisesRegex(NotImplementedError, "W4A4 NVFP4"),
        ):
            BeamForCausalLM(BeamConfig(), quant_config=quant)

    def test_modelopt_variants_without_nvfp4_activations_are_rejected(self):
        for quant in (
            modelopt_quant_config("W4A16_NVFP4"),
            modelopt_quant_config("NVFP4_AWQ"),
            ModelOptFp4Config(),
        ):
            with (
                self.subTest(quant=quant),
                get_context().override_server_args(attention_backend="trtllm_mha"),
                self.assertRaisesRegex(NotImplementedError, "W4A4 NVFP4"),
            ):
                BeamForCausalLM(BeamConfig(), quant_config=quant)

    def test_modelopt_export_keeps_non_expert_modules_in_source_precision(self):
        quant = modelopt_quant_config()
        quant.packed_modules_mapping = BeamForCausalLM.packed_modules_mapping
        for prefix in (
            "lm_head",
            "model.layers.0.mlp.gate_up_proj",
            "model.layers.1.mlp.down_proj",
            "model.layers.3.mlp.gate",
            "model.layers.3.self_attn.qkv_proj",
            "model.layers.3.self_attn.gate_proj",
            "model.layers.3.self_attn.o_proj",
        ):
            with self.subTest(prefix=prefix):
                self.assertTrue(quant.is_layer_excluded(prefix))
        self.assertFalse(quant.is_layer_excluded("model.layers.3.mlp.experts"))
        self.assertFalse(quant.is_layer_excluded("model.layers.10.mlp.experts"))

    def test_nvfp4_routes_with_deepseek_grouping_and_one_scale(self):
        config = BeamConfig(hidden_size=16, num_experts=16, moe_intermediate_size=16)
        quant = nvfp4_quant_config()
        factory = Mock(return_value=torch.nn.Identity())
        with (
            get_context().override_server_args(
                moe_runner_backend="flashinfer_trtllm", moe_a2a_backend="none"
            ),
            patch(
                "sglang.srt.models.beam.get_moe_runner_backend",
                return_value=MoeRunnerBackend.FLASHINFER_TRTLLM,
            ),
            patch(
                "sglang.srt.models.beam.get_moe_a2a_backend",
                return_value=MoeA2ABackend.NONE,
            ),
            patch("sglang.srt.models.beam.get_moe_impl_class", return_value=factory),
            patch(
                "sglang.srt.layers.moe.topk.biased_grouped_topk",
                wraps=biased_grouped_topk_impl._torchdynamo_orig_callable,
            ),
        ):
            model = BeamMoE(config, 0, quant_config=quant)
            # One-token compile materializes routing; the routed kernel adds no scale.
            with torch.no_grad():
                model.gate.e_score_correction_bias.zero_()
            model.topk.enter_torch_compile(num_tokens=1)
            try:
                routing = model.topk(
                    torch.ones(1, config.hidden_size),
                    torch.linspace(1, -1, config.num_experts)[None],
                )
            finally:
                model.topk.leave_torch_compile()
        kwargs = factory.call_args.kwargs
        self.assertIs(kwargs["quant_config"], quant)
        self.assertIsNone(kwargs["quant_method"])
        self.assertEqual(kwargs["routing_method_type"], RoutingMethodType.DeepSeekV3)
        self.assertEqual(
            model.topk.topk_config.output_format, TopKOutputFormat.BYPASSED
        )
        self.assertEqual(model.topk.topk_config.routed_scaling_factor, 2.5)
        torch.testing.assert_close(
            routing.topk_weights.sum(-1), torch.tensor([config.routed_scaling_factor])
        )

    def test_nvfp4_requires_the_trtllm_moe_runner(self):
        config = BeamConfig(hidden_size=16, num_experts=16, moe_intermediate_size=16)
        with (
            get_context().override_server_args(
                moe_runner_backend="triton", moe_a2a_backend="none"
            ),
            patch(
                "sglang.srt.models.beam.get_moe_runner_backend",
                return_value=MoeRunnerBackend.TRITON,
            ),
            patch(
                "sglang.srt.models.beam.get_moe_a2a_backend",
                return_value=MoeA2ABackend.NONE,
            ),
            self.assertRaisesRegex(ValueError, "flashinfer_trtllm for NVFP4"),
        ):
            BeamMoE(config, 0, quant_config=nvfp4_quant_config())


if __name__ == "__main__":
    unittest.main()
