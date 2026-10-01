"""Compressed-tensors NVFP4 contracts for Beam."""

import unittest
from unittest.mock import patch

import torch
from compressed_tensors.quantization import preset_name_to_scheme

from sglang.srt.configs.beam import BeamConfig
from sglang.srt.layers.moe.topk import TopKOutputFormat, biased_grouped_topk_impl
from sglang.srt.layers.moe.utils import (
    MoeA2ABackend,
    MoeRunnerBackend,
    RoutingMethodType,
)
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.models.beam import BeamForCausalLM, BeamMoE
from sglang.srt.runtime_context import (
    get_context,
    get_flags,
    get_parallel,
    override_platform,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def nvfp4_quant_config():
    return CompressedTensorsConfig.from_config(
        {
            "format": "nvfp4-pack-quantized",
            "config_groups": {
                "NVFP4": preset_name_to_scheme("NVFP4", targets=["Linear"]).model_dump()
            },
            "ignore": [],
        }
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
        for quant, layer_id, expected_scale in (
            (nvfp4_quant_config(), 3, config.routed_scaling_factor),
            (modelopt_quant_config(), 3, config.routed_scaling_factor),
            (modelopt_quant_config(), 0, 1.0),  # Excluded experts stay unquantized.
        ):
            with (
                self.subTest(quant=quant, layer_id=layer_id),
                get_context().override_server_args(),
                get_flags().moe.override(
                    runner_backend=MoeRunnerBackend.FLASHINFER_TRTLLM,
                    a2a_backend=MoeA2ABackend.NONE,
                ),
                get_parallel().override(
                    tp_size=1,
                    tp_rank=0,
                    moe_tp_size=1,
                    moe_tp_rank=0,
                    moe_ep_size=1,
                    moe_ep_rank=0,
                ),
                override_platform(is_blackwell=True),
                patch(
                    "sglang.srt.layers.moe.topk.biased_grouped_topk",
                    wraps=biased_grouped_topk_impl._torchdynamo_orig_callable,
                ),
            ):
                model = BeamMoE(
                    config,
                    layer_id,
                    prefix=f"model.layers.{layer_id}.mlp",
                    quant_config=quant,
                )
                with torch.no_grad():
                    model.gate.e_score_correction_bias.zero_()
                hidden = torch.ones(1, config.hidden_size)
                logits = torch.linspace(1, -1, config.num_experts)[None]
                self.assertEqual(
                    model.topk.forward_cuda(hidden, logits).format,
                    TopKOutputFormat.BYPASSED,
                )
                # One-token compile materializes routing; the routed kernel adds no scale.
                model.topk.enter_torch_compile(num_tokens=1)
                try:
                    routing = model.topk(hidden, logits)
                finally:
                    model.topk.leave_torch_compile()
                self.assertEqual(
                    model.experts.routing_method_type, RoutingMethodType.DeepSeekV3
                )
                torch.testing.assert_close(
                    routing.topk_weights.sum(-1),
                    torch.tensor([expected_scale]),
                )

    def test_nvfp4_scheme_rejects_incompatible_runner(self):
        config = BeamConfig(hidden_size=16, num_experts=16, moe_intermediate_size=16)
        with (
            get_context().override_server_args(),
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.TRITON,
                a2a_backend=MoeA2ABackend.NONE,
            ),
            get_parallel().override(
                tp_size=1,
                tp_rank=0,
                moe_tp_size=1,
                moe_tp_rank=0,
                moe_ep_size=1,
                moe_ep_rank=0,
            ),
            override_platform(is_blackwell=True),
            self.assertRaisesRegex(ValueError, "NVFP4 MoE requires"),
        ):
            BeamMoE(config, 0, quant_config=nvfp4_quant_config())


if __name__ == "__main__":
    unittest.main()
