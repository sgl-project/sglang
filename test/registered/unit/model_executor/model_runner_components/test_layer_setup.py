"""Unit tests for model-runner layer discovery."""

import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.model_runner_components.layer_setup import (
    _assert_pp_mtp_compat,
    compute_attention_and_moe_layers,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestComputeAttentionAndMoeLayers(unittest.TestCase):
    def test_deepseek_mla_registers_mha_companion(self):
        attn_mqa = SimpleNamespace()
        attn_mha = SimpleNamespace()
        layer_model = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(attn_mqa=attn_mqa, attn_mha=attn_mha)
                )
            ]
        )

        attention_layers, _, _, _, mha_companion_layers = (
            compute_attention_and_moe_layers(layer_model)
        )

        self.assertEqual(attention_layers, [attn_mqa])
        self.assertEqual(mha_companion_layers, [attn_mha])
        self.assertNotIn("_pcg_mha_companion", vars(attn_mqa))

    def test_pipeline_placeholders_preserve_global_layer_ids(self):
        local_attention = SimpleNamespace()
        layer_model = SimpleNamespace(
            layers=[SimpleNamespace(), SimpleNamespace()]
            + [SimpleNamespace(self_attn=SimpleNamespace(attn=local_attention))]
        )

        attention_layers, _, _, _, mha_companion_layers = (
            compute_attention_and_moe_layers(layer_model)
        )

        self.assertEqual(attention_layers, [None, None, local_attention])
        self.assertEqual(mha_companion_layers, [None, None, None])


class TestPipelineParallelMtpCompatibility(unittest.TestCase):
    def test_qwen35_allows_stage_local_target_layers(self):
        _assert_pp_mtp_compat(
            model_architecture="Qwen3_5MoeForCausalLM",
            model_has_mtp_layers=True,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            num_effective_layers=15,
            model_num_layers=60,
        )

    def test_other_models_keep_existing_guard(self):
        with self.assertRaisesRegex(AssertionError, "PP is not compatible with MTP"):
            _assert_pp_mtp_compat(
                model_architecture="LlamaForCausalLM",
                model_has_mtp_layers=True,
                spec_algorithm=SpeculativeAlgorithm.EAGLE,
                num_effective_layers=15,
                model_num_layers=60,
            )


if __name__ == "__main__":
    unittest.main()
