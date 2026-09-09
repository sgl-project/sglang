"""Unit tests for model-runner layer discovery."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.model_executor.model_runner_components.layer_setup import (
    compute_attention_and_moe_layers,
    resolve_layer_indices,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestComputeAttentionAndMoeLayers(unittest.TestCase):
    @staticmethod
    def _mtp_model_config():
        return SimpleNamespace(
            num_nextn_predict_layers=1,
            num_hidden_layers=48,
            num_attention_layers=12,
            hf_config=SimpleNamespace(
                architectures=["Qwen4ExpForConditionalGeneration"]
            ),
        )

    def test_pd_prefill_mtp_allows_pipeline_layer_shard(self):
        model = SimpleNamespace(start_layer=0, end_layer=24)

        with (
            patch(
                "sglang.srt.model_executor.model_runner_components.layer_setup."
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="prefill"),
            ),
            patch(
                "sglang.srt.model_executor.model_runner_components.layer_setup.is_npu",
                return_value=False,
            ),
        ):
            layer_info = resolve_layer_indices(
                model=model,
                model_config=self._mtp_model_config(),
                is_draft_worker=False,
                spec_algorithm=SpeculativeAlgorithm.EAGLE,
            )

        self.assertEqual(layer_info.start_layer, 0)
        self.assertEqual(layer_info.end_layer, 24)
        self.assertEqual(layer_info.num_effective_layers, 24)

    def test_non_pd_mtp_rejects_pipeline_layer_shard(self):
        model = SimpleNamespace(start_layer=0, end_layer=24)

        with (
            patch(
                "sglang.srt.model_executor.model_runner_components.layer_setup."
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="null"),
            ),
            patch(
                "sglang.srt.model_executor.model_runner_components.layer_setup.is_npu",
                return_value=False,
            ),
            self.assertRaisesRegex(AssertionError, "PP is not compatible with MTP"),
        ):
            resolve_layer_indices(
                model=model,
                model_config=self._mtp_model_config(),
                is_draft_worker=False,
                spec_algorithm=SpeculativeAlgorithm.EAGLE,
            )

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


if __name__ == "__main__":
    unittest.main()
