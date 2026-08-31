"""Unit tests for model-runner layer discovery."""

import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.model_runner_components.layer_setup import (
    compute_attention_and_moe_layers,
    resolve_layer_indices,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


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


class TestResolveLayerIndices(unittest.TestCase):
    @staticmethod
    def _mtp_model_config():
        return SimpleNamespace(
            num_nextn_predict_layers=1,
            num_hidden_layers=48,
            num_attention_layers=48,
            hf_config=SimpleNamespace(
                architectures=["Glm5NextForConditionalGeneration"],
                loop_num=1,
            ),
        )

    def test_mtp_target_model_may_be_pipeline_partitioned(self):
        layer_info = resolve_layer_indices(
            model=SimpleNamespace(start_layer=24, end_layer=48),
            model_config=self._mtp_model_config(),
            is_draft_worker=False,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
        )

        self.assertEqual(layer_info.start_layer, 24)
        self.assertEqual(layer_info.end_layer, 48)
        self.assertEqual(layer_info.num_effective_layers, 24)

    def test_mtp_draft_model_must_own_all_draft_layers(self):
        layer_info = resolve_layer_indices(
            model=SimpleNamespace(),
            model_config=self._mtp_model_config(),
            is_draft_worker=True,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
        )

        self.assertEqual(layer_info.start_layer, 0)
        self.assertEqual(layer_info.end_layer, 1)
        self.assertEqual(layer_info.num_effective_layers, 1)

    def test_partitioned_mtp_draft_model_is_rejected(self):
        with self.assertRaisesRegex(
            AssertionError,
            "Pipeline partitioning is not supported for MTP draft workers",
        ):
            resolve_layer_indices(
                model=SimpleNamespace(start_layer=0, end_layer=0),
                model_config=self._mtp_model_config(),
                is_draft_worker=True,
                spec_algorithm=SpeculativeAlgorithm.EAGLE,
            )


if __name__ == "__main__":
    unittest.main()
