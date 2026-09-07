"""Unit tests for model-runner layer discovery."""

import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.model_runner_components.layer_setup import (
    compute_attention_and_moe_layers,
    resolve_layer_indices,
)
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


class TestResolveMtpLayerIndices(unittest.TestCase):
    def resolve(
        self,
        *,
        start,
        end,
        architecture="Glm5NextForConditionalGeneration",
        is_draft_worker=False,
        mtp_layers=1,
        speculative=True,
    ):
        return resolve_layer_indices(
            model=SimpleNamespace(start_layer=start, end_layer=end),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=[architecture]),
                num_nextn_predict_layers=mtp_layers,
                num_hidden_layers=45,
                num_attention_layers=45,
            ),
            is_draft_worker=is_draft_worker,
            spec_algorithm=SimpleNamespace(is_none=lambda: not speculative),
        )

    def test_glm5_next_mtp_target_preserves_each_pp_stage_range(self):
        for start, end in ((0, 23), (23, 45)):
            with self.subTest(start=start, end=end):
                info = self.resolve(start=start, end=end)
                self.assertEqual((info.start_layer, info.end_layer), (start, end))
                self.assertEqual(info.num_effective_layers, end - start)

    def test_other_mtp_targets_remain_rejected(self):
        for architecture in ("DeepseekV3ForCausalLM", "GlmMoeDsaForCausalLM"):
            with self.subTest(architecture=architecture):
                with self.assertRaisesRegex(AssertionError, "PP is not compatible"):
                    self.resolve(start=0, end=23, architecture=architecture)

    def test_partitioned_glm5_next_mtp_draft_remains_rejected(self):
        for architecture in (
            "Glm5NextForConditionalGeneration",
            "Glm5NextForConditionalGenerationNextN",
        ):
            with self.subTest(architecture=architecture):
                with self.assertRaisesRegex(AssertionError, "PP is not compatible"):
                    self.resolve(
                        start=0,
                        end=1,
                        architecture=architecture,
                        is_draft_worker=True,
                        mtp_layers=2,
                    )

    def test_unpartitioned_glm5_next_mtp_draft_keeps_local_layer_count(self):
        info = self.resolve(
            start=0,
            end=1,
            is_draft_worker=True,
            architecture="Glm5NextForConditionalGenerationNextN",
        )
        self.assertEqual(info.num_effective_layers, 1)

    def test_other_non_mtp_target_still_supports_pp(self):
        info = self.resolve(
            start=23, end=45, architecture="OtherForCausalLM", mtp_layers=0
        )
        self.assertEqual(info.num_effective_layers, 22)

    def test_mtp_target_without_speculation_still_supports_pp(self):
        info = self.resolve(
            start=0, end=23, architecture="DeepseekV3ForCausalLM", speculative=False
        )
        self.assertEqual(info.num_effective_layers, 23)


if __name__ == "__main__":
    unittest.main()
