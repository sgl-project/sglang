import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.models import glm5_next
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGlm5NextModelOpt(CustomTestCase):
    def _config(self, exclude_modules):
        config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=exclude_modules,
        )
        config.apply_weight_name_mapper(
            Glm5NextForConditionalGeneration.hf_to_sglang_mapper
        )
        return config

    def _hf_config(self):
        return SimpleNamespace(
            text_config=SimpleNamespace(
                first_k_dense_replace=3,
                num_hidden_layers=45,
                n_shared_experts=1,
            )
        )

    def test_checkpoint_exclusions_match_sglang_module_names(self):
        config = self._config(
            [
                "model.language_model.embed_tokens",
                "model.language_model.layers.11.self_attn*",
                "model.language_model.layers.11.mlp.shared_experts*",
                "model.visual*",
            ]
        )

        self.assertTrue(config.is_layer_excluded("model.embed_tokens"))
        self.assertTrue(
            config.is_layer_excluded("model.layers.11.self_attn.kv_b_proj")
        )
        self.assertTrue(
            config.is_layer_excluded(
                "model.layers.11.mlp.shared_experts.gate_up_proj"
            )
        )
        self.assertTrue(config.is_layer_excluded("visual.blocks.0.attn.qkv_proj"))

    def test_mixed_precision_shared_experts_disable_fusion(self):
        config = self._config(
            ["model.language_model.layers.3.mlp.shared_experts*"]
        )

        reason = Glm5NextForConditionalGeneration.shared_experts_fusion_disable_reason(
            self._hf_config(), config
        )

        self.assertIn("shared experts unquantized", reason)

    def test_uniform_modelopt_fp4_does_not_disable_fusion(self):
        config = self._config([])
        a2a_backend = SimpleNamespace(is_deepep=lambda: False)

        with (
            patch.object(glm5_next, "_is_cuda", True),
            patch.object(glm5_next, "_device_sm", 100),
            patch.object(
                glm5_next, "get_moe_a2a_backend", return_value=a2a_backend
            ),
            get_parallel().override(moe_ep_size=1),
        ):
            reason = (
                Glm5NextForConditionalGeneration.shared_experts_fusion_disable_reason(
                    self._hf_config(), config
                )
            )

        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
