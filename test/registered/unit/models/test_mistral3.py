import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.models.mistral import Mistral3ForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class FakeMinistral3ForCausalLM:
    def __init__(self, config, quant_config=None, prefix=""):
        self.config = config
        self.quant_config = quant_config
        self.prefix = prefix
        self.loaded_weights = None

    def load_weights(self, weights):
        self.loaded_weights = list(weights)
        return "loaded"


class TestMistral3TextOnlyWrapper(CustomTestCase):
    def test_disable_multimodal_uses_text_model_and_strips_weight_prefix(self):
        text_config = SimpleNamespace(model_type="ministral3")
        config = SimpleNamespace(enable_multimodal=False, text_config=text_config)
        weight_a = object()
        weight_b = object()
        weight_c = object()
        weight_d = object()
        ignored_weight = object()

        with patch(
            "sglang.srt.models.ministral3.Ministral3ForCausalLM",
            FakeMinistral3ForCausalLM,
        ):
            model = Mistral3ForConditionalGeneration(
                config=config,
                quant_config="quant",
                prefix="language_model",
            )
            result = model.load_weights(
                [
                    ("language_model.model.embed_tokens.weight", weight_a),
                    ("model.language_model.layers.0.self_attn.q_proj.weight", weight_c),
                    ("vision_model.embeddings.weight", ignored_weight),
                    ("model.vision_tower.embeddings.weight", ignored_weight),
                    ("lm_head.weight", weight_d),
                    ("language_model.lm_head.weight", weight_b),
                ]
            )

        self.assertTrue(model.text_only)
        self.assertIs(model.inner.config, text_config)
        self.assertEqual(model.inner.quant_config, "quant")
        self.assertEqual(model.inner.prefix, "language_model")
        self.assertEqual(result, "loaded")
        self.assertEqual(
            model.inner.loaded_weights,
            [
                ("model.embed_tokens.weight", weight_a),
                ("model.layers.0.self_attn.q_proj.weight", weight_c),
                ("lm_head.weight", weight_d),
                ("lm_head.weight", weight_b),
            ],
        )


if __name__ == "__main__":
    unittest.main()
