"""Model-owned MTP checkpoint selection, without allocating model weights."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _without_weights(model_class, **config):
    model = model_class.__new__(model_class)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(**config)
    return model


class TestQwenCheckpointSelection(CustomTestCase):
    def test_different_subclass_layout_does_not_inherit_qwen_filter(self):
        from sglang.srt.models.interns2_mobius import (
            InternS2MobiusForConditionalGeneration,
        )

        model = _without_weights(InternS2MobiusForConditionalGeneration)
        self.assertIsNone(getattr(model, "is_unused_checkpoint_weight", None))

    def test_qwen35_main_and_draft_use_distinct_rules(self):
        from sglang.srt.models.qwen3_5 import (
            Qwen3_5ForCausalLM,
            Qwen3_5ForConditionalGeneration,
            Qwen3_5MoeForCausalLM,
            Qwen3_5MoeForConditionalGeneration,
        )
        from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP
        from sglang.srt.models.qwen3_5_text import (
            Qwen3_5ForCausalLM as Qwen3_5TextForCausalLM,
        )

        cases = [
            ("model.layers.0.self_attn.q_proj.weight", False, True),
            ("mtp.layers.0.self_attn.q_proj.weight", True, False),
            ("mtp.layers.0.mlp.experts.0.up_proj.weight_scale", True, False),
            ("mtp.fc.weight", True, False),
            ("model.embed_tokens.weight", False, False),
            ("model.language_model.embed_tokens.weight", False, False),
            ("lm_head.weight", False, True),
            ("mtp.layers.0.rotary_emb.inv_freq", True, True),
        ]
        for cls in (
            Qwen3_5ForCausalLM,
            Qwen3_5ForConditionalGeneration,
            Qwen3_5MoeForCausalLM,
            Qwen3_5MoeForConditionalGeneration,
            Qwen3_5TextForCausalLM,
        ):
            main = _without_weights(cls)
            draft = _without_weights(Qwen3_5ForCausalLMMTP)
            for name, main_unused, draft_unused in cases:
                with self.subTest(model=cls.__name__, name=name):
                    self.assertEqual(
                        main.is_unused_checkpoint_weight(name), main_unused
                    )
                    self.assertEqual(
                        draft.is_unused_checkpoint_weight(name), draft_unused
                    )

    def test_qwen_next_does_not_inherit_qwen35_embedding_exception(self):
        from sglang.srt.models.qwen3_next import Qwen3NextForCausalLM
        from sglang.srt.models.qwen3_next_mtp import Qwen3NextForCausalLMMTP

        main = _without_weights(Qwen3NextForCausalLM)
        draft = _without_weights(Qwen3NextForCausalLMMTP)
        for name, main_unused, draft_unused in (
            ("model.embed_tokens.weight", False, True),
            ("model.layers.0.mlp.up_proj.weight", False, True),
            ("mtp.pre_fc_norm_embedding.weight", True, False),
            ("mtp.layers.0.mlp.up_proj.weight_scale", True, False),
            ("mtp.layers.0.rotary_emb.inv_freq", True, True),
        ):
            with self.subTest(name=name):
                self.assertEqual(main.is_unused_checkpoint_weight(name), main_unused)
                self.assertEqual(draft.is_unused_checkpoint_weight(name), draft_unused)


class TestNextNCheckpointSelection(CustomTestCase):
    def test_remapped_multimodal_draft_does_not_inherit_text_filter(self):
        from sglang.srt.models.glm5_next_nextn import (
            Glm5NextForConditionalGenerationNextN,
        )

        model = _without_weights(Glm5NextForConditionalGenerationNextN)
        self.assertIsNone(getattr(model, "is_unused_checkpoint_weight", None))

    def test_nextn_layer_shared_weights_and_legacy_layout(self):
        from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN
        from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM
        from sglang.srt.models.glm4_moe import Glm4MoeForCausalLM
        from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteForCausalLM
        from sglang.srt.models.glm4_moe_lite_nextn import Glm4MoeLiteForCausalLMNextN
        from sglang.srt.models.glm4_moe_nextn import Glm4MoeForCausalLMNextN

        for main_cls, draft_cls in (
            (DeepseekV2ForCausalLM, DeepseekV3ForCausalLMNextN),
            (Glm4MoeForCausalLM, Glm4MoeForCausalLMNextN),
            (Glm4MoeLiteForCausalLM, Glm4MoeLiteForCausalLMNextN),
        ):
            for num_layers in (1, 61):
                config = dict(num_hidden_layers=num_layers, num_nextn_predict_layers=1)
                main = _without_weights(main_cls, **config)
                draft = _without_weights(draft_cls, **config)
                draft_layer = 0 if num_layers == 1 else num_layers
                prefix = f"model.layers.{draft_layer}."
                for name in (
                    prefix + "eh_proj.weight",
                    prefix + "enorm.weight",
                    prefix + "hnorm.weight",
                    prefix + "shared_head.norm.weight",
                    prefix + "mlp.experts.0.gate_proj.weight_scale_inv",
                ):
                    with self.subTest(model=draft_cls.__name__, name=name):
                        self.assertFalse(draft.is_unused_checkpoint_weight(name))
                for name in (
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                    prefix + "shared_head.head.weight",
                    prefix + "embed_tokens.weight",
                ):
                    with self.subTest(model=draft_cls.__name__, name=name):
                        self.assertTrue(draft.is_unused_checkpoint_weight(name))
                self.assertFalse(
                    main.is_unused_checkpoint_weight("model.layers.0.mlp.weight")
                )
                self.assertTrue(
                    main.is_unused_checkpoint_weight(
                        f"model.layers.{num_layers}.eh_proj.weight"
                    )
                )


if __name__ == "__main__":
    unittest.main()
