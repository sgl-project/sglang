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
    def test_qwen3_moe_main_and_draft_filter_their_own_weights(self):
        from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM
        from sglang.srt.models.qwen3_moe_mtp import Qwen3MoeForCausalLMMTP

        main = _without_weights(Qwen3MoeForCausalLM)
        draft = _without_weights(Qwen3MoeForCausalLMMTP)
        for name, main_unused, draft_unused in (
            ("model.layers.0.mlp.gate_proj.weight", False, True),
            ("mtp.layers.0.mlp.gate_proj.weight", True, False),
            ("mtp.fc.weight", True, False),
            ("mtp.layers.0.rotary_emb.inv_freq", True, True),
        ):
            with self.subTest(name=name):
                self.assertEqual(main.is_unused_checkpoint_weight(name), main_unused)
                self.assertEqual(draft.is_unused_checkpoint_weight(name), draft_unused)

    def test_qwen4_main_skips_draft_weights_without_skipping_vision_shards(self):
        from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration
        from sglang.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP

        main = _without_weights(Qwen4ExpForConditionalGeneration)
        main.language_model_only = True
        draft = _without_weights(Qwen4ExpForCausalLMMTP)
        for name, main_unused, draft_unused in (
            ("model.language_model.layers.0.self_attn.q_proj.weight", False, True),
            ("mtp.layers.0.self_attn.q_proj.weight", True, False),
            ("model.language_model.embed_tokens.weight", False, False),
            ("model.visual.blocks.0.attn.qkv.weight", False, True),
            ("mtp.layers.0.rotary_emb.inv_freq", True, True),
        ):
            with self.subTest(name=name):
                self.assertEqual(main.is_unused_checkpoint_weight(name), main_unused)
                self.assertEqual(draft.is_unused_checkpoint_weight(name), draft_unused)

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


if __name__ == "__main__":
    unittest.main()
