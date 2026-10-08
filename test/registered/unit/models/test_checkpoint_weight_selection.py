"""Model-owned MTP checkpoint selection, without allocating model weights."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

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

    def test_mobius_main_uses_its_strict_load_skip_rule(self):
        from sglang.srt.models.interns2_mobius import (
            InternS2MobiusForConditionalGeneration,
        )

        model = _without_weights(
            InternS2MobiusForConditionalGeneration, tie_word_embeddings=True
        )
        for name, unused in (
            ("mtp.layers.0.mlp.up_proj.weight", True),
            ("model.layers.0.self_attn.rotary_emb.inv_freq", True),
            ("lm_head.weight", True),
            ("model.language_model.layers.0.self_attn.q_proj.weight", False),
            ("model.visual.blocks.0.attn.qkv.weight", False),
        ):
            with self.subTest(name=name):
                self.assertEqual(model.is_unused_checkpoint_weight(name), unused)
        model.config.tie_word_embeddings = False
        self.assertFalse(model.is_unused_checkpoint_weight("lm_head.weight"))

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
    def test_mimo_v2_main_and_draft_select_mtp_without_dropping_multimodal(self):
        from sglang.srt.models.mimo_v2 import MiMoV2ForCausalLM
        from sglang.srt.models.mimo_v2_nextn import MiMoV2MTP

        main = _without_weights(MiMoV2ForCausalLM, tie_word_embeddings=False)
        draft = _without_weights(MiMoV2MTP, tie_word_embeddings=False)
        for name, main_unused, draft_unused in (
            ("model.layers.0.self_attn.qkv_proj.weight", False, True),
            ("model.mtp.layers.0.self_attn.qkv_proj.weight", True, False),
            ("model.mtp.layers.0.eh_proj.weight", True, False),
            ("model.embed_tokens.weight", False, False),
            ("lm_head.weight", False, False),
            ("visual.blocks.0.attn.qkv.weight", False, False),
            ("audio_encoder.layers.0.mtp_proj.weight", False, False),
        ):
            with self.subTest(name=name):
                self.assertEqual(main.is_unused_checkpoint_weight(name), main_unused)
                self.assertEqual(draft.is_unused_checkpoint_weight(name), draft_unused)

    def test_glm5_next_uses_its_raw_multimodal_layer_prefixes(self):
        from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
        from sglang.srt.models.glm5_next_nextn import (
            Glm5NextForConditionalGenerationNextN,
        )

        config = dict(num_hidden_layers=61, num_nextn_predict_layers=1)
        main = _without_weights(Glm5NextForConditionalGeneration, **config)
        draft = _without_weights(Glm5NextForConditionalGenerationNextN, **config)
        for name, main_unused, draft_unused in (
            ("model.language_model.layers.0.self_attn.q_proj.weight", False, True),
            ("model.language_model.layers.61.eh_proj.weight", True, False),
            ("model.layers.61.mlp.experts.0.w1.weight", True, False),
            ("model.language_model.embed_tokens.weight", False, True),
            ("model.visual.blocks.0.attn.qkv.weight", False, True),
        ):
            with self.subTest(name=name):
                self.assertEqual(main.is_unused_checkpoint_weight(name), main_unused)
                self.assertEqual(draft.is_unused_checkpoint_weight(name), draft_unused)

    def test_mimo_main_and_mtp_preserve_shared_embedding_and_head(self):
        from sglang.srt.models.mimo import MiMoForCausalLM
        from sglang.srt.models.mimo_mtp import MiMoMTP

        main = _without_weights(MiMoForCausalLM, tie_word_embeddings=False)
        draft = _without_weights(MiMoMTP, tie_word_embeddings=False)
        for name, main_unused, draft_unused in (
            ("model.layers.0.self_attn.q_proj.weight", False, True),
            ("model.mtp_layers.0.self_attn.q_proj.weight", True, False),
            ("model.mtp_layers.0.token_layernorm.weight", True, False),
            ("model.embed_tokens.weight", False, False),
            ("lm_head.weight", False, False),
            ("model.layers.0.rotary_emb.inv_freq", True, True),
        ):
            with self.subTest(name=name):
                self.assertEqual(main.is_unused_checkpoint_weight(name), main_unused)
                self.assertEqual(draft.is_unused_checkpoint_weight(name), draft_unused)

    def test_glm_ocr_main_and_nextn_select_after_language_model_remapping(self):
        from sglang.srt.models.glm_ocr import GlmOcrForConditionalGeneration
        from sglang.srt.models.glm_ocr_nextn import (
            GlmOcrForConditionalGenerationNextN,
        )

        config = dict(num_hidden_layers=61, num_nextn_predict_layers=1)
        main = _without_weights(GlmOcrForConditionalGeneration, **config)
        draft = _without_weights(GlmOcrForConditionalGenerationNextN, **config)
        for name, main_unused, draft_unused in (
            ("model.language_model.layers.0.self_attn.q_proj.weight", False, True),
            ("model.language_model.layers.61.eh_proj.weight", True, False),
            ("model.layers.61.shared_head.norm.weight", True, False),
            ("model.layers.61.shared_head.head.weight", True, True),
            ("model.language_model.embed_tokens.weight", False, True),
            ("model.visual.blocks.0.attn.q_proj.weight", False, True),
            ("model.layers.61.rotary_emb.inv_freq", True, True),
        ):
            with self.subTest(name=name):
                self.assertEqual(main.is_unused_checkpoint_weight(name), main_unused)
                self.assertEqual(draft.is_unused_checkpoint_weight(name), draft_unused)

        draft.config.num_nextn_predict_layers = 2
        self.assertTrue(
            draft._is_unused_checkpoint_weight(
                "model.layers.0.self_attn.q_proj.weight", is_nextn=True
            )
        )
        self.assertFalse(
            draft.is_unused_checkpoint_weight("model.layers.0.self_attn.q_proj.weight")
        )

    def test_deepseek_v4_main_and_nextn_apply_raw_name_remapping(self):
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
        from sglang.srt.models.deepseek_v4_nextn import DeepseekV4ForCausalLMNextN

        config = dict(num_hidden_layers=61, num_nextn_predict_layers=1)
        main = _without_weights(DeepseekV4ForCausalLM, **config)
        draft = _without_weights(DeepseekV4ForCausalLMNextN, **config)
        for name, main_unused, draft_unused in (
            ("layers.0.attn.wo.weight", False, True),
            ("model.layers.61.self_attn.wo_a.scale", True, False),
            ("mtp.0.e_proj.weight", True, False),
            ("mtp.0.norm.weight", True, False),
            ("mtp.0.head.weight", True, True),
            ("mtp.0.attn.wo.weight", True, False),
            ("embed.weight", False, True),
            ("mtp.0.rotary_emb.inv_freq", True, True),
        ):
            with self.subTest(name=name):
                self.assertEqual(main.is_unused_checkpoint_weight(name), main_unused)
                self.assertEqual(draft.is_unused_checkpoint_weight(name), draft_unused)

        main.config.num_nextn_predict_layers = 0
        self.assertTrue(main.is_unused_checkpoint_weight("mtp.0.attn.wo.weight"))
        self.assertFalse(
            main.is_unused_checkpoint_weight("model.layers.61.self_attn.wo.weight")
        )

        draft.config.num_nextn_predict_layers = 2
        self.assertTrue(
            draft._is_unused_checkpoint_weight(
                "model.layers.0.self_attn.wo.weight", is_nextn=True
            )
        )
        self.assertFalse(draft.is_unused_checkpoint_weight("layers.0.attn.wo.weight"))

    def test_deepseek_v4_dspark_uses_its_name_mapper_for_draft_selection(self):
        from sglang.srt.models.deepseek_v4_dspark import (
            DeepseekV4ForCausalLMDSpark,
        )

        draft = _without_weights(DeepseekV4ForCausalLMDSpark)
        draft.confidence_head = None
        for name, unused in (
            ("layers.0.attn.wq.weight", True),
            ("embed.weight", True),
            ("mtp.0.attn.wo.weight", False),
            ("mtp.0.markov_head.embed.weight", False),
            ("mtp.0.confidence_head.proj.weight", True),
            ("mtp.0.rotary_emb.inv_freq", True),
        ):
            with self.subTest(name=name):
                self.assertEqual(draft.is_unused_checkpoint_weight(name), unused)

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


class TestIndexedMTPCheckpointSelection(CustomTestCase):
    def test_main_models_skip_only_their_proven_draft_weights(self):
        from sglang.srt.models.gigachat35 import GigaChat35ForCausalLM
        from sglang.srt.models.inkling import InklingForConditionalGeneration
        from sglang.srt.models.step3p5 import Step3p5ForCausalLM
        from sglang.srt.models.step3p7 import Step3p7ForConditionalGeneration

        step = _without_weights(
            Step3p5ForCausalLM, num_hidden_layers=61, num_nextn_predict_layers=2
        )
        multimodal_step = _without_weights(Step3p7ForConditionalGeneration)
        multimodal_step.language_model = step
        inkling = _without_weights(InklingForConditionalGeneration)
        giga = _without_weights(
            GigaChat35ForCausalLM, num_hidden_layers=61, num_nextn_predict_layers=2
        )
        for model, name, unused in (
            (step, "model.layers.60.mlp.weight", False),
            (step, "model.layers.61.eh_proj.weight", True),
            (multimodal_step, "model.layers.61.eh_proj.weight", True),
            (multimodal_step, "model.vision_model.blocks.0.weight", False),
            (inkling, "model.mtp.layers.0.attn.weight", True),
            (inkling, "model.llm.layers.0.attn.weight", False),
            (giga, "model.layers.61.eh_proj.weight", True),
            (giga, "model.layers.60.mlp.weight", False),
        ):
            with self.subTest(model=type(model).__name__, name=name):
                self.assertEqual(model.is_unused_checkpoint_weight(name), unused)

    def test_step3p5_draft_selects_only_its_layer_and_shared_embedding(self):
        from sglang.srt.models.step3p5_mtp import Step3p5MTP

        draft = _without_weights(
            Step3p5MTP, num_hidden_layers=61, num_nextn_predict_layers=3
        )
        draft.draft_model_idx = 1
        for name, unused in (
            ("model.layers.60.self_attn.q_proj.weight", True),
            ("model.layers.61.eh_proj.weight", True),
            ("model.layers.62.eh_proj.weight", False),
            ("model.layers.62.shared_head.head.weight", False),
            ("model.layers.63.eh_proj.weight", True),
            ("model.embed_tokens.weight", False),
            ("model.shared_head.head.weight", True),
            ("model.layers.62.rotary_emb.inv_freq", True),
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    draft.is_unused_checkpoint_weight_with_draft_model_idx(name),
                    unused,
                )
                self.assertEqual(draft.is_unused_checkpoint_weight(name), unused)
        draft.draft_model_idx = None
        self.assertFalse(
            draft.is_unused_checkpoint_weight("model.layers.61.eh_proj.weight")
        )

    def test_inkling_draft_selects_its_mtp_layer_and_shared_weights(self):
        from sglang.srt.models.inkling import InklingForConditionalGenerationMTP

        draft = _without_weights(InklingForConditionalGenerationMTP)
        draft.draft_model_idx = 1
        for name, unused in (
            ("model.mtp.layers.0.transformer_block.weight", True),
            ("model.mtp.layers.1.transformer_block.weight", False),
            ("model.mtp.layers.2.transformer_block.weight", True),
            ("model.mtp.chain_norm.weight", False),
            ("model.llm.embed.weight", False),
            ("model.llm.lm_head.weight", False),
            ("model.llm.layers.0.attn.qkv.weight", True),
            ("model.mtp.layers.1.rotary_emb.inv_freq", True),
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    draft.is_unused_checkpoint_weight_with_draft_model_idx(name),
                    unused,
                )

    def test_gigachat_single_draft_reuses_its_loader_rule(self):
        from sglang.srt.models.gigachat35_mtp import GigaChat35ForCausalLMNextN

        draft = _without_weights(
            GigaChat35ForCausalLMNextN,
            num_hidden_layers=61,
            num_nextn_predict_layers=2,
        )
        draft.draft_model_idx = 0
        for name, unused in (
            ("model.layers.60.self_attn.q_proj.weight", True),
            ("model.layers.61.eh_proj.weight", False),
            ("model.layers.62.eh_proj.weight", True),
            ("model.layers.61.shared_head.head.weight", True),
            ("model.embed_tokens.weight", True),
        ):
            with self.subTest(name=name):
                self.assertEqual(draft.is_unused_checkpoint_weight(name), unused)

    def test_gigachat_indexed_draft_selects_and_loads_its_own_layer(self):
        from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
            DeepseekV2WeightLoaderMixin,
        )
        from sglang.srt.models.gigachat35_mtp import GigaChat35ForCausalLMNextN

        class Param:
            def __init__(self, loaded):
                self.loaded = loaded

            def weight_loader(self, _param, weight):
                self.loaded.append(weight.item())

        class Draft(DeepseekV2WeightLoaderMixin):
            config = SimpleNamespace(
                num_hidden_layers=61, num_nextn_predict_layers=2, n_routed_experts=0
            )
            quant_config = None
            num_fused_shared_experts = 0
            model = SimpleNamespace()
            pp_group = SimpleNamespace(is_last_rank=True)
            _initialize_nextn_conf = GigaChat35ForCausalLMNextN._initialize_nextn_conf

            def __init__(self, idx, loaded):
                self.draft_model_idx = idx
                self.param = Param(loaded)

            def named_parameters(self):
                return [("model.eh_proj.weight", self.param)]

            def _maybe_quant_weights_to_fp8_ue8m0(self, weights, *_args):
                return weights

            def post_load_weights(self, **_kwargs):
                pass

        for idx in (0, 1):
            loaded = []
            draft = Draft(idx, loaded)
            selected = 61 + idx
            for name, unused in (
                (f"model.layers.{selected}.eh_proj.weight", False),
                (f"model.layers.{selected}.shared_head.norm.weight", False),
                (f"model.layers.{selected}.shared_head.head.weight", True),
                (f"model.layers.{62 - idx}.eh_proj.weight", True),
            ):
                with self.subTest(idx=idx, name=name):
                    self.assertEqual(
                        GigaChat35ForCausalLMNextN.is_unused_checkpoint_weight_with_draft_model_idx(
                            draft, name
                        ),
                        unused,
                    )
            with patch(
                "sglang.srt.models.deepseek_common.deepseek_weight_loader.FusedMoE.make_expert_params_mapping",
                return_value=[],
            ):
                draft.do_load_weights(
                    [
                        (
                            f"model.layers.{layer}.eh_proj.weight",
                            torch.tensor(float(layer)),
                        )
                        for layer in (61, 62)
                    ],
                    is_nextn=True,
                )
            self.assertEqual(loaded, [float(selected)])


class TestNemotronHCheckpointSelection(CustomTestCase):
    def test_main_and_draft_use_their_own_mapped_name_rules(self):
        from sglang.srt.models.nemotron_h import NemotronHForCausalLM
        from sglang.srt.models.nemotron_h_mtp import NemotronHForCausalLMMTP

        main = _without_weights(NemotronHForCausalLM)
        draft = _without_weights(NemotronHForCausalLMMTP)
        for name, main_unused, draft_unused in (
            ("backbone.layers.0.weight", False, True),
            ("mtp.layers.0.weight", True, False),
            ("backbone.embeddings.weight", False, False),
            ("lm_head.weight", False, False),
        ):
            with self.subTest(name=name):
                self.assertEqual(main.is_unused_checkpoint_weight(name), main_unused)
                self.assertEqual(draft.is_unused_checkpoint_weight(name), draft_unused)

    def test_full_inventory_preserves_shared_head_decision_after_shard_skip(self):
        from sglang.srt.models.nemotron_h import NemotronHForCausalLM
        from sglang.srt.models.nemotron_h_mtp import NemotronHForCausalLMMTP

        draft = _without_weights(NemotronHForCausalLMMTP)
        draft.lm_head = torch.nn.Linear(1, 1, bias=False)
        draft.begin_checkpoint_weight_selection()
        draft.observe_checkpoint_weight_names(
            ("language_model.backbone.layers.0.weight",)
        )
        draft.observe_checkpoint_weight_names(
            ("language_model.mtp.layers.0.weight", "language_model.lm_head.weight")
        )
        self.assertTrue(
            draft.is_unused_checkpoint_weight("language_model.backbone.layers.0.weight")
        )
        with patch.object(
            NemotronHForCausalLM,
            "load_weights",
            side_effect=lambda weights, is_mtp: list(weights),
        ):
            draft.load_weights(
                iter(
                    (
                        ("language_model.mtp.layers.0.weight", torch.ones(1)),
                        ("language_model.lm_head.weight", torch.ones(1)),
                    )
                )
            )
            self.assertFalse(draft._owns_lm_head)
            # A later non-safetensors load has no header observer and must use
            # only its own iterator, not the previous checkpoint's metadata.
            draft.load_weights(
                iter(
                    (
                        ("language_model.mtp.layers.0.weight", torch.ones(1)),
                        ("language_model.lm_head.weight", torch.ones(1)),
                    )
                )
            )
            self.assertTrue(draft._owns_lm_head)


if __name__ == "__main__":
    unittest.main()
