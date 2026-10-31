"""Unit tests for model configuration."""

import json
import os
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from transformers import (
    Cohere2Config,
    Gemma2Config,
    Gemma3TextConfig,
    LlamaConfig,
    Qwen3_5Config,
)

from sglang.srt.arg_groups.overrides import model_config_of
from sglang.srt.configs.model_config import (
    ModelConfig,
    get_hybrid_layer_ids,
    is_embedding_gemma,
    is_hybrid_swa_model,
    is_layer_types_hybrid_swa_model,
    is_multimodal_model,
    load_decision_config,
    register_model_config_factory,
    resolve_spec_hidden_size,
)
from sglang.srt.configs.olmo3 import Olmo3Config
from sglang.srt.configs.qwen4_exp import Qwen4ExpTextConfig
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestHybridLayerIds(CustomTestCase):
    def test_layer_type_architectures(self):
        config = SimpleNamespace(
            num_hidden_layers=4,
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
        )

        for architecture in (
            "Gemma4ForCausalLM",
            "Gemma4ForConditionalGeneration",
            "LagunaForCausalLM",
            "MellumForCausalLM",
        ):
            with self.subTest(architecture=architecture):
                self.assertEqual(
                    get_hybrid_layer_ids([architecture], config),
                    ([0, 2], [1, 3]),
                )

    def test_layer_types_mix_makes_a_model_hybrid_swa(self):
        exaone4_32b = SimpleNamespace(
            num_hidden_layers=8,
            sliding_window=4096,
            sliding_window_pattern="LLLG",
            layer_types=(["sliding_attention"] * 3 + ["full_attention"]) * 2,
        )
        olmo3 = Olmo3Config(num_hidden_layers=8, sliding_window=4096)
        # The full-attention layers each model's own code builds.
        cases = {
            "Gemma2ForCausalLM": (Gemma2Config(num_hidden_layers=6), [1, 3, 5]),
            "Gemma3ForCausalLM": (Gemma3TextConfig(num_hidden_layers=12), [5, 11]),
            "Cohere2ForCausalLM": (Cohere2Config(num_hidden_layers=8), [3, 7]),
            "Exaone4ForCausalLM": (exaone4_32b, [3, 7]),
            olmo3.architectures[0]: (olmo3, [3, 7]),
        }
        for architecture, (config, full) in cases.items():
            with self.subTest(architecture=architecture):
                self.assertTrue(is_hybrid_swa_model([architecture], config))
                self.assertTrue(is_layer_types_hybrid_swa_model([architecture], config))
                swa = [i for i in range(config.num_hidden_layers) if i not in full]
                self.assertEqual(
                    get_hybrid_layer_ids([architecture], config), (swa, full)
                )

    def test_layer_types_without_a_sliding_full_mix_are_not_hybrid_swa(self):
        cases = {
            "no window": (None, ["sliding_attention", "full_attention"]),
            "all full": (4096, ["full_attention"] * 2),
            "all sliding": (4096, ["sliding_attention"] * 2),
            "other kinds": (4096, ["sliding_attention", "linear_attention"]),
            "fewer than the layers": (4096, ["sliding_attention"]),
        }
        for name, (sliding_window, layer_types) in cases.items():
            with self.subTest(name):
                config = SimpleNamespace(
                    num_hidden_layers=2,
                    sliding_window=sliding_window,
                    layer_types=layer_types,
                )
                self.assertFalse(is_hybrid_swa_model(["LlamaForCausalLM"], config))
                self.assertEqual(
                    get_hybrid_layer_ids(["LlamaForCausalLM"], config), (None, None)
                )

    def test_listed_architectures_keep_their_own_rule(self):
        config = SimpleNamespace(
            num_hidden_layers=2,
            sliding_window=4096,
            layer_types=["sliding_attention", "full_attention"],
        )
        self.assertTrue(is_hybrid_swa_model(["Gemma4ForCausalLM"], config))
        self.assertFalse(is_layer_types_hybrid_swa_model(["Gemma4ForCausalLM"], config))
        self.assertTrue(is_layer_types_hybrid_swa_model(["Gemma3ForCausalLM"], config))


class TestEmbeddingGemmaConfig(CustomTestCase):
    def test_detects_bidirectional_gemma3_text_config(self):
        config = SimpleNamespace(
            model_type="gemma3_text", use_bidirectional_attention=True
        )
        self.assertTrue(is_embedding_gemma(config))

    def test_does_not_misclassify_causal_gemma3(self):
        config = SimpleNamespace(
            model_type="gemma3_text", use_bidirectional_attention=False
        )
        self.assertFalse(is_embedding_gemma(config))


class TestDraftModelConfig(CustomTestCase):
    def test_nemotron_h_omni_is_multimodal(self):
        self.assertTrue(is_multimodal_model(["NemotronH_Omni_Reasoning_V3"]))

    def test_qwen35_mtp_depth_is_synced_to_text_config(self):
        config = object.__new__(ModelConfig)
        config.is_draft_model = True
        config.speculative_algorithm = "EAGLE"
        config.hf_config = SimpleNamespace(
            architectures=["Qwen3_5MoeForConditionalGeneration"]
        )
        config.hf_text_config = SimpleNamespace()

        config._config_draft_model()

        self.assertEqual(config.hf_config.architectures, ["Qwen3_5ForCausalLMMTP"])
        self.assertEqual(config.hf_config.num_nextn_predict_layers, 1)
        self.assertEqual(config.hf_text_config.num_nextn_predict_layers, 1)

    def test_draft_is_hybrid_swa_only_by_the_listed_rules(self):
        """FROZEN_KV_MTP gives its draft a single-pool dummy config, so a draft
        whose layer_types mix sliding and full layers keeps one KV pool."""
        text_config = SimpleNamespace(
            num_hidden_layers=4,
            sliding_window=512,
            layer_types=["sliding_attention"] * 3 + ["full_attention"],
        )
        cases = [
            # (architecture, is_draft_model, is_hybrid_swa, is_layer_types_hybrid_swa)
            ("Gemma4AssistantForCausalLM", False, True, True),
            ("Gemma4AssistantForCausalLM", True, False, False),
            ("Gemma4ForCausalLM", True, True, False),
        ]
        for architecture, is_draft_model, hybrid, layer_types_hybrid in cases:
            with self.subTest(architecture=architecture, is_draft_model=is_draft_model):
                config = object.__new__(ModelConfig)
                config.is_draft_model = is_draft_model
                config.disable_hybrid_swa_memory = False
                config.hf_config = SimpleNamespace(architectures=[architecture])
                config.hf_text_config = text_config
                with mock.patch.object(
                    ModelConfig, "_detect_attention_sinks", return_value=False
                ):
                    config._derive_hybrid_model()
                self.assertEqual(config.is_hybrid_swa, hybrid)
                self.assertEqual(config.is_layer_types_hybrid_swa, layer_types_hybrid)

    def test_nemotron_h_omni_mtp_uses_language_model_config(self):
        config = object.__new__(ModelConfig)
        config.is_draft_model = True
        config.speculative_algorithm = "EAGLE"
        config.hf_config = SimpleNamespace(
            architectures=["NemotronH_Omni_Reasoning_V3"]
        )
        config.hf_text_config = SimpleNamespace(architectures=["NemotronHForCausalLM"])

        config._config_draft_model()

        self.assertIs(config.hf_config, config.hf_text_config)
        self.assertEqual(config.hf_config.architectures, ["NemotronHForCausalLMMTP"])
        self.assertEqual(config.hf_config.num_nextn_predict_layers, 1)

    def test_qwen4_exp_spec_hidden_size_keeps_hc_width(self):
        """Qwen4-Exp's MTP draft consumes the hc-flattened target stream,
        so spec_hidden_size must stay hidden_size * hc_mult; hy_v4 collapses first."""
        hidden_size, hc_mult = 2560, 4
        self.assertEqual(Qwen4ExpTextConfig(hc_count=hc_mult).hc_mult, hc_mult)
        for arch in ("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLMMTP"):
            hf_config = SimpleNamespace(architectures=[arch])
            self.assertEqual(
                resolve_spec_hidden_size(
                    hf_config=hf_config, hidden_size=hidden_size, hc_mult=hc_mult
                ),
                (hidden_size * hc_mult, hidden_size * hc_mult),
            )
        hy_v4 = SimpleNamespace(architectures=["HYV4ForCausalLM"])
        self.assertEqual(
            resolve_spec_hidden_size(
                hf_config=hy_v4, hidden_size=hidden_size, hc_mult=hc_mult
            ),
            (hidden_size, None),
        )


class TestExternalModelConfig(CustomTestCase):
    def setUp(self):
        registry = mock.patch.dict(
            "sglang.srt.configs.model_config._MODEL_CONFIG_FACTORIES", clear=True
        )
        registry.start()
        self.addCleanup(registry.stop)

    def test_factory_preserves_arguments_and_uses_the_shared_cache(self):
        class ExternalArgs(ServerArgs):
            pass

        class DerivedArgs(ExternalArgs):
            pass

        result = SimpleNamespace(is_hybrid_swa=False)
        factory = mock.Mock(return_value=result)
        register_model_config_factory(ExternalArgs, factory)
        args = DerivedArgs(model_path="dummy")
        overrides = dict(
            model_path="draft",
            model_revision="draft-revision",
            is_draft_model=True,
            context_length=512,
            dtype="bfloat16",
        )
        self.assertIs(ModelConfig.from_server_args(args, **overrides), result)
        factory.assert_called_once_with(args, **overrides)

        factory.reset_mock()
        self.assertIs(model_config_of(args), result)
        self.assertIs(model_config_of(args), result)
        factory.assert_called_once_with(
            args,
            model_path=None,
            model_revision=None,
            is_draft_model=False,
            context_length=None,
        )

    def test_registration_is_idempotent_and_the_nearest_type_wins(self):
        class ExternalArgs(ServerArgs):
            pass

        class DerivedArgs(ExternalArgs):
            pass

        parent_factory = mock.Mock()
        child_factory = mock.Mock()
        register_model_config_factory(ExternalArgs, parent_factory)
        register_model_config_factory(ExternalArgs, parent_factory)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_model_config_factory(ExternalArgs, child_factory)
        with self.assertRaisesRegex(TypeError, "ServerArgs subclass"):
            register_model_config_factory(SimpleNamespace, child_factory)
        register_model_config_factory(DerivedArgs, child_factory)
        args = DerivedArgs(model_path="dummy")
        self.assertIs(ModelConfig.from_server_args(args), child_factory.return_value)
        parent_factory.assert_not_called()

    def test_capabilities_are_available_during_construction(self):
        class ExternalArgs(ServerArgs):
            pass

        class ExternalConfig(ModelConfig):
            def _derive_multimodal_cuda_graph_support(self, enable_multimodal):
                super()._derive_multimodal_cuda_graph_support(enable_multimodal)
                self.is_multimodal_breakable_cuda_graph_supported = True

            def _derive_model_shapes(self):
                assert self.is_multimodal_breakable_cuda_graph_supported
                super()._derive_model_shapes()

        register_model_config_factory(
            ExternalArgs,
            lambda args, **kwargs: ExternalConfig(model_path=args.model_path),
        )
        with TemporaryDirectory() as checkpoint:
            LlamaConfig(
                architectures=["LlamaForCausalLM"],
                hidden_size=16,
                intermediate_size=32,
                num_attention_heads=2,
                num_hidden_layers=2,
                vocab_size=128,
            ).save_pretrained(checkpoint)
            ordinary = ModelConfig.from_server_args(ServerArgs(model_path=checkpoint))
            external = ModelConfig.from_server_args(ExternalArgs(model_path=checkpoint))
        self.assertIs(type(ordinary), ModelConfig)
        self.assertFalse(ordinary.is_multimodal_breakable_cuda_graph_supported)
        self.assertIs(type(external), ExternalConfig)
        self.assertTrue(external.is_multimodal_breakable_cuda_graph_supported)


class TestReasoningTokenFields(CustomTestCase):
    def test_present_without_a_reasoning_parser(self):
        """Every generated token reads these; only a configured reasoning parser fills
        them, so a server without one must still find them defined."""
        with TemporaryDirectory() as checkpoint:
            LlamaConfig(
                architectures=["LlamaForCausalLM"],
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                vocab_size=128,
            ).save_pretrained(checkpoint)
            config = ModelConfig(model_path=checkpoint)
        self.assertIsNone(config.think_end_ids)
        self.assertIsNone(config.self_opened_think_start_ids)


class TestDecisionCheckpointConfig(CustomTestCase):
    def _checkpoint(self, directory, **decision):
        Qwen3_5Config(architectures=["Qwen3_5Model"]).save_pretrained(directory)
        decision = {"format_version": 1, "codes": ["A"], "token_ids": [32], **decision}
        with open(os.path.join(directory, "decision_config.json"), "w") as f:
            json.dump(decision, f)

    def test_attention_mode_reaches_the_text_config(self):
        """A noncausal decision checkpoint must not be served with causal full attention."""
        for decision, causal in (
            ({}, True),
            ({"attention_mode": "noncausal_full_attention"}, False),
        ):
            with self.subTest(decision=decision), TemporaryDirectory() as checkpoint:
                self._checkpoint(checkpoint, **decision)
                config = ModelConfig(model_path=checkpoint)
                self.assertEqual(
                    config.hf_config.architectures, ["Qwen3_5ForConditionalGeneration"]
                )
                if causal:
                    self.assertFalse(hasattr(config.hf_text_config, "is_causal"))
                else:
                    self.assertIs(config.hf_text_config.is_causal, False)

    def test_unsupported_decision_fields_are_refused(self):
        """An attention mode or pooling the server does not honor must not be served."""
        for decision in ({"attention_mode": "bidirectional"}, {"pooling": "mean"}):
            with self.subTest(decision=decision), TemporaryDirectory() as checkpoint:
                self._checkpoint(checkpoint, **decision)
                with self.assertRaisesRegex(ValueError, "is not supported"):
                    load_decision_config(checkpoint, None)


if __name__ == "__main__":
    unittest.main()
