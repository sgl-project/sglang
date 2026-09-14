"""Unit tests for model configuration."""

import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from transformers import LlamaConfig

from sglang.srt.arg_groups.overrides import model_config_of
from sglang.srt.configs.model_config import (
    ModelConfig,
    get_hybrid_layer_ids,
    is_embedding_gemma,
    is_multimodal_model,
    register_model_config_factory,
    resolve_spec_hidden_size,
)
from sglang.srt.configs.qwen4_exp import Qwen4ExpTextConfig
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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
        self.enterContext(
            mock.patch.dict(
                "sglang.srt.configs.model_config._MODEL_CONFIG_FACTORIES", clear=True
            )
        )

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


if __name__ == "__main__":
    unittest.main()
