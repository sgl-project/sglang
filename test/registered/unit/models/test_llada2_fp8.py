import tempfile
import unittest
from types import SimpleNamespace

from transformers import PretrainedConfig

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.configs.model_config import ModelConfig  # noqa: E402
from sglang.srt.layers.quantization.fp8 import Fp8Config  # noqa: E402
from sglang.srt.model_loader.weight_utils import get_quant_config  # noqa: E402
from sglang.srt.utils.common import LazyValue  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _experts_only_config(**overrides):
    config = {
        "enabled": True,
        "format": "e4m3fn",
        "weight_granularity": "per_expert_2d_block",
        "weight_block_size": [128, 128],
        "activation_method": "dynamic_absmax_rtn",
        "activation_granularity": "per_token",
    }
    config.update(overrides)
    return config


class TestLLaDA2Fp8(CustomTestCase):
    def setUp(self):
        self.model_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.model_dir.cleanup)

    def _model_config(
        self,
        custom_config,
        *,
        architecture="LLaDA2MoeModelLM",
        quantization_config=None,
        compression_config=None,
        use_fp8_experts=None,
    ):
        model_config = ModelConfig.__new__(ModelConfig)
        model_config.model_path = self.model_dir.name
        config_kwargs = dict(
            architectures=[architecture],
            quantization_config=quantization_config,
            compression_config=compression_config,
            llada_fp8_experts=custom_config,
        )
        if use_fp8_experts is not None:
            config_kwargs["use_fp8_experts"] = use_fp8_experts
        model_config.hf_config = PretrainedConfig(**config_kwargs)
        return model_config

    def test_parses_experts_only_block_fp8(self):
        model_config = self._model_config(_experts_only_config())

        parsed = model_config._parse_quant_hf_config()
        self.assertEqual(
            parsed,
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
                "llada_experts_only": True,
            },
        )

        fp8_config = Fp8Config.from_config(parsed)
        self.assertTrue(fp8_config.is_checkpoint_fp8_serialized)
        self.assertTrue(fp8_config.llada_experts_only)
        self.assertEqual(fp8_config.weight_block_size, [128, 128])
        self.assertEqual(fp8_config.activation_scheme, "dynamic")

    def test_model_loader_preserves_experts_only_block_config(self):
        model_config = self._model_config(_experts_only_config())
        model_config.quantization = "fp8"

        fp8_config = get_quant_config(
            model_config,
            SimpleNamespace(),
            packed_modules_mapping={},
        )
        self.assertIsInstance(fp8_config, Fp8Config)
        self.assertTrue(fp8_config.llada_experts_only)
        self.assertEqual(fp8_config.weight_block_size, [128, 128])

    def test_parses_legacy_experts_only_fp8_flag(self):
        model_config = self._model_config(None, use_fp8_experts=True)

        self.assertEqual(
            model_config._parse_quant_hf_config(),
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
                "llada_experts_only": True,
            },
        )

        disabled = self._model_config(None, use_fp8_experts=False)
        self.assertIsNone(disabled._parse_llada_fp8_experts_config())

    def test_legacy_flag_restores_only_missing_runtime_fields(self):
        from sglang.srt.models.llada2 import (
            _apply_legacy_llada2_fp8_config_defaults,
        )

        legacy_config = SimpleNamespace(
            use_fp8_experts=True,
            router_dtype="bfloat16",
        )
        _apply_legacy_llada2_fp8_config_defaults(legacy_config)

        self.assertEqual(legacy_config.embedding_dropout, 0.0)
        self.assertTrue(legacy_config.moe_router_enable_expert_bias)
        self.assertTrue(legacy_config.norm_topk_prob)
        self.assertEqual(legacy_config.score_function, "sigmoid")
        self.assertEqual(legacy_config.router_dtype, "bfloat16")

        standard_config = SimpleNamespace(use_fp8_experts=False)
        _apply_legacy_llada2_fp8_config_defaults(standard_config)
        self.assertFalse(hasattr(standard_config, "embedding_dropout"))
        self.assertFalse(hasattr(standard_config, "score_function"))

    def test_experts_only_mode_requires_llada_specific_opt_in(self):
        non_llada_config = self._model_config(
            _experts_only_config(), architecture="Qwen2ForCausalLM"
        )
        self.assertIsNone(non_llada_config._parse_llada_fp8_experts_config())

        legacy_non_llada = self._model_config(
            None, architecture="Qwen2ForCausalLM", use_fp8_experts=True
        )
        self.assertIsNone(legacy_non_llada._parse_llada_fp8_experts_config())

        generic_fp8_config = Fp8Config.from_config(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "moe_only": True,
            }
        )
        self.assertFalse(generic_fp8_config.llada_experts_only)

    def test_standard_quantization_metadata_takes_precedence(self):
        standard_config = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
        }
        for field in ("quantization_config", "compression_config"):
            with self.subTest(field=field):
                model_config = self._model_config(
                    _experts_only_config(), **{field: standard_config.copy()}
                )
                self.assertEqual(model_config._parse_quant_hf_config(), standard_config)

    def test_rejects_non_dynamic_per_token_activation(self):
        model_config = self._model_config(
            _experts_only_config(
                activation_method="static_absmax",
                activation_granularity="per_tensor",
            )
        )
        with self.assertRaisesRegex(ValueError, "dynamic per-token"):
            model_config._parse_quant_hf_config()

    def test_experts_only_validation_uses_effective_config_and_local_partition(self):
        import torch.nn as nn

        from sglang.srt.models.llada2 import (
            LLaDA2MoeModelLM,
            LLaDA2MoeSparseMoeBlock,
        )

        model = LLaDA2MoeModelLM.__new__(LLaDA2MoeModelLM)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(
            num_experts=4,
            llada_fp8_experts={"enabled": True},
        )
        model.quant_config = SimpleNamespace(llada_experts_only=True)
        model.model = nn.Module()
        model.model.layers = nn.ModuleList()

        model.load_weights([])
        self.assertIsInstance(model.routed_experts_weights_of_layer, LazyValue)
        self.assertEqual(model.routed_experts_weights_of_layer.value, {})

        layer = nn.Module()
        layer.mlp = LLaDA2MoeSparseMoeBlock.__new__(LLaDA2MoeSparseMoeBlock)
        nn.Module.__init__(layer.mlp)
        layer.mlp.get_moe_weights = lambda: ()
        model.model.layers = nn.ModuleList([layer])

        model.quant_config = SimpleNamespace(llada_experts_only=False)
        model.load_weights([])

        model.quant_config = SimpleNamespace(llada_experts_only=True)
        with self.assertRaisesRegex(ValueError, "Incomplete LLaDA FP8 expert weights"):
            model.load_weights([])


if __name__ == "__main__":
    unittest.main()
