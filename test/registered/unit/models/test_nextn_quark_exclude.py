import unittest
from types import SimpleNamespace

from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN
from sglang.srt.models.glm4_moe import GlmMoeDsaForCausalLMNextN
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _FakeQuarkConfig:
    """Minimal stand-in for QuarkConfig: only the bits the nextn quant
    resolution reads (get_name / exclude_layers / quant_config)."""

    def __init__(self, name, exclude_layers, quant_config=None):
        self._name = name
        self.exclude_layers = list(exclude_layers)
        self.quant_config = dict(quant_config or {})

    def get_name(self):
        return self._name


class TestBaseNextNQuarkResolution(unittest.TestCase):
    """Base DeepseekV3ForCausalLMNextN: non-quark / None configs pass through
    unchanged, quark configs are resolved against the mapped MTP prefix."""

    def _resolve(self, num_hidden_layers, quant_config):
        fake_self = SimpleNamespace(
            hf_to_sglang_mapper=DeepseekV3ForCausalLMNextN.hf_to_sglang_mapper
        )
        config = SimpleNamespace(num_hidden_layers=num_hidden_layers)
        return DeepseekV3ForCausalLMNextN._resolve_nextn_quant_config(
            fake_self, config, quant_config
        )

    def test_non_quark_config_unchanged(self):
        quant_config = _FakeQuarkConfig("fp8", ["model.layers.61.eh_proj"])
        self.assertIs(self._resolve(61, quant_config), quant_config)

    def test_none_config_unchanged(self):
        self.assertIsNone(self._resolve(61, None))


class TestGlmNextNQuarkResolution(unittest.TestCase):
    """GLM-5.2 MTP: exclude_layers / layer_quant_config keys recorded under the
    checkpoint prefix "model.layers.<N>.*" must be remapped to the runtime
    "model.*"/"model.decoder.*" names SGLang actually queries, without mutating
    the (possibly shared) input quant_config."""

    _LAYER = 78

    def _resolve(self, quant_config):
        fake_self = SimpleNamespace(
            _map_mtp_ckpt_name=GlmMoeDsaForCausalLMNextN._map_mtp_ckpt_name,
        )
        config = SimpleNamespace(num_hidden_layers=self._LAYER)
        return GlmMoeDsaForCausalLMNextN._resolve_nextn_quant_config(
            fake_self, config, quant_config
        )

    def test_exclude_names_remapped_to_runtime_prefixes(self):
        original_excludes = [
            "model.layers.78.eh_proj",
            "model.layers.78.self_attn.q_proj",
            "model.layers.78.mlp.experts.0.gate_proj",
            "lm_head",
        ]
        quant_config = _FakeQuarkConfig("quark", original_excludes)

        resolved = self._resolve(quant_config)

        # A copy is returned; the input is never mutated in place.
        self.assertIsNot(resolved, quant_config)
        self.assertEqual(quant_config.exclude_layers, original_excludes)

        names = set(resolved.exclude_layers)
        # MTP-specific weight -> model.*
        self.assertIn("model.eh_proj", names)
        # decoder block weight -> model.decoder.*
        self.assertIn("model.decoder.self_attn.q_proj", names)
        # fused routed experts also queried by the coarse module prefix
        self.assertIn("model.decoder.mlp.experts", names)
        # original leaf names are preserved too
        self.assertIn("lm_head", names)

    def test_layer_quant_config_keys_remapped_without_mutation(self):
        original_lqc = {
            "model.layers.78.self_attn": "ptpc_fp8",
            "model.layers.5.self_attn": "keep",
        }
        quant_config = _FakeQuarkConfig(
            "quark", [], quant_config={"layer_quant_config": original_lqc}
        )

        resolved = self._resolve(quant_config)

        # Input dict is untouched (guards the shallow-copy side-effect bug).
        self.assertEqual(
            quant_config.quant_config["layer_quant_config"],
            {
                "model.layers.78.self_attn": "ptpc_fp8",
                "model.layers.5.self_attn": "keep",
            },
        )
        remapped = resolved.quant_config["layer_quant_config"]
        self.assertIn("model.decoder.self_attn", remapped)
        self.assertNotIn("model.layers.78.self_attn", remapped)
        # non-MTP keys are left as-is
        self.assertIn("model.layers.5.self_attn", remapped)

    def test_no_mtp_excludes_returns_same_config(self):
        quant_config = _FakeQuarkConfig("quark", ["model.layers.5.self_attn.q_proj"])
        self.assertIs(self._resolve(quant_config), quant_config)

    def test_non_quark_config_unchanged(self):
        quant_config = _FakeQuarkConfig("fp8", ["model.layers.78.eh_proj"])
        self.assertIs(self._resolve(quant_config), quant_config)


if __name__ == "__main__":
    unittest.main()
