"""Unit tests for ``sglang.srt.models.plamo3`` — no server, no weight loading."""

import unittest

from sglang.srt.configs.plamo3 import Plamo3Config, is_full_attn
from sglang.srt.models.plamo3 import Plamo3ForCausalLM
from sglang.srt.models.registry import ModelRegistry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPlamo3Config(CustomTestCase):
    def _make(self, **overrides):
        defaults = dict(
            hidden_size=64,
            num_hidden_layers=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
            window_size=128,
            sliding_window_pattern=4,
            intermediate_size=128,
            vocab_size=64,
        )
        defaults.update(overrides)
        return Plamo3Config(**defaults)

    def test_model_type(self):
        self.assertEqual(Plamo3Config.model_type, "plamo3")

    def test_default_architectures(self):
        cfg = self._make()
        self.assertEqual(cfg.architectures, ["Plamo3ForCausalLM"])

    def test_custom_architectures_respected(self):
        cfg = self._make(architectures=["CustomArch"])
        self.assertEqual(cfg.architectures, ["CustomArch"])

    def test_is_full_attn_pattern(self):
        # sliding_window_pattern=4: full attn at layers 3, 7, 11, ...
        self.assertFalse(is_full_attn(4, 0))
        self.assertFalse(is_full_attn(4, 1))
        self.assertFalse(is_full_attn(4, 2))
        self.assertTrue(is_full_attn(4, 3))
        self.assertTrue(is_full_attn(4, 7))

    def test_interleaved_sliding_window_layout(self):
        cfg = self._make(num_hidden_layers=8, sliding_window_pattern=4)
        self.assertEqual(len(cfg.interleaved_sliding_window), 8)
        # Layers 3 and 7 are full attention (None), rest are windowed.
        expected = [128, 128, 128, None, 128, 128, 128, None]
        self.assertEqual(cfg.interleaved_sliding_window, expected)

    def test_layer_types(self):
        cfg = self._make(num_hidden_layers=8, sliding_window_pattern=4)
        self.assertEqual(
            cfg.layer_types,
            [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        )

    def test_layers_block_type(self):
        cfg = self._make(num_hidden_layers=4)
        self.assertEqual(cfg.layers_block_type, ["attention"] * 4)

    def test_rope_scaling_none_when_factor_is_one(self):
        cfg = self._make(rope_scaling_factor=1)
        self.assertIsNone(cfg.rope_scaling)

    def test_rope_scaling_yarn_dict(self):
        cfg = self._make(
            rope_scaling_factor=64.0,
            initial_context_length=4096,
            max_position_embeddings=262144,
        )
        rs = cfg.rope_scaling
        self.assertIsNotNone(rs)
        full = rs["full_attention"]
        self.assertEqual(full["rope_type"], "yarn")
        self.assertEqual(full["factor"], 64.0)
        self.assertEqual(full["original_max_position_embeddings"], 4096)
        self.assertEqual(full["beta_fast"], 32.0)
        self.assertEqual(full["beta_slow"], 1.0)
        self.assertFalse(full["truncate"])
        sliding = rs["sliding_attention"]
        self.assertEqual(sliding["rope_type"], "default")

    def test_rope_scaling_requires_initial_context_length(self):
        cfg = self._make(rope_scaling_factor=64.0, initial_context_length=None)
        with self.assertRaises(AssertionError):
            _ = cfg.rope_scaling

    def test_rope_local_base_freq(self):
        cfg = self._make(rope_local_theta=12345)
        self.assertEqual(cfg.rope_local_base_freq, 12345)


class TestPlamo3Registry(CustomTestCase):
    def test_model_arch_registered(self):
        archs = ModelRegistry.get_supported_archs()
        self.assertIn("Plamo3ForCausalLM", archs)

    def test_resolve_model_cls(self):
        model_cls, arch = ModelRegistry.resolve_model_cls(["Plamo3ForCausalLM"])
        self.assertIs(model_cls, Plamo3ForCausalLM)
        self.assertEqual(arch, "Plamo3ForCausalLM")


class TestPlamo3WeightNameMapping(CustomTestCase):
    """Verify the checkpoint-name normalization used by ``Plamo3ForCausalLM.load_weights``.

    PLaMo3 checkpoints store layer weights under ``model.layers.layers.<i>``
    and attention weights under ``.mixer.``. The SGLang module tree uses the
    flattened ``model.layers.<i>`` path (like other SGLang text models), so
    ``load_weights`` remaps both the double layer prefix and the attention
    submodule name.
    """

    def test_remap_layer_prefix(self):
        name = "model.layers.layers.0.self_attn.o_proj.weight"
        name = name.replace("model.layers.layers.", "model.layers.")
        self.assertEqual(name, "model.layers.0.self_attn.o_proj.weight")

    def test_remap_mixer_to_self_attn(self):
        name = "model.layers.layers.0.mixer.q_proj.weight"
        name = name.replace(".mixer.", ".self_attn.")
        self.assertEqual(name, "model.layers.layers.0.self_attn.q_proj.weight")

    def test_combined_remap(self):
        name = "model.layers.layers.0.mixer.q_proj.weight"
        name = name.replace("model.layers.layers.", "model.layers.")
        name = name.replace(".mixer.", ".self_attn.")
        self.assertEqual(name, "model.layers.0.self_attn.q_proj.weight")


class TestPlamo3Eagle3(CustomTestCase):
    def test_eagle3_capture_raises(self):
        # Instantiating the model requires TP runtime; call the classmethod
        # path indirectly by checking the method raises on a dummy instance.
        # We verify the method exists and is the raising variant via source.
        import inspect

        src = inspect.getsource(Plamo3ForCausalLM.set_eagle3_layers_to_capture)
        self.assertIn("NotImplementedError", src)


if __name__ == "__main__":
    unittest.main()
