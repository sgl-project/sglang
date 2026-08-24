"""Unit tests for ``sglang.srt.models.plamo3`` — no server, no weight loading."""

import unittest

import torch

from sglang.srt.configs.plamo3 import Plamo3Config, is_full_attn
from sglang.srt.models.plamo3 import (
    Plamo3ForCausalLM,
    Plamo3Model,
    Plamo3RMSNorm,
)
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

    def test_default_architectures_matches_upstream(self):
        cfg = self._make()
        self.assertIsNone(cfg.architectures)

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

    def test_interleaved_sliding_window_is_derived(self):
        cfg = self._make(num_hidden_layers=4, sliding_window_pattern=2)
        self.assertNotIn("interleaved_sliding_window", cfg.to_dict())
        cfg.window_size = 64
        self.assertEqual(cfg.interleaved_sliding_window, [64, None, 64, None])

    def test_sliding_window_alias(self):
        cfg = self._make(window_size=256)
        self.assertEqual(cfg.sliding_window, 256)

    def test_legacy_rope_global_theta(self):
        cfg = self._make(rope_global_theta=777_777)
        self.assertEqual(cfg.rope_theta, 777_777)
        self.assertFalse(hasattr(cfg, "rope_global_theta"))

    def test_legacy_sliding_window_scalar(self):
        cfg = self._make(sliding_window=256)
        self.assertEqual(cfg.window_size, 256)

    def test_legacy_sliding_window_list(self):
        cfg = self._make(sliding_window=[256, 256, None, 256])
        self.assertEqual(cfg.window_size, 256)

    def test_scale_embedding_default(self):
        self.assertFalse(self._make().scale_embedding)

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

    def test_rope_parameters_default_when_factor_is_one(self):
        cfg = self._make(rope_scaling_factor=1)
        self.assertEqual(
            cfg.rope_parameters,
            {
                "full_attention": {
                    "rope_theta": cfg.rope_theta,
                    "rope_type": "default",
                },
                "sliding_attention": {
                    "rope_theta": cfg.rope_local_theta,
                    "rope_type": "default",
                },
            },
        )

    def test_rope_parameters_yarn_dict(self):
        cfg = self._make(
            rope_scaling_factor=64.0,
            initial_context_length=4096,
            max_position_embeddings=262144,
        )
        rs = cfg.rope_parameters
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

    def test_rope_parameters_requires_initial_context_length(self):
        with self.assertRaises(AssertionError):
            self._make(rope_scaling_factor=64.0, initial_context_length=None)

    def test_default_rope_rejects_initial_context_length(self):
        with self.assertRaises(AssertionError):
            self._make(rope_scaling_factor=1, initial_context_length=4096)

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


class TestPlamo3RMSNorm(CustomTestCase):
    def test_checkpoint_weight_is_loaded_as_offset(self):
        norm = Plamo3RMSNorm(4, offset=0.2)
        loaded_weight = torch.tensor([0.1, -0.1, 0.0, 0.3])

        norm.weight.weight_loader(norm.weight, loaded_weight)

        torch.testing.assert_close(norm.weight, loaded_weight + 0.2)

    def test_forward_matches_reference(self):
        norm = Plamo3RMSNorm(
            4,
            eps=1e-6,
            offset=0.2,
        ).float()
        loaded_weight = torch.tensor([0.1, -0.1, 0.0, 0.3])
        norm.weight.weight_loader(norm.weight, loaded_weight)
        x = torch.tensor([[1.0, -2.0, 3.0, -4.0]])

        expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        expected *= loaded_weight + 0.2

        torch.testing.assert_close(norm.forward_native(x), expected)


class TestPlamo3Embedding(CustomTestCase):
    def test_scale_embedding_matches_upstream(self):
        model = Plamo3Model.__new__(Plamo3Model)
        torch.nn.Module.__init__(model)
        model.config = Plamo3Config(
            hidden_size=4,
            scale_embedding=True,
        )
        model.embed_tokens = torch.nn.Embedding(2, 4)
        torch.nn.init.ones_(model.embed_tokens.weight)

        embeddings = model.embed_input_ids(torch.tensor([0, 1]))

        torch.testing.assert_close(embeddings, torch.full((2, 4), 2.0))


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
