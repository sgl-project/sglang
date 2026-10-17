"""
Unit tests for Hunyuan V3 checkpoint selection and draft weight loading.

Regression test for the released Hy3 MTP checkpoint key
``model.layers.<num_hidden_layers>.final_layernorm.weight``, which must load
into the draft head's output norm (``model.shared_head.norm``) instead of
being remapped to a nonexistent decoder parameter and silently dropped.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.hunyuan_v3 import HYV3ForCausalLM
from sglang.srt.models.hunyuan_v3_nextn import HYV3ForCausalLMNextN


class _FakeParam:
    def __init__(self):
        self.loaded = None

    def weight_loader(self, param, loaded_weight, *args, **kwargs):
        self.loaded = (param, loaded_weight, args, kwargs)


class TestHunyuanV3NextNWeightLoading(unittest.TestCase):
    def _make_minimal_model(self, named_parameters=()):
        model = object.__new__(HYV3ForCausalLMNextN)
        model.config = SimpleNamespace(num_hidden_layers=80, num_experts=2)
        model.named_parameters = lambda: iter(named_parameters)
        return model

    def test_final_layernorm_loads_into_shared_head_norm(self):
        param = _FakeParam()
        model = self._make_minimal_model([("model.shared_head.norm.weight", param)])
        loaded_weight = torch.ones(1)

        model.load_weights([("model.layers.80.final_layernorm.weight", loaded_weight)])

        self.assertEqual(param.loaded, (param, loaded_weight, (), {}))

    def test_spec_weights_map_to_model_prefix(self):
        params = {
            "model.enorm.weight": _FakeParam(),
            "model.hnorm.weight": _FakeParam(),
            "model.eh_proj.weight": _FakeParam(),
        }
        model = self._make_minimal_model(list(params.items()))
        weights = [
            ("model.layers.80.enorm.weight", torch.ones(1)),
            ("model.layers.80.hnorm.weight", torch.ones(1)),
            ("model.layers.80.eh_proj.weight", torch.ones(1)),
        ]

        model.load_weights(weights)

        for name, param in params.items():
            self.assertIsNotNone(param.loaded, f"{name} was not loaded")

    def test_decoder_layer_weight_maps_to_decoder_prefix(self):
        param = _FakeParam()
        model = self._make_minimal_model(
            [("model.decoder.input_layernorm.weight", param)]
        )
        loaded_weight = torch.ones(1)

        model.load_weights([("model.layers.80.input_layernorm.weight", loaded_weight)])

        self.assertEqual(param.loaded, (param, loaded_weight, (), {}))

    def test_embed_tokens_and_lm_head_are_skipped(self):
        params = {
            "model.embed_tokens.weight": _FakeParam(),
            "lm_head.weight": _FakeParam(),
        }
        model = self._make_minimal_model(list(params.items()))
        weights = [
            ("model.embed_tokens.weight", torch.ones(1)),
            ("lm_head.weight", torch.ones(1)),
        ]

        model.load_weights(weights)

        for name, param in params.items():
            self.assertIsNone(param.loaded, f"{name} should have been skipped")

    def test_draft_checkpoint_selection_matches_loaded_weights(self):
        model = self._make_minimal_model()
        for name, unused in (
            ("model.layers.0.self_attn.q_proj.weight", True),
            ("model.layers.80.enorm.weight", False),
            ("model.layers.80.final_layernorm.weight", False),
            ("model.shared_head.norm.weight", False),
            ("model.embed_tokens.weight", True),
            ("model.layers.80.self_attn.rotary_emb.inv_freq", True),
        ):
            with self.subTest(name=name):
                self.assertEqual(model.is_unused_checkpoint_weight(name), unused)


class TestHunyuanV3MainCheckpointSelection(unittest.TestCase):
    def test_main_excludes_nextn_and_tied_head_weights(self):
        model = object.__new__(HYV3ForCausalLM)
        model.config = SimpleNamespace(
            num_hidden_layers=80,
            num_nextn_predict_layers=1,
            tie_word_embeddings=True,
        )
        for name, unused in (
            ("model.layers.0.self_attn.q_proj.weight", False),
            ("model.layers.80.self_attn.q_proj.weight", True),
            ("model.layers.0.self_attn.rotary_emb.inv_freq", True),
            ("lm_head.weight", True),
            ("model.embed_tokens.weight", False),
        ):
            with self.subTest(name=name):
                self.assertEqual(model.is_unused_checkpoint_weight(name), unused)


if __name__ == "__main__":
    unittest.main()
