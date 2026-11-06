"""
Unit tests for HYV3ForCausalLMNextN.load_weights.

Regression test for the released Hy3 MTP checkpoint key
``model.layers.<num_hidden_layers>.final_layernorm.weight``, which must load
into the draft head's output norm (``model.shared_head.norm``) instead of
being remapped to a nonexistent decoder parameter and silently dropped.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

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


if __name__ == "__main__":
    unittest.main()
