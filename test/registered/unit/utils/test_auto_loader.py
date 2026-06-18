# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for the centralized weight loading utilities (auto_loader.py).

These tests are CPU-only and don't require model weights or GPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock

import torch
from torch import nn
from torch.nn import Parameter

from sglang.srt.model_loader.auto_loader import (
    STANDARD_GATE_UP_MAPPING,
    STANDARD_QKV_MAPPING,
    STANDARD_STACKED_MAPPING,
    filter_pp_weights,
    get_weight_remap,
    register_weight_remap,
)
from sglang.srt.models.utils import AutoWeightsLoader, WeightsMapper

# ---------------------------------------------------------------------------
# StackedParamsDispatch Tests
# ---------------------------------------------------------------------------


class TestStackedParamsDispatch(unittest.TestCase):
    def _make_param_with_loader(self, shape):
        """Create a parameter with a mock weight_loader."""
        param = Parameter(torch.zeros(shape))
        param.weight_loader = MagicMock()
        return param

    def test_qkv_mapping_matches_q_proj(self):
        params = {"qkv_proj.weight": self._make_param_with_loader((30, 10))}
        tensor = torch.ones(10, 10)

        result = STANDARD_QKV_MAPPING.try_load("q_proj.weight", tensor, params)

        self.assertEqual(result, "qkv_proj.weight")
        params["qkv_proj.weight"].weight_loader.assert_called_once_with(
            params["qkv_proj.weight"], tensor, "q"
        )

    def test_qkv_mapping_matches_k_proj(self):
        params = {"qkv_proj.weight": self._make_param_with_loader((30, 10))}
        tensor = torch.ones(10, 10)

        result = STANDARD_QKV_MAPPING.try_load("k_proj.weight", tensor, params)

        self.assertEqual(result, "qkv_proj.weight")
        params["qkv_proj.weight"].weight_loader.assert_called_once_with(
            params["qkv_proj.weight"], tensor, "k"
        )

    def test_qkv_mapping_matches_v_proj(self):
        params = {"qkv_proj.weight": self._make_param_with_loader((30, 10))}
        tensor = torch.ones(10, 10)

        result = STANDARD_QKV_MAPPING.try_load("v_proj.weight", tensor, params)

        self.assertEqual(result, "qkv_proj.weight")
        params["qkv_proj.weight"].weight_loader.assert_called_once_with(
            params["qkv_proj.weight"], tensor, "v"
        )

    def test_gate_up_mapping_matches_gate_proj(self):
        params = {"gate_up_proj.weight": self._make_param_with_loader((20, 10))}
        tensor = torch.ones(10, 10)

        result = STANDARD_GATE_UP_MAPPING.try_load("gate_proj.weight", tensor, params)

        self.assertEqual(result, "gate_up_proj.weight")
        params["gate_up_proj.weight"].weight_loader.assert_called_once_with(
            params["gate_up_proj.weight"], tensor, 0
        )

    def test_gate_up_mapping_matches_up_proj(self):
        params = {"gate_up_proj.weight": self._make_param_with_loader((20, 10))}
        tensor = torch.ones(10, 10)

        result = STANDARD_GATE_UP_MAPPING.try_load("up_proj.weight", tensor, params)

        self.assertEqual(result, "gate_up_proj.weight")
        params["gate_up_proj.weight"].weight_loader.assert_called_once_with(
            params["gate_up_proj.weight"], tensor, 1
        )

    def test_no_match_returns_none(self):
        params = {"qkv_proj.weight": self._make_param_with_loader((30, 10))}
        tensor = torch.ones(10, 10)

        result = STANDARD_QKV_MAPPING.try_load("o_proj.weight", tensor, params)

        self.assertIsNone(result)
        params["qkv_proj.weight"].weight_loader.assert_not_called()

    def test_missing_target_param_returns_target_name(self):
        # Parameter doesn't exist in params_dict (e.g. GPTQ bias)
        params = {}  # No qkv_proj.bias
        tensor = torch.ones(10)

        result = STANDARD_QKV_MAPPING.try_load("q_proj.bias", tensor, params)

        # Returns target name for skip tracking
        self.assertEqual(result, "qkv_proj.bias")

    def test_standard_stacked_mapping_covers_all(self):
        params = {
            "qkv_proj.weight": self._make_param_with_loader((30, 10)),
            "gate_up_proj.weight": self._make_param_with_loader((20, 10)),
        }
        tensor = torch.ones(10, 10)

        # All 5 source names should match
        for source, expected_target in [
            ("q_proj.weight", "qkv_proj.weight"),
            ("k_proj.weight", "qkv_proj.weight"),
            ("v_proj.weight", "qkv_proj.weight"),
            ("gate_proj.weight", "gate_up_proj.weight"),
            ("up_proj.weight", "gate_up_proj.weight"),
        ]:
            result = STANDARD_STACKED_MAPPING.try_load(source, tensor, params)
            self.assertEqual(result, expected_target, f"Failed for {source}")


# ---------------------------------------------------------------------------
# WeightsMapper Tests
# ---------------------------------------------------------------------------


class TestWeightsMapper(unittest.TestCase):
    def test_substr_remap(self):
        mapper = WeightsMapper(orig_to_new_substr={".old_name.": ".new_name."})
        weights = [("prefix.old_name.weight", torch.ones(2, 2))]

        result = list(mapper.apply(weights))

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "prefix.new_name.weight")

    def test_prefix_remap(self):
        mapper = WeightsMapper(orig_to_new_prefix={"model.model.": "model."})
        weights = [("model.model.layers.0.weight", torch.ones(2, 2))]

        result = list(mapper.apply(weights))

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "model.layers.0.weight")

    def test_suffix_remap(self):
        mapper = WeightsMapper(orig_to_new_suffix={".activation_scale": ".input_scale"})
        weights = [("layer.activation_scale", torch.ones(2))]

        result = list(mapper.apply(weights))

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "layer.input_scale")

    def test_drop_via_none(self):
        mapper = WeightsMapper(orig_to_new_substr={"rotary_emb": None})
        weights = [
            ("layer.rotary_emb.inv_freq", torch.ones(2)),
            ("layer.o_proj.weight", torch.ones(2, 2)),
        ]

        result = list(mapper.apply(weights))

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "layer.o_proj.weight")

    def test_composition_or(self):
        mapper1 = WeightsMapper(orig_to_new_substr={".old1.": ".new1."})
        mapper2 = WeightsMapper(orig_to_new_suffix={".scale_inv": ".scale"})

        composed = mapper1 | mapper2
        weights = [
            ("prefix.old1.weight", torch.ones(2, 2)),
            ("layer.scale_inv", torch.ones(2)),
        ]

        result = list(composed.apply(weights))

        self.assertEqual(result[0][0], "prefix.new1.weight")
        self.assertEqual(result[1][0], "layer.scale")

    def test_longest_substr_match_wins(self):
        mapper = WeightsMapper(
            orig_to_new_substr={
                "proj": "proj_replaced",
                "q_proj": "qkv_proj",  # Longer match should win
            }
        )
        weights = [("layer.q_proj.weight", torch.ones(2, 2))]

        result = list(mapper.apply(weights))

        self.assertEqual(result[0][0], "layer.qkv_proj.weight")


# ---------------------------------------------------------------------------
# AutoWeightsLoader Tests
# ---------------------------------------------------------------------------


class TestAutoWeightsLoader(unittest.TestCase):
    def test_direct_param_load(self):
        model = nn.Linear(4, 4, bias=False)
        loader = AutoWeightsLoader(model)
        tensor = torch.ones(4, 4)

        loaded = loader.load_weights([("weight", tensor)])

        self.assertIn("weight", loaded)
        self.assertTrue(torch.equal(model.weight.data, tensor))

    def test_nested_module_load(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(4, 4, bias=False)

        model = Model()
        loader = AutoWeightsLoader(model)
        tensor = torch.ones(4, 4)

        loaded = loader.load_weights([("layer.weight", tensor)])

        self.assertIn("layer.weight", loaded)
        self.assertTrue(torch.equal(model.layer.weight.data, tensor))

    def test_skip_prefixes(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.kept = nn.Linear(4, 4, bias=False)
                self.skipped = nn.Linear(4, 4, bias=False)

        model = Model()
        loader = AutoWeightsLoader(model, skip_prefixes=["skipped."])
        tensor = torch.ones(4, 4)

        loaded = loader.load_weights(
            [
                ("kept.weight", tensor),
                ("skipped.weight", tensor * 2),
            ]
        )

        self.assertIn("kept.weight", loaded)
        self.assertNotIn("skipped.weight", loaded)
        self.assertTrue(torch.equal(model.kept.weight.data, tensor))
        # skipped should remain at init value (zeros-like from default)

    def test_skip_rotary_embeddings(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(4))

        model = Model()
        loader = AutoWeightsLoader(model)

        loaded = loader.load_weights(
            [
                ("weight", torch.ones(4)),
                ("rotary_emb.inv_freq", torch.ones(10)),
            ]
        )

        self.assertIn("weight", loaded)
        self.assertNotIn("rotary_emb.inv_freq", loaded)

    def test_module_delegation(self):
        """Walker delegates to child module's load_weights if present."""

        class ChildModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.zeros(4))

            def load_weights(self, weights):
                loaded = set()
                for name, tensor in weights:
                    if name == "source_name":
                        self.param.data.copy_(tensor)
                        loaded.add("param")
                return loaded

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.child = ChildModule()

        model = Model()
        loader = AutoWeightsLoader(model)
        tensor = torch.ones(4)

        loaded = loader.load_weights([("child.source_name", tensor)])

        self.assertIn("child.param", loaded)
        self.assertTrue(torch.equal(model.child.param.data, tensor))

    def test_ignore_unexpected_suffix(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(4, 4, bias=False)

        model = Model()
        loader = AutoWeightsLoader(model, ignore_unexpected_suffixes=[".bias"])

        # Should not raise for unexpected .bias weight nested under layer
        loaded = loader.load_weights(
            [
                ("layer.weight", torch.ones(4, 4)),
                ("layer.bias", torch.ones(4)),  # model's layer has no bias
            ]
        )

        self.assertIn("layer.weight", loaded)

    def test_raises_on_truly_unexpected(self):
        model = nn.Linear(4, 4, bias=False)
        loader = AutoWeightsLoader(model)

        with self.assertRaises(ValueError):
            loader.load_weights([("nonexistent_param", torch.ones(4))])

    def test_mapper_applied(self):
        model = nn.Linear(4, 4, bias=False)
        mapper = WeightsMapper(orig_to_new_substr={"old_weight": "weight"})
        loader = AutoWeightsLoader(model)
        tensor = torch.ones(4, 4)

        loaded = loader.load_weights([("old_weight", tensor)], mapper=mapper)

        self.assertIn("weight", loaded)
        self.assertTrue(torch.equal(model.weight.data, tensor))

    def test_pp_missing_layer_skipped(self):
        """PPMissingLayer modules should be silently skipped."""

        class PPMissingLayer(nn.Module):
            pass

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = PPMissingLayer()
                self.embed = nn.Linear(4, 4, bias=False)

        model = Model()
        loader = AutoWeightsLoader(model)

        # Weights for the missing layer should not raise
        loaded = loader.load_weights(
            [
                ("embed.weight", torch.ones(4, 4)),
                ("layer.weight", torch.ones(4, 4)),
            ]
        )

        self.assertIn("embed.weight", loaded)


# ---------------------------------------------------------------------------
# filter_pp_weights Tests
# ---------------------------------------------------------------------------


class TestFilterPPWeights(unittest.TestCase):
    def test_drops_out_of_range_layers(self):
        weights = [
            ("model.layers.0.self_attn.weight", torch.ones(4)),
            ("model.layers.1.self_attn.weight", torch.ones(4)),
            ("model.layers.2.self_attn.weight", torch.ones(4)),
            ("model.layers.3.self_attn.weight", torch.ones(4)),
        ]

        result = list(filter_pp_weights(weights, start_layer=1, end_layer=3))

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], "model.layers.1.self_attn.weight")
        self.assertEqual(result[1][0], "model.layers.2.self_attn.weight")

    def test_passes_non_layer_weights(self):
        weights = [
            ("model.embed_tokens.weight", torch.ones(4)),
            ("lm_head.weight", torch.ones(4)),
            ("model.layers.0.self_attn.weight", torch.ones(4)),
            ("model.norm.weight", torch.ones(4)),
        ]

        result = list(filter_pp_weights(weights, start_layer=1, end_layer=3))

        # embed_tokens, lm_head, norm pass through (no layer id)
        # layer 0 is filtered out
        names = [name for name, _ in result]
        self.assertIn("model.embed_tokens.weight", names)
        self.assertIn("lm_head.weight", names)
        self.assertIn("model.norm.weight", names)
        self.assertNotIn("model.layers.0.self_attn.weight", names)

    def test_empty_range_drops_all_layers(self):
        weights = [
            ("model.layers.0.weight", torch.ones(4)),
            ("model.embed_tokens.weight", torch.ones(4)),
        ]

        result = list(filter_pp_weights(weights, start_layer=5, end_layer=5))

        names = [name for name, _ in result]
        self.assertNotIn("model.layers.0.weight", names)
        self.assertIn("model.embed_tokens.weight", names)


# ---------------------------------------------------------------------------
# WeightRemapRegistry Tests
# ---------------------------------------------------------------------------


class TestWeightRemapRegistry(unittest.TestCase):
    def test_registered_model_returns_mapper(self):
        # LlamaForCausalLM is registered in auto_loader.py
        class FakeLlama(nn.Module):
            pass

        FakeLlama.__name__ = "LlamaForCausalLM"
        model = FakeLlama()

        mapper = get_weight_remap(model)

        self.assertIsNotNone(mapper)
        self.assertIsInstance(mapper, WeightsMapper)

    def test_unknown_model_returns_none(self):
        class UnknownModel(nn.Module):
            pass

        model = UnknownModel()

        mapper = get_weight_remap(model)

        self.assertIsNone(mapper)

    def test_custom_registration(self):
        @register_weight_remap("TestModelXYZ")
        def _test_remap(model):
            return WeightsMapper(orig_to_new_substr={"foo": "bar"})

        class TestModelXYZ(nn.Module):
            pass

        model = TestModelXYZ()
        mapper = get_weight_remap(model)

        self.assertIsNotNone(mapper)
        weights = [("prefix.foo.weight", torch.ones(2))]
        result = list(mapper.apply(weights))
        self.assertEqual(result[0][0], "prefix.bar.weight")


if __name__ == "__main__":
    unittest.main()
