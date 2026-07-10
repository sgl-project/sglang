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
"""Unit tests for the centralized weight loading utilities (auto_loader.py)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock

import pytest
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


def _make_param_with_loader(shape):
    param = Parameter(torch.zeros(shape))
    param.weight_loader = MagicMock()
    return param


@pytest.mark.parametrize(
    "mapping,source_name,expected_target,expected_shard",
    [
        (STANDARD_QKV_MAPPING, "q_proj.weight", "qkv_proj.weight", "q"),
        (STANDARD_QKV_MAPPING, "k_proj.weight", "qkv_proj.weight", "k"),
        (STANDARD_QKV_MAPPING, "v_proj.weight", "qkv_proj.weight", "v"),
        (STANDARD_GATE_UP_MAPPING, "gate_proj.weight", "gate_up_proj.weight", 0),
        (STANDARD_GATE_UP_MAPPING, "up_proj.weight", "gate_up_proj.weight", 1),
    ],
)
def test_stacked_params_dispatch(mapping, source_name, expected_target, expected_shard):
    params = {expected_target: _make_param_with_loader((30, 10))}
    tensor = torch.ones(10, 10)

    result = mapping.try_load(source_name, tensor, params)

    assert result == expected_target
    params[expected_target].weight_loader.assert_called_once_with(
        params[expected_target], tensor, expected_shard
    )


def test_stacked_params_dispatch_no_match_returns_none():
    params = {"qkv_proj.weight": _make_param_with_loader((30, 10))}
    tensor = torch.ones(10, 10)

    result = STANDARD_QKV_MAPPING.try_load("o_proj.weight", tensor, params)

    assert result is None
    params["qkv_proj.weight"].weight_loader.assert_not_called()


def test_stacked_params_dispatch_missing_target_param_returns_target_name():
    params = {}
    tensor = torch.ones(10)

    result = STANDARD_QKV_MAPPING.try_load("q_proj.bias", tensor, params)

    assert result == "qkv_proj.bias"


def test_standard_stacked_mapping_covers_all():
    params = {
        "qkv_proj.weight": _make_param_with_loader((30, 10)),
        "gate_up_proj.weight": _make_param_with_loader((20, 10)),
    }
    tensor = torch.ones(10, 10)

    for source, expected_target in [
        ("q_proj.weight", "qkv_proj.weight"),
        ("k_proj.weight", "qkv_proj.weight"),
        ("v_proj.weight", "qkv_proj.weight"),
        ("gate_proj.weight", "gate_up_proj.weight"),
        ("up_proj.weight", "gate_up_proj.weight"),
    ]:
        result = STANDARD_STACKED_MAPPING.try_load(source, tensor, params)
        assert result == expected_target, f"Failed for {source}"


@pytest.mark.parametrize(
    "checkpoint_name,expected_name",
    [
        (
            "model.layers.0.self_attn.q_proj.activation_scale",
            "model.layers.0.self_attn.q_proj.input_scale",
        ),
        (
            "model.layers.0.mlp.down_proj.weight_scale_inv",
            "model.layers.0.mlp.down_proj.weight_scale",
        ),
        (
            "model.layers.3.self_attn.k_proj.weight_scale_inv",
            "model.layers.3.self_attn.k_proj.weight_scale",
        ),
    ],
)
def test_llama_remap_registry(checkpoint_name, expected_name):
    class FakeLlama(nn.Module):
        pass

    FakeLlama.__name__ = "LlamaForCausalLM"
    mapper = get_weight_remap(FakeLlama())
    assert mapper is not None

    result = list(mapper.apply([(checkpoint_name, torch.ones(2))]))
    assert len(result) == 1
    assert result[0][0] == expected_name


def test_weights_mapper_drops_rotary_emb():
    mapper = WeightsMapper(orig_to_new_substr={"rotary_emb": None})
    weights = [
        ("layer.rotary_emb.inv_freq", torch.ones(2)),
        ("model.layers.0.self_attn.o_proj.weight", torch.ones(2, 2)),
    ]

    result = list(mapper.apply(weights))

    assert len(result) == 1
    assert result[0][0] == "model.layers.0.self_attn.o_proj.weight"


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

        loaded = loader.load_weights(
            [
                ("layer.weight", torch.ones(4, 4)),
                ("layer.bias", torch.ones(4)),
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
        class PPMissingLayer(nn.Module):
            pass

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = PPMissingLayer()
                self.embed = nn.Linear(4, 4, bias=False)

        model = Model()
        loader = AutoWeightsLoader(model)

        loaded = loader.load_weights(
            [
                ("embed.weight", torch.ones(4, 4)),
                ("layer.weight", torch.ones(4, 4)),
            ]
        )

        self.assertIn("embed.weight", loaded)


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


class TestWeightRemapRegistry(unittest.TestCase):
    def test_registered_model_returns_mapper(self):
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
