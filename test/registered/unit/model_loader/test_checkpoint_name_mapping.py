"""Real parameter assignment through legacy FFN loading paths."""

import unittest

import torch
from torch import nn

from sglang.srt.model_loader.weight_utils import (
    get_checkpoint_name_mapper,
    map_state_dict_names,
)
from sglang.srt.models.utils import AutoWeightsLoader
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCheckpointNameMapping(unittest.TestCase):
    def test_only_declared_paths_are_mapped(self):
        model = nn.Module()
        model.checkpoint_name_mapping = {"feed_forward": "ffn"}
        model.ffn = nn.Linear(3, 2)
        model.projector = nn.Module()
        model.projector.mlp = nn.Linear(2, 1)
        map_name = get_checkpoint_name_mapper(model)
        self.assertEqual(map_name("feed_forward.weight"), "ffn.weight")
        self.assertEqual(map_name("ffn.weight"), "ffn.weight")
        self.assertEqual(map_name("mlp.weight"), "mlp.weight")
        self.assertEqual(map_name("projector.mlp.weight"), "projector.mlp.weight")
        # Presence or absence of a parameter must not select a naming rule.
        self.assertEqual(map_name("feed_forward.missing"), "ffn.missing")
        with self.assertRaises(KeyError):
            dict(model.named_parameters())[map_name("mlp.weight")]

    def test_nested_mapping_and_inherited_declarations(self):
        class Base(nn.Module):
            checkpoint_name_mapping = {"mlps": "ffns"}

        class Layer(Base):
            checkpoint_name_mapping = {"mlp_norm": "ffn_norm"}

        model = Layer()
        child = Layer()
        child.ffn_norm = nn.Linear(2, 2)
        model.ffns = nn.ModuleList([child])
        self.assertEqual(
            get_checkpoint_name_mapper(model)("mlps.0.mlp_norm.weight"),
            "ffns.0.ffn_norm.weight",
        )

    def test_metadata_and_duplicate_destinations(self):
        from collections import OrderedDict

        model = nn.Module()
        model.checkpoint_name_mapping = {"mlp": "ffn"}
        model.ffn = nn.Linear(2, 2)
        state = OrderedDict([("mlp.weight", torch.ones(2, 2))])
        state._metadata = {"": {"version": 1}, "mlp": {"version": 3}}
        mapped = map_state_dict_names(state, get_checkpoint_name_mapper(model))
        self.assertEqual(set(mapped), {"ffn.weight"})
        self.assertEqual(mapped._metadata["ffn"], {"version": 3})
        state["ffn.weight"] = torch.zeros(2, 2)
        with self.assertRaisesRegex(ValueError, "Duplicate state_dict destination"):
            map_state_dict_names(state, get_checkpoint_name_mapper(model))

    def test_auto_loader_assignment_buffers_and_loaded_names(self):
        model = nn.Module()
        layer = nn.Module()
        layer.checkpoint_name_mapping = {"mlp": "ffn"}
        layer.ffn = nn.Linear(3, 2, bias=False)
        layer.ffn.register_buffer("scale", torch.zeros(1))
        model.layers = nn.ModuleList([layer])
        model.projector = nn.Module()
        model.projector.mlp = nn.Linear(3, 1, bias=False)
        pointers = {name: value.data_ptr() for name, value in model.named_parameters()}
        loaded = AutoWeightsLoader(model).load_weights(
            [
                ("layers.0.mlp.weight", torch.full((2, 3), 2.0)),
                ("layers.0.mlp.scale", torch.full((1,), 3.0)),
                ("projector.mlp.weight", torch.full((1, 3), 4.0)),
            ]
        )
        self.assertEqual(
            loaded,
            {"layers.0.ffn.weight", "layers.0.ffn.scale", "projector.mlp.weight"},
        )
        self.assertTrue(torch.all(layer.ffn.weight == 2))
        self.assertTrue(torch.all(layer.ffn.scale == 3))
        self.assertTrue(torch.all(model.projector.mlp.weight == 4))
        self.assertEqual(
            pointers,
            {name: value.data_ptr() for name, value in model.named_parameters()},
        )

    def test_auto_loader_skip_uses_checkpoint_prefix(self):
        model = nn.Module()
        model.checkpoint_name_mapping = {"mlp": "ffn"}
        model.ffn = nn.Linear(2, 2, bias=False)
        before = model.ffn.weight.detach().clone()
        loaded = AutoWeightsLoader(model, skip_prefixes=["mlp."]).load_weights(
            [("mlp.weight", torch.zeros(2, 2))]
        )
        self.assertEqual(loaded, set())
        self.assertTrue(torch.equal(model.ffn.weight, before))

    def test_auto_loader_retains_checkpoint_paths_for_ignore_rules(self):
        model = nn.Module()
        model.checkpoint_name_mapping = {"mlp": "ffn"}
        model.ffn = nn.Linear(2, 2, bias=False)
        loaded = AutoWeightsLoader(
            model, ignore_unexpected_prefixes=["mlp.optional."]
        ).load_weights(
            [
                ("mlp.optional.scale", torch.ones(1)),
                ("mlp.weight", torch.ones(2, 2)),
            ]
        )
        self.assertEqual(loaded, {"ffn.weight"})
        self.assertTrue(torch.all(model.ffn.weight == 1))

    def test_custom_loader_receives_its_own_checkpoint_format(self):
        class Backbone(nn.Module):
            checkpoint_name_mapping = {"mlp": "ffn"}

            def __init__(self):
                super().__init__()
                self.ffn = nn.Linear(2, 2, bias=False)
                self.seen = []

            def load_weights(self, weights):
                params = dict(self.named_parameters())
                map_name = get_checkpoint_name_mapper(self)
                loaded = set()
                for name, weight in weights:
                    self.seen.append(name)
                    target = map_name(name)
                    with torch.no_grad():
                        params[target].copy_(weight)
                    loaded.add(target)
                return loaded

        model = nn.Module()
        model.backbone = Backbone()
        loaded = AutoWeightsLoader(model).load_weights(
            [
                ("backbone.mlp.weight", torch.ones(2, 2)),
            ]
        )
        self.assertEqual(model.backbone.seen, ["mlp.weight"])
        self.assertEqual(loaded, {"backbone.ffn.weight"})
        self.assertTrue(torch.all(model.backbone.ffn.weight == 1))


if __name__ == "__main__":
    unittest.main()
