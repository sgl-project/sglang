"""CPU regression tests for layered checkpoint iteration and materialization."""

import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import sglang.srt.model_loader.loader as loader_mod
from sglang.srt.model_loader.loader import LayeredModelLoader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _CheckpointModel(nn.Module):
    """Small model with the same per-module filtering contract as TorchNative."""

    def __init__(self, names):
        super().__init__()
        self.loaded_names = set()
        for name in names:
            module = self
            components = name.split(".")
            for component in components[:-1]:
                if not hasattr(module, component):
                    module.add_module(component, nn.Module())
                module = getattr(module, component)
            module.register_parameter(components[-1], nn.Parameter(torch.empty(1)))

    def load_weights_to_module(self, fqn, weights):
        module = self.get_submodule(fqn)
        parameters = dict(module.named_parameters(prefix=fqn, recurse=False))
        for name, weight in weights:
            if name in parameters:
                parameters[name].data.copy_(weight)
                self.loaded_names.add(name)


class TestLayeredModelLoader(CustomTestCase):
    def load_model(self, model_factory, weights_factory):
        loader = object.__new__(LayeredModelLoader)
        loader.load_config = object()
        model_config = SimpleNamespace(dtype=torch.float32, model_path="tiny-fixture")
        with (
            patch.object(loader_mod, "_get_quantization_config", return_value=None),
            patch.object(
                loader_mod, "_initialize_model", side_effect=lambda *_: model_factory()
            ),
            patch.object(
                loader, "_get_all_weights", side_effect=lambda *_: weights_factory()
            ),
        ):
            return loader.load_model(
                model_config=model_config,
                device_config=SimpleNamespace(device="cpu"),
            )

    def check_checkpoint(self, expected, order):
        def weights():
            # A real one-shot generator, rather than a reusable list of tensors.
            for name in order:
                yield name, torch.tensor([expected[name]], device="cpu")

        model = self.load_model(lambda: _CheckpointModel(expected), weights)
        self.assertEqual(model.loaded_names, set(expected))
        self.assertFalse(model.training)
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.device.type, "cpu")
            self.assertEqual(parameter.item(), expected[name])

    def test_every_sibling_receives_checkpoint_weights(self):
        expected = {"a.weight": 11.0, "b.weight": 22.0}
        self.check_checkpoint(expected, list(expected))

    def test_nested_modules_and_parent_parameters_in_any_order(self):
        expected = {
            "root_weight": 1.0,
            "block.parent_weight": 2.0,
            "block.left.weight": 11.0,
            "block.left.bias": 12.0,
            "block.right.weight": 21.0,
            "block.right.bias": 22.0,
            "last.weight": 31.0,
        }
        names = list(expected)
        orders = {
            "forward": names,
            "reverse": names[::-1],
            "interleaved": names[::2] + names[1::2],
        }
        for order_name, order in orders.items():
            with self.subTest(order=order_name):
                self.check_checkpoint(expected, order)

    def test_checkpoint_tensors_are_not_retained_between_modules(self):
        expected = {f"layer_{index}.weight": float(index) for index in range(32)}
        live_refs = []
        peak_live = 0

        def weights():
            nonlocal peak_live
            for name, value in reversed(list(expected.items())):
                weight = torch.tensor([value], device="cpu")
                live_refs[:] = [ref for ref in live_refs if ref() is not None]
                live_refs.append(weakref.ref(weight))
                peak_live = max(peak_live, len(live_refs))
                yield name, weight

        model = self.load_model(lambda: _CheckpointModel(expected), weights)
        self.assertEqual(model.loaded_names, set(expected))
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.item(), expected[name])
        # A generator and its consumer may each hold the current/previous item.
        # list(), dict(), or tee() caching the checkpoint would keep all 32.
        self.assertLessEqual(peak_live, 2)
        self.assertTrue(all(ref() is None for ref in live_refs))

    def test_materialization_preserves_custom_shard_loader(self):
        def load_shard(parameter, weight, shard_id):
            parameter.data[shard_id : shard_id + 1].copy_(weight)

        class ShardedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Module()
                self.projection.weight = nn.Parameter(torch.empty(2))
                self.projection.weight.weight_loader = load_shard

            def load_weights_to_module(self, fqn, weights):
                parameters = dict(
                    self.get_submodule(fqn).named_parameters(prefix=fqn, recurse=False)
                )
                for name, weight in weights:
                    if "projection.weight" in parameters:
                        parameter = parameters["projection.weight"]
                        parameter.weight_loader(parameter, weight, int(name))

        def weights():
            yield "1", torch.tensor([22.0], device="cpu")
            yield "0", torch.tensor([11.0], device="cpu")

        model = self.load_model(ShardedModel, weights)
        torch.testing.assert_close(
            model.projection.weight,
            torch.tensor([11.0, 22.0]),
            rtol=0,
            atol=0,
        )

    def test_materialization_preserves_metadata_for_parameter_aliases(self):
        def load_weight(parameter, weight):
            parameter.data.copy_(weight * parameter.load_multiplier)

        class AliasedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = nn.Parameter(torch.empty(1))
                self.first.weight_loader = load_weight
                self.first.load_multiplier = 3
                self.second = self.first

            def load_weights_to_module(self, fqn, weights):
                parameters = dict(
                    self.get_submodule(fqn).named_parameters(
                        prefix=fqn, recurse=False, remove_duplicate=False
                    )
                )
                for name, weight in weights:
                    parameter = parameters[name]
                    parameter.weight_loader(parameter, weight)

        def weights():
            yield "first", torch.tensor([11.0], device="cpu")
            yield "second", torch.tensor([11.0], device="cpu")

        model = self.load_model(AliasedModel, weights)
        # to_empty may replace the two registrations with distinct Parameters;
        # both names must retain the metadata needed to consume their weights.
        self.assertEqual(model.first.item(), 33.0)
        self.assertEqual(model.second.item(), 33.0)


if __name__ == "__main__":
    unittest.main()
