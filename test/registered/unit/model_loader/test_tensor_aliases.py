"""Tensor aliases (views kept as plain attributes, second references, second
registrations) must follow a parameter that a loader replaces."""

import collections
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.srt.model_loader.loader import PreshardedModelLoader
from sglang.srt.model_loader.tensor_aliases import (
    find_tensor_aliases,
    rebase_tensor_aliases,
)
from sglang.srt.weight_cache.ipc_loader import WeightCacheModelLoader
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

Pair = collections.namedtuple("Pair", "first second")


class Holder(nn.Module):
    """Keeps tensors the way attention modules do: as plain attributes."""


class Convolution(nn.Module):
    """A [channels, 1, kernel] weight with a 2-D view kept elsewhere."""

    def __init__(self, device="cpu", channels=6, kernel=4):
        super().__init__()
        self.projection = nn.Linear(kernel, channels, bias=False, device=device)
        # GDN and KDA layers reshape their convolution weight like this.
        self.projection.weight.data = self.projection.weight.data.unsqueeze(1)
        self.attn = Holder()


def replace(module: nn.Module, name: str, value: torch.Tensor) -> None:
    setattr(module, name, nn.Parameter(value, requires_grad=False))


class TestRebaseTensorAliases(unittest.TestCase):
    def test_view_follows_a_replaced_parameter_whatever_its_name(self):
        for cached_name in ("conv_weights", "cached_kernel"):
            with self.subTest(cached_name=cached_name):
                model = Convolution()
                setattr(model.attn, cached_name, model.projection.weight.squeeze(1))
                aliases = find_tensor_aliases(model)
                new = torch.arange(24.0).view(6, 1, 4)
                replace(model.projection, "weight", new)

                rebuilt, problems = rebase_tensor_aliases(model, aliases)

                self.assertEqual((rebuilt, problems), (1, []))
                view = getattr(model.attn, cached_name)
                self.assertEqual(view.data_ptr(), new.data_ptr())
                torch.testing.assert_close(view, new.squeeze(1))

    def test_views_held_in_containers(self):
        model = Convolution()
        weight = model.projection.weight.squeeze(1)
        model.attn.split = (weight[:2], weight[2:4], weight[4:])
        model.attn.pair = Pair(weight[:3], weight[3:])
        model.attn.listed = [weight.t()]
        model.attn.keyed = {"rows": weight[1]}
        aliases = find_tensor_aliases(model)
        new = torch.arange(24.0).view(6, 1, 4)
        replace(model.projection, "weight", new)

        rebuilt, problems = rebase_tensor_aliases(model, aliases)

        self.assertEqual((rebuilt, problems), (7, []))
        expected = new.squeeze(1)
        self.assertIs(type(model.attn.split), tuple)
        for got, want in zip(model.attn.split, expected.split(2)):
            torch.testing.assert_close(got, want)
        self.assertIsInstance(model.attn.pair, Pair)
        torch.testing.assert_close(model.attn.pair.second, expected[3:])
        torch.testing.assert_close(model.attn.listed[0], expected.t())
        torch.testing.assert_close(model.attn.keyed["rows"], expected[1])

    def test_second_reference_and_second_registration(self):
        model = Convolution()
        a_log = nn.Parameter(torch.zeros(3))
        model.A_log = a_log
        model.attn.A_log = a_log  # registered on both, like RadixLinearAttention
        model.attn.params = [a_log]
        aliases = find_tensor_aliases(model)
        replace(model, "A_log", torch.ones(3))

        rebuilt, problems = rebase_tensor_aliases(model, aliases)

        self.assertEqual((rebuilt, problems), (2, []))
        self.assertIs(model.attn.A_log, model.A_log)
        self.assertIs(model.attn.params[0], model.A_log)
        self.assertIn("A_log", model.attn._parameters)
        self.assertNotIn("params", model.attn._parameters)

    def test_registrations_replaced_separately_stay_separate(self):
        # The weight cache maps every name on its own.
        model = Convolution()
        model.A_log = nn.Parameter(torch.zeros(3))
        model.attn.A_log = model.A_log
        aliases = find_tensor_aliases(model)
        replace(model, "A_log", torch.ones(3))
        replace(model.attn, "A_log", torch.full((3,), 2.0))
        attn_a_log = model.attn.A_log

        self.assertEqual(rebase_tensor_aliases(model, aliases), (0, []))
        self.assertIs(model.attn.A_log, attn_a_log)

    def test_meta_model_on_mapped_tensors(self):
        with torch.device("meta"):
            model = Convolution()
        model.attn.conv_weights = model.projection.weight.squeeze(1)
        aliases = find_tensor_aliases(model)
        new = torch.randn(6, 1, 4)
        WeightCacheModelLoader._set_module_tensor(model, "projection.weight", new)

        WeightCacheModelLoader._rebuild_stale_views(model, aliases)

        self.assertFalse(model.attn.conv_weights.is_meta)
        self.assertEqual(model.attn.conv_weights.data_ptr(), new.data_ptr())

    def test_presharded_rebind_keeps_refreshing_gemma_norms(self):
        from sglang.srt.layers.layernorm import GemmaRMSNorm

        model = Convolution()
        model.norm = GemmaRMSNorm(4)
        model.attn.conv_weights = model.projection.weight.squeeze(1)
        aliases = find_tensor_aliases(model)
        model.norm.weight.data.fill_(0.5)
        new = torch.randn(6, 1, 4)
        replace(model.projection, "weight", new)

        PreshardedModelLoader._rebind_parameter_aliases(model, aliases)

        torch.testing.assert_close(model.norm.gemma_weight, torch.full((4,), 1.5))
        self.assertEqual(model.attn.conv_weights.data_ptr(), new.data_ptr())

    def test_copy_in_place_leaves_aliases_alone(self):
        model = Convolution()
        model.attn.conv_weights = model.projection.weight.squeeze(1)
        view = model.attn.conv_weights
        aliases = find_tensor_aliases(model)
        with torch.no_grad():
            model.projection.weight.copy_(torch.ones(6, 1, 4))

        self.assertEqual(rebase_tensor_aliases(model, aliases), (0, []))
        self.assertIs(model.attn.conv_weights, view)

    def test_alias_the_model_rebuilt_itself_is_left_alone(self):
        model = Convolution()
        model.attn.conv_weights = model.projection.weight.squeeze(1)
        aliases = find_tensor_aliases(model)
        replace(model.projection, "weight", torch.ones(6, 1, 4))
        own = torch.zeros(6, 4)
        model.attn.conv_weights = own

        self.assertEqual(rebase_tensor_aliases(model, aliases), (0, []))
        self.assertIs(model.attn.conv_weights, own)

    def test_tensors_that_are_not_aliases_are_ignored(self):
        model = Convolution()
        model.attn.scale = torch.ones(6)  # its own storage
        model.attn.doubled = model.projection.weight.squeeze(1) * 2  # a copy

        self.assertEqual(find_tensor_aliases(model), [])
        replace(model.projection, "weight", torch.ones(6, 1, 4))
        self.assertEqual(rebase_tensor_aliases(model, []), (0, []))
        torch.testing.assert_close(model.attn.scale, torch.ones(6))

    def test_stale_alias_that_cannot_be_rebuilt_is_reported(self):
        model = Convolution()
        model.attn.rows = model.projection.weight.squeeze(1)[:3]
        storage = torch.zeros(10)
        model.first = nn.Parameter(storage[:5])
        model.second = nn.Parameter(storage[5:])
        model.attn.across = storage[3:7]  # spans two parameters
        aliases = find_tensor_aliases(model)
        replace(model.projection, "weight", torch.ones(24))  # new shape
        replace(model, "first", torch.ones(5))

        rebuilt, problems = rebase_tensor_aliases(model, aliases)

        self.assertEqual(rebuilt, 0)
        self.assertEqual(len(problems), 2, problems)
        self.assertIn("attn.rows: projection.weight changed", problems[0])
        self.assertIn("attn.across: it spans more than one tensor", problems[1])
        self.assertEqual(model.attn.rows.shape, (3, 4))

    def test_a_whole_view_follows_a_reshaped_target(self):
        # Post-processing can drop the convolution's unit dimension.
        model = Convolution()
        model.attn.conv_weights = model.projection.weight.squeeze(1)
        aliases = find_tensor_aliases(model)
        new = torch.arange(24.0).view(6, 4)
        replace(model.projection, "weight", new)

        self.assertEqual(rebase_tensor_aliases(model, aliases), (1, []))
        self.assertEqual(model.attn.conv_weights.data_ptr(), new.data_ptr())
        torch.testing.assert_close(model.attn.conv_weights, new)

    def test_a_container_reached_from_two_attributes(self):
        with torch.device("meta"):
            model = Convolution()
        model.attn.a = (model.projection.weight.squeeze(1),)
        model.attn.b = model.attn.a
        aliases = find_tensor_aliases(model)
        replace(model.projection, "weight", torch.ones(6, 1, 4))

        self.assertEqual(rebase_tensor_aliases(model, aliases), (2, []))
        self.assertFalse(model.attn.a[0].is_meta)
        self.assertFalse(model.attn.b[0].is_meta)

    def test_a_view_follows_a_name_rejoined_to_its_replacement(self):
        model = Convolution()
        model.attn.weight = model.projection.weight  # a second registration
        model.attn.conv_weights = model.projection.weight.squeeze(1)
        aliases = find_tensor_aliases(model)
        new = torch.full((6, 1, 4), 9.0)
        replace(model.attn, "weight", new)  # only the second name

        self.assertEqual(rebase_tensor_aliases(model, aliases), (2, []))
        self.assertIs(model.projection.weight, model.attn.weight)
        self.assertEqual(model.attn.conv_weights.data_ptr(), new.data_ptr())

    def test_registrations_replaced_by_different_tensors_are_reported(self):
        model = Convolution()
        a_log = nn.Parameter(torch.zeros(3))
        model.A_log = model.attn.A_log = model.attn.again = a_log
        aliases = find_tensor_aliases(model)
        replace(model, "A_log", torch.ones(3))
        replace(model.attn, "A_log", torch.full((3,), 2.0))

        rebuilt, problems = rebase_tensor_aliases(model, aliases)

        self.assertEqual(rebuilt, 0)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("attn.again: still the tensor that 2 different", problems[0])

    def test_a_registered_view_follows(self):
        with torch.device("meta"):
            model = Convolution()
        model.register_buffer("rows", model.projection.weight.squeeze(1)[:2])
        aliases = find_tensor_aliases(model)
        new = torch.arange(24.0).view(6, 1, 4)
        replace(model.projection, "weight", new)

        self.assertEqual(rebase_tensor_aliases(model, aliases), (1, []))
        self.assertIn("rows", model._buffers)
        torch.testing.assert_close(model.rows, new.squeeze(1)[:2])

    def test_references_to_a_rebuilt_registered_view_follow_it(self):
        with torch.device("meta"):
            model = nn.Module()
            model.w = nn.Parameter(torch.empty(4), requires_grad=False)
        model.register_buffer("b", model.w[:2])
        model.register_buffer("c", model.b)  # the view registered again
        model.cache = [model.b]
        aliases = find_tensor_aliases(model)
        replace(model, "w", torch.arange(4.0))

        self.assertEqual(rebase_tensor_aliases(model, aliases), (3, []))
        self.assertIs(model.c, model.b)
        self.assertIs(model.cache[0], model.b)
        torch.testing.assert_close(model.b, torch.arange(2.0))

    def test_a_buffer_that_holds_a_parameter_stays_a_buffer(self):
        class Legacy(nn.Module):
            def register_buffer(self, name, tensor):  # before ``persistent``
                super().register_buffer(name, tensor)

        class PositionalOnly(nn.Module):
            def register_buffer(self, name, tensor, persistent=True, /):
                super().register_buffer(name, tensor, persistent)

        for make in (nn.Module, Legacy, PositionalOnly):
            for persistent in (True, False):
                with self.subTest(module=make.__name__, persistent=persistent):
                    model = make()
                    model.w = nn.Parameter(torch.zeros(4), requires_grad=False)
                    nn.Module.register_buffer(model, "b", model.w, persistent)
                    aliases = find_tensor_aliases(model)
                    replace(model, "w", torch.ones(4))

                    self.assertEqual(rebase_tensor_aliases(model, aliases), (1, []))
                    self.assertEqual(
                        (list(model._parameters), list(model._buffers)),
                        (["w"], ["b"]),
                    )
                    self.assertEqual(
                        "b" in model._non_persistent_buffers_set, not persistent
                    )
                    self.assertIs(model.b, model.w)

    def test_an_alias_the_loader_repaired_in_place_is_left_alone(self):
        model = nn.Module()
        model.w = nn.Parameter(torch.arange(4.0), requires_grad=False)
        model.cache = model.w[:2]
        cache = model.cache
        aliases = find_tensor_aliases(model)
        replace(model, "w", torch.full((4,), 9.0))
        with torch.no_grad():
            cache.set_(model.w.untyped_storage(), 0, (2,), (1,))

        self.assertEqual(rebase_tensor_aliases(model, aliases), (0, []))
        self.assertIs(model.cache, cache)
        torch.testing.assert_close(model.cache, torch.full((2,), 9.0))

    def test_dict_subclasses_any_keys_deep_containers_and_empty_views(self):
        with torch.device("meta"):
            model = Convolution()
        view = model.projection.weight.squeeze(1)
        model.attn.a = {("x", 0): view, 0.5: view}
        model.attn.b = collections.OrderedDict(x=view)
        model.attn.c = [[[[view]]]]
        model.attn.empty = view[:0]
        aliases = find_tensor_aliases(model)
        replace(model.projection, "weight", torch.ones(6, 1, 4))

        self.assertEqual(rebase_tensor_aliases(model, aliases), (5, []))
        for tensor in (
            model.attn.a[("x", 0)],
            model.attn.a[0.5],
            model.attn.b["x"],
            model.attn.c[0][0][0][0],
            model.attn.empty,
        ):
            self.assertFalse(tensor.is_meta)
        self.assertEqual(model.attn.empty.shape, (0, 4))

    def test_a_container_that_hides_its_items_does_not_hide_an_alias(self):
        class Hidden(dict):
            def items(self):
                return iter(())

            def __iter__(self):
                return iter(())

            def __getitem__(self, key):
                raise KeyError(key)

        with torch.device("meta"):
            model = Convolution()
        model.attn.cache = Hidden(conv=model.projection.weight.squeeze(1))
        aliases = find_tensor_aliases(model)
        replace(model.projection, "weight", torch.ones(6, 1, 4))

        self.assertEqual(rebase_tensor_aliases(model, aliases), (1, []))
        self.assertFalse(dict.get(model.attn.cache, "conv").is_meta)

    def test_an_alias_follows_a_holder_that_replaced_its_container(self):
        model = nn.Module()
        model.w = nn.Parameter(torch.arange(4.0), requires_grad=False)
        model.cache = [model.w[:2]]
        aliases = find_tensor_aliases(model)
        model.cache = collections.deque(model.cache)
        replace(model, "w", torch.full((4,), 9.0))

        self.assertEqual(rebase_tensor_aliases(model, aliases), (1, []))
        torch.testing.assert_close(model.cache[0], torch.full((2,), 9.0))

    def test_conjugate_and_negative_views_are_refused(self):
        for make in (torch.conj, torch._neg_view):
            with self.subTest(view=make.__name__):
                model = nn.Module()
                model.weight = nn.Parameter(torch.tensor([1 + 2j, 3 + 4j]))
                model.view = make(model.weight)
                aliases = find_tensor_aliases(model)
                replace(model, "weight", torch.tensor([5 + 6j, 7 + 8j]))

                rebuilt, problems = rebase_tensor_aliases(model, aliases)

                self.assertEqual(rebuilt, 0)
                self.assertIn("conjugate or negative", problems[0])

    def test_a_view_remade_on_the_same_storage_is_left_alone(self):
        model = Convolution()
        model.attn.rows = model.projection.weight.squeeze(1)[:2]
        aliases = find_tensor_aliases(model)
        remade = model.projection.weight.squeeze(1)[4:]
        model.attn.rows = remade
        replace(model.projection, "weight", torch.ones(6, 1, 4))

        self.assertEqual(rebase_tensor_aliases(model, aliases), (0, []))
        self.assertIs(model.attn.rows, remade)

    def test_a_view_follows_the_tensor_it_was_made_from(self):
        model = nn.Module()
        model.weight = nn.Parameter(torch.arange(8.0))
        model.right = nn.Parameter(model.weight.data)  # same bytes
        model.view = model.right[:4]
        aliases = find_tensor_aliases(model)
        replace(model, "weight", torch.ones(8))
        replace(model, "right", torch.full((8,), 2.0))

        self.assertEqual(rebase_tensor_aliases(model, aliases), (1, []))
        torch.testing.assert_close(model.view, torch.full((4,), 2.0))

    def test_an_unprovable_choice_between_tensors_is_reported(self):
        model = nn.Module()
        model.weight = nn.Parameter(torch.arange(8.0))
        model.right = nn.Parameter(model.weight.data)
        model.view = model.right.detach()[:4]  # detach() keeps no base
        aliases = find_tensor_aliases(model)
        replace(model, "weight", torch.ones(8))
        replace(model, "right", torch.full((8,), 2.0))

        rebuilt, problems = rebase_tensor_aliases(model, aliases)

        self.assertEqual(rebuilt, 0)
        self.assertIn("view: it could have been made from", problems[0])

    def test_a_dtype_view_aligned_in_its_storage_is_rebuilt(self):
        model = nn.Module()
        base = torch.arange(16, dtype=torch.uint8)
        model.register_buffer("weight", base[1:13])
        model.view = base[4:12].view(torch.int32)
        aliases = find_tensor_aliases(model)
        new_base = torch.arange(16, dtype=torch.uint8) + 16
        model.weight = new_base[1:13]

        self.assertEqual(rebase_tensor_aliases(model, aliases), (1, []))
        torch.testing.assert_close(model.view, new_base[4:12].view(torch.int32))

    def test_a_tensor_subclass_without_storage_does_not_break_the_scan(self):
        class NoStorage(torch.Tensor):
            def untyped_storage(self):
                raise RuntimeError("no storage")

        model = nn.Module()
        tensor = torch.ones(2).as_subclass(NoStorage)
        model.register_buffer("a", tensor)
        model.register_buffer("b", tensor)

        self.assertEqual(len(find_tensor_aliases(model)), 1)

    def test_presharded_rebind_requires_the_record_and_fails_loudly(self):
        model = Convolution()
        model.attn.rows = model.projection.weight.squeeze(1)[:3]
        aliases = find_tensor_aliases(model)
        replace(model.projection, "weight", torch.ones(24))

        with self.assertRaises(TypeError):
            PreshardedModelLoader._rebind_parameter_aliases(model)
        with self.assertRaisesRegex(RuntimeError, "attn.rows: projection.weight"):
            PreshardedModelLoader._rebind_parameter_aliases(model, aliases)

    def test_ipc_rejoins_a_buffer_several_modules_registered(self):
        # Every layer registers the same cached table; the daemon exports it
        # under its first name only.
        from sglang.srt.weight_cache import ipc_loader

        with torch.device("meta"):
            model = nn.Module()
            table = torch.empty(4)
            model.a, model.b = nn.Module(), nn.Module()
        model.a.register_buffer("table", table, persistent=False)
        model.b.register_buffer("table", table, persistent=False)
        loaded = torch.arange(4.0)
        loader = WeightCacheModelLoader.__new__(WeightCacheModelLoader)
        loader.load_config = None
        loader._transport_backend = mock.Mock(import_tensor=lambda e: e["tensor"])
        entries = {"a.table": {"tensor": loaded, "is_param": False}}
        with mock.patch.object(ipc_loader, "_initialize_model", return_value=model):
            out = loader._load_zero_copy_mode(
                SimpleNamespace(dtype=torch.float32), None, entries, None
            )

        self.assertIs(out.b.table, out.a.table)
        torch.testing.assert_close(out.b.table, loaded)

    def test_alias_still_on_meta_is_reported(self):
        with torch.device("meta"):
            model = Convolution()
        model.attn.conv_weights = model.projection.weight.squeeze(1)
        aliases = find_tensor_aliases(model)
        replace(model.projection, "weight", torch.empty(6, 1, 4, device="meta"))

        rebuilt, problems = rebase_tensor_aliases(model, aliases)

        self.assertEqual(rebuilt, 0)
        self.assertEqual(
            problems,
            ["attn.conv_weights: projection.weight is still on the meta device"],
        )


if __name__ == "__main__":
    unittest.main()
