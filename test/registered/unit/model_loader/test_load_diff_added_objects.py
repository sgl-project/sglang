"""load_diff, format 2: the objects loading may add. A module of a listed
SGLang class (made again without running its constructor), a NamedTuple of a
listed class, and an object the model was built with that loading moved."""

import sys
import types
import unittest
from collections import defaultdict
from functools import partial
from typing import Any, NamedTuple, Optional
from unittest import mock

import torch
from torch import nn

from sglang.srt.weight_cache import load_diff
from sglang.srt.weight_cache.load_diff_format import (
    REBUILDABLE_CLASSES,
    NewModule,
    SnapshotError,
    decode_manifest,
    encode_manifest,
)
from sglang.srt.weight_cache.schema import SnapshotManifest, StorageMeta
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# A module "of SGLang" for the classes below. It is an imported module, and
# its listed classes are in REBUILDABLE_CLASSES, only while a test class runs
# (AddedObjectsCase).
HOME = "sglang._test_added_objects"
_home = types.ModuleType(HOME)
_listed = set()


def sglang_class(cls):
    cls.__module__ = HOME
    setattr(_home, cls.__qualname__, cls)
    return cls


def rebuildable(cls):
    _listed.add((HOME, cls.__qualname__))
    return sglang_class(cls)


class AddedObjectsCase(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        imported = mock.patch.dict(sys.modules, {HOME: _home})
        imported.start()
        cls.addClassCleanup(imported.stop)
        added = _listed - REBUILDABLE_CLASSES
        REBUILDABLE_CLASSES.update(added)
        cls.addClassCleanup(REBUILDABLE_CLASSES.difference_update, added)


class Method:
    """Like a quant method: built with its layer, a kernel picked then."""

    def __init__(self):
        self.kernel = nn.functional.linear
        self.use_mxfp8 = False


@rebuildable
class Fused(nn.Module):
    """Like MiniMax-M3's fused projection: made by loading, from loaded
    weights and the quant method of a layer it replaces."""

    made = 0

    def __init__(self, method, weight, widths):
        super().__init__()
        type(self).made += 1
        self._qm = method
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight.format_ue8m0 = True
        self.widths = widths

    def forward(self, x):
        return self._qm.kernel(x, self.weight)


@rebuildable
class Op(nn.Module):
    """Like Muse-Spark's finalize op: plain values, resolved on first call."""

    def __init__(self, topk, eps):
        super().__init__()
        self.topk, self.eps = topk, eps
        self._forward_method = None


@sglang_class
class Unlisted(nn.Module):
    pass


@sglang_class
class Cfg(NamedTuple):
    mult: int
    eps: float


@rebuildable
class Hc(NamedTuple):
    """Like DeepSeek-V4.1's HcSubLayer: refers to the layer's parameters."""

    cfg: Cfg
    fn: torch.Tensor
    norm: nn.Module
    parts: Optional[Any] = None


class Foreign(NamedTuple):  # not a class of SGLang
    x: int


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(4, 2, bias=False)
        self.idx = nn.Linear(4, 3, bias=False)
        self.q.quant_method, self.idx.quant_method = Method(), Method()
        self._fused = None
        self._op = None
        self.fn = nn.Parameter(torch.zeros(2, 4), requires_grad=False)
        self.norm = nn.LayerNorm(4)
        self.cfg = Cfg(2, 1e-6)
        self.hc = Hc(self.cfg, self.fn, self.norm)

    def forward(self, x):
        hi, lo = self.hc.parts
        fused = self._fused(self.hc.norm(x)).sum() * self._op.topk
        return fused + (x @ (hi + lo).t()).sum() * self.hc.cfg.mult


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer(), Layer()])

    def forward(self, x):
        return sum(layer(x) for layer in self.layers)


def load(model):
    torch.manual_seed(0)
    for layer in model.layers:
        with torch.no_grad():
            layer.q.weight.copy_(torch.randn(2, 4))
            layer.idx.weight.copy_(torch.randn(3, 4))
            layer.fn.copy_(torch.randn(2, 4))
        # One module from two projections, under two names; they are emptied.
        weight = torch.cat([layer.q.weight.data, layer.idx.weight.data]).contiguous()
        fused = Fused(layer.q.quant_method, weight, [2, 3])
        layer.add_module("fused", fused)
        layer._fused = fused
        for linear in (layer.q, layer.idx):
            linear.quant_method = None
            linear.weight.data = torch.empty(0)
        # A plain attribute that was None becomes a module.
        layer._op = Op(topk=2, eps=1e-6).eval()
        # The NamedTuple is made again, with tensors derived from a weight.
        hi = layer.fn.data.bfloat16().float()
        layer.hc = Hc(layer.cfg, layer.fn, layer.norm, (hi, layer.fn.data - hi))
        layer.hc_again = layer.hc  # one NamedTuple at two places


def exported(change=None, prepare=None):
    """``prepare`` changes a layer before the model counts as built, ``change``
    after it is loaded."""
    owner = Toy()
    if prepare is not None:
        prepare(owner.layers[0])
    constructed = load_diff.record_construction(owner)
    load(owner)
    if change is not None:
        change(owner.layers[0])
    storages, payload = load_diff.export(model=owner, constructed=constructed)
    return owner, payload, storages


def skeleton():
    with torch.device("meta"):
        return Toy()


def snapshot_of(payload, storages):
    return SnapshotManifest(
        version="v",
        spec_digest="",
        storages={
            name: StorageMeta(nbytes=t.nbytes, device=t.device.type)
            for name, t in storages.items()
        },
        load_diff=payload,
        attestation=b"",
        source={},
    )


def install(reader, payload, storages):
    snapshot = snapshot_of(payload, storages)
    copies = {name: t.clone() for name, t in storages.items()}
    return load_diff.install(skeleton=reader, manifest=snapshot, storages=copies)


def preflight(reader, payload, storages):
    load_diff.preflight(skeleton=reader, manifest=snapshot_of(payload, storages))


class TestRoundTrip(AddedObjectsCase):
    def test_a_dry_run_leaves_the_skeleton_as_built(self):
        owner, payload, storages = exported()
        reader = skeleton()
        made = Fused.made
        preflight(reader, payload, storages)
        layer = reader.layers[0]
        self.assertEqual(Fused.made, made)
        self.assertEqual(
            [name for name, _ in layer.named_children()], ["q", "idx", "norm"]
        )
        self.assertIsNone(layer._fused)
        self.assertIsNotNone(layer.q.quant_method)
        self.assertTrue(all(p.is_meta for p in reader.parameters()))
        # The skeleton still installs afterwards
        install(reader, payload, storages)
        x = torch.ones(1, 4)
        torch.testing.assert_close(reader(x), owner(x))

    def test_reader_holds_what_the_owner_loaded(self):
        owner, payload, storages = exported()
        reader = skeleton()
        methods = [layer.q.quant_method for layer in reader.layers]
        made = Fused.made
        install(reader, payload, storages)
        self.assertEqual(Fused.made, made)  # no constructor ran on the reader
        x = torch.randn(5, 4)
        self.assertTrue(torch.equal(reader(x), owner(x)))
        for layer, method, theirs in zip(reader.layers, methods, owner.layers):
            self.assertIs(type(layer.fused), Fused)
            self.assertIs(layer._fused, layer.fused)
            self.assertNotIn("_fused", vars(layer))
            self.assertNotIn("_op", vars(layer))
            self.assertIs(layer.fused._qm, method)  # the reader's own, as built
            self.assertIsNone(layer.q.quant_method)
            self.assertEqual(layer.q.weight.numel(), 0)
            self.assertIs(layer.fused.weight.format_ue8m0, True)
            self.assertEqual(layer.fused.widths, [2, 3])
            op = layer._op
            self.assertEqual(
                (type(op), op.topk, op.eps, op._forward_method), (Op, 2, 1e-6, None)
            )
            self.assertIs(layer.fused.training, True)
            self.assertIs(op.training, False)
            hc = layer.hc
            self.assertIs(type(hc), Hc)
            self.assertIs(layer.hc_again, hc)
            self.assertIs(hc.cfg, layer.cfg)
            self.assertIs(hc.fn, layer.fn)
            self.assertIs(hc.norm, layer.norm)
            self.assertIs(type(hc.parts), tuple)
            for mine, other in zip(hc.parts, theirs.hc.parts):
                self.assertTrue(torch.equal(mine, other))

    def test_a_tensor_of_indices_is_data(self):
        _, payload, storages = exported(
            lambda layer: setattr(layer, "order", torch.tensor([2, 0, 1]))
        )
        reader = skeleton()
        install(reader, payload, storages)
        self.assertEqual(reader.layers[0].order.tolist(), [2, 0, 1])

    def test_the_manifest_names_the_added_modules_and_the_moved_objects(self):
        _, payload, _ = exported()
        m = decode_manifest(payload)
        self.assertEqual(
            [(new.names, new.qualname) for new in m.new_modules[:2]],
            [
                (["layers.0.fused", "layers.0._fused"], "Fused"),
                (["layers.0._op"], "Op"),
            ],
        )
        self.assertEqual(
            m.built,
            [[f"layers.{i}.q", [["a", "quant_method"]]] for i in range(2)],
        )


class TestOwnerRefuses(AddedObjectsCase):
    def refused(self, change, prepare=None):
        with self.assertRaises(SnapshotError) as caught:
            exported(change, prepare)
        return "\n".join(caught.exception.problems)

    def test_a_module_of_a_class_outside_sglang(self):
        why = self.refused(lambda layer: setattr(layer, "extra", nn.Linear(2, 2)))
        self.assertIn("layers.0.extra: a Linear module that loading added", why)

    def test_a_module_of_a_class_that_is_not_listed(self):
        why = self.refused(lambda layer: setattr(layer, "extra", Unlisted()))
        self.assertIn("layers.0.extra: a Unlisted module that loading added", why)

    def test_a_namedtuple_of_a_class_that_is_not_listed(self):
        why = self.refused(
            lambda layer: setattr(layer, "hc", layer.hc._replace(cfg=Cfg(3, 1.0)))
        )
        self.assertIn("layers.0.hc: loading left a", why)

    def test_a_replaced_module(self):
        why = self.refused(lambda layer: setattr(layer, "norm", nn.LayerNorm(4)))
        self.assertIn("layers.0.norm: a LayerNorm module that loading replaced", why)

    def test_a_built_module_under_another_name(self):
        why = self.refused(lambda layer: setattr(layer, "alias", layer.norm))
        self.assertIn("layers.0.alias", why)
        self.assertIn("put under another name", why)

    def test_a_hook_on_an_added_module(self):
        why = self.refused(
            lambda layer: layer.fused.register_forward_hook(lambda *a: None)
        )
        self.assertIn("layers.0.fused: loading added, removed or reordered hooks", why)

    def test_a_moved_object_that_loading_also_changed(self):
        why = self.refused(lambda layer: setattr(layer.fused._qm, "use_mxfp8", True))
        self.assertIn("layers.0.fused._qm", why)
        self.assertIn("changed it", why)

    def test_a_tensor_of_addresses(self):
        # Like the bases of shards mapped from other ranks, which a kernel follows
        def change(layer):
            layer.bases = torch.tensor([7, layer.fn.data_ptr() + 4], dtype=torch.int64)

        why = self.refused(change)
        self.assertIn("layers.0.bases: loading left a tensor that holds", why)
        self.assertIn("an address inside a storage of the model", why)

    def test_addresses_put_in_a_tensor_the_model_was_built_with(self):
        def built(layer):
            layer.register_buffer("table", torch.zeros(2, dtype=torch.int64))

        def replaced(layer):
            layer.table = torch.tensor([7, layer.fn.data_ptr() + 4])

        def filled(layer):
            layer.table[1] = layer.fn.data_ptr() + 4

        for change in (replaced, filled):
            with self.subTest(change=change.__name__):
                why = self.refused(change, built)
                self.assertIn("layers.0.table: loading left a tensor that holds", why)

    def test_a_moved_object_whose_call_or_factory_loading_changed(self):
        def built(layer):
            layer.q.quant_method.scale = partial(torch.mul, other=2)
            layer.q.quant_method.cache = defaultdict(int)

        def other_keyword(layer):
            layer.fused._qm.scale.keywords["other"] = 5

        def other_factory(layer):
            layer.fused._qm.cache.default_factory = list

        for change in (other_keyword, other_factory):
            with self.subTest(change=change.__name__):
                why = self.refused(change, built)
                self.assertIn("layers.0.fused._qm", why)
                self.assertIn("changed it", why)

    def test_a_list_of_a_moved_object_put_at_another_place(self):
        # The reader keeps its own list in the object; a copy made by value at
        # the new place would no longer be that list.
        def built(layer):
            layer.q.quant_method.sizes = [2]

        why = self.refused(
            lambda layer: setattr(layer, "sizes", layer.fused._qm.sizes), built
        )
        self.assertIn("layers.0.sizes: loading put here what the moved object", why)

    def test_added_modules_in_an_order_a_reader_would_not_repeat(self):
        # fused, _fused, _op, again on the owner; a reader puts every name of
        # one module together.
        why = self.refused(lambda layer: layer.add_module("again", layer.fused))
        self.assertIn("layers.0: loading added modules among its children", why)

    def test_an_added_module_with_an_empty_child_slot(self):
        why = self.refused(lambda layer: layer.fused.register_module("post", None))
        self.assertIn("layers.0.fused: a Fused module that loading added", why)
        self.assertIn("empty child slot", why)

    def test_a_built_module_from_a_list_put_among_the_children(self):
        # Built with the model, so whatever refers to it must stay with it: a
        # reader would make a second one.
        def built(layer):
            layer.pending = [Op(topk=1, eps=0.0)]

        why = self.refused(
            lambda layer: setattr(layer, "later", layer.pending.pop()), built
        )
        self.assertIn("layers.0.later: a Op module that loading put under", why)

    def test_a_namedtuple_of_a_class_outside_sglang(self):
        why = self.refused(lambda layer: setattr(layer, "hc", Foreign(1)))
        self.assertIn("layers.0.hc: loading left a", why)
        self.assertIn("Foreign", why)

    def test_a_namedtuple_that_holds_a_function(self):
        why = self.refused(
            lambda layer: setattr(layer, "hc", layer.hc._replace(parts=len))
        )
        self.assertIn("layers.0.hc: loading left a", why)


class TestReaderRefuses(AddedObjectsCase):
    def refused(self, edit):
        _, payload, storages = exported()
        m = decode_manifest(payload)
        edit(m)
        with self.assertRaises(SnapshotError) as caught:
            install(skeleton(), encode_manifest(m), storages)
        # A dry run refuses it too, before any byte moves
        with self.assertRaises(SnapshotError) as dry:
            preflight(skeleton(), encode_manifest(m), storages)
        self.assertEqual(dry.exception.problems, caught.exception.problems)
        return "\n".join(caught.exception.problems)

    def test_a_class_this_process_does_not_have(self):
        def edit(m):
            new = m.new_modules[0]
            m.new_modules[0] = NewModule(new.names, new.module, "Missing", new.training)

        REBUILDABLE_CLASSES.add((HOME, "Missing"))
        try:
            self.assertIn("Missing does not exist here", self.refused(edit))
        finally:
            REBUILDABLE_CLASSES.discard((HOME, "Missing"))

    def test_a_class_outside_sglang(self):
        def edit(m):
            new = m.new_modules[0]
            m.new_modules[0] = NewModule(
                new.names, "torch.nn.modules.linear", "Linear", new.training
            )

        self.assertIn("is not a class a reader makes again", self.refused(edit))

    def test_a_moved_object_this_reader_did_not_build(self):
        def edit(m):
            m.built[0] = ["layers.0.q", [["a", "missing"]]]

        self.assertIn("layers.0.q.missing", self.refused(edit))


if __name__ == "__main__":
    unittest.main()
