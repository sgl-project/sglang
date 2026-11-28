"""CPU tests for the validator of snapshot loading (weight_cache/validator.py)."""

import array
import collections
import enum
import functools
import logging
import operator
import pathlib
import re
import struct
import threading
import unittest
import zlib
from collections import OrderedDict
from types import SimpleNamespace
from unittest import mock

import msgspec
import numpy as np
import torch
from torch import nn

from sglang.srt.weight_cache import validator
from sglang.srt.weight_cache.validator import ValidationError
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_expert_map = validator._expert_map


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 3))
        self.scale = nn.Parameter(torch.zeros(1))
        self.register_buffer("table", torch.zeros(6), persistent=False)
        self.w_kc = None  # the load derives it from the weight, as MLA does
        self.fused = False


class _Model(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.attn = _Attention()
        self.head = nn.Linear(3, 4, bias=False)
        self.head.weight = self.attn.weight  # tied
        self.checkpoint = checkpoint  # as typed: differs between correct processes


class _OtherLinear(nn.Linear):
    pass


class _Scale(nn.Module):
    def __init__(self, by):
        super().__init__()
        self.by = by


class _Slotted(nn.Module):
    __slots__ = ("gain",)


class _SlottedTwice(_Slotted):
    __slots__ = ("gain",)  # the name again: a second slot


class _OtherParameter(nn.Parameter):
    pass


class _Numbers(list):
    __slots__ = ("scale",)


class _Gain(float):
    def hex(self):
        raise AssertionError("the walk asked the object")


class _Flags(set):
    pass


def _raw(tensor):
    """The whole allocation of a tensor as bytes, as an export hands it over."""
    return torch.empty(0, dtype=torch.uint8).set_(tensor.untyped_storage())


def _raw_load(model):
    with torch.no_grad():
        model.attn.weight.copy_(torch.arange(12.0).reshape(4, 3))
        model.attn.scale.fill_(0.5)
        model.attn.table.copy_(torch.arange(6.0))
    model.attn.scale.format_ue8m0 = True
    model.attn.w_kc = model.attn.weight.detach().t()[:2]  # a strided view
    model.attn.fused = True
    return model.eval()


def _publish(model):
    views = {}
    for tensor in (*model.parameters(), *model.buffers(), model.attn.w_kc):
        views.setdefault(tensor.untyped_storage().data_ptr(), _raw(tensor))
    return {str(i): view for i, view in enumerate(views.values())}


def _install(owner, storages, skeleton):
    """A reader's install: every tensor is a view of the fetched copy."""
    fetched = {name: storage.clone() for name, storage in storages.items()}
    names = {s.untyped_storage().data_ptr(): name for name, s in storages.items()}

    def view(source):
        storage = fetched[names[source.untyped_storage().data_ptr()]]
        return torch.empty(0, dtype=source.dtype).set_(
            storage.untyped_storage(),
            source.storage_offset(),
            source.shape,
            source.stride(),
        )

    weight = nn.Parameter(view(owner.attn.weight))
    scale = nn.Parameter(view(owner.attn.scale))
    scale.format_ue8m0 = True
    skeleton.attn.weight = skeleton.head.weight = weight
    skeleton.attn.scale = scale
    skeleton.attn.table = view(owner.attn.table)
    skeleton.attn.w_kc = view(owner.attn.w_kc)
    skeleton.attn.fused = True
    return skeleton.eval(), fetched


def _side(load=None, construct=nn.Module):
    """One process: construct, record, then ``load`` (which may return storages)."""
    model = construct()
    constructed = validator.record_construction(model)
    storages = (load(model) if load else None) or {}
    return dict(model=model, storages=storages, constructed=constructed)


def _held(make):
    """A load that keeps what ``make()`` returns in a plain attribute."""

    def load(model):
        model.held = make()

    return load


class TestValidator(CustomTestCase):
    def setUp(self):
        patcher = mock.patch.object(validator, "_expert_map", return_value=None)
        self.expert_map = patcher.start()
        self.addCleanup(patcher.stop)
        self._snapshot()

    def _snapshot(self):
        self.owner = _Model("/typed/by/the/owner")
        constructed = validator.record_construction(self.owner)
        self.storages = _publish(_raw_load(self.owner))
        self.attestation = validator.attest(
            model=self.owner, storages=self.storages, constructed=constructed
        )
        with torch.device("meta"):  # as a reader builds its skeleton
            skeleton = _Model("hub/id")
        self.constructed = validator.record_construction(skeleton)
        self.model, self.fetched = _install(self.owner, self.storages, skeleton)

    def _validate(self):
        validator.validate(
            model=self.model,
            storages=self.fetched,
            attestation=self.attestation,
            constructed=self.constructed,
        )

    def _refused(self, *expected):
        with self.assertRaises(ValidationError) as caught:
            self._validate()
        for text in expected:
            self.assertIn(text, str(caught.exception))

    def _same(self, owner, reader):
        validator.validate(**reader, attestation=validator.attest(**owner))

    def _differs(self, owner, reader, *expected):
        attestation = validator.attest(**owner)
        with self.assertRaises(ValidationError) as caught:
            validator.validate(**reader, attestation=attestation)
        for text in expected:
            self.assertIn(text, str(caught.exception))

    def _undescribed(self, side, *expected):
        with self.assertRaises(ValidationError) as caught:
            validator.attest(**side)
        self.assertIn("This model cannot be described", str(caught.exception))
        for text in expected:
            self.assertIn(text, str(caught.exception))

    def test_an_installed_copy_of_the_raw_load_passes(self):
        self.assertNotEqual(self.owner.checkpoint, self.model.checkpoint)
        self._validate()

    def test_another_expert_map_is_refused(self):
        self.expert_map.return_value = {"rows": ["value", 1]}
        self._snapshot()
        self._validate()
        self.expert_map.return_value = {"rows": ["value", 2]}
        self._refused("expert map rows: ['value', 1] on the owner, ['value', 2] here")
        self.expert_map.return_value = None
        self._refused("only one of the owner and this rank has an expert map")
        self._snapshot()
        self.expert_map.return_value = {"rows": ["value", 1]}
        self._refused("only one of the owner and this rank has an expert map")

    def test_a_missing_fp8_scale_format_flag_is_refused(self):
        del self.model.attn.scale.format_ue8m0
        self._refused(
            "attn.scale::format_ue8m0: the owner's load left True, here it is absent"
        )

    def test_mla_weights_not_carried_or_derived_again_are_refused(self):
        self.model.attn.w_kc = None
        self._refused(
            "attn::w_kc is only on the owner",
            "attn::w_kc: the owner's load left absent, here it is None",
        )
        self.model.attn.w_kc = torch.ones(2, 4)
        self._refused(
            "attn::w_kc: stride is [1, 3] on the owner, [4, 1] here, "
            "allocation is ['published', '0'] on the owner, ['own', 0] here",
            "own allocation 0, first used by attn::w_kc is only here",
        )

    def test_changed_bytes_in_a_storage_are_refused(self):
        self.fetched["2"][0] += 1
        self._refused("storage 2: its bytes differ from the owner's")
        self.fetched["2"][0] -= 1
        self.fetched["0"][-1] += 1
        self._refused("1 difference(s): storage 0: its bytes differ from the owner's")

    def test_a_tensor_at_another_place_in_its_storage_is_refused(self):
        self.model.attn.w_kc = self.model.attn.weight.detach()[:2]
        self._refused(
            "attn::w_kc: shape is [2, 4] on the owner, [2, 3] here, "
            "stride is [1, 3] on the owner, [3, 1] here"
        )

        def at(first, published_from=0):
            def load(model):
                base = torch.zeros(4)
                model.held = [base[first : first + 2]]
                return {"s": _raw(base)[published_from:]}

            return _side(load)

        self._differs(at(0), at(2), "::held[0]: offset is 0 on the owner, 8 here")
        self._differs(at(1, 4), at(2, 4), "offset is 0 on the owner, 4 here")
        self._same(at(1, 4), at(1, 4))

    def test_a_lost_tie_is_refused(self):
        weight = self.model.attn.weight
        self.model.head.weight = nn.Parameter(weight.detach().clone())
        self._refused(
            "head.weight: ['same as', 'attn.weight'] on the owner, ['tensor', ",
            "own allocation 0, first used by head.weight is only here",
        )
        # Two parameters on the same bytes are not one parameter either.
        self.model.head.weight = nn.Parameter(weight.detach())
        self._refused("1 difference(s): head.weight: ['same as', 'attn.weight']")

    def test_a_non_persistent_buffer_is_compared(self):
        self.assertNotIn("attn.table", self.model.state_dict())
        self.model.attn.table = torch.zeros(6)
        self._refused(
            "attn.table: allocation is ['published', '2'] on the owner, ['own', 0] here"
        )

    def test_plain_values_follow_the_owners_load(self):
        def side(at_construction, loaded=None):
            def construct():
                model = nn.Module()
                model.x = at_construction
                return model

            def load(model):
                if loaded == "deleted":
                    del model.x
                elif loaded is not None:
                    model.x = loaded

            return _side(load, construct)

        self._same(side(1, 2), side(1, 2))
        self._differs(side(1, 2), side(1), "::x: the owner's load left 2, here it is 1")
        # Construction on the reader may already give what the owner's load set,
        self._same(side(1, 2), side(2))
        # and an install may repair what construction on the reader got wrong.
        self._same(side(2), side(1, 2))
        self._differs(side(2), side(1, 3), "::x changed only here, to 3")
        self._differs(
            side(1, "deleted"), side(1), "the owner's load left absent, here it is 1"
        )
        self._differs(side(1), side(1, "deleted"), "::x changed only here, to absent")
        # What no load changes is not compared: it may be a path as typed.
        self._same(side("/a"), side("/b"))

        def config(flag, path, repaired=None):
            def construct():
                model = nn.Module()
                model.config = {"flag": flag, "path": path}
                return model

            def load(model):
                if repaired is not None:
                    model.config["flag"] = repaired

            return _side(load, construct)

        # Member by member: an install repairs one entry next to a path as typed.
        self._same(config(3, "/a"), config(4, "/b", repaired=3))
        self._differs(
            config(3, "/a"),
            config(4, "/b", repaired=5),
            "::config['flag'] changed only here, to 5",
        )

    def test_the_mode_of_every_module_is_compared(self):
        self.model.attn.train()
        self._refused("attn: ['module', ", "False] on the owner, ['module', ")

        def side(training):
            def load(model):
                model.parts = [nn.Identity().train(training)]  # not registered

            return _side(load)

        self._same(side(False), side(False))
        self._differs(side(False), side(True), "::parts[0]: ['module', ")

    def test_a_tensor_without_data_is_refused(self):
        for held in (
            torch.empty(2, 4, device="meta"),
            torch.empty(0, device="meta"),
        ):
            self.model.attn.w_kc = held
            self._refused("attn::w_kc has no data: it is on the meta device")
        self.model.attn.w_kc = None
        self.model.attn.built_on = torch.device("meta")
        self._refused("attn::built_on holds the meta device")

        def returns_it():
            pass

        code = returns_it.__code__
        returns_it.__code__ = code.replace(co_consts=(torch.device("meta"),))
        weight = self.model.attn.weight
        for held in (
            [torch.device("meta")],
            slice(torch.device("meta")),
            {"w": weight, torch.device("meta"): 1},
            slice(weight, torch.device("meta")),
            returns_it,
        ):
            self.model.attn.built_on = held
            self._refused("attn::built_on", " holds the meta device")

    def test_storage_names_and_sizes_must_match(self):
        more = dict(self.fetched, more=torch.zeros(4, dtype=torch.uint8))
        with self.assertRaisesRegex(ValidationError, "storage more is only here"):
            validator.validate(
                model=self.model,
                storages=more,
                attestation=self.attestation,
                constructed=self.constructed,
            )
        del self.fetched["1"]
        self._refused(
            "attn.scale: allocation is ['published', '1'] on the owner, ['own', 0]",
            "storage 1 is only on the owner",
        )

        def unused(nbytes):
            return _side(lambda model: {"u": torch.zeros(nbytes, dtype=torch.uint8)})

        self._differs(unused(4), unused(8), "storage u: 4 bytes on the owner, 8 here")
        self._differs(unused(0), _side(), "storage u is only on the owner")
        self._same(unused(0), unused(0))

    def test_views_swapped_between_equal_storages_are_refused(self):
        def side(first, second):
            def load(model):
                a, b = torch.ones(2), torch.ones(2)
                model.held = [{"a": a, "b": b}[first], {"a": a, "b": b}[second]]
                return {"a": _raw(a), "b": _raw(b)}

            return _side(load)

        self._same(side("a", "b"), side("a", "b"))
        self._differs(
            side("a", "b"),
            side("b", "a"),
            "allocation is ['published', 'a'] on the owner, ['published', 'b'] here",
        )

    def test_an_attestation_it_cannot_read_is_refused(self):
        good = self.attestation
        packed = zlib.decompress(good[1:])
        fields = msgspec.msgpack.decode(packed)
        unreadable = {
            "empty": b"",
            "version": bytes([validator._VERSION + 1]) + good[1:],
            "not compressed": good[:1] + b"not zlib",
            "cut short": good[:-4],
            "followed by more": good + b"x",
            "not msgpack": good[:1] + zlib.compress(b"\xc1"),
            "unknown field": dict(fields, extra=1),
            "missing field": {k: v for k, v in fields.items() if k != "leaves"},
            "wrong type": dict(fields, facts=[1, 2]),
            "wrong inner type": dict(fields, allocations={"published:0": "x"}),
            "not bytes": None,
        }
        for name, attestation in unreadable.items():
            if isinstance(attestation, dict):
                packed = msgspec.msgpack.encode(attestation)
                attestation = good[:1] + zlib.compress(packed)
            self.attestation = attestation
            with self.subTest(name):
                self._refused("Cannot read the owner's attestation")

    def test_module_class_added_module_and_child_order_are_compared(self):
        self.model.head.__class__ = _OtherLinear
        self.model.attn.added = nn.Identity()
        self._refused(
            "head: ['module', 'torch.nn.modules.linear.Linear', False] on the owner",
            "attn.added is only here",
            "attn<modules> is only here",
        )

        def side(*names):
            def load(model):
                layers = {"r": nn.ReLU(), "s": nn.Sigmoid()}
                model.net = nn.Sequential(OrderedDict((n, layers[n]) for n in names))

            return _side(load)

        self._same(side("r", "s"), side("r", "s"))
        self._differs(
            side("r", "s"),
            side("s", "r"),
            "net<modules>: [['r', True], ['s', True]] on the owner, "
            "[['s', True], ['r', True]] here",
        )

    def test_which_object_a_name_refers_to_is_compared(self):
        def side(load):
            def with_two(model):
                model.a, model.b = _Scale(2), _Scale(3)
                load(model)

            return _side(with_two)

        def active(name):
            return side(lambda m: setattr(m, "active", getattr(m, name)))

        self._differs(
            active("a"),
            active("b"),
            "active: ['same as', 'a'] on the owner, ['same as', 'b'] here",
        )
        shadow = side(lambda m: m.__dict__.__setitem__("a", m.b))
        self._undescribed(shadow, "::a shadows a registered parameter, buffer or")

        value = torch.ones(2)
        as_parameter = _side(
            lambda m: m.register_parameter("w", nn.Parameter(value, False))
        )
        as_buffer = _side(lambda m: m.register_buffer("w", nn.Parameter(value, False)))
        self._differs(as_parameter, as_buffer, "<parameters> is only on the owner")
        with_none = _side(lambda m: m.register_buffer("zero", None))
        self._differs(
            with_none, _side(), "<buffers> is only on the owner: [['zero', False]]"
        )

        def grad(requires_grad):
            return _side(
                lambda m: m.register_parameter("w", nn.Parameter(value, requires_grad))
            )

        self._differs(
            grad(True), grad(False), "w: requires_grad is True on the owner, False here"
        )

    def test_each_field_of_a_tensor_is_compared(self):
        def held(make):
            return _side(_held(lambda: [make(torch.zeros(4))]))

        cases = {
            "class is 'torch.nn.parameter.Parameter' on the owner, 'torch.Tensor'": (
                lambda base: nn.Parameter(base, requires_grad=False),
                lambda base: base,
            ),
            "dtype is 'torch.int32' on the owner, 'torch.float32' here": (
                lambda base: base.view(torch.int32),
                lambda base: base,
            ),
            "shape is [2] on the owner, [3] here": (
                lambda base: base.as_strided((2,), (1,)),
                lambda base: base.as_strided((3,), (1,)),
            ),
            "stride is [1] on the owner, [2] here": (
                lambda base: base.as_strided((2,), (1,)),
                lambda base: base.as_strided((2,), (2,)),
            ),
        }
        for expected, (owner, reader) in cases.items():
            with self.subTest(expected):
                self._same(held(owner), held(owner))
                self._differs(
                    held(owner), held(reader), f"1 difference(s): ::held[0]: {expected}"
                )

    def test_views_that_read_other_values_cannot_be_described(self):
        cases = {
            "is a lazy conjugate or negative view": [
                lambda: torch._neg_view(torch.tensor([5.0])),
                lambda: torch.tensor([2 + 3j]).conj(),
            ],
            "is not a plain dense tensor": [
                lambda: torch.quantize_per_tensor(
                    torch.tensor([2.0, 4.0]), 1.0, 0, torch.qint8
                ),
                lambda: torch.eye(2).to_sparse(),
            ],
        }
        for expected, makers in cases.items():
            for make in makers:
                self._undescribed(_side(_held(make)), f"::held {expected}")
        # A class of its own could answer for the tensor.
        other = _side(_held(lambda: _OtherParameter(torch.ones(1))))
        self._undescribed(
            other, "held is a ", "_OtherParameter, a tensor class that is not described"
        )

    def test_a_published_storage_must_be_a_run_of_bytes(self):
        def side(publish, view=lambda base: base):
            def load(model):
                base = torch.arange(4.0)
                model.held = view(base)
                return publish(_raw(base), base)

            return _side(load)

        bad = {
            "a float tensor": lambda raw, base: {"s": base},
            "two dimensions": lambda raw, base: {"s": raw.reshape(4, 4)},
            "every other byte": lambda raw, base: {"s": raw[::2]},
            "no data": lambda raw, base: {
                "s": torch.empty(4, dtype=torch.uint8, device="meta")
            },
            "a lazy negative view": lambda raw, base: {"s": torch._neg_view(raw)},
        }
        for name, publish in bad.items():
            with self.subTest(name):
                self._undescribed(
                    side(publish), "storage 's' is not a 1-D uint8 tensor of bytes"
                )
        unsendable = side(lambda raw, base: {"s\udc80": raw})
        self._undescribed(unsendable, "has a name that cannot be sent")
        self._undescribed(
            side(lambda raw, base: {1: raw}), "a storage has a name that is not a str"
        )
        overlap = side(lambda raw, base: {"a": raw[:9], "b": raw[8:]})
        self._undescribed(overlap, "storage 'a' and storage 'b' overlap in memory")

        # An empty storage has a name and no place: a packed reader may cut it
        # from beyond the end of its allocation.
        self._same(
            side(lambda raw, base: {"s": raw, "e": raw[:0]}),
            side(lambda raw, base: {"s": raw, "e": raw[1::16][1:]}),
        )

        halves = lambda raw, base: {"a": raw[:8], "b": raw[8:]}  # noqa: E731
        self._same(side(halves, lambda b: b[2:]), side(halves, lambda b: b[2:]))
        outside = "::held is not inside a published storage of its allocation"
        self._undescribed(side(halves), outside)
        self._undescribed(side(lambda raw, base: {"s": raw[4:]}), outside)
        gapped = lambda base: base.as_strided((2,), (2,))  # noqa: E731
        self._undescribed(side(lambda raw, base: {"s": raw[:8]}, gapped), outside)

    def test_reading_a_model_leaves_its_storages_resizable(self):
        self._validate()
        own = _side(_held(lambda: torch.ones(3)))
        validator.validate(**own, attestation=validator.attest(**own))
        read = [own["model"].held, self.owner.attn.weight, self.model.attn.weight]
        for tensor in (*read, *self.storages.values(), *self.fetched.values()):
            self.assertTrue(tensor.untyped_storage().resizable())

    def test_an_empty_tensor_has_no_place(self):
        def side(view):
            def load(model):
                base = torch.tensor([3.0, 7.0])
                model.held = view(base)
                return {"s": _raw(base)}

            return _side(load)

        # It reads no byte: a packed install may put it anywhere.
        start = side(lambda b: b[0:0])
        self._same(start, side(lambda b: b[1:1]))
        self._same(start, side(lambda b: torch.empty(0)))
        self._differs(
            start,
            side(lambda b: b[0:0].reshape(0, 3)),
            "::held: shape is [0] on the owner, [0, 3] here",
        )
        self._differs(start, side(lambda b: None), "::held is only on the owner")

    def test_an_allocation_of_the_models_own_is_digested_in_full(self):
        def first_of(*values):
            return _side(_held(lambda: [torch.tensor(values)[:1]]))

        self._differs(
            first_of(1.0, 2.0),
            first_of(1.0, 3.0),
            "own allocation 0, first used by ::held[0]: its bytes differ",
        )
        stepped = _held(lambda: torch.arange(8.0)[::2][:1])
        self._same(_side(stepped), _side(stepped))

        def shrunk():
            held = torch.zeros(2)
            held.untyped_storage().resize_(4)
            return held

        self._undescribed(_side(_held(shrunk)), "::held reaches beyond its allocation")

        def tied():
            base = torch.ones(2)
            return [base, base[:]]

        self._differs(
            _side(_held(tied)),
            _side(_held(lambda: [torch.ones(2), torch.ones(2)])),
            "::held[1]: allocation is ['own', 0] on the owner, ['own', 1] here",
        )
        self._differs(
            _side(_held(lambda: torch.zeros(4)[0:2])),
            _side(_held(lambda: torch.zeros(4)[2:4])),
            "::held: offset is 0 on the owner, 8 here",
        )

    def test_what_objects_and_containers_hold_is_walked(self):
        def kernel(scale, factor):
            return _side(
                _held(lambda: SimpleNamespace(scale=torch.tensor(scale), factor=factor))
            )

        self._same(kernel(1.0, 2), kernel(1.0, 2))
        self._differs(
            kernel(1.0, 2),
            kernel(2.0, 3),
            "::held::factor: the owner's load left 2, here it is 3",
            "first used by ::held::scale: its bytes differ",
        )

        def on_a_tensor(scale):
            def make():
                weight = torch.ones(1)
                weight.scale = torch.tensor(scale)
                return weight

            return _side(_held(make))

        self._differs(
            on_a_tensor(2.0), on_a_tensor(3.0), "first used by ::held::scale: its bytes"
        )

        def nested(value):
            def make():
                held = torch.tensor(value)
                for _ in range(12):
                    held = [held]
                return held

            return _side(_held(make))

        self._same(nested(2.0), nested(2.0))
        self._differs(nested(2.0), nested(3.0), "its bytes differ from the owner's")

        def flags(member):
            return _side(_held(lambda: _Flags([member])))

        self._same(flags(True), flags(True))
        self._differs(flags(True), flags(1), "::held: ['", "[True]] on the owner")

        def build(value):
            def load(model):
                model.parts = [nn.Linear(1, 1, bias=False)]  # not registered
                model.cache = {"k": torch.full((2,), value)}
                with torch.no_grad():
                    model.parts[0].weight.fill_(value)

            return _side(load)

        self._same(build(1.0), build(1.0))
        self._differs(
            build(1.0),
            build(2.0),
            "first used by ::cache['k']: its bytes differ",
            "first used by ::parts[0].weight: its bytes differ",
        )

    def test_a_dict_that_two_objects_share_is_one_dict(self):
        def buffers(shared):
            def load(model):
                model.a, model.b = nn.Module(), nn.Module()
                model.a.register_buffer("w", torch.ones(1))
                if shared:
                    model.b._buffers = model.a._buffers
                else:
                    model.b.register_buffer("w", model.a.w)

            return _side(load)

        self._same(buffers(True), buffers(True))
        self._differs(
            buffers(True),
            buffers(False),
            "b<buffers>: ['same table as', 'a<buffers>'] on the owner, [['w', True]]",
        )

        class Box:
            pass

        def fields(shared):
            def load(model):
                model.a, model.b = Box(), Box()
                model.a.x = 1
                if shared:
                    model.b.__dict__ = model.a.__dict__
                else:
                    model.b.x = 1

            return _side(load)

        self._same(fields(True), fields(True))
        self._differs(
            fields(True),
            fields(False),
            "::b::<fields> is only on the owner: ['same fields as', '::a::<fields>']",
        )

        class Registry(dict):
            pass

        def of_another_class(model):
            model.child = nn.Linear(1, 1)
            model.child._parameters = Registry(model.child._parameters)

        self._undescribed(
            _side(of_another_class), "child has no plain dict at 'child<parameters>'"
        )

    def test_an_object_python_cannot_see_into_must_be_what_construction_left(self):
        def construct():
            model = nn.Module()
            model.lock = threading.Lock()
            model.log = logging.getLogger("test_weight_cache_validator")
            return model

        self._same(_side(None, construct), _side(None, construct))
        replaced = _side(lambda m: setattr(m, "lock", threading.Lock()), construct)
        self._undescribed(
            replaced,
            "::lock holds a _thread.lock that is not described and is not the object "
            "that construction left there",
        )
        self._differs(
            _side(None, construct),
            _side(lambda m: delattr(m, "lock"), construct),
            "::lock is only on the owner: ['opaque', '_thread.lock']",
        )
        # A compiled pattern shows fields, and keeps more than them. Naming the
        # fields of a structured array, or printing a path, would run code.
        made_by_the_load = {
            "Pattern": lambda: re.compile("a"),
            "Generator": lambda: torch.Generator(),
            "Logger": lambda: logging.getLogger("test_weight_cache_validator"),
            "PurePosixPath": lambda: pathlib.PurePosixPath("cfg"),
            "ndarray": lambda: np.array([object()], dtype=object),
        }
        for name, make in made_by_the_load.items():
            with self.subTest(name):
                self._undescribed(
                    _side(_held(make)),
                    "::held holds a ",
                    f"{name} that is not described",
                )
        structured = _side(_held(lambda: np.zeros(2, dtype=[("a", "f4")])))
        self._undescribed(structured, "::held holds a numpy.ndarray that is not")

    def test_what_the_walk_cannot_describe_stops_the_attestation(self):
        class Key:
            pass

        def deep():
            held = torch.ones(1)
            for _ in range(80):
                held = [held]
            return held

        def deep_plain():
            held = 1
            for _ in range(100):
                held = (held,)
            return held

        def deep_in_objects():  # the depth is one count through both
            held = 1
            for _ in range(40):
                held = (held,)
            for _ in range(40):
                held = SimpleNamespace(x=held)
            return held

        undescribed = [
            ("a set with a member that is not a plain value", lambda: {torch.ones(1)}),
            ("a dict with a key that is not a plain value", lambda: {Key(): 1}),
            ("is nested deeper than 64", deep),
            ("is nested deeper than 64", deep_plain),
            ("is nested deeper than 64", deep_in_objects),
        ]
        for expected, make in undescribed:
            with self.subTest(expected):
                self._undescribed(_side(_held(make)), "::held", expected)

        def construct():
            model = nn.Module()
            model.held = {torch.ones(1)}
            return model

        with self.assertRaisesRegex(
            ValidationError, "not recorded: ::held holds a set with a member"
        ):
            validator.attest(**_side(None, construct))

    def test_two_allocations_do_not_share_memory(self):
        def held(*windows):  # (count, first byte) of float32 windows over one buffer
            def load(model):
                buffer = bytearray(struct.pack("4f", 1.0, 2.0, 3.0, 4.0))
                model.held = [
                    torch.frombuffer(
                        buffer, dtype=torch.float32, count=count, offset=offset
                    )
                    for count, offset in windows
                ]

            return _side(load)

        self._same(held((1, 0), (2, 8)), held((1, 0), (2, 8)))
        # One address, two sizes: the second tensor reads past what was digested.
        self._undescribed(
            held((1, 0), (2, 0)),
            "::held[1] and ::held[0] start at one address in allocations of "
            "different sizes",
        )
        self._undescribed(
            held((2, 0), (2, 4)),
            "the allocation of ::held[0] and the allocation of ::held[1] overlap",
        )

    def test_fields_in_slots_are_compared(self):
        def side(derived=None, base=None, cls=_SlottedTwice):
            def load(model):
                model.child = cls()
                if derived is not None:
                    model.child.gain = derived
                if base is not None:
                    _Slotted.gain.__set__(model.child, base)

            return _side(load)

        self._same(side(2, 3), side(2, 3))
        self._differs(
            side(2, cls=_Slotted),
            side(4, cls=_Slotted),
            "child::gain: the owner's load left 2, here it is 4",
        )
        # One name in two classes is two slots.
        self._differs(
            side(base=7),
            side(derived=7),
            "child::<'1:gain'>: the owner's load left 7, here it is absent",
            "child::<'0:gain'> changed only here, to 7",
        )

        # A msgspec.Struct keeps its fields in slots too.
        class Plan(msgspec.Struct):
            gain: int
            table: torch.Tensor

        def plan(gain, fill):
            return _side(_held(lambda: Plan(gain, torch.full((2,), fill))))

        self._same(plan(2, 1.0), plan(2, 1.0))
        self._differs(
            plan(2, 1.0),
            plan(4, 1.0),
            "::held::gain: the owner's load left 2, here it is 4",
        )
        self._differs(
            plan(2, 1.0),
            plan(2, 3.0),
            "first used by ::held::table: its bytes differ from the owner's",
        )

    def test_fields_are_found_wherever_a_value_keeps_them(self):
        class Fields(dict):
            pass

        def in_a_dict_of_its_own_class(model, value):
            model.child = nn.Module()
            model.child.__dict__ = Fields(model.child.__dict__)
            model.child.scale = torch.tensor(value)

        def on_a_list(model, value):
            numbers = _Numbers([1, 2])
            numbers.scale = torch.tensor(value)
            model.held = (numbers,)

        def on_a_function(model, value):
            def function(x):
                return x

            function.scale = torch.tensor(value)
            model.held = function

        def on_an_enum_member(model, value):
            class Kind(enum.Enum):
                A = 1

            Kind.A.scale = torch.tensor(value)
            model.held = Kind.A

        def in_an_ordered_dict_behind_its_back(model, value):
            model.held = collections.OrderedDict(a=1)
            dict.__setitem__(model.held, "w", torch.tensor(value))

        cases = [
            ("child::scale", in_a_dict_of_its_own_class),
            ("::held[0]::scale", on_a_list),
            ("::held::scale", on_a_function),
            ("::held::scale", on_an_enum_member),
            ("::held['w']", in_an_ordered_dict_behind_its_back),
        ]
        for path, load in cases:
            with self.subTest(load.__name__):
                owner = _side(functools.partial(load, value=2.0))
                self._same(owner, _side(functools.partial(load, value=2.0)))
                self._differs(
                    owner,
                    _side(functools.partial(load, value=3.0)),
                    f"first used by {path}: its bytes differ",
                )

        def gain(value):
            return _side(_held(lambda: _Gain(value)))

        self._same(gain(2.0), gain(2.0))
        self._differs(
            gain(2.0),
            gain(3.0),
            "::held::<value>: the owner's load left 2.0, here it is 3.0",
        )

    def test_a_name_is_not_mistaken_for_a_nesting(self):
        def nested(model):
            model.obj = SimpleNamespace(w=torch.tensor(7.0))

        def named(model):
            model.obj = SimpleNamespace()
            setattr(model, "obj::w", torch.tensor(7.0))

        self._differs(
            _side(nested),
            _side(named),
            "::obj::w is only on the owner",
            "::<'obj::w'> is only here",
        )

    def test_the_walk_runs_no_code_of_what_it_reads(self):
        asked = []

        class Watched:
            def __init__(self):
                self.scale = 2

            def __getattribute__(self, name):
                asked.append(name)
                return object.__getattribute__(self, name)

        side = _side(_held(Watched))
        asked.clear()
        validator.record_construction(side["model"])
        validator.attest(**side)
        self.assertEqual(asked, [])

        class Hidden:
            @property
            def __dict__(self):
                asked.append("__dict__")
                return {}

        class Key:
            def __hash__(self):
                asked.append("__hash__")
                return 1

        class Slot:
            __slots__ = ("x",)

        class Shadowed(Slot):
            __dict__ = Slot.__dict__["x"]  # another slot under that name

        def shadowed():
            held = Shadowed()
            held.x, held.scale = {}, torch.ones(1)
            return held

        unreadable = [
            ("cannot be read without running its code", Hidden),
            ("cannot be read without running its code", shadowed),
            (
                "holds a dict with a key that is not a plain value",
                lambda: collections.OrderedDict({Key(): torch.ones(1)}),
            ),
        ]
        for expected, make in unreadable:
            side = _side(_held(make))
            asked.clear()
            self._undescribed(side, f"::held {expected}")
            self.assertEqual(asked, [])

        class Meta(type):
            def __getattribute__(cls, name):
                asked.append(name)
                return type.__getattribute__(cls, name)

        class Numbers(list, metaclass=Meta):
            pass

        class Fake(np.float64):
            __module__ = "numpy"

            def tobytes(self, *args):
                asked.append("tobytes")
                return b""

        # A built-in method bound to an object, a number of a class that says
        # it is NumPy's.
        for make in (lambda: Numbers([1]).append, lambda: Fake(2.0)):
            side = _side(_held(make))
            asked.clear()
            validator.attest(**side)
            self.assertEqual(asked, [])

    def test_the_walk_has_a_budget(self):
        with mock.patch.object(validator, "_MAX_OBJECTS", 1000):
            many = _side(_held(lambda: list(range(5000))))
            with self.assertRaisesRegex(ValidationError, "more than 1000 values"):
                validator.attest(**many)

            def shared():  # 2**20 values through twenty lists
                held = [0]
                for _ in range(20):
                    held = [held, held]
                return held

            with self.assertRaisesRegex(ValidationError, "more than 1000 values"):
                validator.attest(**_side(_held(shared)))

            def sliced():  # 3**7 values through seven slices
                held = 0
                for _ in range(7):
                    held = slice(held, held, held)
                return held

            constant = f"lambda: {tuple(range(5000))!r}"
            names = f"lambda: ({', '.join(f'n{i}' for i in range(5000))})"
            makes = (
                sliced,
                lambda: eval(constant),
                lambda: eval(names),
                lambda: torch.Size(range(5000)),
            )
            for make in makes:
                with self.assertRaisesRegex(ValidationError, "more than 1000 values"):
                    validator.attest(**_side(_held(make)))

            def construct():
                model = nn.Module()
                model.held = list(range(5000))
                return model

            with self.assertRaisesRegex(
                ValidationError, "not recorded: ValidationError: The model reaches"
            ):
                validator.attest(**_side(None, construct))

    def test_functions_are_compared_with_what_they_close_over(self):
        def adder(increment):
            return _side(_held(lambda: lambda x: x + increment))

        self._same(adder(1), adder(1))
        self._differs(
            adder(1),
            adder(2),
            "::held::<closure 0>: the owner's load left 1, here it is 2",
        )

        def partial(by):
            return _side(_held(lambda: functools.partial(operator.mul, by)))

        self._same(partial(2), partial(2))
        self._differs(partial(2), partial(3), "::held::<args>: the owner's load left")

        def table(**ops):
            return _side(_held(lambda: dict(ops)))

        self._same(table(inc=operator.add), table(inc=operator.add))
        self._differs(
            table(inc=operator.add, dec=operator.sub),
            table(dec=operator.add, inc=operator.sub),
            "::held['inc']: ['builtin', '_operator', None, 'add'] on the owner, "
            "['builtin', '_operator', None, 'sub'] here",
            "::held: ['dict', ['inc', 'dec']] on the owner, ['dict', ['dec', 'inc']]",
        )

        def method():
            def load(model):
                model.child = nn.Identity()
                model.held = model.child.forward

            return _side(load)

        self._same(method(), method())

        def coded(kind):  # two functions of one name
            def make():
                if kind == "add":
                    return lambda x: x + 1
                return lambda x: x - 1

            return _side(_held(make))

        self._same(coded("add"), coded("add"))
        self._differs(coded("add"), coded("sub"), "::held: ['function', ")

        def counters(shared):
            def make():
                def pair():
                    count = 0

                    def up():
                        nonlocal count
                        count += 1

                    def down():
                        nonlocal count
                        count -= 1

                    return up, down

                up, down = pair()
                return [up, down if shared else pair()[1]]

            return _side(_held(make))

        self._same(counters(True), counters(True))
        self._differs(
            counters(True),
            counters(False),
            "::held[1]::<closure 0> is only on the owner: "
            "['same cell as', '::held[0]::<closure 0>']",
        )

        def taking(many):  # the same instructions, another way to take arguments
            def make():
                if many:
                    return lambda *x: len(x)
                return lambda x: len(x)

            return _side(_held(make))

        self._differs(taking(False), taking(True), "::held: ['function', ")

        def default(value):
            return _side(_held(lambda: collections.defaultdict(lambda: value)))

        self._same(default(2), default(2))
        self._differs(
            default(2),
            default(3),
            "::held::default_factory::<closure 0>: the owner's load left 2, "
            "here it is 3",
        )

    def test_hooks_compare_in_call_order(self):
        def hooked(increment, earlier=0, **options):
            def load(model):
                for _ in range(earlier):  # other handle ids in this process
                    model.register_forward_hook(lambda *a: None).remove()
                model.register_forward_hook(lambda *a: increment, **options)

            return _side(load)

        self._same(hooked(1), hooked(1, earlier=3))
        self._same(hooked(1, with_kwargs=True), hooked(1, earlier=2, with_kwargs=True))
        self._differs(
            hooked(1),
            hooked(2),
            "::_forward_hooks[0]::<closure 0>: the owner's load left 1, here it is 2",
        )
        self._differs(hooked(1), hooked(1, with_kwargs=True), "hook flags")
        self._differs(hooked(1), _side(), "::_forward_hooks is only on the owner")

    def test_plain_values_compare_by_value(self):
        class Kind(enum.Enum):
            A = 1
            B = 2

        def same(a, b):
            encoded = [validator._plain(a), validator._plain(b)]
            self.assertNotIn(validator._NO, encoded, (a, b))
            return msgspec.msgpack.encode(encoded[0]) == msgspec.msgpack.encode(
                encoded[1]
            )

        equal = [
            (float("nan"), float("nan")),
            ({3, 1, 2}, {1, 2, 3}),
            ({1, "x", (1,)}, {(1,), 1, "x"}),
            (2**70, 2**70),
            (Kind.A, Kind.A),
            (np.float32(1.5), np.float32(1.5)),
            (np.longdouble(1), np.longdouble(1)),
            (np.arange(4), np.arange(4)),
            (torch.device("cuda:0"), torch.device("cuda:1")),
            (torch.float, torch.float32),
            ({"a": [1, (2.0, None)]}, {"a": [1, (2.0, None)]}),
            (array.array("d", [1.0, 2.0]), array.array("d", [1.0, 2.0])),
        ]
        different = [
            (1, True),
            (1, 1.0),
            (0.0, -0.0),
            (float("nan"), -float("nan")),
            ([1, 2], (1, 2)),
            ((1, 2), torch.Size([1, 2])),
            ({"a": 1, "b": 2}, {"b": 2, "a": 1}),  # code may depend on the order
            ({1, 2}, frozenset({1, 2})),
            (Kind.A, Kind.B),
            (np.float32(1.5), np.float64(1.5)),
            (np.arange(4), np.arange(4)[::-1]),
            (torch.float16, torch.bfloat16),
            (torch.device("cpu"), torch.device("cuda:0")),
            (range(3), range(4)),
            (range(1, 4, 1), slice(1, 4, 1)),
            (array.array("d", [1.0]), array.array("f", [1.0])),
        ]
        for a, b in equal:
            self.assertTrue(same(a, b), (a, b))
        for a, b in different:
            self.assertFalse(same(a, b), (a, b))
        unplain_values = (
            torch.ones(1),
            nn.Identity(),
            object(),
            len,
            [torch.ones(1)],
            collections.defaultdict(int),
        )
        for unplain in unplain_values:
            self.assertIs(validator._plain(unplain), validator._NO)

    def test_the_report_is_bounded(self):
        def text(letter):
            return _side(lambda m: setattr(m, "text", letter * 200_000))

        with self.assertRaises(ValidationError) as caught:
            validator.validate(**text("y"), attestation=validator.attest(**text("x")))
        report = str(caught.exception)
        self.assertLess(len(report), 1000)
        self.assertRegex(
            report,
            "::text: the owner's load left a value of 200005 bytes with digest "
            "[0-9a-f]{16}, here it is a value of 200005 bytes with digest [0-9a-f]{16}",
        )
        self._same(text("x"), text("x"))

        def many(value):
            def load(model):
                for i in range(30):
                    setattr(model, f"v{i:02}", value)

            return _side(load)

        with self.assertRaises(ValidationError) as caught:
            validator.validate(**many(2), attestation=validator.attest(**many(1)))
        report = str(caught.exception)
        self.assertIn("30 difference(s): ::v00: the owner's load left 1", report)
        self.assertIn("; and 10 more", report)
        self.assertEqual(report.count("the owner's load left"), 20)

    def test_digests_are_blake3_of_all_the_bytes(self):
        data = bytes(i % 251 for i in range(4097))
        known = {
            b"": "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262",
            data: "9b4052b38f1c5fc8b1f9ff7ac7b27cd242487b3d890d15c96a1c25b8aa0fb995",
            data[:-1] + b"\x00": None,
        }
        got = validator._digests(
            {
                key: torch.frombuffer(bytearray(key), dtype=torch.uint8)
                if key
                else torch.empty(0, dtype=torch.uint8)
                for key in known
            }
        )
        for key, digest in known.items():
            if digest is not None:
                self.assertEqual(got[key].hex(), digest)
        self.assertEqual(len(set(got.values())), 3)

    def test_a_model_without_a_construction_record_cannot_be_described(self):
        model = nn.Module()
        with self.assertRaisesRegex(ValidationError, "no construction record"):
            validator.attest(model=model, storages={}, constructed=None)
        with mock.patch.object(
            validator._Walk, "visit", side_effect=RuntimeError("boom")
        ):
            constructed = validator.record_construction(model)
        with self.assertRaisesRegex(
            ValidationError, "not recorded: RuntimeError: boom"
        ):
            validator.attest(model=model, storages={}, constructed=constructed)

        class Unprintable(Exception):
            def __str__(self):
                raise ValueError("no")

        with mock.patch.object(validator._Walk, "visit", side_effect=Unprintable()):
            unprintable = validator.record_construction(model)
        with self.assertRaisesRegex(ValidationError, "not recorded: Unprintable"):
            validator.attest(model=model, storages={}, constructed=unprintable)
        with self.assertRaisesRegex(
            ValidationError, "not recorded: RuntimeError: boom"
        ):
            validator.validate(
                model=self.model,
                storages=self.fetched,
                attestation=self.attestation,
                constructed=constructed,
            )

    def test_the_expert_map_is_described_with_all_its_fields(self):
        def described(rows=((0, 1),), dtype=torch.int64, dispatch=((0, 1),), ep_size=1):
            metadata = SimpleNamespace(
                physical_to_logical_map_cpu=torch.tensor(rows, dtype=dtype),
                logical_to_all_physical_map=torch.tensor(dispatch),
                ep_size=ep_size,
            )
            with mock.patch(
                "sglang.srt.eplb.expert_location.get_global_expert_location_metadata",
                return_value=metadata,
            ):
                return _expert_map()

        with mock.patch(
            "sglang.srt.eplb.expert_location.get_global_expert_location_metadata",
            return_value=None,
        ):
            self.assertIsNone(_expert_map())
        base = described()
        self.assertEqual(base, described())
        self.assertEqual(base["::ep_size"], ["value", 1])
        for other in (
            described(rows=((1, 0),)),
            described(dtype=torch.int32),
            described(dispatch=((1, 0),)),  # the same CPU rows, another dispatch
            described(ep_size=2),
        ):
            self.assertNotEqual(base, other)
        # It travels in the attestation and is compared field by field.
        self.expert_map.return_value = base
        self._snapshot()
        self._validate()
        self.expert_map.return_value = described(ep_size=2)
        self._refused(
            "expert map ::ep_size: ['value', 1] on the owner, ['value', 2] here"
        )


if __name__ == "__main__":
    unittest.main()
