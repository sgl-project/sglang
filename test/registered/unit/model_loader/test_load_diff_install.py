"""load_diff: record_construction, export and install. An owner loads a model,
a reader builds the same model on the meta device and installs what the owner
exported, through a transport that copies the bytes."""

import types
import unittest
from collections import OrderedDict, defaultdict
from typing import NamedTuple

import torch
from torch import nn

from sglang.srt.weight_cache import load_diff
from sglang.srt.weight_cache.load_diff import check_install
from sglang.srt.weight_cache.load_diff_format import (
    Manifest,
    SnapshotError,
    decode_manifest,
    encode_manifest,
)
from sglang.srt.weight_cache.schema import SnapshotManifest, StorageMeta
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CPU = torch.device("cpu")
SETTINGS = types.SimpleNamespace(scale=1, options={"a": 1})
SIBLING = types.SimpleNamespace(x=1)


class Method:
    """Like a quant method: a plain object that loading fills."""

    def __init__(self):
        self._cache_permute_indices = {}
        self.flag = False
        self.runner = object()  # process-local, never changed by loading


class Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.w13 = nn.Parameter(torch.zeros(2, 4, 6), requires_grad=False)
        self.w13.weight_loader = "built"
        self.scale = nn.Parameter(torch.ones(4), requires_grad=False)
        self.quant_method = Method()
        self.slots = [object(), 1]


class Attention(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.kv_b = nn.Parameter(torch.zeros(4, 6), requires_grad=False)
        self.conv = nn.Parameter(torch.zeros(3, 1, 4), requires_grad=False)
        self.conv_weights = self.conv.view(3, 4)
        self.w_kc = None
        self.register_buffer("table", torch.arange(4.0), persistent=False)
        self.register_parameter("bias", None)
        self.settings = SETTINGS
        self.path = path


class Toy(nn.Module):
    def __init__(self, path="/owner/path"):
        super().__init__()
        self.embed = nn.Embedding(8, 6)
        self.attn = Attention(path)
        self.experts = Experts()
        self.head = nn.Linear(6, 8, bias=False)
        self.topk = types.SimpleNamespace(bias=self.experts.scale)

    def forward(self, ids):
        h = self.embed(ids)
        h = h + (h @ self.attn.w_kc) @ self.attn.kv_b
        h = (
            h
            + self.attn.conv_weights.sum()
            + self.experts.w13.sum() * self.topk.bias.sum()
        )
        return self.head(h)


def load(model):
    """Loading and post-processing, as a loader would do them."""
    torch.manual_seed(0)
    with torch.no_grad():
        model.embed.weight.copy_(torch.randn(8, 6))
        model.attn.kv_b.copy_(torch.randn(4, 6))
        model.attn.conv.copy_(torch.randn(3, 1, 4))
        model.experts.scale.copy_(torch.rand(4))
    attn, experts = model.attn, model.experts
    attn.w_kc = attn.kv_b.t().contiguous()  # MLA: a tensor kept as a plain attribute
    experts.w13 = nn.Parameter(
        torch.randn(2, 4, 6).view(2, 2, 2, 6), requires_grad=False
    )
    experts.scale.format_ue8m0 = True  # a flag on a tensor
    experts.quant_method._cache_permute_indices[("w2", torch.Size([4, 6]), 128)] = (
        torch.arange(6)
    )
    experts.quant_method.flag = True
    experts.use_padded_loading = False
    experts.slots.append(2)  # a list grows; its opaque item stays
    model.head.weight = model.embed.weight  # tied by loading
    attn._buffers.pop("table")
    attn.register_buffer("new_buffer", torch.ones(2))


class Exported(NamedTuple):
    manifest: Manifest  # as a reader decodes it; a test may change it
    storages: dict


def export_of(owner, constructed) -> Exported:
    storages, payload = load_diff.export(model=owner, constructed=constructed)
    return Exported(decode_manifest(payload), storages)


def export(owner_change=None):
    owner = Toy()
    constructed = load_diff.record_construction(owner)
    load(owner)
    if owner_change is not None:
        owner_change(owner)
    return owner, export_of(owner, constructed)


def reader(path="/reader/path"):
    with torch.device("meta"):
        return Toy(path)


def copies_of(storages):
    return {sid: tensor.clone() for sid, tensor in storages.items()}


def install(model, manifest, storages):
    """``manifest`` as the owner published it, installed through the
    interface on copies of ``storages``."""
    payload = encode_manifest(manifest)
    return load_diff.install(
        skeleton=model,
        manifest=snapshot(storages, payload),
        storages=copies_of(storages),
    )


def installed_tensors(model, out):
    """Install without the self-test, for a test that changes the model
    before running it: the tensors by tensor id."""
    tensors, _ = load_diff._install(model, out.manifest, copies_of(out.storages), CPU)
    return tensors


def snapshot(storages, payload):
    """What a publish makes of an export (schema only)."""
    return SnapshotManifest(
        version="v",
        storages={
            name: StorageMeta(nbytes=t.nbytes, device=t.device.type)
            for name, t in storages.items()
        },
        load_diff=payload,
        attestation=b"",
        source={},
    )


class Scratch(nn.Module):
    """A layer whose kernel writes scratch on every call, declared
    process-local, as CUTLASS MoE and Marlin do."""

    _process_local_fields = frozenset({"workspace", "offsets"})

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2), requires_grad=False)
        self.workspace = torch.empty(8)  # built with the model
        self.alias = [None]

    def forward(self, x):
        self.workspace.fill_(float(x.sum()))
        self.offsets.add_(1)
        return x @ self.weight


class TestProcessLocal(CustomTestCase):
    def export(self):
        owner = Scratch()
        constructed = load_diff.record_construction(owner)
        owner.offsets = torch.zeros(3, dtype=torch.int32)  # made by loading
        owner.alias[0] = owner.workspace  # another reference to it
        owner(torch.ones(1, 2))  # the owner's kernel wrote its scratch
        return owner, export_of(owner, constructed)

    def test_scratch_is_not_sent_and_the_reader_makes_its_own(self):
        owner, out = self.export()
        sent = {t.untyped_storage().data_ptr() for t in out.storages.values()}
        self.assertNotIn(owner.workspace.untyped_storage().data_ptr(), sent)
        self.assertNotIn(owner.offsets.untyped_storage().data_ptr(), sent)
        self.assertEqual(
            sorted((e.dtype, e.shape, len(e.places)) for e in out.manifest.local),
            [("float32", [8], 2), ("int32", [3], 1)],
        )
        with torch.device("meta"):
            model = Scratch()
        install(model, out.manifest, out.storages)
        self.assertEqual(model.workspace.device, CPU)
        self.assertIs(model.alias[0], model.workspace)
        self.assertTrue(bool((model.workspace == 0).all()))
        self.assertTrue(bool((model.offsets == 0).all()))
        model(torch.ones(1, 2))  # the reader's kernel writes its own scratch
        self.assertTrue(bool((owner.offsets == 1).all()))
        self.assertTrue(bool((model.offsets == 1).all()))

    def test_a_sent_tensor_where_scratch_belongs_is_refused(self):
        _, out = self.export()
        with torch.device("meta"):
            model = Scratch()
        tensors = installed_tensors(model, out)
        shared = model.weight.detach()  # a view of a sent storage
        vars(model)["workspace"] = shared
        model.alias[0] = shared
        with self.assertRaises(SnapshotError):
            check_install(model, out.manifest, tensors, CPU)

    def test_registered_scratch_is_refused(self):
        owner = Scratch()
        constructed = load_diff.record_construction(owner)
        owner.offsets = torch.zeros(3, dtype=torch.int32)
        owner.register_buffer("kept", owner.offsets)
        with self.assertRaisesRegex(SnapshotError, "registered here"):
            load_diff.export(model=owner, constructed=constructed)

    def refused(self, loading, why):
        owner = Scratch()
        constructed = load_diff.record_construction(owner)
        owner.offsets = torch.zeros(3, dtype=torch.int32)
        loading(owner)
        with self.assertRaisesRegex(SnapshotError, why):
            load_diff.export(model=owner, constructed=constructed)

    def test_a_view_of_scratch_elsewhere_is_refused(self):
        # Another tensor object on the scratch's storage would still be sent.
        def keep_a_view(owner):
            owner.tail = owner.workspace[2:]

        self.refused(keep_a_view, "shares its storage with the process-local")

    def test_scratch_that_is_part_of_a_storage_is_refused(self):
        # A reader makes it as a new contiguous tensor.
        def strided(owner):
            vars(owner)["workspace"] = torch.empty(16)[::2]

        self.refused(strided, "not the whole of its storage")

    def test_the_address_of_scratch_is_refused(self):
        def keep_its_address(owner):
            owner.where = owner.workspace.data_ptr()

        self.refused(keep_its_address, "address")


class TestRoundTrip(CustomTestCase):
    def test_reader_holds_what_the_owner_loaded(self):
        owner, out = export()
        model = reader()
        install(model, out.manifest, out.storages)

        ids = torch.tensor([1, 2, 3])
        torch.testing.assert_close(model(ids), owner(ids))
        attn, experts = model.attn, model.experts
        self.assertEqual(
            attn.conv_weights.untyped_storage().data_ptr(),
            attn.conv.untyped_storage().data_ptr(),
        )
        self.assertIs(model.head.weight, model.embed.weight)
        self.assertTrue(experts.scale.format_ue8m0)
        self.assertEqual(tuple(experts.w13.shape), (2, 2, 2, 6))
        self.assertFalse(hasattr(experts.w13, "weight_loader"))
        self.assertTrue(experts.quant_method.flag)
        cache = experts.quant_method._cache_permute_indices
        torch.testing.assert_close(
            cache[("w2", torch.Size([4, 6]), 128)], torch.arange(6)
        )
        self.assertIs(model.topk.bias, experts.scale)
        self.assertEqual(
            attn.path, "/reader/path"
        )  # a local value loading did not touch
        self.assertNotIn("table", attn._buffers)
        self.assertIn("new_buffer", attn._buffers)
        self.assertIsNone(attn._parameters["bias"])
        self.assertEqual(experts.slots[2], 2)
        self.assertIsNot(experts.quant_method.runner, owner.experts.quant_method.runner)
        self.assertEqual(SETTINGS.scale, 1)

    def test_identity_of_built_parameters_is_kept(self):
        _, out = export()
        model = reader()
        captured = model.attn.kv_b
        install(model, out.manifest, out.storages)
        self.assertIs(model.attn.kv_b, captured)
        self.assertFalse(captured.is_meta)

    def test_through_the_interface(self):
        owner = Toy()
        constructed = load_diff.record_construction(owner)
        load(owner)
        storages, payload = load_diff.export(model=owner, constructed=constructed)
        self.assertIs(type(payload), bytes)
        self.assertTrue(storages and all(type(name) is str for name in storages))
        for tensor in storages.values():
            self.assertEqual((tensor.dtype, tensor.dim()), (torch.uint8, 1))
            self.assertEqual(tensor.untyped_storage().nbytes(), tensor.nbytes)
        model = reader()
        installed = load_diff.install(
            skeleton=model,
            manifest=snapshot(storages, payload),
            storages=copies_of(storages),
        )
        self.assertIs(installed, model)
        ids = torch.tensor([1, 2, 3])
        torch.testing.assert_close(model(ids), owner(ids))

    def test_export_needs_the_record_of_that_model(self):
        owner = Toy()
        constructed = load_diff.record_construction(Toy())
        load(owner)
        with self.assertRaises(SnapshotError):
            load_diff.export(model=owner, constructed=constructed)
        with self.assertRaises(SnapshotError):
            load_diff.export(model=owner, constructed=None)

    def test_install_takes_exactly_the_manifests_storages(self):
        owner = Toy()
        constructed = load_diff.record_construction(owner)
        load(owner)
        storages, payload = load_diff.export(model=owner, constructed=constructed)
        name = next(iter(storages))
        fewer = {k: v for k, v in storages.items() if k != name}
        wrong_size = dict(snapshot(storages, payload).storages)
        wrong_size[name] = StorageMeta(nbytes=wrong_size[name].nbytes + 1, device="cpu")
        cases = {
            "a storage missing": (snapshot(fewer, payload), fewer),
            "a size that is not the export's": (
                SnapshotManifest(
                    version="v", storages=wrong_size, load_diff=payload,
                    attestation=b"", source={},
                ),
                storages,
            ),
            "fewer tensors than the manifest names": (
                snapshot(storages, payload), fewer
            ),
            "load_diff that is not a manifest": (
                snapshot(storages, b"not msgpack"), storages
            ),
        }  # fmt: skip
        for why, (manifest, given) in cases.items():
            with self.subTest(why=why):
                model = reader()
                with self.assertRaises(SnapshotError):
                    load_diff.install(
                        skeleton=model, manifest=manifest, storages=copies_of(given)
                    )

    def test_install_refuses_none_for_a_storage_or_its_meta(self):
        owner = Toy()
        constructed = load_diff.record_construction(owner)
        load(owner)
        storages, payload = load_diff.export(model=owner, constructed=constructed)
        name = next(iter(storages))
        good = snapshot(storages, payload)
        no_meta = SnapshotManifest(
            version="v",
            storages={**good.storages, name: None},
            load_diff=payload,
            attestation=b"",
            source={},
        )
        cases = {
            "a meta that is None": (no_meta, copies_of(storages)),
            "a storage that is None": (good, {**copies_of(storages), name: None}),
        }
        for why, (manifest, given) in cases.items():
            with self.subTest(why=why):
                with self.assertRaises(SnapshotError):
                    load_diff.install(
                        skeleton=reader(), manifest=manifest, storages=given
                    )


class TestOwnerRefuses(CustomTestCase):
    def refused(self, change):
        with self.assertRaises(SnapshotError) as caught:
            export(change)
        return "\n".join(caught.exception.problems)

    def test_a_changed_module_level_object_the_model_refers_to(self):
        def change(model):
            SETTINGS.options["a"] = 2

        try:
            why = self.refused(change)
        finally:
            SETTINGS.options["a"] = 1
        self.assertIn("SETTINGS", why)
        self.assertIn("options", why)

    def test_a_changed_module_level_object_the_model_does_not_refer_to_is_fine(self):
        def change(model):
            SIBLING.x = 2

        try:
            export(change)
        finally:
            SIBLING.x = 1

    def test_a_function_made_by_loading(self):
        why = self.refused(lambda model: setattr(model.experts, "fn", lambda: 1))
        self.assertIn("experts.fn", why)

    def test_a_module_added_by_loading(self):
        why = self.refused(
            lambda model: setattr(model.experts, "extra", nn.Linear(2, 2))
        )
        self.assertIn("experts.extra", why)

    def test_a_hook_added_by_loading(self):
        why = self.refused(
            lambda model: model.head.register_forward_hook(lambda *a: None)
        )
        self.assertIn("head", why)

    def test_an_address_in_a_plain_value(self):
        why = self.refused(
            lambda model: setattr(model.attn, "device_ptr", model.attn.kv_b.data_ptr())
        )
        self.assertIn("attn.device_ptr", why)


class TestReaderRefuses(CustomTestCase):
    def test_wrong_bytes(self):
        _, out = export()
        copies = copies_of(out.storages)
        first = next(iter(copies.values()))
        first.view(-1)[0] ^= 1
        model = reader()
        with self.assertRaisesRegex(SnapshotError, "not the one the owner exported"):
            install(model, out.manifest, copies)

    def test_storages_handed_back_in_the_wrong_order(self):
        _, out = export()
        copies = copies_of(out.storages)
        same_size = {}
        for sid, tensor in copies.items():
            same_size.setdefault(tensor.numel(), []).append(sid)
        a, b = next(sids for sids in same_size.values() if len(sids) >= 2)[:2]
        copies[a], copies[b] = copies[b], copies[a]
        model = reader()
        with self.assertRaises(SnapshotError):
            install(model, out.manifest, copies)

    def test_a_view_installed_at_another_offset(self):
        _, out = export()
        model = reader()
        tensors = installed_tensors(model, out)
        model.attn.conv_weights = (
            model.attn.conv.view(-1)[1:].view(1, 11)[:, :4].view(1, 4)
        )
        with self.assertRaises(SnapshotError):
            check_install(model, out.manifest, tensors, CPU)

    def test_another_model(self):
        _, out = export()
        model = reader()
        model.attn.register_buffer("extra", torch.empty(1, device="meta"))
        with self.assertRaisesRegex(SnapshotError, "another model"):
            install(model, out.manifest, out.storages)

    def test_an_operation_on_a_missing_path_fails(self):
        _, out = export()
        out.manifest.ops.append(["set", ["attn", [["a", "missing"], ["a", "x"]]], 1])
        model = reader()
        with self.assertRaisesRegex(SnapshotError, "attn.missing.x: KeyError"):
            install(model, out.manifest, out.storages)

    def test_an_operation_may_not_write_module_bookkeeping(self):
        _, out = export()
        out.manifest.ops.append(["set", ["attn", [["a", "_parameters"]]], 1])
        model = reader()
        with self.assertRaisesRegex(SnapshotError, "not a place to write"):
            install(model, out.manifest, out.storages)

    def test_an_operation_may_not_write_under_a_module_level_object(self):
        _, out = export()
        out.manifest.ops.append(
            ["set", ["attn", [["a", "settings"], ["a", "scale"]]], 9]
        )
        model = reader()
        with self.assertRaisesRegex(SnapshotError, "not a place to write"):
            install(model, out.manifest, out.storages)
        self.assertEqual(SETTINGS.scale, 1)


GLOBAL = {"opts": {"inner": {"scale": 1}}}


class Item:
    def __init__(self):
        self.scale = 0


class Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(4), requires_grad=False)
        self.lin = nn.Linear(2, 2, bias=False)
        self.items = [Item(), "local"]
        self.d = {"a": 1, "b": 2}
        self.od = OrderedDict(a=1, b=2)
        self.cfg = GLOBAL
        self.extra = None
        self.x = 0
        self.nested = [[1], "local"]
        self.dd = defaultdict(list)
        self.thing = Item()
        self.thing.flag = True
        self.alias = None
        self.w2 = nn.Parameter(torch.zeros(2), requires_grad=False)


def round_trip(change, prepare_reader=lambda m: None):
    """The owner builds Small, loads it and applies ``change``; a reader
    (after ``prepare_reader``) installs the export. Returns (owner, reader)."""
    owner = Small()
    constructed = load_diff.record_construction(owner)
    with torch.no_grad():
        owner.w.copy_(torch.arange(4.0))
    change(owner)
    out = export_of(owner, constructed)
    with torch.device("meta"):
        model = Small()
    prepare_reader(model)
    install(model, out.manifest, out.storages)
    return owner, model


def export_small(change):
    owner = Small()
    constructed = load_diff.record_construction(owner)
    change(owner)
    return export_of(owner, constructed)


class TestLoadingChanges(CustomTestCase):
    def test_an_item_kept_by_a_refill_still_gets_its_changes(self):
        def change(m):
            m.items[0].scale = 1
            m.items.append(3)

        _, model = round_trip(change)
        self.assertEqual(model.items[0].scale, 1)
        self.assertEqual(model.items[1:], ["local", 3])

    def test_a_new_container_at_two_places_stays_one(self):
        def change(m):
            m.a = m.b = {"k": [1]}

        _, model = round_trip(change)
        self.assertIs(model.a, model.b)

    def test_a_new_tensor_keeps_its_attributes(self):
        def change(m):
            m.extra = torch.ones(2)
            m.extra.format_ue8m0 = True

        _, model = round_trip(change)
        self.assertTrue(model.extra.format_ue8m0)

    def test_an_ordered_dict_keeps_its_new_order(self):
        _, model = round_trip(lambda m: m.od.move_to_end("a"))
        self.assertEqual(list(model.od), ["b", "a"])

    def test_a_plain_attribute_that_becomes_a_buffer(self):
        def change(m):
            del m.x
            m.register_buffer("x", torch.ones(1))

        _, model = round_trip(change)
        self.assertIn("x", model._buffers)

    def test_a_deep_change_of_a_module_level_object_is_refused(self):
        def change(m):
            GLOBAL["opts"]["inner"]["scale"] = 2

        try:
            with self.assertRaisesRegex(SnapshotError, "GLOBAL"):
                export_small(change)
        finally:
            GLOBAL["opts"]["inner"]["scale"] = 1

    def test_a_reclassed_module_is_refused(self):
        class Other(nn.Linear):
            pass

        with self.assertRaisesRegex(SnapshotError, "re-classed"):
            export_small(lambda m: setattr(m.lin, "__class__", Other))

    def test_reordered_hooks_are_refused(self):
        owner = Small()
        first = owner.lin.register_forward_hook(lambda *a: None)
        owner.lin.register_forward_hook(lambda *a: None)
        constructed = load_diff.record_construction(owner)
        owner.lin._forward_hooks.move_to_end(first.id)
        with self.assertRaisesRegex(SnapshotError, "reordered"):
            export_of(owner, constructed)

    def test_an_address_as_a_key_or_set_member_is_refused(self):
        for value in (lambda m: {m.w.data_ptr(): 1}, lambda m: {m.w.data_ptr()}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(SnapshotError, "address"):
                    export_small(lambda m: setattr(m, "ptrs", value(m)))

    def test_a_tensor_under_an_address_key_of_an_existing_dict_is_refused(self):
        """cache[w.data_ptr()] = w in a dict the model was built with: a
        reader's addresses differ, so it would not find the tensor there."""
        with self.assertRaisesRegex(SnapshotError, "address"):
            export_small(lambda m: m.d.__setitem__(m.w.data_ptr(), m.w))


class TestMoreLoadingChanges(CustomTestCase):
    def test_a_kept_list_that_grows_inside_a_list_that_grows(self):
        def change(m):
            m.nested[0].append(2)
            m.nested.append(3)

        _, model = round_trip(change)
        self.assertEqual(model.nested, [[1, 2], "local", 3])

    def test_a_refill_keeps_the_readers_own_unchanged_data(self):
        def mine(m):
            m.nested[1] = "mine"

        _, model = round_trip(lambda m: m.nested.append(3), prepare_reader=mine)
        self.assertEqual(model.nested[1], "mine")

    def test_a_container_built_at_two_places_that_loading_fills(self):
        class Shared(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.zeros(2), requires_grad=False)
                self.first = self.second = []
                self.ofirst = self.osecond = OrderedDict()

        owner = Shared()
        constructed = load_diff.record_construction(owner)
        owner.first.append(1)
        owner.ofirst["k"] = 2
        out = export_of(owner, constructed)
        with torch.device("meta"):
            model = Shared()
        install(model, out.manifest, out.storages)
        self.assertEqual(model.first, [1])
        self.assertIs(model.first, model.second)
        self.assertEqual(dict(model.ofirst), {"k": 2})
        self.assertIs(model.ofirst, model.osecond)

    def test_an_existing_list_that_loading_aliases(self):
        _, model = round_trip(lambda m: setattr(m, "alias", m.nested))
        self.assertIs(model.alias, model.nested)

    def test_a_tensor_where_an_object_with_the_same_attribute_was(self):
        def change(m):
            m.thing = torch.ones(1)
            m.thing.flag = True

        _, model = round_trip(change)
        self.assertTrue(model.thing.flag)

    def test_an_ordered_dict_with_a_new_first_key(self):
        def change(m):
            m.od["c"] = 3
            m.od.move_to_end("c", last=False)

        _, model = round_trip(change)
        self.assertEqual(list(model.od), ["c", "a", "b"])

    def test_a_defaultdict_factory_change(self):
        _, model = round_trip(lambda m: setattr(m.dd, "default_factory", int))
        self.assertIs(model.dd.default_factory, int)

    def test_an_empty_tuple_is_not_a_module_level_object(self):
        _, model = round_trip(lambda m: setattr(m, "x", ()))
        self.assertEqual(model.x, ())

    def test_a_registered_tensor_keeps_its_home_when_another_points_at_it(self):
        def change(m):
            m.w.peer = m.w2

        owner, model = round_trip(change)
        self.assertIs(model.w.peer, model.w2)
        self.assertIs(model._parameters["w2"], model.w2)


class TestReaderGuards(CustomTestCase):
    def install(self, manifest, storages):
        with torch.device("meta"):
            model = Small()
        install(model, manifest, storages)
        return model

    def test_no_writes_into_hooks_through_a_longer_path(self):
        out = export_small(lambda m: None)
        out.manifest.ops.append(
            ["set", ["lin", [["a", "_forward_hooks"], ["k", 1]]], 1]
        )
        with self.assertRaisesRegex(SnapshotError, "not a place to write"):
            self.install(out.manifest, out.storages)

    def test_no_refill_of_a_module_level_object(self):
        out = export_small(lambda m: None)
        out.manifest.ops.append(["fill", ["", [["a", "cfg"]]], ["dict", []]])
        with self.assertRaisesRegex(SnapshotError, "not a place to write"):
            self.install(out.manifest, out.storages)
        self.assertEqual(GLOBAL["opts"], {"inner": {"scale": 1}})

    def test_the_self_test_sees_geometry_and_split_storages(self):
        _, out = export()
        for tamper in (
            # the same tensor object, dtype, shape and offset, another stride
            lambda w: w.set_(w.untyped_storage(), 0, (3, 4), (1, 3)),
            # the same bytes, but no longer in the storage of the parameter
            lambda w: w.set_(w.clone().untyped_storage(), 0, w.shape, w.stride()),
        ):
            model = reader()
            tensors = installed_tensors(model, out)
            tamper(model.attn.conv_weights)
            with self.assertRaises(SnapshotError):
                check_install(model, out.manifest, tensors, CPU)


if __name__ == "__main__":
    unittest.main()
