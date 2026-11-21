"""load_diff_format: the load-diff format: where the walk looks, storages and views, values
as data, and the checks a reader runs before it maps anything."""

import collections
import enum
import functools
import types
import unittest

import blake3
import msgspec
import numpy as np
import torch
from torch import nn

from sglang.srt.weight_cache import load_diff_format as load_diff
from sglang.srt.weight_cache.load_diff_format import (
    DecodeContext,
    EncodeContext,
    ModuleLevel,
    SnapshotError,
    Unencodable,
    build_storage_table,
    decode_manifest,
    decode_value,
    encode_manifest,
    encode_value,
    make_view,
    path_str,
    resolve_path,
    storage_digests,
    structure_digest,
    walk_model,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
# The GPU digest test compares the BLAKE3 kernel with the host digest.
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

CPU = torch.device("cpu")


class Kind(enum.Enum):
    A = "a"
    B = "b"


SETTINGS = types.SimpleNamespace(flag=False, options={"scale": 1})


def module_level_function():
    return 1


class Holder:
    """A plain object a module keeps, like a quant method or a top-k config."""

    def __init__(self, tensor):
        self.scale = tensor
        self.flag = False


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.arange(12.0).reshape(3, 4), requires_grad=False
        )
        self.register_buffer("table", torch.ones(4), persistent=False)
        self.register_parameter("bias", None)
        self.view = self.weight[1:]
        self.holder = Holder(self.weight)
        self.cache = {("w2", torch.Size([3, 4]), 128): self.weight.t()}
        self.settings = SETTINGS
        self.kind = Kind.A
        self.dtype = torch.float32

    def forward(self, x):
        return x @ self.weight.t()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = Layer()
        self.b = Layer()
        self.b.weight = self.a.weight  # tied
        self.first = self.a  # a second name for a module
        self.listed = [self.a]  # a module held in a plain list
        self.pair = (self.a.weight, 1)
        self.b.pair = self.pair  # one tuple, two places


def walk(model):
    return walk_model(model, module_level=ModuleLevel(), keep_objects=True)


class TestWalk(CustomTestCase):
    def test_finds_every_tensor_place(self):
        found = walk(Model())
        places = {path_str(p) for p in found.tensors}
        for where in (
            "a.weight",
            "a.table",
            "a.view",
            "a.holder.scale",
            "b.weight",
            "a.cache[('w2', torch.Size([3, 4]), 128)]",
            "<model>.pair[0]",
            "b.pair[0]",
        ):
            self.assertIn(where, places)
        self.assertEqual(found.problems, [])

    def test_registries_keep_none_and_persistence(self):
        found = walk(Model())
        registry = {name: rest for name, *rest in dict(found.registries)["a"]}
        self.assertEqual(registry["bias"][:2], ["p", None])
        self.assertEqual(registry["table"][0], "b")
        self.assertFalse(registry["table"][2])

    def test_describes_plain_values(self):
        found = walk(Model())
        values = found.values
        self.assertEqual(values[("a", ("a", "kind"))][0], "v")
        self.assertEqual(values[("a", ("a", "dtype"))][0], "v")
        self.assertEqual(values[("a", ("a", "holder"))][0], "o")
        self.assertEqual(values[("a", ("a", "holder"), ("a", "flag"))][0], "v")
        self.assertEqual(values[("a", ("a", "cache"))][0], "c")
        self.assertEqual(values[("", ("a", "listed"), ("i", 0))], ("m", "a"))
        # A module-level object is looked at, not into.
        self.assertEqual(values[("a", ("a", "settings"))][0], "g")
        self.assertNotIn(("a", ("a", "settings"), ("a", "flag")), values)
        self.assertIn(("a", ("a", "settings")), found.boundaries)

    def test_reader_stops_where_the_owner_stopped(self):
        owner = walk(Model())
        reader = walk_model(Model(), boundaries=set(owner.boundaries))
        self.assertEqual(set(owner.boundaries), set(reader.boundaries))
        self.assertEqual(set(owner.tensors), set(reader.tensors))

    def test_module_level_state_has_one_level_of_containers(self):
        model = Model()
        before = walk(model).values[("a", ("a", "settings"))]
        SETTINGS.options["scale"] = 2
        try:
            after = walk(model).values[("a", ("a", "settings"))]
        finally:
            SETTINGS.options["scale"] = 1
        self.assertEqual(before[:3], after[:3])
        self.assertNotEqual(before[3], after[3])

    def test_floats_compare_by_their_bytes(self):
        model = nn.Module()
        model.x, model.y, model.z = float("nan"), 0.0, -0.0
        values = walk(model).values
        self.assertEqual(walk(model).values[("", ("a", "x"))], values[("", ("a", "x"))])
        self.assertNotEqual(values[("", ("a", "y"))], values[("", ("a", "z"))])

    def test_too_deep_is_a_problem(self):
        model = nn.Module()
        nested = []
        model.deep = nested
        for _ in range(load_diff.MAX_DEPTH + 2):
            nested.append([])
            nested = nested[0]
        self.assertTrue(any("deeper than" in p for p in walk(model).problems))

    def test_cycles_stop(self):
        model = nn.Module()
        loop = []
        loop.append(loop)
        model.loop = loop
        found = walk(model)
        self.assertEqual(found.values[("", ("a", "loop"), ("i", 0))][0], "y")

    def test_process_local_fields_are_walked_apart(self):
        class Kernel:
            _process_local_fields = frozenset({"scratch"})

            def __init__(self):
                self.scratch = torch.zeros(3)
                self.table = torch.ones(2)

        class Layer(nn.Module):
            _process_local_fields = frozenset({"workspace", "unset"})

            def __init__(self):
                super().__init__()
                self.workspace = torch.zeros(4)
                self.unset = None
                self.kernel = Kernel()

        found = walk(Layer())
        workspace = ("", ("a", "workspace"))
        scratch = ("", ("a", "kernel"), ("a", "scratch"))
        self.assertEqual(found.values[workspace], ("l",))
        self.assertEqual(set(found.local), {workspace, scratch})
        self.assertNotIn(workspace, found.tensors)
        self.assertIn(("", ("a", "kernel"), ("a", "table")), found.tensors)
        # A declared field that holds no tensor is a plain value.
        self.assertEqual(found.values[("", ("a", "unset"))][0], "v")

    def test_a_declared_field_that_is_registered_is_a_problem(self):
        class Layer(nn.Module):
            _process_local_fields = frozenset({"workspace"})

            def __init__(self):
                super().__init__()
                self.register_buffer("workspace", torch.zeros(4))

        (problem,) = walk(Layer()).problems
        self.assertIn("workspace", problem)
        self.assertIn("declared process-local, but registered", problem)

    def test_resolve_path_runs_no_property(self):
        class Trap:
            @property
            def value(self):
                raise AssertionError("a property ran")

        model = nn.Module()
        model.trap = Trap()
        vars(model.trap)["value"] = 3
        modules = dict(model.named_modules(remove_duplicate=False))
        self.assertEqual(resolve_path(modules, ("", ("a", "trap"), ("a", "value"))), 3)


class TestStorageTable(CustomTestCase):
    def test_views_tied_and_ordered_by_first_use(self):
        model = Model()
        table = build_storage_table(walk(model).tensors, CPU)
        self.assertEqual(table.problems, [])
        weight = model.a.weight
        self.assertEqual(table.views[id(weight)], (0, 0, "float32", [3, 4], [4, 1]))
        self.assertEqual(table.views[id(model.a.view)][:2], (0, 16))
        transposed = model.a.cache[("w2", torch.Size([3, 4]), 128)]
        self.assertEqual(table.views[id(transposed)][4], [1, 4])
        self.assertEqual(len({v[0] for v in table.views.values()}), len(table.storages))
        self.assertTrue(all(where == "model" for where, _, _ in table.storages))

    def test_contained_storage_joins_its_container(self):
        base = torch.arange(16, dtype=torch.uint8)
        inner = torch.frombuffer(memoryview(base.numpy())[4:], dtype=torch.uint8)
        self.assertNotEqual(
            base.untyped_storage()._cdata, inner.untyped_storage()._cdata
        )
        table = build_storage_table(
            {("", ("a", "x")): base, ("", ("a", "y")): inner}, CPU
        )
        self.assertEqual(table.problems, [])
        self.assertEqual(len(table.storages), 1)
        self.assertEqual(table.views[id(inner)][:2], (0, 4))

    def test_partial_overlap_is_refused(self):
        buffer = np.zeros(16, dtype=np.uint8)
        left = torch.frombuffer(memoryview(buffer)[0:8], dtype=torch.uint8)
        right = torch.frombuffer(memoryview(buffer)[4:12], dtype=torch.uint8)
        table = build_storage_table(
            {("", ("a", "l")): left, ("", ("a", "r")): right}, CPU
        )
        self.assertTrue(any("overlap" in p for p in table.problems))

    def test_empty_storages_stay_apart(self):
        x, y = torch.empty(0), torch.empty(0)
        table = build_storage_table({("", ("a", "x")): x, ("", ("a", "y")): y}, CPU)
        self.assertEqual(len(table.storages), 2)

    def test_meta_tensor_is_a_problem(self):
        meta = {("", ("a", "m")): torch.empty(2, device="meta")}
        self.assertTrue(
            any("meta" in p for p in build_storage_table(meta, CPU).problems)
        )

    def test_views_rebuild_on_copied_storages(self):
        model = Model()
        table = build_storage_table(walk(model).tensors, CPU)
        copies = [whole.clone().untyped_storage() for _, _, whole in table.storages]
        for tensor in (model.a.weight, model.a.view, model.a.table):
            sid, offset, dtype, shape, stride = table.views[id(tensor)]
            rebuilt = make_view(copies[sid], offset, dtype, shape, stride)
            self.assertEqual(rebuilt.stride(), tensor.stride())
            torch.testing.assert_close(rebuilt, tensor.detach())


class TestValues(CustomTestCase):
    def setUp(self):
        self.model = Model()
        self.modules = dict(self.model.named_modules(remove_duplicate=False))
        self.tensors = [self.model.a.weight.detach()]
        self.enc_ctx = EncodeContext(
            tensor_ids={id(self.model.a.weight): 0},
            first_paths={id(self.model.a.holder): ("a", ("a", "holder"))},
            module_names={id(self.model.a): "a"},
        )
        self.dec_ctx = DecodeContext(self.modules, self.tensors, CPU)

    def round_trip(self, value, path=("b", ("a", "x"))):
        return decode_value(encode_value(value, path, self.enc_ctx), path, self.dec_ctx)

    def test_data_round_trips(self):
        values = [
            None, True, 3, 2.5, float("inf"), "s", b"bytes",
            torch.bfloat16, torch.device("cpu"), torch.Size([2, 3]), Kind.B,
            {1, 2}, frozenset({"a"}),
            (1, [2, {"k": (3,)}]),
            collections.OrderedDict(b=1, a=2),
            {torch.float16: 1, (Kind.A, torch.Size([1])): 2},
        ]  # fmt: skip
        for value in values:
            with self.subTest(value=value):
                out = self.round_trip(value)
                self.assertEqual(out, value)
                self.assertIs(type(out), type(value))

    def test_defaultdict_keeps_a_builtin_factory(self):
        out = self.round_trip(collections.defaultdict(list, {"a": [1]}))
        self.assertIs(out.default_factory, list)
        self.assertEqual(out["a"], [1])

    def test_references_to_what_the_reader_has(self):
        self.assertIs(self.round_trip(self.model.a.weight), self.tensors[0])
        self.assertIs(self.round_trip(self.model.a), self.model.a)
        self.assertIs(self.round_trip(self.model.a.holder), self.model.a.holder)

    def test_keep_gives_what_the_reader_holds_there(self):
        path = ("a", ("a", "cache"))
        ctx = EncodeContext(keep=lambda here, value: here == path + (("i", 1),))
        encoded = encode_value([1, object()], path, ctx)
        self.assertEqual(encoded, ["list", [1, ["keep"]]])
        reader_value = ["x", "kept"]
        vars(self.model.a)["cache"] = reader_value
        self.assertEqual(decode_value(encoded, path, self.dec_ctx), [1, "kept"])

    def test_addresses_are_refused(self):
        ctx = EncodeContext(addresses=[(0x1000, 0x2000)])
        self.assertEqual(encode_value(0x2000, ("", ("a", "x")), ctx), 0x2000)
        with self.assertRaises(Unencodable):
            encode_value(0x1800, ("", ("a", "x")), ctx)

    def test_code_and_new_objects_are_refused(self):
        refused = [
            lambda: 1,
            module_level_function,
            functools.partial(module_level_function),
            self.model.a.forward,
            Holder(None),
            SETTINGS,
            np.zeros(2),
            collections.namedtuple("Pair", "a b")(1, 2),
            collections.defaultdict(module_level_function),
            {object(): 1},
        ]
        for value in refused:
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(Unencodable):
                    encode_value(value, ("b", ("a", "x")), self.enc_ctx)
        loop = []
        loop.append(loop)
        with self.assertRaises(Unencodable):
            encode_value(loop, ("b", ("a", "x")), self.enc_ctx)

    def test_decoding_imports_nothing(self):
        with self.assertRaises(KeyError):
            decode_value(
                ["enum", "a_module_nobody_imported", "E", "A"], ("",), self.dec_ctx
            )
        with self.assertRaises(ValueError):
            decode_value(["dtype", "nn"], ("",), self.dec_ctx)


def manifest(**overrides):
    m = {
        "version": load_diff.FORMAT_VERSION,
        "structure": [bytes(32), 3, 4],
        "storages": [[48, "model", bytes(32)]],
        "tensors": [
            [
                0,
                0,
                "float32",
                [3, 4],
                [4, 1],
                "parameter",
                False,
                ["a", [["p", "weight"]]],
            ]
        ],
        "registered": [["a", [["weight", "p", 0, True]]]],
        "places": [[["a", [["a", "view"]]], 0]],
        "ops": [["set", ["a", [["a", "flag"]]], True]],
        "boundaries": [],
        "removed_modules": [],
        "local": [],
    }
    m.update(overrides)
    return m


def check(m):
    """``m`` (plain data) as a reader decodes and checks it."""
    return decode_manifest(msgspec.msgpack.encode(m))


class TestValidateManifest(CustomTestCase):
    def test_accepts_a_good_one(self):
        check(manifest())
        check(manifest(local=[["int32", [2, 3], "model", [["a", [["a", "scratch"]]]]]]))

    def test_refuses_bad_ones(self):
        home = ["a", [["p", "weight"]]]
        deep_key = functools.reduce(
            lambda acc, _: ["tuple", [acc]], range(load_diff.MAX_DEPTH + 2), 0
        )
        bad = {
            "version": manifest(version=99),
            "version True": manifest(version=True),
            "a list as a dict key": manifest(
                ops=[["set", ["a", [["a", "x"]]], ["dict", [[["list", []], 1]]]]]
            ),
            "a key too deep": manifest(ops=[["del", ["a", [["a", "x"], ["k", deep_key]]]]]),
            "home not a place of its tensor": manifest(
                tensors=[[0, 0, "float32", [3, 4], [4, 1], "parameter", False, ["a", [["a", "y"]]]]]
            ),
            "one path twice": manifest(
                ops=[["set", ["a", [["a", "x"]]], 1], ["del", ["a", [["a", "x"]]]]]
            ),
            "negative size": manifest(storages=[[-1, "model", bytes(32)]]),
            "short digest": manifest(storages=[[48, "model", bytes(8)]]),
            "an extra column": manifest(storages=[[48, "model", bytes(32), 0]]),
            "a dtype that is a list": manifest(ops=[["set", ["a", [["a", "x"]]], ["dtype", []]]]),
            "a factory that is a list": manifest(
                ops=[["set", ["a", [["a", "x"]]], ["ddict", [], []]]]
            ),
            "view past its storage": manifest(
                tensors=[[0, 4, "float32", [3, 4], [4, 1], "parameter", False, home]]
            ),
            "misaligned view": manifest(tensors=[[0, 2, "float32", [2], [1], "tensor", False, home]]),
            "unknown dtype": manifest(tensors=[[0, 0, "nn", [1], [1], "tensor", False, home]]),
            "grad on an int": manifest(tensors=[[0, 0, "int32", [1], [1], "tensor", True, home]]),
            "bool as an int": manifest(tensors=[[True, 0, "float32", [1], [1], "tensor", False, home]]),
            "tensor id out of range": manifest(places=[[["a", [["a", "v"]]], 7]]),
            "registered twice": manifest(
                registered=[["a", [["weight", "p", 0, True], ["weight", "b", 0, True]]]]
            ),
            "unknown tag": manifest(ops=[["set", ["a", [["a", "x"]]], ["exec", "code"]]]),
            "a local of an unknown dtype": manifest(
                local=[["nn", [2], "model", [["a", [["a", "s"]]]]]]
            ),
            "a local at one place twice": manifest(
                local=[["int32", [2], "model", [["a", [["a", "s"]]], ["a", [["a", "s"]]]]]]
            ),
            "a local at a module": manifest(local=[["int32", [2], "model", [["a", []]]]]),
            "a local where a tensor is": manifest(
                local=[["int32", [2], "model", [["a", [["a", "view"]]]]]]
            ),
            "a local with no place": manifest(local=[["int32", [2], "model", []]]),
            "unknown step": manifest(ops=[["del", ["a", [["z", "x"]]]]]),
            "view past a cpu storage": manifest(storages=[[4, "cpu", bytes(32)]]),
            "missing field": {k: v for k, v in manifest().items() if k != "ops"},
            "too deep": manifest(
                ops=[["set", ["a", [["a", "x"]]], functools.reduce(
                    lambda acc, _: ["list", [acc]], range(load_diff.MAX_DEPTH + 2), 0)]]
            ),
        }  # fmt: skip
        for why, m in bad.items():
            with self.subTest(why=why):
                with self.assertRaises(SnapshotError):
                    check(m)

    def test_an_empty_view_fits_anywhere_in_bounds(self):
        check(
            manifest(
                tensors=[[0, 48, "float32", [0], [1], "tensor", False, ["a", [["p", "weight"]]]]]
            )
        )  # fmt: skip


class TestManifestBytes(CustomTestCase):
    def test_round_trip(self):
        m = manifest(
            ops=[
                ["set", ["a", [["a", "flag"]]], ["tuple", [1, 2.5, "x", b"y", None]]],
                ["fill", ["a", [["a", "cfg"]]], ["dict", [["k", ["list", [True]]]]]],
            ]
        )
        data = msgspec.msgpack.encode(m)
        # The typed manifest encodes to the very bytes of the plain data.
        self.assertEqual(encode_manifest(decode_manifest(data)), data)

    def test_refuses_what_is_not_a_manifest(self):
        bad = {
            "not bytes": bytearray(encode_manifest(manifest())),
            "not msgpack": b"\xc1",
            "truncated": encode_manifest(manifest())[:-3],
            "not a manifest": msgspec.msgpack.encode({"version": 1}),
        }
        for why, data in bad.items():
            with self.subTest(why=why):
                with self.assertRaises(SnapshotError):
                    decode_manifest(data)

    def test_an_int_past_64_bits_cannot_be_encoded(self):
        with self.assertRaises(SnapshotError):
            encode_manifest(manifest(ops=[["set", ["a", [["a", "x"]]], 1 << 70]]))


class TestDigests(CustomTestCase):
    def test_cpu_digest_is_blake3_of_the_bytes(self):
        data = torch.arange(100, dtype=torch.uint8)
        empty = torch.empty(0, dtype=torch.uint8)
        self.assertEqual(
            storage_digests([data, empty]),
            [blake3.blake3(bytes(range(100))).digest(), blake3.blake3(b"").digest()],
        )

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_gpu_digest_equals_cpu_digest(self):
        data = torch.randint(0, 255, (3 << 20,), dtype=torch.uint8)
        self.assertEqual(storage_digests([data.cuda()]), storage_digests([data]))


class TestStructureDigest(CustomTestCase):
    def test_same_construction_same_digest_on_any_device(self):
        with torch.device("meta"):
            on_meta = Model()
        self.assertEqual(structure_digest(Model()), structure_digest(on_meta))

    def test_a_different_shape_changes_it(self):
        other = Model()
        other.a.extra = nn.Parameter(torch.zeros(2))
        self.assertNotEqual(structure_digest(Model())[0], structure_digest(other)[0])

    def test_module_sharing_changes_it(self):
        other = Model()
        other.first = Layer()  # same names and classes, no longer the same module
        self.assertNotEqual(structure_digest(Model())[0], structure_digest(other)[0])


if __name__ == "__main__":
    unittest.main()
