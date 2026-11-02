"""model_walk: where a model holds its tensors and other values, by the same
rules for every caller."""

import collections
import enum
import unittest

import torch
from torch import nn

from sglang.srt.model_loader.model_walk import (
    MODULE_FIELDS,
    Place,
    container_items,
    get_at,
    get_item,
    plain_attributes,
    registered_tensors,
    replace_item,
    tensor_names,
    walk_plain_attributes,
    walk_value,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

Pair = collections.namedtuple("Pair", "first second")


class Color(enum.Enum):
    RED = 1


class Shape(tuple, enum.Enum):
    SQUARE = (4, 4)


class Hidden(dict):
    """A dict that shows nothing of what it stores."""

    def items(self):
        return iter(())

    def __iter__(self):
        return iter(())

    def __getitem__(self, key):
        raise KeyError(key)


class HiddenList(list):
    def __iter__(self):
        return iter(())

    def __getitem__(self, index):
        raise IndexError(index)


class HiddenTuple(tuple):
    def __iter__(self):
        return iter(())


def tensors(model: nn.Module, **kwargs):
    """{where: tensor} of the tensors a walk of ``model`` finds."""
    return {
        held.place.where: held.value
        for held in walk_plain_attributes(model, max_depth=5, **kwargs)
        if isinstance(held.value, torch.Tensor)
    }


class TestWalk(unittest.TestCase):
    def test_containers_paths_and_the_containers_on_the_way(self):
        model = nn.Module()
        t = torch.zeros(2)
        model.flat = t
        model.pair = Pair(t, (1, t))
        model.listed = [[t]]
        model.keyed = collections.OrderedDict(a={"b": t})
        model.size = torch.Size([2, 3])

        found = list(walk_plain_attributes(model, max_depth=5))

        where = {held.place.where: held for held in found}
        self.assertEqual(
            sorted(w for w, h in where.items() if h.value is t),
            sorted(
                [
                    "flat",
                    "pair[0]",
                    "pair[1][1]",
                    "listed[0][0]",
                    "keyed['a']['b']",
                ]
            ),
        )
        self.assertEqual(where["size[1]"].value, 3)
        self.assertEqual(where["pair[1][0]"].value, 1)
        self.assertEqual(where["flat"].containers, ())
        chain = where["keyed['a']['b']"].containers
        self.assertIs(chain[0], model.keyed)
        self.assertIs(chain[1], model.keyed["a"])
        self.assertEqual(where["pair[1][1]"].place, Place("", ("pair", 1, 1)))

    def test_items_are_read_as_the_builtin_type_stores_them(self):
        model = nn.Module()
        t = torch.zeros(2)
        model.d = Hidden(x=t)
        model.l = HiddenList([t])
        model.t = HiddenTuple((t,))

        self.assertEqual(sorted(tensors(model)), ["d['x']", "l[0]", "t[0]"])
        self.assertIs(get_item(model.d, "x"), t)
        self.assertIs(get_item(model.l, 0), t)
        self.assertEqual(container_items(model.t), [(0, t)])

    def test_dict_keys_of_any_type(self):
        model = nn.Module()
        t = torch.zeros(2)
        key = object()
        model.d = {("x", 0): t, 0.5: t, None: t, Color.RED: t, key: t}

        found = [h for h in walk_plain_attributes(model, max_depth=5)]

        self.assertEqual(
            [h.place.path[1] for h in found], [("x", 0), 0.5, None, Color.RED, key]
        )
        for held in found:
            self.assertIs(get_at(model, held.place), t)

    def test_a_shared_container_is_visited_along_each_path(self):
        model = nn.Module()
        t = torch.zeros(2)
        inner = [t]
        model.a = model.b = inner
        model.c = [inner, inner]

        self.assertEqual(sorted(tensors(model)), ["a[0]", "b[0]", "c[0][0]", "c[1][0]"])

    def test_cycles_and_depth_are_cut_and_reported(self):
        t = torch.zeros(2)
        loop = [t]
        loop.append(loop)
        deep = [[[[[[t]]]]]]  # six levels
        cuts = []

        found = list(
            walk_value(
                {"loop": loop, "deep": deep},
                ("x",),
                max_depth=5,
                on_cut=lambda path, reason: cuts.append((path, reason)),
            )
        )

        self.assertEqual([path for path, value, _ in found], [("x", "loop", 0)])
        self.assertEqual(
            cuts,
            [(("x", "loop", 1), "cycle"), (("x", "deep", 0, 0, 0, 0), "depth")],
        )
        self.assertEqual(list(walk_value([t], max_depth=0)), [])
        self.assertEqual(list(walk_value(t, ("t",), max_depth=0)), [(("t",), t, ())])

    def test_leaves_are_not_looked_into(self):
        model = nn.Module()
        t = torch.zeros(2)
        holder = type("Holder", (), {})()
        holder.t = t
        model.objects = [holder, {t}, "text", Color.RED, nn.Linear(1, 1)]
        model.shape = Shape.SQUARE  # an enum member that is a tuple

        leaves = {
            h.place.where: h.value for h in walk_plain_attributes(model, max_depth=5)
        }

        self.assertEqual(
            sorted(leaves),
            sorted(
                [
                    "objects[0]",
                    "objects[1]",
                    "objects[2]",
                    "objects[3]",
                    "objects[4]",
                    "shape[0]",
                    "shape[1]",
                ]
            ),
        )
        self.assertIs(leaves["objects[0]"], holder)

    def test_modules_once_and_no_bookkeeping(self):
        model = nn.Module()
        shared = nn.Module()
        shared.cached = torch.zeros(2)
        model.first, model.second = shared, shared
        model.weight = nn.Parameter(torch.zeros(2))
        model.register_buffer("table", torch.zeros(2))

        self.assertEqual(list(tensors(model)), ["first.cached"])
        names = [attr for attr, _ in plain_attributes(model)]
        self.assertFalse(set(names) & MODULE_FIELDS)
        self.assertNotIn("weight", names)


class TestRegistration(unittest.TestCase):
    def test_every_name_of_every_parameter_and_buffer(self):
        model = nn.Module()
        model.a, model.b = nn.Module(), nn.Module()
        weight = nn.Parameter(torch.zeros(2))
        model.a.weight = model.b.weight = weight
        table = torch.zeros(3)
        model.a.register_buffer("table", table)
        model.b.register_buffer("table", table)

        registered = registered_tensors(model)

        self.assertEqual(
            list(registered), ["a.weight", "b.weight", "a.table", "b.table"]
        )
        self.assertEqual(
            tensor_names(registered),
            {id(weight): ["a.weight", "b.weight"], id(table): ["a.table", "b.table"]},
        )
        self.assertEqual(Place.of_registered("a.table"), Place("a", ("table",), True))
        self.assertEqual(Place.of_registered("x"), Place("", ("x",), True))


class TestAccess(unittest.TestCase):
    def test_get_at(self):
        model = nn.Module()
        model.child = nn.Module()
        model.child.weight = nn.Parameter(torch.zeros(2))
        model.child.kept = {"k": [1, 2]}

        self.assertIs(
            get_at(model, Place.of_registered("child.weight")), model.child.weight
        )
        self.assertEqual(get_at(model, Place("child", ("kept", "k", 1))), 2)
        for place, error in (
            (Place("child", ("kept", "missing")), KeyError),
            (Place("child", ("kept", "k", 5)), IndexError),
            (Place("child", ("kept", "k", 0, 0)), TypeError),
            (Place("child", ("absent",)), KeyError),
            (Place("nobody", ("kept",)), AttributeError),
        ):
            with self.subTest(place=place.where):
                with self.assertRaises(error):
                    get_at(model, place)

    def test_a_missing_key_adds_nothing(self):
        cache = collections.defaultdict(list)
        with self.assertRaises(KeyError):
            get_item(cache, "x")
        with self.assertRaises(KeyError):
            replace_item(cache, "x", 1)
        self.assertEqual(dict(cache), {})

    def test_replace_item(self):
        ordered = collections.OrderedDict(a=1, b=2)
        ordered.move_to_end("a")
        listed = [1, 2]

        replace_item(ordered, "a", 3)
        replace_item(listed, 1, 4)
        replace_item(Hidden(x=1), "x", 5)

        self.assertEqual(list(ordered.items()), [("b", 2), ("a", 3)])
        self.assertEqual(listed, [1, 4])
        with self.assertRaises(IndexError):
            replace_item(listed, 2, 0)
        with self.assertRaises(TypeError):
            replace_item((1,), 0, 2)


if __name__ == "__main__":
    unittest.main()
