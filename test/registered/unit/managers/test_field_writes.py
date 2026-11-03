"""The owned-field write scan the request-state guards share."""

import ast
import tempfile
import unittest
from pathlib import Path

from sglang.test import field_writes
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_FIELDS = {"a", "b", "items"}
_SCAN = dict(
    owner_scopes={"Owner"},
    kwarg_funcs={"Owner", "replace"},
    owned_only_fields={"items"},
)


def _writes(source):
    return [
        field
        for node in ast.walk(ast.parse(source))
        for field in field_writes.writes(
            node,
            _FIELDS,
            kwarg_funcs=_SCAN["kwarg_funcs"],
            owned_only_fields=_SCAN["owned_only_fields"],
        )
    ]


def _scan(source):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mod.py"
        path.write_text(source)
        return field_writes.scan(path, _FIELDS, **_SCAN)


class TestFieldWrites(CustomTestCase):
    def test_catches_indirect_writes(self):
        cases = {
            "x.y.a = 1": "a",
            "x.a += 2": "a",
            "self.b = 0": "b",
            "p, x.b = 1, 2": "b",
            "setattr(x, 'a', 3)": "a",
            "object.__setattr__(x, 'b', 3)": "b",
            "delattr(x, 'a')": "a",
            "Owner(a=4)": "a",
            "dataclasses.replace(x, b=4)": "b",
            "del x.items[k]": "items",
            "d = x.items": "items",
            "f(x.items)": "items",
        }
        for source, field in cases.items():
            with self.subTest(source=source):
                self.assertIn(field, _writes(source))

        for source in ("y = x.a", "x.set(t, 3)", "s.replace('a', 'b')", "Owner()"):
            with self.subTest(source=source):
                self.assertEqual(_writes(source), [])

    def test_owner_scopes_may_write(self):
        source = (
            "class Owner:\n"
            "    def set(self, v):\n"
            "        self.a = v\n"
            "class Other:\n"
            "    def set(self, x, v):\n"
            "        x.a = v\n"
        )
        self.assertEqual(_scan(source), [("Other.set", "a", 6)])

    def test_skips_a_class_own_field_of_the_same_name(self):
        source = (
            "class Params:\n"
            "    items: dict\n"
            "    def set(self, k, v):\n"
            "        self.items[k] = v\n"
            "    def take(self, x):\n"
            "        return x.items\n"
        )
        self.assertEqual(_scan(source), [("Params.take", "items", 6)])


if __name__ == "__main__":
    unittest.main()
