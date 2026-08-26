"""Ratchet guard: module-level choice lists in ``server_args.py`` may only decrease.

A module-level ``*_CHOICES``-style list in ``server_args.py`` means one thing:
out-of-tree platforms and plugins extend it before ``ServerArgs`` is built. A
field whose choices nobody extends does not need a name -- the values go inline
in ``Arg(choices=[...])``.

So the count never grows on its own. Adding an extension point is a deliberate
public-API decision: raise the baseline in the same change, with a note saying
who extends it. When a list is inlined or deleted, lower the baseline.
"""

import ast
import unittest
from pathlib import Path

import sglang.srt.server_args as server_args_module
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Extension points: a Choices list plus its `add_*` alias on the next line.
_EXTENSION_POINT_BASELINE = 17

# Shared choice lists: referenced by more than one field, so they keep a name,
# but nothing extends them and they get no adder.
# DSA_TOPK_BACKEND_CHOICES is shared by target and speculative-draft settings.
_SHARED_LIST_BASELINE = 4


def _module_level_lists():
    """Names assigned at module level in server_args.py, split by kind.

    Read from the source rather than ``dir()``: the module also imports choice
    lists that other modules own (``SUPPORTED_LORA_TARGET_MODULES``), and those
    are not this file's extension points.
    """
    tree = ast.parse(Path(server_args_module.__file__).read_text())
    extension_points, shared = [], []
    for node in tree.body:
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            continue
        name = node.targets[0].id
        if not name.isupper():
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "Choices"
        ):
            extension_points.append(name)
        elif isinstance(value, (ast.List, ast.Name)):
            shared.append(name)
    return sorted(extension_points), sorted(shared)


class TestChoicesExtensionPointRatchet(CustomTestCase):
    def test_extension_points_do_not_grow(self):
        extension_points, _ = _module_level_lists()
        self.assertLessEqual(
            len(extension_points),
            _EXTENSION_POINT_BASELINE,
            "New module-level Choices list(s) in server_args.py: "
            f"{extension_points}. A choice list only belongs at module level "
            "if out-of-tree code extends it; otherwise inline the values into "
            "the field's Arg(choices=[...]). If this really is a new extension "
            "point, raise _EXTENSION_POINT_BASELINE in this file and say who "
            "extends it.",
        )

    def test_shared_lists_do_not_grow(self):
        _, shared = _module_level_lists()
        self.assertLessEqual(
            len(shared),
            _SHARED_LIST_BASELINE,
            f"New module-level plain choice list(s) in server_args.py: {shared}. "
            "Inline the values into the field instead of hoisting a name.",
        )

    def test_every_extension_point_has_an_adder(self):
        """Guards the list-added-without-its-adder case: a Choices list with no
        add_* alias is an extension point out-of-tree code cannot reach."""
        extension_points, _ = _module_level_lists()
        missing = []
        for name in extension_points:
            target = getattr(server_args_module, name)
            if not any(
                getattr(getattr(server_args_module, candidate), "__self__", None)
                is target
                for candidate in dir(server_args_module)
                if candidate.startswith("add_")
            ):
                missing.append(name)
        self.assertEqual(
            missing,
            [],
            f"Choices list(s) with no add_* alias: {missing}. Either bind one on "
            "the line below the list, or make it a plain list (or inline it).",
        )


if __name__ == "__main__":
    unittest.main()
