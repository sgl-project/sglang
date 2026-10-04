"""Each decoder layer builds its stage boundaries once."""

import ast
import unittest
from pathlib import Path

import sglang.srt.models
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_BUILDERS = {"make_stages", "make_attn_stage", "make_ffn_stage"}


def _called(node):
    return {
        call.func.attr
        if isinstance(call.func, ast.Attribute)
        else getattr(call.func, "id", None)
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    }


def _builders(trees):
    """Functions that build stage boundaries, directly or through another
    such function. Derived from the code, so a new helper joins unlisted."""
    names, functions = set(_BUILDERS), {}
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name != "__init__":
                functions.setdefault(node.name, []).append(node)
    changed = True
    while changed:
        changed = False
        for name, nodes in functions.items():
            if name not in names and any(_called(n) & names for n in nodes):
                names.add(name)
                changed = True
    return names


def _base_init_calls(init, bases):
    """super().__init__(...) and Base.__init__(self, ...) calls."""
    for call in ast.walk(init):
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "__init__"
        ):
            continue
        owner = call.func.value
        if (
            isinstance(owner, ast.Call) and getattr(owner.func, "id", None) == "super"
        ) or (ast.unparse(owner).split(".")[-1] in bases):
            yield call


def _skips_base_build(call):
    return any(
        keyword.arg == "build_stages"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is False
        for keyword in call.keywords
    )


def _layer_classes(trees):
    classes = {}
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                init = next(
                    (
                        m
                        for m in node.body
                        if isinstance(m, ast.FunctionDef) and m.name == "__init__"
                    ),
                    None,
                )
                bases = {ast.unparse(b).split(".")[-1] for b in node.bases}
                # Same-named classes in different files: keep every one, so a
                # name builds if any class of that name does (fails closed).
                classes.setdefault(node.name, []).append((path, node, init, bases))
    return classes


def _twice_built(trees):
    """Classes whose __init__ builds stages after a base __init__ that
    already built them; and every class whose construction builds stages."""
    builders = _builders(trees)
    classes = _layer_classes(trees)
    building, found = set(), []

    def builds(name, seen=frozenset()):
        if name in building:
            return True
        if name in seen or name not in classes:
            return False
        for _, _, init, bases in classes[name]:
            if init is None:
                if any(builds(b, seen | {name}) for b in bases):
                    return True
                continue
            if _called(init) & builders:
                return True
            if any(
                not _skips_base_build(call)
                and any(builds(b, seen | {name}) for b in bases)
                for call in _base_init_calls(init, bases)
            ):
                return True
        return False

    for name, defs in classes.items():
        if builds(name):
            building.add(name)
        for path, node, init, bases in defs:
            if init is None or not _called(init) & builders:
                continue
            if any(
                not _skips_base_build(call) and any(builds(b) for b in bases)
                for call in _base_init_calls(init, bases)
            ):
                found.append(f"{path.name}:{node.lineno} {name}")
    return building, found


class TestStageBoundariesBuiltOnce(CustomTestCase):
    def test_no_layer_builds_its_stage_boundaries_twice(self):
        # A second build discards the first pair, which was bound with the
        # base class's norms; the base must be told not to build.
        (models,) = map(Path, sglang.srt.models.__path__)
        trees = {p: ast.parse(p.read_text()) for p in sorted(models.rglob("*.py"))}
        building, found = _twice_built(trees)
        # The builders are found from the code; if none were, nothing is checked.
        self.assertGreater(len(building), 40)
        self.assertEqual(found, [])

    def test_a_subclass_that_builds_again_is_reported(self):
        source = (
            "class Base(nn.Module):\n"
            "    def __init__(self, build_stages=True):\n"
            "        super().__init__()\n"
            "        if build_stages:\n"
            "            self.stages = self._build_stages()\n"
            "    def _build_stages(self):\n"
            "        return make_stages((declare_attn(), norm), (declare_ffn(), norm))\n"
            "class Twice(Base):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.stages = self._build_stages()\n"
            "class Once(Base):\n"
            "    def __init__(self):\n"
            "        super().__init__(build_stages=False)\n"
            "        self.stages = self._build_stages()\n"
            "class Inherits(Base):\n"
            "    pass\n"
        )
        building, found = _twice_built({Path("layers.py"): ast.parse(source)})
        self.assertEqual(building, {"Base", "Twice", "Once", "Inherits"})
        self.assertEqual(found, ["layers.py:8 Twice"])


if __name__ == "__main__":
    unittest.main()
