"""The KV storage package imports downward only.

Order, lowest first: declarations; layouts; the plan and the state policy;
sizing; memory and allocation; the request table and its state; the router;
the storage; construction; the public module. A module may import its own
layer and the layers below it, never a layer above. Imports under
``if TYPE_CHECKING:`` are annotations only and are not counted.
"""

import ast
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PACKAGE = "sglang.srt.mem_cache.kv"
ROOT = Path(__file__).resolve().parents[5] / "python" / "sglang" / "srt" / "mem_cache"

# Module (relative to the package) -> layer; a directory entry covers its files.
LAYERS = {
    "decl": 0,
    "layout": 1,
    "plan": 2,
    "state_policy": 2,
    "sizing": 3,
    "memory": 4,
    "alloc": 4,
    "requests": 5,
    "advance": 5,
    "state_storage": 5,
    "router": 6,
    "routed_cache": 6,
    "storage": 7,
    "build": 8,
    "": 9,  # the package's public module
}

# Allowed upward imports, each with its reason. The test fails when one
# disappears (shrink this list) or a new one appears.
KNOWN_UPWARD = {
    # The static pools' per-family classes live in their layout modules and
    # build their own views and routers.
    ("layout.mha", "alloc.id_space"),
    ("layout.mha", "memory.views"),
    ("layout.mha", "router"),
    ("layout.mla", "alloc.id_space"),
    ("layout.mla", "memory.views"),
    ("layout.mla", "router"),
}


def _layer(module: str) -> int:
    head = module.split(".")[0] if module else ""
    if module in LAYERS:
        return LAYERS[module]
    if head not in LAYERS:
        raise AssertionError(f"{PACKAGE}.{module} has no layer in LAYERS")
    return LAYERS[head]


def _module_name(path: Path) -> str:
    rel = path.relative_to(ROOT / "kv").with_suffix("")
    parts = [p for p in rel.parts if p != "__init__"]
    return ".".join(parts)


def _type_checking_lines(tree: ast.AST) -> set:
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and (
            (isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING")
            or (
                isinstance(node.test, ast.Attribute)
                and node.test.attr == "TYPE_CHECKING"
            )
        ):
            for child in node.body:
                for sub in ast.walk(child):
                    if hasattr(sub, "lineno"):
                        lines.add(sub.lineno)
    return lines


def _package_imports(path: Path):
    tree = ast.parse(path.read_text())
    skip = _type_checking_lines(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.lineno in skip:
            continue
        mod = node.module or ""
        if mod == PACKAGE:
            yield ""
        elif mod.startswith(PACKAGE + "."):
            yield mod[len(PACKAGE) + 1 :]


def upward_imports():
    found = set()
    for path in sorted((ROOT / "kv").rglob("*.py")):
        name = _module_name(path)
        own = _layer(name)
        for target in _package_imports(path):
            if _layer(target) > own:
                found.add((name, target))
    return found


class TestImportOrder(CustomTestCase):
    def test_package_imports_downward(self):
        found = upward_imports()
        self.assertEqual(
            found - KNOWN_UPWARD,
            set(),
            "modules import a layer above their own",
        )
        self.assertEqual(
            KNOWN_UPWARD - found,
            set(),
            "an allowed upward import is gone: remove it from KNOWN_UPWARD",
        )


if __name__ == "__main__":
    unittest.main()
