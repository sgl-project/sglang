# SPDX-License-Identifier: Apache-2.0
"""The monokernels import from aiter only what the stock aiter of the tested image defines (e7d2453f, the manifest
aiter_e7d2453f.json next to this file, generated from its sources): every ``from aiter... import name``, every
attribute read off an imported aiter module (``bo.buffer_load``) and every attribute of an imported aiter class
(``UncachedIpcHeap.alloc_uncached``). Reads sources only, so it runs without aiter and never skips."""

import ast
import json
import pathlib
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PLUGIN = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python/sglang/experimental/monokernels"
)
MANIFEST = json.loads(
    (pathlib.Path(__file__).with_name("aiter_e7d2453f.json")).read_text()
)


def needs():
    """(file, aiter module, name or Class.attr) for every aiter name the plugin uses."""
    modules = MANIFEST["modules"]
    out = []
    for path in sorted(PLUGIN.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        mods, classes = {}, {}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and (node.module or "").split(".")[0] == "aiter"
            ):
                for a in node.names:
                    full = f"{node.module}.{a.name}"
                    if full in modules:
                        mods[a.asname or a.name] = full
                    else:
                        out.append((path.name, node.module, a.name))
                        classes[a.asname or a.name] = (node.module, a.name)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] == "aiter":
                        out.append((path.name, a.name, ""))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id in mods:
                    out.append((path.name, mods[node.value.id], node.attr))
                elif node.value.id in classes:
                    mod, cls = classes[node.value.id]
                    out.append((path.name, mod, f"{cls}.{node.attr}"))
    return sorted(set(out))


class TestAiterImports(unittest.TestCase):
    def test_manifest_is_the_stock_aiter(self):
        self.assertTrue(MANIFEST["aiter"].startswith("e7d2453f"))

    def test_every_name_is_in_the_manifest(self):
        missing, checked = [], 0
        for fname, module, name in needs():
            checked += 1
            entry = MANIFEST["modules"].get(module)
            if entry is None:
                missing.append(f"{fname}: module {module} is not in the manifest")
                continue
            cls, _, attr = name.partition(".")
            ok = (not name or cls in entry["names"]) and (
                not attr or attr in entry["classes"].get(cls, [])
            )
            if not ok:
                missing.append(f"{fname}: {module}.{name}")
        self.assertGreater(checked, 20)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
