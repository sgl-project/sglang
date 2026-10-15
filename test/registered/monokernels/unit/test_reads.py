# SPDX-License-Identifier: Apache-2.0
"""The contract between the plugin and SGLang (RFC #43949 section 11), checked on CPU from both sides.

Each plugin module lists the SGLang names it reads in ``SGLANG_READS``, by file. Every ``sglang.srt`` import of a
plugin module must name a declared file and name, and every declared entry must still exist in that file: a plain
name as a definition, an assignment target, an attribute assignment, a class field or an imported name; ``f(p)`` as
function ``f`` with parameter ``p``; ``'"x"'`` as the string literal ``x``. Both sides are read from source, so the
GPU-only kernel modules are checked without importing them."""

import ast
import pathlib
import re
import unittest

import sglang
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ROOT = pathlib.Path(sglang.__file__).parent
PLUGIN = ROOT / "experimental" / "monokernels"
CALL = re.compile(r"^(\w+)\((\w+)\)$")
LITERAL = re.compile(r'^"(.*)"$')


def srt_key(module: str) -> str:
    """The SGLANG_READS key of an sglang.srt module: its file relative to python/sglang."""
    rel = pathlib.Path(*module.split(".")[1:])
    path = ROOT / rel.with_suffix(".py")
    if not path.exists():
        path = ROOT / rel / "__init__.py"
    return str(path.relative_to(ROOT)).replace("\\", "/")


def declared_reads(tree: ast.Module) -> dict:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SGLANG_READS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    return {}


def srt_imports(tree: ast.AST) -> dict:
    """sglang.srt module -> names imported from it (at any depth, including lazy imports)."""
    out = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("sglang.srt")
        ):
            out.setdefault(node.module, set()).update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("sglang.srt"):
                    out.setdefault(a.name, set())
    return out


def defined_names(tree: ast.AST) -> set:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Name):
                        names.add(sub.id)
                    elif isinstance(sub, ast.Attribute):
                        names.add(sub.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
    return names


def function_params(tree: ast.AST) -> dict:
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            out.setdefault(node.name, set()).update(
                p.arg for p in a.posonlyargs + a.args + a.kwonlyargs
            )
    return out


def string_literals(tree: ast.AST) -> set:
    return {
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }


def plugin_files():
    return sorted(PLUGIN.rglob("*.py"))


class TestSGLangReads(unittest.TestCase):
    def test_every_srt_import_is_declared(self):
        for path in plugin_files():
            tree = ast.parse(path.read_text())
            reads = declared_reads(tree)
            for module, names in srt_imports(tree).items():
                key = srt_key(module)
                with self.subTest(file=str(path.relative_to(ROOT)), imports=module):
                    self.assertIn(
                        key, reads, f"imports {module} without SGLANG_READS[{key!r}]"
                    )
                    for n in names:
                        self.assertIn(n, reads[key], f"reads {module}.{n} undeclared")

    def test_every_declared_name_exists(self):
        checked = 0
        for path in plugin_files():
            for rel, entries in declared_reads(ast.parse(path.read_text())).items():
                target = ROOT / rel
                with self.subTest(file=str(path.relative_to(ROOT)), reads=rel):
                    self.assertTrue(target.exists(), f"{rel} no longer exists")
                    tree = ast.parse(target.read_text())
                    names, params, literals = (
                        defined_names(tree),
                        function_params(tree),
                        string_literals(tree),
                    )
                    for entry in entries:
                        checked += 1
                        call, lit = CALL.match(entry), LITERAL.match(entry)
                        if call:
                            f, p = call.groups()
                            self.assertIn(
                                p, params.get(f, set()), f"{rel}: {f}({p}) is gone"
                            )
                        elif lit:
                            self.assertIn(
                                lit.group(1),
                                literals,
                                f"{rel}: literal {entry} is gone",
                            )
                        else:
                            self.assertIn(entry, names, f"{rel}: {entry} is gone")
        self.assertGreater(checked, 50)

    def test_grammar(self):
        tree = ast.parse(
            "def f(a, *, b):\n    return 'x'\nclass C:\n    y: int = 0\nobj.z = 1\n"
        )
        self.assertEqual(function_params(tree)["f"], {"a", "b"})
        self.assertTrue({"f", "C", "y", "z"} <= defined_names(tree))
        self.assertIn("x", string_literals(tree))
        self.assertEqual(CALL.match("f(b)").groups(), ("f", "b"))
        self.assertEqual(LITERAL.match('"null"').group(1), "null")


if __name__ == "__main__":
    unittest.main()
