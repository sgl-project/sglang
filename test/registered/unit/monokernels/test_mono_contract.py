"""The SGLang names the monokernel framework reads, pinned on CPU.

The framework keeps its runtime integration out of srt, so it reads SGLang
internals by name, mostly in calls no CPU test runs. Every framework module that
imports from ``sglang.srt`` lists what it reads in ``SGLANG_READS``. A refactor
that renames one fails here, naming the module to repair, instead of failing at a
GPU server's startup. The checks are static (AST).
"""

import ast
import unittest
from functools import lru_cache
from pathlib import Path

from sglang.experimental.monokernels import spec
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, stage="base-a", runner_config="cpu")

ROOT = Path(spec.__file__).resolve().parents[2]  # python/sglang


@lru_cache(maxsize=None)
def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


@lru_cache(maxsize=None)
def _provided(path: Path) -> frozenset:
    """What a source file provides: definitions, assignment targets (``x.name = ...``
    too), imported names, parameters as ``f(p)`` and string literals as ``"x"``."""
    names = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if not isinstance(node, ast.ClassDef):
                params = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                names.update(f"{node.name}({p.arg})" for p in params)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for t in ast.walk(target):
                    if isinstance(t, ast.Name):
                        names.add(t.id)
                    elif isinstance(t, ast.Attribute):
                        names.add(t.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(f'"{node.value}"')
    return frozenset(names)


def _declared(path: Path) -> dict:
    """The module's ``SGLANG_READS``, or {} without one."""
    for node in _tree(path).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SGLANG_READS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    return {}


def _srt_imports(path: Path) -> set:
    return {
        a.asname or a.name
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("sglang.srt")
        for a in node.names
    }


READERS = [
    p
    for p in sorted((ROOT / "experimental/monokernels").rglob("*.py"))
    if _declared(p) or _srt_imports(p)
]


class TestSglangReads(CustomTestCase):
    def test_every_srt_import_is_declared(self):
        for path in READERS:
            with self.subTest(module=str(path.relative_to(ROOT))):
                declared = {n for names in _declared(path).values() for n in names}
                undeclared = _srt_imports(path) - declared
                self.assertFalse(
                    undeclared, f"declare {sorted(undeclared)} in SGLANG_READS"
                )

    def test_sglang_provides_what_the_framework_reads(self):
        for path in READERS:
            with self.subTest(module=str(path.relative_to(ROOT))):
                missing = []
                for pattern, names in _declared(path).items():
                    files = sorted(ROOT.glob(pattern))
                    self.assertTrue(files, f"no file matches {pattern}")
                    provided = frozenset().union(*map(_provided, files))
                    missing += [f"{pattern}: {n}" for n in names if n not in provided]
                self.assertFalse(
                    missing,
                    "SGLang no longer provides what this module reads; repair it "
                    "and its SGLANG_READS:\n  " + "\n  ".join(missing),
                )


if __name__ == "__main__":
    unittest.main()
