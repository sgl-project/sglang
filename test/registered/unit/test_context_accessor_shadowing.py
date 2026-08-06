"""No local may shadow a ``runtime_context`` accessor it also calls.

A mechanical sweep that rewrites ``self.server_args.mamba_cache_chunk_size``
into ``mamba_cache_chunk_size()`` turns

    mamba_cache_chunk_size = self.server_args.mamba_cache_chunk_size

into ``mamba_cache_chunk_size = mamba_cache_chunk_size()``, which is a
self-referential local: the name is local for the whole function, so the call
raises ``UnboundLocalError`` the first time that line runs. Five of these
shipped in one sweep and only one had unit coverage — a mamba model on the
radix-cache-v2 path found it at request time.

This scans for the shape directly: a function-scope assignment whose target
name is an imported accessor.
"""

import ast
import unittest
from pathlib import Path

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PACKAGE_ROOT = Path(next(iter(sglang.__path__)))
_CONTEXT_MODULE = "sglang.srt.runtime_context"


def _imported_accessors(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == _CONTEXT_MODULE:
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _shadowing_assignments(tree: ast.AST, accessors: set[str]):
    """Assignments whose target shadows an accessor the module imported."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                continue
            targets = inner.targets if isinstance(inner, ast.Assign) else [inner.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in accessors:
                    yield node.name, target.id, inner.lineno


class TestNoAccessorShadowing(CustomTestCase):
    def test_no_local_shadows_a_context_accessor(self):
        offenders = []
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            rel = path.relative_to(_PACKAGE_ROOT).as_posix()
            if rel.startswith("srt/runtime_context.py"):
                continue
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            accessors = _imported_accessors(tree)
            if not accessors:
                continue
            for func, name, lineno in _shadowing_assignments(tree, accessors):
                offenders.append(f"{rel}:{lineno}: {func}() binds {name!r}")
        self.assertFalse(
            offenders,
            "locals shadow a runtime_context accessor imported in the same "
            "module; the name is local for the whole function, so any call to "
            "the accessor there raises UnboundLocalError:\n" + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
