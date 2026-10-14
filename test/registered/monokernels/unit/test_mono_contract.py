"""The SGLang names the monokernel plugin reads (RFC #43949), pinned on CPU.

The plugin keeps SGLang's runtime integration out of srt, so it reads SGLang
internals by name. Every plugin module that imports from ``sglang.srt`` lists what
it reads in ``SGLANG_READS``. A refactor that renames one fails here, naming the
plugin file to repair, instead of failing at a GPU server's startup. The checks
are static (AST): no plugin module is imported.
"""

import ast
from functools import lru_cache
from pathlib import Path

import pytest

from sglang.experimental.monokernels import spec
from sglang.test.ci.ci_register import register_cpu_ci

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
        targets = getattr(node, "targets", [getattr(node, "target", None)])
        if any(getattr(t, "id", None) == "SGLANG_READS" for t in targets):
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


def _id(path: Path) -> str:
    return str(path.relative_to(ROOT))


@pytest.mark.parametrize("path", READERS, ids=_id)
def test_every_srt_import_is_declared(path):
    declared = {n for names in _declared(path).values() for n in names}
    undeclared = _srt_imports(path) - declared
    assert not undeclared, f"{_id(path)}: declare {sorted(undeclared)} in SGLANG_READS"


@pytest.mark.parametrize("path", READERS, ids=_id)
def test_sglang_provides_what_the_plugin_reads(path):
    missing = []
    for pattern, names in _declared(path).items():
        files = sorted(ROOT.glob(pattern))
        assert files, f"{_id(path)}: no file matches {pattern}"
        provided = frozenset().union(*map(_provided, files))
        missing += [f"{pattern}: {n}" for n in names if n not in provided]
    assert not missing, (
        f"SGLang no longer provides what {_id(path)} reads; repair it and its "
        "SGLANG_READS:\n  " + "\n  ".join(missing)
    )
