"""Check that every `sglang` import in the pinned sgl-amd-mono resolves in this tree.

sgl-amd-mono (installed through the `monokernels_hip` extra in
python/pyproject_other.toml) imports helpers from sglang, often inside the
function that uses them. A sglang PR that renames or removes such a helper
passes sglang's CI, and the monokernel server then fails at runtime, e.g.
`ImportError: cannot import name 'qwen_moe_topk_output'` during decode graph
capture.

This check reads the source only (no torch, no GPU): it parses every module of
sgl-amd-mono, collects each `from sglang... import name` and `import sglang...`
at any depth, and looks the module and name up in python/sglang.

Imports inside a `try` that catches ImportError are reported as warnings: mono
falls back to SGLang's own path there, so the server runs but loses the
monokernel. Imports in such an `except` body are the back-compat path for an
older SGLang and are skipped. Unguarded imports that do not resolve fail the
check.

Usage:
    python scripts/ci/amd/check_sgl_amd_mono_imports.py              # clone the pin
    python scripts/ci/amd/check_sgl_amd_mono_imports.py --mono-root ../sgl-amd-mono
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = REPO_ROOT / "python" / "pyproject_other.toml"
SGLANG_PKG = REPO_ROOT / "python" / "sglang"
PIN_RE = re.compile(r"sgl-amd-mono\s*@\s*git\+(?P<url>\S+?\.git)@(?P<rev>[0-9a-fA-F]+)")


@dataclass
class Ref:
    file: Path
    line: int
    module: str
    name: str | None  # None: `import module`
    guarded: bool


# --- sgl-amd-mono side -------------------------------------------------------


def read_pin(pyproject: Path) -> tuple[str, str]:
    m = PIN_RE.search(pyproject.read_text(encoding="utf-8"))
    if not m:
        sys.exit(f"no sgl-amd-mono git pin found in {pyproject}")
    return m.group("url"), m.group("rev")


def clone_pin(url: str, rev: str, dest: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(dest)], check=True)
    subprocess.run(
        ["git", "-C", str(dest), "fetch", "-q", "--depth", "1", url, rev], check=True
    )
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", "FETCH_HEAD"], check=True)
    return dest


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(
        isinstance(n, ast.Name)
        and n.id in ("ImportError", "ModuleNotFoundError", "Exception")
        for n in names
    )


class _Collector(ast.NodeVisitor):
    def __init__(self, path: Path):
        self.path = path
        self.refs: list[Ref] = []
        self._guard = 0

    def visit_Try(self, node: ast.Try) -> None:
        guarded = any(_catches_import_error(h) for h in node.handlers)
        self._guard += guarded
        for stmt in node.body:
            self.visit(stmt)
        self._guard -= guarded
        for h in node.handlers:
            # `except ImportError:` bodies import the older SGLang's names on
            # purpose (back-compat), so they are not expected to resolve here.
            if not (guarded and _catches_import_error(h)):
                for stmt in h.body:
                    self.visit(stmt)
        for part in (node.orelse, node.finalbody):
            for stmt in part:
                self.visit(stmt)

    visit_TryStar = visit_Try

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        if node.level == 0 and (mod == "sglang" or mod.startswith("sglang.")):
            for alias in node.names:
                self.refs.append(
                    Ref(self.path, node.lineno, mod, alias.name, self._guard > 0)
                )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "sglang" or alias.name.startswith("sglang."):
                self.refs.append(
                    Ref(self.path, node.lineno, alias.name, None, self._guard > 0)
                )


def collect_refs(mono_root: Path) -> list[Ref]:
    pkg = mono_root / "sgl_amd_mono"
    if not pkg.is_dir():
        sys.exit(f"{pkg} not found; is --mono-root an sgl-amd-mono checkout?")
    refs: list[Ref] = []
    for path in sorted(pkg.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as e:
            sys.exit(f"cannot parse {path}: {e}")
        c = _Collector(path.relative_to(mono_root))
        c.visit(tree)
        refs.extend(c.refs)
    return refs


# --- sglang side -------------------------------------------------------------


def module_file(module: str) -> Path | None:
    base = SGLANG_PKG.parent.joinpath(*module.split("."))
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.is_file():
            return cand
    # namespace package (a directory without __init__.py)
    return base if base.is_dir() else None


_defined_cache: dict[Path, tuple[set[str], bool]] = {}


def _bind_targets(target: ast.expr, out: set[str]) -> None:
    if isinstance(target, ast.Name):
        out.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for t in target.elts:
            _bind_targets(t, out)


def defined_names(path: Path) -> tuple[set[str], bool]:
    """Top-level names of a module, and whether it may define more dynamically."""
    if path in _defined_cache:
        return _defined_cache[path]
    names: set[str] = set()
    dynamic = False
    if path.is_dir():
        _defined_cache[path] = (names, dynamic)
        return names, dynamic
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def walk(stmts: list[ast.stmt]) -> None:
        nonlocal dynamic
        for s in stmts:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(s.name)
                if s.name == "__getattr__":
                    dynamic = True
            elif isinstance(s, ast.Assign):
                for t in s.targets:
                    _bind_targets(t, names)
            elif isinstance(s, (ast.AnnAssign, ast.AugAssign)):
                _bind_targets(s.target, names)
            elif isinstance(s, (ast.Import, ast.ImportFrom)):
                for a in s.names:
                    if a.name == "*":
                        dynamic = True
                    else:
                        names.add(a.asname or a.name.split(".")[0])
            elif isinstance(s, ast.If):
                walk(s.body)
                walk(s.orelse)
            elif isinstance(s, (ast.Try, getattr(ast, "TryStar", ast.Try))):
                walk(s.body)
                for h in s.handlers:
                    walk(h.body)
                walk(s.orelse)
                walk(s.finalbody)
            elif isinstance(s, (ast.With, ast.AsyncWith, ast.For, ast.While)):
                walk(s.body)

    walk(tree.body)
    _defined_cache[path] = (names, dynamic)
    return names, dynamic


def resolve(ref: Ref) -> str | None:
    """None if the import resolves, else the reason it does not."""
    path = module_file(ref.module)
    if path is None:
        return f"module `{ref.module}` does not exist"
    if ref.name is None or ref.name == "*":
        return None
    if module_file(f"{ref.module}.{ref.name}") is not None:
        return None  # a submodule
    names, dynamic = defined_names(path)
    if ref.name in names or dynamic:
        return None
    return f"`{ref.module}` has no `{ref.name}`"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--mono-root", type=Path, help="sgl-amd-mono checkout (default: clone the pin)"
    )
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        if args.mono_root:
            mono_root = args.mono_root
            print(f"sgl-amd-mono: {mono_root}")
        else:
            url, rev = read_pin(PYPROJECT)
            print(f"sgl-amd-mono pin: {url}@{rev}")
            mono_root = clone_pin(url, rev, Path(tmp) / "sgl-amd-mono")

        refs = collect_refs(mono_root)
        errors = warnings = 0
        for ref in refs:
            reason = resolve(ref)
            if reason is None:
                continue
            what = (
                f"from {ref.module} import {ref.name}"
                if ref.name
                else f"import {ref.module}"
            )
            msg = f"{ref.file.as_posix()}:{ref.line}: `{what}`: {reason}"
            if ref.guarded:
                warnings += 1
                print(f"::warning::{msg} (guarded: mono falls back to SGLang's path)")
            else:
                errors += 1
                print(f"::error::{msg}")

    print(
        f"checked {len(refs)} sglang imports in sgl-amd-mono: {errors} broken, {warnings} guarded"
    )
    if errors:
        print(
            "A change here removed or renamed something sgl-amd-mono uses. Keep the old "
            "name (or an alias), or fix sgl-amd-mono first and bump its pin in "
            "python/pyproject_other.toml in this PR."
        )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
