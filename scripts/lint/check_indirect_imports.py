#!/usr/bin/env python3
"""
Pre-commit hook: reject importing a name through a module that only imports it.

`from sglang.a import f`, where `sglang/a.py` does not define `f` but itself
imports it from `sglang.b`, ties the caller to `sglang.a`: loading `f` loads all
of `sglang.a`, and `sglang.a` can no longer drop or move that import without
breaking callers it does not know about. Import `f` from `sglang.b` instead.

A module may still export a name it imports when it says so:
  * the name is in its literal `__all__`;
  * it imports the name as `f as f`;
  * it is a package `__init__.py`;
  * it binds the name inside a runtime `if` / `try`, or more than once, which
    makes it the place that picks an implementation (per platform, or a
    fallback).

With file arguments, only those files are checked as importers; with none,
every Python file under python/ and test/ is.
"""

import ast
import os
import sys
import warnings
from functools import lru_cache
from pathlib import Path

PKG_ROOT = Path("python")
SCAN_ROOTS = ("python", "test")


def _module_path(module: str):
    base = PKG_ROOT / module.replace(".", "/")
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            return path
    return None


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(PKG_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve(current: str, is_package: bool, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    base = current.split(".")
    if not is_package:
        base = base[:-1]
    if node.level > 1:
        base = base[: len(base) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _top_level(stmts, type_checking=False, conditional=False):
    """Yield (stmt, under TYPE_CHECKING, under a runtime if/try)."""
    for stmt in stmts:
        if isinstance(stmt, ast.If):
            if _is_type_checking(stmt.test):
                yield from _top_level(stmt.body, True, conditional)
            else:
                yield from _top_level(stmt.body, type_checking, True)
            yield from _top_level(stmt.orelse, type_checking, True)
        elif isinstance(stmt, ast.Try):
            for block in (stmt.body, stmt.orelse, stmt.finalbody):
                yield from _top_level(block, type_checking, True)
            for handler in stmt.handlers:
                yield from _top_level(handler.body, type_checking, True)
        elif isinstance(stmt, ast.With):
            yield from _top_level(stmt.body, type_checking, conditional)
        else:
            yield stmt, type_checking, conditional


@lru_cache(maxsize=None)
def _module_info(module: str):
    """What `module` defines, exports, and imports at its top level, or None."""
    path = _module_path(module)
    if path is None:
        return None
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return None
    is_package = path.name == "__init__.py"
    defined, exported, imported, selectors = set(), set(), {}, set()
    bindings = {}
    for stmt, type_checking, conditional in _top_level(tree.body):
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(stmt.name)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for target in targets:
                for node in ast.walk(target):
                    if isinstance(node, ast.Name):
                        defined.add(node.id)
                if isinstance(target, ast.Name) and target.id == "__all__":
                    try:
                        exported |= set(ast.literal_eval(stmt.value))
                    except (ValueError, TypeError, SyntaxError):
                        pass
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)) and not type_checking:
            source = (
                _resolve(module, is_package, stmt)
                if isinstance(stmt, ast.ImportFrom)
                else None
            )
            for alias in stmt.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name.split(".")[0]
                bindings[local] = bindings.get(local, 0) + 1
                if conditional:
                    selectors.add(local)
                if alias.asname == alias.name:
                    exported.add(local)
                imported[local] = (
                    (source, alias.name) if source is not None else (alias.name, None)
                )
    selectors |= {name for name, count in bindings.items() if count > 1}
    return is_package, defined, exported, imported, selectors


def _declares(module: str, name: str) -> bool:
    info = _module_info(module)
    if info is None:
        return True
    is_package, defined, exported, imported, selectors = info
    return (
        is_package
        or name in defined
        or name in exported
        or name in selectors
        or name not in imported
    )


def _source_of(module: str, name: str) -> str:
    """The nearest module after `module` that declares `name`."""
    for _ in range(10):
        source, original = _module_info(module)[3][name]
        if original is None:
            return source
        if _module_path(f"{source}.{original}") is not None:
            return f"{source}.{original}"
        if not source.startswith("sglang") or _declares(source, original):
            return source
        module, name = source, original
    return module


def _import_froms(stmts):
    """Every `from ... import` statement, without walking into expressions."""
    for stmt in stmts:
        if isinstance(stmt, ast.ImportFrom):
            yield stmt
        for field in ("body", "orelse", "finalbody", "handlers", "cases"):
            block = getattr(stmt, field, None)
            if block:
                yield from _import_froms(block)


def check_file(path: Path):
    try:
        source_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    in_package = PKG_ROOT in path.parents
    if "sglang" not in source_text and not (in_package and "from ." in source_text):
        return []
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return []
    current = _module_name(path) if in_package else ""
    is_package = path.name == "__init__.py"
    errors = []
    for node in _import_froms(tree.body):
        if node.level and not in_package:
            continue
        module = _resolve(current, is_package, node)
        if not module.startswith("sglang") or _module_info(module) is None:
            continue
        for alias in node.names:
            name = alias.name
            if name == "*" or _module_path(f"{module}.{name}") is not None:
                continue
            if _declares(module, name):
                continue
            source = _source_of(module, name)
            if source == current:
                continue
            errors.append(
                f"{path}:{node.lineno}: `{name}` is imported from `{module}`, "
                f"which only re-imports it; import it from `{source}`"
            )
    return errors


def main(argv) -> int:
    # Old escape sequences in some sources are not this check's business.
    warnings.simplefilter("ignore", SyntaxWarning)
    if argv:
        files = [Path(f) for f in argv if f.endswith(".py")]
    else:
        files = [p for root in SCAN_ROOTS for p in sorted(Path(root).rglob("*.py"))]
    errors = [e for f in files if os.path.isfile(f) for e in check_file(f)]
    if errors:
        print("ERROR: names imported through a module that only re-imports them:")
        for error in errors:
            print(f"  {error}")
        print(
            "Import each name from the module that defines it. A module that means "
            "to export an imported name lists it in `__all__` or imports it as "
            "`name as name`."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
