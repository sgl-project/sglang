#!/usr/bin/env python3
"""
Pre-commit hook: sglang.srt.utils is the bottom layer of srt, so its modules
import no srt domain package at module level.

Other srt packages import sglang.srt.utils. When a utils module imports a
domain package (managers, layers, multimodal, ...) at module level, importing
the small helper loads that package too, and the two can import each other in
a cycle. A helper that needs a domain package belongs to that package, as
`sglang.srt.<pkg>.utils` or a sibling module.

Imports inside a function or under `TYPE_CHECKING` do not run at import time
and are not checked.
"""

import ast
import sys
from pathlib import Path

PKG_ROOT = Path("python")
UTILS_DIR = PKG_ROOT / "sglang" / "srt" / "utils"

# This check does not follow imports transitively: add a module here only if it
# imports no srt domain package at module level itself.
ALLOWED = (
    "sglang.srt.utils",
    "sglang.srt.environ",
    "sglang.srt.runtime_context",
    "sglang.srt.platforms",
    "sglang.srt.observability.func_timer",
)


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _module_level_imports(path: Path) -> list[str]:
    """Absolute names of the modules imported when `path` is imported."""
    package = path.relative_to(PKG_ROOT).parent.parts
    modules = []
    pending = list(ast.parse(path.read_text(encoding="utf-8")).body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            pending.extend(node.orelse)
            continue
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - node.level + 1]
                modules.append(".".join(base + ((node.module,) if node.module else ())))
            else:
                modules.append(node.module)
        else:
            pending.extend(
                child
                for child in ast.iter_child_nodes(node)
                if isinstance(child, ast.stmt)
            )
    return modules


def _is_allowed(module: str) -> bool:
    return any(module == a or module.startswith(a + ".") for a in ALLOWED)


def main() -> int:
    paths = sorted(UTILS_DIR.rglob("*.py"))
    if not paths:
        print(f"ERROR: no Python files under {UTILS_DIR}; update UTILS_DIR.")
        return 1
    errors = []
    for path in paths:
        packages = {
            ".".join(module.split(".")[:3])
            for module in _module_level_imports(path)
            if module.startswith("sglang.srt.") and not _is_allowed(module)
        }
        if packages:
            errors.append(f"  {path}: {sorted(packages)}")
    if errors:
        print("ERROR: sglang.srt.utils imports an srt domain package at module level:")
        print("\n".join(errors))
        print(
            "Move the helper to the package that owns it, import inside the "
            "function, or import under TYPE_CHECKING when only annotations use it."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
