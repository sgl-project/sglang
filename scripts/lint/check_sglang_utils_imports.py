#!/usr/bin/env python3
"""
Pre-commit hook: the runtime packages (sglang.srt, sglang.kernels) do not
import sglang.utils.

sglang.utils holds helpers for the client, tests, and docs. A helper the
runtime needs belongs to the runtime: a generic one in sglang.srt.utils, one
that a single package uses in that package. Every import is checked, including
imports inside a function.

With file arguments, only those files are checked; without, every file under
the runtime packages is.
"""

import ast
import sys
from pathlib import Path

RUNTIME_DIRS = (Path("python/sglang/srt"), Path("python/sglang/kernels"))
FORBIDDEN = "sglang.utils"


def _forbidden_imports(path: Path) -> list[int]:
    source = path.read_text(encoding="utf-8")
    if FORBIDDEN not in source and "from sglang import" not in source:
        return []
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            if any(alias.name == FORBIDDEN for alias in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            if node.module == FORBIDDEN or (
                node.module == "sglang"
                and any(alias.name == "utils" for alias in node.names)
            ):
                lines.append(node.lineno)
    return lines


def _in_runtime(path: Path) -> bool:
    return any(path.is_relative_to(d) for d in RUNTIME_DIRS)


def main(argv: list[str]) -> int:
    if argv:
        paths = [Path(a) for a in argv if a.endswith(".py")]
    else:
        paths = [p for d in RUNTIME_DIRS for p in sorted(d.rglob("*.py"))]
    found = [
        f"  {path}:{line}"
        for path in paths
        if _in_runtime(path) and path.exists()
        for line in _forbidden_imports(path)
    ]
    if found:
        print("ERROR: sglang.srt and sglang.kernels must not import sglang.utils:")
        print("\n".join(found))
        print(
            "Move the helper into the runtime: sglang.srt.utils for a generic "
            "one, or the package that uses it."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
