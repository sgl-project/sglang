#!/usr/bin/env python3
"""Reject a method defined twice in one class body.

Python binds the last definition, so the earlier one never runs. That is
usually a merge artifact: two branches each added the method, and the survivor
silently drops whatever the other one did. `HybridAttnBackend.forward` sat like
that for months, and the shadowed copy still read as live code.

Only same-class collisions count. Overloads, property accessors, and
`typing.overload` stubs are legitimate repeats and are skipped.
"""

import ast
import pathlib
import sys

_ALLOWED_DECORATORS = ("overload", "setter", "getter", "deleter", "register")


def _is_intentional_repeat(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Property accessors and dispatch registrations reuse a name on purpose."""
    for decorator in node.decorator_list:
        text = ast.unparse(decorator)
        if any(marker in text for marker in _ALLOWED_DECORATORS):
            return True
    return False


def find_shadowed_defs(path: pathlib.Path) -> list[tuple[str, str, int, int]]:
    """Return (class, method, shadowed_line, winning_line) for each collision."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    shadowed = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        # Direct children only: a nested class keeps its own namespace.
        seen: dict[str, int] = {}
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _is_intentional_repeat(item):
                continue
            if item.name in seen:
                shadowed.append((node.name, item.name, seen[item.name], item.lineno))
            seen[item.name] = item.lineno
    return shadowed


def main(paths: list[str]) -> int:
    offenders = []
    for path_string in paths:
        path = pathlib.Path(path_string)
        for class_name, method, shadowed_line, winning_line in find_shadowed_defs(path):
            offenders.append(
                f"  {path}:{shadowed_line}: {class_name}.{method}() is shadowed "
                f"by the definition on line {winning_line}"
            )

    if not offenders:
        return 0

    print(
        "ERROR: a method defined twice in one class body only runs the last "
        "definition. Delete the dead one, or merge what it does into the "
        "survivor:"
    )
    for offender in offenders:
        print(offender)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
