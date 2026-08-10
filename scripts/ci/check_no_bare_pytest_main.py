#!/usr/bin/env python3

import ast
import pathlib
import re
import sys

_PYTEST_MAIN = re.compile(r"pytest\s*\.\s*main")


def is_main_guard(node: ast.expr) -> bool:
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return False
    if not isinstance(node.ops[0], ast.Eq):
        return False
    sides = [node.left, *node.comparators]
    has_name = any(
        isinstance(side, ast.Name) and side.id == "__name__" for side in sides
    )
    has_main = any(
        isinstance(side, ast.Constant) and side.value == "__main__" for side in sides
    )
    return has_name and has_main


def is_bare_pytest_main_call(node: ast.stmt) -> bool:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    func = node.value.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "main"
        and isinstance(func.value, ast.Name)
        and func.value.id == "pytest"
    )


def find_bare_pytest_main(path: pathlib.Path) -> int | None:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if "__main__" not in source or _PYTEST_MAIN.search(source) is None:
        return None
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not is_main_guard(node.test):
            continue
        for statement in node.body:
            if is_bare_pytest_main_call(statement):
                return statement.lineno
    return None


def main(paths: list[str]) -> int:
    offenders = []
    for path_string in paths:
        path = pathlib.Path(path_string)
        line = find_bare_pytest_main(path)
        if line is not None:
            offenders.append(f"{path}:{line}")

    if not offenders:
        return 0

    print(
        "ERROR: pytest.main(...) in an __main__ block must propagate its exit "
        "code with sys.exit(...):"
    )
    for offender in offenders:
        print(f"  {offender}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
