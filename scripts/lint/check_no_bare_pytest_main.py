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


def is_pytest_main_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "main"
        and isinstance(func.value, ast.Name)
        and func.value.id == "pytest"
    )


def propagates_exit_code(node: ast.Call, parents: dict[int, ast.AST]) -> bool:
    parent = parents.get(id(node))
    if not isinstance(parent, ast.Call) or node not in parent.args:
        return False
    func = parent.func
    if (
        isinstance(func, ast.Attribute)
        and func.attr == "exit"
        and isinstance(func.value, ast.Name)
        and func.value.id == "sys"
    ):
        return True
    grandparent = parents.get(id(parent))
    return (
        isinstance(func, ast.Name)
        and func.id == "SystemExit"
        and isinstance(grandparent, ast.Raise)
        and grandparent.exc is parent
    )


def runtime_nodes(node: ast.AST):
    yield node
    if isinstance(
        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    ):
        return
    for child in ast.iter_child_nodes(node):
        yield from runtime_nodes(child)


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
            nodes = list(runtime_nodes(statement))
            parents = {
                id(child): parent
                for parent in nodes
                for child in ast.iter_child_nodes(parent)
            }
            for candidate in nodes:
                if is_pytest_main_call(candidate) and not propagates_exit_code(
                    candidate, parents
                ):
                    return candidate.lineno
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
        "code with sys.exit(...) or raise SystemExit(...):"
    )
    for offender in offenders:
        print(f"  {offender}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
