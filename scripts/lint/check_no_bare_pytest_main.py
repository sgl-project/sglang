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


def is_exit_call(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    """``sys.exit(...)``, or a ``SystemExit(...)`` that is actually raised."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if (
        isinstance(func, ast.Attribute)
        and func.attr == "exit"
        and isinstance(func.value, ast.Name)
        and func.value.id == "sys"
    ):
        return True
    parent = parents.get(id(node))
    return (
        isinstance(func, ast.Name)
        and func.id == "SystemExit"
        and isinstance(parent, ast.Raise)
        and parent.exc is node
    )


def assigned_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
        return [node.target.id] if isinstance(node.target, ast.Name) else []
    return []


def exited_names(nodes: list[ast.AST], parents: dict[int, ast.AST]) -> set[str]:
    """Names handed to an exit call, so the two-step form still propagates."""
    return {
        arg.id
        for node in nodes
        if is_exit_call(node, parents)
        for arg in node.args
        if isinstance(arg, ast.Name)
    }


def propagates_exit_code(
    node: ast.Call, parents: dict[int, ast.AST], exited: set[str]
) -> bool:
    parent = parents.get(id(node))
    if (
        isinstance(parent, ast.Call)
        and node in parent.args
        and is_exit_call(parent, parents)
    ):
        return True
    return any(name in exited for name in assigned_names(parent))


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
        # Whole body at once: the call and the sys.exit() that propagates it
        # are separate statements.
        nodes = [n for statement in node.body for n in runtime_nodes(statement)]
        parents = {
            id(child): parent
            for parent in nodes
            for child in ast.iter_child_nodes(parent)
        }
        exited = exited_names(nodes, parents)
        for candidate in nodes:
            if is_pytest_main_call(candidate) and not propagates_exit_code(
                candidate, parents, exited
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
