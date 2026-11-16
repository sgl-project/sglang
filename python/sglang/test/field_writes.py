"""AST scan for writes of owned fields: the guards that keep a class's fields
written only through its named methods share it."""

import ast
import warnings
from pathlib import Path
from typing import Collection, Iterator, List, Tuple

_SETATTR_FUNCS = {"setattr", "delattr", "__setattr__", "__delattr__"}
# The node types `writes` reports on.
_WRITE_NODES = (
    ast.Assign,
    ast.AugAssign,
    ast.AnnAssign,
    ast.Delete,
    ast.Call,
    ast.Attribute,
)


def _func_name(func):
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_own_field(node, own_fields):
    """`self.field` where the enclosing class declares a field of that name."""
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr in own_fields
    )


def _target_fields(target, fields, own_fields):
    """Fields a store or delete target names: `x.field` and `x.field[k]`."""
    for sub in ast.walk(target):
        if (
            isinstance(sub, ast.Attribute)
            and sub.attr in fields
            and not _is_own_field(sub, own_fields)
        ):
            yield sub.attr


def writes(
    node: ast.AST,
    fields: Collection[str],
    *,
    kwarg_funcs: Collection[str] = (),
    owned_only_fields: Collection[str] = (),
    own_fields: Collection[str] = frozenset(),
) -> Iterator[str]:
    """The fields a node writes, or could write. `kwarg_funcs` are calls whose
    keyword arguments set fields (a constructor, dataclasses.replace);
    `owned_only_fields` are mutable fields any read can change through an
    alias; `own_fields` are the enclosing class's own fields of the same
    names."""
    if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
        if isinstance(node, (ast.Assign, ast.Delete)):
            targets = node.targets
        else:
            targets = [node.target]
        for target in targets:
            yield from _target_fields(target, fields, own_fields)
    elif isinstance(node, ast.Call):
        name = _func_name(node.func)
        if name in _SETATTR_FUNCS:
            for arg in node.args[:2]:
                if isinstance(arg, ast.Constant) and arg.value in fields:
                    yield arg.value
        elif name in kwarg_funcs:
            for keyword in node.keywords:
                if keyword.arg in fields:
                    yield keyword.arg
    elif (
        isinstance(node, ast.Attribute)
        and node.attr in owned_only_fields
        and isinstance(node.ctx, ast.Load)
        and not _is_own_field(node, own_fields)
    ):
        yield node.attr


def _class_fields(node, fields):
    """The field names a class body declares as its own fields."""
    declared = set()
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            declared.add(stmt.target.id)
        elif isinstance(stmt, ast.Assign):
            declared.update(t.id for t in stmt.targets if isinstance(t, ast.Name))
    return frozenset(declared & set(fields))


def _in_scopes(scope, scopes):
    return any(scope == s or scope.startswith(s + ".") for s in scopes)


def scan(
    path: Path,
    fields: Collection[str],
    *,
    owner_scopes: Collection[str],
    kwarg_funcs: Collection[str] = (),
    owned_only_fields: Collection[str] = (),
) -> List[Tuple[str, str, int]]:
    """(scope, field, line) for each write outside `owner_scopes`, the dotted
    `Class` or `Class.method` scopes allowed to write the fields."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    found = []

    def visit(node, scope, own_fields):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = f"{scope}.{node.name}" if scope else node.name
            if _in_scopes(scope, owner_scopes):
                return
        if isinstance(node, ast.ClassDef):
            own_fields = _class_fields(node, fields)
        if isinstance(node, _WRITE_NODES):
            for field in writes(
                node,
                fields,
                kwarg_funcs=kwarg_funcs,
                owned_only_fields=owned_only_fields,
                own_fields=own_fields,
            ):
                found.append((scope, field, node.lineno))
        for child in ast.iter_child_nodes(node):
            visit(child, scope, own_fields)

    visit(tree, "", frozenset())
    return found


def scan_tree(root: Path, fields: Collection[str], **kwargs) -> List[str]:
    """`rel:line scope writes field` for every write under root."""
    return [
        f"{path.relative_to(root).as_posix()}:{line} {scope} writes {field}"
        for path in sorted(root.rglob("*.py"))
        for scope, field, line in scan(path, fields, **kwargs)
    ]
