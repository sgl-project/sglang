"""Who installs the startup record into the runtime context, derived from code.

Two guards need this answer and neither should keep its own list: matching the
spellings by hand is how the constructors' old defensive publish once read as
*not* publishing, which turned a correct module into a reported violation. A publisher is
defined by what it does -- it reaches ``RuntimeContext.set_server_args`` --
and a *constructor* publisher is an ``__init__`` that calls one.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Dict, Set, Tuple

_DEFINING_MODULES = ("runtime_context.py", "server_args.py")


def publisher_names(srt_root: pathlib.Path) -> frozenset:
    """Module-level functions that transitively install the record."""
    reaches: Set[str] = set()
    graph: Dict[str, Set[str]] = {}
    for relative in _DEFINING_MODULES:
        tree = ast.parse((srt_root / relative).read_text(encoding="utf-8-sig"))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            called: Set[str] = set()
            installs = False
            for inner in ast.walk(node):
                if not isinstance(inner, ast.Call):
                    continue
                if (
                    isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "set_server_args"
                ):
                    installs = True
                elif isinstance(inner.func, ast.Name):
                    called.add(inner.func.id)
            graph[node.name] = called
            if installs:
                reaches.add(node.name)
    growing = True
    while growing:
        growing = False
        for name, called in graph.items():
            if name not in reaches and called & reaches:
                reaches.add(name)
                growing = True
    return frozenset(reaches)


def constructor_publishers(srt_root: pathlib.Path) -> Set[Tuple[str, str, str]]:
    """``{(module, class, publisher)}`` for every ``__init__`` that publishes.

    Keyed by the owning class, not just the module: two constructors in one
    module would otherwise collapse into a single entry, and a *new* defensive
    publish added next to a listed one would leave the census unchanged --
    which is exactly the "adding one fails the pin" property it exists for.

    Reached through a local name or through a module attribute
    (``runtime_context.publish(...)``), and through a helper defined in the
    same module -- a constructor that publishes one hop away is the same
    hazard as one that publishes directly.
    """
    publishers = publisher_names(srt_root)
    found: Set[Tuple[str, str, str]] = set()
    for path in sorted(srt_root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except SyntaxError:
            raise AssertionError(f"unparsable module in the census: {path}")
        local: Dict[str, ast.AST] = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        def publishes(node, seen=None):
            """The publisher this callable reaches, if any."""
            seen = seen if seen is not None else set()
            for inner in ast.walk(node):
                if not isinstance(inner, ast.Call):
                    continue
                if isinstance(inner.func, ast.Name):
                    name = inner.func.id
                elif isinstance(inner.func, ast.Attribute):
                    name = inner.func.attr
                else:
                    continue
                if name in publishers:
                    return name
                if name in local and name not in seen:
                    seen.add(name)
                    reached = publishes(local[name], seen)
                    if reached is not None:
                        return reached
            return None

        for owner in ast.walk(tree):
            if not isinstance(owner, ast.ClassDef):
                continue
            for node in owner.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name != "__init__":
                    continue
                reached = publishes(node)
                if reached is not None:
                    found.add(
                        (
                            path.relative_to(srt_root).as_posix(),
                            owner.name,
                            reached,
                        )
                    )
    return found
