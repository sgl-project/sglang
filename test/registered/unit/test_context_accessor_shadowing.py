"""No local may shadow a ``runtime_context`` accessor it also calls.

A mechanical sweep that rewrites ``self.server_args.mamba_cache_chunk_size``
into ``mamba_cache_chunk_size()`` turns

    mamba_cache_chunk_size = self.server_args.mamba_cache_chunk_size

into ``mamba_cache_chunk_size = mamba_cache_chunk_size()``, which is a
self-referential local: the name is local for the whole function, so the call
raises ``UnboundLocalError`` the first time that line runs. Five of these
shipped in one sweep and only one had unit coverage — a mamba model on the
radix-cache-v2 path found it at request time.

This scans for the shape directly: a function-scope assignment whose target
name is an imported accessor.
"""

import ast
import unittest
from pathlib import Path

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=23, suite="base-a-test-cpu")

_PACKAGE_ROOT = Path(next(iter(sglang.__path__)))
_CONTEXT_MODULE = "sglang.srt.runtime_context"


def _module_level_accessor_imports(tree: ast.AST) -> set[str]:
    """Accessors imported at module scope — visible in every function.

    A *function-local* import is visible only inside its own scope, so it is
    collected per function in the scan below: charging it file-wide would flag
    an unrelated sibling function that binds the same name, where no shadowing
    can occur.
    """
    names: set[str] = set()
    stack = list(tree.body)
    while stack:
        stmt = stack.pop()
        if isinstance(
            stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
        ):
            continue
        if isinstance(stmt, ast.ImportFrom) and stmt.module == _CONTEXT_MODULE:
            for alias in stmt.names:
                names.add(alias.asname or alias.name)
        stack.extend(ast.iter_child_nodes(stmt))
    return names


def _bound_names(target):
    """Every name a binding target introduces, unpacking included.

    ``a, (b, c) = ...`` and ``for x, y in ...`` bind through Tuple/List/Starred
    nodes, so a check that only accepts a bare ``ast.Name`` misses them.
    """
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, ast.Starred):
        yield from _bound_names(target.value)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _bound_names(element)


def _own_scope_statements(node) -> tuple:
    """This function's OWN scope: its statements, plus the (name, lineno) of
    each nested ``def``/``class`` — the definition's *name* is a binding in
    this scope (an earlier accessor call raises UnboundLocalError just like an
    assignment), while its *body* is the nested scope's own and descending into
    it would misattribute bindings."""
    own_scope = []
    nested_def_bindings = []
    pending = list(node.body)
    while pending:
        stmt = pending.pop()
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nested_def_bindings.append((stmt.name, stmt.lineno))
            continue
        if isinstance(stmt, ast.Lambda):
            continue
        own_scope.append(stmt)
        pending.extend(ast.iter_child_nodes(stmt))
    return own_scope, nested_def_bindings


def _child_functions(body) -> list:
    """Function defs directly beneath this scope — descending through plain
    statements and class bodies (a method closes over the enclosing function's
    names, not the class's), but never into another function."""
    funcs = []
    pending = list(body)
    while pending:
        stmt = pending.pop()
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(stmt)
            continue
        if isinstance(stmt, ast.Lambda):
            continue
        pending.extend(ast.iter_child_nodes(stmt))
    return funcs


def _shadowing_assignments(tree: ast.AST, module_accessors: set[str]):
    """Function-local bindings whose name shadows an accessor visible in that
    scope -- every statement form that binds a local, not just ``=``.

    Python decides a name is local from *any* binding in the function, so a
    loop variable, a ``with ... as``, a walrus, a comprehension target, or an
    ``except ... as`` all shadow the accessor for the whole function body,
    exactly like an assignment does.

    Visibility follows lexical scope: module-level imports reach every
    function; a function-local import reaches its own scope and nested
    functions (closure), but NOT an unrelated sibling — charging it file-wide
    would flag bindings where no shadowing occurs. A function-scope *re-import*
    of the accessor is itself fine: it binds the name to the same callable, so
    calls after it behave identically (and the module is full of deliberate
    local imports).
    """
    stack = [(fn, module_accessors) for fn in _child_functions(tree.body)]
    while stack:
        node, inherited = stack.pop()
        own_scope, nested_def_bindings = _own_scope_statements(node)
        local_imports = {
            alias.asname or alias.name
            for stmt in own_scope
            if isinstance(stmt, ast.ImportFrom) and stmt.module == _CONTEXT_MODULE
            for alias in stmt.names
        }
        visible = inherited | local_imports
        # ``def get_exec(): ...`` nested in the function binds the name in
        # THIS scope, exactly like an assignment would.
        for name, lineno in nested_def_bindings:
            if name in visible:
                yield node.name, name, lineno
        for inner in own_scope:
            targets = []
            if isinstance(inner, ast.Assign):
                targets = inner.targets
            elif isinstance(inner, (ast.AnnAssign, ast.AugAssign)):
                targets = [inner.target]
            elif isinstance(inner, (ast.For, ast.AsyncFor, ast.comprehension)):
                targets = [inner.target]
            elif isinstance(inner, ast.NamedExpr):
                targets = [inner.target]
            elif isinstance(inner, (ast.With, ast.AsyncWith)):
                targets = [i.optional_vars for i in inner.items if i.optional_vars]
            elif isinstance(inner, ast.ExceptHandler) and inner.name:
                targets = [ast.Name(id=inner.name, ctx=ast.Store())]
            for target in targets:
                for name in _bound_names(target):
                    if name in visible:
                        yield node.name, name, getattr(inner, "lineno", node.lineno)
        for nested in _child_functions(node.body):
            stack.append((nested, visible))


class TestNoAccessorShadowing(CustomTestCase):
    def test_no_local_shadows_a_context_accessor(self):
        offenders = []
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            rel = path.relative_to(_PACKAGE_ROOT).as_posix()
            if rel.startswith("srt/runtime_context.py"):
                continue
            source = path.read_text()
            # A file that never names the module cannot import an accessor
            # from it, at module scope or inside any function.
            if _CONTEXT_MODULE not in source:
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            module_accessors = _module_level_accessor_imports(tree)
            for func, name, lineno in _shadowing_assignments(tree, module_accessors):
                offenders.append(f"{rel}:{lineno}: {func}() binds {name!r}")
        self.assertFalse(
            offenders,
            "locals shadow a runtime_context accessor imported in the same "
            "module; the name is local for the whole function, so any call to "
            "the accessor there raises UnboundLocalError:\n" + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
