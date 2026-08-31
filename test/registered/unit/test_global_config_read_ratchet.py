"""Ratchet guard: process-global config reads may only decrease.

``get_server_args()`` returns the published ``ServerArgs`` — one process's
startup record. Config decisions read the namespace accessors instead
(``get_exec()`` / ``get_memory()`` / …), which carry the resolved value
including post-publish overrides, and per-runner values come from the runner
that owns them.

Business code no longer reads the published record for a config value at all:
both baselines are zero, over the whole package minus the modules that own the
slot.

The reads that remain live in ``runtime_context.py`` (exempt by module): the
``@property`` / method members computed from several fields plus the HF config,
which are not namespace leaves and have no home but ``ServerArgs``.

What the scan sees: ``get_server_args().field``, an alias (``sa =
get_server_args()`` then ``sa.field`` -- function-local, module-level, or parked
on an instance attribute), a local copy of an alias (``cfg = sa``), and the
``getattr(<either>, "field")`` spelling of each. It matches the accessors by
their literal names, which is why import-renaming them is banned below. A name
computed at runtime, or indirection deeper than a local name copy, is invisible
here -- the census tool in the context repo audits that shape. A whole-object
pass (``def f(server_args)``) is not a global read and is not counted: there the
caller decided which instance to hand over.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import ast
import unittest
from pathlib import Path

import sglang
from sglang.test.test_utils import CustomTestCase

# srt is the migrated surface; the rest of the package has no reads today and is
# scanned so a new one cannot appear there unnoticed.
_PACKAGE_ROOT = Path(next(iter(sglang.__path__)))

# The modules that own the slot: runtime_context publishes it and exposes the
# named accessors for the derived members, server_args/arg_groups ARE the
# resolution pipeline.
_SLOT_OWNERS = ("srt/runtime_context.py", "srt/server_args.py", "srt/arg_groups/")

_DIRECT_BASELINE = 0
_ALIAS_BASELINE = 0


def _is_global_call(node) -> bool:
    """``get_server_args()`` however it is spelled: bare, or module-qualified
    (``ctx.get_server_args()``), which an ast.Name check alone would miss."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "get_server_args"
    return isinstance(func, ast.Attribute) and func.attr == "get_server_args"


def _collect(rel: str, tree: ast.AST):
    """The (direct, alias) field reads in one module."""
    direct, alias = [], []

    def _getattr_name(node):
        """``getattr(<record>, "field")`` names a field just as ``.field`` does;
        matching only ast.Attribute would let a dynamic read walk past."""
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            return None
        return node.args[1].value

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _is_global_call(node.value):
            direct.append(f"{rel}:{node.lineno}: get_server_args().{node.attr}")

        name = _getattr_name(node)
        if name is not None and _is_global_call(node.args[0]):
            direct.append(f"{rel}:{node.lineno}: getattr(get_server_args(), {name!r})")

        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        bound = {}
        for inner in ast.walk(node):
            # ``sa = get_server_args()`` and its annotated form
            # ``sa: ServerArgs = get_server_args()``.
            if isinstance(inner, (ast.Assign, ast.AnnAssign)) and _is_global_call(
                getattr(inner, "value", None)
            ):
                targets = (
                    inner.targets if isinstance(inner, ast.Assign) else [inner.target]
                )
                for target in targets:
                    if not isinstance(target, ast.Name):
                        continue
                    # A parameter reassigned from the global is the
                    # optional-injection shape (``f(server_args=None)`` then
                    # ``server_args = get_server_args()``): the reads that
                    # follow are global reads wearing a parameter's name, so
                    # they count from the bind on.
                    bound.setdefault(target.id, inner.lineno)
        if not bound:
            continue
        # A copy of an alias reaches the same record (``cfg = sa`` after
        # ``sa = get_server_args()``), so follow Name-to-Name assignments to a
        # fixpoint. Deeper indirection (through containers, attributes of
        # other objects, cross-scope copies) stays census-tool territory.
        changed = True
        while changed:
            changed = False
            for inner in ast.walk(node):
                if not isinstance(inner, (ast.Assign, ast.AnnAssign)):
                    continue
                value = getattr(inner, "value", None)
                if not (isinstance(value, ast.Name) and value.id in bound):
                    continue
                targets = (
                    inner.targets if isinstance(inner, ast.Assign) else [inner.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name) and target.id not in bound:
                        bound[target.id] = inner.lineno
                        changed = True
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Attribute)
                and isinstance(inner.value, ast.Name)
                and inner.value.id in bound
                and inner.lineno >= bound[inner.value.id]
            ):
                alias.append(
                    f"{rel}:{inner.lineno}: {inner.value.id}.{inner.attr} "
                    f"(bound from get_server_args() at line {bound[inner.value.id]})"
                )
            name = _getattr_name(inner)
            if (
                name is not None
                and isinstance(inner.args[0], ast.Name)
                and inner.args[0].id in bound
                and inner.lineno >= bound[inner.args[0].id]
            ):
                alias.append(
                    f"{rel}:{inner.lineno}: getattr({inner.args[0].id}, {name!r}) "
                    f"(bound from get_server_args() at line {bound[inner.args[0].id]})"
                )
    # A module-level alias is visible to every function in the file, so it needs
    # its own pass -- the per-function scan above deliberately does not reach
    # across scopes.
    module_bound = {}
    module_stack = list(tree.body)
    while module_stack:
        stmt = module_stack.pop()
        # A module-level bind can sit inside an `if` / `try` / `with`, so the
        # walk descends into those bodies -- but not into a nested function or
        # class, whose binds are that scope's own.
        if isinstance(
            stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        module_stack.extend(ast.iter_child_nodes(stmt))
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and _is_global_call(
            getattr(stmt, "value", None)
        ):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    module_bound.setdefault(target.id, stmt.lineno)
    if module_bound:
        # Shadowing is per lexical scope: a function with its own `sa` hides the
        # module alias *inside that function only*. Aggregating the names
        # file-wide would suppress every read in the module, including the
        # top-level ones and the ones in functions that do resolve to the alias.
        parents = {}
        scope_binds = {}
        stack = [tree]
        while stack:
            node = stack.pop()
            enclosing = parents.get(id(node))
            for child in ast.iter_child_nodes(node):
                parents[id(child)] = (
                    node
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    else enclosing
                )
                stack.append(child)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names = {
                    a.arg for a in list(node.args.args) + list(node.args.kwonlyargs)
                }
                # Only this scope's own stores: a nested function's local `sa`
                # shadows the alias inside *that* function, not in its parent.
                pending = list(node.body)
                while pending:
                    inner = pending.pop()
                    if isinstance(
                        inner,
                        (
                            ast.FunctionDef,
                            ast.AsyncFunctionDef,
                            ast.Lambda,
                            ast.ClassDef,
                        ),
                    ):
                        continue
                    if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Store):
                        names.add(inner.id)
                    pending.extend(ast.iter_child_nodes(inner))
                scope_binds[id(node)] = names

        def _shadowed(node, name):
            scope = parents.get(id(node))
            while scope is not None:
                if name in scope_binds.get(id(scope), ()):
                    return True
                scope = parents.get(id(scope))
            return False

        for node in ast.walk(tree):
            base = attr = None
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in module_bound
            ):
                base, attr = node.value.id, node.attr
                shown = f"{base}.{attr}"
            else:
                attr_name = _getattr_name(node)
                if (
                    attr_name is not None
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id in module_bound
                ):
                    base, attr = node.args[0].id, attr_name
                    shown = f"getattr({base}, {attr!r})"
            if base and not _shadowed(node, base):
                alias.append(
                    f"{rel}:{node.lineno}: {shown} "
                    f"(module-level bind from get_server_args() at line "
                    f"{module_bound[base]})"
                )

    # An alias parked on an instance attribute (``self._sa = get_server_args()``
    # in one method, ``self._sa.field`` in another) reaches the same slot and
    # crosses function scopes, so it is collected per class rather than per
    # function.
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        attr_bound = {}
        for inner in ast.walk(node):
            if isinstance(inner, (ast.Assign, ast.AnnAssign)) and _is_global_call(
                getattr(inner, "value", None)
            ):
                targets = (
                    inner.targets if isinstance(inner, ast.Assign) else [inner.target]
                )
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id in ("self", "cls")
                    ):
                        attr_bound.setdefault(
                            (target.value.id, target.attr), inner.lineno
                        )
        if not attr_bound:
            continue

        def _bound_attr(value):
            """``self._sa`` when that attribute was bound from the global."""
            if (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and (value.value.id, value.attr) in attr_bound
            ):
                return (value.value.id, value.attr)
            return None

        for inner in ast.walk(node):
            key = shown = None
            if isinstance(inner, ast.Attribute):
                key = _bound_attr(inner.value)
                if key is not None:
                    shown = f"{key[0]}.{key[1]}.{inner.attr}"
            else:
                name = _getattr_name(inner)
                if name is not None:
                    key = _bound_attr(inner.args[0])
                    if key is not None:
                        shown = f"getattr({key[0]}.{key[1]}, {name!r})"
            if shown is not None:
                alias.append(
                    f"{rel}:{inner.lineno}: {shown} "
                    f"(attribute bind from get_server_args() at line "
                    f"{attr_bound[key]})"
                )
    return direct, alias


def _field_reads():
    direct, alias = [], []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(_PACKAGE_ROOT).as_posix()
        if rel.startswith(_SLOT_OWNERS):
            continue
        source = path.read_text()
        if "get_server_args" not in source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        module_direct, module_alias = _collect(rel, tree)
        direct += module_direct
        alias += module_alias
    return direct, alias


class TestGlobalConfigReadRatchet(CustomTestCase):
    def _check(self, kind, reads, baseline):
        if len(reads) > baseline:
            self.fail(
                f"{kind} process-global config field reads grew: {len(reads)} > "
                f"baseline {baseline}. Read the namespace accessor for the "
                "field's namespace, or the owning runner for a per-runner "
                "field:\n" + "\n".join(reads)
            )
        if len(reads) < baseline:
            self.fail(
                f"{kind} process-global config field reads shrank: {len(reads)} < "
                f"baseline {baseline}. Lower the baseline in this file to lock "
                "in the progress."
            )

    def test_global_field_reads_match_the_baseline(self):
        direct, alias = _field_reads()
        self._check("direct", direct, _DIRECT_BASELINE)
        self._check("alias-form", alias, _ALIAS_BASELINE)


class TestNoRenamedAccessorImports(CustomTestCase):
    """The baseline scanner matches ``get_server_args`` by its literal name, so
    an ``import ... as`` rename would walk a read straight past the zero
    baseline. Renaming the accessor buys nothing (the name is already short and
    unambiguous), so it is banned outright — which is exactly what makes
    literal-name matching sound. (The configured-size registry resolves
    ``get_parallel`` aliases itself, so it needs no such ban.)"""

    def test_the_scanned_accessors_are_never_import_renamed(self):
        offenders = []
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            rel = path.relative_to(_PACKAGE_ROOT).as_posix()
            source = path.read_text()
            if "get_server_args" not in source:
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.ImportFrom, ast.Import)):
                    continue
                for imported in node.names:
                    if imported.asname is None or imported.asname == imported.name:
                        continue
                    base = imported.name.rsplit(".", 1)[-1]
                    if base == "get_server_args":
                        offenders.append(
                            f"{rel}:{node.lineno}: {imported.name} as "
                            f"{imported.asname}"
                        )
        self.assertFalse(
            offenders,
            "get_server_args imported under another name; the read ratchet "
            "matches it by its literal name, so a rename silently escapes the "
            "baseline:\n" + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
