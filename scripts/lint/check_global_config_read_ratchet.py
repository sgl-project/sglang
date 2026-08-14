"""Guard: business code never reads a config field off the process-global record.

``get_server_args()`` returns the published ``ServerArgs`` -- one process's
startup record. Config decisions read the namespace accessors instead
(``get_exec()`` / ``get_memory()`` / ...); per-runner values come from the
runner that owns them. Both baselines are zero, over the whole package minus
the modules that own the slot.

The scanners match ``get_server_args`` and ``configured_*_size`` by their
literal names, which is why import-renaming them is banned below. A name
computed at runtime, or indirection deeper than a local name copy, is invisible
here -- the census tool in the context repo audits that shape.
"""

import ast
from functools import cache
from pathlib import Path

# srt is the migrated surface; the rest of the package has no reads today and is
# scanned so a new one cannot appear there unnoticed.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "python" / "sglang"

# The modules that own the slot: runtime_context publishes it and exposes the
# named accessors for the derived members, server_args/arg_groups ARE the
# resolution pipeline.
_SLOT_OWNERS = ("srt/runtime_context.py", "srt/server_args.py", "srt/arg_groups/")

# Every call site of a ``configured_*_size()`` accessor, with the reason the
# live topology cannot answer there. The checker below asserts this map is exactly
# the set of call sites, so the reasons cannot drift away from the code.
_CONFIGURED_SIZE_CALL_SITES = {
    ("srt/layers/attention/dsa/dsa_indexer.py", "configured_pp_size"): (
        "gates `pp_size > 1 and not get_pp_group()...`; the short circuit is the "
        "point, since with PP off the group is never touched, which is what lets "
        "the Indexer be constructed before distributed init"
    ),
    ("srt/mem_cache/kv_cache_configurator.py", "configured_pp_size"): (
        "decides whether the token capacity needs a cross-PP all-reduce at all; "
        "asking the configured size keeps that decision independent of whether a "
        "PP group is installed in this process"
    ),
    ("srt/layers/dp_attention.py", "configured_attn_cp_size"): (
        "compared against the configured moe_dp_size below"
    ),
    ("srt/layers/dp_attention.py", "configured_moe_dp_size"): (
        "the configuration this predicate detects (attn_cp_size > moe_dp_size) is "
        "the one where initialize_model_parallel aliases _MOE_DP to _ATTN_CP, so "
        "the live sizes are equal there and a live comparison is always false"
    ),
    ("srt/model_loader/loader.py", "configured_moe_dp_size"): (
        "the same dict already carries the live moe_dp_size under 'dp'; this entry "
        "is the configured intent"
    ),
    ("srt/models/kimi_k25.py", "configured_tp_size"): (
        "the IPC refcount must match the configured TP consumer count captured "
        "when the tokenizer creates MmItemMemoryPool; a live attention subgroup "
        "size could strand leases in the bounded pool"
    ),
    ("srt/models/kimi_k3.py", "configured_tp_size"): (
        "same as kimi_k25: the IPC refcount must agree with the recycler's waiter"
    ),
}

# A dynamic read whose name is set nowhere in the tree, so the predicate it
# feeds is inert (the ``getattr`` default decides it). Converting it would mean
# choosing what it should have named, which is the CP path's call, not this
# sweep's -- so it is listed here rather than silently counted or "fixed".
_INERT_DYNAMIC_READS = frozenset({("srt/layers/cp/base.py", "_is_dsa_model_arch")})

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


def _collect(rel: str, tree: ast.AST, inert: frozenset = frozenset()):
    """The (direct, alias) field reads in one module.

    ``inert`` names the fields listed in ``_INERT_DYNAMIC_READS`` for this file;
    they are dropped here, at the point the read is recognized, so the filter
    matches on the field name rather than on the rendered message.
    """
    direct, alias = [], []

    def counted(attr: str) -> bool:
        return attr not in inert

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
        if (
            isinstance(node, ast.Attribute)
            and _is_global_call(node.value)
            and counted(node.attr)
        ):
            direct.append(f"{rel}:{node.lineno}: get_server_args().{node.attr}")

        name = _getattr_name(node)
        if name is not None and _is_global_call(node.args[0]) and counted(name):
            direct.append(f"{rel}:{node.lineno}: getattr(get_server_args(), {name!r})")

        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = {a.arg for a in list(node.args.args) + list(node.args.kwonlyargs)}
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
                and counted(inner.attr)
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
                and counted(name)
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
            if base and not _shadowed(node, base) and counted(attr):
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
                if key is not None and counted(inner.attr):
                    shown = f"{key[0]}.{key[1]}.{inner.attr}"
            else:
                name = _getattr_name(inner)
                if name is not None:
                    key = _bound_attr(inner.args[0])
                    if key is not None and counted(name):
                        shown = f"getattr({key[0]}.{key[1]}, {name!r})"
            if shown is not None:
                alias.append(
                    f"{rel}:{inner.lineno}: {shown} "
                    f"(attribute bind from get_server_args() at line "
                    f"{attr_bound[key]})"
                )
    return direct, alias


@cache
def _parsed_modules():
    """(rel, tree) per parseable module; the three scanners below share it."""
    modules = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        modules.append((path.relative_to(_PACKAGE_ROOT).as_posix(), tree))
    return modules


def _field_reads():
    direct, alias = [], []
    for rel, tree in _parsed_modules():
        if rel.startswith(_SLOT_OWNERS):
            continue
        inert = frozenset(name for path_, name in _INERT_DYNAMIC_READS if path_ == rel)
        module_direct, module_alias = _collect(rel, tree, inert)
        direct += module_direct
        alias += module_alias
    return direct, alias


def _configured_size_call_sites():
    found = set()
    for rel, tree in _parsed_modules():
        if rel.startswith(_SLOT_OWNERS):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else (func.attr if isinstance(func, ast.Attribute) else None)
            )
            if name and name.startswith("configured_") and name.endswith("_size"):
                found.add((rel, name))
    return found


def _renamed_accessor_imports():
    offenders = []
    for rel, tree in _parsed_modules():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ImportFrom, ast.Import)):
                continue
            for imported in node.names:
                if imported.asname is None or imported.asname == imported.name:
                    continue
                base = imported.name.rsplit(".", 1)[-1]
                if base == "get_server_args" or (
                    base.startswith("configured_") and base.endswith("_size")
                ):
                    offenders.append(
                        f"{rel}:{node.lineno}: {imported.name} as {imported.asname}"
                    )
    return offenders


def _check_count(kind, reads, baseline):
    if len(reads) > baseline:
        raise AssertionError(
            f"{kind} process-global config field reads grew: {len(reads)} > "
            f"baseline {baseline}. Read the namespace accessor for the "
            "field's namespace, or the owning runner for a per-runner "
            "field:\n" + "\n".join(reads)
        )


def check_global_config_read_ratchet():
    direct, alias = _field_reads()
    _check_count("direct", direct, _DIRECT_BASELINE)
    _check_count("alias-form", alias, _ALIAS_BASELINE)


def check_configured_size_call_sites():
    """The configured-vs-live exceptions are enumerated, with reasons.

    ``configured_*_size()`` answers what the user asked for where
    ``get_parallel()`` would answer what the process ended up with. Each such
    exception is listed above with why the live property cannot serve it, and
    this case fails if the code and that list disagree.

    The unit is **(file, accessor)**, not the individual call: a second
    `configured_pp_size()` in a file already registered for it collapses into
    the same entry, so the reason has to cover the file's use of that accessor
    rather than one line. A new file, or a new accessor in a listed file, is
    what this catches -- in either call form (bare or module-qualified).
    """

    found = _configured_size_call_sites()
    documented = set(_CONFIGURED_SIZE_CALL_SITES)
    if documented != found:
        raise AssertionError(
            "configured-size call sites drifted from their documented reasons.\n"
            f"  undocumented: {sorted(found - documented)}\n"
            f"  stale entries: {sorted(documented - found)}",
        )


def check_no_renamed_accessor_imports():
    """The scanners above match ``get_server_args`` and ``configured_*_size``
    by their literal names, so an ``import ... as`` rename would walk a read
    straight past both the zero baseline and the call-site registry. Renaming
    these accessors buys nothing (the names are already short and unambiguous),
    so it is banned outright — which is exactly what makes literal-name
    matching sound."""

    offenders = _renamed_accessor_imports()
    if offenders:
        raise AssertionError(
            "get_server_args / configured_*_size imported under another name; "
            "the read ratchet and the configured-size registry match these "
            "accessors by their literal names, so a rename silently escapes "
            "both:\n" + "\n".join(offenders),
        )


if __name__ == "__main__":
    check_global_config_read_ratchet()
    check_configured_size_call_sites()
    check_no_renamed_accessor_imports()
