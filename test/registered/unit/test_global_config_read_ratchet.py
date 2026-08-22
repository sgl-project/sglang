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
which are not namespace leaves and have no home but ``ServerArgs``, and the
``configured_*_size()`` accessors for the sizes ``get_parallel()`` shadows with
the live topology. ``_CONFIGURED_SIZE_CALL_SITES`` registers every one of the
latter with the reason the live property cannot serve it.

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

# Every call site of a ``configured_*_size()`` accessor, with the reason the
# live topology cannot answer there. The test below asserts this map is exactly
# the set of call sites, so the reasons cannot drift away from the code.
_CONFIGURED_SIZE_CALL_SITES = {
    ("srt/entrypoints/engine.py", "configured_pp_size"): (
        "the launch path decides how many scheduler processes to spawn; it runs "
        "before any of them exists, so there is no group to ask"
    ),
    ("srt/ray/engine.py", "configured_pp_size"): (
        "the Ray driver sizes the actor placement group; the actors it is about "
        "to create are the ones that will hold the process groups"
    ),
    ("srt/ray/data_parallel_controller.py", "configured_pp_size"): (
        "same placement arithmetic on the DP path -- ranks per TP group, "
        "computed in the driver before the actors start"
    ),
    ("srt/ray/data_parallel_controller.py", "configured_attn_cp_size"): (
        "the attention-CP factor of that same placement arithmetic, and the one "
        "size whose live value cannot express the configured intent when "
        "attn_cp_size > moe_dp_size aliases the groups"
    ),
    ("srt/layers/attention/dsa/dsa_indexer.py", "configured_pp_size"): (
        "gates `pp_size > 1 and not get_pp_group()...`; the short circuit is the "
        "point, since with PP off the group is never touched, which is what lets "
        "the Indexer be constructed before distributed init"
    ),
    ("srt/managers/scheduler.py", "configured_pp_size"): (
        "dispatch_event_loop picks the PP event loop; the MLX runner stub never "
        "initializes torch.distributed, so the live property asserts before the "
        "MLX loop can start -- the configured leaf answers the same value "
        "wherever the live groups exist"
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
    ("srt/managers/scheduler.py", "configured_tp_size"): (
        "configure_scheduler_process runs before the scheduler's own process "
        "groups exist -- configuring the process is what it is for -- so there "
        "is nothing live to ask yet"
    ),
    ("srt/managers/scheduler.py", "configured_moe_dp_size"): (
        "same pre-distributed-init arithmetic in configure_scheduler_process"
    ),
    ("srt/managers/scheduler.py", "configured_attn_cp_size"): (
        "same pre-distributed-init arithmetic in configure_scheduler_process"
    ),
    ("srt/utils/cuda_vmm_transport_utils.py", "configured_tp_size"): (
        "the consumer count is configured fan-out arithmetic (tp_size // "
        "dp_size), which is what the record answered before"
    ),
    ("srt/disaggregation/encoder/runtime.py", "configured_tp_size"): (
        "the encode server's launch entry sizes its workers before it has "
        "spawned any of them"
    ),
    ("srt/utils/common.py", "configured_tp_size"): (
        "the require_*_tp_gather predicates compared the configured tp_size "
        "when they read the record; the live property answers a different "
        "question wherever the groups alias, so the configured accessor is the "
        "mechanical substitution and the live one would be a semantic change"
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
        try:
            tree = ast.parse(path.read_text())
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


class TestConfiguredSizeCallSites(CustomTestCase):
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

    def test_the_call_sites_match_the_documented_set(self):
        found = set()
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            rel = path.relative_to(_PACKAGE_ROOT).as_posix()
            if rel.startswith(_SLOT_OWNERS):
                continue
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
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
        documented = set(_CONFIGURED_SIZE_CALL_SITES)
        self.assertEqual(
            documented,
            found,
            "configured-size call sites drifted from their documented reasons.\n"
            f"  undocumented: {sorted(found - documented)}\n"
            f"  stale entries: {sorted(documented - found)}",
        )


class TestNoRenamedAccessorImports(CustomTestCase):
    """The scanners above match ``get_server_args`` and ``configured_*_size``
    by their literal names, so an ``import ... as`` rename would walk a read
    straight past both the zero baseline and the call-site registry. Renaming
    these accessors buys nothing (the names are already short and unambiguous),
    so it is banned outright — which is exactly what makes literal-name
    matching sound."""

    def test_the_scanned_accessors_are_never_import_renamed(self):
        offenders = []
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            rel = path.relative_to(_PACKAGE_ROOT).as_posix()
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
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
                            f"{rel}:{node.lineno}: {imported.name} as "
                            f"{imported.asname}"
                        )
        self.assertFalse(
            offenders,
            "get_server_args / configured_*_size imported under another name; "
            "the read ratchet and the configured-size registry match these "
            "accessors by their literal names, so a rename silently escapes "
            "both:\n" + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
