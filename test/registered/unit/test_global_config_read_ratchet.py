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
Separately, ``_CONFIGURED_SIZE_CALL_SITES`` registers every business read of
``get_parallel().config.<size>`` — the config tier of a size whose bare name is
the live topology — with the reason the live property cannot serve it.

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

# Every configured read of a live-shadowed size (``get_parallel().config.pp_size``
# and its four siblings), with the reason the live topology cannot answer there.
# The test below asserts this map is exactly the set of such reads, so the
# reasons cannot drift away from the code.
_CONFIGURED_SIZE_CALL_SITES = {
    ("srt/layers/cp/base.py", "attn_cp_size"): (
        "the lazy strategy bind in a worker: the CP group is what the strategy "
        "is being built for, and the configured width is what describes it"
    ),
    ("benchmark/one_batch.py", "pp_size"): (
        "CPU affinity for this rank, computed right after the work function "
        "publishes and before dist init, so the groups do not exist yet"
    ),
    ("benchmark/one_batch.py", "tp_size"): (
        "the same affinity computation: the layout is the configured one, and "
        "the live group is not up at this point in the work function"
    ),
    ("srt/entrypoints/engine.py", "pp_size"): (
        "the launch path decides how many scheduler processes to spawn; it runs "
        "before any of them exists, so there is no group to ask"
    ),
    ("srt/entrypoints/engine.py", "attn_cp_size"): (
        "the launcher's per-TP-rank layout, computed while deciding what to "
        "spawn -- the groups it is laying out do not exist yet"
    ),
    ("srt/entrypoints/engine.py", "moe_dp_size"): (
        "the MoE factor of that same pre-spawn layout"
    ),
    ("srt/ray/engine.py", "pp_size"): (
        "the Ray driver sizes the actor placement group; the actors it is about "
        "to create are the ones that will hold the process groups"
    ),
    ("srt/ray/engine.py", "tp_size"): (
        "the same placement arithmetic as the stage count: the driver sizes "
        "the actors that will hold the process groups"
    ),
    ("srt/ray/data_parallel_controller.py", "tp_size"): (
        "the same arithmetic on the DP path, also in the driver"
    ),
    ("srt/ray/data_parallel_controller.py", "pp_size"): (
        "same placement arithmetic on the DP path -- ranks per TP group, "
        "computed in the driver before the actors start"
    ),
    ("srt/ray/data_parallel_controller.py", "attn_cp_size"): (
        "the attention-CP factor of that same placement arithmetic, and the one "
        "size whose live value cannot express the configured intent when "
        "attn_cp_size > moe_dp_size aliases the groups"
    ),
    ("srt/layers/attention/dsa/dsa_indexer.py", "pp_size"): (
        "gates `pp_size > 1 and not get_pp_group()...`; the short circuit is the "
        "point, since with PP off the group is never touched, which is what lets "
        "the Indexer be constructed before distributed init"
    ),
    ("srt/managers/scheduler.py", "pp_size"): (
        "dispatch_event_loop picks the PP event loop; the MLX runner stub never "
        "initializes torch.distributed, so the live property asserts before the "
        "MLX loop can start -- the configured leaf answers the same value "
        "wherever the live groups exist"
    ),
    ("srt/mem_cache/kv_cache_configurator.py", "pp_size"): (
        "decides whether the token capacity needs a cross-PP all-reduce at all; "
        "asking the configured size keeps that decision independent of whether a "
        "PP group is installed in this process"
    ),
    ("srt/layers/dp_attention.py", "attn_cp_size"): (
        "compared against the configured moe_dp_size below"
    ),
    ("srt/layers/dp_attention.py", "moe_dp_size"): (
        "the configuration this predicate detects (attn_cp_size > moe_dp_size) is "
        "the one where initialize_model_parallel aliases _MOE_DP to _ATTN_CP, so "
        "the live sizes are equal there and a live comparison is always false"
    ),
    ("srt/managers/scheduler.py", "tp_size"): (
        "configure_scheduler_process runs before the scheduler's own process "
        "groups exist -- configuring the process is what it is for -- so there "
        "is nothing live to ask yet"
    ),
    ("srt/managers/scheduler.py", "moe_dp_size"): (
        "same pre-distributed-init arithmetic in configure_scheduler_process"
    ),
    ("srt/managers/scheduler.py", "attn_cp_size"): (
        "same pre-distributed-init arithmetic in configure_scheduler_process"
    ),
    ("srt/managers/scheduler.py", "dcp_size"): (
        "same pre-distributed-init arithmetic in configure_scheduler_process"
    ),
    ("srt/model_executor/runner/base_runner.py", "tp_size"): (
        "the same window as the stage count next to it: a draft runner shares "
        "the target's groups, so the live property would answer for the wrong "
        "runner"
    ),
    ("srt/model_executor/cpu_graph_runner.py", "tp_size"): (
        "the same window, on the CPU graph path"
    ),
    ("srt/entrypoints/v1_loads.py", "tp_size"): (
        "the accelerator count is arithmetic over the launch shape, reported "
        "from the tokenizer process, which holds no model groups"
    ),
    ("srt/disaggregation/nixl/conn.py", "tp_size"): (
        "the NIXL rank arithmetic runs on the transfer path, which the CPU-only "
        "conn tests exercise without starting torch.distributed"
    ),
    ("srt/managers/tokenizer_control_mixin.py", "tp_size"): (
        "the tokenizer divides its worker count by the launch width; it holds "
        "no model groups"
    ),
    ("srt/model_executor/runner/base_runner.py", "pp_size"): (
        "the runner's layer window is arithmetic over the configured stage "
        "count; a draft runner shares the target's groups, so the live "
        "property would answer for the wrong runner"
    ),
    ("srt/model_executor/cpu_graph_runner.py", "pp_size"): (
        "the same window, on the CPU graph path"
    ),
    (
        "srt/managers/scheduler_components/metrics_reporter.py",
        "pp_size",
    ): (
        "the reporter labels its metrics with the stage count it was launched "
        "with, which is configuration; the live group answers per process"
    ),
    ("srt/speculative/eagle_draft_cuda_graph_runner.py", "pp_size"): (
        "the draft runner's window over the target's stages: its own groups are "
        "the target's, so the configured count is the one that describes it"
    ),
    (
        "srt/speculative/eagle_draft_extend_cuda_graph_runner.py",
        "pp_size",
    ): ("the same draft window, on the extend path"),
    (
        "srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py",
        "pp_size",
    ): ("the same draft window, multi-layer extend"),
    ("srt/speculative/frozen_kv_mtp_cuda_graph_runner.py", "pp_size"): (
        "the same draft window, frozen-KV MTP"
    ),
    ("srt/managers/data_parallel_controller.py", "pp_size"): (
        "the controller lays out its schedulers' ranks before spawning them, so "
        "the groups it is sizing for do not exist yet"
    ),
    ("srt/managers/data_parallel_controller.py", "attn_cp_size"): (
        "the same pre-spawn rank arithmetic"
    ),
    ("srt/managers/data_parallel_controller.py", "moe_dp_size"): (
        "the same pre-spawn rank arithmetic"
    ),
    ("srt/entrypoints/v1_loads.py", "pp_size"): (
        "the /v1/loads accelerator count is arithmetic over the launch shape, "
        "reported from the tokenizer process, which holds no model groups"
    ),
    ("srt/disaggregation/common/conn.py", "pp_size"): (
        "the bootstrap connection is built by the KV manager on the transfer "
        "path, which the CPU-only conn tests exercise without ever starting "
        "torch.distributed"
    ),
    ("srt/elastic_ep/elastic_ep.py", "tp_size"): (
        "the joiner's rank window is computed against the size the process was "
        "configured with, not the size of the group it is about to join"
    ),
    ("srt/elastic_ep/expert_backup_manager.py", "tp_size"): (
        "the backup server counts the clients it expects to report in, which "
        "is how many the launch configured -- the live group is what they are "
        "still joining"
    ),
    (
        "srt/model_executor/model_runner_components/startup_weight_load.py",
        "tp_size",
    ): (
        "the load options are assembled in ModelRunner.__init__ for a runner "
        "that may be a draft, whose groups are the target's; the configured "
        "sizes are what the record answered before"
    ),
    (
        "srt/model_executor/model_runner_components/startup_weight_load.py",
        "pp_size",
    ): ("same options object, same reason"),
    (
        "srt/model_executor/model_runner_components/startup_weight_load.py",
        "attn_cp_size",
    ): ("same options object, same reason"),
    (
        "srt/model_executor/model_runner_components/startup_weight_load.py",
        "dcp_size",
    ): ("same options object, same reason"),
    (
        "srt/model_executor/model_runner_components/spec_aux_hidden_state.py",
        "tp_size",
    ): (
        "the draft KV bytes/token estimate sizes the memory pool before the "
        "draft runner exists, so its shard count is configuration"
    ),
    ("srt/eplb/expert_location.py", "tp_size"): (
        "the elastic-EP joiner window, used to size the expert layout: the "
        "size the process was configured with, not the group it is joining"
    ),
    ("srt/utils/cuda_vmm_transport_utils.py", "tp_size"): (
        "the consumer count is configured fan-out arithmetic (tp_size // "
        "dp_size), which is what the record answered before"
    ),
    ("srt/disaggregation/encoder/runtime.py", "tp_size"): (
        "the encode server's launch entry sizes its workers before it has "
        "spawned any of them"
    ),
    ("srt/disaggregation/encoder/grpc_server.py", "tp_size"): (
        "the same worker-count arithmetic on the gRPC entry: it spawns the TP "
        "workers, so their groups do not exist yet"
    ),
    ("srt/disaggregation/encoder/server.py", "tp_size"): (
        "`MMEncoder` builds its own TP group from this size -- "
        "`initialize_model_parallel` is the call being handed it, so there is "
        "nothing live to ask"
    ),
    ("srt/disaggregation/encoder/receiver.py", "tp_size"): (
        "the receiver labels and shards by the launch width; it runs in the "
        "tokenizer process, which holds no encoder groups"
    ),
    ("srt/managers/rust_server.py", "tp_size"): (
        "the rust server decides its transport from the launch width, in the "
        "tokenizer process, which holds no model groups"
    ),
    ("compile_deep_gemm.py", "tp_size"): (
        "the warm-up request fans bootstrap rooms across the launch's ranks; it "
        "runs in the tokenizer process, which holds no model groups"
    ),
    ("srt/utils/common.py", "tp_size"): (
        "the require_*_tp_gather predicates compared the configured tp_size "
        "when they read the record; the live property answers a different "
        "question wherever the groups alias, so the configured accessor is the "
        "mechanical substitution and the live one would be a semantic change"
    ),
    ("srt/model_loader/loader.py", "moe_dp_size"): (
        "the same dict already carries the live moe_dp_size under 'dp'; this entry "
        "is the configured intent"
    ),
    ("srt/models/kimi_k25.py", "tp_size"): (
        "the IPC refcount must match the configured TP consumer count captured "
        "when the tokenizer creates MmItemMemoryPool; a live attention subgroup "
        "size could strand leases in the bounded pool"
    ),
    ("srt/models/kimi_k3.py", "tp_size"): (
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


def _live_shadowed_sizes() -> frozenset:
    """Names that are BOTH a live ``ParallelContext`` property and a ``parallel``
    config leaf.

    Derived from the two sides themselves: a size that gains a live property, or
    a live property that gains a leaf, joins the registry's subject set without a
    list here.
    """
    from sglang.srt.arg_groups.arg_utils import namespace_of
    from sglang.srt.runtime_context import ParallelContext
    from sglang.srt.server_args import ServerArgs

    live = {
        name
        for name, value in vars(ParallelContext).items()
        if isinstance(value, property)
    }
    leaves = {
        field for field, path in namespace_of(ServerArgs).items() if path == "parallel"
    }
    shadowed = frozenset(live & leaves)
    assert shadowed, "no live-shadowed size found; the derivation is broken"
    return shadowed


def _parallel_config_reads(tree, subjects):
    """Names in ``subjects`` read through the parallel bag's ``config`` hop.

    Sees ``get_parallel().config.pp_size``, the module-qualified spelling, a
    local bound to either hop (``p = get_parallel()`` / ``cfg = p.config``), and
    the ``getattr`` form of each.
    """
    fns, modules = set(), set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.endswith("runtime_context")
        ):
            fns |= {a.asname or a.name for a in node.names if a.name == "get_parallel"}
        elif isinstance(node, ast.ImportFrom) and node.module:
            # `from sglang.srt import runtime_context as rc` binds the module.
            for a in node.names:
                if f"{node.module}.{a.name}".endswith("runtime_context"):
                    modules.add(a.asname or a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.endswith("runtime_context"):
                    # Unaliased, the call site spells the whole dotted path.
                    modules.add(a.asname or a.name)

    def dotted(node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        parts.append(node.id)
        return ".".join(reversed(parts))

    def is_bag_call(node):
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        if isinstance(func, ast.Name):
            return func.id in fns
        return (
            isinstance(func, ast.Attribute)
            and func.attr == "get_parallel"
            and dotted(func.value) in modules
        )

    bag_aliases, config_aliases = set(), set()
    for _ in range(2):  # a local copy of a local is still the same object
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            if is_bag_call(value) or (
                isinstance(value, ast.Name) and value.id in bag_aliases
            ):
                bucket = bag_aliases
            elif (
                isinstance(value, ast.Attribute)
                and value.attr == "config"
                and (
                    is_bag_call(value.value)
                    or (
                        isinstance(value.value, ast.Name)
                        and value.value.id in bag_aliases
                    )
                )
            ) or (isinstance(value, ast.Name) and value.id in config_aliases):
                bucket = config_aliases
            else:
                continue
            bucket |= {t.id for t in node.targets if isinstance(t, ast.Name)}

    def is_config_hop(node):
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "config"
            and (
                is_bag_call(node.value)
                or (isinstance(node.value, ast.Name) and node.value.id in bag_aliases)
            )
        ) or (isinstance(node, ast.Name) and node.id in config_aliases)

    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in subjects:
            base, name = node.value, node.attr
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in subjects
        ):
            base, name = node.args[0], node.args[1].value
        else:
            continue
        if is_config_hop(base):
            found.add(name)
    return found


_READ_SPELLINGS = (
    "from sglang.srt.runtime_context import get_parallel\nx = get_parallel().config.tp_size",
    "from sglang.srt.runtime_context import get_parallel as gp\nx = gp().config.tp_size",
    "from sglang.srt import runtime_context as rc\nx = rc.get_parallel().config.tp_size",
    "import sglang.srt.runtime_context\nx = sglang.srt.runtime_context.get_parallel().config.tp_size",
    "from sglang.srt.runtime_context import get_parallel\np = get_parallel()\nx = p.config.tp_size",
    "from sglang.srt.runtime_context import get_parallel\nc = get_parallel().config\nx = c.tp_size",
    'from sglang.srt.runtime_context import get_parallel\nx = getattr(get_parallel().config, "tp_size")',
)


class TestParallelConfigReadSpellings(CustomTestCase):
    """``_parallel_config_reads`` resolves every spelling it claims to.

    The scan below decides what the documented set is compared against, so a
    spelling it cannot resolve does not fail anything -- it drops the read.
    """

    def test_every_documented_spelling_resolves(self):
        for source in _READ_SPELLINGS:
            with self.subTest(source=source):
                found = _parallel_config_reads(ast.parse(source), {"tp_size"})
                self.assertEqual({"tp_size"}, set(found))

    def test_the_live_property_is_not_a_config_read(self):
        source = (
            "from sglang.srt.runtime_context import get_parallel\n"
            "x = get_parallel().tp_size"
        )
        self.assertEqual(
            set(), set(_parallel_config_reads(ast.parse(source), {"tp_size"}))
        )


class TestConfiguredSizeCallSites(CustomTestCase):
    """The configured-vs-live exceptions are enumerated, with reasons.

    ``get_parallel().config.tp_size`` answers what the process was configured
    with where the bare ``get_parallel().tp_size`` answers what the process ended
    up with. Each site that needs the former is listed above with why the live
    property cannot serve it, and this case fails if the code and that list
    disagree.

    The unit is **(file, size)**, not the individual read: a second
    ``.config.pp_size`` in a file already registered for it collapses into the
    same entry, so the reason has to cover the file's use of that size rather
    than one line. A new file, or a new size in a listed file, is what this
    catches -- through any spelling of the hop.
    """

    def test_the_call_sites_match_the_documented_set(self):
        subjects = _live_shadowed_sizes()
        found = set()
        scanned = 0
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            rel = path.relative_to(_PACKAGE_ROOT).as_posix()
            if rel.startswith(_SLOT_OWNERS):
                continue
            source = path.read_text()
            # Every spelling `_parallel_config_reads` resolves -- the direct
            # call, an aliased import, a module-qualified call, a local bound to
            # either hop -- needs the name in the source, so skipping the rest is
            # free. Filtering on anything narrower silently empties the scan.
            if "get_parallel" not in source:
                continue
            scanned += 1
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            found |= {(rel, name) for name in _parallel_config_reads(tree, subjects)}
        self.assertGreater(
            scanned,
            50,
            f"the pre-filter left only {scanned} files to scan; the derivation "
            "is broken, not the tree",
        )
        documented = set(_CONFIGURED_SIZE_CALL_SITES)
        self.assertEqual(
            documented,
            found,
            "configured-size reads drifted from their documented reasons.\n"
            f"  undocumented: {sorted(found - documented)}\n"
            f"  stale entries: {sorted(documented - found)}",
        )


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
