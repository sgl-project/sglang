"""Ratchet guards for runtime_context / ServerArgs adoption.

These used to live as several tiny one-concern files at ``test/registered/unit/``.
Each pin is still an exact count: a grown number means new code bypassed the
runtime_context accessors, a shrunk number means a removal forgot to lower the
baseline. The accessor-shadowing scan is here too: a local that binds the same
name as an imported ``runtime_context`` accessor raises ``UnboundLocalError``.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import ast
import re
import unittest
from pathlib import Path

import sglang
import sglang.srt
from sglang.test.test_utils import CustomTestCase

_SRT_ROOT = Path(next(iter(sglang.srt.__path__)))
_SGLANG_ROOT = Path(next(iter(sglang.__path__)))

# Baselines counted over python/sglang/srt/**/*.py, including each function's
# own def line. Ratchet: decrease-only.
_LEGACY_GLOBAL_RATCHETS = [
    # Down to the shim definition itself; every call-site now goes through
    # runtime_context.get_server_args().
    ("get_global_server_args", r"\bget_global_server_args\s*\(", 1),
    (
        "set_global_server_args_for_*",
        r"\bset_global_server_args_for_(?:scheduler|tokenizer)\s*\(",
        2,
    ),
]

_PINNED_GLOBALS = {
    "layers/moe/utils.py": frozenset(),
    "layers/dp_attention.py": frozenset(
        {
            # DP-attention topology (parallel vertical scope).
            "_ATTN_DP_RANK",
            "_ATTN_DP_SIZE",
        }
    ),
}

_BANNED_PARALLEL_CALLS = re.compile(
    r"\b(?:dcp_enabled|get_(?:"
    r"tensor_model_parallel_(?:world_size|rank)"
    r"|pipeline_model_parallel_(?:world_size|rank)"
    r"|moe_expert_parallel_(?:world_size|rank)"
    r"|moe_tensor_parallel_(?:world_size|rank)"
    r"|moe_data_parallel_(?:world_size|rank)"
    r"|attn_tensor_model_parallel_(?:world_size|rank)"
    r"|attn_context_model_parallel_(?:world_size|rank)"
    r"|dcp_(?:world_size|rank)"
    r"|dcp_group(?:_no_assert)?"
    r"|attention_dcp_(?:world_size|rank)"
    r"|attention_(?:tp|cp)_(?:group|rank|size)"
    r"))\(\)"
)

# The whole package is swept; the exemptions are the substrate itself.
_PARALLEL_SWEPT_DIRS = ("",)
_PARALLEL_EXEMPT = (
    "distributed/",  # parallel_state: defines the canonical getters
    "runtime_context.py",  # delegates DCP reads to canonical getters
    "layers/dp_attention.py",  # delegation substrate for the attn-DP dims
    "layers/dcp/comm.py",  # deprecated out-of-tree DCP compatibility shims
    # The dumper's megatron plugin calls third-party getters that share the
    # parallel_state names (self._mpu.get_tensor_model_parallel_rank()).
    "debug_utils/dumper.py",
)

# Assignments to a server_args attribute (``server_args.x = ...``,
# ``self.server_args.x = ...``, and the ``sa`` alias used by a few helpers).
# ``==`` comparisons are excluded by the negative lookahead.
_MUTATION_PATTERNS = [
    re.compile(r"\bserver_args\.[a-z0-9_]+\s*=(?![=}])"),
    re.compile(r"\bsa\.[a-z0-9_]+\s*=(?![=}])"),
    re.compile(r"get_(?:global_)?server_args\(\)\.[a-z0-9_]+\s*=(?![=}])"),
    re.compile(
        r"setattr\(\s*(?:[\w.]+\.)?(?:server_args|sa|get_(?:global_)?server_args\(\))\s*,"
    ),
]

# The resolution pipeline itself (mutation is its job) and multimodal_gen,
# whose ServerArgs is a different class outside this contract.
_MUTATION_EXCLUDED = (
    "srt/server_args.py",
    "srt/arg_groups",
    "multimodal_gen",
)
_MUTATION_BASELINE = 0


class TestLegacyGlobalRatchet(CustomTestCase):
    def test_legacy_accessor_call_sites_match_the_baselines(self):
        sources = [
            path.read_text(encoding="utf-8", errors="replace")
            for path in sorted(_SRT_ROOT.rglob("*.py"))
        ]
        for name, pattern, baseline in _LEGACY_GLOBAL_RATCHETS:
            count = sum(len(re.findall(pattern, source)) for source in sources)
            if count > baseline:
                self.fail(
                    f"{name} call-sites grew: {count} > baseline {baseline}. "
                    "New code must use the sglang.srt.runtime_context accessors "
                    "(get_server_args() / get_context().set_server_args())."
                )
            if count < baseline:
                self.fail(
                    f"{name} call-sites shrank: {count} < baseline {baseline}. "
                    "Lower the baseline in this file to lock in the progress."
                )


class TestModuleStateRatchet(CustomTestCase):
    def test_global_statements_match_the_pins(self):
        for rel, pinned in _PINNED_GLOBALS.items():
            tree = ast.parse((_SRT_ROOT / rel).read_text())
            declared = {
                name
                for node in ast.walk(tree)
                if isinstance(node, ast.Global)
                for name in node.names
            }
            grown = declared - pinned
            self.assertFalse(
                grown,
                f"{rel} declares new module-level runtime state {sorted(grown)}; "
                "put runtime flags on a get_flags() group instead "
                "(see runtime_context.MoeFlags / DpFlags).",
            )
            shrunk = pinned - declared
            self.assertFalse(
                shrunk,
                f"{rel} no longer declares {sorted(shrunk)}; "
                "shrink the pin in this file to lock in the progress.",
            )


class TestParallelAdoptionRatchet(CustomTestCase):
    def test_no_legacy_parallel_getters_in_swept_dirs(self):
        offenders = []
        for top in _PARALLEL_SWEPT_DIRS:
            for path in sorted((_SRT_ROOT / top).rglob("*.py")):
                rel = path.relative_to(_SRT_ROOT).as_posix()
                if rel.startswith(_PARALLEL_EXEMPT):
                    continue
                for i, line in enumerate(path.read_text().split("\n"), 1):
                    if _BANNED_PARALLEL_CALLS.search(line):
                        offenders.append(f"{rel}:{i}")
        self.assertFalse(
            offenders,
            "legacy parallel-getter calls in swept directories (use "
            f"get_parallel().<dim> instead): {offenders}",
        )


class TestServerArgsMutationRatchet(CustomTestCase):
    def test_out_of_pipeline_mutations_match_the_baseline(self):
        count = 0
        for path in sorted(_SGLANG_ROOT.rglob("*.py")):
            rel = path.relative_to(_SGLANG_ROOT).as_posix()
            if rel.startswith(_MUTATION_EXCLUDED):
                continue
            source = path.read_text()
            count += sum(len(p.findall(source)) for p in _MUTATION_PATTERNS)
        if count > _MUTATION_BASELINE:
            self.fail(
                f"server_args mutations outside the resolution pipeline grew: "
                f"{count} > baseline {_MUTATION_BASELINE}. Configuration is resolved in "
                "ServerArgs.__post_init__; declare through the pipeline "
                "(passes / declare_late_resolution), change resolved config "
                "with get_context().override(source, ...), or hand the value "
                "to its runner as a constructor argument — do not assign fields."
            )
        if count < _MUTATION_BASELINE:
            self.fail(
                f"server_args mutations outside the resolution pipeline "
                f"shrank: {count} < baseline {_MUTATION_BASELINE}. Lower the baseline "
                "in this file to lock in the progress."
            )


_CONTEXT_MODULE = "sglang.srt.runtime_context"


def _module_level_accessor_imports(tree: ast.AST) -> set[str]:
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
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, ast.Starred):
        yield from _bound_names(target.value)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _bound_names(element)


def _own_scope_statements(node) -> tuple:
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
        for path in sorted(_SGLANG_ROOT.rglob("*.py")):
            rel = path.relative_to(_SGLANG_ROOT).as_posix()
            if rel.startswith("srt/runtime_context.py"):
                continue
            source = path.read_text()
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
