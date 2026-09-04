"""The supplied-instance surface is measured on two axes, and may only shrink.

A callee that takes ``server_args`` keeps the supplied-instance contract: the
caller chose the object, so no global-read ratchet counts it. Step 12 changes
what that object *carries* — the instance stays at the user's raw input — so a
callee reading a field **resolution fills in** would start seeing the CLI default
instead of the effective value.

This pins that intersection. Each entry is one (file, field) pair where a
parameter named ``server_args`` is read for a field resolution writes; the plan
doc carries the proposed disposition per field
(``global_context/12-raw-input-config.md``, "the supplied-instance conversion
list"). New pairs fail: a new one is new step-12 work, and the moment to decide
where the value should come from is when the read is written, not during the
flip. Pairs that disappear also fail, with the entry to delete — the list is the
measurement, not a memory of one.

The written-field set is derived here rather than hardcoded: the
representative configs in ``_MATRIX`` (one per resolution family it exercises)
are resolved and compared against the dataclass defaults, the same matrix the
context repo's audit tool uses. Ambient environment is normalized per entry --
resolution branches on CI detection and leaves sticky process state, so each
entry resolves from the pristine snapshot, and the CI shape is an explicit
entry rather than an accident of the runner. The read scan mirrors that
tool's three shapes — a parameter attribute, ``getattr(server_args, "literal")``,
and the parameter parked on ``self`` — because two implementations of one census
that disagree are worse than either alone.

The second axis is **already wrong today**, not after a flip. Some config is
decided *after* publish and recorded with ``get_context().override(...)`` —
elastic-EP resizing `ep_size`, a weight update rewriting `model_path` /
`load_format`, HiCache attach naming a storage backend, adaptive speculative
decoding moving `speculative_num_steps`. That write reaches the bags and never
the record, so a supplied-instance read of one of those fields answers with the
startup value from the moment the override lands. Whether that is a defect
depends on ordering — a value copied at construction, before any override, is
fine — so this axis is pinned as a measurement with the same growth guard rather
than as a list of bugs. One of them *was* a defect and is fixed at the base of
this stack: the linear-attn dispatch table rebuilt itself from the record after
the SM100 GDN prefill decision had been recorded in the bag, so a second runner's
rebuild dropped it. That choice is a per-runner stamp now and is not recorded
process-wide at all, so neither the read nor the field is on this axis.
"""

import ast
import dataclasses
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import sglang
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")
# Also on a CUDA runner: the written set is derived by resolving on the running
# host, and `is_cuda()` / capability gates only open on real hardware. The pin
# is split by host so both registrations stay exact: `_EXPOSED` is asserted
# everywhere, and a pair whose write only happens on CUDA belongs in
# `_EXPOSED_CUDA_ONLY` -- pinned on the CUDA runner, invisible to the CPU
# assertion. Without the split, one shared exact list could not hold such a
# pair at all: pinning it fails the CPU run as "gone", omitting it fails the
# CUDA run as "new". (No AMD registration: an `is_hip()`-gated write would
# shift the exact sets in ways none of the pinning hosts can verify; the ROCm
# resolution surface is covered by `test_resolution_is_reproducible.py`
# instead, whose assertion is device-agnostic.)
register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

_PACKAGE_ROOT = Path(next(iter(sglang.__path__))) / "srt"

# The config the resolution pipeline owns; reading the in-flight record is their
# job, not a supplied-instance read.
_OWNERS = ("server_args.py", "runtime_context.py", "arg_groups/")

_MINI_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "max_position_embeddings": 2048,
}

# One config resolves only its own decisions, so the written set is a union.
_MATRIX = (
    {},
    {
        "speculative_algorithm": "EAGLE",
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 4,
    },
    {"dp_size": 2, "tp_size": 2, "enable_dp_attention": True},
    # DWDP resolves dp_size and enable_dp_attention *itself* -- the plain DP
    # entry above passes them in, and passed-in fields are excluded from the
    # written set, so without this entry the dp_size readers would never pin.
    {"tp_size": 2, "dwdp_size": 2},
    {"enable_hierarchical_cache": True, "hicache_ratio": 2.0},
    {"disaggregation_mode": "prefill"},
    {"tp_size": 2, "attn_cp_size": 2},
    {"enable_lora": True, "max_lora_rank": 16},
    {"kv_cache_dtype": "fp8_e4m3", "page_size": 64},
    # MIS resolves disable_radix_cache (and friends) itself; the backend is
    # passed in because the handler asserts flashinfer rather than switching.
    {"enable_mis": True, "attention_backend": "flashinfer"},
)

# `declare_late_resolution` call sites whose keyword expansion is built
# dynamically; the written fields are spelled out here and drift-guarded.
_LATE_RESOLUTION_DYNAMIC_SITES = {
    "parser/template_detection.py": frozenset({"reasoning_parser", "tool_call_parser"}),
}

# `get_context().override(...)` declares through the same seam, but the fields
# are the caller's -- a test names them one call at a time. There is no static
# set to collect, and nothing resolution decides: whatever a caller overrides
# there is exposure only through that caller's own reads.
_CALLER_SUPPLIED_LATE_SITES = frozenset({"runtime_context.py"})

# Resolution also branches on ambient environment; those shapes are explicit
# entries so the written set is the same on every host. `SGLANG_IS_IN_CI`
# makes resolution fill `soft_watchdog_timeout`.
_ENV_MATRIX = (({}, {"SGLANG_IS_IN_CI": "true"}),)

# Only true constructor inputs: `tokenizer_path` / `served_model_name` are
# resolution-written (filled from `model_path` when unset), so their readers
# are step-12 exposure like any other pair.
_PASSED = frozenset({"model_path", "device", "random_seed"})

# Empty. A pair belongs here when a reader has no bag to read -- it runs before
# its process publishes -- and cannot use `resolving_view` either. The launcher's
# pre-publish reads (`_set_envs_and_config`, the auto-parser gate) and the
# late-resolution detection it calls all read the declarations now, so nothing
# qualifies. A new entry needs that kind of reason next to it.
_EXPOSED: frozenset = frozenset()

# Pairs whose resolution write only happens on a CUDA host (capability or
# `is_cuda()` gated): asserted on the CUDA registration, invisible to the CPU
# one. Empty today -- the current written sets coincide across the two hosts --
# but this is where a GPU-only write's readers get pinned without breaking the
# CPU-exact assertion.
_EXPOSED_CUDA_ONLY: frozenset = frozenset()


# Axis two: (file, field) pairs where a supplied-instance read names a field that
# some code overrides post-publish. Each needs an ordering judgment, not a blanket
# conversion; the list exists so a new one is a decision made when it is written.
_OVERRIDDEN_AND_READ: frozenset = frozenset()


def _expanded_override_keys(rel, tree, call, kw) -> set:
    """The statically visible keys behind an ``override(..., **expr)``.

    Handles a dict literal, a conditional between dict literals, and a name
    bound to a dict literal in the enclosing function (plus constant-subscript
    stores onto it -- the HiCache attach shape). One expansion is unresolvable
    by design and exempted by name: ``update_server_args`` forwards
    operator-chosen fields, so its key set is the API's, not this file's.
    Anything else unresolvable fails -- a silently skipped expansion would
    shrink the written set.
    """
    for a in call.args:
        if isinstance(a, ast.Constant) and a.value == "update_server_args":
            return set()
    for k in call.keywords:
        if (
            k.arg == "source"
            and isinstance(k.value, ast.Constant)
            and k.value.value == "update_server_args"
        ):
            return set()

    def loop_variable_values(name: str) -> set:
        """The values a `for name, ... in (<literal tuples>)` loop binds.

        A handler that records one field per loop iteration spells the field
        names in the loop's own literal, so they are still static.
        """
        values = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.For):
                continue
            target = node.target
            names = (
                [target]
                if isinstance(target, ast.Name)
                else list(getattr(target, "elts", []))
            )
            if not names or not isinstance(names[0], ast.Name) or names[0].id != name:
                continue
            if not (node.lineno <= call.lineno <= (node.end_lineno or node.lineno)):
                continue
            for item in getattr(node.iter, "elts", []):
                first = (
                    item.elts[0] if isinstance(item, ast.Tuple) and item.elts else item
                )
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    values.add(first.value)
        return values

    def dict_keys(node) -> set:
        assert isinstance(node, ast.Dict), (
            f"non-literal dict in override expansion at {rel}:{call.lineno}"
        )
        keys = set()
        for key in node.keys:
            if isinstance(key, ast.Constant):
                keys.add(key.value)
                continue
            assert isinstance(key, ast.Name), (
                f"non-literal dict key in override expansion at {rel}:{call.lineno}"
            )
            bound = loop_variable_values(key.id)
            assert bound, (
                f"dict key {key.id!r} at {rel}:{call.lineno} is not bound by a "
                "literal loop; extend the resolver"
            )
            keys |= bound
        return keys

    if isinstance(kw.value, ast.Dict):
        return dict_keys(kw.value)
    if isinstance(kw.value, ast.IfExp):
        keys = set()
        for branch in (kw.value.body, kw.value.orelse):
            if isinstance(branch, ast.Dict) and branch.keys:
                keys |= dict_keys(branch)
            elif isinstance(branch, ast.Dict):
                pass
            else:
                raise AssertionError(
                    f"unresolvable override expansion at {rel}:{call.lineno}"
                )
        return keys
    assert isinstance(kw.value, ast.Name), (
        f"unresolvable override expansion at {rel}:{call.lineno}"
    )
    name = kw.value.id
    enclosing = None
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                fn.lineno
                <= call.lineno
                <= max(getattr(fn, "end_lineno", fn.lineno), fn.lineno)
            ):
                if enclosing is None or fn.lineno > enclosing.lineno:
                    enclosing = fn
    assert enclosing is not None, (
        f"override expansion outside any function at {rel}:{call.lineno}"
    )
    keys = set()
    found = False
    for node in ast.walk(enclosing):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
            and isinstance(node.value, ast.Dict)
        ):
            found = True
            keys |= dict_keys(node.value)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == name
            and isinstance(node.targets[0].slice, ast.Constant)
        ):
            keys.add(node.targets[0].slice.value)
    assert found, (
        f"override expansion '{name}' at {rel}:{call.lineno} has no "
        "dict-literal assignment in its function; extend the resolver"
    )
    return keys


class TestSuppliedInstanceExposure(CustomTestCase):
    def _callTestMethod(self, method):
        # No CI retry: a failed first attempt has already resolved the matrix
        # and mutated process state; a retry against that contamination could
        # pass on a drifted written set or mask a real drift.
        return unittest.TestCase._callTestMethod(self, method)

    def setUp(self):
        # Resolving the matrix writes process state on the way through (the
        # multimodal transport handler sets SGLANG_USE_CUDA_IPC_TRANSPORT, and
        # `EnvField.set()` flips a descriptor flag `os.environ` does not carry).
        # Leaking it makes *later* files in the same worker fail, which is how
        # this was found -- so the case restores what it touched.
        super().setUp()
        state = (dict(os.environ), self._env_field_flags())
        self.addCleanup(self._restore_process_state, state)

    @staticmethod
    def _env_field_flags() -> dict:
        from sglang.srt.environ import EnvField, envs

        flags = {}
        for klass in reversed(type(envs).__mro__):
            for name, field in vars(klass).items():
                if isinstance(field, EnvField):
                    flags[name] = field._set_to_none
        return flags

    @staticmethod
    def _restore_process_state(state) -> None:
        from sglang.srt.environ import envs

        saved_environ, saved_flags = state
        os.environ.clear()
        os.environ.update(saved_environ)
        for name, was_none in saved_flags.items():
            getattr(type(envs), name)._set_to_none = was_none

    def _config_dir(self) -> str:
        config_dir = tempfile.mkdtemp(prefix="supplied_instance_")
        self.addCleanup(shutil.rmtree, config_dir, ignore_errors=True)
        with open(os.path.join(config_dir, "config.json"), "w") as handle:
            json.dump(_MINI_CONFIG, handle)
        return config_dir

    def _resolution_written_fields(self) -> set:
        """The union of what resolution fills in across the matrix.

        Every entry must resolve. A silently skipped one would shrink this set,
        which makes pinned pairs look like they disappeared -- the list would
        then drift by environment rather than by code, and the failure would
        point at the wrong thing. Each entry resolves from the pristine
        process snapshot (resolution writes env and EnvField flags on the way
        through, and DWDP flips `SGLANG_SCHEDULER_SKIP_ALL_GATHER`), so the
        union does not depend on matrix order; and the ambient CI marker is
        cleared, so a runner's identity cannot leak into the measurement --
        the CI-conditioned writes come from `_ENV_MATRIX`'s explicit entry.
        Late resolution counts too: `declare_late_resolution` writers run at
        launcher stage (LoRA normalization, parser auto-detection), so their
        target fields are collected statically from the call sites -- they are
        resolution writes by definition, just staged after `__post_init__`.
        """
        pristine = (dict(os.environ), self._env_field_flags())
        written = set()

        def resolve_one(extra, env):
            self._restore_process_state(pristine)
            os.environ.pop("SGLANG_IS_IN_CI", None)
            os.environ.update(env)
            model_path = self._config_dir()
            try:
                resolved = ServerArgs(
                    model_path=model_path, device="cuda", random_seed=42, **extra
                )
                resolved.resolve_once()
            except Exception as exc:
                self.fail(
                    f"the matrix entry {extra} (env={env}) did not resolve in "
                    f"this environment ({type(exc).__name__}: {exc}); the "
                    "written-field union would be short and the pinned list "
                    "would drift"
                )
            defaults = {}
            for field in dataclasses.fields(resolved):
                if field.default is not dataclasses.MISSING:
                    defaults[field.name] = field.default
                elif field.default_factory is not dataclasses.MISSING:
                    defaults[field.name] = field.default_factory()
            for field_name, default in defaults.items():
                if field_name in _PASSED or field_name in extra:
                    continue
                if getattr(resolved, field_name) != default:
                    written.add(field_name)

        for extra in _MATRIX:
            resolve_one(extra, {})
        for extra, env in _ENV_MATRIX:
            resolve_one(extra, env)
        self._restore_process_state(pristine)
        written |= self._late_resolution_written_fields()
        written |= self._hook_assignment_targets()
        written |= self._record_method_assignment_targets()
        written |= self._declarative_override_fields()
        return written

    def _hook_assignment_targets(self) -> set:
        """Fields any resolution hook can write, collected statically.

        The matrix can only enumerate families someone thought to add -- the
        DFLASH hole (its hook is the sole writer of
        `speculative_draft_attention_backend`, and no entry ran it) showed
        that a family nobody listed leaves its readers unpinned. The hook
        modules under `arg_groups/` are the resolution pipeline's extension
        points -- along with the NPU default helper, which the pipeline calls
        the same way -- and their write surface is the may-write set,
        family-blind by
        construction. A hook writes two ways: `server_args.field = ...`, and
        `declare_resolution(server_args, source, field=...)`, which records
        the write in the declaration stash on its way to the field. Counting
        only the assignment would read a hook's conversion to a declaration as
        the field having stopped being written. Collected like the
        late-resolution keywords: statically, failing loudly on an
        unparsable module. Underscore-prefixed targets are pipeline
        bookkeeping, not config leaves.
        """
        targets = set()
        modules = sorted((_PACKAGE_ROOT / "arg_groups").glob("*.py"))
        modules.append(_PACKAGE_ROOT / "hardware_backend/npu/utils.py")
        for path in modules:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            except SyntaxError:
                self.fail(f"unparsable hook module in the census: {path.name}")
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    tgts = node.targets
                elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                    tgts = [node.target]
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "declare_resolution"
                ):
                    targets |= {
                        kw.arg
                        for kw in node.keywords
                        if kw.arg and not kw.arg.startswith("_")
                    }
                    continue
                else:
                    continue
                for tgt in tgts:
                    if (
                        isinstance(tgt, ast.Attribute)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "server_args"
                        and not tgt.attr.startswith("_")
                    ):
                        targets.add(tgt.attr)
        return targets

    def _record_method_assignment_targets(self) -> set:
        """Fields ``ServerArgs``'s own methods can write, collected statically.

        The record's handlers are as family-conditional as the hooks -- the
        mooncake/layer_first layout rewrite, the deepseek-EP mode defaults,
        the seed fill that only runs when the caller did *not* supply one (so
        construct-and-diff can never see it: measuring requires supplying).
        A write site that can never fire is a dead branch to delete upstream,
        not a census exemption. Only names that are declared dataclass fields
        count; underscore bookkeeping does not.

        Two spellings write: an assignment, and ``self._declare(source,
        field=value)``, which records the write in the declaration stash on
        its way to the field. Counting only assignments would read a handler's
        conversion to a declaration as the field having stopped being written,
        which would quietly retire every pinned pair that reads it.
        """
        tree = ast.parse(
            (_PACKAGE_ROOT / "server_args.py").read_text(encoding="utf-8-sig")
        )
        sa_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ServerArgs"
        )
        declared = {
            node.target.id
            for node in sa_class.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        targets = set()
        for node in ast.walk(sa_class):
            if isinstance(node, ast.Assign):
                tgts = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                tgts = [node.target]
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "declare_resolution"
            ):
                targets |= {
                    kw.arg
                    for kw in node.keywords
                    if kw.arg in declared and not kw.arg.startswith("_")
                }
                continue
            else:
                continue
            for tgt in tgts:
                if (
                    isinstance(tgt, ast.Attribute)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self"
                    and tgt.attr in declared
                    and not tgt.attr.startswith("_")
                ):
                    targets.add(tgt.attr)
        # The deprecated-alias normalization declares through `**renamed`, so
        # the keyword scan sees no names; its field set is pinned here.
        alias_fields = {
            "attention_backend",
            "decode_attention_backend",
            "prefill_attention_backend",
            "speculative_draft_attention_backend",
        }

        # The handler lives in `arg_groups/serving_hook.py`, reached either as a
        # record method or as a bare-name call, so look the loop up by both.
        def _deprecated_alias_handler():
            for node in ast.walk(sa_class):
                if (
                    isinstance(node, ast.FunctionDef)
                    and node.name == "_handle_deprecated_args"
                    and any(isinstance(n, ast.For) for n in ast.walk(node))
                ):
                    return node
            for path in sorted((_PACKAGE_ROOT / "arg_groups").glob("*.py")):
                tree = ast.parse(path.read_text(encoding="utf-8-sig"))
                for node in tree.body:
                    if (
                        isinstance(node, ast.FunctionDef)
                        and node.name == "handle_deprecated_args"
                    ):
                        return node
            raise AssertionError("the deprecated-alias handler was not found")

        deprecated = _deprecated_alias_handler()
        found_tuples = [
            {elt.value for elt in node.iter.elts if isinstance(elt, ast.Constant)}
            for node in ast.walk(deprecated)
            if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple)
        ]
        self.assertIn(
            alias_fields,
            found_tuples,
            "the deprecated-alias normalization loop moved or changed its "
            "field tuple; update alias_fields to match",
        )
        return targets | alias_fields

    def _declarative_override_fields(self) -> set:
        """Fields the declarative override registry can write.

        ``MODEL_OVERRIDES`` maps arch -> {field: value}, and the
        ``@register_model_override``(-``_predicate``) providers return (or
        build by subscript) {field: value} dicts, which go straight into the
        declaration stash, so no assignment scan sees these writes and a
        llama-only matrix never triggers them. Keys must be
        string literals; anything else fails loudly.
        """
        tree = ast.parse(
            (_PACKAGE_ROOT / "arg_groups" / "overrides.py").read_text(
                encoding="utf-8-sig"
            )
        )
        fields = set()
        for node in tree.body:
            target = None
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                target = node.targets[0].id
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target = node.target.id
            if target != "MODEL_OVERRIDES" or node.value is None:
                continue
            for inner in ast.walk(node.value):
                if not isinstance(inner, ast.Dict):
                    continue
                for key, value in zip(inner.keys, inner.values):
                    if isinstance(value, ast.Dict):
                        continue  # arch -> {…} outer layer
                    self.assertIsInstance(key, ast.Constant, "non-literal override key")
                    fields.add(key.value)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not any(
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Name)
                and dec.func.id.startswith("register_model_override")
                for dec in node.decorator_list
            ):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Assign) and isinstance(
                    inner.targets[0], ast.Subscript
                ):
                    key = inner.targets[0].slice
                    self.assertIsInstance(
                        key, ast.Constant, f"non-literal override key in {node.name}"
                    )
                    fields.add(key.value)
                if isinstance(inner, ast.Dict):
                    for key in inner.keys:
                        self.assertIsInstance(
                            key,
                            ast.Constant,
                            f"non-literal override key in {node.name}",
                        )
                        fields.add(key.value)
        return fields

    def _late_resolution_written_fields(self) -> set:
        """Fields `declare_late_resolution` writes, collected statically.

        These are resolution's launcher-stage writes (they need a tokenizer or
        adapter load, so they cannot run in `__post_init__`), which the
        construct-and-diff pass above never sees. The keywords at the call
        sites are the written fields; an expansion this cannot resolve fails
        loudly like the override collector's, except the named dynamic sites
        below, whose field sets are spelled out and drift-guarded (each name
        must still appear as a constant in the file)."""
        written = set()
        root = _PACKAGE_ROOT
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            source = path.read_text(encoding="utf-8-sig")
            if "declare_late_resolution" not in source:
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                self.fail(f"unparsable module in the census: {rel}")
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and (
                        (
                            isinstance(node.func, ast.Name)
                            and node.func.id == "declare_late_resolution"
                        )
                        or (
                            isinstance(node.func, ast.Attribute)
                            and node.func.attr
                            in ("declare_late_resolution", "_late_resolution")
                        )
                    )
                ):
                    continue
                if all(kw.arg is None for kw in node.keywords) and any(
                    isinstance(kw.value, ast.Name) and kw.value.id == "fields"
                    for kw in node.keywords
                ):
                    # The forwarding shim (`ServerArgs._late_resolution` /
                    # the helper's own body) re-expands its caller's kwargs;
                    # the write sites are the callers.
                    continue
                for kw in node.keywords:
                    if kw.arg and kw.arg != "source":
                        written.add(kw.arg)
                    elif kw.arg is None:
                        if rel in _CALLER_SUPPLIED_LATE_SITES:
                            continue
                        dynamic = _LATE_RESOLUTION_DYNAMIC_SITES.get(rel)
                        if dynamic is not None:
                            constants = {
                                c.value
                                for c in ast.walk(tree)
                                if isinstance(c, ast.Constant)
                            }
                            missing = dynamic - constants
                            self.assertFalse(
                                missing,
                                f"{rel}: the declared dynamic field set drifted "
                                f"from the file ({sorted(missing)} not found)",
                            )
                            written |= dynamic
                        else:
                            written |= _expanded_override_keys(rel, tree, node, kw)
        return written

    _READS_CACHE = None

    def _supplied_instance_reads(self) -> set:
        """Three spellings of the same read: ``server_args.field`` off the
        parameter, ``getattr(server_args, "field", default)`` with a literal
        name, and the *parked* form -- ``self.x = server_args`` in a method
        that takes the parameter, read as ``self.x.field`` anywhere in the
        class. Parking under a different object, a container, or a computed
        name stays invisible, like in every census of this family. The
        loudest boundary *was* the *chain* spelling,
        ``model_runner.server_args.field`` off some other object, which this
        census still does not count -- but those reads are gone for every
        resolution-written field and ``test_chain_read_ratchet.py`` holds them
        at zero, so the gap is no longer where the risk is."""
        if TestSuppliedInstanceExposure._READS_CACHE is not None:
            return TestSuppliedInstanceExposure._READS_CACHE
        pairs = set()
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            rel = path.relative_to(_PACKAGE_ROOT).as_posix()
            if rel.startswith(_OWNERS):
                continue
            source = path.read_text(encoding="utf-8-sig")
            if "server_args" not in source:
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                # A silently dropped module shrinks `found` and reads as
                # intentional surface shrinkage under the bidirectional pin.
                self.fail(f"unparsable module in the census: {rel}")
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                params = {a.arg for a in list(fn.args.args) + list(fn.args.kwonlyargs)}
                if "server_args" not in params:
                    continue
                for node in ast.walk(fn):
                    if (
                        isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)
                        and node.value.id == "server_args"
                        and isinstance(node.ctx, ast.Load)
                    ):
                        pairs.add((rel, node.attr))
                    elif (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "getattr"
                        and len(node.args) >= 2
                        and isinstance(node.args[0], ast.Name)
                        and node.args[0].id == "server_args"
                        and isinstance(node.args[1], ast.Constant)
                        and isinstance(node.args[1].value, str)
                    ):
                        # The same read in optional clothing. Only a literal
                        # name is censusable; a computed one is not.
                        pairs.add((rel, node.args[1].value))
            for cls in ast.walk(tree):
                if not isinstance(cls, ast.ClassDef):
                    continue
                # Parked: `self.x = server_args` in a method that takes the
                # parameter, read as `self.x.field` anywhere in the class.
                parked = set()
                for fn in cls.body:
                    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    if "server_args" not in {
                        a.arg for a in list(fn.args.args) + list(fn.args.kwonlyargs)
                    }:
                        continue
                    for node in ast.walk(fn):
                        if (
                            isinstance(node, ast.Assign)
                            and len(node.targets) == 1
                            and isinstance(node.targets[0], ast.Attribute)
                            and isinstance(node.targets[0].value, ast.Name)
                            and node.targets[0].value.id == "self"
                            and isinstance(node.value, ast.Name)
                            and node.value.id == "server_args"
                        ):
                            parked.add(node.targets[0].attr)
                for node in ast.walk(cls):
                    if (
                        isinstance(node, ast.Attribute)
                        and isinstance(node.ctx, ast.Load)
                        and isinstance(node.value, ast.Attribute)
                        and node.value.attr in parked
                        and isinstance(node.value.value, ast.Name)
                        and node.value.value.id == "self"
                    ):
                        pairs.add((rel, node.attr))
        TestSuppliedInstanceExposure._READS_CACHE = pairs
        return pairs

    @staticmethod
    def _override_written_fields() -> set:
        """Fields written post-publish through ``get_context().override(...)``."""
        written = set()
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            rel = path.relative_to(_PACKAGE_ROOT).as_posix()
            source = path.read_text(encoding="utf-8-sig")
            if "record_config_updates" not in source and not (
                "get_context" in source and "override" in source
            ):
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                raise AssertionError(f"unparsable module in the census: {rel}")
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                ):
                    continue
                base = node.func.value
                is_override = node.func.attr == "override" and (
                    isinstance(base, ast.Call)
                    and isinstance(base.func, ast.Name)
                    and base.func.id == "get_context"
                )
                # `record_config_updates` is a named wrapper over override, so
                # its call sites are override sites. Its body forwards **kwargs
                # and names no field, so skip the forwarding call itself.
                is_wrapper = node.func.attr == "record_config_updates"
                if not (is_override or is_wrapper):
                    continue
                inside_wrapper = any(
                    isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and fn.name == "record_config_updates"
                    and fn.lineno <= node.lineno <= (fn.end_lineno or fn.lineno)
                    for fn in ast.walk(tree)
                )
                if inside_wrapper:
                    continue
                for kw in node.keywords:
                    if kw.arg == "source":
                        # Override metadata, not a config field.
                        continue
                    if kw.arg:
                        written.add(kw.arg)
                    else:
                        written |= _expanded_override_keys(rel, tree, node, kw)
        return written

    def test_the_post_publish_override_surface_matches_the_pinned_list(self):
        written = self._override_written_fields()
        self.assertGreater(
            len(written), 5, "found almost no override targets; the scan broke"
        )
        found = {pair for pair in self._supplied_instance_reads() if pair[1] in written}
        new = sorted(found - _OVERRIDDEN_AND_READ)
        gone = sorted(_OVERRIDDEN_AND_READ - found)
        self.assertEqual(
            ([], []),
            (new, gone),
            "the post-publish override surface drifted. A read here answers with "
            "the startup value once the override lands, so a new pair needs an "
            "ordering judgment: copied before any override (fine), or read after "
            "one (then it must come from the bags).\n"
            f"  new: {new}\n"
            f"  gone (delete from _OVERRIDDEN_AND_READ): {gone}",
        )

    def test_the_exposed_set_matches_the_pinned_list(self):
        import torch

        written = self._resolution_written_fields()
        found = {pair for pair in self._supplied_instance_reads() if pair[1] in written}
        expected = set(_EXPOSED)
        if torch.cuda.is_available():
            expected |= _EXPOSED_CUDA_ONLY
        new = sorted(found - expected)
        gone = sorted(expected - found)
        self.assertEqual(
            ([], []),
            (new, gone),
            "the supplied-instance step-12 surface drifted.\n"
            f"  new (decide where the resolved value comes from): {new}\n"
            f"  gone (delete from _EXPOSED / _EXPOSED_CUDA_ONLY): {gone}",
        )


if __name__ == "__main__":
    unittest.main()
