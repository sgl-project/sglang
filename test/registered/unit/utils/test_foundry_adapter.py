"""Unit tests for the Foundry adapter and its call sites (--cuda-graph-persistence).

TestFoundryAdapter: the no-op adapter without the flag, the process contract (a
record without the flag in a persistence-enabled process is an error), the
install error when the flag is set without Foundry, the integration-API version
check, that every adapter method reaches foundry.integration.sglang.api (a
fake api module stands in for Foundry), and the fallback: an ``Unsupported``
from any step turns every later method into a no-op.

TestDispatcherPrebuildProtocol: which dispatchers implement ``prebuild``, and that
native prebuilding leaves the others alone.

TestFoundryCallSites: Foundry's archive is only valid if the call sites stay at
their sequence points (before or after the work they bracket), so a refactor
that moves or drops one fails here instead of at a user's LOAD. It reads
source only.

No GPU, no Foundry.

Run:  python -m pytest test/registered/unit/utils/test_foundry_adapter.py -v
"""

import ast
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import sglang
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    prebuild_dispatchers,
)
from sglang.srt.utils import foundry_adapter
from sglang.srt.utils.foundry_adapter import (
    FOUNDRY_INTEGRATION_API_MAJOR,
    FOUNDRY_INTEGRATION_API_MIN_MINOR,
    FoundryAdapter,
    activate_foundry,
    get_foundry_adapter,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_API_MODULE = "foundry.integration.sglang.api"


class _Unsupported(Exception):
    def __init__(self, step, reason, *, inherited=False, quiet=False):
        super().__init__(f"unsupported at {step}: {reason}")
        self.step = step
        self.reason = reason
        self.inherited = inherited
        self.quiet = quiet


def _fake_api(
    version=(FOUNDRY_INTEGRATION_API_MAJOR, FOUNDRY_INTEGRATION_API_MIN_MINOR),
    unsupported_at=None,
):
    """``unsupported_at``: the api function that raises ``Unsupported``."""
    api = types.ModuleType(_API_MODULE)
    api.INTEGRATION_API_VERSION = version
    api.Unsupported = _Unsupported
    api.calls = []

    def record(name, result=None):
        def fn(*args, **kwargs):
            api.calls.append((name, args, kwargs))
            if name == unsupported_at:
                raise _Unsupported(name, "fake reason")
            return result

        return fn

    for name in (
        "activate",
        "deactivate",
        "pin_server_args",
        "validate_resolved_server_args",
        "before_parallel_init",
        "record_memory_pool_config",
        "log_init_summary",
    ):
        setattr(api, name, record(name))
    api.capture_one = record("capture_one", ("graph", "out"))
    api.replay_saved_memory_pool_config = record(
        "replay_saved_memory_pool_config", "cfg"
    )

    @contextmanager
    def scope(name, *args):
        api.calls.append((name, args, {}))
        if name == unsupported_at:
            raise _Unsupported(name, "fake reason")
        yield

    api.configure_subprocess = lambda server_args=None: scope(
        "configure_subprocess", server_args
    )
    api.capture_scope = (
        lambda phase, *, model, req_slots=None, kind=None, flashinfer_tactics=None: (
            scope("capture_scope", phase, model, req_slots, kind, flashinfer_tactics)
        )
    )
    return api


def _sa(mode=None, config=None):
    return SimpleNamespace(
        cuda_graph_persistence=mode, cuda_graph_persistence_config=config
    )


@contextmanager
def _installed(api):
    """``api`` importable as foundry.integration.sglang.api and a fresh
    process-level adapter."""
    parents = {
        "foundry": types.ModuleType("foundry"),
        "foundry.integration": types.ModuleType("foundry.integration"),
        "foundry.integration.sglang": types.ModuleType("foundry.integration.sglang"),
    }
    parents["foundry.integration.sglang"].api = api
    with (
        mock.patch.dict(sys.modules, {**parents, _API_MODULE: api}),
        mock.patch.object(foundry_adapter, "_active", foundry_adapter._NOOP),
    ):
        yield


class TestFoundryAdapter(CustomTestCase):
    def test_noop_without_the_flag(self):
        with mock.patch.object(foundry_adapter, "_active", foundry_adapter._NOOP):
            adapter = activate_foundry(_sa())
            self.assertIs(adapter, get_foundry_adapter())
            self.assertFalse(adapter.enabled)
            self.assertIsNone(adapter.replay_saved_memory_pool_config())
            with (
                adapter.capture_scope(phase="decode", model=object()),
                adapter.configure_subprocess(),
            ):
                pass

    def test_flag_without_foundry_raises_with_install_hint(self):
        with (
            mock.patch.dict(sys.modules, {"foundry": None}),
            mock.patch.object(foundry_adapter, "_active", foundry_adapter._NOOP),
            self.assertRaisesRegex(ImportError, r"sglang\[foundry\]"),
        ):
            activate_foundry(_sa("save"))

    def test_api_version_mismatch_is_refused(self):
        for version in (
            (FOUNDRY_INTEGRATION_API_MAJOR + 1, FOUNDRY_INTEGRATION_API_MIN_MINOR),
            (FOUNDRY_INTEGRATION_API_MAJOR - 1, FOUNDRY_INTEGRATION_API_MIN_MINOR),
        ):
            with self.subTest(version=version):
                api = _fake_api(version=version)
                with (
                    _installed(api),
                    self.assertRaisesRegex(RuntimeError, "is not supported"),
                ):
                    activate_foundry(_sa("save"))
                self.assertEqual(api.calls, [])
                self.assertIs(get_foundry_adapter(), foundry_adapter._NOOP)

    def test_record_without_the_flag_in_an_enabled_process_is_refused(self):
        """Foundry's hook, region and env pins cannot be undone in a live
        process: a later record without the flag must not run there."""
        api = _fake_api()
        with _installed(api):
            enabled = activate_foundry(_sa("save"))
            with self.assertRaisesRegex(RuntimeError, "cannot be undone"):
                activate_foundry(_sa())
            # The enabled adapter stays the process's adapter.
            self.assertIs(get_foundry_adapter(), enabled)

    def test_activation_passes_mode_and_config(self):
        api = _fake_api()
        with _installed(api):
            sa = _sa("load", "/x/foundry.toml")
            adapter = activate_foundry(sa)
            self.assertIs(get_foundry_adapter(), adapter)
            self.assertTrue(adapter.enabled)
            self.assertEqual(adapter.mode, "load")
            # Idempotent: a second record re-activates with its own arguments.
            self.assertIs(activate_foundry(sa), adapter)
        activations = [c for c in api.calls if c[0] == "activate"]
        self.assertEqual(len(activations), 2)
        for _, _, kwargs in activations:
            self.assertEqual(kwargs, {"mode": "load", "config_path": "/x/foundry.toml"})

    def test_every_adapter_method_reaches_the_api(self):
        """A no-op method added without a real override would silently do
        nothing with the flag set."""
        api = _fake_api()
        with _installed(api):
            adapter = activate_foundry(_sa("save"))
            api.calls.clear()
            sa = object()
            adapter.pin_server_args(sa)
            adapter.validate_resolved_server_args(sa)
            with adapter.configure_subprocess(sa):
                pass
            adapter.before_parallel_init("cuda")
            adapter.replay_saved_memory_pool_config()
            adapter.record_memory_pool_config(object())
            with adapter.capture_scope(phase="decode", model=object()):
                pass
            adapter.capture_one(
                shape_key="key", forward_fn=None, pool="pool", stream="stream"
            )
            adapter.log_init_summary()
        called = {name for name, _, _ in api.calls}
        methods = {
            name
            for name, value in vars(FoundryAdapter).items()
            if callable(value) and not name.startswith("_")
        }
        self.assertEqual(called, methods)

    def _run_init(self, adapter, body):
        """The scheduler-side calls in process order; ``body`` runs inside the
        capture scope. Returns what the pool-config replay and capture_one gave."""
        adapter.before_parallel_init("cuda")
        replayed = adapter.replay_saved_memory_pool_config()
        adapter.record_memory_pool_config(object())
        with adapter.capture_scope(phase="decode", model=object()):
            body()
            captured = adapter.capture_one(
                shape_key="key", forward_fn=None, pool="pool", stream="stream"
            )
        adapter.log_init_summary()
        return replayed, captured

    def test_unsupported_at_any_step_falls_back_natively(self):
        """After ``Unsupported`` at any step every later call is a no-op: the
        pool config resolves natively (None), the capture loop runs exactly once
        and each shape is captured natively (None); Foundry is told once."""
        steps = (
            "activate",
            "pin_server_args",
            "validate_resolved_server_args",
            "before_parallel_init",
            "replay_saved_memory_pool_config",
            "capture_scope",
            "capture_one",
        )
        for step in steps:
            with self.subTest(step=step):
                api = _fake_api(unsupported_at=step)
                with _installed(api):
                    adapter = activate_foundry(_sa("save"))
                    sa = object()
                    adapter.pin_server_args(sa)
                    adapter.validate_resolved_server_args(sa)
                    loops = []
                    replayed, captured = self._run_init(
                        adapter, lambda: loops.append(1)
                    )
                    self.assertFalse(adapter.enabled)
                    self.assertIs(activate_foundry(_sa("save")), adapter)
                self.assertEqual(loops, [1])
                self.assertIsNone(captured)
                in_scope = step in ("capture_scope", "capture_one")
                self.assertEqual(replayed, "cfg" if in_scope else None)
                names = [name for name, _, _ in api.calls]
                self.assertEqual(
                    [kw for name, _, kw in api.calls if name == "deactivate"],
                    [{"step": step, "reason": "fake reason"}],
                )
                after = names[names.index("deactivate") + 1 :]
                self.assertNotIn("capture_one", after)
                self.assertNotIn("record_memory_pool_config", after)
                self.assertNotIn("log_init_summary", after)

    def test_other_foundry_errors_propagate(self):
        """Only ``Unsupported`` falls back; a Foundry bug must stay an error."""
        api = _fake_api()

        def broken(**kwargs):
            raise RuntimeError("restore failed")

        api.capture_one = broken
        with _installed(api):
            adapter = activate_foundry(_sa("load"))
            with self.assertRaisesRegex(RuntimeError, "restore failed"):
                adapter.capture_one(
                    shape_key="key", forward_fn=None, pool="pool", stream="stream"
                )
            self.assertTrue(adapter.enabled)

    def test_spawn_after_a_fallback_still_reaches_foundry(self):
        """The children learn about the fallback from Foundry's spawn wrapper."""
        api = _fake_api(unsupported_at="validate_resolved_server_args")
        with _installed(api):
            adapter = activate_foundry(_sa("save"))
            adapter.validate_resolved_server_args(object())
            with adapter.configure_subprocess(object()):
                pass
        self.assertEqual(api.calls[-1][0], "configure_subprocess")


class _Dispatcher(BaseDispatcher):
    def dispatch(self, hidden_states, topk_output):
        raise NotImplementedError

    def combine(self, combine_input):
        raise NotImplementedError


class _LazyDispatcher(_Dispatcher):
    """Builds its state at the first dispatch, as a new backend would."""

    @property
    def prebuild_pending(self):
        return True


class _PrebuiltDispatcher(_Dispatcher):
    built = False

    @property
    def prebuild_pending(self):
        return not self.built

    def prebuild(self):
        self.built = True


class TestDispatcherPrebuildProtocol(CustomTestCase):
    def test_only_an_overridden_prebuild_implements_the_protocol(self):
        self.assertFalse(_LazyDispatcher().implements_prebuild)
        self.assertTrue(_PrebuiltDispatcher().implements_prebuild)

    def test_native_prebuild_skips_dispatchers_without_the_protocol(self):
        """A dispatcher without ``prebuild`` keeps its lazy build natively, even
        when it reports pending state; one with it is prebuilt."""
        model = torch.nn.Module()
        model.lazy = torch.nn.Module()
        model.lazy.dispatcher = _LazyDispatcher()
        model.prebuilt = torch.nn.Module()
        model.prebuilt.dispatcher = _PrebuiltDispatcher()
        self.assertEqual(prebuild_dispatchers(model), 1)
        self.assertTrue(model.prebuilt.dispatcher.built)
        self.assertEqual(prebuild_dispatchers(model), 0)


_SRT = Path(sglang.__file__).parent / "srt"
_DECODE = "model_executor/runner/decode_cuda_graph_runner.py"
_PREFILL = "model_executor/runner/prefill_cuda_graph_runner.py"

# (file, function qualname, adapter call, where): "first" = the first
# statement after the docstring, "last" = the last statement before the
# return, "before:X" / "after:X" = before the next / after the last use of X,
# "any" = somewhere in the body.
_CALL_SITES = [
    ("distributed/bootstrap.py", "init_parallel_runtime",
     "get_foundry_adapter().before_parallel_init", "before:_resolve_backend"),
    ("mem_cache/kv_cache_configurator.py",
     "KVCacheConfigurator._resolve_memory_pool_config",
     "get_foundry_adapter().replay_saved_memory_pool_config",
     "before:self._profile_available_bytes"),
    ("mem_cache/kv_cache_configurator.py",
     "KVCacheConfigurator._resolve_memory_pool_config",
     "get_foundry_adapter().record_memory_pool_config", "last"),
    (_DECODE, "DecodeCudaGraphRunner.capture",
     "get_foundry_adapter().capture_scope", "any"),
    (_PREFILL, "PrefillCudaGraphRunner.capture",
     "get_foundry_adapter().capture_scope", "any"),
    ("model_executor/runner_backend/full_cuda_graph_backend.py",
     "FullCudaGraphBackend.capture_one", "foundry.capture_one",
     "before:self._cuda_graph_runner"),
    ("managers/scheduler.py", "run_scheduler_process", "activate_foundry",
     "before:publish"),
    ("managers/scheduler.py", "run_scheduler_process",
     "get_foundry_adapter().log_init_summary", "before:pipe_writer.send"),
    ("entrypoints/engine.py", "Engine._launch_scheduler_processes",
     "activate_foundry", "before:proc.start"),
    ("entrypoints/engine.py", "Engine._launch_scheduler_processes",
     "foundry_adapter.configure_subprocess", "before:proc.start"),
]  # fmt: skip

# Resolution steps, each right after the step named (None: right before the
# next one), in pipeline order.
_RESOLUTION_STEPS = [
    ("handle_cuda_graph_persistence", "before", "apply_inkling_prefill_cuda_graph_default"),
    ("validate_cuda_graph_persistence", "after", "handle_other_validations"),
]  # fmt: skip

# Names Foundry reads: get_parallel().<name>, get_device().<name>,
# get_flags().dp.<name>, get_context().<name>() and the accessors themselves.
_READ = [
    ("arg_groups/fields/parallel.py", "Parallel",
     {"tp_size", "pp_size", "tp_rank", "pp_rank", "dp_rank", "attn_dp_enabled"}),
    ("arg_groups/fields/device.py", "Device", {"device", "gpu_id"}),
    ("runtime_context.py", "DpFlags", {"prefill_graph_has_dp_gather"}),
    ("runtime_context.py", "RuntimeContext", {"override", "overrides_log"}),
    ("runtime_context.py", None,
     {"get_parallel", "get_device", "get_flags", "get_context"}),
]  # fmt: skip


def _tree(relpath):
    return ast.parse((_SRT / relpath).read_text(encoding="utf-8"))


def _find(relpath, qualname):
    node = _tree(relpath)
    for part in qualname.split("."):
        node = next(
            n
            for n in node.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part
        )
    return node


def _call_lines(fn, prefix):
    """Lines of calls whose callee source starts with ``prefix``."""
    return sorted(
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and ast.unparse(n.func).startswith(prefix)
    )


def _name_lines(fn, prefix):
    return sorted(
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, (ast.Call, ast.Attribute, ast.Name))
        and ast.unparse(n).startswith(prefix)
    )


def _body(fn):
    body = list(fn.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        body = body[1:]
    return body


def _members(node):
    names = set()
    for n in node.body:
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names.add(n.target.id)
        elif isinstance(n, ast.Assign):
            names.update(t.id for t in n.targets if isinstance(t, ast.Name))
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            names.add(n.name)
    return names


class TestFoundryCallSites(CustomTestCase):
    def test_call_sites_stay_at_their_sequence_points(self):
        for path, qualname, call, where in _CALL_SITES:
            with self.subTest(site=qualname, call=call):
                fn = _find(path, qualname)
                lines = _call_lines(fn, call)
                self.assertTrue(lines, f"{qualname} no longer calls {call}")
                body = _body(fn)
                if where == "first":
                    self.assertIn(call, ast.unparse(body[0]))
                elif where == "last":
                    tail = [s for s in body if not isinstance(s, ast.Return)][-1]
                    self.assertIn(call, ast.unparse(tail))
                elif where.startswith("before:"):
                    other = _name_lines(fn, where.split(":", 1)[1])
                    self.assertTrue(any(o > lines[0] for o in other), where)
                elif where.startswith("after:"):
                    other = _name_lines(fn, where.split(":", 1)[1])
                    self.assertTrue(other, where)
                    self.assertGreater(lines[-1], other[-1])

    def test_every_engine_scheduler_spawn_is_wrapped(self):
        """Every scheduler spawn and the DP-controller spawn: the controller's
        schedulers inherit the hook preload from its environment."""
        fn = _find("entrypoints/engine.py", "Engine._launch_scheduler_processes")
        self.assertEqual(
            len(_call_lines(fn, "foundry_adapter.configure_subprocess")),
            len(_call_lines(fn, "proc.start")),
        )

    def test_capture_scope_wraps_the_whole_capture_loop(self):
        """Foundry's scope must bracket warmup and every shape, inside the
        runner's model context: capture() enters model_context() and then the
        scope, around _capture() alone."""
        for path, cls in (
            (_DECODE, "DecodeCudaGraphRunner"),
            (_PREFILL, "PrefillCudaGraphRunner"),
        ):
            with self.subTest(runner=cls):
                body = _body(_find(path, f"{cls}.capture"))
                scope = body[-1]
                self.assertIsInstance(scope, ast.With)
                items = [ast.unparse(item.context_expr) for item in scope.items]
                self.assertEqual(len(items), 2)
                self.assertEqual(items[0], "self.model_context()")
                self.assertTrue(
                    items[1].startswith("get_foundry_adapter().capture_scope(")
                )
                self.assertEqual(
                    [ast.unparse(s) for s in scope.body], ["self._capture()"]
                )
                inner = _find(path, f"{cls}._capture")
                self.assertTrue(_call_lines(inner, "self.warmup"))

    def test_resolution_steps_keep_their_positions(self):
        overridable = next(
            n.value
            for n in _tree("arg_groups/resolution_hooks.py").body
            if isinstance(n, ast.AnnAssign) and n.target.id == "_OVERRIDABLE_HOOKS"
        )
        overridable = {
            n.value for n in ast.walk(overridable) if isinstance(n, ast.Constant)
        }
        steps = [
            ast.unparse(n.args[0])
            for n in ast.walk(
                _find("arg_groups/pipeline.py", "run_resolution_pipeline")
            )
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "run_hook"
        ]
        for name, relation, anchor in _RESOLUTION_STEPS:
            with self.subTest(step=name):
                self.assertIn(name, overridable)
                offset = 1 if relation == "after" else -1
                self.assertEqual(steps[steps.index(anchor) + offset], name)
        order = [steps.index(name) for name, _, _ in _RESOLUTION_STEPS]
        self.assertEqual(order, sorted(order))

    def test_read_names_exist(self):
        """Renamed fields would give Foundry a wrong workspace rank or device."""
        for path, cls, names in _READ:
            with self.subTest(owner=cls or path):
                owner = _find(path, cls) if cls else _tree(path)
                self.assertLessEqual(names, _members(owner))


if __name__ == "__main__":
    unittest.main()
