"""Unit tests for the Foundry adapter and its call sites (--cuda-graph-persistence).

TestFoundryAdapter: the no-op adapter without the flag, the process contract (a
record without the flag in a persistence-enabled process is an error), the
install error when the flag is set without Foundry, the integration-API version
check, and delegation of every adapter method to foundry.integration.sglang.api
(a fake api module stands in for Foundry).

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

import sglang
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


def _fake_api(
    version=(FOUNDRY_INTEGRATION_API_MAJOR, FOUNDRY_INTEGRATION_API_MIN_MINOR),
):
    api = types.ModuleType(_API_MODULE)
    api.INTEGRATION_API_VERSION = version
    api.calls = []

    def record(name, result=None):
        def fn(*args, **kwargs):
            api.calls.append((name, args, kwargs))
            return result

        return fn

    for name in (
        "activate",
        "pin_server_args",
        "validate_graph_config",
        "validate_resolved_server_args",
        "apply_env_pins",
        "before_parallel_init",
        "after_parallel_init",
        "after_runner_distributed_init",
        "record_memory_pool_overrides",
        "before_alloc_memory_pool",
        "after_alloc_memory_pool",
    ):
        setattr(api, name, record(name))
    api.capture_one = record("capture_one", ("graph", "out"))
    api.replay_saved_memory_pool_config = record(
        "replay_saved_memory_pool_config", "cfg"
    )

    @contextmanager
    def scope(name, *args):
        api.calls.append((name, args, {}))
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
    """``api`` importable as foundry.integration.sglang.api, version check
    passing, and a fresh process-level adapter."""
    parents = {
        "foundry": types.ModuleType("foundry"),
        "foundry.integration": types.ModuleType("foundry.integration"),
        "foundry.integration.sglang": types.ModuleType("foundry.integration.sglang"),
    }
    parents["foundry.integration.sglang"].api = api
    with (
        mock.patch.dict(sys.modules, {**parents, _API_MODULE: api}),
        mock.patch("sglang.srt.utils.common.assert_pkg_version") as pkg_check,
        mock.patch.object(foundry_adapter, "_active", foundry_adapter._NOOP),
    ):
        yield pkg_check


class TestFoundryAdapter(CustomTestCase):
    def test_noop_without_the_flag(self):
        with mock.patch.object(foundry_adapter, "_active", foundry_adapter._NOOP):
            adapter = activate_foundry(_sa())
            self.assertIs(adapter, get_foundry_adapter())
            self.assertFalse(adapter.enabled)
            self.assertIsNone(adapter.replay_saved_memory_pool_config())
            with (
                adapter.capture_scope("decode", model=object()),
                adapter.configure_subprocess(),
            ):
                pass

    def test_flag_without_foundry_raises_with_install_hint(self):
        with (
            mock.patch.dict(sys.modules, {"foundry": None}),
            mock.patch.object(foundry_adapter, "_active", foundry_adapter._NOOP),
            self.assertLogs(foundry_adapter.logger, level="WARNING") as logs,
            self.assertRaises(ImportError),
        ):
            activate_foundry(_sa("save"))
        self.assertIn("sglang[foundry]", "\n".join(logs.output))

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
                    FoundryAdapter.create(True, mode="save")
                self.assertEqual(api.calls, [])

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

    def test_activation_checks_the_package_and_passes_mode_and_config(self):
        api = _fake_api()
        with _installed(api) as pkg_check:
            sa = _sa("load", "/x/foundry.toml")
            adapter = activate_foundry(sa)
            self.assertIs(get_foundry_adapter(), adapter)
            self.assertTrue(adapter.enabled)
            self.assertEqual(adapter.mode, "load")
            # Idempotent: a second record re-activates with its own arguments.
            self.assertIs(activate_foundry(sa), adapter)
        pkg_check.assert_called_once()
        self.assertEqual(pkg_check.call_args.args[0], "foundry-core")
        activations = [c for c in api.calls if c[0] == "activate"]
        self.assertEqual(len(activations), 2)
        for _, _, kwargs in activations:
            self.assertEqual(kwargs, {"mode": "load", "config_path": "/x/foundry.toml"})

    def test_every_method_delegates_to_the_api(self):
        api = _fake_api()
        with _installed(api):
            adapter = FoundryAdapter.create(True, mode="save")
            api.calls.clear()
            sa, model, cfg = object(), object(), object()
            adapter.pin_server_args(sa)
            adapter.validate_graph_config(sa)
            adapter.validate_resolved_server_args(sa)
            adapter.apply_env_pins(sa)
            with adapter.configure_subprocess(sa):
                pass
            adapter.before_parallel_init("cuda")
            adapter.after_parallel_init()
            adapter.after_runner_distributed_init(False)
            self.assertEqual(adapter.replay_saved_memory_pool_config(), "cfg")
            adapter.record_memory_pool_overrides()
            adapter.before_alloc_memory_pool(False)
            adapter.after_alloc_memory_pool(cfg, False)
            with adapter.capture_scope(
                "prefill", model=model, req_slots=4, flashinfer_tactics="/t.json"
            ):
                pass
            by_name = {name: (args, kwargs) for name, args, kwargs in api.calls}
            self.assertEqual(by_name["after_runner_distributed_init"], ((False,), {}))
            self.assertEqual(by_name["after_alloc_memory_pool"], ((cfg, False), {}))
            self.assertEqual(
                by_name["capture_scope"],
                (("prefill", model, 4, None, "/t.json"), {}),
            )
            with adapter.capture_scope("decode", model=model, kind="draft_extend"):
                pass
            self.assertEqual(
                api.calls[-1],
                ("capture_scope", ("decode", model, None, "draft_extend", None), {}),
            )
            self.assertEqual(
                adapter.capture_one("key", None, pool="pool", stream="stream"),
                ("graph", "out"),
            )
            name, args, kwargs = api.calls[-1]
            self.assertEqual((name, args), ("capture_one", ("key", None)))
            self.assertEqual(kwargs, {"pool": "pool", "stream": "stream"})
        called = sorted({name for name, _, _ in api.calls})
        methods = [
            name
            for name, value in vars(FoundryAdapter).items()
            if callable(value) and not name.startswith("_") and name != "create"
        ]
        self.assertEqual(sorted(called), sorted(methods))


_SRT = Path(sglang.__file__).parent / "srt"
_DECODE = "model_executor/runner/decode_cuda_graph_runner.py"
_PREFILL = "model_executor/runner/prefill_cuda_graph_runner.py"
_MR = "model_executor/model_runner.py"

# (file, function qualname, adapter call, where): "first" = the first
# statement after the docstring, "last" = the last statement before the
# return, "before:X" / "after:X" = before the next / after the last use of X,
# "any" = somewhere in the body.
_CALL_SITES = [
    ("distributed/bootstrap.py", "init_parallel_runtime",
     "get_foundry_adapter().before_parallel_init", "before:_resolve_backend"),
    ("distributed/bootstrap.py", "init_parallel_runtime",
     "get_foundry_adapter().after_parallel_init", "after:_init_parallel_groups"),
    (_MR, "ModelRunner.init_torch_distributed",
     "get_foundry_adapter().after_runner_distributed_init", "last"),
    (_MR, "ModelRunner.alloc_memory_pool",
     "get_foundry_adapter().before_alloc_memory_pool", "first"),
    (_MR, "ModelRunner.alloc_memory_pool",
     "get_foundry_adapter().after_alloc_memory_pool", "last"),
    ("mem_cache/kv_cache_configurator.py",
     "KVCacheConfigurator._resolve_memory_pool_config",
     "get_foundry_adapter().replay_saved_memory_pool_config",
     "before:self._profile_available_bytes"),
    ("mem_cache/kv_cache_configurator.py",
     "KVCacheConfigurator._resolve_memory_pool_config",
     "get_foundry_adapter().record_memory_pool_overrides", "last"),
    (_DECODE, "DecodeCudaGraphRunner.capture",
     "get_foundry_adapter().capture_scope", "any"),
    (_PREFILL, "PrefillCudaGraphRunner.capture",
     "get_foundry_adapter().capture_scope", "any"),
    ("model_executor/runner_backend/full_cuda_graph_backend.py",
     "FullCudaGraphBackend.capture_one", "foundry.capture_one",
     "before:self._cuda_graph_runner"),
    ("managers/scheduler.py", "run_scheduler_process", "activate_foundry",
     "before:publish"),
    ("managers/data_parallel_controller.py", "run_data_parallel_controller_process",
     "activate_foundry", "before:publish"),
    ("managers/data_parallel_controller.py",
     "DataParallelController.launch_tensor_parallel_group",
     "get_foundry_adapter().configure_subprocess", "before:proc.start"),
    ("entrypoints/engine.py", "Engine._launch_scheduler_processes",
     "foundry_adapter.configure_subprocess", "before:proc.start"),
    ("entrypoints/engine.py", "Engine._launch_scheduler_processes",
     "activate_foundry(server_args).configure_subprocess", "before:proc.start"),
    ("entrypoints/engine.py", "_set_envs_and_config",
     "activate_foundry(server_args).apply_env_pins", "before:is_mnnvl_fabric_device"),
]  # fmt: skip

# Resolution steps, each right after the step named (None: right before the
# next one), in pipeline order.
_RESOLUTION_STEPS = [
    ("handle_cuda_graph_persistence", "before", "apply_inkling_prefill_cuda_graph_default"),
    ("validate_cuda_graph_persistence_graph_config", "after", "handle_cuda_graph_config"),
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

# What each call site hands Foundry: values SGLang owns (the loaded model, the
# draft-worker flag, the resolved pool config, the prefill request slots),
# never a runner or backend object. capture_one is checked separately
# (test_capture_one_passes_arguments_and_stores_the_result).
_CALL_ARGS = [
    (_MR, "ModelRunner.init_torch_distributed",
     "get_foundry_adapter().after_runner_distributed_init(self.is_draft_worker)"),
    (_MR, "ModelRunner.alloc_memory_pool",
     "get_foundry_adapter().before_alloc_memory_pool(self.is_draft_worker)"),
    (_MR, "ModelRunner.alloc_memory_pool",
     "get_foundry_adapter().after_alloc_memory_pool("
     "memory_pool_config=self.memory_pool_config, is_draft_worker=self.is_draft_worker)"),
    (_DECODE, "DecodeCudaGraphRunner.capture",
     "get_foundry_adapter().capture_scope('decode', model=self.model_runner.model, "
     "kind=self._foundry_graph_kind(), flashinfer_tactics=tactics)"),
    (_PREFILL, "PrefillCudaGraphRunner.capture",
     "get_foundry_adapter().capture_scope('prefill', model=self.model_runner.model, "
     "req_slots=self._capture_req_slots, flashinfer_tactics=tactics)"),
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

    def test_capture_one_passes_arguments_and_stores_the_result(self):
        """Foundry gets what it needs as arguments and SGLang assigns the
        returned graph and output into its own backend fields."""
        fn = _find(
            "model_executor/runner_backend/full_cuda_graph_backend.py",
            "FullCudaGraphBackend.capture_one",
        )
        call = next(
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "foundry.capture_one"
        )
        self.assertEqual(
            [ast.unparse(a) for a in call.args], ["shape_key", "forward_fn"]
        )
        self.assertEqual(
            {k.arg: ast.unparse(k.value) for k in call.keywords},
            {
                "pool": "self._pool",
                "stream": "self._capture_stream",
            },
        )
        assign = next(
            n for n in ast.walk(fn) if isinstance(n, ast.Assign) and n.value is call
        )
        self.assertEqual(
            [ast.unparse(t) for t in assign.targets[0].elts],
            ["self._graphs[shape_key]", "self._outputs[shape_key]"],
        )
        # Nothing hands Foundry the backend itself.
        self.assertNotIn("self", [ast.unparse(a) for a in call.args])

    def test_capture_scope_wraps_the_whole_capture_loop(self):
        """Foundry's scope must bracket warmup and every shape: the runner's
        capture() is only the scope around _capture_graphs()."""
        for path, cls in (
            (_DECODE, "DecodeCudaGraphRunner"),
            (_PREFILL, "PrefillCudaGraphRunner"),
        ):
            with self.subTest(runner=cls):
                body = _body(_find(path, f"{cls}.capture"))
                self.assertEqual(len(body), 1)
                self.assertIsInstance(body[0], ast.With)
                inner = ast.unparse(body[0].body[0])
                self.assertEqual(inner, "self._capture_graphs()")
                graphs = _find(path, f"{cls}._capture_graphs")
                self.assertTrue(_call_lines(graphs, "self.warmup"))

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

    def test_call_sites_pass_values_not_objects(self):
        """Every call hands Foundry the exact values it needs; no runner or
        backend object crosses the boundary."""
        for path, qualname, expected in _CALL_ARGS:
            func = ast.unparse(ast.parse(expected, mode="eval").body.func)
            with self.subTest(site=qualname, call=func):
                calls = [
                    ast.unparse(n)
                    for n in ast.walk(_find(path, qualname))
                    if isinstance(n, ast.Call) and ast.unparse(n.func) == func
                ]
                self.assertIn(expected, calls)
                for call in calls:
                    self.assertNotRegex(call, r"\(self[,)]|[=,] ?self[,)]")


if __name__ == "__main__":
    unittest.main()
