"""Unit tests for the Foundry adapter, against a fake Foundry api, and its call sites;
Foundry's archive is valid only while the call sites keep their order."""

import ast
import io
import sys
import types
import unittest
from contextlib import contextmanager, redirect_stderr
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

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

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
    # ``unsupported_at``: the api function that raises ``Unsupported``.
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


def _graph_sa(**fields):
    # The graph fields Foundry pins, at SGLang's defaults unless given.
    defaults = dict(
        cuda_graph_backend_decode=None,
        cuda_graph_backend_prefill=None,
        enable_profile_cuda_graph=False,
        disable_flashinfer_autotune=False,
    )
    return SimpleNamespace(**{**defaults, **fields})


def _hook_sa(**fields):
    # What the persistence hooks read of a record, at SGLang's defaults unless given.
    defaults = dict(
        vars(_graph_sa()),
        cuda_graph_persistence="save",
        cuda_graph_persistence_config=None,
        disable_cuda_graph=False,
        disable_decode_cuda_graph=False,
        enable_unified_memory=False,
    )
    return SimpleNamespace(**{**defaults, **fields})


def _pin_lines(api, server_args):
    # The ``[Foundry] pin:`` lines the adapter prints while pinning ``server_args``.
    with _installed(api), redirect_stderr(io.StringIO()) as err:
        activate_foundry(_sa("save")).pin_server_args(server_args)
    return [
        line
        for line in err.getvalue().splitlines()
        if line.startswith("[Foundry] pin:")
    ]


@contextmanager
def _installed(api):
    # Makes ``api`` importable as foundry.integration.sglang.api; fresh adapter.
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


def _drive(adapter, body=lambda: None):
    # Every adapter method in process order, ``body`` inside the capture scope;
    # returns what the pool-config replay and capture_one gave.
    sa = _graph_sa()
    adapter.pin_server_args(sa)
    adapter.validate_resolved_server_args(sa)
    with adapter.configure_subprocess(sa):
        pass
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


class TestFoundryAdapter(CustomTestCase):
    def test_noop_without_the_flag(self):
        with mock.patch.object(foundry_adapter, "_active", foundry_adapter._NOOP):
            adapter = activate_foundry(_sa())
            self.assertIs(adapter, get_foundry_adapter())
            self.assertFalse(adapter.enabled)
            self.assertEqual(_drive(adapter), (None, None))

    def test_flag_without_foundry_raises_with_install_hint(self):
        with (
            mock.patch.dict(sys.modules, {"foundry": None}),
            mock.patch.object(foundry_adapter, "_active", foundry_adapter._NOOP),
            self.assertRaisesRegex(ImportError, r"sglang\[foundry\]"),
        ):
            activate_foundry(_sa("save"))

    def test_api_version_mismatch_is_refused(self):
        for major in (
            FOUNDRY_INTEGRATION_API_MAJOR + 1,
            FOUNDRY_INTEGRATION_API_MAJOR - 1,
        ):
            with self.subTest(major=major):
                api = _fake_api(version=(major, FOUNDRY_INTEGRATION_API_MIN_MINOR))
                with (
                    _installed(api),
                    self.assertRaisesRegex(RuntimeError, "is not supported"),
                ):
                    activate_foundry(_sa("save"))
                self.assertEqual(api.calls, [])

    def test_record_without_the_flag_in_an_enabled_process_is_refused(self):
        """Foundry's hook, region and env pins cannot be undone in a live process;
        a later record without the flag must not run there."""
        from sglang.srt.arg_groups.cuda_graph_hook import (
            validate_cuda_graph_persistence,
        )

        api = _fake_api()
        with _installed(api):
            enabled = activate_foundry(_sa("save"))
            validate_cuda_graph_persistence(_sa())
            self.assertNotIn(
                "validate_resolved_server_args", [name for name, *_ in api.calls]
            )
            with self.assertRaisesRegex(RuntimeError, "cannot be undone"):
                activate_foundry(_sa())
            self.assertIs(get_foundry_adapter(), enabled)
            self.assertIs(activate_foundry(_sa("save")), enabled)

    def test_every_adapter_method_reaches_the_api(self):
        """A method without a real override would silently no-op with the flag set."""
        api = _fake_api()
        with _installed(api):
            adapter = activate_foundry(_sa("save"))
            api.calls.clear()
            _drive(adapter)
        methods = {
            name
            for name, value in vars(FoundryAdapter).items()
            if callable(value) and not name.startswith("_")
        }
        self.assertEqual({name for name, _, _ in api.calls}, methods)

    def test_unsupported_config_falls_back_natively(self):
        """A config resolution rejects never activates Foundry, whose pins stay for good;
        after Foundry's ``Unsupported`` at any step, later calls no-op, capture is native."""
        from sglang.srt.arg_groups.cuda_graph_hook import (
            handle_cuda_graph_persistence,
            validate_cuda_graph_persistence,
        )
        from sglang.srt.arg_groups.model_override_base import resolving_view

        for field, value in (
            ("disable_cuda_graph", True),
            ("disable_decode_cuda_graph", True),
            ("cuda_graph_backend_decode", "disabled"),
            ("enable_unified_memory", True),
        ):
            with self.subTest(rejected_at_resolution=field):
                api = _fake_api()
                server_args = _hook_sa(**{field: value})
                with _installed(api):
                    handle_cuda_graph_persistence(server_args)
                    validate_cuda_graph_persistence(server_args)
                    # A spawned process activates from the same record.
                    self.assertFalse(activate_foundry(server_args).enabled)
                self.assertEqual(api.calls, [])
                self.assertIsNone(resolving_view(server_args).cuda_graph_persistence)
        api = _fake_api()
        with _installed(api):
            handle_cuda_graph_persistence(_hook_sa())
        self.assertIn("activate", [name for name, *_ in api.calls])

        before_spawn = ("activate", "pin_server_args", "validate_resolved_server_args")
        in_scope = ("capture_scope", "capture_one")
        for step in (
            *before_spawn,
            "before_parallel_init",
            "replay_saved_memory_pool_config",
            *in_scope,
        ):
            with self.subTest(step=step):
                api = _fake_api(unsupported_at=step)
                loops = []
                with _installed(api):
                    adapter = activate_foundry(_sa("save"))
                    replayed, captured = _drive(
                        adapter=adapter, body=lambda: loops.append(1)
                    )
                    self.assertFalse(adapter.enabled)
                    self.assertIs(activate_foundry(_sa("save")), adapter)
                self.assertEqual(loops, [1])
                self.assertIsNone(captured)
                self.assertEqual(replayed, "cfg" if step in in_scope else None)
                # Foundry is told once; spawns still pass its wrapper to tell the children.
                self.assertEqual(
                    [kw for name, _, kw in api.calls if name == "deactivate"],
                    [{"step": step, "reason": "fake reason", "quiet": False}],
                )
                names = [name for name, _, _ in api.calls]
                after = names[names.index("deactivate") + 1 :]
                self.assertEqual(
                    after, ["configure_subprocess"] if step in before_spawn else []
                )

    def test_pin_line_per_field_foundry_changed(self):
        """Foundry declares the graph pins without a log line of its own."""
        from sglang.srt.arg_groups.overrides import declare_resolution

        api = _fake_api()
        api.pin_server_args = lambda server_args: declare_resolution(
            server_args,
            "foundry_test",
            cuda_graph_backend_decode="full",
            enable_profile_cuda_graph=False,
            disable_flashinfer_autotune=True,
        )
        self.assertEqual(
            _pin_lines(api=api, server_args=_graph_sa()),
            [
                "[Foundry] pin: cuda_graph_backend_decode='full' (was None)",
                "[Foundry] pin: disable_flashinfer_autotune=True (was False)",
            ],
        )

    def test_no_pin_line_when_nothing_changed(self):
        from sglang.srt.arg_groups.overrides import declare_resolution

        same = _fake_api()
        same.pin_server_args = lambda server_args: declare_resolution(
            server_args, "foundry_test", cuda_graph_backend_decode="full"
        )
        for name, api, server_args in (
            ("nothing declared", _fake_api(), _graph_sa()),
            ("value already set", same, _graph_sa(cuda_graph_backend_decode="full")),
            ("unsupported", _fake_api(unsupported_at="pin_server_args"), _graph_sa()),
        ):
            with self.subTest(case=name):
                self.assertEqual(_pin_lines(api=api, server_args=server_args), [])

    def test_other_foundry_errors_propagate(self):
        """Only ``Unsupported`` falls back; a Foundry bug must stay an error."""
        api = _fake_api()
        api.capture_one = mock.Mock(side_effect=RuntimeError("restore failed"))
        with _installed(api):
            adapter = activate_foundry(_sa("load"))
            with self.assertRaisesRegex(RuntimeError, "restore failed"):
                adapter.capture_one(
                    shape_key="key", forward_fn=None, pool="pool", stream="stream"
                )
            self.assertTrue(adapter.enabled)


_KV = (
    "mem_cache/kv_cache_configurator.py",
    "KVCacheConfigurator._resolve_memory_pool_config",
)
_SCHEDULER = ("managers/scheduler.py", "run_scheduler_process")
_ENGINE = ("entrypoints/engine.py", "Engine._launch_scheduler_processes")
_PIPELINE = ("arg_groups/pipeline.py", "run_resolution_pipeline")
_CAPTURES = [
    (
        "model_executor/runner/decode_cuda_graph_runner.py",
        "DecodeCudaGraphRunner.capture",
    ),
    (
        "model_executor/runner/prefill_cuda_graph_runner.py",
        "PrefillCudaGraphRunner.capture",
    ),
]
# (file, function, first, then): in the function's source, ``then`` follows ``first``.
_ORDER = [
    ("distributed/bootstrap.py", "init_parallel_runtime",
     "get_foundry_adapter().before_parallel_init(", "_resolve_backend("),
    (*_KV, "get_foundry_adapter().replay_saved_memory_pool_config(",
     "self._profile_available_bytes("),
    (*_KV, "self._profile_available_bytes(",
     "get_foundry_adapter().record_memory_pool_config("),
    *((*site, "get_foundry_adapter().capture_scope(", "self._capture()")
      for site in _CAPTURES),
    ("model_executor/runner_backend/full_cuda_graph_backend.py",
     "FullCudaGraphBackend.capture_one", "foundry.capture_one(", "forward_fn()"),
    (*_SCHEDULER, "activate_foundry(server_args)", "publish("),
    (*_SCHEDULER, "Scheduler(server_args, port_args)",
     "get_foundry_adapter().log_init_summary()"),
    (*_SCHEDULER, "get_foundry_adapter().log_init_summary()", "pipe_writer.send("),
    (*_ENGINE, "activate_foundry(server_args)", "proc.start()"),
    (*_PIPELINE, "run_hook(handle_cuda_graph_persistence",
     "run_hook(apply_inkling_prefill_cuda_graph_default"),
    (*_PIPELINE, "run_hook(handle_other_validations",
     "run_hook(validate_cuda_graph_persistence"),
]  # fmt: skip


def _source(relpath, qualname):
    text = (Path(sglang.__file__).parent / "srt" / relpath).read_text(encoding="utf-8")
    node = ast.parse(text)
    for part in qualname.split("."):
        node = next(
            n
            for n in node.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part
        )
    return ast.get_source_segment(text, node)


class TestFoundryCallSites(CustomTestCase):
    def test_call_sites_keep_their_order(self):
        """A call site dropped or moved across the work it brackets fails only at LOAD;
        that work: parallel init, memory profile, capture loop, first publish."""
        for path, qualname, first, then in _ORDER:
            with self.subTest(site=qualname, first=first, then=then):
                source = _source(relpath=path, qualname=qualname)
                at = source.find(first)
                self.assertGreaterEqual(at, 0, f"{qualname} lost {first}")
                self.assertGreater(source.find(then, at + 1), at)

    def test_every_scheduler_spawn_is_wrapped(self):
        """The DP controller's schedulers inherit the hook preload from its spawn."""
        source = _source(relpath=_ENGINE[0], qualname=_ENGINE[1])
        self.assertEqual(
            source.count("foundry_adapter.configure_subprocess("),
            source.count("proc.start()"),
        )


if __name__ == "__main__":
    unittest.main()
