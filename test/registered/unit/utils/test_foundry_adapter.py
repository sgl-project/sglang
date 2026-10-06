"""Unit tests for the Foundry adapter (--cuda-graph-persistence).

Covers: the no-op adapter without the flag, the process contract (a record
without the flag in a persistence-enabled process is an error), the install
error when the flag is set without Foundry, the integration-API version check,
and delegation of every adapter method to foundry.integration.sglang.api.
No GPU, no Foundry (a fake api module stands in for it).

Run:  python -m pytest test/registered/unit/utils/test_foundry_adapter.py -v
"""

import sys
import types
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

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
    api.capture_scope = lambda runner: scope("capture_scope", runner)

    def run_capture_loop(runner, loop_fn):
        api.calls.append(("run_capture_loop", (runner,), {}))
        return loop_fn()

    api.run_capture_loop = run_capture_loop
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
            with adapter.capture_scope(object()), adapter.configure_subprocess():
                pass
            # The capture loop runs exactly once without Foundry.
            runs = []
            self.assertEqual(
                adapter.run_capture_loop(object(), lambda: runs.append(1) or "r"), "r"
            )
            self.assertEqual(runs, [1])

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
            (FOUNDRY_INTEGRATION_API_MAJOR, FOUNDRY_INTEGRATION_API_MIN_MINOR - 1),
        ):
            with self.subTest(version=version):
                api = _fake_api(version=version)
                with (
                    _installed(api),
                    self.assertRaisesRegex(RuntimeError, "is not supported"),
                ):
                    FoundryAdapter.create(True, mode="save")
                self.assertEqual(api.calls, [])

    def test_record_without_the_flag_gets_the_noop_adapter(self):
        """A process that never enabled persistence keeps the no-op adapter."""
        api = _fake_api()
        with _installed(api):
            self.assertIs(activate_foundry(_sa()), foundry_adapter._NOOP)
            self.assertIs(get_foundry_adapter(), foundry_adapter._NOOP)
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
            sa, runner = object(), object()
            adapter.pin_server_args(sa)
            adapter.validate_graph_config(sa)
            adapter.validate_resolved_server_args(sa)
            adapter.apply_env_pins(sa)
            with adapter.configure_subprocess(sa):
                pass
            adapter.before_parallel_init("cuda")
            adapter.after_parallel_init()
            adapter.after_runner_distributed_init(runner)
            self.assertEqual(adapter.replay_saved_memory_pool_config(), "cfg")
            adapter.record_memory_pool_overrides()
            adapter.before_alloc_memory_pool(runner)
            adapter.after_alloc_memory_pool(runner)
            with adapter.capture_scope(runner):
                pass
            self.assertEqual(adapter.run_capture_loop(runner, lambda: "looped"), "looped")
            hook, group = object(), object()
            self.assertEqual(
                adapter.capture_one(
                    "key",
                    None,
                    pool="pool",
                    stream="stream",
                    prefill_req_slots=4,
                    post_warmup_hook=hook,
                    tp_group=group,
                ),
                ("graph", "out"),
            )
            name, args, kwargs = api.calls[-1]
            self.assertEqual((name, args), ("capture_one", ("key", None)))
            self.assertEqual(
                kwargs,
                {
                    "pool": "pool",
                    "stream": "stream",
                    "prefill_req_slots": 4,
                    "post_warmup_hook": hook,
                    "tp_group": group,
                },
            )
        called = [name for name, _, _ in api.calls]
        methods = [
            name
            for name, value in vars(FoundryAdapter).items()
            if callable(value) and not name.startswith("_") and name != "create"
        ]
        self.assertEqual(sorted(called), sorted(methods))


if __name__ == "__main__":
    unittest.main()
