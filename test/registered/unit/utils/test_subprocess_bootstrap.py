"""Unit tests for subprocess bootstrap helpers."""

import importlib
import pickle
import sys
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from sglang.srt.utils.subprocess_bootstrap import (
    DEFAULT_DATA_PARALLEL_CONTROLLER_TARGET,
    DEFAULT_DETOKENIZER_TARGET,
    DEFAULT_MULTI_DETOKENIZER_ROUTER_TARGET,
    DEFAULT_SCHEDULER_TARGET,
    get_subprocess_target_args,
    resolve_subprocess_target,
    run_subprocess_target,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSubprocessBootstrap(TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self._tempdir.name)
        self.module_names = [
            "bootstrap_test_state",
            "bootstrap_test_target",
        ]
        sys.path.insert(0, self._tempdir.name)

        (self.temp_path / "bootstrap_test_state.py").write_text(
            "events = []\n",
            encoding="utf-8",
        )
        (self.temp_path / "bootstrap_test_target.py").write_text(
            "from bootstrap_test_state import events\n"
            "events.append('target_import')\n"
            "not_callable = 1\n"
            "def target(*args, **kwargs):\n"
            "    events.append(('target_call', args, kwargs))\n"
            "    return {'args': args, 'kwargs': kwargs}\n"
            "def fail():\n"
            "    raise RuntimeError('target boom')\n",
            encoding="utf-8",
        )

    def tearDown(self):
        sys.path.remove(self._tempdir.name)
        for module_name in self.module_names:
            sys.modules.pop(module_name, None)
        self._tempdir.cleanup()

    def test_startup_plugins_run_before_target_import(self):
        state = importlib.import_module("bootstrap_test_state")

        def record_startup():
            state.events.append("startup")

        with patch(
            "sglang.srt.plugins.load_startup_plugins",
            side_effect=record_startup,
        ) as mock_load_startup_plugins:
            result = run_subprocess_target(
                "bootstrap_test_target:target", "arg", key="value"
            )

        mock_load_startup_plugins.assert_called_once()
        self.assertEqual(result, {"args": ("arg",), "kwargs": {"key": "value"}})
        self.assertEqual(
            state.events,
            [
                "startup",
                "target_import",
                ("target_call", ("arg",), {"key": "value"}),
            ],
        )

    def test_resolve_subprocess_target_invalid_string(self):
        for target in ["bootstrap_test_target", ":target", "bootstrap_test_target:"]:
            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError, "module:function"):
                    resolve_subprocess_target(target)

    def test_resolve_subprocess_target_rejects_non_callable(self):
        with self.assertRaisesRegex(TypeError, "not callable"):
            resolve_subprocess_target("bootstrap_test_target:not_callable")

    def test_target_exception_propagates(self):
        with patch("sglang.srt.plugins.load_startup_plugins"):
            with self.assertRaisesRegex(RuntimeError, "target boom"):
                run_subprocess_target("bootstrap_test_target:fail")

    def test_run_subprocess_target_is_pickleable(self):
        self.assertIs(
            pickle.loads(pickle.dumps(run_subprocess_target)), run_subprocess_target
        )

    def test_get_subprocess_target_args_wraps_string_target(self):
        target, args = get_subprocess_target_args(
            "bootstrap_test_target:target", "arg", 1
        )

        self.assertIs(target, run_subprocess_target)
        self.assertEqual(args, ("bootstrap_test_target:target", "arg", 1))

    def test_default_targets_are_module_function_strings(self):
        for target in [
            DEFAULT_DATA_PARALLEL_CONTROLLER_TARGET,
            DEFAULT_DETOKENIZER_TARGET,
            DEFAULT_MULTI_DETOKENIZER_ROUTER_TARGET,
            DEFAULT_SCHEDULER_TARGET,
        ]:
            with self.subTest(target=target):
                module_name, separator, function_name = target.partition(":")
                self.assertTrue(module_name)
                self.assertEqual(separator, ":")
                self.assertTrue(function_name)

    def test_get_subprocess_target_args_wraps_non_scheduler_targets(self):
        for target in [
            DEFAULT_DATA_PARALLEL_CONTROLLER_TARGET,
            DEFAULT_DETOKENIZER_TARGET,
            DEFAULT_MULTI_DETOKENIZER_ROUTER_TARGET,
        ]:
            with self.subTest(target=target):
                subprocess_target, args = get_subprocess_target_args(target, "arg")
                self.assertIs(subprocess_target, run_subprocess_target)
                self.assertEqual(args, (target, "arg"))

    def test_get_subprocess_target_args_preserves_callable_target(self):
        def target_func():
            return None

        target, args = get_subprocess_target_args(target_func, "arg", 1)

        self.assertIs(target, target_func)
        self.assertEqual(args, ("arg", 1))


if __name__ == "__main__":
    import unittest

    unittest.main()
