"""CPU-only unit tests for serving-time Triton load diagnostics."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.utils import triton_load_watch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _fake_triton_knobs(*, hook=None, compilation=None):
    """Build importable ``triton``/``triton.knobs`` modules for install tests."""
    triton = types.ModuleType("triton")
    triton.__path__ = []
    knobs = types.ModuleType("triton.knobs")
    knobs.runtime = SimpleNamespace(kernel_load_start_hook=hook)
    knobs.compilation = compilation
    triton.knobs = knobs
    return {"triton": triton, "triton.knobs": knobs}


class TestTritonLoadWatch(CustomTestCase):
    def setUp(self):
        self._original_state = (
            triton_load_watch._installed,
            triton_load_watch._serving_started,
            triton_load_watch._prev_compile_listener,
        )
        triton_load_watch._installed = False
        triton_load_watch._serving_started = False
        triton_load_watch._prev_compile_listener = None

    def tearDown(self):
        (
            triton_load_watch._installed,
            triton_load_watch._serving_started,
            triton_load_watch._prev_compile_listener,
        ) = self._original_state

    def test_install_registers_once_and_chains_existing_listener(self):
        hook = Mock()
        previous_listener = Mock()
        compilation = SimpleNamespace(listener=previous_listener)

        with patch.dict(
            sys.modules,
            _fake_triton_knobs(hook=hook, compilation=compilation),
        ):
            triton_load_watch.install()
            triton_load_watch.install()

        hook.add.assert_called_once_with(triton_load_watch._on_kernel_load)
        self.assertIs(compilation.listener, triton_load_watch._on_compilation)
        self.assertIs(triton_load_watch._prev_compile_listener, previous_listener)
        self.assertTrue(triton_load_watch._installed)

    def test_install_is_noop_when_triton_knobs_cannot_be_imported(self):
        real_import = __import__

        def import_without_triton_knobs(name, *args, **kwargs):
            if name == "triton.knobs":
                raise ImportError("triton knobs unavailable")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_triton_knobs):
            triton_load_watch.install()

        self.assertFalse(triton_load_watch._installed)
        self.assertIsNone(triton_load_watch._prev_compile_listener)

    def test_install_is_noop_when_required_hooks_are_unavailable(self):
        unsupported_cases = (
            (None, SimpleNamespace(listener=None)),
            (Mock(), None),
            (Mock(), SimpleNamespace()),
        )

        for hook, compilation in unsupported_cases:
            with self.subTest(hook=hook, compilation=compilation):
                triton_load_watch._installed = False
                with patch.dict(
                    sys.modules,
                    _fake_triton_knobs(hook=hook, compilation=compilation),
                ):
                    triton_load_watch.install()
                self.assertFalse(triton_load_watch._installed)

    def test_mark_serving_started_arms_diagnostics(self):
        triton_load_watch.mark_serving_started()
        self.assertTrue(triton_load_watch._serving_started)

    def test_compilation_chains_existing_listener(self):
        previous_listener = Mock()
        triton_load_watch._prev_compile_listener = previous_listener
        src = SimpleNamespace(name="cached_kernel")
        times = SimpleNamespace(total=100)

        triton_load_watch._on_compilation(
            src=src,
            metadata="metadata",
            metadata_group="group",
            times=times,
            cache_hit=True,
        )

        previous_listener.assert_called_once_with(
            src=src,
            metadata="metadata",
            metadata_group="group",
            times=times,
            cache_hit=True,
        )

    def test_compilation_warning_gates(self):
        cases = (
            # Startup compilation is expected.
            (False, False, 2_000_000, False),
            # A cache hit does not compile a new specialization.
            (True, True, 2_000_000, False),
            # Fast serving-time compilation stays below the threshold.
            (True, False, 999_999, False),
            # Equality is intentionally warning-worthy.
            (True, False, 1_000_000, True),
        )
        src = SimpleNamespace(name="late_kernel")

        with patch.object(
            triton_load_watch.envs.SGLANG_TRITON_SLOW_COMPILE_THRESHOLD_SECS,
            "get",
            return_value=1.0,
        ):
            for serving_started, cache_hit, total, should_warn in cases:
                with (
                    self.subTest(
                        serving_started=serving_started,
                        cache_hit=cache_hit,
                        total=total,
                    ),
                    patch.object(triton_load_watch.logger, "warning") as warning,
                ):
                    triton_load_watch._serving_started = serving_started
                    triton_load_watch._on_compilation(
                        src=src,
                        metadata=None,
                        metadata_group=None,
                        times=SimpleNamespace(total=total),
                        cache_hit=cache_hit,
                    )
                    self.assertEqual(warning.called, should_warn)

    def test_kernel_load_before_serving_does_not_query_cuda(self):
        with patch.object(triton_load_watch.torch.cuda, "is_available") as available:
            triton_load_watch._on_kernel_load(None, None, "startup_kernel", None, None)
        available.assert_not_called()

    def test_kernel_load_warns_only_below_memory_threshold(self):
        triton_load_watch._serving_started = True

        with (
            patch.object(
                triton_load_watch.envs.SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY,
                "get",
                return_value=False,
            ),
            patch.object(
                triton_load_watch.envs.SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB,
                "get",
                return_value=1.0,
            ),
            patch.object(
                triton_load_watch.torch.cuda, "is_available", return_value=True
            ),
            patch.object(
                triton_load_watch.torch.cuda, "current_device", return_value=3
            ),
            patch.object(triton_load_watch, "get_available_gpu_memory") as free_memory,
        ):
            for free_gb, should_warn in ((1.0, False), (0.5, True)):
                with (
                    self.subTest(free_gb=free_gb),
                    patch.object(triton_load_watch.logger, "warning") as warning,
                ):
                    free_memory.return_value = free_gb
                    triton_load_watch._on_kernel_load(
                        None, None, "late_kernel", None, None
                    )
                    self.assertEqual(warning.called, should_warn)

        free_memory.assert_called_with("cuda", 3, empty_cache=False)

    def test_kernel_load_query_failure_is_safe_unless_crash_mode_is_enabled(self):
        triton_load_watch._serving_started = True

        with (
            patch.object(
                triton_load_watch.torch.cuda, "is_available", return_value=True
            ),
            patch.object(
                triton_load_watch.torch.cuda, "current_device", return_value=0
            ),
            patch.object(
                triton_load_watch,
                "get_available_gpu_memory",
                side_effect=RuntimeError("driver error"),
            ),
            patch.object(triton_load_watch.logger, "debug") as debug,
            patch.object(triton_load_watch.logger, "warning") as warning,
        ):
            with patch.object(
                triton_load_watch.envs.SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY,
                "get",
                return_value=False,
            ):
                triton_load_watch._on_kernel_load(
                    None, None, "unknown_memory", None, None
                )
            warning.assert_not_called()

            with (
                patch.object(
                    triton_load_watch.envs.SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY,
                    "get",
                    return_value=True,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "unknown_memory.*free device mem: unknown",
                ),
            ):
                triton_load_watch._on_kernel_load(
                    None, None, "unknown_memory", None, None
                )

        self.assertEqual(debug.call_count, 2)


if __name__ == "__main__":
    unittest.main()
