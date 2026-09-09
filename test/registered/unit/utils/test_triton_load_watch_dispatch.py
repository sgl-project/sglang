"""Accelerator-dispatch tests for triton_load_watch — CPU-only, fully mocked.

Kept separate from test_triton_load_watch.py, which is register_cuda_ci: that
file launches a real Triton kernel on ``device="cuda"`` and imports ``triton``
at module scope, so it cannot be collected on the CPU runner. The cases here
mock the memory query instead and need no accelerator.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.utils import triton_load_watch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestTritonLoadWatchAcceleratorDispatch(CustomTestCase):
    # The watch keeps process-global state shared with test_triton_load_watch.py,
    # so reset on entry as well as exit: a selective run, a --last-failed rerun or
    # a shuffling plugin can otherwise start mid-state and see no warning at all.
    def setUp(self):
        self._reset()

    def tearDown(self):
        self._reset()

    @staticmethod
    def _reset():
        triton_load_watch._serving_started = False
        triton_load_watch._unknown_memory_warning_emitted = False

    def test_late_xpu_load_uses_xpu_memory_and_neutral_warning(self):
        triton_load_watch._serving_started = True
        with (
            patch.object(
                torch.accelerator,
                "current_accelerator",
                return_value=torch.device("xpu"),
            ),
            patch.object(torch.accelerator, "current_device_index", return_value=3),
            patch.object(
                triton_load_watch,
                "get_available_gpu_memory",
                return_value=0.125,
            ) as get_memory,
            self.assertLogs(triton_load_watch.logger, level="WARNING") as logs,
        ):
            triton_load_watch._on_kernel_load(None, None, "probe", None, None)

        get_memory.assert_called_once_with("xpu", 3, empty_cache=False)
        self.assertIn("stall serving", logs.output[0])
        # Neutral text: no CUDA-specific wording on a non-CUDA accelerator.
        self.assertNotIn("CUDA", logs.output[0])

    def test_invalid_memory_query_contract_is_not_hidden(self):
        with (
            patch.object(
                torch.accelerator,
                "current_accelerator",
                return_value=torch.device("xpu"),
            ),
            patch.object(torch.accelerator, "current_device_index", return_value=0),
            patch.object(
                triton_load_watch,
                "get_available_gpu_memory",
                side_effect=AssertionError("invalid device type"),
            ),
            self.assertRaisesRegex(AssertionError, "invalid device type"),
        ):
            triton_load_watch._free_device_memory_gb()

    def test_accelerator_without_visible_devices_reports_unknown(self):
        # current_accelerator() names the compiled-in accelerator even when no
        # device is visible; the index query then raises. That has to degrade to
        # "unknown" rather than escape the Triton kernel-load hook.
        triton_load_watch._serving_started = True
        with (
            patch.object(
                torch.accelerator,
                "current_accelerator",
                return_value=torch.device("xpu"),
            ),
            patch.object(
                torch.accelerator,
                "current_device_index",
                side_effect=RuntimeError("No XPU devices are available."),
            ),
            self.assertLogs(triton_load_watch.logger, level="WARNING") as logs,
        ):
            triton_load_watch._on_kernel_load(None, None, "probe", None, None)

        self.assertIn("free device mem: unknown", logs.output[0])

    def test_runtime_memory_query_failure_warns_once_with_unknown_memory(self):
        triton_load_watch._serving_started = True
        with (
            patch.object(
                torch.accelerator,
                "current_accelerator",
                return_value=torch.device("xpu"),
            ),
            patch.object(torch.accelerator, "current_device_index", return_value=0),
            patch.object(
                triton_load_watch,
                "get_available_gpu_memory",
                side_effect=RuntimeError("query failed"),
            ),
            self.assertLogs(triton_load_watch.logger, level="WARNING") as logs,
        ):
            triton_load_watch._on_kernel_load(None, None, "first", None, None)
            triton_load_watch._on_kernel_load(None, None, "second", None, None)

        self.assertEqual(len(logs.output), 1)
        self.assertIn("free device mem: unknown", logs.output[0])


if __name__ == "__main__":
    unittest.main()
