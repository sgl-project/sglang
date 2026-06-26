from __future__ import annotations

import logging
import unittest
from unittest.mock import Mock

import torch

from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag
from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.runner import stats_logger as stats_logger_module
from sglang.srt.kv_canary.runner.health_checker import KernelRunCounterHealthChecker
from sglang.srt.kv_canary.state import CanaryDeviceState
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kv_canary.runner_test_base import (
    CanaryManagerTestCase,
    make_config,
    make_manager,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=45, suite="extra-a-test-1-gpu-small-amd")


class TestSelfUnitManagerHealth(CanaryManagerTestCase):
    def test_kernel_run_counter_watchdog_raises_on_zero(self) -> None:
        """Verify the kernel watchdog raises when counters stop advancing."""
        manager = make_manager(device=self.device)
        manager._outer_step_counter = 1000
        manager._device_state.kernel_run_counters.zero_()
        manager._health_checker.step()
        manager._outer_step_counter = 2000
        with self.assertRaises(RuntimeError):
            manager._health_checker.step()

    def test_kernel_run_counter_watchdog_ignores_sweep_when_sweep_is_disabled(
        self,
    ) -> None:
        """Verify the watchdog ignores disabled sweep counters."""
        config = make_config(sweep_interval=0)
        manager = make_manager(device=self.device, config=config)
        manager._device_state.kernel_run_counters.zero_()
        for tag in (
            CanaryLaunchTag.HEAD_K_FULL,
            CanaryLaunchTag.HEAD_V_FULL,
            CanaryLaunchTag.TAIL_K_FULL,
            CanaryLaunchTag.TAIL_V_FULL,
        ):
            manager._device_state.kernel_run_counters[tag.value] = 1

        manager._outer_step_counter = 1000
        manager._health_checker.step()
        manager._outer_step_counter = 2000
        manager._health_checker.step()

    def test_periodic_stats_log_every_n_step(self) -> None:
        """Verify periodic stats are logged at the configured interval."""
        config = make_config(stats_print_every_n_steps=5)
        manager = make_manager(device=self.device, config=config)
        manager._device_state.slot_run_counters.fill_(7)

        with self.assertLogs(stats_logger_module.logger.name, level=logging.INFO) as cm:
            for _ in range(11):
                manager._stats_logger.step()
                manager._outer_step_counter += 1
        log_text = "\n".join(cm.output)
        self.assertIn("protected_tokens=", log_text)
        self.assertTrue("step=5" in log_text or "step=10" in log_text)


class TestKernelRunCounterDeltaCheck(CustomTestCase):
    """Pure host-side regression tests for the watchdog's delta semantics."""

    def _make_checker(
        self,
        *,
        active_tags: tuple[CanaryLaunchTag, ...],
        outer_step: int,
    ) -> KernelRunCounterHealthChecker:
        config = Mock(spec=CanaryConfig)
        config.sweep_interval = 0
        num_tags = len(CanaryLaunchTag)
        device_state = Mock(spec=CanaryDeviceState)
        device_state.kernel_run_counters = torch.zeros(num_tags, dtype=torch.int64)
        return KernelRunCounterHealthChecker(
            config=config,
            device_state=device_state,
            active_tags=active_tags,
            outer_step_counter_getter=lambda: outer_step,
            d2h_stream=Mock(),
        )

    def _host_tensor(self, value: int) -> torch.Tensor:
        return torch.full((len(CanaryLaunchTag),), value, dtype=torch.int64)

    def test_first_check_raises_when_counter_never_incremented(self) -> None:
        active_tags = (CanaryLaunchTag.HEAD_K_FULL, CanaryLaunchTag.TAIL_K_FULL)
        checker = self._make_checker(active_tags=active_tags, outer_step=100)

        host_counters = torch.zeros(len(CanaryLaunchTag), dtype=torch.int64)
        with self.assertRaises(RuntimeError) as cm:
            checker._postprocess_on_host(host_counters)
        message = str(cm.exception)
        self.assertIn(CanaryLaunchTag.HEAD_K_FULL.name, message)
        self.assertIn(CanaryLaunchTag.TAIL_K_FULL.name, message)
        self.assertIn("did not increase", message)

    def test_second_check_raises_when_delta_is_zero(self) -> None:
        active_tags = (CanaryLaunchTag.HEAD_K_FULL,)
        checker = self._make_checker(active_tags=active_tags, outer_step=100)

        checker._postprocess_on_host(self._host_tensor(1))
        with self.assertRaises(RuntimeError) as cm:
            checker._postprocess_on_host(self._host_tensor(1))
        message = str(cm.exception)
        self.assertIn(CanaryLaunchTag.HEAD_K_FULL.name, message)
        self.assertIn("did not increase", message)

    def test_no_raise_when_counter_increases(self) -> None:
        active_tags = (CanaryLaunchTag.HEAD_K_FULL, CanaryLaunchTag.TAIL_K_FULL)
        checker = self._make_checker(active_tags=active_tags, outer_step=100)

        for value in (1, 5, 12, 100):
            checker._postprocess_on_host(self._host_tensor(value))


if __name__ == "__main__":
    unittest.main()
