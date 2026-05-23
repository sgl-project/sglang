from __future__ import annotations

import logging
import unittest

from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag
from sglang.srt.kv_canary.runner import canary_manager as manager_module
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.runner_test_base import (
    CanaryManagerTestCase,
    make_config,
    make_manager,
)

register_cuda_ci(est_time=45, stage="extra-a", runner_config="1-gpu-large")


class TestSelfUnitManagerHealth(CanaryManagerTestCase):
    def test_kernel_run_counter_watchdog_raises_on_zero(self) -> None:
        """Verify the kernel watchdog raises when counters stop advancing."""
        manager = make_manager(device=self.device)
        manager._step_counter = 1000
        manager._device_state.kernel_run_counters.zero_()
        manager._health_checker.step()
        manager._step_counter = 2000
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

        manager._step_counter = 1000
        manager._health_checker.step()
        manager._step_counter = 2000
        manager._health_checker.step()

    def test_periodic_stats_log_every_n_step(self) -> None:
        """Verify periodic stats are logged at the configured interval."""
        config = make_config(stats_print_every_n_steps=5)
        manager = make_manager(device=self.device, config=config)
        manager._device_state.slot_run_counters.fill_(7)

        with self.assertLogs(manager_module.logger.name, level=logging.INFO) as cm:
            for _ in range(11):
                manager._stats_logger.step()
                manager._step_counter += 1
        log_text = "\n".join(cm.output)
        self.assertIn("protected_tokens=", log_text)
        self.assertTrue("step=5" in log_text or "step=10" in log_text)


if __name__ == "__main__":
    unittest.main()
