from __future__ import annotations

import logging
import unittest

from kv_canary_runner_unit_utils import CanaryRunnerTestCase, make_config, make_runner

from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag
from sglang.srt.kv_canary.runner import canary_runner as runner_module
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="extra-a", runner_config="1-gpu-large")


class TestSelfUnitRunnerHealth(CanaryRunnerTestCase):
    def test_kernel_run_counter_watchdog_raises_on_zero(self) -> None:
        """Verify the kernel watchdog raises when counters stop advancing."""
        runner = make_runner(device=self.device)
        runner._step_counter = 1000
        runner._device_state.kernel_run_counters.zero_()
        runner._health_checker.step()
        runner._step_counter = 2000
        with self.assertRaises(RuntimeError):
            runner._health_checker.step()

    def test_kernel_run_counter_watchdog_ignores_sweep_when_sweep_is_disabled(
        self,
    ) -> None:
        """Verify the watchdog ignores disabled sweep counters."""
        config = make_config(sweep_interval=0)
        runner = make_runner(device=self.device, config=config)
        runner._device_state.kernel_run_counters.zero_()
        for tag in (
            CanaryLaunchTag.HEAD_K_FULL,
            CanaryLaunchTag.HEAD_V_FULL,
            CanaryLaunchTag.TAIL_K_FULL,
            CanaryLaunchTag.TAIL_V_FULL,
        ):
            runner._device_state.kernel_run_counters[tag.value] = 1

        runner._step_counter = 1000
        runner._health_checker.step()
        runner._step_counter = 2000
        runner._health_checker.step()

    def test_periodic_stats_log_every_n_step(self) -> None:
        """Verify periodic stats are logged at the configured interval."""
        config = make_config(stats_print_every_n_steps=5)
        runner = make_runner(device=self.device, config=config)
        runner._device_state.slot_run_counters.fill_(7)

        with self.assertLogs(runner_module.logger.name, level=logging.INFO) as cm:
            for _ in range(11):
                runner._stats_logger.step()
                runner._step_counter += 1
        log_text = "\n".join(cm.output)
        self.assertIn("protected_tokens=", log_text)
        self.assertTrue("step=5" in log_text or "step=10" in log_text)


if __name__ == "__main__":
    unittest.main()
