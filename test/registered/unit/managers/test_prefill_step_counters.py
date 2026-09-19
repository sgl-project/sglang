"""Completed prefill bursts must contribute both busy time and tokens."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestPrefillStepCounters(CustomTestCase):
    def test_completed_bursts_count_time_and_tokens(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._prev_step = None
        scheduler._prev_prefill_end_ts = None
        scheduler.total_prefill_busy_us = 0
        scheduler.total_prefill_uncached_tokens = 0

        # Two overlapping batches, then an isolated batch after an idle gap.
        # Check every completion so the burst tail needs no subsequent launch.
        completions = [
            (0.0, 0.1875, True, 64, 187_500, 64),
            (0.0625, 0.25, False, 128, 250_000, 192),
            (1.0, 1.125, True, 256, 375_000, 448),
        ]
        for iteration, (launch, end, idle, tokens, busy_us, total_tokens) in enumerate(
            completions, start=1
        ):
            with self.subTest(completion=iteration):
                batch = ScheduleBatch(
                    reqs=[SimpleNamespace(rid=f"request-{iteration}")],
                    forward_mode=ForwardMode.EXTEND,
                    forward_iter=iteration,
                    launch_ts=launch,
                    after_idle_gap=idle,
                    extend_num_tokens=tokens,
                )
                with patch(
                    "sglang.srt.managers.scheduler.time.monotonic", return_value=end
                ):
                    scheduler._record_step_counters(batch, GenerationBatchResult())
                self.assertEqual(scheduler.total_prefill_busy_us, busy_us)
                self.assertEqual(scheduler.total_prefill_uncached_tokens, total_tokens)


if __name__ == "__main__":
    unittest.main()
