"""Prefill time partitions must include burst boundaries and pair with tokens."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from parameterized import parameterized

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _BeforeModelForward(Exception):
    pass


class TestPrefillBusyAccounting(CustomTestCase):
    def setUp(self):
        self.scheduler = Scheduler.__new__(Scheduler)
        self.scheduler._prev_step = None
        self.scheduler._prev_prefill_end_ts = None
        self.scheduler.total_prefill_busy_us = 0
        self.scheduler.total_prefill_uncached_tokens = 0
        self.scheduler.decode_moment_totals = [0.0] * 6

    def record(self, batch, end_ts):
        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=end_ts):
            self.scheduler._record_step_counters(batch, GenerationBatchResult())

    def batch(
        self, iteration, launch_ts, tokens, after_idle=False, mode=ForwardMode.EXTEND
    ):
        return ScheduleBatch(
            reqs=[SimpleNamespace(rid=f"request-{iteration}")],
            forward_mode=mode,
            forward_iter=iteration,
            launch_ts=launch_ts,
            after_idle_gap=after_idle,
            extend_num_tokens=tokens,
        )

    @parameterized.expand(
        [
            (
                "isolated",
                (0, 1, 2, 3),
                (0.125, 1.125, 2.125, 3.125),
                (True, True, True, True),
                (0.125, 0.25, 0.375, 0.5),
            ),
            (
                "overlapped_bursts",
                (0, 0.0625, 1, 1.0625),
                (0.1875, 0.25, 1.1875, 1.25),
                (False, False, True, False),
                (0.1875, 0.25, 0.4375, 0.5),
            ),
            (
                "overlapped_continuous",
                (0, 0.0625, 0.25, 0.3125),
                (0.1875, 0.25, 0.4375, 0.5),
                (False, False, False, False),
                (0.1875, 0.25, 0.4375, 0.5),
            ),
            (
                "scheduler_overhead",
                (0, 0.25, 0.5, 0.75),
                (0.125, 0.375, 0.625, 0.875),
                (False, False, False, False),
                (0.125, 0.375, 0.625, 0.875),
            ),
        ]
    )
    def test_complete_bursts(self, name, launches, ends, idle_flags, expected_busy):
        tokens = (64, 128, 256, 512)
        for i in range(len(tokens)):
            with self.subTest(completed_batches=i + 1):
                self.record(
                    self.batch(i + 1, launches[i], tokens[i], idle_flags[i]), ends[i]
                )
                # Assert after every completion: the final result must be counted
                # even if no subsequent batch ever arrives.
                self.assertEqual(
                    self.scheduler.total_prefill_busy_us, int(expected_busy[i] * 1e6)
                )
                self.assertEqual(
                    self.scheduler.total_prefill_uncached_tokens, sum(tokens[: i + 1])
                )
        self.assertEqual(self.scheduler.decode_moment_totals, [0.0] * 6)

    @parameterized.expand(
        [
            ("missing_iteration", None),
            ("decode", ForwardMode.DECODE),
            ("target_verify", ForwardMode.TARGET_VERIFY),
            ("idle_mode", ForwardMode.IDLE),
            ("health_check", ForwardMode.EXTEND),
        ]
    )
    def test_discontinuity_starts_at_own_launch(self, name, mode):
        self.record(self.batch(1, 0, 64), 0.25)
        if mode is not None:
            skipped = self.batch(2, 1, 1024, mode=mode)
            if name == "health_check":
                with patch(
                    "sglang.srt.managers.scheduler.is_health_check_generate_req",
                    return_value=True,
                ):
                    self.record(skipped, 1.125)
            else:
                self.record(skipped, 1.125)
        self.record(self.batch(3, 2, 128), 2.125)
        self.record(self.batch(4, 2.25, 256), 2.375)
        self.assertEqual(self.scheduler.total_prefill_busy_us, 625_000)
        self.assertEqual(self.scheduler.total_prefill_uncached_tokens, 448)

    def test_overlapping_mode_transition_does_not_count_time_twice(self):
        self.record(self.batch(1, 0, 64), 0.5)
        self.record(self.batch(2, 0.125, 0, mode=ForwardMode.DECODE), 0.625)
        self.record(self.batch(3, 0.25, 128), 0.75)
        self.assertEqual(self.scheduler.total_prefill_busy_us, 750_000)
        self.assertEqual(self.scheduler.total_prefill_uncached_tokens, 192)

    @parameterized.expand([("zero", 0.0), ("cap", 2.0), ("over_cap", 2.125)])
    def test_rejected_interval_drops_tokens_and_resets_boundary(self, name, end_ts):
        self.record(self.batch(1, 0, 64), end_ts)
        self.assertEqual(self.scheduler.total_prefill_busy_us, 0)
        self.assertEqual(self.scheduler.total_prefill_uncached_tokens, 0)
        self.record(self.batch(2, end_ts + 0.125, 128), end_ts + 0.25)
        self.assertEqual(self.scheduler.total_prefill_busy_us, 250_000)
        self.assertEqual(self.scheduler.total_prefill_uncached_tokens, 128)

    @parameterized.expand(
        [("decode", ForwardMode.DECODE), ("verify", ForwardMode.TARGET_VERIFY)]
    )
    def test_decode_keeps_launch_cadence_and_boundary_guards(self, name, mode):
        with patch(
            "sglang.srt.managers.scheduler.time.monotonic",
            side_effect=AssertionError("Decode must not use completion time"),
        ):
            for i, launch in enumerate((0, 0.125, 1, 1.125)):
                batch = self.batch(i + 1, launch, 0, after_idle=(i == 2), mode=mode)
                self.scheduler._record_step_counters(batch, GenerationBatchResult())
        self.assertEqual(self.scheduler.decode_moment_totals[0], 2)
        self.assertEqual(self.scheduler.decode_moment_totals[2], 250_000)
        self.assertEqual(self.scheduler.total_prefill_busy_us, 0)
        self.assertEqual(self.scheduler.total_prefill_uncached_tokens, 0)

    @parameterized.expand([("prefill_only", False), ("interleaved_decode", True)])
    def test_split_prefill_preserves_first_launch_and_idle_flag(
        self, name, interleaved
    ):
        scheduler = self.scheduler
        scheduler._sched_idled = True
        scheduler.forward_ct = 0
        scheduler.processed_tokens_counter = 0
        scheduler.scheduler_stage_metrics = None
        scheduler.metrics_reporter = SimpleNamespace(record_scheduler_active=Mock())
        scheduler.scripted_scheduler_hook = SimpleNamespace(
            on_run_batch=Mock(side_effect=_BeforeModelForward)
        )

        def launch(batch, split_index, launch_ts):
            batch.split_index = split_index
            with patch(
                "sglang.srt.managers.scheduler.time.monotonic", return_value=launch_ts
            ):
                with self.assertRaises(_BeforeModelForward):
                    Scheduler.run_batch(scheduler, batch)

        first = self.batch(0, 0, 64, mode=ForwardMode.SPLIT_PREFILL)
        for split_index, launch_ts in enumerate((0, 0.125, 0.25)):
            launch(first, split_index, launch_ts)
            self.assertTrue(first.after_idle_gap)
            self.assertFalse(scheduler._sched_idled)
        self.assertEqual(first.forward_iter, 3)
        self.assertEqual(first.launch_ts, 0.25)
        copied = first.copy()
        self.assertEqual(copied.split_prefill_start, (1, 0))
        self.record(copied, 0.5)
        self.assertEqual(scheduler.total_prefill_busy_us, 500_000)
        self.assertEqual(scheduler.total_prefill_uncached_tokens, 64)

        # The next split batch is contiguous at its FIRST slice. Preserve the
        # scheduling overhead between full results, not just the final slice.
        second = self.batch(0, 0, 128, mode=ForwardMode.SPLIT_PREFILL)
        launch(second, 0, 0.75)
        if interleaved:
            decode = self.batch(0, 0, 0, mode=ForwardMode.DECODE)
            launch(decode, 0, 0.8125)
            self.record(decode, 0.84375)
        launch(second, 1, 0.875)
        self.assertFalse(second.after_idle_gap)
        self.assertEqual(second.split_prefill_start, (4, 0.75))
        self.record(second, 1.0)
        self.assertEqual(
            scheduler.total_prefill_busy_us, 750_000 if interleaved else 1_000_000
        )
        self.assertEqual(scheduler.total_prefill_uncached_tokens, 192)


if __name__ == "__main__":
    unittest.main()
