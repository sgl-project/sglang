import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.observability.scheduler_stage_metrics import (
    FORWARD_OVERLAP_FULL,
    FORWARD_OVERLAP_NONE,
    FORWARD_OVERLAP_PARTIAL,
    SCHEDULER_STAGE_GET_NEXT_BATCH,
    SCHEDULER_STAGE_PROCESS_QUEUE,
    SCHEDULER_STAGE_PROCESS_REQUESTS,
    SCHEDULER_STAGE_RUN_BATCH,
    SchedulerStageMetricsRecorder,
    scheduler_stage_method,
)
from sglang.srt.utils.device_timer import DeviceTimer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class TestSchedulerStageMetricsRecorder(CustomTestCase):
    def test_nested_stages_are_exclusive(self):
        # Each exclusive span uses its own endpoints, including the resumed parent.
        active = iter([False, False, True, True, False, False])
        recorder = SchedulerStageMetricsRecorder(
            enabled=True, query_forward_active=lambda: next(active)
        )
        recorder.start(wall_ns=0)

        with patch(
            "sglang.srt.observability.scheduler_stage_metrics.time.monotonic_ns",
            side_effect=[10, 30, 50, 80],
        ):
            outer = recorder.enter(SCHEDULER_STAGE_GET_NEXT_BATCH)
            inner = recorder.enter(SCHEDULER_STAGE_PROCESS_QUEUE)
            recorder.exit(inner)
            recorder.exit(outer)

        wall_ns = recorder.drain(wall_ns=100)

        self.assertEqual(
            wall_ns,
            {
                ("other", FORWARD_OVERLAP_NONE): 30,
                ("get_next_batch_to_run", FORWARD_OVERLAP_PARTIAL): 50,
                ("process_queue", FORWARD_OVERLAP_FULL): 20,
            },
        )
        self.assertEqual(sum(wall_ns.values()), 100)

    def test_decorator_restores_stage_after_exception(self):
        recorder = SchedulerStageMetricsRecorder(enabled=True)
        recorder.start(wall_ns=0)

        class SchedulerLike:
            scheduler_stage_metrics = recorder

            @scheduler_stage_method(SCHEDULER_STAGE_RUN_BATCH)
            def fail(self):
                raise RuntimeError("boom")

        with (
            patch(
                "sglang.srt.observability.scheduler_stage_metrics.time.monotonic_ns",
                side_effect=[10, 40],
            ),
            self.assertRaisesRegex(RuntimeError, "boom"),
        ):
            SchedulerLike().fail()

        wall_ns = recorder.drain(wall_ns=50)
        self.assertEqual(
            wall_ns,
            {
                ("other", FORWARD_OVERLAP_NONE): 20,
                ("run_batch", FORWARD_OVERLAP_NONE): 30,
            },
        )

    def test_nested_same_stage_does_not_double_count(self):
        recorder = SchedulerStageMetricsRecorder(enabled=True)
        recorder.start(wall_ns=0)

        with patch(
            "sglang.srt.observability.scheduler_stage_metrics.time.monotonic_ns",
            side_effect=[10, 40],
        ):
            with recorder.record(SCHEDULER_STAGE_RUN_BATCH):
                with recorder.record(SCHEDULER_STAGE_RUN_BATCH):
                    pass

        wall_ns = recorder.drain(wall_ns=50)
        self.assertEqual(
            wall_ns,
            {
                ("other", FORWARD_OVERLAP_NONE): 20,
                ("run_batch", FORWARD_OVERLAP_NONE): 30,
            },
        )

    def test_trace_spans_do_not_require_python_stacks(self):
        recorder = SchedulerStageMetricsRecorder(enabled=False)

        class SchedulerLike:
            scheduler_stage_metrics = recorder

            @scheduler_stage_method(SCHEDULER_STAGE_RUN_BATCH)
            def run(self):
                with self.scheduler_stage_metrics.record(SCHEDULER_STAGE_PROCESS_QUEUE):
                    pass

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            with_stack=False,
            acc_events=True,
        ) as profiler:
            SchedulerLike().run()

        stage_events = [
            event for event in profiler.events() if event.key.startswith("scheduler.")
        ]
        self.assertEqual(
            [event.key for event in stage_events],
            ["scheduler.run_batch", "scheduler.process_queue"],
        )
        self.assertTrue(all(not event.stack for event in stage_events))

    def test_forward_query_distinguishes_queued_active_and_completed_intervals(self):
        samples = []
        timer = DeviceTimer(lambda **sample: samples.append(sample))
        start, end, next_start, next_end = (
            Mock(query=Mock(return_value=False)) for _ in range(4)
        )
        start.elapsed_time.return_value = 2.0
        next_start.elapsed_time.return_value = 3.0
        with patch(
            "sglang.srt.utils.device_timer.torch.cuda.Event",
            side_effect=[start, end, next_start, next_end],
        ):
            self.assertFalse(timer.is_active())
            with timer.wrap(metadata={"category": "extend"}):
                self.assertFalse(timer.is_active())  # Queued, not started.
                start.query.return_value = True
                self.assertTrue(timer.is_active())  # End not recorded yet.
            self.assertEqual(samples, [])

            with timer.wrap(metadata={"category": "decode"}):
                # A later queued forward must not hide the earlier active one.
                self.assertTrue(timer.is_active())
                end.query.return_value = True
                self.assertFalse(timer.is_active())
                next_start.query.return_value = True
                self.assertTrue(timer.is_active())
            self.assertEqual(samples, [{"category": "extend", "t": 0.002}])
            next_end.query.return_value = True
            self.assertFalse(timer.is_active())
            timer._report()
            timer._report()
        self.assertEqual(
            samples,
            [
                {"category": "extend", "t": 0.002},
                {"category": "decode", "t": 0.003},
            ],
        )

    def test_decorator_preserves_existing_trace_names(self):
        recorder = SchedulerStageMetricsRecorder(enabled=False)

        class SchedulerLike:
            scheduler_stage_metrics = recorder

            @scheduler_stage_method(SCHEDULER_STAGE_PROCESS_REQUESTS)
            def process_input_requests(self):
                pass

            @scheduler_stage_method(SCHEDULER_STAGE_GET_NEXT_BATCH)
            def get_next_batch_to_run(self):
                pass

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            with_stack=False,
            acc_events=True,
        ) as profiler:
            scheduler = SchedulerLike()
            scheduler.process_input_requests()
            scheduler.get_next_batch_to_run()

        stage_events = [
            event for event in profiler.events() if event.key.startswith("scheduler.")
        ]
        self.assertEqual(
            [event.key for event in stage_events],
            [
                "scheduler.process_input_requests",
                "scheduler.get_next_batch_to_run",
            ],
        )


if __name__ == "__main__":
    unittest.main()
