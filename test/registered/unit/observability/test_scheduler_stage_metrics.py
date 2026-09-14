import unittest
from unittest.mock import patch

import torch

from sglang.srt.observability.scheduler_stage_metrics import (
    SCHEDULER_STAGE_CATEGORIES,
    SCHEDULER_STAGE_GET_NEXT_BATCH,
    SCHEDULER_STAGE_PROCESS_QUEUE,
    SCHEDULER_STAGE_PROCESS_REQUESTS,
    SCHEDULER_STAGE_RUN_BATCH,
    SchedulerStageMetricsRecorder,
    scheduler_stage_method,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class TestSchedulerStageMetricsRecorder(CustomTestCase):
    def test_category_names(self):
        self.assertEqual(
            set(SCHEDULER_STAGE_CATEGORIES),
            {
                "other",
                "recv_requests",
                "process_input_requests",
                "process_batch_result",
                "process_queue",
                "get_next_batch_to_run",
                "run_batch",
                "sanity_check_cache",
                "idle",
            },
        )

    def test_nested_stages_are_exclusive(self):
        recorder = SchedulerStageMetricsRecorder(enabled=True)
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
                "other": 30,
                "get_next_batch_to_run": 50,
                "process_queue": 20,
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
        self.assertEqual(wall_ns, {"other": 20, "run_batch": 30})

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
        self.assertEqual(wall_ns, {"other": 20, "run_batch": 30})

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
