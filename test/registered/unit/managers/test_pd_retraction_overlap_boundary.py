"""PD offload must save the same boundary in CPU metadata and GPU state."""

import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang.srt.managers.scheduler as scheduler_module
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class Batch:
    def __init__(self, events, enough):
        self.events = events
        self.enough = enough
        self.reqs = [
            SimpleNamespace(
                output_ids=list(range(100)), is_retracted=False, finished=False
            )
            for _ in range(2)
        ]
        self.batch_is_full = True
        self.snapshots = []

    def batch_size(self):
        return len(self.reqs)

    def filter_batch(self):
        self.reqs = [req for req in self.reqs if not req.finished]

    def is_empty(self):
        return not self.reqs

    def check_decode_mem(self):
        return self.enough

    def retract_decode(self):
        req = self.reqs.pop()
        self.snapshots.append(list(req.output_ids))
        self.events.append("offload")
        req.is_retracted = True
        return [req], 0.5, []

    def prepare_for_decode(self):
        self.events.append("prepare")


class RetractionOverlapBoundaryTest(CustomTestCase):
    def setup_case(
        self,
        *,
        forced=True,
        enough=True,
        mode=DisaggregationMode.DECODE,
        overlap=True,
        completed=0,
        free_on_result=False,
    ):
        self.enterContext(patch.object(scheduler_module, "TEST_RETRACT", forced))
        self.enterContext(patch.object(scheduler_module, "TEST_RETRACT_INTERVAL", 16))
        events = []
        batch = Batch(events, enough)
        pending = list(batch.reqs)
        queue = deque([(pending, [100, 101, 102, 103])])

        def process_batch_result(reqs, accepted):
            events.append("settle")
            for index, req in enumerate(reqs):
                # Matches output processing's intentional retracted-request guard.
                if not req.is_retracted:
                    req.output_ids.extend(accepted)
                    req.finished = index < completed
            if free_on_result:
                batch.enough = True

        # A real instance, so helper methods update_running_batch calls resolve.
        scheduler = Scheduler.__new__(Scheduler)
        vars(scheduler).update(
            enable_hierarchical_cache=False,
            enable_overlap=overlap,
            disaggregation_mode=mode,
            forward_ct=16,
            result_queue=queue,
            last_batch=pending,
            process_batch_result=process_batch_result,
            token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda: 1024),
            new_token_ratio_tracker=SimpleNamespace(
                current=0.5, decay_step=lambda: events.append("decay")
            ),
            tree_cache=SimpleNamespace(req_to_token_pool=SimpleNamespace()),
            metrics_reporter=SimpleNamespace(enable_metrics=False),
            server_args=SimpleNamespace(),
            _add_request_to_queue=lambda req, is_retracted: events.append("enqueue"),
        )
        return scheduler, batch, events

    def test_pending_accepted_block_is_settled_before_pd_snapshot(self):
        for forced, enough in ((True, True), (False, False)):
            with self.subTest(forced=forced):
                scheduler, batch, events = self.setup_case(forced=forced, enough=enough)
                Scheduler.update_running_batch(scheduler, batch)
                self.assertEqual(batch.snapshots, [list(range(104))])
                self.assertLess(events.index("settle"), events.index("offload"))
                self.assertFalse(scheduler.result_queue)
                self.assertIsNone(scheduler.last_batch)

    def test_settlement_filters_completions_and_rechecks_capacity(self):
        for completed in (0, 1, 2):
            with self.subTest(completed=completed):
                scheduler, batch, events = self.setup_case(
                    forced=False, enough=False, completed=completed, free_on_result=True
                )
                self.assertIs(Scheduler.update_running_batch(scheduler, batch), batch)
                self.assertEqual(batch.batch_size(), 2 - completed)
                self.assertEqual(
                    [len(req.output_ids) for req in batch.reqs], [104] * (2 - completed)
                )
                self.assertEqual(batch.snapshots, [])
                self.assertEqual(
                    events,
                    ["settle"] if completed == 2 else ["settle", "decay", "prepare"],
                )
                self.assertIsNone(scheduler.last_batch)
                if completed:
                    self.assertFalse(batch.batch_is_full)

    def test_other_steps_preserve_pending_results(self):
        for mode, overlap, forced in (
            (DisaggregationMode.DECODE, True, False),
            (DisaggregationMode.NULL, True, True),
            (DisaggregationMode.DECODE, False, True),
        ):
            with self.subTest(mode=mode, overlap=overlap, forced=forced):
                scheduler, batch, events = self.setup_case(
                    mode=mode, overlap=overlap, forced=forced
                )
                Scheduler.update_running_batch(scheduler, batch)
                self.assertNotIn("settle", events)
                self.assertEqual(len(scheduler.result_queue), 1)
                self.assertIsNotNone(scheduler.last_batch)
                if not forced:
                    self.assertEqual(events, ["decay", "prepare"])


if __name__ == "__main__":
    unittest.main()
