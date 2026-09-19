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


class PendingBlock:
    def __init__(self):
        self.output_ids = list(range(100))
        self.origin_input_ids = [1] * 20
        self.is_retracted = False
        self.finished = False
        self.gpu_committed = 104


class Batch:
    def __init__(self, events, capacity):
        self.events = events
        self.capacity = capacity
        self.reqs = [PendingBlock(), PendingBlock()]
        self.batch_is_full = True
        self.snapshots = []

    def batch_size(self):
        return len(self.reqs)

    def filter_batch(self):
        self.reqs = [req for req in self.reqs if not req.finished]

    def is_empty(self):
        return not self.reqs

    def check_decode_mem(self):
        return self.capacity["enough"]

    def retract_decode(self):
        req = self.reqs.pop()
        self.snapshots.append((len(req.output_ids), req.gpu_committed))
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
        complete=False,
        free_on_result=False,
    ):
        method = Scheduler.update_running_batch
        self.enterContext(patch.object(scheduler_module, "TEST_RETRACT", forced))
        self.enterContext(patch.object(scheduler_module, "TEST_RETRACT_INTERVAL", 16))
        events = []
        capacity = {"enough": enough}
        batch = Batch(events, capacity)
        pending = list(batch.reqs)
        queue = deque([(pending, [100, 101, 102, 103])])

        def process_batch_result(reqs, accepted):
            events.append("settle")
            for index, req in enumerate(reqs):
                # Matches output processing's intentional retracted-request guard.
                if not req.is_retracted:
                    req.output_ids.extend(accepted)
                    req.finished = complete is True or (
                        complete == "first" and index == 0
                    )
            if free_on_result:
                capacity["enough"] = True

        scheduler = SimpleNamespace(
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
        return method, scheduler, batch, events

    def test_pending_accepted_block_is_settled_before_pd_snapshot(self):
        for forced, enough in ((True, True), (False, False)):
            with self.subTest(forced=forced):
                method, scheduler, batch, events = self.setup_case(
                    forced=forced, enough=enough
                )
                method(scheduler, batch)
                self.assertEqual(batch.snapshots, [(104, 104)])
                self.assertLess(events.index("settle"), events.index("offload"))
                self.assertFalse(scheduler.result_queue)
                self.assertIsNone(scheduler.last_batch)

    def test_settling_finished_requests_can_avoid_retraction(self):
        method, scheduler, batch, events = self.setup_case(
            forced=False, enough=False, free_on_result=True
        )
        method(scheduler, batch)
        self.assertEqual(batch.snapshots, [])
        self.assertEqual(events, ["settle", "decay", "prepare"])
        self.assertIsNone(scheduler.last_batch)

    def test_all_finished_during_settlement_returns_empty_batch(self):
        method, scheduler, batch, events = self.setup_case(
            forced=False, enough=False, complete=True
        )
        self.assertIs(method(scheduler, batch), batch)
        self.assertTrue(batch.is_empty())
        self.assertFalse(batch.batch_is_full)
        self.assertEqual(events, ["settle"])

    def test_partial_completion_is_filtered_before_memory_recheck(self):
        method, scheduler, batch, events = self.setup_case(
            forced=False, enough=False, complete="first", free_on_result=True
        )
        method(scheduler, batch)
        self.assertEqual(batch.batch_size(), 1)
        self.assertEqual(len(batch.reqs[0].output_ids), 104)
        self.assertEqual(batch.snapshots, [])
        self.assertIsNone(scheduler.last_batch)

    def test_healthy_pd_steps_keep_overlap(self):
        method, scheduler, batch, events = self.setup_case(forced=False)
        method(scheduler, batch)
        self.assertEqual(events, ["decay", "prepare"])
        self.assertEqual(len(scheduler.result_queue), 1)
        self.assertIsNotNone(scheduler.last_batch)

    def test_other_scheduler_modes_do_not_drain_here(self):
        for mode, overlap in (
            (DisaggregationMode.NULL, True),
            (DisaggregationMode.DECODE, False),
        ):
            with self.subTest(mode=mode, overlap=overlap):
                method, scheduler, batch, events = self.setup_case(
                    mode=mode, overlap=overlap
                )
                method(scheduler, batch)
                self.assertNotIn("settle", events)
                self.assertEqual(len(scheduler.result_queue), 1)


if __name__ == "__main__":
    unittest.main()
