import time
import unittest
from collections import deque
from types import SimpleNamespace

from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core import Req


class TestSchedulerARPrefetch(unittest.TestCase):
    def _scheduler(self, *, can_prefetch):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._sequential_prefetch_enabled = lambda: True
        scheduler._can_dynamic_batch = lambda _base, _candidate: True
        scheduler._batching_max_size = 4
        scheduler._batching_delay_s = 0.0
        scheduler._batch_metrics_enabled = False
        scheduler._batch_admission = SimpleNamespace(
            batch_is_full=lambda _reqs: False,
            reject_reason_for_candidate=lambda _reqs, _candidate: None,
            limit_reason_for_batch=lambda _reqs: None,
            max_admissible_batch_size=lambda _req: 4,
        )
        scheduler._record_batch_dispatch_metrics = lambda **_kwargs: None
        scheduler.worker = SimpleNamespace(
            can_prepare_forward_sequential_group=can_prefetch
        )
        return scheduler

    def _req(self, *, num_outputs_per_prompt: int):
        return Req(
            request_id=f"req-{num_outputs_per_prompt}",
            prompt="A simple product sketch",
            image_path=None,
            height=1024,
            width=1024,
            num_outputs_per_prompt=num_outputs_per_prompt,
        )

    def _enabled_scheduler(self, **server_arg_overrides):
        pipeline_config = SimpleNamespace(
            supports_native_grouped_requests=lambda: True,
            supports_sequential_dit_inference=lambda: True,
            supports_async_ar_prefetch=lambda: True,
        )
        server_args = SimpleNamespace(
            enable_ar_dit_overlap=True,
            num_gpus=1,
            dp_size=1,
            tp_size=1,
            sp_degree=1,
            enable_cfg_parallel=False,
            pipeline_config=pipeline_config,
        )
        for key, value in server_arg_overrides.items():
            setattr(server_args, key, value)

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.server_args = server_args
        scheduler.receiver = object()
        scheduler._dynamic_batching_enabled = lambda: True
        return scheduler

    def test_prefetch_enabled_for_single_rank_dit(self):
        scheduler = self._enabled_scheduler()

        self.assertTrue(scheduler._sequential_prefetch_enabled())

    def test_prefetch_disabled_for_multi_rank_dit(self):
        scheduler = self._enabled_scheduler(num_gpus=2)

        self.assertFalse(scheduler._sequential_prefetch_enabled())

    def test_multi_output_request_is_not_removed_for_ar_prefetch(self):
        req = self._req(num_outputs_per_prompt=2)
        scheduler = self._scheduler(
            can_prefetch=lambda batch: all(
                request.num_outputs_per_prompt == 1 for request in batch
            )
        )
        scheduler.waiting_queue = deque([(b"id", req, time.monotonic())])

        items = scheduler._get_next_sequential_prefetch_items()

        self.assertIsNone(items)
        self.assertEqual(len(scheduler.waiting_queue), 1)
        self.assertIs(scheduler.waiting_queue[0][1], req)

    def test_partial_prefetch_batch_waits_for_batching_delay(self):
        req = self._req(num_outputs_per_prompt=1)
        scheduler = self._scheduler(can_prefetch=lambda _batch: True)
        scheduler._batching_delay_s = 0.1
        scheduler.waiting_queue = deque([(b"id", req, time.monotonic())])

        items = scheduler._get_next_sequential_prefetch_items()

        self.assertIsNone(items)
        self.assertEqual(len(scheduler.waiting_queue), 1)
        self.assertIs(scheduler.waiting_queue[0][1], req)

    def test_partial_prefetch_batch_dispatches_after_batching_delay(self):
        req = self._req(num_outputs_per_prompt=1)
        scheduler = self._scheduler(can_prefetch=lambda _batch: True)
        scheduler._batching_delay_s = 0.1
        scheduler.waiting_queue = deque([(b"id", req, time.monotonic() - 0.2)])

        items = scheduler._get_next_sequential_prefetch_items()

        self.assertEqual(items, [(b"id", req)])
        self.assertEqual(len(scheduler.waiting_queue), 0)


if __name__ == "__main__":
    unittest.main()
