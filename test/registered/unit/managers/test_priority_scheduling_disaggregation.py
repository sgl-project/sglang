import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.decode import (  # noqa: E402
    DecodePreallocQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.schedule_batch import FINISH_ABORT  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="stage-a", runner_config="1-gpu-small")


class TestDisaggregationPriorityQueueing(unittest.TestCase):
    def _new_scheduler(self, disaggregation_mode: DisaggregationMode) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = disaggregation_mode
        scheduler.enable_priority_scheduling = True
        scheduler.schedule_low_priority_values_first = False
        scheduler.abort_on_priority_when_disabled = False
        scheduler.waiting_queue = []
        scheduler._prefetch_kvcache = MagicMock()
        scheduler._abort_on_queued_limit = MagicMock(return_value=False)
        scheduler.model_config = SimpleNamespace(num_key_value_heads=8)
        scheduler.disagg_prefill_bootstrap_queue = MagicMock()
        scheduler.disagg_decode_prealloc_queue = MagicMock()
        scheduler.send_to_tokenizer = MagicMock()
        return scheduler

    def _new_req(self, priority=None):
        req = MagicMock()
        req.priority = priority
        req.rid = "req"
        req.time_stats = MagicMock()
        req.time_stats.trace_ctx = MagicMock()
        return req

    def test_prefill_mode_assigns_default_priority_before_bootstrap_queue(self):
        scheduler = self._new_scheduler(DisaggregationMode.PREFILL)
        req = self._new_req(priority=None)

        scheduler._add_request_to_queue(req)

        self.assertEqual(req.priority, -sys.maxsize - 1)
        scheduler.disagg_prefill_bootstrap_queue.add.assert_called_once_with(req, 8)
        req.time_stats.set_prefill_bootstrap_queue_entry_time.assert_called_once()

    def test_decode_mode_assigns_default_priority_before_prealloc_queue(self):
        scheduler = self._new_scheduler(DisaggregationMode.DECODE)
        req = self._new_req(priority=None)

        scheduler._add_request_to_queue(req)

        self.assertEqual(req.priority, -sys.maxsize - 1)
        scheduler.disagg_decode_prealloc_queue.add.assert_called_once_with(
            req, is_retracted=False
        )
        req.time_stats.set_decode_prealloc_queue_entry_time.assert_called_once()

    def test_priority_disabled_abort_validation_applies_to_decode_mode(self):
        scheduler = self._new_scheduler(DisaggregationMode.DECODE)
        scheduler.enable_priority_scheduling = False
        scheduler.abort_on_priority_when_disabled = True
        req = self._new_req(priority=10)

        scheduler._add_request_to_queue(req)

        scheduler.disagg_decode_prealloc_queue.add.assert_not_called()
        scheduler.send_to_tokenizer.send_output.assert_called_once()
        req.time_stats.trace_ctx.abort.assert_called_once()


class TestDecodePreallocQueuePriority(unittest.TestCase):
    def _new_decode_req(self, rid: str, priority: int, *, failed: bool = False):
        req = SimpleNamespace(
            rid=rid,
            priority=priority,
            origin_input_ids=[1, 2, 3],
            output_ids=[],
            req_pool_idx=int(priority) % 8,
            finished_reason=FINISH_ABORT("failed") if failed else None,
            return_logprob=False,
            sampling_params=SimpleNamespace(max_new_tokens=8),
            cache_protected_len=0,
            time_stats=MagicMock(),
        )
        return SimpleNamespace(
            req=req,
            waiting_for_input=True,
            kv_receiver=MagicMock(),
            metadata_buffer_index=-1,
        )

    def _new_queue(self, decode_reqs, *, low_priority_values_first: bool = False):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.queue = list(decode_reqs)
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = 0
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._allocatable_tokens = MagicMock(return_value=1000)
        queue._pre_alloc = MagicMock(
            side_effect=lambda req, prefix_indices=None, prefix_len=0: torch.arange(
                len(req.origin_input_ids) - prefix_len, dtype=torch.int64
            )
        )

        queue.req_to_token_pool = MagicMock()
        queue.req_to_token_pool.available_size.return_value = 100
        queue.req_to_token_pool.req_to_token = torch.arange(
            8 * 16, dtype=torch.int64
        ).reshape(8, 16)

        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 100
        queue.req_to_metadata_buffer_idx_allocator.alloc.side_effect = iter(range(100))

        queue.token_to_kv_pool_allocator = MagicMock()
        queue.token_to_kv_pool_allocator.page_size = 1
        queue.token_to_kv_pool_allocator.available_size.return_value = 1000
        queue.token_to_kv_pool = MagicMock()
        queue.transfer_queue = SimpleNamespace(queue=[], enable_staging=False)
        queue.kv_manager = SimpleNamespace(kv_args=SimpleNamespace(state_types=[]))
        queue.tree_cache = MagicMock()

        scheduler = MagicMock()
        scheduler.enable_priority_scheduling = True
        scheduler.schedule_low_priority_values_first = low_priority_values_first
        scheduler.running_batch.reqs = []
        scheduler.server_args.disaggregation_decode_enable_radix_cache = False
        scheduler.enable_hisparse = False
        scheduler.waiting_queue = []
        scheduler.last_batch = None
        scheduler.stream_output = MagicMock()
        queue.scheduler = scheduler
        return queue

    def test_prealloc_queue_schedules_higher_priority_values_first_by_default(self):
        reqs = [
            self._new_decode_req("low", 1),
            self._new_decode_req("high", 10),
            self._new_decode_req("mid", 5),
        ]
        queue = self._new_queue(reqs)

        with patch("sglang.srt.disaggregation.decode.CLIP_MAX_NEW_TOKEN", 4096):
            preallocated, failed = queue.pop_preallocated()

        self.assertEqual(
            [decode_req.req.rid for decode_req in preallocated],
            [
                "high",
                "mid",
                "low",
            ],
        )
        self.assertEqual(failed, [])

    def test_prealloc_queue_can_schedule_lower_priority_values_first(self):
        reqs = [
            self._new_decode_req("mid", 5),
            self._new_decode_req("high", 10),
            self._new_decode_req("low", 1),
        ]
        queue = self._new_queue(reqs, low_priority_values_first=True)

        with patch("sglang.srt.disaggregation.decode.CLIP_MAX_NEW_TOKEN", 4096):
            preallocated, failed = queue.pop_preallocated()

        self.assertEqual(
            [decode_req.req.rid for decode_req in preallocated],
            [
                "low",
                "mid",
                "high",
            ],
        )
        self.assertEqual(failed, [])

    def test_failed_request_indices_stay_valid_after_priority_sort(self):
        failed_low = self._new_decode_req("failed-low", 1, failed=True)
        healthy_high = self._new_decode_req("healthy-high", 10)
        queue = self._new_queue([failed_low, healthy_high])

        with patch("sglang.srt.disaggregation.decode.CLIP_MAX_NEW_TOKEN", 4096):
            preallocated, failed = queue.pop_preallocated()

        self.assertEqual(
            [decode_req.req.rid for decode_req in preallocated], ["healthy-high"]
        )
        self.assertEqual([decode_req.req.rid for decode_req in failed], ["failed-low"])
        self.assertEqual(queue.queue, [])
        queue.scheduler.stream_output.assert_called_once_with(
            [failed_low.req], failed_low.req.return_logprob
        )


class TestDecodePrebuiltPriority(unittest.TestCase):
    def test_waiting_queue_is_sorted_before_prebuilt_selection(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.grammar_manager = MagicMock()
        scheduler.grammar_manager.has_waiting_grammars.return_value = False
        original_waiting_queue = [MagicMock(rid="low"), MagicMock(rid="high")]
        scheduler.waiting_queue = original_waiting_queue
        scheduler.waiting_queue[0].priority = 1
        scheduler.waiting_queue[1].priority = 10
        scheduler.enable_priority_scheduling = True
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.batch_size.return_value = 0
        scheduler.req_to_token_pool = MagicMock(size=1)
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.tree_cache = MagicMock()
        scheduler.model_config = MagicMock()
        scheduler.enable_overlap = False
        scheduler.spec_algorithm = MagicMock()
        scheduler.max_running_requests = 1
        scheduler.server_args = SimpleNamespace(
            disaggregation_decode_enable_radix_cache=False
        )
        scheduler.future_map = MagicMock()
        scheduler.policy = MagicMock()
        scheduler.policy.calc_priority.side_effect = (
            lambda waiting_queue, _: waiting_queue.sort(key=lambda req: -req.priority)
        )

        new_batch = MagicMock()
        with patch(
            "sglang.srt.disaggregation.decode.ScheduleBatch.init_new",
            return_value=new_batch,
        ) as init_new:
            ret = SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch(scheduler)

        self.assertIs(ret, new_batch)
        scheduler.policy.calc_priority.assert_called_once_with(
            original_waiting_queue, scheduler.running_batch
        )
        selected_reqs = init_new.call_args.args[0]
        self.assertEqual([req.rid for req in selected_reqs], ["high"])
        self.assertEqual([req.rid for req in scheduler.waiting_queue], ["low"])


if __name__ == "__main__":
    unittest.main()
