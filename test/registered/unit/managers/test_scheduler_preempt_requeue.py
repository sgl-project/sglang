"""Regression test: requests released by priority preemption are requeued."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _PreemptingAdder:
    """Preempts the running request for the candidate, then rejects the candidate."""

    def __init__(self, page_size, tree_cache, token_pool, running_batch, *_, **__):
        self.running_batch = running_batch
        self.can_run_list = []
        self.preempt_list = []

    def preempt_to_schedule(self, req):
        self.preempt_list.extend(self.running_batch.reqs)
        self.running_batch.reqs = []
        return True

    def add_one_req(self, req, **_):
        return AddReqResult.NO_TOKEN


def _make_scheduler(*, running_req, waiting_req):
    s = Scheduler.__new__(Scheduler)
    s.grammar_manager = MagicMock()
    s.grammar_manager.has_waiting_grammars.return_value = False
    s.enable_priority_preemption = True
    s.is_hybrid_swa = False
    s.waiting_queue = [waiting_req]
    s.chunked_req = None
    s.min_free_slots_delayer = None
    s.get_num_allocatable_reqs = MagicMock(return_value=0)
    s.policy = MagicMock()
    s.processed_tokens_counter = 0
    s.dynamic_chunk_sizer = None
    s.chunked_prefill_size = 8192
    s.tp_worker = MagicMock()
    s.page_size = 1
    s.tree_cache = MagicMock()
    s.tree_cache.buffer_pipeline = None
    s.token_to_kv_pool_allocator = MagicMock()
    s.new_token_ratio_tracker = MagicMock()
    s.max_prefill_tokens = 16384
    s.is_mixed_chunk = False
    s.priority_scheduling_preemption_threshold = 0
    s.max_prefill_bs = 0
    s.max_running_requests = 1
    s.dllm_config = None
    s.enable_lora = False
    s.req_to_token_pool = SimpleNamespace()
    s.disaggregation_mode = DisaggregationMode.NULL
    s.enable_hicache_storage = False
    s.enable_lmcache = False
    s.enable_hierarchical_cache = False
    s.enable_unified_cache_external_linker = False
    s.truncation_align_size = None
    s._add_request_to_queue = MagicMock()
    running_batch = MagicMock()
    running_batch.batch_is_full = False
    running_batch.reqs = [running_req]
    running_batch.is_empty.return_value = False
    return s, running_batch


class TestPreemptedRequestsRequeuedWhenNothingAdmitted(CustomTestCase):
    def test_preempted_request_is_requeued(self):
        # Preemption releases a running request before its candidate is added;
        # if the candidate is then rejected, the released request must be queued
        # again, or it is in no queue and never finishes.
        running_req = MagicMock(name="running_req")
        waiting_req = MagicMock(name="waiting_req")
        waiting_req.token_indices_to_pool = None
        waiting_req.beam_group = None
        waiting_req.kv.holds_mamba = False
        s, running_batch = _make_scheduler(
            running_req=running_req, waiting_req=waiting_req
        )

        with (
            patch.object(scheduler_module, "PrefillAdder", _PreemptingAdder),
            patch.object(
                scheduler_module,
                "get_schedule",
                return_value=SimpleNamespace(prefill_max_requests=None),
            ),
        ):
            batch, _ = Scheduler._get_new_batch_prefill_raw(
                s, prefill_delayer_single_pass=None, running_batch=running_batch
            )

        self.assertIsNone(batch)
        s._add_request_to_queue.assert_called_once_with(running_req)


if __name__ == "__main__":
    unittest.main()
