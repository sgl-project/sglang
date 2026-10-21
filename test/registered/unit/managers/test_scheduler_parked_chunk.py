"""A chunked prefill parked for lack of KV builds no prefill batch at all."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Req:
    def __init__(self):
        self.kv = SimpleNamespace(holds_kv=False)
        self.beam_group = None
        self.token_indices_to_pool = None
        self.inflight_middle_chunks = 0

    def init_next_round_input(self, tree_cache=None):
        pass


class _ParkingAdder:
    """Hands the chunk back unadmitted, then admits whatever else is offered."""

    def __init__(self, *args, **kwargs):
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        self.rem_chunk_tokens = 0

    def add_chunked_req(self, req):
        return req

    def budget_state(self):
        return AddReqResult.CONTINUE

    def add_one_req(self, req, has_chunked_req, truncation_align_size):
        self.can_run_list.append(req)
        return AddReqResult.CONTINUE


class _BatchBuilt(Exception):
    pass


class TestParkedChunkSkipsPrefillStep(CustomTestCase):
    """Every prefill batch counts the chunked request as a chunk in flight. A
    batch built while the chunk is parked would count a chunk that never runs,
    and that count never comes back down, so no batch may be built."""

    def _scheduler(self, chunk, waiting) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        s.enable_priority_preemption = False
        s.is_hybrid_swa = False
        s.waiting_queue = waiting
        s.chunked_req = chunk
        s.min_free_slots_delayer = None
        s.get_num_allocatable_reqs = Mock(return_value=8)
        s.policy = Mock()
        s.policy.shortest_prefill_chunk_limit.return_value = None
        s.processed_tokens_counter = 0
        s.chunked_prefill_size = 8192
        s.dynamic_chunk_sizer = None
        s.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=SimpleNamespace())
        )
        s.get_prefill_adder_class = lambda: _ParkingAdder
        s.page_size = 1
        s.tree_cache = SimpleNamespace(
            buffer_pipeline=None, storage_prefetch_retries=None
        )
        s.token_to_kv_pool_allocator = None
        s.new_token_ratio_tracker = SimpleNamespace(current=0.0)
        s.max_prefill_tokens = 1 << 20
        s.is_mixed_chunk = False
        s.priority_scheduling_preemption_threshold = 0
        s.max_prefill_bs = 0
        s.max_running_requests = 8
        s.dllm_config = None
        s.enable_lora = False
        s.enable_hicache_storage = False
        s.enable_lmcache = False
        s.truncation_align_size = None
        s.model_config = None
        s.enable_overlap = False
        s.spec_algorithm = None
        s.req_to_token_pool = SimpleNamespace()
        return s

    def test_parked_chunk_builds_no_batch(self):
        chunk, waiting = _Req(), _Req()
        running_batch = ScheduleBatch(reqs=[])
        s = self._scheduler(chunk, [waiting])

        with (
            published_topology(),
            patch.object(ScheduleBatch, "init_new", side_effect=_BatchBuilt),
            patch("sglang.srt.managers.scheduler.set_time_batch"),
        ):
            batch, _ = s._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None, running_batch=running_batch
            )

        self.assertIsNone(batch)
        self.assertIs(s.chunked_req, chunk)
        self.assertEqual(s.waiting_queue, [waiting])
        self.assertEqual(chunk.inflight_middle_chunks, 0)


if __name__ == "__main__":
    unittest.main()
