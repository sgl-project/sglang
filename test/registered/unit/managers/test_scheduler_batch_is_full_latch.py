"""Regression tests for the empty-batch batch_is_full latch (#41841)."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefResult, IncLockRefResult
from sglang.srt.mem_cache.prefill_budget import PrefillBudget
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_tree_cache() -> MagicMock:
    tree_cache = MagicMock()
    tree_cache.supports_mamba.return_value = False
    tree_cache.full_evictable_size.return_value = 0
    tree_cache.evictable_size.return_value = 0
    tree_cache.inc_lock_ref.return_value = IncLockRefResult()
    tree_cache.dec_lock_ref.return_value = DecLockRefResult()
    tree_cache.buffer_pipeline = None
    tree_cache.storage_prefetch_retries = None
    return tree_cache


def _make_token_allocator(available_size: int) -> MagicMock:
    allocator = MagicMock()
    allocator.available_size.return_value = available_size
    allocator.full_available_size.return_value = available_size
    allocator.swa_available_size.return_value = 0
    allocator.size_swa = 1_000_000
    allocator.swa_req_ring = False
    allocator.page_size = 1
    allocator.create_prefill_budget.side_effect = lambda tree_cache, **kwargs: (
        PrefillBudget(allocator, tree_cache, **kwargs)
    )
    return allocator


def _make_req(rid: str, fill_len: int = 8) -> MagicMock:
    req = MagicMock(spec=Req)
    req.rid = rid
    req.prefix_indices = []
    req.full_untruncated_fill_ids = list(range(fill_len))
    req.output_ids = []
    req.sampling_params = SimpleNamespace(max_new_tokens=4, ignore_eos=False)
    req.host_hit_length = 0
    req.swa_host_hit_length = 0
    req.token_indices_to_pool = None
    req.beam_group = None
    req.session = None
    req.kv = ReqKvInfo(req_pool_idx=0)
    req.last_node = None
    return req


def _scheduler_for_prefill_raw(tree_cache, token_allocator, waiting_queue) -> Scheduler:
    """Stub just enough of Scheduler for _get_new_batch_prefill_raw to run its
    plain (non-hicache, non-lmcache) admission path for real."""
    s = Scheduler.__new__(Scheduler)
    s.grammar_manager = MagicMock()
    s.grammar_manager.has_waiting_grammars.return_value = False
    s.enable_priority_preemption = False
    s.is_hybrid_swa = False
    s.waiting_queue = list(waiting_queue)
    s.chunked_req = None
    s.min_free_slots_delayer = None
    s.get_num_allocatable_reqs = MagicMock(return_value=8)
    s.policy = MagicMock()
    s.processed_tokens_counter = 0
    s.chunked_prefill_size = 4096
    s.dynamic_chunk_sizer = None
    s.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=SimpleNamespace())
    )
    s.page_size = 1
    s.tree_cache = tree_cache
    s.token_to_kv_pool_allocator = token_allocator
    s.new_token_ratio_tracker = SimpleNamespace(current=1.0)
    s.max_prefill_tokens = 10000
    s.is_mixed_chunk = False
    s.priority_scheduling_preemption_threshold = 0
    s.max_prefill_bs = 8
    s.max_running_requests = 128
    s.dllm_config = None
    s.enable_lora = False
    s.req_to_token_pool = SimpleNamespace()
    s.disaggregation_mode = DisaggregationMode.NULL
    s.enable_hierarchical_cache = False
    s.enable_hicache_storage = False
    s.enable_lmcache = False
    s.enable_unified_cache_external_linker = False
    s.truncation_align_size = None
    return s


class TestEmptyBatchFullLatch(CustomTestCase):
    """batch_is_full is only ever cleared on the decode path
    (update_running_batch, last-batch merge). An empty running batch never
    reaches decode, so latching it there stalls every later prefill."""

    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def _build(self, *, available_size: int):
        tree_cache = _make_tree_cache()
        allocator = _make_token_allocator(available_size)
        req = _make_req("req-0")
        s = _scheduler_for_prefill_raw(tree_cache, allocator, [req])
        running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
        return s, req, running_batch

    def test_no_token_on_empty_batch_does_not_latch_full(self):
        # Empty running batch plus an unadmittable request must leave
        # batch_is_full False: nothing will reach decode to clear it.
        s, req, running_batch = self._build(available_size=0)

        new_batch, running_batch = Scheduler._get_new_batch_prefill_raw(
            s, None, running_batch
        )

        self.assertIsNone(new_batch)
        self.assertFalse(running_batch.batch_is_full)

    def test_prefill_retried_after_no_token_on_empty_batch(self):
        # The round after a NO_TOKEN miss must walk the admission path again
        # instead of short-circuiting on the stale latch. calc_priority only
        # runs once the batch_is_full gate is passed, so its call count tells
        # the two apart.
        s, req, running_batch = self._build(available_size=0)
        Scheduler._get_new_batch_prefill_raw(s, None, running_batch)

        new_batch, running_batch = Scheduler._get_new_batch_prefill_raw(
            s, None, running_batch
        )

        self.assertIsNone(new_batch)
        self.assertEqual(s.policy.calc_priority.call_count, 2)
        self.assertFalse(running_batch.batch_is_full)


if __name__ == "__main__":
    unittest.main()
