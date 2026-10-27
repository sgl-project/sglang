"""HiCache progress must not depend on whether a prefill batch is admitted."""

import unittest
from concurrent.futures import Future
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, call

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from test_request_class_override_points import _recv, _scheduler

from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.managers.schedule_batch import NextBatchPlan
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestSchedulerHiCacheEvents(unittest.TestCase):
    def test_cache_prefetch_follows_grammar_admission_on_every_pp_stage(self):
        for rank in range(3):
            with (
                self.subTest(rank=rank),
                published_topology(
                    tp_size=2,
                    pp_size=3,
                    ranks={"world_rank": rank * 2},
                    hicache_host_memory_mode="cache",
                ),
            ):
                s = _scheduler({}, [])
                s.max_req_input_len = 1024
                s.validate_dllm_request = lambda req: None
                s.spec_algorithm = Mock()
                s.spec_algorithm.is_dflash_family.return_value = False
                s.spec_algorithm.is_uno.return_value = False
                s.tree_cache = Mock()
                s._set_or_validate_priority = lambda req: True
                s._abort_on_queued_limit = lambda req: False
                s.waiting_queue = []
                s.processed_tokens_counter = 0
                s._add_request_to_queue = MethodType(Scheduler._add_request_to_queue, s)
                issued = []
                s._prefetch_kvcache = lambda req: issued.append(req.rid)
                gm = s.grammar_manager = GrammarManager.__new__(GrammarManager)
                gm.grammar_queue = []
                gm._enable_strict_thinking = False
                gm.grammar_backend = Mock()
                future = Future()
                gm.grammar_backend.get_cached_or_future_value.return_value = (
                    future,
                    False,
                )
                structured = _recv("r")
                structured.sampling_params.json_schema = '{"type":"object"}'
                s.handle_generate_request(structured)
                s.handle_generate_request(_recv("q"))
                self.assertEqual(issued, ["q"])
                s.tree_cache.bind_prefetch_ticket.assert_not_called()
                # Native grammar readiness moves r through the same enqueue path.
                future.set_result(Mock())
                s._add_request_to_queue(gm.grammar_queue.pop())
                self.assertEqual(issued, ["q", "r"])

    def setUp(self):
        self.calls = Mock()
        self.scheduler = s = Scheduler.__new__(Scheduler)
        s.scheduler_stage_metrics = None
        s.enable_hierarchical_cache = True
        s.enable_unified_cache_external_linker = False
        s.enable_lmcache = False
        s.enable_hicache_storage = True
        s.tree_cache = SimpleNamespace(check_hicache_events=self.calls.drain)
        s._process_storage_prefetch_retries = self.calls.retry
        s.process_pending_chunked_abort = Mock()
        s.process_prefill_chunk = Mock()
        s.dp_attn_adapter = Mock()
        s.dp_attn_adapter.maybe_prepare_mlp_sync_batch.return_value = None
        self.running_batch = SimpleNamespace(
            batch_is_full=False,
            is_prefill_only=False,
            is_empty=lambda: True,
            batch_size=lambda: 0,
            filter_batch=lambda: None,
        )

    def test_pd_prefill_drains_before_admission_even_with_empty_queue(self):
        s = self.scheduler
        s.resolve_waiting_queue_bootstrap = Mock()
        s.get_new_batch_prefill = self.calls.admit
        s.get_new_batch_prefill.return_value = NextBatchPlan(
            batch_to_run=None, running_batch=self.running_batch
        )
        for waiting_queue in ([], [SimpleNamespace(rid="pending_l3")]):
            with self.subTest(waiting=bool(waiting_queue)):
                s.waiting_queue = waiting_queue
                self.calls.reset_mock()
                for _ in range(2):
                    plan = s.get_next_disagg_prefill_batch_to_run(
                        running_batch=self.running_batch, last_batch=None
                    )
                    self.assertIsNone(plan.batch_to_run)
                self.assertEqual(
                    self.calls.mock_calls,
                    [call.drain(), call.retry(), call.admit(self.running_batch)] * 2,
                )

    def test_unified_drains_when_prefill_is_deferred(self):
        s = self.scheduler
        s.enable_fpm = False
        s._abort_on_waiting_timeout = Mock()
        s._abort_on_running_timeout = Mock()
        s.dllm_config = None
        s.chunked_req = None
        s.enable_hisparse = False
        s.require_mlp_sync = False
        s._should_defer_prefill = self.calls.defer
        s._should_defer_prefill.return_value = True
        s.get_new_batch_prefill = Mock()
        s.dp_attn_adapter.maybe_convert_decode_to_extend.return_value = None
        s._arm_prefill_decode_interval = Mock()
        s.ngram_embedding_manager = Mock()
        s.ngram_embedding_manager.prepare_for_forward.return_value = None

        plan = s.get_next_batch_to_run(self.running_batch, None)

        self.assertIsNone(plan.batch_to_run)
        s.get_new_batch_prefill.assert_not_called()
        self.assertEqual(
            self.calls.mock_calls,
            [
                call.drain(),
                call.retry(),
                call.defer(self.running_batch),
            ],
        )

    def test_pp_prefill_drains_before_admission(self):
        s = self.scheduler
        s.init_pp_loop_state = Mock()
        s.pp_loop_size = 1
        enter_scope(self, published_topology(pp_size=2))
        s.pp_group = SimpleNamespace(is_last_rank=True)
        s.running_mbs = [self.running_batch]
        s.last_mbs = [None]
        s.ingest_requests = Mock(return_value=[])
        s._pp_prefetch_step = self.calls.completion
        s.waiting_queue = []
        s.disagg_prefill_bootstrap_queue = SimpleNamespace(pop_bootstrapped=lambda: [])
        s._pp_pd_get_prefill_transferred_ids = Mock(return_value=[])
        s._pp_commit_comm_work = Mock()
        s.get_new_batch_prefill = self.calls.admit
        # Stop the infinite loop at admission; no PP transport or GPU is needed.
        s.get_new_batch_prefill.side_effect = StopIteration

        with self.assertRaises(StopIteration):
            s.event_loop_pp_disagg_prefill()

        self.assertEqual(
            self.calls.mock_calls,
            [
                call.completion(),
                call.drain(),
                call.retry(),
                call.admit(self.running_batch),
            ],
        )


if __name__ == "__main__":
    unittest.main()
