import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
)
from sglang.srt.model_executor.forward_mode import ForwardMode

register_cpu_ci(est_time=4, suite="base-a-test-cpu")
TRACE_LOG = "sglang.srt.managers.scheduler_components.prefill_admission_trace"


class TestPrefillAdmissionTrace(unittest.TestCase):
    def scheduler(self, *requests):
        s = Scheduler.__new__(Scheduler)
        s.dp_round_trace = True
        s.forward_ct = 17
        s.prefill_delayer = None
        s.grammar_manager = Mock()
        s.grammar_manager.has_waiting_grammars.return_value = False
        s.enable_priority_preemption = s.is_hybrid_swa = False
        s.chunked_reqs = IndexedReqList(ListKind.CHUNKED)
        s.waiting_queue = IndexedReqList(ListKind.WAITING)
        s.waiting_queue.join(requests)
        return s

    def test_multiple_scans_without_a_forward_keep_their_own_outcomes(self):
        head = SimpleNamespace(time_stats=SimpleNamespace(wait_queue_entry_time=1.0))
        s = self.scheduler(head)
        running = ScheduleBatch(reqs=[object()], batch_is_full=True)
        with self.assertLogs(TRACE_LOG) as records:
            self.assertIsNone(s.get_new_batch_prefill(running).batch_to_run)
            running.batch_is_full = False
            s.waiting_queue.clear()
            self.assertIsNone(s.get_new_batch_prefill(running).batch_to_run)
        self.assertIn(
            "scan=1 after_fct=17 queue=1 visited=0 admitted=0", records.output[0]
        )
        self.assertIn("stop=batch_full", records.output[0])
        self.assertIn(
            "scan=2 after_fct=17 queue=0 visited=0 admitted=0", records.output[1]
        )
        self.assertIn(
            "head_wait_ms=None skips=none stop=queue_empty", records.output[1]
        )
        s.dp_round_trace = False
        with self.assertNoLogs(TRACE_LOG):
            s.get_new_batch_prefill(running)
        self.assertEqual(s._dp_admission_scan_seq, 2)

    def test_prefetch_skip_then_pass_budget_does_not_count_unvisited_rows(self):
        reqs = [Mock(beam_group=None, token_indices_to_pool=None) for _ in range(3)]
        for req in reqs:
            req.time_stats.wait_queue_entry_time = 0.0
        s = self.scheduler(*reqs)
        s.min_free_slots_delayer = None
        s.get_num_allocatable_reqs = lambda *args, **kwargs: 4
        s.processed_tokens_counter = 0
        s.disagg_prefill_bootstrap_queue = None
        s.policy = Mock(calc_priority=lambda waiting, *args, **kwargs: list(waiting))
        s.chunked_prefill_size = 2048
        s.tp_worker = Mock()
        adder = Mock(can_run_list=[], preempt_list=[])
        adder.budget_state.side_effect = (AddReqResult.CONTINUE, AddReqResult.OTHER)
        s.get_prefill_adder_class = lambda: lambda *args, **kwargs: adder
        s.page_size = 128
        s.tree_cache = Mock(buffer_pipeline=None)
        s.tree_cache.check_prefetch_progress.return_value = False
        s.token_to_kv_pool_allocator = Mock(state_slots=None)
        s.new_token_ratio_tracker = Mock(current=0.5)
        s.max_prefill_tokens = s.max_prefill_bs = s.max_running_requests = 16
        s.is_mixed_chunk = s.enable_lora = s.enable_lmcache = False
        s.priority_scheduling_preemption_threshold = 0
        s.dllm_config = None
        s.enable_hicache_storage = True
        s.running_reqs = Mock()
        with (
            patch.object(
                scheduler_module,
                "get_schedule",
                return_value=Mock(prefill_max_requests=None),
            ),
            self.assertLogs(TRACE_LOG) as records,
        ):
            self.assertIsNone(
                s.get_new_batch_prefill(ScheduleBatch(reqs=[])).batch_to_run
            )
        self.assertIn("queue=3 visited=2 admitted=0", records.output[0])
        self.assertIn("skips=prefetch:1 stop=pass_budget", records.output[0])
        self.assertEqual(list(s.waiting_queue), reqs)

    def test_round_joins_only_the_producing_scan_and_reuses_balance_sample(self):
        s = self.scheduler()
        batch = ScheduleBatch(
            reqs=[], forward_mode=ForwardMode.EXTEND, extend_num_tokens=32
        )

        def select(*, trace, running_batch, **kwargs):
            trace.admitted = 1
            return batch, running_batch

        s._get_new_batch_prefill_raw = select
        with self.assertLogs(TRACE_LOG):
            self.assertIs(s.get_new_batch_prefill(batch).batch_to_run, batch)
        batch.dp_balance_stats = SimpleNamespace(
            local_tokens=32,
            max_tokens=64,
            sync_wait_seconds=0.0125,
            local_attention_pairs=64,
            max_attention_pairs=8192,
        )
        s.metrics_reporter = s.profiler_manager = s.invariant_checker = Mock()
        s._join_prefill_launched = Mock()
        s.scripted_scheduler_hook = None
        s._sched_idled = False
        s.processed_tokens_counter = 0
        with self.assertLogs(scheduler_module.logger) as rounds:
            s._on_step_launch(batch)
            batch.forward_mode = ForwardMode.IDLE
            batch.extend_num_tokens = 0
            batch.dp_balance_stats = None
            s._on_step_launch(batch)
        self.assertIn(
            "fct=18 mode=extend bs=0 new=32 scan=1 local_tokens=32 max_tokens=64 gather_ms=12.5",
            rounds.output[0],
        )
        self.assertIn("local_pairs=64 max_pairs=8192", rounds.output[0])
        self.assertIn(
            "fct=19 mode=idle bs=0 new=0 scan=None local_tokens=None max_tokens=None gather_ms=None",
            rounds.output[1],
        )


if __name__ == "__main__":
    unittest.main()
