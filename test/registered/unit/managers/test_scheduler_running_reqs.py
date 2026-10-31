"""The scheduler's running-request owner and the hooks that keep it."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    NextBatchPlan,
    Req,
    ScheduleBatch,
)
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.running_reqs import RunningReqs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _req(rid):
    return Req(rid, "", array("q", [1, 2, 3]), SamplingParams())


def _finish(req):
    req.finished_reason = FINISH_LENGTH(length=1)


def _batch(reqs, mode, chunked_req=None):
    batch = ScheduleBatch(reqs=reqs, forward_mode=mode, chunked_req=chunked_req)
    n = len(reqs)
    batch.device = "cpu"
    batch.model_config = MagicMock(is_encoder_decoder=False)
    batch.sampling_info = MagicMock()
    batch.req_pool_indices = torch.arange(n)
    batch.req_pool_indices_cpu = torch.arange(n)
    batch.seq_lens = torch.full((n,), 4)
    batch.seq_lens_cpu = torch.full((n,), 4)
    batch.orig_seq_lens = torch.full((n,), 4, dtype=torch.int32)
    return batch


class _ForwardLaunched(Exception):
    pass


class _Scheduler(Scheduler):
    """Real scheduling up to the model forward, which run_batch never reaches."""

    @property
    def is_generation(self):
        raise _ForwardLaunched


def _scheduler():
    s = _Scheduler.__new__(_Scheduler)
    s.running_reqs = RunningReqs()
    s.disaggregation_mode = DisaggregationMode.NULL
    s.dllm_config = None
    s.enable_hisparse = False
    s.chunked_req = None
    # run_batch up to the forward
    s.scheduler_stage_metrics = None
    s.metrics_reporter = MagicMock()
    s.forward_ct = 0
    s._sched_idled = False
    s.processed_tokens_counter = 0
    s.scripted_scheduler_hook = None
    s.profiler_manager = MagicMock()
    s.forward_sleep_time = None
    s.enable_staging = False
    return s


def _launch(s, batch):
    try:
        s.run_batch(batch)
    except _ForwardLaunched:
        pass


class TestRunningReqs(CustomTestCase):
    def test_join_keeps_order_and_leaving_mid_iteration_is_safe(self):
        a, b, c, outsider = (_req(rid) for rid in "abcx")
        running = RunningReqs()
        running.join([a, b])
        running.join([c, b])
        _finish(b)

        seen = []
        for req in running:
            seen.append(req)
            running.leave_finished()
            running.leave([a, outsider])

        self.assertEqual(seen, [a, b, c])
        self.assertEqual(list(running), [c])
        self.assertNotIn(a, running)

    def test_check_names_both_differences(self):
        a, b, chunk, done, stray = (_req(rid) for rid in ("a", "b", "c", "d", "s"))
        _finish(done)
        batch = SimpleNamespace(reqs=[a, b, chunk, done])
        running = RunningReqs()
        running.join([a, stray])

        with self.assertRaisesRegex(
            RuntimeError, r"not running: \['b'\].*no batch: \['s'\]"
        ):
            running.check([batch, None], chunked_req=chunk)

        running.leave([stray])
        running.join([b])
        running.check([batch, None], chunked_req=chunk)


class TestRunningReqsHooks(CustomTestCase):
    def setUp(self):
        super().setUp()
        enter_scope(self, published_topology())

    def test_launch_joins_requests_whose_last_chunk_runs(self):
        s = _scheduler()
        a, chunk, new, decoding = (_req(rid) for rid in ("a", "c", "n", "d"))
        s.running_reqs.join([decoding])

        _launch(s, _batch([a, chunk], ForwardMode.EXTEND, chunked_req=chunk))
        _launch(s, _batch([new, decoding], ForwardMode.MIXED))
        _launch(s, _batch([_req("decode-only")], ForwardMode.DECODE))

        self.assertEqual(list(s.running_reqs), [decoding, a, new])

        for field, value in (
            ("disaggregation_mode", DisaggregationMode.PREFILL),
            ("disaggregation_mode", DisaggregationMode.DECODE),
            ("dllm_config", object()),
            ("enable_hisparse", True),
        ):
            with self.subTest(field=field, value=value):
                s = _scheduler()
                setattr(s, field, value)
                _launch(s, _batch([_req("p")], ForwardMode.EXTEND))
                self.assertEqual(len(s.running_reqs), 0)

    def test_converted_step_launched_after_its_result_rejoins_no_finished_row(self):
        # DP attention converts this rank's decode step to extend while a peer
        # prefills, so with SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP the loop
        # processes the previous result between scheduling and launching it.
        # A row that result finishes still rides the launched step.
        s = Scheduler.__new__(Scheduler)
        done, kept = _req("done"), _req("kept")
        for req in (done, kept):
            req.kv.kv_committed_len = 4
        running = _batch([done, kept], ForwardMode.DECODE)
        running.spec_algorithm = SpeculativeAlgorithm.NONE
        s.running_batch = running
        s.last_batch = None
        s.running_reqs = RunningReqs()
        s.running_reqs.join(running.reqs)
        s.disaggregation_mode = DisaggregationMode.NULL
        s.dllm_config = None
        s.enable_hisparse = False
        s.chunked_req = None
        s.invariant_checker = SchedulerInvariantChecker(
            disaggregation_mode=DisaggregationMode.NULL,
            page_size=1,
            tree_cache=None,
            token_to_kv_pool_allocator=None,
            req_to_token_pool=None,
            pool_stats_observer=None,
            get_last_batch=Mock(),
            get_running_batch=Mock(),
            scheduler_stage_metrics=None,
            get_chunked_req=lambda: s.chunked_req,
            get_running_reqs=lambda: s.running_reqs,
        )
        # Event loop
        s.gracefully_exit = False
        s._engine_paused = False
        s.increment_scheduler_iterations = Mock()
        s.ingest_requests = Mock(side_effect=[None, None, None, StopIteration])
        s.is_generation = True
        s.enable_unified_memory = False
        s._apply_war_barrier = Mock()
        s.launch_batch_sample_if_needed = Mock()
        # Scheduling: no admission; every step decodes and is converted to extend.
        s.scheduler_stage_metrics = None
        s.process_pending_chunked_abort = Mock()
        s._process_hicache_events = Mock()
        s.require_mlp_sync = True
        s.spec_algorithm = SpeculativeAlgorithm.NONE
        s._should_defer_prefill = Mock(return_value=False)
        s.get_new_batch_prefill = lambda b: NextBatchPlan(None, b)

        def decode(batch):
            batch.forward_mode = ForwardMode.DECODE
            return batch

        def mlp_sync(batch, **_):
            batch.is_extend_in_batch = True
            return batch

        s.update_running_batch = decode
        s.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=mlp_sync,
            maybe_convert_decode_to_extend=lambda b: b.convert_decode_to_extend() or b,
        )
        s._arm_prefill_decode_interval = Mock()
        s.ngram_embedding_manager = SimpleNamespace(
            prepare_for_forward=lambda b, **_: b
        )
        # Launch up to the forward.
        s.metrics_reporter = MagicMock()
        s.forward_ct = 0
        s._sched_idled = False
        s.processed_tokens_counter = 0
        s.scripted_scheduler_hook = None
        s.profiler_manager = MagicMock()
        s.forward_sleep_time = None
        s.enable_overlap = True
        s.future_map = Mock()
        s.future_map.resolve_seq_lens_cpu.side_effect = _ForwardLaunched

        def run_batch(batch):
            try:
                Scheduler.run_batch(s, batch)
            except _ForwardLaunched:
                return SimpleNamespace()

        s.run_batch = run_batch
        # The first result finishes `done`.
        processed = []

        def process_batch_result(batch, result):
            if not processed:
                _finish(done)
            processed.append(batch)
            s.running_reqs.leave_finished()

        s.process_batch_result = process_batch_result

        with (
            envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.override(True),
            envs.SGLANG_ENABLE_RUNNING_REQS_CHECK.override(True),
            self.assertRaises(StopIteration),
        ):
            s.event_loop_overlap()

        self.assertEqual(len(processed), 2)
        self.assertEqual(list(s.running_reqs), [kept])

    def test_preemption_leaves_even_when_nothing_is_admitted(self):
        s = _scheduler()
        victim, other = _req("v"), _req("o")
        running_batch = ScheduleBatch(reqs=[victim, other])
        s.running_reqs.join(running_batch.reqs)
        waiting = SimpleNamespace(
            beam_group=None,
            init_next_round_input=Mock(),
            kv=SimpleNamespace(holds_mamba=False),
        )

        class _Adder:
            def __init__(self, page_size, tree_cache, allocator, batch, *_, **__):
                self.running_batch = batch
                self.can_run_list = []
                self.preempt_list = []

            def budget_state(self):
                return AddReqResult.CONTINUE

            def preempt_to_schedule(self, req):
                self.preempt_list.append(self.running_batch.reqs.pop(0))
                return True

            def add_one_req(self, req, **_):
                return AddReqResult.NO_TOKEN

        s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        s.enable_priority_preemption = True
        s.is_hybrid_swa = False
        s.waiting_queue = [waiting]
        s.min_free_slots_delayer = None
        s.get_num_allocatable_reqs = lambda *args, **kwargs: 0
        s.policy = Mock()
        s.disagg_prefill_bootstrap_queue = None
        s.chunked_prefill_size = 8192
        s.dynamic_chunk_sizer = None
        s.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=SimpleNamespace())
        )
        s.get_prefill_adder_class = lambda: _Adder
        s.page_size = 1
        s.tree_cache = SimpleNamespace(buffer_pipeline=None)
        s.token_to_kv_pool_allocator = SimpleNamespace(state_slots=None)
        s.new_token_ratio_tracker = SimpleNamespace(current=0.0)
        s.max_prefill_tokens = 1 << 20
        s.is_mixed_chunk = False
        s.priority_scheduling_preemption_threshold = 0
        s.max_prefill_bs = 0
        s.max_running_requests = 4
        s.enable_lora = False
        s.enable_hicache_storage = False
        s.enable_lmcache = False
        s.truncation_align_size = None
        s._add_request_to_queue = Mock()

        new_batch, _ = s._get_new_batch_prefill_raw(None, running_batch)

        self.assertIsNone(new_batch)
        s._add_request_to_queue.assert_called_once_with(victim)
        self.assertEqual(list(s.running_reqs), [other])


if __name__ == "__main__":
    unittest.main()
