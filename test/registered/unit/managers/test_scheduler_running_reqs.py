import unittest
from array import array
from functools import partial
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

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.req_types import FINISH_LENGTH
from sglang.srt.managers.schedule_batch import NextBatchPlan, Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.pause_state import PauseState
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
    move,
)
from sglang.srt.managers.scheduler_components.state_check import CheckPhase
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.scheduler_test_utils import empty_request_lists, leaving_retirer

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


def _invariant_checker(s):
    return SchedulerInvariantChecker(
        disaggregation_mode=DisaggregationMode.NULL,
        page_size=1,
        tree_cache=None,
        token_to_kv_pool_allocator=None,
        req_to_token_pool=None,
        pool_stats_observer=None,
        scheduler_stage_metrics=None,
        get_request_lists=s.request_lists,
        get_chunked_req=lambda: s.chunked_req,
        get_running_batches=lambda: (s.running_batch,),
    )


def _scheduler():
    s = _Scheduler.__new__(_Scheduler)
    empty_request_lists(s)
    s.dllm_manager = SimpleNamespace(waiting_queue=IndexedReqList(ListKind.DLLM))
    s.running_batch = ScheduleBatch(reqs=[])
    s.invariant_checker = _invariant_checker(s)
    s.disaggregation_mode = DisaggregationMode.NULL
    s.enable_hisparse = False
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
    # process_batch_result, with the mode-specific processing mocked
    s.publish_load_snapshot = Mock()
    s.load_publisher = MagicMock()
    s.load_inquirer = MagicMock()
    s.batch_result_processor = MagicMock()
    s.tree_cache = MagicMock()
    s._record_step_counters = Mock()
    s.enable_fpm = False
    s._maybe_clear_mm_inputs = Mock()
    s.maybe_send_health_check_signal = Mock()
    return s


def _schedule_without_admission(s):
    s.process_pending_chunked_abort = Mock()
    s._process_hicache_events = Mock()
    s.require_mlp_sync = False
    s.spec_algorithm = SpeculativeAlgorithm.NONE
    s._should_defer_prefill = Mock(return_value=True)
    s.update_running_batch = lambda b: b
    s.dp_attn_adapter = SimpleNamespace(
        maybe_prepare_mlp_sync_batch=lambda b, **_: b,
        maybe_convert_decode_to_extend=lambda b: b,
    )
    s._arm_prefill_decode_interval = Mock()
    s.ngram_embedding_manager = SimpleNamespace(prepare_for_forward=lambda b, **_: b)


def _launch(s, batch):
    try:
        s.run_batch(batch)
    except _ForwardLaunched:
        pass


class TestRequestListsCheck(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.s = _scheduler()
        self.check = partial(self.s.invariant_checker.run, CheckPhase.LAUNCH)
        enter_scope(self, published_topology())
        enter_scope(self, envs.SGLANG_ENABLE_RUNNING_REQS_CHECK.override(True))

    def _start_running(self, reqs):
        self.s.running_batch = ScheduleBatch(reqs=reqs)
        move(reqs, self.s.running_reqs)

    def test_fails_a_request_owned_by_another_list(self):
        req = _req("a")
        move([req], IndexedReqList(ListKind.WAITING))
        # Corrupt the lists: running also holds the request another list owns.
        self.s.running_reqs._reqs[req] = None

        with self.assertRaisesRegex(
            RuntimeError, "a is in DECODING but owned by WAITING"
        ):
            self.check(ScheduleBatch(reqs=[]))

    def test_fails_a_launched_row_in_no_list(self):
        orphan, done = _req("o"), _req("d")
        _finish(done)

        with self.assertRaisesRegex(RuntimeError, r"in no list: \['o'\]"):
            self.check(ScheduleBatch(reqs=[orphan, done]))

    def test_fails_a_running_batch_row_not_running(self):
        decoding, retracted, chunk = _req("r"), _req("w"), _req("c")
        self._start_running([decoding])
        self.s.running_batch.reqs += [retracted, chunk]
        move([retracted], self.s.waiting_queue)
        move([chunk], self.s.chunked_reqs)

        with self.assertRaisesRegex(
            RuntimeError, r"running batch but not in running: \['w'\]"
        ):
            self.check(self.s.running_batch)

    def test_fails_a_running_request_held_by_nothing(self):
        decoding, lost = _req("r"), _req("l")
        self._start_running([decoding])
        move([lost], self.s.running_reqs)

        with self.assertRaisesRegex(
            RuntimeError, r"in no running batch or step: \['l'\]"
        ):
            self.check(self.s.running_batch)

    def test_fails_a_launched_prefill_in_a_running_batch(self):
        merged = _req("m")
        self.s.running_batch = ScheduleBatch(reqs=[merged])
        move([merged], self.s.prefill_launched_reqs)

        with self.assertRaisesRegex(
            RuntimeError, r"prefill_launched but in a running batch: \['m'\]"
        ):
            self.check(self.s.running_batch)

    def test_passes_a_request_its_queued_result_finishes(self):
        # Overlap took it out of the running batch; its queued result finishes it.
        decoding, finishing = _req("r"), _req("f")
        self._start_running([decoding, finishing])
        _launch(self.s, _batch([decoding, finishing], ForwardMode.DECODE))
        self.s.running_batch = ScheduleBatch(reqs=[decoding])

        self.assertEqual(finishing.inflight_steps, 1)
        self.check(_batch([decoding], ForwardMode.DECODE))

    def test_passes_the_decode_rows_of_another_microbatch_mixed_step(self):
        # A mixed step of another microbatch took these decode rows out of its
        # running batch.
        new, decoding = _req("n"), _req("d")
        mixed = _batch([new, decoding], ForwardMode.MIXED)
        move([decoding], self.s.running_reqs)
        move([new], self.s.prefill_launched_reqs)
        _launch(self.s, mixed)
        self.s.invariant_checker.get_pp_mbs = lambda: [mixed, None]
        self.s.invariant_checker.get_running_batches = lambda: (
            self.s.running_batch,
            ScheduleBatch(reqs=[]),
        )

        self.assertEqual(list(self.s.prefill_launched_reqs), [new])
        self.check(_batch([], ForwardMode.IDLE))


class TestRunningReqsHooks(CustomTestCase):
    def setUp(self):
        super().setUp()
        enter_scope(self, published_topology())

    def test_prefills_wait_in_prefill_launched_until_the_merge(self):
        s = _scheduler()
        a, chunk, new, decoding = (_req(rid) for rid in ("a", "c", "n", "d"))
        s.running_batch = ScheduleBatch(reqs=[decoding])
        move([decoding], s.running_reqs)
        move([chunk], s.chunked_reqs)
        # A parked chunk: scheduling has no KV of it to stash.
        chunk.extend_end = chunk.prefix_len
        # As admission leaves them.
        move([a, new], s.prefill_launched_reqs)

        # A launch moves no request between lists.
        _launch(s, _batch([a, chunk], ForwardMode.EXTEND, chunked_req=chunk))
        mixed = _batch([new, decoding], ForwardMode.MIXED)
        _launch(s, mixed)
        _launch(s, _batch([decoding], ForwardMode.DECODE))

        self.assertEqual(list(s.running_reqs), [decoding])

        # The merge moves the mixed step's prefill; its decode row stays put.
        _schedule_without_admission(s)
        with envs.SGLANG_ENABLE_RUNNING_REQS_CHECK.override(True):
            Scheduler.get_next_batch_to_run(
                s, ScheduleBatch(reqs=[], batch_is_full=False), mixed
            )
        self.assertEqual(list(s.prefill_launched_reqs), [a])
        self.assertEqual(list(s.running_reqs), [decoding, new])

        # dLLM keeps its requests in its own queue.
        s = _scheduler()
        s.dllm_config = object()
        move([_req("p")], s.dllm_manager.waiting_queue)
        _launch(s, _batch(list(s.dllm_manager.waiting_queue), ForwardMode.EXTEND))
        self.assertEqual(list(s.prefill_launched_reqs), [])

    def test_ngram_grammar_step_launches_no_finished_row(self):
        # A grammar row under NGRAM processes the previous result early; once it
        # finishes, the loop overlaps again.
        done, kept = _req("done"), _req("kept")
        done.grammar = MagicMock()
        for req in (done, kept):
            req.kv.kv_committed_len = 4
        running = _batch([done, kept], ForwardMode.DECODE)
        running.spec_algorithm = SpeculativeAlgorithm.NGRAM
        running.has_grammar = True
        s = Scheduler.__new__(Scheduler)
        empty_request_lists(s)
        s.running_batch = running
        s.last_batch = None
        move(running.reqs, s.running_reqs)
        s.disaggregation_mode = DisaggregationMode.NULL
        s.enable_hisparse = False
        s.invariant_checker = _invariant_checker(s)
        s._confidence_budget_prepare = None
        # Event loop
        s.gracefully_exit = False
        s.pause_state = PauseState()
        s.increment_scheduler_iterations = Mock()
        s.ingest_requests = Mock(side_effect=[None, None, None, StopIteration])
        s.is_generation = True
        s.enable_unified_memory = False
        s._apply_war_barrier = Mock()
        s.launch_batch_sample_if_needed = Mock()
        # Scheduling: no admission; every step decodes.
        s.scheduler_stage_metrics = None
        s.process_pending_chunked_abort = Mock()
        s._process_hicache_events = Mock()
        s.require_mlp_sync = False
        s.spec_algorithm = SpeculativeAlgorithm.NGRAM
        s._should_defer_prefill = Mock(return_value=False)
        s.get_new_batch_prefill = lambda b: NextBatchPlan(None, b)
        s.update_running_batch = lambda b: b
        s.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=lambda b, **_: b,
            maybe_convert_decode_to_extend=lambda b: b,
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
        # Each launch's rows, and whether each has an earlier step unprocessed.
        launched = []

        def run_batch(batch):
            rids = [req.rid for req in batch.reqs]
            try:
                Scheduler.run_batch(s, batch)
            except _ForwardLaunched:
                launched.append((rids, [r.inflight_steps > 1 for r in batch.reqs]))
                return SimpleNamespace()

        def process_batch_result(batch, result):
            _finish(done)
            leaving_retirer().release(done, checkpoint=True)
            s._on_step_processed(batch, result)

        s.run_batch = run_batch
        s.process_batch_result = process_batch_result

        with (
            envs.SGLANG_ENABLE_RUNNING_REQS_CHECK.override(True),
            self.assertRaises(StopIteration),
        ):
            s.event_loop_overlap()

        self.assertEqual(
            launched,
            [
                (["done", "kept"], [False, False]),
                (["kept"], [False]),
                (["kept"], [True]),
            ],
        )

    def test_preemption_leaves_even_when_nothing_is_admitted(self):
        s = _scheduler()
        victim, other = _req("v"), _req("o")
        running_batch = ScheduleBatch(reqs=[victim, other])
        move(running_batch.reqs, s.running_reqs)
        waiting = SimpleNamespace(
            owner_list=None,
            beam_group=None,
            init_next_round_input=Mock(),
            kv=SimpleNamespace(holds_mamba=False),
        )

        class _Adder(PrefillAdder):
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
        move([waiting], s.waiting_queue)
        s.min_free_slots_delayer = None
        s.get_num_allocatable_reqs = lambda *args, **kwargs: 0
        s.policy = Mock(calc_priority=lambda waiting, *args, **kwargs: list(waiting))
        s.disagg_prefill_bootstrap_queue = None
        s.chunked_prefill_size = 8192
        s.dynamic_chunk_sizer = None
        s.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=SimpleNamespace())
        )
        s.get_prefill_adder_class = lambda: _Adder
        s.request_retirer = None
        s.page_size = 1
        s.tree_cache = SimpleNamespace(buffer_pipeline=None)
        s.token_to_kv_pool_allocator = SimpleNamespace(
            state_slots=None, pools_share_memory=False
        )
        s.new_token_ratio_tracker = SimpleNamespace(current=0.0)
        s.max_prefill_tokens = 1 << 20
        s.is_mixed_chunk = False
        s.priority_scheduling_preemption_threshold = 0
        s.max_prefill_bs = 0
        s.max_running_requests = 4
        s.enable_lora = False
        s.enable_hicache_storage = False
        s.enable_unified_cache_external_linker = False
        s.enable_lmcache = False
        s.truncation_align_size = None
        s.uocr_runtime = None
        s._set_or_validate_priority = Mock(return_value=True)
        s._abort_on_queued_limit = Mock(return_value=False)
        s._prefetch_kvcache = Mock()

        new_batch, _ = s._get_new_batch_prefill_raw(None, running_batch)

        self.assertIsNone(new_batch)
        self.assertEqual(list(s.waiting_queue), [waiting, victim])
        self.assertEqual(list(s.running_reqs), [other])

    def test_launch_counts_in_flight_steps_and_processing_drops_them(self):
        s = _scheduler()
        a, b = _req("a"), _req("b")
        first = _batch([a, b], ForwardMode.DECODE)
        second = _batch([a], ForwardMode.DECODE)
        s.running_batch = ScheduleBatch(reqs=[a, b])
        move([a, b], s.running_reqs)
        _launch(s, first)
        _launch(s, second)
        self.assertEqual((a.inflight_steps, b.inflight_steps), (2, 1))

        with envs.SGLANG_ENABLE_RUNNING_REQS_CHECK.override(True):
            s.process_batch_result(first, MagicMock())
            s.process_batch_result(second, MagicMock())
            self.assertEqual((a.inflight_steps, b.inflight_steps), (0, 0))
            with self.assertRaisesRegex(RuntimeError, r"launched with \[\]"):
                never_launched = _batch([_req("c")], ForwardMode.DECODE)
                s.process_batch_result(never_launched, MagicMock())


class TestPriorityPreemptionUnderUnifiedMemory(CustomTestCase):
    """Priority preemption is off under unified memory, while priority
    scheduling itself stays on."""

    def _preemption(self, **fields):
        override = get_context().override_server_args(
            enable_priority_scheduling=True, **fields
        )
        override.install()
        self.addCleanup(override.restore)
        s = Scheduler.__new__(Scheduler)
        s.schedule_policy = "fcfs"
        s.tree_cache = MagicMock()
        s.enable_hierarchical_cache = False
        s.enable_priority_scheduling = True
        s.schedule_low_priority_values_first = False
        s.metrics_reporter = SimpleNamespace(enable_metrics=False)
        Scheduler.init_schedule_policy(s)
        return s.enable_priority_preemption

    def test_off_with_unified_memory(self):
        self.assertFalse(self._preemption(enable_unified_memory=True))

    def test_on_without_it(self):
        self.assertTrue(self._preemption(enable_unified_memory=False))


if __name__ == "__main__":
    unittest.main()
