"""The scheduler's running-request owner and the hooks that keep it."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

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
    FINISH_ABORT,
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

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _req(rid):
    return Req(rid, "", [1, 2, 3], SamplingParams())


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


def _running_reqs_checker(s):
    return SchedulerInvariantChecker(
        disaggregation_mode=s.disaggregation_mode,
        page_size=1,
        tree_cache=Mock(),
        token_to_kv_pool_allocator=Mock(),
        req_to_token_pool=Mock(),
        pool_stats_observer=Mock(),
        get_last_batch=Mock(),
        get_running_batch=Mock(),
        scheduler_stage_metrics=Mock(),
        get_chunked_req=lambda: s.chunked_req,
        get_running_reqs=lambda: s.running_reqs,
    )


def _scheduler():
    s = _Scheduler.__new__(_Scheduler)
    s.running_reqs = RunningReqs()
    s.disaggregation_mode = DisaggregationMode.NULL
    s.dllm_config = None
    s.enable_hisparse = False
    s.chunked_req = None
    s.invariant_checker = _running_reqs_checker(s)
    # run_batch up to the forward
    s.metrics_reporter = MagicMock()
    s.forward_ct = 0
    s._sched_idled = False
    s.processed_tokens_counter = 0
    s.scripted_scheduler_hook = None
    s.profiler_manager = MagicMock()
    s.forward_sleep_time = None
    s.enable_staging = False
    # get_next_batch_to_run, with admission and decode prep stubbed
    s.scheduler_stage_metrics = None
    s.process_pending_chunked_abort = Mock()
    s._process_hicache_events = Mock()
    s.stash_chunked_request = Mock()
    s.require_mlp_sync = False
    s._should_defer_prefill = Mock(return_value=False)
    s.dp_attn_adapter = Mock()
    s.dp_attn_adapter.maybe_prepare_mlp_sync_batch.side_effect = lambda b, **_: b
    s.dp_attn_adapter.maybe_convert_decode_to_extend.side_effect = lambda b: b
    s._arm_prefill_decode_interval = Mock()
    s.ngram_embedding_manager = Mock()
    s.ngram_embedding_manager.prepare_for_forward.side_effect = lambda b, **_: b
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

    def test_check_holds_through_overlap_prefill_merge_decode_finish(self):
        s = _scheduler()
        a, b, chunk = _req("a"), _req("b"), _req("c")
        chunk.extend_end = chunk.prefix_len + 1
        prefill = _batch([a, b, chunk], ForwardMode.EXTEND, chunked_req=chunk)

        def admit(running_batch):
            batch, admit.next = admit.next, None
            return NextBatchPlan(batch_to_run=batch, running_batch=running_batch)

        def decode(batch):
            batch.forward_mode = ForwardMode.DECODE
            return batch

        s.get_new_batch_prefill = admit
        s.update_running_batch = decode

        def step(running_batch, last_batch):
            plan = s.get_next_batch_to_run(running_batch, last_batch)
            if plan.batch_to_run is not None:
                _launch(s, plan.batch_to_run)
            return plan

        with envs.SGLANG_ENABLE_RUNNING_REQS_CHECK.override(True):
            # Launch a prefill whose chunked request continues.
            admit.next, s.chunked_req = prefill, chunk
            plan = step(ScheduleBatch(reqs=[]), None)
            # Overlap: merge the prefill before its result, then decode it.
            plan = step(plan.running_batch, plan.batch_to_run)
            self.assertIs(plan.batch_to_run, prefill)
            self.assertEqual(list(s.running_reqs), [a, b])
            # Processing the prefill result finishes a, and it leaves.
            _finish(a)
            s.running_reqs.leave_finished()
            plan = step(plan.running_batch, plan.batch_to_run)
            self.assertEqual(list(s.running_reqs), [b])

            # A request that never joined fails the check.
            s.running_reqs.join([_req("stray")])
            with self.assertRaisesRegex(RuntimeError, "stray"):
                step(plan.running_batch, plan.batch_to_run)

    def test_retract_leaves_retracted_and_aborted(self):
        s = _scheduler()
        keep, retracted, aborted = _req("k"), _req("r"), _req("x")
        aborted.to_finish = FINISH_ABORT("KV cache pool is full")
        batch = _batch([keep, retracted, aborted], ForwardMode.DECODE)
        s.running_reqs.join(batch.reqs)
        s.enable_skip_finishing_decode = False
        s.decode_offload_manager = None
        s.token_to_kv_pool_allocator = MagicMock()
        s.token_to_kv_pool_allocator.available_size.return_value = 0
        s.tree_cache = MagicMock()
        s.tree_cache.token_to_kv_pool_allocator.state_slots = None
        s.new_token_ratio_tracker = SimpleNamespace(current=1.0)
        s.send_abort_output = Mock()
        s.beam_coordinator = Mock()
        s._add_request_to_queue = Mock()

        def retract_decode():
            batch.reqs = [keep]
            return [retracted], 0.5, [aborted]

        with (
            patch.object(batch, "check_decode_mem", return_value=False),
            patch.object(batch, "retract_decode", side_effect=retract_decode),
            patch.object(batch, "prepare_for_decode"),
            patch("sglang.srt.managers.scheduler._make_abort_req"),
        ):
            s.update_running_batch(batch)

        self.assertEqual(list(s.running_reqs), [keep])

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
