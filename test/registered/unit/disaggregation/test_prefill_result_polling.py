"""CPU coverage for continuous input polling and coordinated prefill completion."""

import multiprocessing
import tempfile
import unittest
from array import array
from collections import Counter, deque
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch
import torch.distributed as dist

from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    ReqToMetadataIdxAllocator,
)
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    ContinueGenerationReqInput,
    PauseGenerationReqInput,
    ShutdownReq,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import NextBatchPlan, Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import SchedulePolicy
from sglang.srt.managers.scheduler import Scheduler, _dispatch_event_loop_once
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class _StopLoop(Exception):
    pass


class _Event:
    def __init__(self, ready=False, recorded=True):
        self.ready = ready
        self.recorded = recorded
        self.queries = 0
        self.synchronizations = 0
        self.blocking_synchronizations = 0
        self.records = 0

    def record(self):
        self.recorded = True
        self.records += 1

    def query(self):
        assert self.recorded, "Polled a copy before submission"
        self.queries += 1
        return self.ready

    def synchronize(self):
        assert self.recorded, "Waited for a copy before submission"
        self.synchronizations += 1
        if not self.ready:
            self.blocking_synchronizations += 1
        self.ready = True


def _batch(forward_iter, reqs=None):
    if reqs is None:
        reqs = [Req(str(forward_iter), "", [1, 2, 3, 4], SamplingParams())]
    return ScheduleBatch(
        reqs=reqs,
        forward_iter=forward_iter,
        forward_mode=ForwardMode.EXTEND,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )


def _result(event=None, delayed=False):
    result = GenerationBatchResult(
        next_token_ids=torch.tensor([11]),
        copy_done=event if event is not None else _Event(),
    )
    if delayed:
        result.delay_sample_func = lambda: result
    return result


def _input(batch):
    return TokenizedGenerateReqInput(
        rid=str(batch.forward_iter),
        input_text="",
        input_ids=array("q", [1, 2, 3, 4]),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
    )


class _Scheduler(SchedulerDisaggregationPrefillMixin):
    """Run real scheduling and result handling with controlled intake and GPU events."""

    launch_batch_sample_if_needed = Scheduler.launch_batch_sample_if_needed
    pause_generation = Scheduler.pause_generation
    continue_generation = Scheduler.continue_generation

    def __init__(
        self,
        arrivals,
        results,
        ready_at,
        pause_at=None,
        resume_at=None,
        pause_mode="in_place",
    ):
        self.arrivals = arrivals
        self.batches = {b.forward_iter: b for b in arrivals if b is not None}
        self.results = results
        self.copy_events = {name: result.copy_done for name, result in results.items()}
        self.ready_at = ready_at
        self.ready_batches = deque()
        self.request_receiver = SimpleNamespace(recv_requests=self.recv_requests)
        self._deferred_input_requests = []
        self._incoming_requests = []
        self.pause_at = pause_at
        self.resume_at = resume_at
        self.pause_mode = pause_mode
        self.paused_states = []
        self.iteration = -1
        self.actions = []
        self.admission_attempts = []
        self.chunk_processes = Counter()
        self.result_processes = Counter()
        self.depths = []
        self.scheduler_stage_metrics = None
        self.disaggregation_mode = DisaggregationMode.PREFILL
        self.spec_algorithm = SpeculativeAlgorithm.NONE
        self.is_generation = True
        self.dllm_config = None
        self.tp_cpu_group = None
        self.tp_size = 1
        self.attn_cp_cpu_group = None
        self.attn_tp_cpu_group = None
        self.last_batch = None
        self.result_queue = deque()
        self.chunked_req = None
        self.decode_offload_manager = None
        self.running_batch = ScheduleBatch(reqs=[])
        self.waiting_queue = []
        self.cur_batch_for_debug = None
        self._engine_paused = False
        self.gracefully_exit = False
        self.enable_staging = False
        self.enable_overlap = True
        self.disagg_prefill_bootstrap_queue = SimpleNamespace(
            pop_bootstrapped=lambda: []
        )
        self.ngram_embedding_manager = SimpleNamespace(
            prepare_for_forward=lambda batch, chunked_req: batch
        )
        self.forward_stream_ctx = nullcontext()
        self.copy_stream_ctx = nullcontext()
        self.forward_stream = Mock(spec=["wait_stream"])
        self.copy_stream = Mock(spec=["wait_stream"])
        self.schedule_stream = object()
        self._relay_forward_payload = Mock()
        self.tree_cache = Mock(spec=["flush_pending_backups", "finish"])
        self.batch_result_processor = Mock(
            spec=["snapshot_auxiliary_output_starts", "move_logprobs_to_cpu"]
        )
        self.metrics_reporter = Mock(
            spec=[
                "report_prefill_stats",
                "log_batch_result_stats",
                "update_device_timer",
                "record_scheduler_active",
            ]
        )
        self.metrics_reporter.current_scheduler_metrics_enabled = False
        self.kv_events_publisher = Mock(spec=["publish_kv_events"])
        self.publish_load_snapshot = Mock()
        self.load_publisher = Mock(spec=["publish_load_stat"])
        self.load_inquirer = SimpleNamespace(get_loads=Mock())
        self.enable_fpm = False
        self._record_step_counters = Mock()
        self._maybe_clear_mm_inputs = Mock()
        self.maybe_send_health_check_signal = Mock()
        self.disagg_prefill_inflight_queue = []
        self.send_kv_chunk = Mock()
        self.process_pending_chunked_abort = Mock()
        self.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=lambda batch: batch
        )

    def _process_hicache_events(self, should_retry_storage_prefetch=True):
        pass

    def _poll_timeout_aborts(self):
        return []

    def ingest_requests(self, stop_at_pause=False):
        # GPU completion and incoming traffic advance even when intake is held.
        self.iteration += 1
        if self.iteration == len(self.arrivals):
            raise _StopLoop
        self.depths.append(len(self.result_queue))
        for name, iteration in self.ready_at.items():
            event = self.copy_events[name]
            event.ready = event.ready or self.iteration >= iteration
        self._incoming_requests.extend(self._inputs_for_step())
        return Scheduler.ingest_requests(self, stop_at_pause=stop_at_pause)

    def recv_requests(self, local_reqs=None):
        self.actions.append((self.iteration, "receive", None))
        if self.tp_cpu_group is not None:
            # A completion decision must never reorder the next intake collective.
            tick = torch.tensor([self.iteration])
            dist.broadcast(tick, src=0, group=self.tp_cpu_group)
            assert tick.item() == self.iteration
        inputs = self._incoming_requests
        self._incoming_requests = []
        return inputs + (local_reqs or [])

    def _inputs_for_step(self):
        inputs = []
        if self.iteration == self.pause_at:
            inputs.append(PauseGenerationReqInput(mode=self.pause_mode))
        if self.iteration == self.resume_at:
            inputs.append(ContinueGenerationReqInput(torch_empty_cache=False))
        if (batch := self.arrivals[self.iteration]) is not None:
            inputs.append(_input(batch))
        return inputs

    def process_input_requests(self, recv_reqs):
        for req in recv_reqs:
            name = (
                req.rid
                if isinstance(req, TokenizedGenerateReqInput)
                else type(req).__name__
            )
            self.actions.append((self.iteration, "dispatch", name))
            if isinstance(req, TokenizedGenerateReqInput):
                self.ready_batches.append(self.batches[int(req.rid)])
            elif isinstance(req, PauseGenerationReqInput):
                assert len(self.result_queue) <= 1, "Pause bypassed a result wait"
                self.pause_generation(req)
            elif isinstance(req, ContinueGenerationReqInput):
                self.continue_generation(req)
            elif isinstance(req, ShutdownReq):
                Scheduler.handle_shutdown(self, req)
            else:
                raise AssertionError(f"Unexpected input: {req}")

    def _record_scheduler_state_for_paused_engine(self):
        self.actions.append((self.iteration, "pause", None))
        self.paused_states.append((self.last_batch, tuple(self.result_queue)))

    def process_prefill_chunk(self, last_batch, running_batch):
        if last_batch is not None:
            self.chunk_processes[last_batch.forward_iter] += 1
        SchedulerDisaggregationPrefillMixin.process_prefill_chunk(
            self, last_batch=last_batch, running_batch=running_batch
        )

    def get_new_batch_prefill(self, running_batch):
        self.admission_attempts.append(self.iteration)
        batch = self.ready_batches.popleft() if self.ready_batches else None
        if batch is not None:
            self.actions.append((self.iteration, "prepare_batch", batch.forward_iter))
            self.waiting_queue = [
                req for req in self.waiting_queue if req not in batch.reqs
            ]
        return NextBatchPlan(batch_to_run=batch, running_batch=running_batch)

    def run_batch(self, batch):
        assert len(self.result_queue) < 2, "Overwrote a live forward's keep-alive slot"
        self.actions.append((self.iteration, "launch", batch.forward_iter))
        result = self.results[batch.forward_iter]
        if result.delay_sample_func is not None:

            def sample():
                self.actions.append((self.iteration, "sample", batch.forward_iter))
                return result

            result.delay_sample_func = sample
        return result

    def _apply_war_barrier(self):
        pass

    def process_batch_result(self, batch, result):
        assert all(pending is not result for _, pending in self.result_queue)
        self.result_processes[batch.forward_iter] += 1
        self.actions.append((self.iteration, "process_result", batch.forward_iter))
        Scheduler.process_batch_result(self, batch, result)
        event = self.copy_events[batch.forward_iter]
        assert event.ready and event.recorded
        assert result.delay_sample_func is None
        self.actions.append((self.iteration, "result", batch.forward_iter))

    def process_disagg_prefill_inflight_queue(self):
        self.actions.append((self.iteration, "inflight", None))

    def on_idle(self):
        self.actions.append((self.iteration, "idle", None))


def _run_loop(scheduler, enabled=True, **topology):
    with (
        published_topology(**topology),
        envs.SGLANG_ENABLE_DISAGG_PREFILL_CONTINUOUS_INPUT_POLLING.override(enabled),
        patch("sglang.srt.disaggregation.prefill.checkpoint_kv_cache"),
    ):
        scheduler.tp_size = get_parallel().tp_size
        try:
            _dispatch_event_loop_once(scheduler)
        except _StopLoop:
            return
        if scheduler.gracefully_exit:
            return
    raise AssertionError("Scheduler exited before all arrivals were polled")


@contextmanager
def _single_rank_gloo_group():
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _rank_consensus_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        scheduler = _Scheduler(
            [_batch(1), None, _batch(2)] + [None] * 6,
            {name: _result() for name in (1, 2)},
            {1: (2, 5)[rank], 2: (7, 6)[rank]},
        )
        scheduler.tp_cpu_group = dist.group.WORLD
        _run_loop(scheduler, tp_size=2, ranks={"world_rank": rank})
        finished = [a for a in scheduler.actions if a[1] == "result"]
        assert finished == [(5, "result", 1), (7, "result", 2)], finished
        assert max(scheduler.depths) == 2
        for event in scheduler.copy_events.values():
            assert event.synchronizations == 1
            assert event.blocking_synchronizations == 0

        for depth in (1, 2):
            for mode in ("in_place", "retract"):
                arrivals = [
                    _batch(1),
                    _batch(2) if depth == 2 else None,
                    None,
                    _batch(3),
                ]
                names = (1, 2, 3) if depth == 2 else (1, 3)
                scheduler = _Scheduler(
                    arrivals + [None] * 8,
                    {
                        name: _result(_Event(recorded=False), delayed=True)
                        for name in names
                    },
                    {
                        1: (3, 5)[rank],
                        3: (10, 9)[rank],
                        **({2: (8, 7)[rank]} if depth == 2 else {}),
                    },
                    pause_at=2,
                    resume_at=7,
                    pause_mode=mode,
                )
                scheduler.tp_cpu_group = dist.group.WORLD
                _run_loop(scheduler, tp_size=2, ranks={"world_rank": rank})
                assert (5, "result", 1) in scheduler.actions
                assert (6, "dispatch", "PauseGenerationReqInput") in scheduler.actions
                assert (6, "dispatch", "3") in scheduler.actions
                assert (7, "launch", 3) in scheduler.actions
                assert not set(range(2, 7)).intersection(scheduler.admission_attempts)
                assert all(
                    (step, "receive", None) in scheduler.actions for step in range(3, 7)
                )
                assert scheduler.result_processes == {name: 1 for name in names}
                assert [a[2] for a in scheduler.actions if a[1] == "result"] == list(
                    names
                )
                assert not scheduler._deferred_input_requests
                for name, event in scheduler.copy_events.items():
                    assert event.synchronizations == 1
                    assert event.blocking_synchronizations == int(
                        name == 2 and mode == "retract"
                    )
                    assert event.records == 1

        # An unsubmitted copy on either rank is a lifecycle error on every rank.
        with published_topology(tp_size=2, ranks={"world_rank": rank}):
            batch_result_completion_status = torch.empty(1, dtype=torch.int32)
            for missing_event in (True, False):
                result = _result()
                if rank == 0:
                    if missing_event:
                        result.copy_done = None
                    else:
                        result.delay_sample_func = lambda: None
                        result.copy_done.recorded = False
                scheduler.result_queue = deque([(_batch(99), result)])
                try:
                    scheduler.is_disagg_prefill_batch_result_ready(
                        result, batch_result_completion_status
                    )
                except RuntimeError as error:
                    assert "must be submitted" in str(error)
                else:
                    raise AssertionError("An unsubmitted copy was polled")
    finally:
        dist.destroy_process_group()


class TestPrefillResultPolling(CustomTestCase):
    def test_intake_retains_pause_suffix_while_continuing_to_receive(self):
        for stop_at_pause in (False, True):
            for prefix_size in (0, 1):
                with self.subTest(stop_at_pause=stop_at_pause, prefix_size=prefix_size):
                    prefix = [object()] * prefix_size
                    suffix = [
                        PauseGenerationReqInput(mode="retract"),
                        object(),
                        ShutdownReq(),
                    ]
                    newer = [object() for _ in range(4)]
                    receiver = Mock(
                        side_effect=[prefix + suffix, *[[req] for req in newer]]
                    )
                    scheduler = SimpleNamespace(
                        _deferred_input_requests=[],
                        _poll_timeout_aborts=Mock(return_value=[]),
                        request_receiver=SimpleNamespace(recv_requests=receiver),
                        metrics_reporter=Mock(),
                        process_input_requests=Mock(),
                    )
                    with published_topology():
                        received = [
                            Scheduler.ingest_requests(scheduler, stop_at_pause=stop)
                            for stop in (stop_at_pause,) * 3 + (False, False)
                        ]
                    self.assertEqual(
                        received,
                        [prefix, [], [], suffix + newer[:3], newer[3:]]
                        if stop_at_pause
                        else [prefix + suffix, *[[req] for req in newer]],
                    )
                    self.assertEqual(receiver.call_count, 5)
                    # A held pause must not accumulate repeated timeout aborts.
                    self.assertEqual(
                        scheduler._poll_timeout_aborts.call_count,
                        2 if stop_at_pause else 5,
                    )
                    self.assertFalse(scheduler._deferred_input_requests)
                    dispatched = [
                        req
                        for args in scheduler.process_input_requests.call_args_list
                        for req in args.args[0]
                    ]
                    self.assertEqual(dispatched, prefix + suffix + newer)

    def test_deferred_pause_stops_preparation_with_a_free_forward_slot(self):
        class ReadyAfterPauseScheduler(_Scheduler):
            def get_new_batch_prefill(self, running_batch):
                if self.iteration == 1:
                    self.admission_attempts.append(self.iteration)
                    return NextBatchPlan(batch_to_run=None, running_batch=running_batch)
                return super().get_new_batch_prefill(running_batch)

        scheduler = ReadyAfterPauseScheduler(
            [_batch(1), _batch(2), None, _batch(3)] + [None] * 8,
            {name: _result() for name in (1, 2, 3)},
            {1: 4, 2: 8, 3: 10},
            pause_at=2,
            resume_at=6,
        )
        _run_loop(scheduler)

        self.assertIn((4, "result", 1), scheduler.actions)
        self.assertIn((5, "dispatch", "PauseGenerationReqInput"), scheduler.actions)
        self.assertIn((5, "dispatch", "3"), scheduler.actions)
        self.assertIn((6, "launch", 2), scheduler.actions)
        for iteration in range(2, 6):
            self.assertNotIn(iteration, scheduler.admission_attempts)
        for iteration in range(3, 6):
            self.assertIn((iteration, "receive", None), scheduler.actions)
        self.assertEqual(scheduler.result_processes, {1: 1, 2: 1, 3: 1})
        self.assertFalse(scheduler._deferred_input_requests)

    @unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
    def test_empty_polls_and_arrival_bursts_preserve_storage_retry_pacing(self):
        class CapacityBlockedScheduler(_Scheduler):
            _process_hicache_events = Scheduler._process_hicache_events
            _process_storage_prefetch_retries = (
                Scheduler._process_storage_prefetch_retries
            )
            _retry_storage_prefetch = Scheduler._retry_storage_prefetch

            def _prefetch_kvcache(self, req, storage_hit_end):
                self.reissues.append(self.iteration)
                self.tree_cache.storage_prefetch_retries.poll_miss(req.rid)

            def get_new_batch_prefill(self, running_batch):
                if self.iteration > 0 and not self.result_processes[1]:
                    self.admission_attempts.append(self.iteration)
                    self.policy.calc_priority(self.waiting_queue, running_batch)
                    return NextBatchPlan(batch_to_run=None, running_batch=running_batch)
                return super().get_new_batch_prefill(running_batch)

        for arrivals_during_wait in (False, True):
            with self.subTest(arrivals_during_wait=arrivals_during_wait):
                arrivals = [_batch(1)] + [None] * 13
                if arrivals_during_wait:
                    arrivals[2:12] = [_batch(i) for i in range(2, 12)]
                scheduler = CapacityBlockedScheduler(arrivals, {1: _result()}, {1: 13})
                scheduler.enable_hierarchical_cache = True
                scheduler.enable_hicache_storage = True
                scheduler.reissues = []
                retries = StoragePrefetchRetries()
                radix_cache = RadixCache.create_simulated()
                matched_at = []

                def match_prefix(params):
                    matched_at.append(scheduler.iteration)
                    return radix_cache.match_prefix(params)

                scheduler.tree_cache = SimpleNamespace(
                    check_hicache_events=Mock(),
                    flush_pending_backups=Mock(),
                    storage_prefetch_retries=retries,
                    supports_fast_match_prefix=lambda: True,
                    swa_reprefill_tail_tokens=lambda: 0,
                    match_prefix=match_prefix,
                )
                scheduler.policy = SchedulePolicy(
                    "fcfs", scheduler.tree_cache, False, False, False
                )
                scheduler.ngram_embedding_manager.prepare_for_forward = Mock(
                    side_effect=lambda batch, chunked_req: batch
                )
                head, paced = (_batch(i).reqs[0] for i in (90, 91))
                for req in (head, paced):
                    req.origin_input_ids = array("q", req.origin_input_ids)
                    req.pending_bootstrap = False
                    req.disagg_kv_sender = SimpleNamespace(
                        poll=lambda: KVPoll.WaitingForInput
                    )
                scheduler.waiting_queue = [head, paced]
                retries.poll_miss(paced.rid)

                with _single_rank_gloo_group():
                    _run_loop(scheduler, hicache_storage_prefetch_retry_poll_interval=2)

                attempts = list(range(12)) if arrivals_during_wait else [0, 1]
                self.assertEqual(scheduler.admission_attempts, attempts)
                self.assertEqual(
                    matched_at, [step for step in attempts[1:] for _ in range(2)]
                )
                self.assertEqual(
                    scheduler.ngram_embedding_manager.prepare_for_forward.call_count,
                    len(attempts),
                )
                # Empty polls still receive inputs and query completion after a miss.
                self.assertEqual(scheduler.copy_events[1].queries, 13)
                self.assertEqual(
                    [
                        step
                        for step, action, _ in scheduler.actions
                        if action == "receive"
                    ],
                    list(range(14)),
                )
                self.assertEqual(scheduler.reissues, [])
                self.assertEqual(paced.storage_prefetch_retry_attempts, 0)
                self.assertIn((13, "result", 1), scheduler.actions)
                # The next outer admission still gets the retry due at step three.
                self.assertEqual(
                    retries.pop_ready([head, paced], 2, 8), [(paced, None)]
                )

    def test_deferred_readiness_retries_after_completion_or_while_idle(self):
        class ReadinessBlockedScheduler(_Scheduler):
            _process_hicache_events = Scheduler._process_hicache_events

            def process_input_requests(self, inputs):
                super().process_input_requests(inputs)
                if self.readiness_source == "grammar" and any(
                    isinstance(req, TokenizedGenerateReqInput) and req.rid == "2"
                    for req in inputs
                ):
                    req = self.ready_batches.pop().reqs[0]
                    req.grammar = self.grammar_future
                    req.grammar_key = ("regex", "[ab]")
                    self.grammar_manager.grammar_queue.append(req)

            def _inputs_for_step(self):
                if self.readiness_source == "grammar" and self.iteration == 3:
                    self.grammar_future.set_result(self.compiled_grammar)
                return super()._inputs_for_step()

            def get_new_batch_prefill(self, running_batch):
                if self.readiness_source == "grammar":
                    if self.grammar_manager.has_waiting_grammars():
                        ready = self.grammar_manager.get_ready_grammar_requests()
                        self.waiting_queue.extend(ready)
                        if ready:
                            self.ready_batches.append(self.batches[2])
                elif (
                    self.ready_batches
                    and self.iteration > 0
                    and not self.prefetch_ready
                ):
                    self.admission_attempts.append(self.iteration)
                    return NextBatchPlan(batch_to_run=None, running_batch=running_batch)
                return super().get_new_batch_prefill(running_batch)

        for readiness_source in ("cache", "grammar"):
            for pending_forward, enabled in (
                (False, False),
                (False, True),
                (True, False),
                (True, True),
            ):
                with self.subTest(
                    readiness_source=readiness_source,
                    pending_forward=pending_forward,
                    continuous_input_polling=enabled,
                ):
                    names = (1, 2) if pending_forward else (2,)
                    scheduler = ReadinessBlockedScheduler(
                        [_batch(1) if pending_forward else None, _batch(2)]
                        + [None] * 8,
                        {name: _result() for name in names},
                        {2: 7, **({1: 5} if pending_forward else {})},
                    )
                    scheduler.readiness_source = readiness_source
                    scheduler.prefetch_ready = False
                    scheduler.enable_hierarchical_cache = readiness_source == "cache"
                    scheduler.enable_unified_cache_external_linker = False
                    scheduler.enable_lmcache = False
                    scheduler.enable_hicache_storage = False

                    def drain_cache_events():
                        # Readiness is visible only after the real cache-drain path.
                        if scheduler.iteration >= 3:
                            scheduler.prefetch_ready = True

                    scheduler.tree_cache = SimpleNamespace(
                        check_hicache_events=drain_cache_events,
                        flush_pending_backups=Mock(),
                        finish=Mock(),
                    )
                    if readiness_source == "grammar":
                        scheduler.grammar_future = Future()
                        scheduler.compiled_grammar = Mock(
                            spec=["copy", "accept_token", "finished"]
                        )
                        with published_topology(skip_tokenizer_init=True):
                            scheduler.grammar_manager = GrammarManager(
                                SimpleNamespace(
                                    server_args=SimpleNamespace(),
                                    dp_tp_cpu_group=None,
                                    dp_tp_group=SimpleNamespace(
                                        world_size=1, first_rank=0, is_first_rank=True
                                    ),
                                    pp_group=None,
                                )
                            )
                        scheduler.grammar_manager.grammar_backend = Mock(
                            spec=["set_cache"]
                        )
                    _run_loop(scheduler, enabled=enabled)

                    # Long-tail readiness waits for completion, but an empty pipeline
                    # retries normally and must not retain a failed-creation marker.
                    deferred = pending_forward and enabled
                    self.assertIn(
                        (6 if deferred else 3, "launch", 2), scheduler.actions
                    )
                    if deferred:
                        self.assertEqual(
                            [step for step in scheduler.admission_attempts if step < 6],
                            [0, 1],
                        )
                    else:
                        self.assertEqual(scheduler.admission_attempts[:4], [0, 1, 2, 3])
                    if pending_forward:
                        self.assertIn(
                            (5 if enabled else 1, "result", 1), scheduler.actions
                        )
                    self.assertEqual(
                        scheduler.result_processes, {name: 1 for name in names}
                    )
                    if readiness_source == "grammar":
                        self.assertIs(
                            scheduler.batches[2].reqs[0].grammar,
                            scheduler.compiled_grammar,
                        )
                        self.assertFalse(
                            scheduler.grammar_manager.has_waiting_grammars()
                        )

    @unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
    def test_bootstrap_completion_admits_without_new_input(self):
        second = _batch(2)
        req = second.reqs[0]
        req.pending_bootstrap = False
        req.disagg_kv_sender = SimpleNamespace(poll=lambda: KVPoll.WaitingForInput)
        scheduler = _Scheduler(
            [_batch(1)] + [None] * 7,
            {1: _result(), 2: _result()},
            {1: 5, 2: 6},
        )

        def pop_bootstrapped():
            if scheduler.iteration == 3:
                scheduler.ready_batches.append(second)
                return [req]
            return []

        scheduler.disagg_prefill_bootstrap_queue.pop_bootstrapped = pop_bootstrapped
        with _single_rank_gloo_group():
            _run_loop(scheduler)

        self.assertIn((3, "launch", 2), scheduler.actions)
        self.assertIn((5, "result", 1), scheduler.actions)
        self.assertEqual(scheduler.result_processes, {1: 1, 2: 1})

    @unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
    def test_transfer_retirement_waits_for_batch_result_before_freeing_metadata(self):
        second = _batch(2)

        class MetadataBlockedScheduler(_Scheduler):
            process_disagg_prefill_inflight_queue = SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue

            def run_batch(self, batch):
                result = super().run_batch(batch)
                if batch.forward_iter == 1:
                    self.ready_batches.append(second)
                    self.waiting_queue.extend(second.reqs)
                return result

            def get_new_batch_prefill(self, running_batch):
                if self.iteration > 0 and second in self.ready_batches:
                    index = self.req_to_metadata_buffer_idx_allocator.alloc()
                    if index is None:
                        self.admission_attempts.append(self.iteration)
                        return NextBatchPlan(
                            batch_to_run=None, running_batch=running_batch
                        )
                    second.reqs[0].metadata_buffer_index = index
                return super().get_new_batch_prefill(running_batch)

        first = _batch(1)
        scheduler = MetadataBlockedScheduler(
            [first] + [None] * 7,
            {1: _result(), 2: _result()},
            {1: 5, 2: 6},
        )
        allocator = ReqToMetadataIdxAllocator(2)
        scheduler.req_to_metadata_buffer_idx_allocator = allocator
        scheduler.output_streamer = SimpleNamespace(stream_output=Mock())
        prior = _batch(0).reqs[0]
        prior.metadata_buffer_index = allocator.alloc()
        prior.pending_bootstrap = False
        prior.bootstrap_host = FAKE_BOOTSTRAP_HOST
        prior.disagg_kv_sender = SimpleNamespace(
            poll=lambda: (
                KVPoll.Success if scheduler.iteration >= 3 else KVPoll.Transferring
            ),
            clear=Mock(),
        )
        scheduler.disagg_prefill_inflight_queue = [prior]
        first.reqs[0].metadata_buffer_index = allocator.alloc()
        for req in (first.reqs[0], second.reqs[0]):
            req.pending_bootstrap = False
            req.disagg_kv_sender = SimpleNamespace(poll=lambda: KVPoll.Transferring)
        with (
            _single_rank_gloo_group(),
            patch("sglang.srt.disaggregation.prefill.release_kv_cache") as release_kv,
        ):
            _run_loop(scheduler)

        release_kv.assert_called_once_with(prior, scheduler.tree_cache, checkpoint=True)
        self.assertEqual(prior.metadata_buffer_index, -1)
        self.assertEqual(second.reqs[0].metadata_buffer_index, 0)
        self.assertIn((6, "launch", 2), scheduler.actions)
        self.assertIn((5, "result", 1), scheduler.actions)
        self.assertEqual(scheduler.admission_attempts[:3], [0, 1, 6])
        self.assertEqual(scheduler.copy_events[1].blocking_synchronizations, 0)
        self.assertEqual(scheduler.result_processes, {1: 1, 2: 1})

    def test_input_polling_continues_at_both_queue_depths(self):
        scheduler = _Scheduler(
            [_batch(1), None, _batch(2), _batch(3)] + [None] * 6,
            {name: _result() for name in (1, 2, 3)},
            {1: 6, 2: 4, 3: 8},
        )
        _run_loop(scheduler)

        finished = [a for a in scheduler.actions if a[1] == "result"]
        self.assertEqual(
            finished,
            [
                (6, "result", 1),
                (7, "result", 2),
                (8, "result", 3),
            ],
        )
        self.assertIn((2, "launch", 2), scheduler.actions)
        self.assertIn((7, "launch", 3), scheduler.actions)
        self.assertEqual(
            [a for a in scheduler.actions if a[1] == "prepare_batch"],
            [(0, "prepare_batch", 1), (2, "prepare_batch", 2), (7, "prepare_batch", 3)],
        )
        self.assertEqual(max(scheduler.depths), 2)
        self.assertEqual(scheduler.chunk_processes, {1: 1, 2: 1, 3: 1})
        self.assertEqual(scheduler.result_processes, {1: 1, 2: 1, 3: 1})
        self.assertEqual(scheduler.load_publisher.publish_load_stat.call_count, 3)
        self.assertEqual(scheduler.tree_cache.flush_pending_backups.call_count, 3)
        self.assertEqual(scheduler._record_step_counters.call_count, 3)
        for iteration in range(1, 8):
            self.assertIn((iteration, "receive", None), scheduler.actions)
            self.assertNotIn((iteration, "idle", None), scheduler.actions)
        self.assertEqual(
            [a for a in scheduler.actions if a[1] == "process_result"],
            [
                (6, "process_result", 1),
                (7, "process_result", 2),
                (8, "process_result", 3),
            ],
        )
        self.assertFalse(scheduler.result_queue)
        self.assertIsNone(scheduler.last_batch)
        self.assertEqual(
            [req.rid for req in scheduler.disagg_prefill_inflight_queue],
            ["1", "2", "3"],
        )
        self.assertEqual(
            [list(req.output_ids) for req in scheduler.disagg_prefill_inflight_queue],
            [[11], [11], [11]],
        )
        for event in scheduler.copy_events.values():
            self.assertEqual(event.synchronizations, 1)
            self.assertEqual(event.blocking_synchronizations, 0)

    def test_waiting_and_paused_tp1_passes_allow_worker_progress(self):
        scheduler = _Scheduler(
            [_batch(1)] + [None] * 9,
            {1: _result()},
            {1: 5},
            pause_at=3,
            resume_at=8,
        )
        parked_passes = []

        def park(interval_s):
            self.assertGreater(interval_s, 0)
            self.assertLessEqual(interval_s, 0.001)
            parked_passes.append(scheduler.iteration)

        with patch("sglang.srt.disaggregation.prefill.time.sleep", side_effect=park):
            _run_loop(scheduler)

        # Copy waits 1..4 and paused passes 6..7 must let workers reacquire the GIL.
        self.assertEqual(parked_passes, [1, 2, 3, 4, 6, 7])

    def test_new_batches_process_each_forward_once(self):
        for overlap, enabled in ((False, False), (True, False), (True, True)):
            with self.subTest(overlap=overlap, continuous_input_polling=enabled):
                first, second = _batch(1), _batch(2)
                results = {i: _result(_Event(ready=True)) for i in (1, 2)}
                scheduler = _Scheduler(
                    [first, None, None, second, None, None], results, ready_at={}
                )
                scheduler.enable_overlap = overlap
                _run_loop(scheduler, enabled=enabled)

                self.assertEqual(scheduler.chunk_processes, {1: 1, 2: 1})
                self.assertEqual(scheduler.result_processes, {1: 1, 2: 1})
                self.assertEqual(
                    scheduler.load_publisher.publish_load_stat.call_count, 2
                )
                first_done = 1 if overlap else 0
                self.assertEqual(
                    [a for a in scheduler.actions if a[1] == "result"],
                    [
                        (first_done, "result", 1),
                        (first_done + 3, "result", 2),
                    ],
                )
                self.assertFalse(scheduler.result_queue)

    def test_delayed_sample_is_submitted_after_predecessor_before_query(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                scheduler = _Scheduler(
                    [_batch(1), _batch(2)] + [None] * 5,
                    {
                        1: _result(_Event(recorded=False), delayed=True),
                        2: _result(_Event(recorded=False), delayed=True),
                    },
                    {1: 4, 2: 5},
                )
                _run_loop(scheduler, enabled=enabled)
                first_done, second_done = (4, 5) if enabled else (1, 2)
                self.assertIn((0, "sample", 1), scheduler.actions)
                self.assertLess(
                    scheduler.actions.index((first_done, "result", 1)),
                    scheduler.actions.index((first_done, "sample", 2)),
                )
                self.assertIn((second_done, "result", 2), scheduler.actions)
                for name, result in scheduler.results.items():
                    event = scheduler.copy_events[name]
                    self.assertEqual(event.records, 1)
                    self.assertIsNone(result.delay_sample_func)
                    self.assertEqual(event.synchronizations, 1)
                    self.assertEqual(event.blocking_synchronizations, int(not enabled))

    def test_in_place_pause_accepts_inputs_and_preserves_pending_results(self):
        for third_at in (3, 4):
            with self.subTest(third_at=third_at):
                first, second, third = _batch(1), _batch(2), _batch(3)
                arrivals = [first, second] + [None] * 8
                arrivals[third_at] = third
                scheduler = _Scheduler(
                    arrivals,
                    {name: _result() for name in (1, 2, 3)},
                    {1: 3, 2: 6, 3: 8},
                    pause_at=3,
                    resume_at=5,
                )
                _run_loop(scheduler)

                self.assertEqual(len(scheduler.paused_states), 1)
                for last_batch, pending in scheduler.paused_states:
                    self.assertIs(last_batch, second)
                    self.assertEqual([b.forward_iter for b, _ in pending], [2])
                self.assertEqual(
                    [a for a in scheduler.actions if a[1] == "dispatch"],
                    [
                        (0, "dispatch", "1"),
                        (1, "dispatch", "2"),
                        (4, "dispatch", "PauseGenerationReqInput"),
                        (4, "dispatch", "3"),
                        (5, "dispatch", "ContinueGenerationReqInput"),
                    ],
                )
                self.assertEqual(
                    [a for a in scheduler.actions if a[1] == "result"],
                    [(3, "result", 1), (6, "result", 2), (8, "result", 3)],
                )
                self.assertIn((5, "launch", 3), scheduler.actions)
                self.assertEqual(scheduler.result_processes, {1: 1, 2: 1, 3: 1})
                for event in scheduler.copy_events.values():
                    self.assertEqual(event.synchronizations, 1)
                    self.assertEqual(event.blocking_synchronizations, 0)

    def test_retract_during_copy_wait_finishes_each_result_once(self):
        for enabled, second_at in ((False, None), (True, None), (True, 1), (True, 2)):
            with self.subTest(enabled=enabled, second_at=second_at):
                pending_count = 1 if second_at is None else 2
                pause_at = 3 if enabled else 1
                results = {
                    name: _result(_Event(recorded=False), delayed=True)
                    for name in range(1, pending_count + 1)
                }
                results[3] = _result()
                arrivals = [None] * 12
                arrivals[0] = _batch(1)
                if second_at is not None:
                    arrivals[second_at] = _batch(2)
                arrivals[5] = _batch(3)
                scheduler = _Scheduler(
                    arrivals,
                    results,
                    {
                        1: 4,
                        **({2: 20} if pending_count == 2 else {}),
                        3: 10,
                    },
                    pause_at=pause_at,
                    resume_at=7,
                    pause_mode="retract",
                )
                _run_loop(scheduler, enabled=enabled)

                self.assertEqual(
                    [a for a in scheduler.actions if a[1] == "result"],
                    [(4 if enabled else 1, "result", 1)]
                    + ([(5, "result", 2)] if pending_count == 2 else [])
                    + [(10 if enabled else 8, "result", 3)],
                )
                self.assertEqual(
                    scheduler.result_processes, {name: 1 for name in results}
                )
                self.assertEqual(
                    scheduler.chunk_processes,
                    # Pause finishes results; resume owns the next chunk preparation.
                    {1: 1, 3: 1} if enabled else {3: 1},
                )
                self.assertEqual(
                    [
                        list(req.output_ids)
                        for req in scheduler.disagg_prefill_inflight_queue
                    ],
                    [[11]] * len(results),
                )
                for name, event in scheduler.copy_events.items():
                    self.assertEqual(event.synchronizations, 1)
                    self.assertEqual(
                        event.blocking_synchronizations, int(name == 2 or not enabled)
                    )
                    if name != 3:
                        self.assertEqual(event.records, 1)
                        self.assertIsNone(results[name].delay_sample_func)
                self.assertFalse(scheduler.result_queue)
                self.assertIsNone(scheduler.last_batch)

    def test_shutdown_during_intake_stops_new_batches_in_both_modes(self):
        class StoppingScheduler(_Scheduler):
            def _inputs_for_step(self):
                inputs = super()._inputs_for_step()
                if self.iteration == shutdown_at:
                    inputs.insert(0, ShutdownReq())
                return inputs

        for enabled in (False, True):
            for shutdown_at in (0, 1, 2):
                with self.subTest(enabled=enabled, shutdown_at=shutdown_at):
                    scheduler = StoppingScheduler(
                        [_batch(1), _batch(2), _batch(3), None],
                        {
                            name: _result(
                                _Event(ready=True, recorded=False), delayed=True
                            )
                            for name in (1, 2, 3)
                        },
                        {},
                    )
                    _run_loop(scheduler, enabled=enabled)

                    self.assertTrue(scheduler.gracefully_exit)
                    self.assertEqual(
                        scheduler.admission_attempts, list(range(shutdown_at))
                    )
                    self.assertEqual(
                        [a[2] for a in scheduler.actions if a[1] == "launch"],
                        list(range(1, shutdown_at + 1)),
                    )
                    self.assertEqual(
                        scheduler.result_processes,
                        {name: 1 for name in range(1, shutdown_at + 1)},
                    )
                    self.assertFalse(scheduler.result_queue)

    def test_shutdown_during_copy_wait_finishes_oldest_result_before_exiting(self):
        class StoppingScheduler(_Scheduler):
            def _inputs_for_step(self):
                inputs = super()._inputs_for_step()
                if self.iteration == 2:
                    inputs.insert(0, ShutdownReq())
                return inputs

        for pending_count, pause_on_shutdown in (
            (1, False),
            (1, True),
            (2, False),
            (2, True),
        ):
            with self.subTest(
                pending_count=pending_count, pause_on_shutdown=pause_on_shutdown
            ):
                second = _batch(2) if pending_count == 2 else None
                scheduler = StoppingScheduler(
                    [_batch(1), second, _batch(3), None, None],
                    {name: _result() for name in (1, 2, 3)},
                    {name: 20 for name in (1, 2, 3)},
                    pause_at=2 if pause_on_shutdown else None,
                )
                _run_loop(scheduler)

                self.assertTrue(scheduler.gracefully_exit)
                self.assertEqual(
                    [a for a in scheduler.actions if a[1] == "launch"],
                    [
                        (name - 1, "launch", name)
                        for name in range(1, pending_count + 1)
                    ],
                )
                self.assertEqual(scheduler.result_processes, {1: 1})
                self.assertEqual(scheduler.copy_events[1].synchronizations, 1)
                self.assertEqual(scheduler.copy_events[1].blocking_synchronizations, 1)
                self.assertIn((2, "dispatch", "ShutdownReq"), scheduler.actions)
                self.assertEqual(
                    [batch.forward_iter for batch, _ in scheduler.result_queue],
                    [2] if pending_count == 2 else [],
                )
                self.assertEqual(
                    [req.rid for req in scheduler.disagg_prefill_inflight_queue], ["1"]
                )
                self.assertEqual(
                    list(scheduler.disagg_prefill_inflight_queue[0].output_ids), [11]
                )

    def test_shutdown_behind_deferred_pause_keeps_input_order(self):
        class StoppingScheduler(_Scheduler):
            def _inputs_for_step(self):
                inputs = super()._inputs_for_step()
                if self.iteration == 2:
                    inputs.append(ShutdownReq())
                return inputs

        for depth in (1, 2):
            for mode in ("in_place", "retract"):
                with self.subTest(depth=depth, mode=mode):
                    second = _batch(2) if depth == 2 else None
                    scheduler = StoppingScheduler(
                        [_batch(1), second, _batch(3)] + [None] * 5,
                        {name: _result() for name in (1, 2, 3)},
                        {1: 4, 2: 20, 3: 20},
                        pause_at=2,
                        pause_mode=mode,
                    )
                    _run_loop(scheduler)

                    self.assertTrue(scheduler.gracefully_exit)
                    self.assertEqual(
                        [a for a in scheduler.actions if a[1] == "dispatch"][-3:],
                        [
                            (5, "dispatch", "PauseGenerationReqInput"),
                            (5, "dispatch", "3"),
                            (5, "dispatch", "ShutdownReq"),
                        ],
                    )
                    self.assertIn((4, "result", 1), scheduler.actions)
                    self.assertEqual(
                        [a[2] for a in scheduler.actions if a[1] == "launch"],
                        list(range(1, depth + 1)),
                    )
                    self.assertEqual(
                        scheduler.result_processes,
                        {1: 1, 2: 1} if depth == 2 and mode == "retract" else {1: 1},
                    )
                    self.assertEqual(
                        [batch.forward_iter for batch, _ in scheduler.result_queue],
                        [2] if depth == 2 and mode == "in_place" else [],
                    )
                    self.assertFalse(scheduler._deferred_input_requests)

    def test_only_disabled_and_unsupported_modes_keep_blocking_results(self):
        for mode in (
            "disabled",
            "independent_dp",
            "attention_dp",
            "cp",
            "speculation",
            "embedding",
            "dllm",
        ):
            with self.subTest(mode=mode):
                scheduler = _Scheduler([_batch(1), None, None], {1: _result()}, {1: 2})
                topology = {}
                if mode == "independent_dp":
                    topology = {"dp_size": 2}
                elif mode == "attention_dp":
                    topology = {"dp_size": 2, "tp_size": 2, "enable_dp_attention": True}
                elif mode == "cp":
                    topology = {"attn_cp_size": 2, "tp_size": 2}
                elif mode == "speculation":
                    scheduler.spec_algorithm = SpeculativeAlgorithm.EAGLE
                elif mode == "embedding":
                    scheduler.is_generation = False
                elif mode == "dllm":
                    scheduler.dllm_config = object()
                _run_loop(scheduler, enabled=mode != "disabled", **topology)
                blocking = mode != "independent_dp"
                self.assertIn((1 if blocking else 2, "result", 1), scheduler.actions)
                self.assertEqual(scheduler.copy_events[1].queries == 0, blocking)
                self.assertEqual(
                    scheduler.copy_events[1].blocking_synchronizations, int(blocking)
                )

    def test_dispatch_clears_polling_state_outside_prefill_overlap(self):
        scheduler = _Scheduler([], {}, {})
        for mode, pp_size, overlap, loop in (
            (DisaggregationMode.PREFILL, 2, True, "event_loop_pp_disagg_prefill"),
            (DisaggregationMode.PREFILL, 1, False, "event_loop_normal_disagg_prefill"),
            (DisaggregationMode.DECODE, 1, True, "event_loop_overlap_disagg_decode"),
        ):
            with self.subTest(mode=mode, pp_size=pp_size, overlap=overlap):
                scheduler.disaggregation_mode = mode
                scheduler.enable_overlap = overlap
                scheduler.enable_continuous_input_polling = True
                with (
                    published_topology(pp_size=pp_size),
                    patch.object(scheduler, loop, create=True) as run_loop,
                    patch.object(
                        scheduler, "_is_continuous_input_polling_enabled"
                    ) as check_support,
                ):
                    _dispatch_event_loop_once(scheduler)
                run_loop.assert_called_once_with()
                check_support.assert_not_called()
                self.assertFalse(scheduler.enable_continuous_input_polling)

    def test_chunk_handoffs_across_polling_and_retract_pause(self):
        for enabled, pause_at, blocking_ends in (
            (True, None, ()),
            (False, 1, (4,)),
            (True, 1, (4,)),
            (True, 2, (8,)),
            (True, 4, (12,)),
        ):
            with self.subTest(enabled=enabled, pause_at=pause_at):
                req = Req("chunked", "", list(range(12)), SamplingParams())
                req.metadata_buffer_index = 1
                req.pending_bootstrap = False
                batches = [_batch(end, [req]) for end in (4, 8, 12)]
                results = {
                    b.forward_iter: GenerationBatchResult(
                        next_token_ids=torch.tensor([11]), copy_done=_Event()
                    )
                    for b in batches
                }
                checkpoint_ends = []

                class ChunkScheduler(_Scheduler):
                    def run_batch(self, batch):
                        end = int(batch.forward_iter)
                        req.extend_end = end
                        self.chunked_req = req if end < 12 else None
                        batch.chunked_req = self.chunked_req
                        if self.chunked_req is not None:
                            req.inflight_middle_chunks += 1
                        return super().run_batch(batch)

                    def checkpoint_disagg_prefill(self, req):
                        checkpoint_ends.append(req.extend_end)
                        super().checkpoint_disagg_prefill(req)

                if pause_at == 1:
                    arrivals = [batches[0], None, None, batches[1], batches[2]]
                    ready_at = {4: 20, 8: 7, 12: 8}
                elif pause_at == 2:
                    arrivals = [batches[0], batches[1], None, None, batches[2]]
                    ready_at = {4: 3, 8: 20, 12: 8}
                elif pause_at == 4:
                    # The second pending result is the request's final chunk.
                    arrivals = [batches[0], batches[1], None, batches[2], None]
                    ready_at = {4: 1, 8: 5, 12: 20}
                else:
                    arrivals = [batches[0], None, batches[1], None, batches[2]]
                    ready_at = {4: 4, 8: 6, 12: 8}
                scheduler = ChunkScheduler(
                    arrivals + [None] * 5,
                    results,
                    ready_at,
                    pause_at=pause_at,
                    resume_at=pause_at + 2 if pause_at is not None else None,
                    pause_mode="retract",
                )
                _run_loop(scheduler, enabled=enabled)

                self.assertEqual(
                    scheduler.send_kv_chunk.call_args_list,
                    [
                        call(req, last_chunk=False, end_idx=4),
                        call(req, last_chunk=False, end_idx=8),
                        call(req, last_chunk=True),
                    ],
                )
                # Pause must not take a cache snapshot that resume repeats.
                self.assertEqual(checkpoint_ends, [4, 8, 12])
                self.assertEqual(req.inflight_middle_chunks, 0)
                self.assertEqual(list(req.output_ids), [11])
                self.assertEqual(scheduler.disagg_prefill_inflight_queue, [req])
                self.assertEqual(scheduler.result_processes, {4: 1, 8: 1, 12: 1})
                for end, event in scheduler.copy_events.items():
                    self.assertEqual(event.synchronizations, 1)
                    self.assertEqual(
                        event.blocking_synchronizations,
                        int(not enabled or end in blocking_ends),
                    )

    @unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
    def test_retract_resume_requeues_optimistic_chunk_after_result_finishes(self):
        for depth, yield_before_pause in ((1, False), (2, False), (2, True)):
            with self.subTest(depth=depth, yield_before_pause=yield_before_pause):
                req = Req("optimistic", "", list(range(12)), SamplingParams())
                req.metadata_buffer_index = 1
                req.pending_bootstrap = True
                req.kv.req_pool_idx = 1
                req.kv.kv_allocated_len = 8
                req.disagg_kv_sender = SimpleNamespace(
                    poll=lambda: KVPoll.Bootstrapping
                )
                chunk_count = 1 if yield_before_pause else depth
                chunks = [_batch(end, [req]) for end in (4, 8)[:chunk_count]]
                other = _batch(100)
                other.reqs[0].disagg_kv_sender = SimpleNamespace(
                    poll=lambda: KVPoll.WaitingForInput
                )

                class OptimisticScheduler(_Scheduler):
                    _release_aborted_request = Scheduler._release_aborted_request

                    def run_batch(self, batch):
                        if req in batch.reqs:
                            req.extend_end = batch.forward_iter
                            self.chunked_req = batch.chunked_req = req
                            req.inflight_middle_chunks += 1
                        return super().run_batch(batch)

                    def process_input_requests(self, inputs):
                        super().process_input_requests(inputs)
                        if any(
                            isinstance(item, TokenizedGenerateReqInput)
                            and item.rid == "100"
                            for item in inputs
                        ):
                            self.waiting_queue.extend(other.reqs)

                pause_at = 2 if yield_before_pause else (1 if depth == 1 else 3)
                ready_at = {**{batch.forward_iter: 20 for batch in chunks}, 100: 7}
                if depth == 2:
                    ready_at[4] = pause_at + 1
                scheduler = OptimisticScheduler(
                    chunks + [other] + [None] * 8,
                    {batch.forward_iter: _result() for batch in [*chunks, other]},
                    ready_at,
                    pause_at=pause_at,
                    resume_at=pause_at + 2,
                    pause_mode="retract",
                )
                scheduler.disagg_prefill_pending_chunk_rids = set()
                scheduler.processed_tokens_counter = 0
                scheduler.metrics_reporter.enable_metrics = False
                release_steps = []

                def release(req, tree_cache, checkpoint):
                    self.assertEqual(req.inflight_middle_chunks, 0)
                    release_steps.append(scheduler.iteration)
                    req.kv.req_pool_idx = None
                    req.kv.mark_kv_released()

                with (
                    _single_rank_gloo_group(),
                    patch(
                        "sglang.srt.disaggregation.prefill.release_kv_cache",
                        side_effect=release,
                    ) as release_kv,
                ):
                    _run_loop(scheduler, optimistic_prefill_attempts=2)

                release_kv.assert_called_once_with(
                    req, scheduler.tree_cache, checkpoint=False
                )
                self.assertEqual(
                    release_steps, [3 if depth == 1 or yield_before_pause else 5]
                )
                self.assertEqual(scheduler.waiting_queue, [req])
                self.assertEqual(req.prefill_attempt_count, 1)
                self.assertEqual(req.inflight_middle_chunks, 0)
                self.assertEqual(list(req.output_ids), [])
                self.assertIsNone(scheduler.chunked_req)
                self.assertEqual(
                    scheduler.result_processes,
                    {b.forward_iter: 1 for b in [*chunks, other]},
                )
                scheduler.send_kv_chunk.assert_called_once_with(
                    other.reqs[0], last_chunk=True
                )

    @unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
    def test_ranks_agree_on_fifo_completion_and_copy_submission(self):
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = str(Path(directory) / "rendezvous")
            processes = [
                context.Process(target=_rank_consensus_worker, args=(rank, rendezvous))
                for rank in range(2)
            ]
            try:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(timeout=60)
                self.assertEqual([process.exitcode for process in processes], [0, 0])
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
