"""CPU coverage for continuous input polling and coordinated prefill completion."""

import multiprocessing
import tempfile
import unittest
from array import array
from collections import Counter, deque
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch
import torch.distributed as dist

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.prefill import DisaggPrefill
from sglang.srt.disaggregation.req_state import DisaggReqState
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
)
from sglang.srt.environ import envs
from sglang.srt.managers import scheduler_assembly
from sglang.srt.managers.io_struct import (
    ContinueGenerationReqInput,
    PauseGenerationReqInput,
    ShutdownReq,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.request_lists import CoreLists
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import SchedulePolicy
from sglang.srt.managers.scheduler import Scheduler, _dispatch_event_loop_once
from sglang.srt.managers.scheduler_assembly import init_mode_stages
from sglang.srt.managers.scheduler_components.engine_state import (
    EngineFlags,
    LaunchCounters,
)
from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
from sglang.srt.managers.scheduler_components.mode_stages import ScheduleSlot
from sglang.srt.managers.scheduler_components.pending_steps import PendingSteps
from sglang.srt.managers.scheduler_components.request_retirer import RequestRetirer
from sglang.srt.managers.scheduler_components.step_launcher import StepLauncher
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.disagg_owner_shells import result_processor_shell
from sglang.test.scheduler_request_intake import make_request_intake
from sglang.test.scheduler_request_lists import (
    assert_one_list_per_request,
    request_registry_of,
    step_planner_of,
)
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


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
    for req in reqs:
        # The bootstrap queue attaches it when it takes the request.
        if req.disagg is None:
            req.disagg = DisaggReqState()
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


class _Owner(DisaggPrefill):
    """The PD prefill owner over the fake scheduler's collaborators, which the
    tests replace on the scheduler; the KV transfer poll only records."""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.bootstrap_queue = SimpleNamespace(pop_bootstrapped=lambda: [], queue=[])
        self.prefilling = []
        self.transfer = []
        self.pending_chunk_rids = set()
        self.kv_senders = {}
        self.result_completion_status = torch.empty(1, dtype=torch.int32)
        self.enable_staging = False
        self.kv_checksum_computer = None
        self.counters = scheduler.counters

    def after_commit(self, waiting, rids_to_check=None):
        self.scheduler.actions.append((self.scheduler.iteration, "inflight", None))
        return waiting


for _name, _source in (
    ("tp_size", "tp_size"),
    ("tp_cpu_group", "tp_cpu_group"),
    ("attn_cp_cpu_group", "attn_cp_cpu_group"),
    ("attn_tp_cpu_group", "attn_tp_cpu_group"),
    ("tree_cache", "tree_cache"),
    ("enable_overlap", "enable_overlap"),
    ("spec_algorithm", "spec_algorithm"),
    ("scheduler_stage_metrics", "scheduler_stage_metrics"),
    ("output_streamer", "output_streamer"),
    ("metrics_sink", "metrics_sink"),
    ("metrics_collector", "metrics_collector"),
    ("batch_result_processor", "batch_result_processor"),
    ("send_kv_chunk", "send_kv_chunk"),
    ("request_retirer", "request_retirer"),
    ("core_lists", "lists"),
):
    setattr(_Owner, _name, property(lambda self, n=_source: getattr(self.scheduler, n)))


class _Launcher:
    """The real delayed sample and its host copy start; the forward is the
    scheduler double's run_batch."""

    launch_batch_sample_if_needed = StepLauncher.launch_batch_sample_if_needed
    _copy_result_to_cpu = StepLauncher._copy_result_to_cpu

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.forward_stream_ctx = nullcontext()
        self.copy_stream_ctx = nullcontext()
        self.forward_stream = Mock(spec=["wait_stream"])
        self.copy_stream = Mock(spec=["wait_stream"])
        self.async_d2h_worker = None
        self.schedule_stream = object()
        self._relay_beam_tokens = Mock()

    def run_batch(self, batch):
        return self.scheduler.run_batch(batch)


class _Scheduler:
    """Run real scheduling and result handling with controlled intake and GPU events."""

    event_loop_normal_disagg_prefill = Scheduler.event_loop_normal_disagg_prefill
    event_loop_overlap_disagg_prefill = Scheduler.event_loop_overlap_disagg_prefill
    schedule_step = Scheduler.schedule_step
    stash_chunked_request = Scheduler.stash_chunked_request
    finalize_step = Scheduler.finalize_step
    loop_schedules = Scheduler.loop_schedules
    # The stages the loop's stage table names.
    _admit_prefill = Scheduler._admit_prefill
    _admit_prefill_after_interval = Scheduler._admit_prefill_after_interval
    _admit_dllm_step = Scheduler._admit_dllm_step
    _select_decode = Scheduler._select_decode
    process_bootstrapped_queue = Scheduler.process_bootstrapped_queue
    process_prefill_transfer_queue = Scheduler.process_prefill_transfer_queue
    paused_pass = Scheduler.paused_pass
    _yield_gil_if_needed = Scheduler._yield_gil_if_needed
    pause_generation = Scheduler.pause_generation
    continue_generation = Scheduler.continue_generation
    run_flat_loop = Scheduler.run_flat_loop
    is_disable_overlap_for_batch = Scheduler.is_disable_overlap_for_batch
    may_disable_overlap = Scheduler.may_disable_overlap
    launch_step = Scheduler.launch_step
    join_launched_step = Scheduler.join_launched_step
    commit_step = Scheduler.commit_step
    commit_pending_steps = Scheduler.commit_pending_steps
    leave_finished_reqs = Scheduler.leave_finished_reqs
    disagg_prefill = Scheduler.disagg_prefill
    disagg_decode = Scheduler.disagg_decode
    dllm = Scheduler.dllm
    hisparse_coordinator = Scheduler.hisparse_coordinator
    # The double plays the scheduler and the parts, mode and hub it holds.
    parts = mode = hub = property(lambda self: self)

    @property
    def kv(self):
        return SimpleNamespace(
            tree_cache=self.tree_cache, token_to_kv_pool_allocator=None
        )

    def init_mode_stages(self):
        self.stages = init_mode_stages(self, self.loop_schedules(), lora_ids_in_use=set)

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
        self.result_processes = Counter()
        self.depths = []
        self.scheduler_stage_metrics = None
        self.scripted_scheduler_hook = None
        self.disaggregation_mode = DisaggregationMode.PREFILL
        self.spec_algorithm = SpeculativeAlgorithm.NONE
        self.is_generation = True
        self.mode_owners = ModeOwners()
        self.tp_cpu_group = None
        self.tp_size = 1
        self.attn_cp_cpu_group = None
        self.attn_tp_cpu_group = None
        self.pending = PendingSteps()
        self.lists = CoreLists()
        self.decode_offload_manager = None
        self.flat_slot = ScheduleSlot()
        self.recv_skipper = None
        self._sched_idled = False
        self.counters = LaunchCounters()
        self.flags = EngineFlags()
        # Config leaves, applied to the bags once a topology is published.
        self.enable_overlap = True
        self.enable_hierarchical_cache = False
        self.enable_unified_cache_external_linker = False
        self.require_mlp_sync = False
        self.grammar_manager = SimpleNamespace(grammar_queue=[], abort_requests=Mock())
        self.ngram_embedding_manager = SimpleNamespace(
            prepare_for_forward=lambda batch, chunked_req: batch
        )
        self.step_launcher = _Launcher(self)
        self.tree_cache = Mock(spec=["flush_pending_backups", "finish"])
        self.batch_result_processor = result_processor_shell()
        self.metrics_reporter = Mock(spec=["report_prefill_stats"])
        self.metrics_sink = Mock(
            spec=[
                "log_batch_result_stats",
                "update_device_timer",
                "record_scheduler_active",
            ]
        )
        self.metrics_sink.current_scheduler_metrics_enabled = False
        self.metrics_sink.enable_fpm = False
        self.kv_events_publisher = Mock(spec=["publish_kv_events"])
        self.publish_load_snapshot = Mock()
        self.load_publisher = Mock(spec=["publish_load_stat"])
        self.load_inquirer = SimpleNamespace(get_loads=Mock())
        self.step_counters = Mock()
        self.request_retirer = RequestRetirer(
            kv=SimpleNamespace(tree_cache=self.tree_cache),
            beam_coordinator=None,
            hisparse_coordinator=None,
            decode_offload_manager=None,
            pending=PendingSteps(),
            output_streamer=None,
            ipc_channels=None,
        )
        self.request_retirer.clear_finished_media = Mock()
        self.output_streamer = Mock(spec=["maybe_reply_health_check"])
        self.send_kv_chunk = Mock()
        self.process_pending_chunked_abort = Mock()
        self.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=lambda batch, need_sync=None: batch,
            maybe_set_extend_view=lambda batch: None,
        )
        self.mode_owners.disagg_prefill = _Owner(self)

    def apply_config(self):
        get_context().override(
            "test",
            disable_overlap_schedule=not self.enable_overlap,
            enable_hierarchical_cache=self.enable_hierarchical_cache,
            enable_unified_cache_external_linker=(
                self.enable_unified_cache_external_linker
            ),
        )

    def _process_hicache_events(self, should_retry_storage_prefetch=True):
        pass

    @property
    def request_registry(self):
        # Over the collaborators as the tests last replaced them.
        return request_registry_of(self)

    # Grammar readiness is the admission double's own (get_new_batch_prefill).
    request_intake = SimpleNamespace(requeue_ready_grammars=lambda: None)

    @property
    def step_planner(self):
        # The real pause retraction over the same collaborators; admission is
        # this fake's own.
        return step_planner_of(
            self,
            get_new_batch_prefill=self.get_new_batch_prefill,
            arm_prefill_decode_interval=lambda step: None,
        )

    def ingest_requests(self, stop_at_pause=False):
        assert_one_list_per_request(self)
        # GPU completion and incoming traffic advance even when intake is held.
        self.iteration += 1
        if self.iteration == len(self.arrivals):
            raise _StopLoop
        self.depths.append(len(self.pending))
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
                assert len(self.pending) <= 1, "Pause bypassed a result wait"
                self.pause_generation(req)
            elif isinstance(req, ContinueGenerationReqInput):
                self.continue_generation(req)
            elif isinstance(req, ShutdownReq):
                Scheduler.handle_shutdown(self, req)
            else:
                raise AssertionError(f"Unexpected input: {req}")

    def _record_scheduler_state_for_paused_engine(self):
        self.actions.append((self.iteration, "pause", None))
        self.paused_states.append(tuple(record.step for record in self.pending))

    def get_new_batch_prefill(self, slot):
        self.admission_attempts.append(self.iteration)
        batch = self.ready_batches.popleft() if self.ready_batches else None
        if batch is not None:
            self.actions.append((self.iteration, "prepare_batch", batch.forward_iter))
        return batch

    def run_batch(self, batch):
        assert len(self.pending) < 2, "Overwrote a live forward's keep-alive slot"
        self.actions.append((self.iteration, "launch", batch.forward_iter))
        result = self.results[batch.forward_iter]
        if result.delay_sample_func is not None:

            def sample():
                self.actions.append((self.iteration, "sample", batch.forward_iter))
                return result

            result.delay_sample_func = sample
        return result

    war_barrier_is_coarse = True

    def _war_barrier_is_coarse(self):
        return self.war_barrier_is_coarse

    def _apply_war_barrier(self):
        self.actions.append((self.iteration, "barrier", None))

    def process_batch_result(self, batch, result):
        assert all(record.result is not result for record in self.pending)
        self.result_processes[batch.forward_iter] += 1
        self.actions.append((self.iteration, "process_result", batch.forward_iter))
        Scheduler.process_batch_result(self, batch, result)
        event = self.copy_events[batch.forward_iter]
        assert event.ready and event.recorded
        assert result.delay_sample_func is None
        self.actions.append((self.iteration, "result", batch.forward_iter))

    def on_idle(self):
        self.actions.append((self.iteration, "idle", None))


def _run_loop(scheduler, enabled=True, **topology):
    with (
        published_topology(**topology),
        envs.SGLANG_ENABLE_DISAGG_PREFILL_CONTINUOUS_INPUT_POLLING.override(enabled),
        patch(
            "sglang.srt.disaggregation.prefill.checkpoint_kv_cache",
            side_effect=lambda req, cache: setattr(req, "prefix_len", req.extend_end),
        ),
    ):
        scheduler.tp_size = get_parallel().tp_size
        scheduler.apply_config()
        scheduler.init_mode_stages()
        try:
            _dispatch_event_loop_once(scheduler)
        except _StopLoop:
            return
        if scheduler.flags.gracefully_exit:
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
            for missing_event in (True, False):
                result = _result()
                if rank == 0:
                    if missing_event:
                        result.copy_done = None
                    else:
                        result.delay_sample_func = lambda: None
                        result.copy_done.recorded = False
                try:
                    scheduler.disagg_prefill.is_result_ready(result)
                except RuntimeError as error:
                    assert "must be submitted" in str(error)
                else:
                    raise AssertionError("An unsubmitted copy was polled")
    finally:
        dist.destroy_process_group()


class TestPrefillResultPolling(CustomTestCase):
    @unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
    def test_empty_polls_and_arrival_bursts_preserve_storage_retry_pacing(self):
        class CapacityBlockedScheduler(_Scheduler):
            _process_hicache_events = Scheduler._process_hicache_events

            def get_new_batch_prefill(self, slot):
                if self.iteration > 0 and not self.result_processes[1]:
                    self.admission_attempts.append(self.iteration)
                    self.policy.calc_priority(
                        self.lists.waiting_queue, self.lists.decoding
                    )
                    return None
                return super().get_new_batch_prefill(slot)

        for arrivals_during_wait in (False, True):
            with self.subTest(arrivals_during_wait=arrivals_during_wait):
                arrivals = [_batch(1)] + [None] * 13
                if arrivals_during_wait:
                    arrivals[2:12] = [_batch(i) for i in range(2, 12)]
                scheduler = CapacityBlockedScheduler(arrivals, {1: _result()}, {1: 13})
                scheduler.enable_hierarchical_cache = True
                storage = patch(
                    "sglang.srt.managers.scheduler.hicache_storage_enabled",
                    return_value=True,
                )
                storage.start()
                self.addCleanup(storage.stop)
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
                    swa_reprefill_tail_tokens=lambda req=None: 0,
                    match_prefix=match_prefix,
                )
                # The intake reads its config when built; _run_loop publishes
                # the loop's own.
                with published_topology():
                    scheduler.request_intake = make_request_intake(
                        core_lists=scheduler.lists,
                        kv=SimpleNamespace(
                            tree_cache=scheduler.tree_cache,
                            token_to_kv_pool_allocator=None,
                        ),
                    )

                def reissue(req, storage_hit_end):
                    scheduler.reissues.append(scheduler.iteration)
                    retries.poll_miss(req.rid)

                scheduler.request_intake._prefetch_kvcache = reissue
                scheduler.policy = SchedulePolicy(
                    "fcfs", scheduler.tree_cache, False, False, False
                )
                scheduler.ngram_embedding_manager.prepare_for_forward = Mock(
                    side_effect=lambda batch, chunked_req: batch
                )
                head, paced = (_batch(i).reqs[0] for i in (90, 91))
                for req in (head, paced):
                    req.origin_input_ids = array("q", req.origin_input_ids)
                    req.disagg.pending_bootstrap = False
                    scheduler.disagg_prefill.kv_senders[req.rid] = SimpleNamespace(
                        poll=lambda: KVPoll.WaitingForInput
                    )
                scheduler.lists.waiting_queue = [head, paced]
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

    def test_chunk_processing_runs_once_per_launched_step(self):
        """A polling pass that schedules again while the same step is the last
        one launched does not repeat its chunk processing (bootstrap check,
        optimistic yield); a pass after a step without a launch does."""
        first, second = _batch(1), _batch(2)
        results = {i: _result(_Event()) for i in (1, 2)}
        scheduler = _Scheduler(
            [first, None, second, None, None, None],
            results,
            ready_at={1: 3, 2: 4},
        )
        chunk_passes = []

        def process_prefill_chunk(chunked_req, waiting, pending):
            chunk_passes.append(scheduler.iteration)
            return chunked_req, waiting

        scheduler.disagg_prefill.process_prefill_chunk = process_prefill_chunk
        _run_loop(scheduler, enabled=True)

        # Pass 2 schedules (the second request arrives) with the first step
        # still the last one launched: its chunk was processed in pass 1.
        self.assertEqual(chunk_passes, [0, 1, 4, 5])
        self.assertEqual(scheduler.result_processes, {1: 1, 2: 1})

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

                    self.assertTrue(scheduler.flags.gracefully_exit)
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
                    self.assertFalse(scheduler.pending)

    def test_only_disabled_and_unsupported_modes_keep_blocking_results(self):
        for mode in (
            "disabled",
            "independent_dp",
            "attention_dp",
            "cp",
            "speculation",
            "embedding",
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
                _run_loop(scheduler, enabled=mode != "disabled", **topology)
                blocking = mode != "independent_dp"
                self.assertIn((1 if blocking else 2, "result", 1), scheduler.actions)
                self.assertEqual(scheduler.copy_events[1].queries == 0, blocking)
                self.assertEqual(
                    scheduler.copy_events[1].blocking_synchronizations, int(blocking)
                )

    def test_readiness_gate_only_under_the_overlap_flat_prefill_driver(self):
        scheduler = _Scheduler([], {}, {})
        for pp_size, overlap, loop, gated in (
            (2, True, "event_loop_pp", False),
            (1, False, "event_loop_normal_disagg_prefill", False),
            (1, True, "event_loop_overlap_disagg_prefill", True),
        ):
            with self.subTest(pp_size=pp_size, overlap=overlap):
                scheduler.enable_overlap = overlap
                with (
                    published_topology(pp_size=pp_size),
                    patch.object(scheduler, loop, create=True) as run_loop,
                    patch.object(
                        scheduler_assembly,
                        "continuous_input_polling_enabled",
                        return_value=True,
                    ) as check_support,
                ):
                    scheduler.apply_config()
                    scheduler.init_mode_stages()
                    _dispatch_event_loop_once(scheduler)
                run_loop.assert_called_once_with()
                self.assertEqual(check_support.called, gated)
                self.assertEqual(scheduler.stages.is_result_ready is not None, gated)

    def test_coarse_war_barrier_follows_previous_result_processing(self):
        steps = ("launch", "process_result", "inflight", "barrier")
        for coarse, polling, ready_at, expected in (
            # A whole-forward barrier waits until the previous result is processed.
            (True, False, 1, ["launch", "process_result", "inflight", "barrier"]),
            (True, True, 1, ["launch", "process_result", "inflight", "barrier"]),
            # Polling for an unfinished result still applies it before intake.
            (True, True, 2, ["launch", "barrier"]),
            # A read-done barrier keeps its place right after the launch.
            (False, False, 1, ["launch", "barrier", "process_result", "inflight"]),
            (False, True, 2, ["launch", "barrier"]),
        ):
            with self.subTest(coarse=coarse, polling=polling, ready_at=ready_at):
                scheduler = _Scheduler(
                    [_batch(1), _batch(2)] + [None] * 4,
                    {name: _result() for name in (1, 2)},
                    {1: ready_at, 2: ready_at + 1},
                )
                scheduler.war_barrier_is_coarse = coarse
                _run_loop(scheduler, enabled=polling)
                # Step 1 launches batch 2 while batch 1's result is pending.
                self.assertEqual(
                    [a[1] for a in scheduler.actions if a[0] == 1 and a[1] in steps],
                    expected,
                )

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
                req.disagg = DisaggReqState(
                    metadata_buffer_index=1, pending_bootstrap=False
                )
                batches = [_batch(end, [req]) for end in (4, 8, 12)]
                for start, batch in zip((0, 4, 8), batches):
                    batch.prefix_lens = [start]
                    batch.extend_lens = [batch.forward_iter - start]
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
                        batch.chunked_req = req if end < 12 else None
                        return super().run_batch(batch)

                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        checkpoint = self.disagg_prefill.checkpoint

                        def record(req):
                            checkpoint_ends.append(req.extend_end)
                            checkpoint(req)

                        self.disagg_prefill.checkpoint = record

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
                self.assertEqual(list(req.output_ids), [11])
                self.assertEqual(scheduler.disagg_prefill.transfer, [req])
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
                req.disagg = DisaggReqState(
                    metadata_buffer_index=1, pending_bootstrap=True
                )
                req.kv.req_pool_idx = 1
                req.kv.kv_allocated_len = 8
                chunk_count = 1 if yield_before_pause else depth
                chunks = [_batch(end, [req]) for end in (4, 8)[:chunk_count]]
                other = _batch(100)

                class OptimisticScheduler(_Scheduler):
                    def run_batch(self, batch):
                        if req in batch.reqs:
                            req.extend_end = batch.forward_iter
                            batch.chunked_req = req
                        return super().run_batch(batch)

                    def process_input_requests(self, inputs):
                        super().process_input_requests(inputs)
                        if any(
                            isinstance(item, TokenizedGenerateReqInput)
                            and item.rid == "100"
                            for item in inputs
                        ):
                            self.lists.waiting_queue.extend(other.reqs)

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
                scheduler.request_retirer = RequestRetirer(
                    kv=SimpleNamespace(tree_cache=scheduler.tree_cache),
                    beam_coordinator=None,
                    hisparse_coordinator=None,
                    decode_offload_manager=None,
                    pending=PendingSteps(),
                    output_streamer=None,
                    ipc_channels=None,
                )
                scheduler.disagg_prefill.pending_chunk_rids = set()
                scheduler.disagg_prefill.kv_senders.update(
                    {
                        req.rid: SimpleNamespace(poll=lambda: KVPoll.Bootstrapping),
                        other.reqs[0].rid: SimpleNamespace(
                            poll=lambda: KVPoll.WaitingForInput
                        ),
                    }
                )
                scheduler.counters.processed_tokens_counter = 0
                scheduler.metrics_sink.enable_metrics = False
                release_steps = []

                def release(req, tree_cache, checkpoint):
                    # No launched chunk of the request is still in flight.
                    self.assertFalse(
                        any(r.step.chunked_req is req for r in scheduler.pending)
                    )
                    release_steps.append(scheduler.iteration)
                    req.kv.req_pool_idx = None
                    req.kv.mark_kv_released()

                with (
                    _single_rank_gloo_group(),
                    patch(
                        "sglang.srt.managers.scheduler_components.request_retirer.release_kv_cache",
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
                self.assertEqual(scheduler.lists.waiting_queue, [req])
                self.assertEqual(req.disagg.prefill_attempt_count, 1)
                self.assertEqual(list(req.output_ids), [])
                self.assertIsNone(scheduler.lists.chunked_req)
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


class _ForwardLaunched(Exception):
    pass


class _EarlySendScheduler(SchedulerDisaggregationPrefillMixin):
    """Just enough Scheduler state for run_batch to reach the forward launch."""

    def __init__(self):
        self.scheduler_stage_metrics = None
        self.metrics_reporter = SimpleNamespace(record_scheduler_active=Mock())
        self.forward_ct = 0
        self._sched_idled = False
        self.processed_tokens_counter = 0
        self.scripted_scheduler_hook = None
        self.profiler_manager = SimpleNamespace(_profile_batch_predicate=Mock())
        self.forward_sleep_time = None
        self.disaggregation_mode = DisaggregationMode.PREFILL
        self.enable_staging = False
        self.enable_overlap = False
        self.token_to_kv_pool_allocator = SimpleNamespace(page_size=64)
        self.send_kv_chunk = Mock()

    @property
    def is_generation(self):
        raise _ForwardLaunched


class TestCachedPrefixEarlySend(CustomTestCase):
    def test_batch_that_starts_a_hicache_load_back_does_not_early_send(self):
        """A request can match pages whose HiCache load-back was submitted in the same
        build; early-sending them would read KV the H2D has not written yet (#44070)."""
        for consumer_index, expect_send in ((-1, True), (0, False)):
            scheduler = _EarlySendScheduler()
            req = SimpleNamespace(
                prefix_len=128,
                host_hit_length=0,
                start_send_idx=0,
                pending_bootstrap=False,
                early_send_prefix_end=None,
            )
            batch = SimpleNamespace(
                reqs=[req],
                hicache_consumer_index=consumer_index,
                extend_num_tokens=32,
                forward_mode=SimpleNamespace(
                    is_split_prefill=lambda: False, is_prebuilt=lambda: False
                ),
            )
            with self.subTest(consumer_index=consumer_index):
                with envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.override(True):
                    with self.assertRaises(_ForwardLaunched):
                        Scheduler.run_batch(scheduler, batch)
                if expect_send:
                    scheduler.send_kv_chunk.assert_called_once_with(
                        req, last_chunk=False, end_idx=128
                    )
                else:
                    scheduler.send_kv_chunk.assert_not_called()


if __name__ == "__main__":
    unittest.main()
