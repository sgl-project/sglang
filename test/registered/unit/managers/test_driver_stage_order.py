"""The stage order each driver runs per mode: the flat driver (normal, overlap,
PD prefill with and without the commit readiness gate, PD decode, dLLM) and the
PP ring (plain, PD prefill, PD decode), over a few scripted iterations on fakes.

The fakes stand at the owners and at the scheduling and commit leaves, so the
order pinned here is the drivers' and the schedule stages', not the leaves'.
"""

import itertools
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_request_lists import make_request_retirer
from sglang.test.test_utils import (
    CustomTestCase,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.request_lists import CoreLists  # noqa: E402
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch  # noqa: E402
from sglang.srt.managers.scheduler import (  # noqa: E402
    _dispatch_event_loop_once,
)
from sglang.srt.managers.scheduler_assembly import init_mode_stages  # noqa: E402
from sglang.srt.managers.scheduler_components.bundles import Streams  # noqa: E402
from sglang.srt.managers.scheduler_components.engine_state import (  # noqa: E402
    EngineFlags,
    LaunchCounters,
)
from sglang.srt.managers.scheduler_components.mode_owners import (  # noqa: E402
    ModeOwners,
)
from sglang.srt.managers.scheduler_components.mode_stages import (  # noqa: E402
    ScheduleSlot,
)
from sglang.srt.managers.scheduler_components.pending_steps import (  # noqa: E402
    PendingSteps,
)
from sglang.srt.managers.scheduler_components.pp_ring import PPRing  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.runtime_context import get_context, get_parallel  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402
from sglang.test.scheduler_request_lists import partial_scheduler  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Stop(Exception):
    pass


def _step(name, mode=ForwardMode.EXTEND):
    req = Req(name, "", [1, 2, 3], SamplingParams())
    step = ScheduleBatch(
        reqs=[req], forward_mode=mode, spec_algorithm=SpeculativeAlgorithm.NONE
    )
    step.name = name
    return step


class _Result:
    def __init__(self, name, delayed, ready_at):
        self.name = name
        self.delay_sample_func = (lambda: self) if delayed else None
        self.ready_at = ready_at


class _PrefillOwner:
    def __init__(self, shell):
        self.shell = shell
        self.transfer = []

    def intake(self):
        self.shell.log("dp.intake")
        return []

    def resolve_waiting_bootstrap(self, waiting):
        self.shell.log("dp.resolve")
        return waiting

    def process_prefill_chunk(self, chunked_req, waiting, pending):
        self.shell.log("dp.chunk")
        return chunked_req, waiting

    def before_forward(self, step):
        # The fake run_batch stands for the forward and what runs before it.
        raise AssertionError("not reached")

    def on_launch(self, reqs):
        self.shell.log("dp.on_launch", *[r.rid for r in reqs])

    def after_commit(self, waiting, rids_to_check=None):
        self.shell.log("dp.after_commit")
        return waiting

    def commit_extend(self, step, result):
        self.shell.log("dp.commit_extend", step.name)

    def is_result_ready(self, result, *status):
        ready = self.shell.iteration >= result.ready_at
        self.shell.log("dp.ready", result.name, ready)
        return ready

    def poll_transfer_rids(self):
        self.shell.log("dp.poll_transfer")
        return []


class _DecodeOwner:
    def __init__(self, shell):
        self.shell = shell
        self.ready = []
        self.transfer_queue = SimpleNamespace(queue=[])
        self.prealloc_queue = SimpleNamespace(queue=[])

    def prefetch_prefill_dp_rank_queries(self):
        self.shell.log("dd.prefetch")

    def intake(self, *, lora_ids_in_use):
        self.shell.log("dd.intake")

    def join_ready(self, slot):
        self.shell.log("join", slot.index)
        return []

    def check_hicache_events(self):
        self.shell.log("hicache")

    def retracted_rids(self, mb_id):
        self.shell.log("dd.retracted")
        return []

    def poll_prealloc_rids(self):
        self.shell.log("dd.poll_prealloc")
        return [], []

    def aborted_prealloc_rids(self):
        return []

    def poll_transfer_rids(self):
        self.shell.log("dd.poll_transfer")
        return []

    def apply_retract_consensus(self, rids, slot):
        self.shell.log("dd.apply_retract", slot.index)
        return rids

    def apply_prealloc_consensus(self, rids, slot, *, lora_ids_in_use):
        self.shell.log("dd.apply_prealloc", slot.index)
        return rids

    def apply_transfer_consensus(self, rids, slot):
        self.shell.log("dd.apply_transfer", slot.index)
        return rids


class _DllmOwner:
    def __init__(self, shell):
        self.shell = shell

    def schedule(self, waiting):
        step = self.shell.script("prefill")
        self.shell.log("dllm.schedule")
        return step, waiting

    def on_launch(self, step):
        self.shell.log("dllm.on_launch", step.name)

    def commit_extend(self, step, result):
        self.shell.log("dllm.commit_extend", step.name)


class _HiSparse:
    def __init__(self, shell):
        self.shell = shell

    def on_launch(self, reqs):
        self.shell.log("hisparse.on_launch", *[r.rid for r in reqs])

    def leave_finished(self):
        pass

    def admit_request_into_staging(self, req):
        self.shell.log("hisparse.stage", req.rid)

    def free_remapped_decode_slots(self):
        self.shell.log("hisparse.free_remapped")

    def join_ready(self, slot, *, future_map, device):
        self.shell.log("join", slot.index)
        return []


class _Hook:
    def __init__(self, shell):
        self.shell = shell

    def on_schedule(self, step, *slot):
        self.shell.log("hook.on_schedule")


class _Shell:
    """A Scheduler built with __new__ whose leaves record into one log."""

    def __init__(
        self,
        mode,
        iterations,
        prefill=(),
        decode=(),
        ready_at=None,
        shutdown_at=None,
        paused=(),
        hisparse=False,
    ):
        self.events = []
        self.groups = {}
        self.shutdown_at = shutdown_at
        # The iterations whose intake leaves the engine paused.
        self.paused = set(paused)
        self.iteration = -1
        self.iterations = iterations
        self.prefill = dict(prefill)
        self.decode = dict(decode)
        self.ready_at = ready_at or {}
        self.results = {}
        s = self.s = partial_scheduler()
        s.parts.mode_owners = ModeOwners()
        s.parts.counters = LaunchCounters()
        s.parts.flags = EngineFlags()
        s.parts.lists = CoreLists()
        s.parts.flags.gracefully_exit = False
        s.parts.flags._engine_paused = False
        s._deferred_input_requests = []
        s.scheduler_stage_metrics = None
        s.parts.lists.waiting_queue = []
        s.parts.lists.decoding = []
        s.parts.lists.chunked_req = None
        s.parts.pending = PendingSteps()
        s.parts.request_retirer = make_request_retirer(pending=s.parts.pending)
        s.flat_slot = ScheduleSlot()
        s.parts.flags.enable_skip_finishing_decode = False
        s._sched_idled = False
        s.parts.counters.cur_batch_for_debug = None
        s.parts.recv_skipper = None
        s.enable_overlap = False
        s.enable_unified_memory = False
        s.parts.metrics_sink = SimpleNamespace(enable_fpm=False)
        s.enable_hisparse = False
        s.parts.mode_owners.hisparse_coordinator = _HiSparse(self) if hisparse else None
        s.parts.future_map = None
        s.parts.streams.device = "cpu"
        s.parts.require_mlp_sync = False
        s.parts.is_generation = True
        s.tp_size = 2
        s.parts.spec_algorithm = SpeculativeAlgorithm.NONE
        s.parts.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        s.parts.scripted_scheduler_hook = _Hook(self)
        s.parts.mode_owners.dllm = None
        s.parts.mode_owners.disagg_prefill = None
        s.parts.mode_owners.disagg_decode = None
        s.mode.disaggregation_mode = DisaggregationMode.NULL
        if mode == "prefill":
            s.parts.mode_owners.disagg_prefill = _PrefillOwner(self)
            s.mode.disaggregation_mode = DisaggregationMode.PREFILL
        elif mode == "decode":
            s.parts.mode_owners.disagg_decode = _DecodeOwner(self)
            s.mode.disaggregation_mode = DisaggregationMode.DECODE
            s.parts.decode_offload_manager = None
        elif mode == "dllm":
            s.parts.mode_owners.dllm = _DllmOwner(self)

        s.ingest_requests = self.ingest_requests
        s._record_scheduler_state_for_paused_engine = lambda: self.log("paused")
        s._yield_gil_if_needed = lambda: None
        s.process_pending_chunked_abort = lambda: self.log("chunked_abort")
        s._start_fpm_clock = lambda: self.log("fpm")
        s.stash_chunked_request = lambda: self.log("stash")
        s._process_hicache_events = self.hicache_events
        s.hub.request_intake = SimpleNamespace(
            requeue_ready_grammars=lambda: self.log("grammar")
        )
        s.hub.step_planner = SimpleNamespace(
            should_defer_prefill=lambda: False,
            get_new_batch_prefill=lambda *slot: self.schedule_leaf("prefill"),
            get_decode_step=lambda *slot: self.schedule_leaf("decode"),
            arm_prefill_decode_interval=lambda step: self.log("arm_interval"),
        )
        s.parts.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=self.mlp_sync,
            maybe_set_extend_view=lambda step: self.log("extend_view"),
        )
        s.parts.ngram_embedding_manager = SimpleNamespace(
            prepare_for_forward=self.ngram
        )
        s.mode.step_launcher = SimpleNamespace(
            run_batch=self.run_batch, launch_batch_sample_if_needed=self.sample
        )
        s._apply_war_barrier = lambda: self.log("war_barrier")
        s.process_batch_result = self.process_batch_result
        s.parts.step_counters = SimpleNamespace(record=lambda record: None)
        s.parts.batch_result_processor = SimpleNamespace(
            process_batch_result_prefill=self.commit_prefill
        )
        s.on_idle = lambda: self.log("idle")

    def log(self, *event):
        self.events.append((self.iteration,) + event)

    def script(self, kind):
        table = self.prefill if kind == "prefill" else self.decode
        return table.pop(self.iteration, None)

    def ingest_requests(self, stop_at_pause=False):
        self.iteration += 1
        if self.iteration == self.iterations:
            raise _Stop
        self.log("ingest", stop_at_pause) if stop_at_pause else self.log("ingest")
        if self.iteration == self.shutdown_at:
            self.s.parts.flags.gracefully_exit = True
        self.s.parts.flags._engine_paused = self.iteration in self.paused
        return []

    def hicache_events(self, should_retry_storage_prefetch=True):
        if should_retry_storage_prefetch:
            self.log("hicache")
        else:
            self.log("hicache", "no_retry")

    def schedule_leaf(self, kind):
        step = self.script(kind)
        self.log(kind, step.name if step is not None else None)
        return step

    def mlp_sync(self, step, need_sync=None):
        self.log("mlp_sync")
        return step

    def ngram(self, step, chunked_req):
        self.log("ngram")
        return step

    def run_batch(self, step, pp_proxy_tensors=None):
        self.log("run_batch", step.name)
        result = _Result(step.name, self.s.enable_overlap, self.ready_at.get(step.name))
        self.results[step.name] = result
        return result

    def sample(self, result, step):
        if result is None or result.delay_sample_func is None:
            return
        result.delay_sample_func = None
        self.log("sample", result.name)

    def process_batch_result(self, step, result):
        self.log("commit", step.name)

    def commit_prefill(self, step, result, on_checkpoint=None):
        self.log("commit_prefill", step.name)
        if on_checkpoint is not None:
            on_checkpoint(step.reqs[0])

    def run(self, **topology):
        with (
            published_topology(**topology),
            # The process groups the PP ring is built over.
            get_parallel().override(**self.groups),
            patch(
                "sglang.srt.managers.scheduler.set_schedule_time_batch",
                lambda step: self.log("schedule_time"),
            ),
        ):
            get_context().override(
                "test", disable_overlap_schedule=not self.s.enable_overlap
            )
            self.s.mode.stages = init_mode_stages(
                self.s.parts, self.s.loop_schedules(), lora_ids_in_use=set
            )
            try:
                _dispatch_event_loop_once(self.s)
            except _Stop:
                pass
        return self.events


def _decode_step(name):
    return _step(name, ForwardMode.DECODE)


def _at(iteration, *events):
    """The events of one iteration; a bare name is a one-field event."""
    return [(iteration,) + (e if isinstance(e, tuple) else (e,)) for e in events]


# The core steps every schedule runs first, in every mode and driver.
_PROLOGUE = ("chunked_abort", "hicache", "fpm", "stash")
# The tail every schedule runs, on a step and on no step.
_HOOK = "hook.on_schedule"
_FINALIZE_NONE = ("mlp_sync", "extend_view", "arm_interval", "ngram", _HOOK)
_FINALIZE = _FINALIZE_NONE[:-1] + ("schedule_time", _HOOK)


class TestFlatDriverStageOrder(CustomTestCase):
    def test_normal(self):
        shell = _Shell(
            "normal", 3, prefill={0: _step("p0")}, decode={1: _decode_step("d1")}
        )
        schedule = ("ingest", *_PROLOGUE, "grammar")
        self.assertEqual(
            shell.run(),
            _at(0, *schedule, ("prefill", "p0"), *_FINALIZE)
            + _at(0, ("run_batch", "p0"), ("commit", "p0"))
            + _at(1, *schedule, ("prefill", None), ("decode", "d1"), *_FINALIZE)
            + _at(1, ("run_batch", "d1"), ("commit", "d1"))
            + _at(2, *schedule, ("prefill", None), ("decode", None))
            + _at(2, *_FINALIZE_NONE, "idle"),
        )

    def test_overlap(self):
        shell = _Shell(
            "normal", 4, prefill={0: _step("p0")}, decode={1: _decode_step("d1")}
        )
        shell.s.enable_overlap = True
        schedule = {*_PROLOGUE, "grammar", "prefill", "decode"}
        schedule.update(_FINALIZE)
        self.assertEqual(
            [e for e in shell.run() if e[1] not in schedule],
            _at(0, "ingest", ("run_batch", "p0"), "war_barrier", ("sample", "p0"))
            + _at(1, "ingest", ("run_batch", "d1"), "war_barrier")
            + _at(1, ("commit", "p0"), ("sample", "d1"))
            + _at(2, "ingest", ("commit", "d1"))
            + _at(3, "ingest", "idle"),
        )
        self.assertTrue(shell.s.parts.flags.enable_skip_finishing_decode)

    def test_shutdown_stops_new_launches(self):
        """The pass that receives the shutdown commits what is pending and
        launches nothing; the loop then exits."""
        shell = _Shell(
            "normal",
            3,
            prefill={0: _step("p0")},
            decode={1: _decode_step("d1")},
            shutdown_at=1,
        )
        shell.s.enable_overlap = True
        self.assertEqual(
            [e for e in shell.run() if e[1] in ("ingest", "run_batch", "commit")],
            _at(0, "ingest", ("run_batch", "p0")) + _at(1, "ingest", ("commit", "p0")),
        )

    def test_deferred_prefill_requeues_ready_grammars(self):
        """A pass whose prefill the prefill/decode interval defers still
        queues the requests whose grammar is ready."""
        shell = _Shell("normal", 1)
        shell.s.hub.step_planner.should_defer_prefill = lambda: True
        events = [e[1] for e in shell.run()]
        self.assertIn("grammar", events)
        self.assertNotIn("prefill", events)

    def test_dllm(self):
        shell = _Shell("dllm", 2, prefill={0: _step("b0", ForwardMode.DLLM_EXTEND)})
        schedule = ("ingest", *_PROLOGUE, "grammar", "dllm.schedule")
        self.assertEqual(
            shell.run(),
            _at(0, *schedule, *_FINALIZE)
            + _at(0, ("run_batch", "b0"), ("dllm.on_launch", "b0"), ("commit", "b0"))
            + _at(1, *schedule, ("decode", None), *_FINALIZE_NONE, "idle"),
        )

    def test_pd_prefill(self):
        shell = _Shell("prefill", 2, prefill={0: _step("p0")})
        schedule = ("ingest", "dp.intake", *_PROLOGUE)
        schedule += ("dp.resolve", "dp.chunk", "grammar")
        self.assertEqual(
            shell.run(),
            _at(0, *schedule, ("prefill", "p0"), *_FINALIZE)
            + _at(0, ("run_batch", "p0"), ("dp.on_launch", "p0"))
            + _at(0, ("commit", "p0"), "dp.after_commit")
            + _at(1, *schedule, ("prefill", None), *_FINALIZE_NONE, "idle")
            + _at(1, "dp.after_commit"),
        )

    def test_pd_prefill_overlap(self):
        shell = _Shell("prefill", 3, prefill={0: _step("p0"), 1: _step("p1")})
        shell.s.enable_overlap = True
        schedule = ("ingest", "dp.intake", *_PROLOGUE)
        schedule += ("dp.resolve", "dp.chunk", "grammar")
        self.assertEqual(
            shell.run(),
            _at(0, *schedule, ("prefill", "p0"), *_FINALIZE)
            + _at(0, ("run_batch", "p0"), "war_barrier")
            + _at(0, ("dp.on_launch", "p0"), "dp.after_commit", ("sample", "p0"))
            + _at(1, *schedule, ("prefill", "p1"), *_FINALIZE)
            + _at(1, ("run_batch", "p1"), "war_barrier")
            + _at(1, ("dp.on_launch", "p1"), ("commit", "p0"), "dp.after_commit")
            + _at(1, ("sample", "p1"))
            + _at(2, *schedule, ("prefill", None), *_FINALIZE_NONE)
            + _at(2, ("commit", "p1"), "dp.after_commit"),
        )

    def test_pd_prefill_overlap_with_readiness_gate(self):
        """While the oldest pending step is not ready, intake repeats (holding
        a pause) and the commit waits; no step is built while two are pending."""
        shell = _Shell(
            "prefill",
            7,
            prefill={0: _step("p0"), 1: _step("p1"), 4: _step("p2")},
            ready_at={"p0": 3, "p1": 3, "p2": 5},
        )
        shell.s.enable_overlap = True
        schedule = ("ingest", "dp.intake", *_PROLOGUE)
        schedule += ("dp.resolve", "dp.chunk", "grammar")
        with envs.SGLANG_ENABLE_DISAGG_PREFILL_CONTINUOUS_INPUT_POLLING.override(True):
            events = shell.run()
        self.assertEqual(
            events,
            _at(0, *schedule, ("prefill", "p0"), *_FINALIZE)
            + _at(0, ("run_batch", "p0"), "war_barrier")
            + _at(0, ("dp.on_launch", "p0"), "dp.after_commit", ("sample", "p0"))
            + _at(1, *schedule, ("prefill", "p1"), *_FINALIZE)
            + _at(1, ("run_batch", "p1"), "war_barrier")
            + _at(1, ("dp.on_launch", "p1"), ("dp.ready", "p0", False))
            + _at(2, ("ingest", True), "dp.intake", ("dp.ready", "p0", False))
            + _at(3, ("ingest", True), "dp.intake", ("dp.ready", "p0", True))
            + _at(3, ("commit", "p0"), "dp.after_commit", ("sample", "p1"))
            + _at(4, *schedule, ("prefill", "p2"), *_FINALIZE)
            + _at(4, ("run_batch", "p2"), "war_barrier")
            + _at(4, ("dp.on_launch", "p2"), ("dp.ready", "p1", True))
            + _at(4, ("commit", "p1"), "dp.after_commit", ("sample", "p2"))
            + _at(5, *schedule, ("prefill", None), *_FINALIZE_NONE)
            + _at(5, ("dp.ready", "p2", True), ("commit", "p2"), "dp.after_commit")
            + _at(6, *schedule, ("prefill", None), *_FINALIZE_NONE, "idle")
            + _at(6, "dp.after_commit"),
        )

    def test_pd_decode(self):
        # The prefetch runs at the top of an iteration, before the ingest that
        # numbers it, so the log stamps it with the iteration before.
        schedule = ("ingest", *_PROLOGUE, "dd.intake", "grammar", ("join", None))
        for overlap in (False, True):
            with self.subTest(overlap=overlap):
                shell = _Shell("decode", 2, decode={0: _decode_step("d0")})
                shell.s.enable_overlap = overlap
                launch = (
                    [("run_batch", "d0"), ("commit", "d0")]
                    if not overlap
                    else [("run_batch", "d0"), "war_barrier", ("sample", "d0")]
                )
                self.assertEqual(
                    shell.run(),
                    _at(-1, "dd.prefetch")
                    + _at(0, *schedule, ("decode", "d0"), *_FINALIZE)
                    + _at(0, *launch, "dd.prefetch")
                    + _at(1, *schedule, ("decode", None), *_FINALIZE_NONE)
                    + _at(1, ("commit", "d0") if overlap else "idle", "dd.prefetch"),
                )
                self.assertFalse(shell.s.parts.flags.enable_skip_finishing_decode)


class _RingShell(_Shell):
    """The ring on PP rank 0 of 2: it relays requests, proxies and PD ids to
    rank 1; the outputs of a slot's step come back one ring round later."""

    def __init__(self, mode, iterations, **script):
        super().__init__(mode, iterations, **script)
        s = self.s
        self.groups = dict(
            pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=False),
            world_group=None,
            attn_tp_group=None,
        )
        s.parts.streams = Streams(
            device="cpu",
            device_module=SimpleNamespace(
                Event=lambda: SimpleNamespace(record=lambda stream: None),
                current_stream=lambda: None,
                stream=lambda stream: nullcontext(),
            ),
            schedule_stream=None,
            forward_stream=None,
            copy_stream=None,
            async_d2h_worker=None,
        )
        s.parts.kv = SimpleNamespace(token_to_kv_pool_allocator=None)
        s.parts.future_map = s.parts.workers.tp_worker = None
        s.parts.pp_spec_forward = None
        self.channel = SimpleNamespace(
            is_first_rank=True,
            is_last_rank=False,
            commit=lambda work, fence_next_forward=False: None,
            commit_proxy_send=lambda: None,
            send_pyobj=self.send,
            send_next=lambda data: self.send(data, async_send=True),
            recv_prev=self.recv,
            recv_proxy=lambda: None,
            send_proxy=lambda result: self.log("send_proxy"),
            wait_forward_dependencies=lambda: None,
            launch_event=None,
        )

    def send(self, data, async_send=False):
        self.log("send")
        return []

    def recv(self):
        self.log("recv")
        return []

    def exchange(self, next_first_rank_mb_id, next_mb_id):
        self.log("exchange", next_mb_id)
        done = SimpleNamespace(synchronize=lambda: None)
        return None, SimpleNamespace(relayed=next_mb_id), done

    def run(self):
        with (
            patch("sglang.srt.managers.scheduler_components.pp_ring.set_time_batch"),
            patch(
                "sglang.srt.managers.scheduler_assembly.PPChannel",
                return_value=self.channel,
            ),
            patch.object(
                PPRing,
                "_commit_send_output_work_and_preprocess_output_tensors",
                lambda ring, *args: self.exchange(*args),
            ),
        ):
            return super().run(pp_size=2)


class TestRingDriverStageOrder(CustomTestCase):
    def test_plain(self):
        shell = _RingShell(
            "normal", 4, prefill={0: _step("p0")}, decode={1: _decode_step("d1")}
        )
        schedule = {*_PROLOGUE, "grammar"}
        schedule.update(_FINALIZE)
        self.assertEqual(
            [e for e in shell.run() if e[1] not in schedule],
            _at(0, "ingest", "send", ("prefill", "p0"), ("run_batch", "p0"))
            + _at(0, ("exchange", 1), "send_proxy")
            + _at(1, "ingest", "send", ("prefill", None), ("decode", "d1"))
            + _at(1, ("run_batch", "d1"), ("exchange", 0), ("commit", "p0"))
            + _at(1, "send_proxy")
            + _at(2, "ingest", "send", ("prefill", None), ("decode", None))
            + _at(2, ("exchange", 1), ("commit", "d1"))
            + _at(3, "ingest", "send", ("prefill", None), ("decode", None))
            + _at(3, ("exchange", 0), "idle"),
        )

    def test_pd_prefill(self):
        shell = _RingShell("prefill", 3, prefill={0: _step("p0")})
        schedule = ("ingest", "dp.intake", "dp.poll_transfer", *_PROLOGUE)
        schedule += ("dp.chunk", "grammar")
        self.assertEqual(
            shell.run(),
            _at(0, *schedule, ("prefill", "p0"), *_FINALIZE)
            + _at(0, ("run_batch", "p0"))
            + _at(0, ("dp.on_launch", "p0"), ("exchange", 1), "send", "send")
            + _at(0, "send_proxy")
            + _at(1, *schedule, ("prefill", None), *_FINALIZE_NONE, ("exchange", 0))
            + _at(1, "recv", ("commit", "p0"), "dp.after_commit", "send", "send")
            + _at(2, *schedule, ("prefill", None), *_FINALIZE_NONE, ("exchange", 1))
            + _at(2, "send", "recv", "dp.after_commit", "send", "send"),
        )

    def test_pd_prefill_idles_while_a_transfer_is_in_flight(self):
        """An idle ring round runs on_idle, which checks the PD lists itself,
        as the flat loop does."""
        shell = _RingShell("prefill", 2)
        shell.s.parts.mode_owners.disagg_prefill.transfer.append(object())
        self.assertEqual(shell.run()[-1], (1, "idle"))

    def test_pause_launches_nothing(self):
        """A paused slot launches nothing but still exchanges and commits the
        step in flight, and a paused round runs no idle checks; the first
        pass after the resume schedules again."""
        for mode in ("normal", "prefill"):
            with self.subTest(mode):
                shell = _RingShell(
                    mode, 4, prefill={0: _step("p0"), 3: _step("p3")}, paused=(1, 2)
                )
                events = shell.run()
                paused = [e[0] for e in events if e[1] == "paused"]
                self.assertEqual(paused, [1, 2])
                launched = [e for e in events if e[1] == "run_batch"]
                self.assertEqual(
                    launched, [(0, "run_batch", "p0"), (3, "run_batch", "p3")]
                )
                self.assertIn((1, "commit", "p0"), events)
                self.assertNotIn("idle", [e[1] for e in events])

    def test_shutdown_ends_the_ring_once_nothing_is_in_flight(self):
        """The pass that receives the shutdown launches nothing; the ring
        ends at the first pass with nothing in flight."""
        for mode in ("normal", "prefill"):
            with self.subTest(mode):
                shell = _RingShell(mode, 6, prefill={0: _step("p0")}, shutdown_at=1)
                events = shell.run()
                self.assertIn((1, "commit", "p0"), events)
                self.assertNotIn((1, "run_batch"), [e[:2] for e in events])
                self.assertEqual(shell.iteration, 2)
                self.assertEqual(events[-2:], _at(2, "ingest", "send"))

    def test_pd_decode(self):
        shell = _RingShell("decode", 2, decode={0: _decode_step("d0")})
        schedule = ("ingest", "dd.retracted", "dd.poll_prealloc", "dd.poll_transfer")
        schedule += (*_PROLOGUE, "grammar")

        # Slot 1 applies the consensus of slot 0, the slot scheduled next,
        # and budgets it against that slot.
        def consensus(slot):
            return (
                "recv",
                ("dd.apply_retract", slot),
                "recv",
                ("dd.apply_prealloc", slot),
                "recv",
                ("dd.apply_transfer", slot),
            )

        self.assertEqual(
            shell.run(),
            _at(0, *schedule, ("join", 0), ("decode", "d0"), *_FINALIZE)
            + _at(0, ("run_batch", "d0"), ("exchange", 1))
            + _at(0, "send", "send", "send", "send", "send_proxy")
            + _at(1, *schedule, ("join", 1), ("decode", None), *_FINALIZE_NONE)
            + _at(1, ("exchange", 0), *consensus(0))
            + _at(1, ("commit", "d0"), "send", "send", "send", "send"),
        )


# Where each schedule event may sit: the core prologue, the mode's steps
# before admission, the grammar requeue, the join of ready requests,
# admission, decode selection, then the tail.
_RANK = {
    **{name: rank for rank, name in enumerate(_PROLOGUE)},
    "dd.intake": 4,
    "dp.resolve": 4,
    "dp.chunk": 4,
    "grammar": 5,
    "join": 6,
    "prefill": 7,
    "dllm.schedule": 7,
    "decode": 8,
    **{name: 9 for name in _FINALIZE + _FINALIZE_NONE},
}


class TestOnePrologue(CustomTestCase):
    """Every mode on every driver, with HiSparse on and off, runs one
    schedule: the core prologue in one order, the mode's own steps only at
    their places, and the tail."""

    _CASES = (
        ("flat", "normal", {}),
        ("flat", "normal", {"overlap": True}),
        ("flat", "dllm", {}),
        ("flat", "prefill", {}),
        ("flat", "prefill", {"overlap": True, "gate": True}),
        ("flat", "decode", {}),
        ("ring", "normal", {}),
        ("ring", "prefill", {}),
        ("ring", "decode", {}),
    )
    # The mode's own steps each schedule runs, per driver.
    _MODE_STEPS = {
        ("flat", "normal"): ("join?", "prefill", "decode"),
        ("flat", "dllm"): ("join?", "dllm.schedule", "decode"),
        ("flat", "prefill"): ("dp.resolve", "dp.chunk", "prefill"),
        ("flat", "decode"): ("dd.intake", "join", "decode"),
        ("ring", "normal"): ("join?", "prefill", "decode"),
        ("ring", "prefill"): ("dp.chunk", "prefill"),
        ("ring", "decode"): ("join", "decode"),
    }

    def _schedules(self, events):
        """The events of each schedule call, from its first core step to
        the tail's hook."""
        schedules, current = [], None
        for event in events:
            name = event[1]
            if name == _PROLOGUE[0]:
                current = []
            if current is not None and name in _RANK:
                current.append(name)
            if name == _HOOK and current is not None:
                schedules.append(current)
                current = None
        return schedules

    def test_every_mode_and_driver(self):
        for driver, mode, options in self._CASES:
            for hisparse in (False, True):
                with self.subTest(
                    driver=driver, mode=mode, hisparse=hisparse, **options
                ):
                    shell_cls = _RingShell if driver == "ring" else _Shell
                    shell = shell_cls(
                        mode,
                        3,
                        prefill={0: _step("p0", ForwardMode.DLLM_EXTEND)}
                        if mode == "dllm"
                        else {0: _step("p0")},
                        hisparse=hisparse,
                        ready_at={"p0": 1},
                    )
                    shell.s.enable_overlap = options.get("overlap", False)
                    with envs.SGLANG_ENABLE_DISAGG_PREFILL_CONTINUOUS_INPUT_POLLING.override(
                        options.get("gate", False)
                    ):
                        schedules = self._schedules(shell.run())
                    self.assertGreaterEqual(len(schedules), 2)
                    joins = hisparse and mode in ("normal", "dllm")
                    expected_steps = tuple(
                        step.rstrip("?")
                        for step in self._MODE_STEPS[(driver, mode)]
                        if not step.endswith("?") or joins
                    )
                    for names in schedules:
                        self.assertEqual(
                            [n for n in names if n in _PROLOGUE or n == "grammar"],
                            [*_PROLOGUE, "grammar"],
                        )
                        ranks = [_RANK[n] for n in names]
                        self.assertEqual(ranks, sorted(ranks), names)
                        steps = tuple(
                            n for n in names if 4 <= _RANK[n] <= 8 and n != "grammar"
                        )
                        # Admission builds the step; decode runs when it built none.
                        self.assertEqual(
                            [n for n in steps if n != "decode"],
                            [n for n in expected_steps if n != "decode"],
                        )


class TestCommitExtendStage(CustomTestCase):
    """Every mode's stages, flat or PP and with HiSparse on or off, commit an
    extend step through their own stage: dLLM and PD prefill through their
    owner, the rest through the prefill result processor."""

    def test_binding(self):
        expected = {
            "normal": "commit_prefill",
            "decode": "commit_prefill",
            "dllm": "dllm.commit_extend",
            "prefill": "dp.commit_extend",
        }
        for (mode, event), pp_size, hisparse in itertools.product(
            expected.items(), (1, 2), (False, True)
        ):
            with self.subTest(mode=mode, pp_size=pp_size, hisparse=hisparse):
                shell = _Shell(mode, 0, hisparse=hisparse)
                with published_topology(pp_size=pp_size):
                    stages = init_mode_stages(
                        shell.s.parts, shell.s.loop_schedules(), lora_ids_in_use=set
                    )
                stages.commit_extend(_step("p0"), None)
                staged = hisparse and event == "commit_prefill"
                self.assertEqual(
                    shell.events,
                    [(-1, event, "p0")] + [(-1, "hisparse.stage", "p0")] * staged,
                )
                # HiSparse frees a decode step's remapped slots in every mode.
                self.assertEqual(stages.after_decode_commit is not None, hisparse)


class _StageRecorder:
    """Stands for the scheduler's stage recorder: the stage it is in."""

    def __init__(self):
        self.stage = None
        self._trace_stage = None

    def enter(self, stage):
        previous, self.stage = self.stage, stage
        return previous

    def exit(self, previous):
        self.stage = previous


class TestScheduleStageLabel(CustomTestCase):
    """Every mode's schedule, flat and ring, runs under the
    get_next_batch_to_run stage; the PD prefill intake runs outside it."""

    _SCHEDULE_EVENTS = (*_PROLOGUE, "grammar", "join", "prefill", "decode", _HOOK)

    def _stages_by_event(self, shell):
        recorder = shell.s.scheduler_stage_metrics = _StageRecorder()
        stages = {}
        log = shell.log

        def log_with_stage(*event):
            stages.setdefault(event[0], set()).add(recorder.stage)
            log(*event)

        shell.log = log_with_stage
        shell.run()
        return stages

    def test_flat_and_ring(self):
        shells = {
            "flat normal": _Shell("normal", 2, decode={1: _decode_step("d1")}),
            "flat dllm": _Shell("dllm", 1),
            "flat pd prefill": _Shell("prefill", 1),
            "flat pd decode": _Shell("decode", 1),
            "ring normal": _RingShell("normal", 2),
            "ring pd prefill": _RingShell("prefill", 2),
            "ring pd decode": _RingShell("decode", 2),
        }
        for name, shell in shells.items():
            with self.subTest(name):
                stages = self._stages_by_event(shell)
                for event in self._SCHEDULE_EVENTS:
                    if event in stages:
                        self.assertEqual(
                            stages[event], {"get_next_batch_to_run"}, event
                        )
                self.assertLessEqual(stages.get("dp.intake", set()), {None})


if __name__ == "__main__":
    unittest.main()
