"""PDMux standard-prefill lane: submission, event order, completion pipeline.

The standard lane submits a prefill as one ordinary EXTEND on the prefill
green-context stream and finalizes it later, once every rank has seen the
result's `copy_done`. Three properties carry that design and none of them is
visible from the layer_split tests:

- Submission must not block. The synchronous branch it replaced read
  `new_seq_lens` back to the host right after the forward, which pins the
  scheduler thread to the prefill and serializes decode behind it.
- The completion decision must be rank-uniform. A rank-local `copy_done.query()`
  would let ranks finalize on different iterations, after which their decode
  batches diverge -- under TP that hangs on the next collective rather than
  returning a wrong answer.
- The prefill stream must publish no event the decode lane waits on between
  submitting the forward and its copy_done, or every decode step queues behind
  the whole prefill.

The loop is driven for real over stubbed streams, events and collaborators, the
way test_pdmux_hicache_events.py drives the layer_split loop. The stream groups
are three distinct (prefill, decode) pairs and the group index is real state,
so a submit that lands on the wrong lane after a group switch is observable.
"""

from __future__ import annotations

import ast
import contextlib
import pathlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import sglang.srt.managers.scheduler as scheduler_module
import sglang.srt.multiplex.multiplexing_mixin as multiplexing_mixin
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_forward_context,
)
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.multiplex.multiplexing_mixin import (
    PdmuxPrefillInflight,
    SchedulerMultiplexMixin,
)
from sglang.srt.multiplex.pdmux_context import (
    decode_lane_attn_backend,
    is_pdmux_standard_prefill,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TP_SIZE = 2
NUM_GROUPS = 3

# The loop's stream-group index, as the real pdmux_context globals would hold
# it. A list so the patched getter/setter share one mutable cell.
_STREAM_IDX = [0]


class _LoopFinished(Exception):
    """Breaks out of the event loop, which otherwise never returns."""


class _ScriptedEvent:
    """A CUDA event whose readiness follows a script, then stays ready."""

    def __init__(self, name, script=None):
        self.name = name
        self._script = list(script or [])

    def query(self):
        return self._script.pop(0) if self._script else True


class _Stream:
    """Records the cross-lane event traffic the loop publishes.

    `lane` is "P" or "D"; `name` also carries the group index so two groups'
    streams are distinguishable in the journal.
    """

    def __init__(self, lane, group, journal):
        self.lane = lane
        self.name = f"{lane}{group}"
        self.journal = journal

    def record_event(self):
        event = _ScriptedEvent(f"{self.name}:{len(self.journal)}")
        self.journal.append(("record", self, event))
        return event

    def wait_event(self, event):
        self.journal.append(("wait", self, event))

    def wait_stream(self, other):
        self.journal.append(("wait_stream", self, other))

    def synchronize(self):
        self.journal.append(("sync", self, None))


class _Work:
    """Stand-in for the gloo allreduce Work handle.

    `wait` folds the other ranks' votes into this rank's flags, which is what
    makes a locally-ready-but-globally-not-ready iteration expressible.
    """

    def __init__(self, flags, peer_votes):
        self._flags = flags
        self._peer_votes = peer_votes

    def wait(self):
        self._flags[0] = int(self._flags[0]) + sum(self._peer_votes)


class _Batch:
    """Minimal ScheduleBatch stand-in."""

    def __init__(self, name, empty=False):
        self.name = name
        self._empty = empty
        self.batch_is_full = False
        self.chunked_req = None
        self.reqs = []
        self.split_prefill_finished = False

    def is_empty(self):
        return self._empty

    def batch_size(self):
        return 0 if self._empty else 1

    def filter_batch(self, chunked_req_to_exclude=None):
        pass

    def merge_batch(self, other):
        pass


class _FakeScheduler(SchedulerMultiplexMixin):
    """Drives the real standard loop over stubbed collaborators."""

    HICACHE_PUMP_INTERVAL = 1

    def __init__(
        self,
        *,
        max_iterations,
        copy_done_script=None,
        peer_votes_script=None,
        prefill_batches=None,
        decode_empty=False,
        fail_prefill_finalize=False,
    ):
        self.max_iterations = max_iterations
        self.iteration = -1
        self.journal = []
        self.submitted = []
        self.pumps = []
        self.processed_prefills = []
        self.issued_votes = []
        self.decode_backend_switches = []
        self._fail_prefill_finalize = fail_prefill_finalize

        # Per-query readiness for this rank's copy_done, and the peer ranks'
        # votes per allreduce (default: the peers agree with this rank).
        self._copy_done_script = list(copy_done_script or [])
        self._peer_votes_script = list(peer_votes_script or [])

        self._prefill_batches = list(
            prefill_batches if prefill_batches is not None else [_Batch("prefill")]
        )

        self.pdmux_standard = True
        self.draft_worker = None
        self.chunked_req = None
        self.split_prefill_batch = None
        # Set by Scheduler.__init__ / init_pdmux in the real scheduler.
        self._pdmux_prefill_batch = None
        self._pdmux_prefill_pending = None
        self._pdmux_prefill_inflight = None
        self.running_batch = _Batch("decode", empty=decode_empty)
        self.tree_cache = Mock()

        self.ps = SimpleNamespace(tp_size=TP_SIZE, gpu_id=0)
        # No manual divisions: with three groups, a non-empty decode batch that
        # runs alongside a prefill selects the single shared group, index 1.
        self.pdmux_config = SimpleNamespace(
            manual_divisions=[], decode_bs_divisor=36, split_forward_token_budget=1000
        )
        self.real_sm_group_num = NUM_GROUPS
        self.sm_counts = [(1, 0), (1, 1), (0, 1)]
        self.stream_groups = [
            (_Stream("P", i, self.journal), _Stream("D", i, self.journal))
            for i in range(NUM_GROUPS)
        ]
        self.pdmux_prefill_stream = None
        self.tp_cpu_group = SimpleNamespace(allreduce=self._allreduce)
        self.request_receiver = SimpleNamespace(recv_requests=self._recv_requests)
        self.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=lambda batch: batch
        )

    # --- collaborators the loop drives -------------------------------------

    def _recv_requests(self):
        self.iteration += 1
        if self.iteration >= self.max_iterations:
            raise _LoopFinished
        return []

    def _allreduce(self, flags, op):
        local_vote = int(flags[0])
        peers = (
            self._peer_votes_script.pop(0)
            if self._peer_votes_script
            else [local_vote] * (TP_SIZE - 1)
        )
        self.issued_votes.append((self.iteration, local_vote))
        return _Work(flags, peers)

    def process_input_requests(self, recv_reqs):
        pass

    def process_pending_chunked_abort(self):
        pass

    def get_new_batch_prefill(self, running_batch):
        batch = self._prefill_batches.pop(0) if self._prefill_batches else None
        return SimpleNamespace(batch_to_run=batch, running_batch=running_batch)

    def update_running_batch(self, running_batch):
        return running_batch

    def on_idle(self):
        pass

    def check_hicache_events_if_enabled(self):
        self.pumps.append(self.iteration)
        return False

    def run_batch(self, batch):
        is_prefill = self._pdmux_prefill_batch is batch
        if is_prefill:
            # What the real submit path would issue on: the lane the loop
            # published, read at submit time.
            self.submitted.append((self.iteration, batch, self.pdmux_prefill_stream))
            self.journal.append(("submit", self.pdmux_prefill_stream, batch))
        # The result remembers what it was submitted as. The batch object cannot
        # tell: after a finalize into an empty decode batch, the prefill batch
        # *is* the running batch and runs decode steps under the same name.
        return SimpleNamespace(
            is_prefill=is_prefill,
            extra_keep_alive_refs=[batch, "attrs"],
            copy_done=_ScriptedEvent(
                "copy_done", self._copy_done_script if is_prefill else None
            ),
        )

    def process_batch_result(self, batch, result):
        if result.is_prefill:
            self.processed_prefills.append((self.iteration, batch))
            self.journal.append(("finalize", None, batch))
            if self._fail_prefill_finalize:
                raise RuntimeError("finalize failed")

    def stash_chunked_request(self, req):
        pass

    def _update_decode_attn_backends(self, stream_idx):
        self.decode_backend_switches.append(stream_idx)
        self.journal.append(("switch", None, stream_idx))


@contextlib.contextmanager
def _stubbed_cuda():
    _STREAM_IDX[0] = 0
    with (
        patch("torch.cuda.stream", lambda _stream: contextlib.nullcontext()),
        patch("torch.cuda.empty_cache", lambda: None),
        patch(
            "sglang.srt.multiplex.multiplexing_mixin.pdmux_prefill_tp_group",
            contextlib.nullcontext,
        ),
        patch(
            "sglang.srt.multiplex.multiplexing_mixin.get_current_stream_idx",
            lambda: _STREAM_IDX[0],
        ),
        patch(
            "sglang.srt.multiplex.multiplexing_mixin.set_current_stream_idx",
            lambda idx: _STREAM_IDX.__setitem__(0, idx),
        ),
        patch(
            "sglang.srt.multiplex.multiplexing_mixin.dist.get_world_size",
            lambda group: TP_SIZE,
        ),
    ):
        yield


def _run_loop(**kwargs):
    scheduler = _FakeScheduler(**kwargs)
    with _stubbed_cuda():
        try:
            scheduler.event_loop_pdmux()
        except _LoopFinished:
            pass
    return scheduler


def _events(journal, kind, lane):
    return [
        event
        for k, stream, event in journal
        if k == kind and stream is not None and stream.lane == lane
    ]


class TestPDMuxStandardPrefillLoop(unittest.TestCase):
    def test_marker_is_cleared_once_the_submit_returns(self):
        """The identity marker must not outlive the run_batch call.

        The prefill batch object is merged into the decode batch at finalize, so
        a marker left set would send the next decode step down run_batch's
        prefill branch -- submitting a decode as if it were a prefill.
        """
        scheduler = _run_loop(max_iterations=4)

        self.assertEqual(len(scheduler.submitted), 1)
        self.assertIsNone(scheduler._pdmux_prefill_batch)

    def test_pending_prefill_is_submitted_on_the_lane_selected_after_the_switch(self):
        """The group switch runs between forming a prefill and submitting it.

        With a decode batch running, formation on group 0 is followed by a
        switch to the shared group 1, and the submit must issue on group 1's
        prefill stream. A submit that captured the stream at formation, or a
        loop that never republished the lane, lands on group 0's stream -- the
        prefill-only partition -- while decode runs on group 1.
        """
        scheduler = _run_loop(max_iterations=3)

        self.assertEqual(scheduler.decode_backend_switches[:1], [1])
        _, _, submit_stream = scheduler.submitted[0]
        self.assertIs(submit_stream, scheduler.stream_groups[1][0])
        self.assertIsNot(submit_stream, scheduler.stream_groups[0][0])

    def test_no_switch_while_a_prefill_is_in_flight(self):
        """Switching rebinds decode backends and drains the prefill stream.

        Doing that under an in-flight prefill would either block the scheduler
        on the whole forward or swap the backend a running forward reads. The
        switch to the shared group happens before the submit; nothing switches
        again, and the prefill stream is never drained, until the work item is
        finalized. The decode lane's own per-step `synchronize()` is expected
        in that window -- it is the synchronous decode path, not a switch.
        """
        scheduler = _run_loop(max_iterations=6, copy_done_script=[False] * 3)

        journal = scheduler.journal
        submit_at = next(i for i, e in enumerate(journal) if e[0] == "submit")
        finalize_at = next(i for i, e in enumerate(journal) if e[0] == "finalize")
        window = journal[submit_at + 1 : finalize_at]

        self.assertEqual([e for e in window if e[0] == "switch"], [])
        self.assertEqual(
            [e for e in window if e[0] == "sync" and e[1].lane == "P"], []
        )
        # The decode lane kept stepping (and draining itself) meanwhile.
        self.assertTrue(any(e[0] == "sync" and e[1].lane == "D" for e in window))
        self.assertEqual(scheduler.decode_backend_switches[:1], [1])

    def test_one_completion_vote_per_iteration_while_in_flight(self):
        """A vote is issued every iteration and consumed the next one.

        Issuing only on alternate iterations (consume, else sample) would double
        the detection latency. With copy_done not ready at submit and ready from
        the next sample on, the votes land on consecutive iterations.
        """
        scheduler = _run_loop(max_iterations=5, copy_done_script=[False])

        self.assertEqual([it for it, _ in scheduler.issued_votes], [0, 1])
        self.assertEqual([vote for _, vote in scheduler.issued_votes], [0, 1])
        self.assertEqual(len(scheduler.processed_prefills), 1)

    def test_a_consumed_not_ready_vote_is_resampled_in_the_same_iteration(self):
        """Consuming a not-ready vote must not skip that iteration's sample."""
        scheduler = _run_loop(
            max_iterations=4,
            copy_done_script=[False],
            peer_votes_script=[[0], [1], [1]],
        )

        self.assertEqual([it for it, _ in scheduler.issued_votes], [0, 1])

    def test_a_peer_that_is_not_ready_blocks_the_finalize(self):
        """A locally ready result is not enough; the reduction must carry TP.

        This is exactly the case a rank-local `copy_done.query()` decision would
        get wrong: this rank sees the result ready while a peer does not.
        """
        scheduler = _run_loop(
            max_iterations=4,
            peer_votes_script=[[0], [0], [0], [0]],
        )

        self.assertEqual(scheduler.processed_prefills, [])
        self.assertIsNotNone(scheduler._pdmux_prefill_inflight)
        # A work item with an outstanding vote is still reachable by an abort.
        self.assertTrue(scheduler._extra_inflight_batches())

    def test_completion_progresses_with_no_decode_batch(self):
        """The completion check is not gated on decode work existing.

        Otherwise a prefill submitted into an idle server would never be
        detected complete and the lane would wedge.
        """
        scheduler = _run_loop(max_iterations=4, decode_empty=True)

        self.assertTrue(scheduler.issued_votes)
        self.assertEqual(len(scheduler.processed_prefills), 1)

    def test_a_second_work_item_starts_without_the_first_ones_vote(self):
        """A new work item must not inherit the previous item's completion vote.

        Carrying the Work or its flags across items would finalize the second on
        evidence gathered for the first.
        """
        scheduler = _run_loop(
            max_iterations=8,
            prefill_batches=[_Batch("prefill-a"), _Batch("prefill-b")],
        )

        self.assertEqual(len(scheduler.submitted), 2)
        finalized = [batch.name for _, batch in scheduler.processed_prefills]
        self.assertEqual(finalized, ["prefill-a", "prefill-b"])

    def test_empty_formation_still_publishes_its_dependency(self):
        """Formation publishes E1 even when it produced no batch.

        Formation is not free when it admits nothing: it runs the HiCache pump,
        and admission's host-to-device load-back allocates device pages and can
        evict to make room. Recording only on a non-empty batch leaves those
        writes unordered against the decode lane's allocation.
        """
        scheduler = _run_loop(max_iterations=2, prefill_batches=[])

        self.assertEqual(scheduler.submitted, [])
        prefill_records = _events(scheduler.journal, "record", "P")
        waited_by_decode = set(_events(scheduler.journal, "wait", "D"))
        self.assertTrue(prefill_records)
        self.assertTrue(any(event in waited_by_decode for event in prefill_records))

    def test_formation_waits_on_the_decode_lane(self):
        """Formation is ordered after the decode lane's device work.

        Input handling runs prefix matching (a device concatenation) and the
        previous iteration's decode result processing frees pages; formation
        allocates from that same free list.
        """
        scheduler = _run_loop(max_iterations=3)

        waited_by_prefill = _events(scheduler.journal, "wait", "P")
        self.assertTrue(any(event.name.startswith("D") for event in waited_by_prefill))

    def test_no_event_is_recorded_on_the_prefill_stream_while_in_flight(self):
        """The scheduling invariant, stated as a test.

        Between the submit and the finalize the prefill stream records nothing:
        an event recorded there lands after the whole prefill forward, so a
        decode step waiting on it would block for the entire prefill. It does
        not say decode never waits on prefill -- formation and the merge both
        publish dependencies, and both sit outside this window.
        """
        scheduler = _run_loop(max_iterations=6, copy_done_script=[False, False])

        journal = scheduler.journal
        submit_at = next(i for i, e in enumerate(journal) if e[0] == "submit")
        finalize_at = next(
            i for i, e in enumerate(journal) if e[0] == "finalize" and i > submit_at
        )
        recorded_between = [
            entry
            for entry in journal[submit_at + 1 : finalize_at]
            if entry[0] == "record" and entry[1].lane == "P"
        ]
        self.assertEqual(recorded_between, [])

    def test_finalize_failure_keeps_the_work_item_and_propagates(self):
        """A failed finalize must not clear the in-flight state.

        Clearing it would let a later iteration finalize the same work item
        again. Keeping it is a guard, not a retry: the exception propagates and
        nothing re-enters finalize for this batch.
        """
        scheduler = _FakeScheduler(max_iterations=5, fail_prefill_finalize=True)
        with _stubbed_cuda():
            with self.assertRaises(RuntimeError):
                scheduler.event_loop_pdmux()

        self.assertIsNotNone(scheduler._pdmux_prefill_inflight)
        self.assertEqual(len(scheduler.processed_prefills), 1)

    def test_pending_and_inflight_prefills_are_reported_as_inflight(self):
        """`abort_request` and `is_fully_idle` both consume this list.

        A prefill between formation and finalize is in neither running_batch nor
        last_batch, so without it an abort would be ignored until the request
        reached the decode batch, and flush_cache could reset the pools under a
        running forward.
        """
        scheduler = _FakeScheduler(max_iterations=1)
        batch = _Batch("prefill")

        self.assertEqual(scheduler._extra_inflight_batches(), [])
        scheduler._pdmux_prefill_pending = batch
        self.assertEqual(scheduler._extra_inflight_batches(), [batch])
        scheduler._pdmux_prefill_pending = None
        scheduler._pdmux_prefill_inflight = PdmuxPrefillInflight(
            batch=batch, result=object()
        )
        self.assertEqual(scheduler._extra_inflight_batches(), [batch])

    def test_layer_split_mode_reports_no_extra_inflight_batches(self):
        """layer_split keeps its existing abort / idle behaviour untouched."""
        scheduler = _FakeScheduler(max_iterations=1)
        scheduler.pdmux_standard = False
        scheduler._pdmux_prefill_pending = _Batch("prefill")

        self.assertEqual(scheduler._extra_inflight_batches(), [])


class _RunnerStub(EagerRunner):
    """An EagerRunner with only the fields the backend resolvers read."""

    def __init__(self, *, enable_pdmux, pdmux_standard, model_runner):
        self.enable_pdmux = enable_pdmux
        self.pdmux_standard = pdmux_standard
        self.model_runner = model_runner


def _target_verify_batch():
    return SimpleNamespace(forward_mode=SimpleNamespace(is_target_verify=lambda: True))


def _extend_batch():
    return SimpleNamespace(forward_mode=SimpleNamespace(is_target_verify=lambda: False))


class TestEagerRunnerBackendResolution(unittest.TestCase):
    """Which attention backend an eager decode-lane forward runs under.

    The multi-step EAGLE draft binds `draft_attn_backend.attn_backends[i]` per
    step and marks the forward's metadata ready against it. With draft graphs
    off (which the standard lane forces), every step is eager, and the runner's
    pdmux override replaced that per-step backend with the per-stream decode
    backend: metadata init was skipped because it was marked ready, so the
    model read a backend that had never planned this forward.
    """

    def setUp(self):
        self.default = object()
        self.group = object()
        self.per_step = object()
        self.model_runner = SimpleNamespace(
            attn_backend=self.default, decode_attn_backend=self.group
        )

    def _resolve_decode_under(self, ctx_backend, *, pdmux_standard, enable_pdmux=True):
        runner = _RunnerStub(
            enable_pdmux=enable_pdmux,
            pdmux_standard=pdmux_standard,
            model_runner=self.model_runner,
        )
        # `_forward_raw` publishes the runner default when nobody else did; a
        # caller such as the draft loop publishes its own before calling in.
        with forward_context(ForwardContext(attn_backend=ctx_backend)):
            backend, pdmux_ctx = runner._resolve_decode_pdmux()
            with pdmux_ctx:
                active = get_forward_context().attn_backend
        return backend, active

    def test_standard_lane_keeps_a_caller_published_backend(self):
        """The multi-step draft regression: a per-step backend must survive."""
        backend, active = self._resolve_decode_under(
            self.per_step, pdmux_standard=True
        )

        self.assertIs(backend, self.per_step)
        self.assertIs(active, self.per_step)

    def test_standard_lane_overrides_only_the_runner_default(self):
        """With no caller choice, decode still routes to the per-stream backend.

        This is the routing DSpark's draft block and the target verify rely on;
        the exception for caller-published backends must not disable it.
        """
        backend, active = self._resolve_decode_under(
            self.default, pdmux_standard=True
        )

        self.assertIs(backend, self.group)
        self.assertIs(active, self.group)

    def test_layer_split_keeps_its_original_override(self):
        """layer_split behaviour is unchanged, caller context or not."""
        backend, active = self._resolve_decode_under(
            self.per_step, pdmux_standard=False
        )

        self.assertIs(backend, self.group)
        self.assertIs(active, self.group)

    def test_target_verify_uses_the_decode_backend_only_on_the_standard_lane(self):
        """TARGET_VERIFY is extend-classified but decode-lane work.

        Standard routes it to the per-stream decode backend; layer_split and
        non-PDMux keep the runner default they always used.
        """
        cases = [
            (True, True, self.group),
            (True, False, self.default),
            (False, False, self.default),
        ]
        for enable_pdmux, pdmux_standard, expected in cases:
            with self.subTest(enable_pdmux=enable_pdmux, pdmux_standard=pdmux_standard):
                runner = _RunnerStub(
                    enable_pdmux=enable_pdmux,
                    pdmux_standard=pdmux_standard,
                    model_runner=self.model_runner,
                )
                with forward_context(ForwardContext(attn_backend=self.default)):
                    backend, _ = runner._resolve_extend_pdmux(_target_verify_batch())
                self.assertIs(backend, expected)

    def test_a_real_prefill_never_takes_the_decode_backend(self):
        runner = _RunnerStub(
            enable_pdmux=True, pdmux_standard=True, model_runner=self.model_runner
        )
        with forward_context(ForwardContext(attn_backend=self.default)):
            backend, _ = runner._resolve_extend_pdmux(_extend_batch())

        self.assertIs(backend, self.default)


class TestPdmuxStandardPrefillMode(unittest.TestCase):
    """`is_pdmux_standard_prefill` is the single switch every lane gate reads."""

    @staticmethod
    @contextlib.contextmanager
    def _disagg(*, enable_pdmux, mode):
        with patch(
            "sglang.srt.runtime_context.get_disagg",
            lambda: SimpleNamespace(
                enable_pdmux=enable_pdmux, pdmux_prefill_mode=mode
            ),
        ):
            yield

    def test_only_pdmux_plus_standard_selects_the_lane(self):
        """Both terms are load-bearing.

        Dropping the mode check would put every PDMux run on the lane's gates
        (single-stream models, no draft graphs, no symmetric-memory gather),
        silently changing layer_split. Dropping the pdmux check would apply them
        to every non-PDMux server.
        """
        cases = {
            (True, "standard"): True,
            (True, "layer_split"): False,
            (False, "standard"): False,
            (False, "layer_split"): False,
        }
        for (enable_pdmux, mode), expected in cases.items():
            with self.subTest(enable_pdmux=enable_pdmux, mode=mode):
                with self._disagg(enable_pdmux=enable_pdmux, mode=mode):
                    self.assertIs(is_pdmux_standard_prefill(), expected)

    def test_decode_lane_backend_follows_the_mode(self):
        """The pre-forward planners (DSpark verify, EAGLE's trivial verify mask)
        must plan into the same backend the eager runner will resolve."""
        default, group = object(), object()
        model_runner = SimpleNamespace(attn_backend=default, decode_attn_backend=group)

        with self._disagg(enable_pdmux=True, mode="standard"):
            self.assertIs(decode_lane_attn_backend(model_runner), group)
        with self._disagg(enable_pdmux=True, mode="layer_split"):
            self.assertIs(decode_lane_attn_backend(model_runner), default)
        with self._disagg(enable_pdmux=False, mode="layer_split"):
            self.assertIs(decode_lane_attn_backend(model_runner), default)


class TestPdmuxStandardPrefillAdmission(unittest.TestCase):
    """The standard lane's server-arg admission.

    Each rejection names a resource the lane cannot place on the prefill green
    context. They are asserts rather than silent downgrades because every one of
    them would otherwise fail as a hang or a wrong SM assignment at runtime,
    long after launch.
    """

    @staticmethod
    def _args(**overrides):
        from sglang.srt.model_executor.cuda_graph_config import Backend

        base = dict(
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend=Backend.DISABLED)
            ),
            enable_multi_layer_eagle=False,
            enable_two_batch_overlap=False,
            enable_unified_memory=False,
            enable_dp_attention=False,
            ep_size=1,
            attn_cp_size=1,
            dcp_size=1,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def _check(self, **overrides):
        from sglang.srt.arg_groups.validation_hook import (
            _check_pdmux_standard_prefill,
        )

        _check_pdmux_standard_prefill(self._args(**overrides))

    def test_a_clean_profile_is_accepted(self):
        self._check()

    def test_every_incompatible_feature_is_rejected(self):
        """Completeness: each gate must actually fire.

        The failure mode is a gate that silently degrades to always-true after a
        field is renamed or its default changes.
        """
        rejections = [
            dict(
                cuda_graph_config=SimpleNamespace(
                    prefill=SimpleNamespace(backend="breakable")
                )
            ),
            dict(enable_multi_layer_eagle=True),
            dict(enable_two_batch_overlap=True),
            dict(enable_unified_memory=True),
            dict(ep_size=2),
            dict(attn_cp_size=2),
            dict(dcp_size=2),
        ]
        for overrides in rejections:
            with self.subTest(**{k: str(v) for k, v in overrides.items()}):
                with self.assertRaises(AssertionError):
                    self._check(**overrides)

    def test_dp_attention_is_supported(self):
        self._check(enable_dp_attention=True)


def _function_node(module, name: str) -> ast.FunctionDef:
    source = pathlib.Path(module.__file__).read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return node
    raise AssertionError(f"{name} not found in {module.__name__}")


def _called_attribute_names(node: ast.AST) -> list[str]:
    """Names of every `x.<name>(...)` call in the node, docstrings excluded."""
    return [
        n.func.attr
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]


def _to_cpu_calls(node: ast.AST) -> list[ast.Call]:
    """`x.to("cpu")` / `x.to(device="cpu")` calls in the node."""
    calls = []
    for n in ast.walk(node):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)):
            continue
        if n.func.attr != "to":
            continue
        targets = [a for a in n.args if isinstance(a, ast.Constant)] + [
            k.value for k in n.keywords if isinstance(k.value, ast.Constant)
        ]
        if any(c.value == "cpu" for c in targets):
            calls.append(n)
    return calls


class TestPDMuxStandardPrefillWiring(unittest.TestCase):
    """Invariants that live in code shape rather than in the loop's behaviour.

    These inspect the AST, never the raw text, so a docstring that happens to
    mention `synchronize` or `query()` cannot fail or pass them.
    """

    def test_overlap_branch_keeps_every_side_path(self):
        """Extracting the shared primitives must not drop overlap behaviour.

        The overlap branch owns four paths the PDMux lane deliberately does not
        use -- the grammar barrier, the DSpark confidence budget, deferred
        sampling and the unified-memory forward_done hook. A refactor that
        folded the whole branch into one shared helper would silently lose them.
        """
        run_batch = _function_node(scheduler_module, "run_batch")
        names = {
            n.attr for n in ast.walk(run_batch) if isinstance(n, ast.Attribute)
        } | {n.id for n in ast.walk(run_batch) if isinstance(n, ast.Name)}
        constants = {
            n.value
            for n in ast.walk(run_batch)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }

        for marker in (
            "_confidence_budget_prepare",
            "delay_sample_func",
            "set_inflight_forward",
        ):
            self.assertIn(marker, names, f"overlap branch lost {marker}")
        self.assertIn(
            "grammar_barrier", constants, "overlap branch lost grammar_barrier"
        )

    def test_overlap_isolation_still_pins_the_two_iteration_ring(self):
        """`_forward_isolation(overlap=True)` keeps the overlap keep-alive.

        The PDMux lane calls the field-isolation primitive directly because it
        holds its refs for the life of one work item instead. If the pinning
        moved out of `_forward_isolation`, the overlap loop would lose it.
        """
        isolation = _function_node(scheduler_module, "_forward_isolation")

        self.assertIn("record_batch_in_overlap", _called_attribute_names(isolation))

    def test_standard_submit_contains_no_host_readback(self):
        """The submit path must not call anything that reads a device tensor.

        The synchronous branch it replaced ended with `new_seq_lens.to("cpu")`
        and a `seq_lens_cpu.sum()`, which blocks the scheduler thread until the
        prefill forward has finished -- what this lane exists to avoid.
        """
        submit = _function_node(scheduler_module, "_run_pdmux_standard_prefill")
        called = set(_called_attribute_names(submit))

        for forbidden in ("tolist", "item", "cpu", "synchronize"):
            self.assertNotIn(forbidden, called, f"submit path calls .{forbidden}()")
        self.assertEqual(_to_cpu_calls(submit), [])

    def test_completion_sample_is_reduced_before_it_gates_finalize(self):
        """The local copy_done sample may only feed the reduction.

        Deciding locally -- via `is_completed()` or a bare `query()` branch --
        would let ranks finalize on different iterations.
        """
        advance = _function_node(multiplexing_mixin, "_advance_standard_prefill")
        called = _called_attribute_names(advance)

        self.assertNotIn("is_completed", called)
        self.assertIn("allreduce", called)
        self.assertEqual(called.count("query"), 1)

        # The single query() feeds the vote tensor; the finalize call sits in a
        # branch on the reduced value, never on the query itself.
        query_if = next(
            n
            for n in ast.walk(advance)
            if isinstance(n, ast.If)
            and any(
                isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute)
                and c.func.attr == "query"
                for c in ast.walk(n.test)
            )
        )
        self.assertNotIn(
            "_finalize_standard_prefill", _called_attribute_names(query_if)
        )


if __name__ == "__main__":
    unittest.main()
