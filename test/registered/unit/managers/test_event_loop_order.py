"""Each CUDA event loop makes the same scheduler calls in the same order.

Drives every flat and pipeline loop a few iterations on a fake scheduler that
logs each call it receives (with its arguments, rendered when the call is made),
each write to a loop-owned attribute, and each change to the in-flight queue.
The expected logs were recorded on the hand-written loops; a change to a loop's
call order fails here.
"""

import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, NonCallableMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.inflight_queue import (  # noqa: E402
    InflightQueue,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.runtime_context import get_disagg, get_parallel  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# Attribute writes the loops make between calls.
_WATCHED = frozenset(
    {
        "inflight_steps",
        "enable_skip_finishing_decode",
        "rows_join_before_plan",
        "enable_continuous_input_polling",
        "cur_batch_for_debug",
        "_sched_idled",
        "last_batch",
        "decoding_batch",
        "launch_event",
        "send_req_work",
        "pp_outputs",
    }
)
# Queue changes; its queries have no effect.
_QUEUE_CHANGES = ("push", "pop_due", "drain", "end_step")


_PREFETCH = "disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries("


class _Stop(Exception):
    pass


class _B:
    """A labelled stand-in for a batch or any other value the log names."""

    def __init__(self, label, mode=ForwardMode.EXTEND):
        self.label = label
        self.forward_mode = mode
        self.is_extend_in_batch = mode.is_extend()

    def copy(self):
        return _B(self.label + "c", self.forward_mode)


def _render(value):
    if isinstance(value, _B):
        return value.label
    if isinstance(value, InflightQueue):
        return f"IQ({value.depth},{value.blocking},{value.ring})"
    if isinstance(value, NonCallableMock):
        name = value._extract_mock_name()
        return name[2:] if name.startswith("s.") else name
    if isinstance(value, torch.Tensor):
        return "tensor"
    if isinstance(value, SimpleNamespace):
        return "md"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_render(v) for v in value) + "]"
    return repr(value)


def _render_call(name, args, kwargs):
    parts = [_render(a) for a in args]
    parts += [f"{k}={_render(v)}" for k, v in kwargs.items()]
    return f"{name}({','.join(parts)})"


class _LoggedMock(MagicMock):
    def _get_child_mock(self, **kwargs):
        return _LoggedMock(**kwargs)

    def __call__(self, *args, **kwargs):
        name = self._extract_mock_name()
        self._log_sink.append(_render_call(name[2:], args, kwargs))
        return super().__call__(*args, **kwargs)


class _FakeScheduler(_LoggedMock):
    def __setattr__(self, name, value):
        if name in _WATCHED:
            self._log_sink.append(f"{name}={_render(value)}")
        super().__setattr__(name, value)


def _make_scheduler(log):
    _LoggedMock._log_sink = log
    s = _FakeScheduler(name="s")
    s.gracefully_exit = False
    s._engine_paused = False
    s.is_generation = True
    s.enable_staging = True
    s.enable_unified_memory = True
    s._deferred_input_requests = []
    s.chunked_req = None
    s.last_batch = None
    s.ngram_embedding_manager.prepare_for_forward.side_effect = (
        lambda batch, chunked_req: batch
    )
    s.dp_attn_adapter.maybe_prepare_mlp_sync_batch.side_effect = lambda batch: batch
    s.run_batch.side_effect = lambda batch: _B("r" + batch.label)
    s._commit_due_step.side_effect = MethodType(Scheduler._commit_due_step, s)
    del log[:]
    return s


def _script_ingest(s, n, paused=(), exit_at=None, inputs=None):
    """ingest_requests pauses the engine on the listed calls, asks the loop to
    exit after call exit_at, and stops the run after n calls."""
    calls = iter(range(n + 1))

    def ingest(*args, **kwargs):
        i = next(calls)
        if i == n:
            raise _Stop
        s._engine_paused = i in paused
        if exit_at is not None and i >= exit_at:
            s.gracefully_exit = True
        return (inputs or {}).get(i, [])

    s.ingest_requests.side_effect = ingest


def _plans(batches):
    plans = iter(batches)

    def plan(*args, **kwargs):
        return SimpleNamespace(batch_to_run=next(plans, None))

    return plan


def _values(values, default):
    it = iter(values)
    return lambda *args, **kwargs: next(it, default)


def _lines(log):
    """One line per loop iteration or pipeline slot: split at each intake, or
    at the prefetch before it, and at each slot's row binding."""
    lines = []
    for entry in log:
        prefetched = bool(lines) and lines[-1][-1].startswith(_PREFETCH)
        starts = entry.startswith(_PREFETCH) or entry.startswith("decoding_batch=")
        if (
            not lines
            or starts
            or (entry.startswith("ingest_requests(") and not prefetched)
        ):
            lines.append([])
        lines[-1].append(entry)
    return ["; ".join(line) for line in lines]


class _QueueLog:
    def __init__(self, log):
        self.log = log
        self.patches = []
        for name in _QUEUE_CHANGES:
            orig = getattr(InflightQueue, name)

            def logged(q, *args, _name=name, _orig=orig, **kwargs):
                self.log.append(_render_call("iq." + _name, args, kwargs))
                return _orig(q, *args, **kwargs)

            self.patches.append(patch.object(InflightQueue, name, logged))

    def __enter__(self):
        for p in self.patches:
            p.__enter__()

    def __exit__(self, *exc):
        for p in reversed(self.patches):
            p.__exit__(*exc)


class _LoopOrderCase(CustomTestCase):
    maxDiff = None

    def setUp(self):
        super().setUp()
        enter_scope(self, published_topology(role="scheduler"))
        enter_scope(self, envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1))
        self.log = []
        enter_scope(self, _QueueLog(self.log))

    def run_loop(self, loop, s):
        try:
            loop(s)
        except _Stop:
            pass
        return _lines(self.log)

    def check(self, name, lines):
        self.assertEqual(lines, GOLDEN[name].strip().splitlines(), name)


A, B, C = (_B(label) for label in "ABC")


class TestFlatLoopOrder(_LoopOrderCase):
    def flat_scheduler(self, batches, disable=(), paused=(2,), n=8, exit_at=None):
        s = _make_scheduler(self.log)
        _script_ingest(s, n, paused=paused, exit_at=exit_at)
        s.get_next_batch_to_run.side_effect = _plans(batches)
        s.get_next_disagg_prefill_batch_to_run.side_effect = _plans(batches)
        s.get_next_disagg_decode_batch_to_run.side_effect = _plans(batches)
        s.is_disable_overlap_for_batch.side_effect = _values(disable, False)
        s.disagg_prefill_bootstrap_queue.pop_bootstrapped.return_value = []
        return s

    def test_flat_loops(self):
        schedule = [A, B, None, None, C, None]
        for name, loop in (
            ("normal", Scheduler.event_loop_normal),
            ("overlap", Scheduler.event_loop_overlap),
            ("pd_prefill_normal", Scheduler.event_loop_normal_disagg_prefill),
            ("pd_decode_normal", Scheduler.event_loop_normal_disagg_decode),
            ("pd_decode_overlap", Scheduler.event_loop_overlap_disagg_decode),
        ):
            with self.subTest(name):
                s = self.flat_scheduler(schedule, disable=[False, True])
                self.check(name, self.run_loop(loop, s))

    def test_overlap_exit_and_non_generation(self):
        s = self.flat_scheduler([A, B, C], exit_at=2)
        s.is_generation = False
        s.enable_unified_memory = False
        s.is_disable_overlap_for_batch.side_effect = _values([False, True], False)
        self.check("overlap_exit", self.run_loop(Scheduler.event_loop_overlap, s))

    def test_pd_overlap_loops_sample_for_non_generation(self):
        for name, loop in (
            (
                "pd_prefill_overlap_non_generation",
                Scheduler.event_loop_overlap_disagg_prefill,
            ),
            (
                "pd_decode_overlap_non_generation",
                Scheduler.event_loop_overlap_disagg_decode,
            ),
        ):
            with self.subTest(name):
                s = self.flat_scheduler([A, B], paused=(), n=3)
                s.is_generation = False
                s._is_continuous_input_polling_enabled.return_value = False
                self.check(name, self.run_loop(loop, s))

    def test_pd_prefill_overlap_blocking(self):
        s = self.flat_scheduler([A, B, None, None, C, None])
        s._is_continuous_input_polling_enabled.return_value = False
        self.check(
            "pd_prefill_overlap",
            self.run_loop(Scheduler.event_loop_overlap_disagg_prefill, s),
        )

    def test_pd_prefill_overlap_polling(self):
        # A and B launch; A's copy is pending for two passes, during which new
        # input clears the skip memo and the queue is full; B is ready at
        # once; then a failed plan sets the skip memo, a deferred request and
        # a bootstrap arrival gate the next plans.
        s = self.flat_scheduler([A, B, None, None, C, None], paused=(), n=11)
        s._is_continuous_input_polling_enabled.return_value = True
        s.is_disagg_prefill_batch_result_ready.side_effect = _values(
            [False, False, True, True, False, True], True
        )
        _script_ingest(s, 11, inputs={2: ["req"]})
        bootstrapped = {6: ["b"]}
        calls = iter(range(100))
        s.disagg_prefill_bootstrap_queue.pop_bootstrapped.side_effect = lambda: (
            bootstrapped.get(next(calls), [])
        )
        deferred = iter([[], [], [], [], [], ["d"]])
        orig_ingest = s.ingest_requests.side_effect

        def ingest(*args, **kwargs):
            out = orig_ingest(*args, **kwargs)
            s._deferred_input_requests = next(deferred, [])
            return out

        s.ingest_requests.side_effect = ingest
        self.check(
            "pd_prefill_polling",
            self.run_loop(Scheduler.event_loop_overlap_disagg_prefill, s),
        )

    def test_pd_prefill_polling_shutdown_while_waiting(self):
        s = self.flat_scheduler([A, B, C], paused=(), n=6, exit_at=2)
        s._is_continuous_input_polling_enabled.return_value = True
        s.is_disagg_prefill_batch_result_ready.side_effect = _values([False], True)
        self.check(
            "pd_prefill_polling_exit",
            self.run_loop(Scheduler.event_loop_overlap_disagg_prefill, s),
        )

    def test_pd_prefill_polling_pause_while_waiting(self):
        s = self.flat_scheduler([A, B, C], paused=(2,), n=6)
        s._is_continuous_input_polling_enabled.return_value = True
        s.is_disagg_prefill_batch_result_ready.side_effect = _values([False], True)
        self.check(
            "pd_prefill_polling_pause",
            self.run_loop(Scheduler.event_loop_overlap_disagg_prefill, s),
        )


def _pp_init(s, loop_size, spec_relay=False, metadata=None):
    def init():
        s.pp_loop_size = loop_size
        s.inflight_steps = InflightQueue(depth=loop_size - 1, ring=True)
        s.decoding_batches = [_B(f"db{i}") for i in range(loop_size)]
        s.mb_metadata = list(metadata or [None] * loop_size)
        s.pp_outputs = None
        s.last_rank_comm_queue = _B("lrcq")
        s._pp_spec_relay = spec_relay
        s.send_req_work = []

    s.init_pp_loop_state.side_effect = init


class TestRingLoopOrder(_LoopOrderCase):
    def ring_scheduler(self, batches, loop_size, last_rank, rounds=3, **init):
        s = _make_scheduler(self.log)
        _pp_init(s, loop_size, **init)
        _script_ingest(s, rounds * loop_size)
        s.pp_group = SimpleNamespace(is_last_rank=last_rank)
        s.waiting_queue = []
        s.disagg_prefill_inflight_queue = []
        s.disagg_decode_transfer_queue.queue = []
        s.disagg_decode_prealloc_queue.queue = []
        s.decode_offload_manager.ongoing_offload = []
        s.get_next_batch_to_run.side_effect = _plans(batches)
        s.get_next_disagg_decode_batch_to_run.side_effect = _plans(batches)
        s.get_new_batch_prefill.side_effect = _plans(batches)
        for name in (
            "_pp_slot_last_batch",
            "_pp_pop_due_batch",
            "_pp_debug_check_slot",
        ):
            getattr(s, name).side_effect = MethodType(getattr(Scheduler, name), s)
        s._pp_recv_proxy_tensors.return_value = _B("proxy")
        s._pp_launch_batch.side_effect = lambda mb_id, batch, *rest: (
            _B("res" + batch.label),
            _B("ev" + batch.label),
        )
        s._pp_commit_send_output_work_and_preprocess_output_tensors.side_effect = (
            lambda first, nxt: (_B(f"out{nxt}"), _B(f"nres{nxt}"), s.d2h_event)
        )
        s._pp_send_pyobj_to_next_stage.side_effect = lambda data, async_send: _B(
            "work:" + _render(data)
        )
        s._pp_recv_pyobj_from_prev_stage.return_value = _B("recv")
        s._pp_pd_send_consensus_release_ids.side_effect = lambda tmbs, first, rel, ids: (
            _B("relwork"),
            _B("rel:" + _render(ids)),
        )
        s._pp_pd_send_consensus_bootstrapped_ids.side_effect = (
            lambda mbs, first, cons, ids: (_B("conswork"), _B("cons:" + _render(ids)))
        )
        s.process_retract_queue.side_effect = lambda ids: _B("pr:" + _render(ids))
        s.process_prealloc_queue.side_effect = lambda ids: _B("pp:" + _render(ids))
        s.process_decode_transfer_queue.side_effect = lambda ids: _B(
            "pt:" + _render(ids)
        )
        return s

    def run_ring(self, name, loop, s, pp_size, depth, offload=False):
        with get_parallel().override(pp_size=pp_size, pp_async_batch_depth=depth):
            with get_disagg().override(
                disaggregation_decode_enable_offload_kvcache=offload
            ):
                self.check(name, self.run_loop(loop, s))

    def test_pp(self):
        d = ForwardMode.DECODE
        for name, kwargs, last_rank, depth in (
            ("pp_l2", {}, False, 0),
            ("pp_l3_last", {}, True, 1),
            (
                "pp_l2_spec_relay",
                dict(
                    spec_relay=True,
                    metadata=[None, SimpleNamespace(fwd_batch=_B("F1"))],
                ),
                False,
                0,
            ),
        ):
            with self.subTest(name):
                batches = [A, _B("D", d), None, _B("E", d), None, None]
                s = self.ring_scheduler(batches, 2 + depth, last_rank, **kwargs)
                self.run_ring(name, Scheduler.event_loop_pp, s, 2, depth)

    def test_pp_disagg_prefill(self):
        for name, last_rank, depth in (("pdp_l2", False, 0), ("pdp_l3_last", True, 1)):
            with self.subTest(name):
                s = self.ring_scheduler(
                    [A, B, None, C, None, None], 2 + depth, last_rank
                )
                transferred = iter(
                    [None, ["t1"], None, ["t3"], None, None, None, None, None]
                )
                s._pp_pd_get_prefill_transferred_ids.side_effect = lambda: next(
                    transferred
                )
                if name == "pdp_l2":
                    s.disagg_prefill_inflight_queue = [object()]
                self.run_ring(name, Scheduler.event_loop_pp_disagg_prefill, s, 2, depth)

    def test_pp_disagg_decode(self):
        d = ForwardMode.DECODE
        for name, last_rank, depth, offload in (
            ("pdd_l2", False, 0, True),
            ("pdd_l3_last", True, 1, False),
        ):
            with self.subTest(name):
                batches = [_B("A", d), _B("B", d), None, _B("C", d), None, None]
                s = self.ring_scheduler(batches, 2 + depth, last_rank)
                ids = {
                    "_pp_pd_get_retract_ids": [["r0"], None, None, ["r3"]],
                    "_pp_pd_get_prealloc_ids": [None, ["p1"], None],
                    "_pp_pd_get_decode_transferred_ids": [None, None, ["t2"]],
                }
                for method, values in ids.items():
                    getattr(s, method).side_effect = _values(values, None)
                self.run_ring(
                    name, Scheduler.event_loop_pp_disagg_decode, s, 2, depth, offload
                )


# Recorded on the hand-written loops; never edit to make a change pass.
GOLDEN = {
    "normal": """
inflight_steps=IQ(0,True,False); rows_join_before_plan=True
ingest_requests(); get_next_batch_to_run(decoding_batch=decoding_batch,last_batch=None); cur_batch_for_debug=A; run_batch(A); _update_rows_at_launch(A); iq.push(A,rA); _commit_due_step(); iq.pop_due(); process_batch_result(A,rA); leave_finished_rows(decoding_batch); last_batch=A; iq.end_step(A); _debug_check_last_launched(); invariant_checker.self_check_during_busy()
ingest_requests(); get_next_batch_to_run(decoding_batch=decoding_batch,last_batch=A); cur_batch_for_debug=B; run_batch(B); _update_rows_at_launch(B); iq.push(B,rB); _commit_due_step(); iq.pop_due(); process_batch_result(B,rB); leave_finished_rows(decoding_batch); last_batch=B; iq.end_step(B); _debug_check_last_launched(); invariant_checker.self_check_during_busy()
ingest_requests(); _record_scheduler_state_for_paused_engine()
ingest_requests(); get_next_batch_to_run(decoding_batch=decoding_batch,last_batch=B); cur_batch_for_debug=None; _sched_idled=True; on_idle(); last_batch=None; iq.end_step(None); _debug_check_last_launched(); invariant_checker.self_check_during_busy()
ingest_requests(); get_next_batch_to_run(decoding_batch=decoding_batch,last_batch=None); cur_batch_for_debug=None; _sched_idled=True; on_idle(); last_batch=None; iq.end_step(None); _debug_check_last_launched(); invariant_checker.self_check_during_busy()
ingest_requests(); get_next_batch_to_run(decoding_batch=decoding_batch,last_batch=None); cur_batch_for_debug=C; run_batch(C); _update_rows_at_launch(C); iq.push(C,rC); _commit_due_step(); iq.pop_due(); process_batch_result(C,rC); leave_finished_rows(decoding_batch); last_batch=C; iq.end_step(C); _debug_check_last_launched(); invariant_checker.self_check_during_busy()
ingest_requests(); get_next_batch_to_run(decoding_batch=decoding_batch,last_batch=C); cur_batch_for_debug=None; _sched_idled=True; on_idle(); last_batch=None; iq.end_step(None); _debug_check_last_launched(); invariant_checker.self_check_during_busy()
ingest_requests(); get_next_batch_to_run(decoding_batch=decoding_batch,last_batch=None); cur_batch_for_debug=None; _sched_idled=True; on_idle(); last_batch=None; iq.end_step(None); _debug_check_last_launched(); invariant_checker.self_check_during_busy()
ingest_requests()
""",
    "overlap": """
enable_skip_finishing_decode=True; inflight_steps=IQ(1,True,False); rows_join_before_plan=True
ingest_requests(); get_next_batch_to_run(decoding_batch=decoding_batch,last_batch=None); cur_batch_for_debug=A; is_disable_overlap_for_batch(A,last_batch=None); run_batch(A); _apply_war_barrier(); iq.push(Ac,rA); _update_rows_at_launch(A); _debug_check_commit_gate(False); leave_finished_rows(decoding_batch); launch_batch_sample_if_needed(rA,A); last_batch=A; iq.end_step(A); _debug_check_last_launched(); invariant_checker.self_check_during_busy()
ingest_requests(); get_next_batch_to_run(decoding_batch=decoding_batch,last_batch=A); cur_batch_for_debug=B; is_disable_overlap_for_batch(B,last_batch=A); _commit_due_step(); iq.pop_due(); process_batch_result(Ac,rA); token_to_kv_pool_allocator.flush_opportunistic(); run_batch(B); _apply_war_barrier(); iq.push(Bc,rB); _update_rows_at_launch(B); _debug_check_commit_gate(False); leave_finished_rows(decoding_batch); launch_batch_sample_if_needed(rB,B); last_batch=B; iq.end_step(B); _debug_check_last_launched(); invariant_checker.self_check_during_busy()
ingest_requests(); _record_scheduler_state_for_paused_engine()
ingest_requests(); get_next_batch_to_run(decoding_batch=decoding_batch,last_batch=B); cur_batch_for_debug=None; is_disable_overlap_for_batch(None,last_batch=B); _sched_idled=True; _debug_check_commit_gate(True); _commit_due_step(); iq.pop_due(); process_batch_result(Bc,rB); leave_finished_rows(decoding_batch); launch_batch_sample_if_needed(None,None); last_batch=None; iq.end_step(None); _debug_check_last_launched(); invariant_checker.self_check_during_busy()
ingest_requests(); get_next_batch_to_run(decoding_batch=decoding_batch,last_batch=None); cur_batch_for_debug=None; is_disable_overlap_for_batch(None,last_batch=None); _sched_idled=True; _debug_check_commit_gate(False); on_idle(); leave_finished_rows(decoding_batch); launch_batch_sample_if_needed(None,None); last_batch=None; iq.end_step(None); _debug_check_last_launched(); invariant_checker.self_check_during_busy()
ingest_requests(); get_next_batch_to_run(decoding_batch=decoding_batch,last_batch=None); cur_batch_for_debug=C; is_disable_overlap_for_batch(C,last_batch=None); run_batch(C); _apply_war_barrier(); iq.push(Cc,rC); _update_rows_at_launch(C); _debug_check_commit_gate(False); leave_finished_rows(decoding_batch); launch_batch_sample_if_needed(rC,C); last_batch=C; iq.end_step(C); _debug_check_last_launched(); invariant_checker.self_check_during_busy()
ingest_requests(); get_next_batch_to_run(decoding_batch=decoding_batch,last_batch=C); cur_batch_for_debug=None; is_disable_overlap_for_batch(None,last_batch=C); _sched_idled=True; _debug_check_commit_gate(True); _commit_due_step(); iq.pop_due(); process_batch_result(Cc,rC); leave_finished_rows(decoding_batch); launch_batch_sample_if_needed(None,None); last_batch=None; iq.end_step(None); _debug_check_last_launched(); invariant_checker.self_check_during_busy()
ingest_requests(); get_next_batch_to_run(decoding_batch=decoding_batch,last_batch=None); cur_batch_for_debug=None; is_disable_overlap_for_batch(None,last_batch=None); _sched_idled=True; _debug_check_commit_gate(False); on_idle(); leave_finished_rows(decoding_batch); launch_batch_sample_if_needed(None,None); last_batch=None; iq.end_step(None); _debug_check_last_launched(); invariant_checker.self_check_during_busy()
ingest_requests()
""",
    "overlap_exit": """
enable_skip_finishing_decode=True; inflight_steps=IQ(1,True,False); rows_join_before_plan=True
ingest_requests(); get_next_batch_to_run(decoding_batch=decoding_batch,last_batch=None); cur_batch_for_debug=A; is_disable_overlap_for_batch(A,last_batch=None); run_batch(A); _apply_war_barrier(); iq.push(Ac,rA); _update_rows_at_launch(A); _debug_check_commit_gate(False); leave_finished_rows(decoding_batch); last_batch=A; iq.end_step(A); _debug_check_last_launched(); invariant_checker.self_check_during_busy()
ingest_requests(); get_next_batch_to_run(decoding_batch=decoding_batch,last_batch=A); cur_batch_for_debug=B; is_disable_overlap_for_batch(B,last_batch=A); _commit_due_step(); iq.pop_due(); process_batch_result(Ac,rA); run_batch(B); _apply_war_barrier(); iq.push(Bc,rB); _update_rows_at_launch(B); _debug_check_commit_gate(False); leave_finished_rows(decoding_batch); last_batch=B; iq.end_step(B); _debug_check_last_launched(); invariant_checker.self_check_during_busy()
ingest_requests(); _record_scheduler_state_for_paused_engine()
""",
    "pd_decode_normal": """
inflight_steps=IQ(0,True,False)
disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries(); ingest_requests(); process_decode_queue(); get_next_disagg_decode_batch_to_run(decoding_batch=decoding_batch); ngram_embedding_manager.prepare_for_forward(A,chunked_req=None); cur_batch_for_debug=A; run_batch(A); decoding_batch.absorb(); iq.push(A,rA); _commit_due_step(); iq.pop_due(); process_batch_result(A,rA); leave_finished_rows(decoding_batch); last_batch=A; iq.end_step(A); _debug_check_last_launched()
disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries(); ingest_requests(); process_decode_queue(); get_next_disagg_decode_batch_to_run(decoding_batch=decoding_batch); ngram_embedding_manager.prepare_for_forward(B,chunked_req=None); cur_batch_for_debug=B; run_batch(B); decoding_batch.absorb(); iq.push(B,rB); _commit_due_step(); iq.pop_due(); process_batch_result(B,rB); leave_finished_rows(decoding_batch); last_batch=B; iq.end_step(B); _debug_check_last_launched()
disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries(); ingest_requests(); _record_scheduler_state_for_paused_engine()
ingest_requests(); process_decode_queue(); get_next_disagg_decode_batch_to_run(decoding_batch=decoding_batch); ngram_embedding_manager.prepare_for_forward(None,chunked_req=None); cur_batch_for_debug=None; _sched_idled=True; on_idle(); last_batch=None; iq.end_step(None); _debug_check_last_launched()
disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries(); ingest_requests(); process_decode_queue(); get_next_disagg_decode_batch_to_run(decoding_batch=decoding_batch); ngram_embedding_manager.prepare_for_forward(None,chunked_req=None); cur_batch_for_debug=None; _sched_idled=True; on_idle(); last_batch=None; iq.end_step(None); _debug_check_last_launched()
disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries(); ingest_requests(); process_decode_queue(); get_next_disagg_decode_batch_to_run(decoding_batch=decoding_batch); ngram_embedding_manager.prepare_for_forward(C,chunked_req=None); cur_batch_for_debug=C; run_batch(C); decoding_batch.absorb(); iq.push(C,rC); _commit_due_step(); iq.pop_due(); process_batch_result(C,rC); leave_finished_rows(decoding_batch); last_batch=C; iq.end_step(C); _debug_check_last_launched()
disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries(); ingest_requests(); process_decode_queue(); get_next_disagg_decode_batch_to_run(decoding_batch=decoding_batch); ngram_embedding_manager.prepare_for_forward(None,chunked_req=None); cur_batch_for_debug=None; _sched_idled=True; on_idle(); last_batch=None; iq.end_step(None); _debug_check_last_launched()
disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries(); ingest_requests(); process_decode_queue(); get_next_disagg_decode_batch_to_run(decoding_batch=decoding_batch); ngram_embedding_manager.prepare_for_forward(None,chunked_req=None); cur_batch_for_debug=None; _sched_idled=True; on_idle(); last_batch=None; iq.end_step(None); _debug_check_last_launched()
disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries(); ingest_requests()
""",
    "pd_decode_overlap": """
inflight_steps=IQ(1,True,False); last_batch=None
disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries(); ingest_requests(); process_decode_queue(); get_next_disagg_decode_batch_to_run(decoding_batch=decoding_batch); ngram_embedding_manager.prepare_for_forward(A,chunked_req=None); cur_batch_for_debug=A; is_disable_overlap_for_batch(A,last_batch=None); run_batch(A); _apply_war_barrier(); iq.push(Ac,rA); decoding_batch.absorb(); _debug_check_commit_gate(False); leave_finished_rows(decoding_batch); launch_batch_sample_if_needed(rA,A); last_batch=A; iq.end_step(A); _debug_check_last_launched()
disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries(); ingest_requests(); process_decode_queue(); get_next_disagg_decode_batch_to_run(decoding_batch=decoding_batch); ngram_embedding_manager.prepare_for_forward(B,chunked_req=None); cur_batch_for_debug=B; is_disable_overlap_for_batch(B,last_batch=A); _commit_due_step(); iq.pop_due(); process_batch_result(Ac,rA); run_batch(B); _apply_war_barrier(); iq.push(Bc,rB); decoding_batch.absorb(); _debug_check_commit_gate(False); leave_finished_rows(decoding_batch); launch_batch_sample_if_needed(rB,B); last_batch=B; iq.end_step(B); _debug_check_last_launched()
disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries(); ingest_requests(); _record_scheduler_state_for_paused_engine()
ingest_requests(); process_decode_queue(); get_next_disagg_decode_batch_to_run(decoding_batch=decoding_batch); ngram_embedding_manager.prepare_for_forward(None,chunked_req=None); cur_batch_for_debug=None; is_disable_overlap_for_batch(None,last_batch=B); _sched_idled=True; _debug_check_commit_gate(True); _commit_due_step(); iq.pop_due(); process_batch_result(Bc,rB); leave_finished_rows(decoding_batch); launch_batch_sample_if_needed(None,None); last_batch=None; iq.end_step(None); _debug_check_last_launched()
disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries(); ingest_requests(); process_decode_queue(); get_next_disagg_decode_batch_to_run(decoding_batch=decoding_batch); ngram_embedding_manager.prepare_for_forward(None,chunked_req=None); cur_batch_for_debug=None; is_disable_overlap_for_batch(None,last_batch=None); _sched_idled=True; _debug_check_commit_gate(False); on_idle(); leave_finished_rows(decoding_batch); launch_batch_sample_if_needed(None,None); last_batch=None; iq.end_step(None); _debug_check_last_launched()
disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries(); ingest_requests(); process_decode_queue(); get_next_disagg_decode_batch_to_run(decoding_batch=decoding_batch); ngram_embedding_manager.prepare_for_forward(C,chunked_req=None); cur_batch_for_debug=C; is_disable_overlap_for_batch(C,last_batch=None); run_batch(C); _apply_war_barrier(); iq.push(Cc,rC); decoding_batch.absorb(); _debug_check_commit_gate(False); leave_finished_rows(decoding_batch); launch_batch_sample_if_needed(rC,C); last_batch=C; iq.end_step(C); _debug_check_last_launched()
disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries(); ingest_requests(); process_decode_queue(); get_next_disagg_decode_batch_to_run(decoding_batch=decoding_batch); ngram_embedding_manager.prepare_for_forward(None,chunked_req=None); cur_batch_for_debug=None; is_disable_overlap_for_batch(None,last_batch=C); _sched_idled=True; _debug_check_commit_gate(True); _commit_due_step(); iq.pop_due(); process_batch_result(Cc,rC); leave_finished_rows(decoding_batch); launch_batch_sample_if_needed(None,None); last_batch=None; iq.end_step(None); _debug_check_last_launched()
disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries(); ingest_requests(); process_decode_queue(); get_next_disagg_decode_batch_to_run(decoding_batch=decoding_batch); ngram_embedding_manager.prepare_for_forward(None,chunked_req=None); cur_batch_for_debug=None; is_disable_overlap_for_batch(None,last_batch=None); _sched_idled=True; _debug_check_commit_gate(False); on_idle(); leave_finished_rows(decoding_batch); launch_batch_sample_if_needed(None,None); last_batch=None; iq.end_step(None); _debug_check_last_launched()
disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries(); ingest_requests()
""",
    "pd_decode_overlap_non_generation": """
inflight_steps=IQ(1,True,False); last_batch=None
disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries(); ingest_requests(); process_decode_queue(); get_next_disagg_decode_batch_to_run(decoding_batch=decoding_batch); ngram_embedding_manager.prepare_for_forward(A,chunked_req=None); cur_batch_for_debug=A; is_disable_overlap_for_batch(A,last_batch=None); run_batch(A); _apply_war_barrier(); iq.push(Ac,rA); decoding_batch.absorb(); _debug_check_commit_gate(False); leave_finished_rows(decoding_batch); launch_batch_sample_if_needed(rA,A); last_batch=A; iq.end_step(A); _debug_check_last_launched()
disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries(); ingest_requests(); process_decode_queue(); get_next_disagg_decode_batch_to_run(decoding_batch=decoding_batch); ngram_embedding_manager.prepare_for_forward(B,chunked_req=None); cur_batch_for_debug=B; is_disable_overlap_for_batch(B,last_batch=A); run_batch(B); _apply_war_barrier(); iq.push(Bc,rB); decoding_batch.absorb(); _debug_check_commit_gate(True); _commit_due_step(); iq.pop_due(); process_batch_result(Ac,rA); leave_finished_rows(decoding_batch); launch_batch_sample_if_needed(rB,B); last_batch=B; iq.end_step(B); _debug_check_last_launched()
disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries(); ingest_requests(); process_decode_queue(); get_next_disagg_decode_batch_to_run(decoding_batch=decoding_batch); ngram_embedding_manager.prepare_for_forward(None,chunked_req=None); cur_batch_for_debug=None; is_disable_overlap_for_batch(None,last_batch=B); _sched_idled=True; _debug_check_commit_gate(True); _commit_due_step(); iq.pop_due(); process_batch_result(Bc,rB); leave_finished_rows(decoding_batch); launch_batch_sample_if_needed(None,None); last_batch=None; iq.end_step(None); _debug_check_last_launched()
disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries(); ingest_requests()
""",
    "pd_prefill_normal": """
inflight_steps=IQ(0,True,False)
ingest_requests(); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=None); ngram_embedding_manager.prepare_for_forward(A,chunked_req=None); cur_batch_for_debug=A; maybe_prefetch_staging_for_batch(A); run_batch(A); iq.push(A,rA); _commit_due_step(); iq.pop_due(); process_batch_result(A,rA); process_disagg_prefill_inflight_queue(); last_batch=A; iq.end_step(A); _debug_check_last_launched()
ingest_requests(); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=A); ngram_embedding_manager.prepare_for_forward(B,chunked_req=None); cur_batch_for_debug=B; maybe_prefetch_staging_for_batch(B); run_batch(B); iq.push(B,rB); _commit_due_step(); iq.pop_due(); process_batch_result(B,rB); process_disagg_prefill_inflight_queue(); last_batch=B; iq.end_step(B); _debug_check_last_launched()
ingest_requests(); _record_scheduler_state_for_paused_engine()
ingest_requests(); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=B); ngram_embedding_manager.prepare_for_forward(None,chunked_req=None); cur_batch_for_debug=None; _sched_idled=True; on_idle(); process_disagg_prefill_inflight_queue(); last_batch=None; iq.end_step(None); _debug_check_last_launched()
ingest_requests(); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=None); ngram_embedding_manager.prepare_for_forward(None,chunked_req=None); cur_batch_for_debug=None; _sched_idled=True; on_idle(); process_disagg_prefill_inflight_queue(); last_batch=None; iq.end_step(None); _debug_check_last_launched()
ingest_requests(); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=None); ngram_embedding_manager.prepare_for_forward(C,chunked_req=None); cur_batch_for_debug=C; maybe_prefetch_staging_for_batch(C); run_batch(C); iq.push(C,rC); _commit_due_step(); iq.pop_due(); process_batch_result(C,rC); process_disagg_prefill_inflight_queue(); last_batch=C; iq.end_step(C); _debug_check_last_launched()
ingest_requests(); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=C); ngram_embedding_manager.prepare_for_forward(None,chunked_req=None); cur_batch_for_debug=None; _sched_idled=True; on_idle(); process_disagg_prefill_inflight_queue(); last_batch=None; iq.end_step(None); _debug_check_last_launched()
ingest_requests(); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=None); ngram_embedding_manager.prepare_for_forward(None,chunked_req=None); cur_batch_for_debug=None; _sched_idled=True; on_idle(); process_disagg_prefill_inflight_queue(); last_batch=None; iq.end_step(None); _debug_check_last_launched()
ingest_requests()
""",
    "pd_prefill_overlap": """
_is_continuous_input_polling_enabled(); enable_continuous_input_polling=False; inflight_steps=IQ(1,True,False)
ingest_requests(stop_at_pause=False); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=None,should_retry_storage_prefetch=True); ngram_embedding_manager.prepare_for_forward(A,chunked_req=None); cur_batch_for_debug=A; maybe_prefetch_staging_for_batch(A); run_batch(A); _apply_war_barrier(); iq.push(Ac,rA); _debug_check_commit_gate(False); process_disagg_prefill_inflight_queue(); launch_batch_sample_if_needed(rA,A); _sched_idled=False; last_batch=A; iq.end_step(A); _debug_check_last_launched()
ingest_requests(stop_at_pause=False); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=A,should_retry_storage_prefetch=True); ngram_embedding_manager.prepare_for_forward(B,chunked_req=None); cur_batch_for_debug=B; maybe_prefetch_staging_for_batch(B); run_batch(B); _apply_war_barrier(); iq.push(Bc,rB); _debug_check_commit_gate(True); _commit_due_step(); iq.pop_due(); process_batch_result(Ac,rA); process_disagg_prefill_inflight_queue(); launch_batch_sample_if_needed(rB,B); _sched_idled=False; last_batch=B; iq.end_step(B); _debug_check_last_launched()
ingest_requests(stop_at_pause=False); _record_scheduler_state_for_paused_engine(); _yield_gil_if_needed()
ingest_requests(stop_at_pause=False); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=B,should_retry_storage_prefetch=True); ngram_embedding_manager.prepare_for_forward(None,chunked_req=None); cur_batch_for_debug=None; _debug_check_commit_gate(True); _commit_due_step(); iq.pop_due(); process_batch_result(Bc,rB); process_disagg_prefill_inflight_queue(); launch_batch_sample_if_needed(None,None); _sched_idled=True; last_batch=None; iq.end_step(None); _debug_check_last_launched()
ingest_requests(stop_at_pause=False); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=None,should_retry_storage_prefetch=True); ngram_embedding_manager.prepare_for_forward(None,chunked_req=None); cur_batch_for_debug=None; _debug_check_commit_gate(False); on_idle(); process_disagg_prefill_inflight_queue(); launch_batch_sample_if_needed(None,None); _sched_idled=True; last_batch=None; iq.end_step(None); _debug_check_last_launched()
ingest_requests(stop_at_pause=False); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=None,should_retry_storage_prefetch=True); ngram_embedding_manager.prepare_for_forward(C,chunked_req=None); cur_batch_for_debug=C; maybe_prefetch_staging_for_batch(C); run_batch(C); _apply_war_barrier(); iq.push(Cc,rC); _debug_check_commit_gate(False); process_disagg_prefill_inflight_queue(); launch_batch_sample_if_needed(rC,C); _sched_idled=False; last_batch=C; iq.end_step(C); _debug_check_last_launched()
ingest_requests(stop_at_pause=False); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=C,should_retry_storage_prefetch=True); ngram_embedding_manager.prepare_for_forward(None,chunked_req=None); cur_batch_for_debug=None; _debug_check_commit_gate(True); _commit_due_step(); iq.pop_due(); process_batch_result(Cc,rC); process_disagg_prefill_inflight_queue(); launch_batch_sample_if_needed(None,None); _sched_idled=True; last_batch=None; iq.end_step(None); _debug_check_last_launched()
ingest_requests(stop_at_pause=False); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=None,should_retry_storage_prefetch=True); ngram_embedding_manager.prepare_for_forward(None,chunked_req=None); cur_batch_for_debug=None; _debug_check_commit_gate(False); on_idle(); process_disagg_prefill_inflight_queue(); launch_batch_sample_if_needed(None,None); _sched_idled=True; last_batch=None; iq.end_step(None); _debug_check_last_launched()
ingest_requests(stop_at_pause=False)
""",
    "pd_prefill_overlap_non_generation": """
_is_continuous_input_polling_enabled(); enable_continuous_input_polling=False; inflight_steps=IQ(1,True,False)
ingest_requests(stop_at_pause=False); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=None,should_retry_storage_prefetch=True); ngram_embedding_manager.prepare_for_forward(A,chunked_req=None); cur_batch_for_debug=A; maybe_prefetch_staging_for_batch(A); run_batch(A); _apply_war_barrier(); iq.push(Ac,rA); _debug_check_commit_gate(False); process_disagg_prefill_inflight_queue(); launch_batch_sample_if_needed(rA,A); _sched_idled=False; last_batch=A; iq.end_step(A); _debug_check_last_launched()
ingest_requests(stop_at_pause=False); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=A,should_retry_storage_prefetch=True); ngram_embedding_manager.prepare_for_forward(B,chunked_req=None); cur_batch_for_debug=B; maybe_prefetch_staging_for_batch(B); run_batch(B); _apply_war_barrier(); iq.push(Bc,rB); _debug_check_commit_gate(True); _commit_due_step(); iq.pop_due(); process_batch_result(Ac,rA); process_disagg_prefill_inflight_queue(); launch_batch_sample_if_needed(rB,B); _sched_idled=False; last_batch=B; iq.end_step(B); _debug_check_last_launched()
ingest_requests(stop_at_pause=False); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=B,should_retry_storage_prefetch=True); ngram_embedding_manager.prepare_for_forward(None,chunked_req=None); cur_batch_for_debug=None; _debug_check_commit_gate(True); _commit_due_step(); iq.pop_due(); process_batch_result(Bc,rB); process_disagg_prefill_inflight_queue(); launch_batch_sample_if_needed(None,None); _sched_idled=True; last_batch=None; iq.end_step(None); _debug_check_last_launched()
ingest_requests(stop_at_pause=False)
""",
    "pd_prefill_polling": """
_is_continuous_input_polling_enabled(); enable_continuous_input_polling=True; inflight_steps=IQ(1,False,False)
ingest_requests(stop_at_pause=False); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=None,should_retry_storage_prefetch=True); ngram_embedding_manager.prepare_for_forward(A,chunked_req=None); cur_batch_for_debug=A; maybe_prefetch_staging_for_batch(A); run_batch(A); _apply_war_barrier(); iq.push(Ac,rA); _debug_check_commit_gate(False); process_disagg_prefill_inflight_queue(); launch_batch_sample_if_needed(rA,A); _sched_idled=False; last_batch=A; iq.end_step(A); _debug_check_last_launched()
ingest_requests(stop_at_pause=False); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=A,should_retry_storage_prefetch=True); ngram_embedding_manager.prepare_for_forward(B,chunked_req=None); cur_batch_for_debug=B; maybe_prefetch_staging_for_batch(B); run_batch(B); _apply_war_barrier(); iq.push(Bc,rB); _debug_check_commit_gate(True); is_disagg_prefill_batch_result_ready(rA,tensor); _yield_gil_if_needed()
ingest_requests(stop_at_pause=True); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); _debug_check_commit_gate(True); is_disagg_prefill_batch_result_ready(rA,tensor); _yield_gil_if_needed()
ingest_requests(stop_at_pause=True); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); _debug_check_commit_gate(True); is_disagg_prefill_batch_result_ready(rA,tensor); _commit_due_step(); iq.pop_due(); process_batch_result(Ac,rA); process_disagg_prefill_inflight_queue(); launch_batch_sample_if_needed(rB,B); _sched_idled=False; last_batch=B; iq.end_step(B); _debug_check_last_launched()
ingest_requests(stop_at_pause=False); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=B,should_retry_storage_prefetch=True); ngram_embedding_manager.prepare_for_forward(None,chunked_req=None); cur_batch_for_debug=None; _debug_check_commit_gate(True); is_disagg_prefill_batch_result_ready(rB,tensor); _commit_due_step(); iq.pop_due(); process_batch_result(Bc,rB); process_disagg_prefill_inflight_queue(); launch_batch_sample_if_needed(None,None); _sched_idled=True; last_batch=None; iq.end_step(None); _debug_check_last_launched()
ingest_requests(stop_at_pause=False); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); _debug_check_commit_gate(False); on_idle(); process_disagg_prefill_inflight_queue(); launch_batch_sample_if_needed(None,None); _sched_idled=True; last_batch=None; iq.end_step(None); _debug_check_last_launched()
ingest_requests(stop_at_pause=False); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend(['b']); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=None,should_retry_storage_prefetch=True); ngram_embedding_manager.prepare_for_forward(None,chunked_req=None); cur_batch_for_debug=None; _debug_check_commit_gate(False); on_idle(); process_disagg_prefill_inflight_queue(); launch_batch_sample_if_needed(None,None); _sched_idled=True; last_batch=None; iq.end_step(None); _debug_check_last_launched()
ingest_requests(stop_at_pause=False); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=None,should_retry_storage_prefetch=True); ngram_embedding_manager.prepare_for_forward(C,chunked_req=None); cur_batch_for_debug=C; maybe_prefetch_staging_for_batch(C); run_batch(C); _apply_war_barrier(); iq.push(Cc,rC); _debug_check_commit_gate(False); process_disagg_prefill_inflight_queue(); launch_batch_sample_if_needed(rC,C); _sched_idled=False; last_batch=C; iq.end_step(C); _debug_check_last_launched()
ingest_requests(stop_at_pause=False); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=C,should_retry_storage_prefetch=True); ngram_embedding_manager.prepare_for_forward(None,chunked_req=None); cur_batch_for_debug=None; _debug_check_commit_gate(True); is_disagg_prefill_batch_result_ready(rC,tensor); _yield_gil_if_needed()
ingest_requests(stop_at_pause=True); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); _debug_check_commit_gate(True); is_disagg_prefill_batch_result_ready(rC,tensor); _commit_due_step(); iq.pop_due(); process_batch_result(Cc,rC); process_disagg_prefill_inflight_queue(); launch_batch_sample_if_needed(None,None); _sched_idled=True; last_batch=None; iq.end_step(None); _debug_check_last_launched()
ingest_requests(stop_at_pause=False); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=None,should_retry_storage_prefetch=True); ngram_embedding_manager.prepare_for_forward(None,chunked_req=None); cur_batch_for_debug=None; _debug_check_commit_gate(False); on_idle(); process_disagg_prefill_inflight_queue(); launch_batch_sample_if_needed(None,None); _sched_idled=True; last_batch=None; iq.end_step(None); _debug_check_last_launched()
ingest_requests(stop_at_pause=False)
""",
    "pd_prefill_polling_exit": """
_is_continuous_input_polling_enabled(); enable_continuous_input_polling=True; inflight_steps=IQ(1,False,False)
ingest_requests(stop_at_pause=False); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=None,should_retry_storage_prefetch=True); ngram_embedding_manager.prepare_for_forward(A,chunked_req=None); cur_batch_for_debug=A; maybe_prefetch_staging_for_batch(A); run_batch(A); _apply_war_barrier(); iq.push(Ac,rA); _debug_check_commit_gate(False); process_disagg_prefill_inflight_queue(); launch_batch_sample_if_needed(rA,A); _sched_idled=False; last_batch=A; iq.end_step(A); _debug_check_last_launched()
ingest_requests(stop_at_pause=False); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=A,should_retry_storage_prefetch=True); ngram_embedding_manager.prepare_for_forward(B,chunked_req=None); cur_batch_for_debug=B; maybe_prefetch_staging_for_batch(B); run_batch(B); _apply_war_barrier(); iq.push(Bc,rB); _debug_check_commit_gate(True); is_disagg_prefill_batch_result_ready(rA,tensor); _yield_gil_if_needed()
ingest_requests(stop_at_pause=True); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); _debug_check_commit_gate(True); _commit_due_step(); iq.pop_due(); process_batch_result(Ac,rA); process_disagg_prefill_inflight_queue(); launch_batch_sample_if_needed(rB,B); _sched_idled=False; last_batch=B; iq.end_step(B); _debug_check_last_launched()
""",
    "pd_prefill_polling_pause": """
_is_continuous_input_polling_enabled(); enable_continuous_input_polling=True; inflight_steps=IQ(1,False,False)
ingest_requests(stop_at_pause=False); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=None,should_retry_storage_prefetch=True); ngram_embedding_manager.prepare_for_forward(A,chunked_req=None); cur_batch_for_debug=A; maybe_prefetch_staging_for_batch(A); run_batch(A); _apply_war_barrier(); iq.push(Ac,rA); _debug_check_commit_gate(False); process_disagg_prefill_inflight_queue(); launch_batch_sample_if_needed(rA,A); _sched_idled=False; last_batch=A; iq.end_step(A); _debug_check_last_launched()
ingest_requests(stop_at_pause=False); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=A,should_retry_storage_prefetch=True); ngram_embedding_manager.prepare_for_forward(B,chunked_req=None); cur_batch_for_debug=B; maybe_prefetch_staging_for_batch(B); run_batch(B); _apply_war_barrier(); iq.push(Bc,rB); _debug_check_commit_gate(True); is_disagg_prefill_batch_result_ready(rA,tensor); _yield_gil_if_needed()
ingest_requests(stop_at_pause=True); _record_scheduler_state_for_paused_engine(); _yield_gil_if_needed()
ingest_requests(stop_at_pause=True); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); _debug_check_commit_gate(True); is_disagg_prefill_batch_result_ready(rA,tensor); _commit_due_step(); iq.pop_due(); process_batch_result(Ac,rA); process_disagg_prefill_inflight_queue(); launch_batch_sample_if_needed(rB,B); _sched_idled=False; last_batch=B; iq.end_step(B); _debug_check_last_launched()
ingest_requests(stop_at_pause=False); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=B,should_retry_storage_prefetch=True); ngram_embedding_manager.prepare_for_forward(C,chunked_req=None); cur_batch_for_debug=C; maybe_prefetch_staging_for_batch(C); run_batch(C); _apply_war_barrier(); iq.push(Cc,rC); _debug_check_commit_gate(True); is_disagg_prefill_batch_result_ready(rB,tensor); _commit_due_step(); iq.pop_due(); process_batch_result(Bc,rB); process_disagg_prefill_inflight_queue(); launch_batch_sample_if_needed(rC,C); _sched_idled=False; last_batch=C; iq.end_step(C); _debug_check_last_launched()
ingest_requests(stop_at_pause=False); disagg_prefill_bootstrap_queue.pop_bootstrapped(); waiting_queue.extend([]); get_next_disagg_prefill_batch_to_run(decoding_batch=decoding_batch,last_batch=C,should_retry_storage_prefetch=True); ngram_embedding_manager.prepare_for_forward(None,chunked_req=None); cur_batch_for_debug=None; _debug_check_commit_gate(True); is_disagg_prefill_batch_result_ready(rC,tensor); _commit_due_step(); iq.pop_due(); process_batch_result(Cc,rC); process_disagg_prefill_inflight_queue(); launch_batch_sample_if_needed(None,None); _sched_idled=True; last_batch=None; iq.end_step(None); _debug_check_last_launched()
ingest_requests(stop_at_pause=False)
""",
    "pdd_l2": """
init_pp_loop_state(); inflight_steps=IQ(1,True,True); pp_outputs=None; send_req_work=[]
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=None
ingest_requests(); _pp_commit_comm_work([]); _pp_pd_get_retract_ids(0); _pp_commit_comm_work([]); _pp_pd_get_prealloc_ids(); _pp_commit_comm_work([]); _pp_pd_get_decode_transferred_ids(); _pp_commit_comm_work([]); get_next_disagg_decode_batch_to_run(decoding_batch=db0); _pp_debug_check_slot(0); iq.push(A,None); cur_batch_for_debug=A; _pp_recv_proxy_tensors(); _pp_commit_proxy_send_work(); _pp_launch_batch(0,A,proxy,[None,None],lrcq); launch_event=evA; _pp_commit_send_output_work_and_preprocess_output_tensors(0,1); _pp_pd_send_consensus_bootstrapped_ids([['r0'],None],0,None,['r0']); _pp_pd_send_consensus_bootstrapped_ids([None,None],0,None,None); _pp_pd_send_consensus_release_ids([None,None],0,None,None); decode_offload_manager.check_offload_progress(); _pp_commit_comm_work(conswork); _pp_commit_comm_work(conswork); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); _pp_send_pyobj_to_next_stage([],async_send=True); send_req_work=work:[]; _pp_send_pyobj_to_next_stage(['r0'],async_send=True); _pp_send_pyobj_to_next_stage(None,async_send=True); _pp_send_pyobj_to_next_stage(None,async_send=True); _pp_send_proxy_to_next_stage(resA); pp_outputs=out1; iq.end_step(A)
decoding_batch=db1; _pp_slot_last_batch(1); last_batch=None
ingest_requests(); _pp_commit_comm_work(work:[]); _pp_pd_get_retract_ids(1); _pp_commit_comm_work(work:['r0']); _pp_pd_get_prealloc_ids(); _pp_commit_comm_work(work:None); _pp_pd_get_decode_transferred_ids(); _pp_commit_comm_work(work:None); get_next_disagg_decode_batch_to_run(decoding_batch=db1); _pp_debug_check_slot(1); iq.push(B,None); cur_batch_for_debug=B; _pp_recv_proxy_tensors(); _pp_commit_proxy_send_work(); _pp_launch_batch(1,B,proxy,[None,None],lrcq); launch_event=evB; _pp_commit_send_output_work_and_preprocess_output_tensors(1,0); _pp_pd_send_consensus_bootstrapped_ids([['r0'],None],1,None,None); _pp_pd_send_consensus_bootstrapped_ids([None,['p1']],1,None,['p1']); _pp_pd_send_consensus_release_ids([None,None],1,None,None); decode_offload_manager.check_offload_progress(); _pp_recv_pyobj_from_prev_stage(); process_retract_queue(recv); _pp_commit_comm_work(conswork); _pp_commit_comm_work(conswork); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); iq.pop_due(); d2h_event.synchronize(); _pp_process_batch_result(A,nres0); _pp_update_rows_at_commit(0,A); _pp_send_pyobj_to_next_stage([],async_send=True); send_req_work=work:[]; _pp_send_pyobj_to_next_stage(None,async_send=True); _pp_send_pyobj_to_next_stage(['p1'],async_send=True); _pp_send_pyobj_to_next_stage(None,async_send=True); _pp_send_proxy_to_next_stage(resB); pp_outputs=out0; iq.end_step(B)
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=A
ingest_requests(); _pp_commit_comm_work(work:[]); _pp_pd_get_retract_ids(0); _pp_commit_comm_work(work:None); _pp_pd_get_prealloc_ids(); _pp_commit_comm_work(work:['p1']); _pp_pd_get_decode_transferred_ids(); _pp_commit_comm_work(work:None); get_next_disagg_decode_batch_to_run(decoding_batch=db0); _pp_debug_check_slot(0); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_proxy_send_work(); _pp_commit_send_output_work_and_preprocess_output_tensors(0,1); _pp_pd_send_consensus_bootstrapped_ids([None,None],0,pr:recv,None); _pp_pd_send_consensus_bootstrapped_ids([None,['p1']],0,None,None); _pp_pd_send_consensus_release_ids([['t2'],None],0,None,['t2']); decode_offload_manager.check_offload_progress(); _pp_commit_comm_work(conswork); _pp_recv_pyobj_from_prev_stage(); process_prealloc_queue(recv); _pp_commit_comm_work(conswork); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); iq.pop_due(); d2h_event.synchronize(); _pp_process_batch_result(B,nres1); _pp_update_rows_at_commit(1,B); _pp_send_pyobj_to_next_stage([],async_send=True); send_req_work=work:[]; _pp_send_pyobj_to_next_stage(None,async_send=True); _pp_send_pyobj_to_next_stage(None,async_send=True); _pp_send_pyobj_to_next_stage(['t2'],async_send=True); pp_outputs=out1; iq.end_step(None)
decoding_batch=db1; _pp_slot_last_batch(1); last_batch=B
ingest_requests(); _pp_commit_comm_work(work:[]); _pp_pd_get_retract_ids(1); _pp_commit_comm_work(work:None); _pp_pd_get_prealloc_ids(); _pp_commit_comm_work(work:None); _pp_pd_get_decode_transferred_ids(); _pp_commit_comm_work(work:['t2']); get_next_disagg_decode_batch_to_run(decoding_batch=db1); _pp_debug_check_slot(1); iq.push(C,None); cur_batch_for_debug=C; _pp_recv_proxy_tensors(); _pp_commit_proxy_send_work(); _pp_launch_batch(1,C,proxy,[None,None],lrcq); launch_event=evC; _pp_commit_send_output_work_and_preprocess_output_tensors(1,0); _pp_pd_send_consensus_bootstrapped_ids([None,['r3']],1,None,['r3']); _pp_pd_send_consensus_bootstrapped_ids([None,None],1,pp:recv,None); _pp_pd_send_consensus_release_ids([['t2'],None],1,None,None); decode_offload_manager.check_offload_progress(); _pp_commit_comm_work(conswork); _pp_commit_comm_work(conswork); _pp_recv_pyobj_from_prev_stage(); process_decode_transfer_queue(recv); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); iq.pop_due(); _pp_send_pyobj_to_next_stage([],async_send=True); send_req_work=work:[]; _pp_send_pyobj_to_next_stage(['r3'],async_send=True); _pp_send_pyobj_to_next_stage(None,async_send=True); _pp_send_pyobj_to_next_stage(None,async_send=True); _pp_send_proxy_to_next_stage(resC); pp_outputs=out0; iq.end_step(C)
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=None
ingest_requests(); _pp_commit_comm_work(work:[]); _pp_pd_get_retract_ids(0); _pp_commit_comm_work(work:['r3']); _pp_pd_get_prealloc_ids(); _pp_commit_comm_work(work:None); _pp_pd_get_decode_transferred_ids(); _pp_commit_comm_work(work:None); get_next_disagg_decode_batch_to_run(decoding_batch=db0); _pp_debug_check_slot(0); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_proxy_send_work(); _pp_commit_send_output_work_and_preprocess_output_tensors(0,1); _pp_pd_send_consensus_bootstrapped_ids([None,['r3']],0,None,None); _pp_pd_send_consensus_bootstrapped_ids([None,None],0,None,None); _pp_pd_send_consensus_release_ids([None,None],0,pt:recv,None); decode_offload_manager.check_offload_progress(); _pp_recv_pyobj_from_prev_stage(); process_retract_queue(recv); _pp_commit_comm_work(conswork); _pp_commit_comm_work(conswork); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); iq.pop_due(); d2h_event.synchronize(); _pp_process_batch_result(C,nres1); _pp_update_rows_at_commit(1,C); _pp_send_pyobj_to_next_stage([],async_send=True); send_req_work=work:[]; _pp_send_pyobj_to_next_stage(None,async_send=True); _pp_send_pyobj_to_next_stage(None,async_send=True); _pp_send_pyobj_to_next_stage(None,async_send=True); pp_outputs=out1; iq.end_step(None)
decoding_batch=db1; _pp_slot_last_batch(1); last_batch=C
ingest_requests(); _pp_commit_comm_work(work:[]); _pp_pd_get_retract_ids(1); _pp_commit_comm_work(work:None); _pp_pd_get_prealloc_ids(); _pp_commit_comm_work(work:None); _pp_pd_get_decode_transferred_ids(); _pp_commit_comm_work(work:None); get_next_disagg_decode_batch_to_run(decoding_batch=db1); _pp_debug_check_slot(1); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_proxy_send_work(); _pp_commit_send_output_work_and_preprocess_output_tensors(1,0); _pp_pd_send_consensus_bootstrapped_ids([None,None],1,pr:recv,None); _pp_pd_send_consensus_bootstrapped_ids([None,None],1,None,None); _pp_pd_send_consensus_release_ids([None,None],1,None,None); decode_offload_manager.check_offload_progress(); _pp_commit_comm_work(conswork); _pp_commit_comm_work(conswork); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); iq.pop_due(); _pp_send_pyobj_to_next_stage([],async_send=True); send_req_work=work:[]; _pp_send_pyobj_to_next_stage(None,async_send=True); _pp_send_pyobj_to_next_stage(None,async_send=True); _pp_send_pyobj_to_next_stage(None,async_send=True); pp_outputs=out0; iq.end_step(None); _sched_idled=True; on_idle()
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=None
ingest_requests()
""",
    "pdd_l3_last": """
init_pp_loop_state(); inflight_steps=IQ(2,True,True); pp_outputs=None; send_req_work=[]
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=None
ingest_requests(); _pp_pd_get_retract_ids(0); _pp_commit_comm_work([]); _pp_pd_get_prealloc_ids(); _pp_commit_comm_work([]); _pp_pd_get_decode_transferred_ids(); _pp_commit_comm_work([]); get_next_disagg_decode_batch_to_run(decoding_batch=db0); _pp_debug_check_slot(0); iq.push(A,None); cur_batch_for_debug=A; _pp_recv_proxy_tensors(); _pp_commit_send_output_work_and_preprocess_output_tensors(2,1); _pp_commit_proxy_send_work(); _pp_launch_batch(0,A,proxy,[None,None,None],lrcq); launch_event=evA; _pp_pd_send_consensus_bootstrapped_ids([['r0'],None,None],2,None,['r0']); _pp_pd_send_consensus_bootstrapped_ids([None,None,None],2,None,None); _pp_pd_send_consensus_release_ids([None,None,None],2,None,None); _pp_commit_comm_work(conswork); _pp_commit_comm_work(conswork); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); pp_outputs=out1; iq.end_step(A)
decoding_batch=db1; _pp_slot_last_batch(1); last_batch=None
ingest_requests(); _pp_pd_get_retract_ids(1); _pp_commit_comm_work([]); _pp_pd_get_prealloc_ids(); _pp_commit_comm_work([]); _pp_pd_get_decode_transferred_ids(); _pp_commit_comm_work([]); get_next_disagg_decode_batch_to_run(decoding_batch=db1); _pp_debug_check_slot(1); iq.push(B,None); cur_batch_for_debug=B; _pp_recv_proxy_tensors(); _pp_commit_send_output_work_and_preprocess_output_tensors(0,2); _pp_commit_proxy_send_work(); _pp_launch_batch(1,B,proxy,[None,None,None],lrcq); launch_event=evB; _pp_pd_send_consensus_bootstrapped_ids([['r0'],None,None],0,None,None); _pp_pd_send_consensus_bootstrapped_ids([None,['p1'],None],0,None,['p1']); _pp_pd_send_consensus_release_ids([None,None,None],0,None,None); _pp_commit_comm_work(conswork); _pp_commit_comm_work(conswork); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); pp_outputs=out2; iq.end_step(B)
decoding_batch=db2; _pp_slot_last_batch(2); last_batch=None
ingest_requests(); _pp_pd_get_retract_ids(2); _pp_commit_comm_work([]); _pp_pd_get_prealloc_ids(); _pp_commit_comm_work([]); _pp_pd_get_decode_transferred_ids(); _pp_commit_comm_work([]); get_next_disagg_decode_batch_to_run(decoding_batch=db2); _pp_debug_check_slot(2); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_send_output_work_and_preprocess_output_tensors(1,0); _pp_commit_proxy_send_work(); _pp_pd_send_consensus_bootstrapped_ids([['r0'],None,None],1,None,None); _pp_pd_send_consensus_bootstrapped_ids([None,['p1'],None],1,None,None); _pp_pd_send_consensus_release_ids([None,None,['t2']],1,None,['t2']); _pp_recv_pyobj_from_prev_stage(); process_retract_queue(recv); _pp_commit_comm_work(conswork); _pp_commit_comm_work(conswork); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); iq.pop_due(); d2h_event.synchronize(); _pp_process_batch_result(A,nres0); _pp_update_rows_at_commit(0,A); pp_outputs=out0; iq.end_step(None)
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=A
ingest_requests(); _pp_pd_get_retract_ids(0); _pp_commit_comm_work([]); _pp_pd_get_prealloc_ids(); _pp_commit_comm_work([]); _pp_pd_get_decode_transferred_ids(); _pp_commit_comm_work([]); get_next_disagg_decode_batch_to_run(decoding_batch=db0); _pp_debug_check_slot(0); iq.push(C,None); cur_batch_for_debug=C; _pp_recv_proxy_tensors(); _pp_commit_send_output_work_and_preprocess_output_tensors(2,1); _pp_commit_proxy_send_work(); _pp_launch_batch(0,C,proxy,[None,None,None],lrcq); launch_event=evC; _pp_pd_send_consensus_bootstrapped_ids([['r3'],None,None],2,pr:recv,['r3']); _pp_pd_send_consensus_bootstrapped_ids([None,['p1'],None],2,None,None); _pp_pd_send_consensus_release_ids([None,None,['t2']],2,None,None); _pp_commit_comm_work(conswork); _pp_recv_pyobj_from_prev_stage(); process_prealloc_queue(recv); _pp_commit_comm_work(conswork); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); iq.pop_due(); d2h_event.synchronize(); _pp_process_batch_result(B,nres1); _pp_update_rows_at_commit(1,B); pp_outputs=out1; iq.end_step(C)
decoding_batch=db1; _pp_slot_last_batch(1); last_batch=B
ingest_requests(); _pp_pd_get_retract_ids(1); _pp_commit_comm_work([]); _pp_pd_get_prealloc_ids(); _pp_commit_comm_work([]); _pp_pd_get_decode_transferred_ids(); _pp_commit_comm_work([]); get_next_disagg_decode_batch_to_run(decoding_batch=db1); _pp_debug_check_slot(1); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_send_output_work_and_preprocess_output_tensors(0,2); _pp_commit_proxy_send_work(); _pp_pd_send_consensus_bootstrapped_ids([['r3'],None,None],0,None,None); _pp_pd_send_consensus_bootstrapped_ids([None,None,None],0,pp:recv,None); _pp_pd_send_consensus_release_ids([None,None,['t2']],0,None,None); _pp_commit_comm_work(conswork); _pp_commit_comm_work(conswork); _pp_recv_pyobj_from_prev_stage(); process_decode_transfer_queue(recv); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); iq.pop_due(); pp_outputs=out2; iq.end_step(None)
decoding_batch=db2; _pp_slot_last_batch(2); last_batch=None
ingest_requests(); _pp_pd_get_retract_ids(2); _pp_commit_comm_work([]); _pp_pd_get_prealloc_ids(); _pp_commit_comm_work([]); _pp_pd_get_decode_transferred_ids(); _pp_commit_comm_work([]); get_next_disagg_decode_batch_to_run(decoding_batch=db2); _pp_debug_check_slot(2); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_send_output_work_and_preprocess_output_tensors(1,0); _pp_commit_proxy_send_work(); _pp_pd_send_consensus_bootstrapped_ids([['r3'],None,None],1,None,None); _pp_pd_send_consensus_bootstrapped_ids([None,None,None],1,None,None); _pp_pd_send_consensus_release_ids([None,None,None],1,pt:recv,None); _pp_recv_pyobj_from_prev_stage(); process_retract_queue(recv); _pp_commit_comm_work(conswork); _pp_commit_comm_work(conswork); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); iq.pop_due(); d2h_event.synchronize(); _pp_process_batch_result(C,nres0); _pp_update_rows_at_commit(0,C); pp_outputs=out0; iq.end_step(None)
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=C
ingest_requests(); _pp_pd_get_retract_ids(0); _pp_commit_comm_work([]); _pp_pd_get_prealloc_ids(); _pp_commit_comm_work([]); _pp_pd_get_decode_transferred_ids(); _pp_commit_comm_work([]); get_next_disagg_decode_batch_to_run(decoding_batch=db0); _pp_debug_check_slot(0); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_send_output_work_and_preprocess_output_tensors(2,1); _pp_commit_proxy_send_work(); _pp_pd_send_consensus_bootstrapped_ids([None,None,None],2,pr:recv,None); _pp_pd_send_consensus_bootstrapped_ids([None,None,None],2,None,None); _pp_pd_send_consensus_release_ids([None,None,None],2,None,None); _pp_commit_comm_work(conswork); _pp_commit_comm_work(conswork); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); iq.pop_due(); pp_outputs=out1; iq.end_step(None)
decoding_batch=db1; _pp_slot_last_batch(1); last_batch=None
ingest_requests(); _pp_pd_get_retract_ids(1); _pp_commit_comm_work([]); _pp_pd_get_prealloc_ids(); _pp_commit_comm_work([]); _pp_pd_get_decode_transferred_ids(); _pp_commit_comm_work([]); get_next_disagg_decode_batch_to_run(decoding_batch=db1); _pp_debug_check_slot(1); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_send_output_work_and_preprocess_output_tensors(0,2); _pp_commit_proxy_send_work(); _pp_pd_send_consensus_bootstrapped_ids([None,None,None],0,None,None); _pp_pd_send_consensus_bootstrapped_ids([None,None,None],0,None,None); _pp_pd_send_consensus_release_ids([None,None,None],0,None,None); _pp_commit_comm_work(conswork); _pp_commit_comm_work(conswork); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); iq.pop_due(); pp_outputs=out2; iq.end_step(None)
decoding_batch=db2; _pp_slot_last_batch(2); last_batch=None
ingest_requests(); _pp_pd_get_retract_ids(2); _pp_commit_comm_work([]); _pp_pd_get_prealloc_ids(); _pp_commit_comm_work([]); _pp_pd_get_decode_transferred_ids(); _pp_commit_comm_work([]); get_next_disagg_decode_batch_to_run(decoding_batch=db2); _pp_debug_check_slot(2); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_send_output_work_and_preprocess_output_tensors(1,0); _pp_commit_proxy_send_work(); _pp_pd_send_consensus_bootstrapped_ids([None,None,None],1,None,None); _pp_pd_send_consensus_bootstrapped_ids([None,None,None],1,None,None); _pp_pd_send_consensus_release_ids([None,None,None],1,None,None); _pp_commit_comm_work(conswork); _pp_commit_comm_work(conswork); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); iq.pop_due(); pp_outputs=out0; iq.end_step(None); _sched_idled=True; on_idle()
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=None
ingest_requests()
""",
    "pdp_l2": """
init_pp_loop_state(); inflight_steps=IQ(1,True,True); pp_outputs=None; send_req_work=[]
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=None
ingest_requests(); _pp_commit_comm_work([]); process_bootstrapped_queue(); _pp_pd_get_prefill_transferred_ids(); _pp_commit_comm_work([]); process_prefill_chunk(last_batch=None); _process_hicache_events(); get_new_batch_prefill(db0); dp_attn_adapter.maybe_prepare_mlp_sync_batch(A); _pp_debug_check_slot(0); iq.push(A,None); cur_batch_for_debug=A; _pp_recv_proxy_tensors(); _pp_commit_proxy_send_work(); maybe_prefetch_staging_for_batch(A); _pp_launch_batch(0,A,proxy,[None,None],lrcq); launch_event=evA; _pp_commit_send_output_work_and_preprocess_output_tensors(0,1); _pp_pd_send_consensus_release_ids([None,None],0,None,None); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); _pp_send_pyobj_to_next_stage([],async_send=True); send_req_work=work:[]; _pp_send_pyobj_to_next_stage(None,async_send=True); _pp_send_proxy_to_next_stage(resA); pp_outputs=out1; iq.end_step(A)
decoding_batch=db1; _pp_slot_last_batch(1); last_batch=None
ingest_requests(); _pp_commit_comm_work(work:[]); process_bootstrapped_queue(); _pp_pd_get_prefill_transferred_ids(); _pp_commit_comm_work(work:None); process_prefill_chunk(last_batch=None); _process_hicache_events(); get_new_batch_prefill(db1); dp_attn_adapter.maybe_prepare_mlp_sync_batch(B); _pp_debug_check_slot(1); iq.push(B,None); cur_batch_for_debug=B; _pp_recv_proxy_tensors(); _pp_commit_proxy_send_work(); maybe_prefetch_staging_for_batch(B); _pp_launch_batch(1,B,proxy,[None,None],lrcq); launch_event=evB; _pp_commit_send_output_work_and_preprocess_output_tensors(1,0); _pp_pd_send_consensus_release_ids([None,['t1']],1,None,['t1']); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); iq.pop_due(); d2h_event.synchronize(); _pp_process_batch_result(A,nres0); _pp_send_pyobj_to_next_stage([],async_send=True); send_req_work=work:[]; _pp_send_pyobj_to_next_stage(['t1'],async_send=True); _pp_send_proxy_to_next_stage(resB); pp_outputs=out0; iq.end_step(B)
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=A
ingest_requests(); _pp_commit_comm_work(work:[]); process_bootstrapped_queue(); _pp_pd_get_prefill_transferred_ids(); _pp_commit_comm_work(work:['t1']); process_prefill_chunk(last_batch=A); _process_hicache_events(); get_new_batch_prefill(db0); dp_attn_adapter.maybe_prepare_mlp_sync_batch(None); _pp_debug_check_slot(0); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_proxy_send_work(); _pp_commit_send_output_work_and_preprocess_output_tensors(0,1); _pp_pd_send_consensus_release_ids([None,['t1']],0,None,None); _pp_recv_pyobj_from_prev_stage(); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); iq.pop_due(); d2h_event.synchronize(); _pp_process_batch_result(B,nres1); process_disagg_prefill_inflight_queue(recv); _pp_send_pyobj_to_next_stage([],async_send=True); send_req_work=work:[]; _pp_send_pyobj_to_next_stage(None,async_send=True); pp_outputs=out1; iq.end_step(None)
decoding_batch=db1; _pp_slot_last_batch(1); last_batch=B
ingest_requests(); _pp_commit_comm_work(work:[]); process_bootstrapped_queue(); _pp_pd_get_prefill_transferred_ids(); _pp_commit_comm_work(work:None); process_prefill_chunk(last_batch=B); _process_hicache_events(); get_new_batch_prefill(db1); dp_attn_adapter.maybe_prepare_mlp_sync_batch(C); _pp_debug_check_slot(1); iq.push(C,None); cur_batch_for_debug=C; _pp_recv_proxy_tensors(); _pp_commit_proxy_send_work(); maybe_prefetch_staging_for_batch(C); _pp_launch_batch(1,C,proxy,[None,None],lrcq); launch_event=evC; _pp_commit_send_output_work_and_preprocess_output_tensors(1,0); _pp_pd_send_consensus_release_ids([None,['t3']],1,recv,['t3']); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); iq.pop_due(); _pp_send_pyobj_to_next_stage([],async_send=True); send_req_work=work:[]; _pp_send_pyobj_to_next_stage(['t3'],async_send=True); _pp_send_proxy_to_next_stage(resC); pp_outputs=out0; iq.end_step(C)
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=None
ingest_requests(); _pp_commit_comm_work(work:[]); process_bootstrapped_queue(); _pp_pd_get_prefill_transferred_ids(); _pp_commit_comm_work(work:['t3']); process_prefill_chunk(last_batch=None); _process_hicache_events(); get_new_batch_prefill(db0); dp_attn_adapter.maybe_prepare_mlp_sync_batch(None); _pp_debug_check_slot(0); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_proxy_send_work(); _pp_commit_send_output_work_and_preprocess_output_tensors(0,1); _pp_pd_send_consensus_release_ids([None,['t3']],0,None,None); _pp_recv_pyobj_from_prev_stage(); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); iq.pop_due(); d2h_event.synchronize(); _pp_process_batch_result(C,nres1); process_disagg_prefill_inflight_queue(recv); _pp_send_pyobj_to_next_stage([],async_send=True); send_req_work=work:[]; _pp_send_pyobj_to_next_stage(None,async_send=True); pp_outputs=out1; iq.end_step(None)
decoding_batch=db1; _pp_slot_last_batch(1); last_batch=C
ingest_requests(); _pp_commit_comm_work(work:[]); process_bootstrapped_queue(); _pp_pd_get_prefill_transferred_ids(); _pp_commit_comm_work(work:None); process_prefill_chunk(last_batch=C); _process_hicache_events(); get_new_batch_prefill(db1); dp_attn_adapter.maybe_prepare_mlp_sync_batch(None); _pp_debug_check_slot(1); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_proxy_send_work(); _pp_commit_send_output_work_and_preprocess_output_tensors(1,0); _pp_pd_send_consensus_release_ids([None,None],1,recv,None); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); iq.pop_due(); _pp_send_pyobj_to_next_stage([],async_send=True); send_req_work=work:[]; _pp_send_pyobj_to_next_stage(None,async_send=True); pp_outputs=out0; iq.end_step(None); _sched_idled=True
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=None
ingest_requests()
""",
    "pdp_l3_last": """
init_pp_loop_state(); inflight_steps=IQ(2,True,True); pp_outputs=None; send_req_work=[]
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=None
ingest_requests(); process_bootstrapped_queue(); _pp_pd_get_prefill_transferred_ids(); _pp_commit_comm_work([]); process_prefill_chunk(last_batch=None); _process_hicache_events(); get_new_batch_prefill(db0); dp_attn_adapter.maybe_prepare_mlp_sync_batch(A); _pp_debug_check_slot(0); iq.push(A,None); cur_batch_for_debug=A; _pp_recv_proxy_tensors(); _pp_commit_send_output_work_and_preprocess_output_tensors(2,1); _pp_commit_proxy_send_work(); maybe_prefetch_staging_for_batch(A); _pp_launch_batch(0,A,proxy,[None,None,None],lrcq); launch_event=evA; _pp_pd_send_consensus_release_ids([None,None,None],2,None,None); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); pp_outputs=out1; iq.end_step(A)
decoding_batch=db1; _pp_slot_last_batch(1); last_batch=None
ingest_requests(); process_bootstrapped_queue(); _pp_pd_get_prefill_transferred_ids(); _pp_commit_comm_work([]); process_prefill_chunk(last_batch=None); _process_hicache_events(); get_new_batch_prefill(db1); dp_attn_adapter.maybe_prepare_mlp_sync_batch(B); _pp_debug_check_slot(1); iq.push(B,None); cur_batch_for_debug=B; _pp_recv_proxy_tensors(); _pp_commit_send_output_work_and_preprocess_output_tensors(0,2); _pp_commit_proxy_send_work(); maybe_prefetch_staging_for_batch(B); _pp_launch_batch(1,B,proxy,[None,None,None],lrcq); launch_event=evB; _pp_pd_send_consensus_release_ids([None,['t1'],None],0,None,['t1']); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); pp_outputs=out2; iq.end_step(B)
decoding_batch=db2; _pp_slot_last_batch(2); last_batch=None
ingest_requests(); process_bootstrapped_queue(); _pp_pd_get_prefill_transferred_ids(); _pp_commit_comm_work([]); process_prefill_chunk(last_batch=None); _process_hicache_events(); get_new_batch_prefill(db2); dp_attn_adapter.maybe_prepare_mlp_sync_batch(None); _pp_debug_check_slot(2); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_send_output_work_and_preprocess_output_tensors(1,0); _pp_commit_proxy_send_work(); _pp_pd_send_consensus_release_ids([None,['t1'],None],1,None,None); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); iq.pop_due(); d2h_event.synchronize(); _pp_process_batch_result(A,nres0); pp_outputs=out0; iq.end_step(None)
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=A
ingest_requests(); process_bootstrapped_queue(); _pp_pd_get_prefill_transferred_ids(); _pp_commit_comm_work([]); process_prefill_chunk(last_batch=A); _process_hicache_events(); get_new_batch_prefill(db0); dp_attn_adapter.maybe_prepare_mlp_sync_batch(C); _pp_debug_check_slot(0); iq.push(C,None); cur_batch_for_debug=C; _pp_recv_proxy_tensors(); _pp_commit_send_output_work_and_preprocess_output_tensors(2,1); _pp_commit_proxy_send_work(); maybe_prefetch_staging_for_batch(C); _pp_launch_batch(0,C,proxy,[None,None,None],lrcq); launch_event=evC; _pp_pd_send_consensus_release_ids([['t3'],['t1'],None],2,None,['t3']); _pp_recv_pyobj_from_prev_stage(); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); iq.pop_due(); d2h_event.synchronize(); _pp_process_batch_result(B,nres1); process_disagg_prefill_inflight_queue(recv); pp_outputs=out1; iq.end_step(C)
decoding_batch=db1; _pp_slot_last_batch(1); last_batch=B
ingest_requests(); process_bootstrapped_queue(); _pp_pd_get_prefill_transferred_ids(); _pp_commit_comm_work([]); process_prefill_chunk(last_batch=B); _process_hicache_events(); get_new_batch_prefill(db1); dp_attn_adapter.maybe_prepare_mlp_sync_batch(None); _pp_debug_check_slot(1); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_send_output_work_and_preprocess_output_tensors(0,2); _pp_commit_proxy_send_work(); _pp_pd_send_consensus_release_ids([['t3'],None,None],0,recv,None); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); iq.pop_due(); pp_outputs=out2; iq.end_step(None)
decoding_batch=db2; _pp_slot_last_batch(2); last_batch=None
ingest_requests(); process_bootstrapped_queue(); _pp_pd_get_prefill_transferred_ids(); _pp_commit_comm_work([]); process_prefill_chunk(last_batch=None); _process_hicache_events(); get_new_batch_prefill(db2); dp_attn_adapter.maybe_prepare_mlp_sync_batch(None); _pp_debug_check_slot(2); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_send_output_work_and_preprocess_output_tensors(1,0); _pp_commit_proxy_send_work(); _pp_pd_send_consensus_release_ids([['t3'],None,None],1,None,None); _pp_recv_pyobj_from_prev_stage(); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); iq.pop_due(); d2h_event.synchronize(); _pp_process_batch_result(C,nres0); process_disagg_prefill_inflight_queue(recv); pp_outputs=out0; iq.end_step(None)
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=C
ingest_requests(); process_bootstrapped_queue(); _pp_pd_get_prefill_transferred_ids(); _pp_commit_comm_work([]); process_prefill_chunk(last_batch=C); _process_hicache_events(); get_new_batch_prefill(db0); dp_attn_adapter.maybe_prepare_mlp_sync_batch(None); _pp_debug_check_slot(0); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_send_output_work_and_preprocess_output_tensors(2,1); _pp_commit_proxy_send_work(); _pp_pd_send_consensus_release_ids([None,None,None],2,recv,None); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); iq.pop_due(); pp_outputs=out1; iq.end_step(None)
decoding_batch=db1; _pp_slot_last_batch(1); last_batch=None
ingest_requests(); process_bootstrapped_queue(); _pp_pd_get_prefill_transferred_ids(); _pp_commit_comm_work([]); process_prefill_chunk(last_batch=None); _process_hicache_events(); get_new_batch_prefill(db1); dp_attn_adapter.maybe_prepare_mlp_sync_batch(None); _pp_debug_check_slot(1); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_send_output_work_and_preprocess_output_tensors(0,2); _pp_commit_proxy_send_work(); _pp_pd_send_consensus_release_ids([None,None,None],0,None,None); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); iq.pop_due(); pp_outputs=out2; iq.end_step(None)
decoding_batch=db2; _pp_slot_last_batch(2); last_batch=None
ingest_requests(); process_bootstrapped_queue(); _pp_pd_get_prefill_transferred_ids(); _pp_commit_comm_work([]); process_prefill_chunk(last_batch=None); _process_hicache_events(); get_new_batch_prefill(db2); dp_attn_adapter.maybe_prepare_mlp_sync_batch(None); _pp_debug_check_slot(2); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_send_output_work_and_preprocess_output_tensors(1,0); _pp_commit_proxy_send_work(); _pp_pd_send_consensus_release_ids([None,None,None],1,None,None); _pp_commit_comm_work(relwork); _pp_pop_due_batch(); iq.pop_due(); pp_outputs=out0; iq.end_step(None); _sched_idled=True; on_idle()
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=None
ingest_requests()
""",
    "pp_l2": """
init_pp_loop_state(); inflight_steps=IQ(1,True,True); pp_outputs=None; send_req_work=[]; rows_join_before_plan=True
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=None
ingest_requests(); _pp_commit_comm_work([]); _pp_send_pyobj_to_next_stage([],async_send=True); send_req_work=work:[]; get_next_batch_to_run(decoding_batch=db0,last_batch=None); _pp_debug_check_slot(0); iq.push(A,None); cur_batch_for_debug=A; _pp_recv_proxy_tensors(); _pp_commit_proxy_send_work(); _pp_launch_batch(0,A,proxy,[None,None],lrcq); launch_event=evA; _pp_commit_send_output_work_and_preprocess_output_tensors(0,1); _pp_pop_due_batch(); _pp_send_proxy_to_next_stage(resA); pp_outputs=out1; iq.end_step(A)
decoding_batch=db1; _pp_slot_last_batch(1); last_batch=None
ingest_requests(); _pp_commit_comm_work(work:[]); _pp_send_pyobj_to_next_stage([],async_send=True); send_req_work=work:[]; get_next_batch_to_run(decoding_batch=db1,last_batch=None); _pp_debug_check_slot(1); iq.push(D,None); cur_batch_for_debug=D; _pp_recv_proxy_tensors(); _pp_commit_proxy_send_work(); _pp_launch_batch(1,D,proxy,[None,None],lrcq); launch_event=evD; _pp_commit_send_output_work_and_preprocess_output_tensors(1,0); _pp_pop_due_batch(); iq.pop_due(); d2h_event.synchronize(); _pp_process_batch_result(A,nres0); _pp_update_rows_at_commit(0,A); _pp_send_proxy_to_next_stage(resD); pp_outputs=out0; iq.end_step(D)
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=A
ingest_requests(); _pp_commit_comm_work(work:[]); _pp_send_pyobj_to_next_stage([],async_send=True); send_req_work=work:[]; get_next_batch_to_run(decoding_batch=db0,last_batch=A); _pp_debug_check_slot(0); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_proxy_send_work(); _pp_commit_send_output_work_and_preprocess_output_tensors(0,1); _pp_pop_due_batch(); iq.pop_due(); d2h_event.synchronize(); _pp_process_batch_result(D,nres1); _pp_update_rows_at_commit(1,D); pp_outputs=out1; iq.end_step(None)
decoding_batch=db1; _pp_slot_last_batch(1); last_batch=D
ingest_requests(); _pp_commit_comm_work(work:[]); _pp_send_pyobj_to_next_stage([],async_send=True); send_req_work=work:[]; get_next_batch_to_run(decoding_batch=db1,last_batch=D); _pp_debug_check_slot(1); iq.push(E,None); cur_batch_for_debug=E; _pp_recv_proxy_tensors(); _pp_commit_proxy_send_work(); _pp_launch_batch(1,E,proxy,[None,None],lrcq); launch_event=evE; _pp_commit_send_output_work_and_preprocess_output_tensors(1,0); _pp_pop_due_batch(); iq.pop_due(); _pp_send_proxy_to_next_stage(resE); pp_outputs=out0; iq.end_step(E)
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=None
ingest_requests(); _pp_commit_comm_work(work:[]); _pp_send_pyobj_to_next_stage([],async_send=True); send_req_work=work:[]; get_next_batch_to_run(decoding_batch=db0,last_batch=None); _pp_debug_check_slot(0); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_proxy_send_work(); _pp_commit_send_output_work_and_preprocess_output_tensors(0,1); _pp_pop_due_batch(); iq.pop_due(); d2h_event.synchronize(); _pp_process_batch_result(E,nres1); _pp_update_rows_at_commit(1,E); pp_outputs=out1; iq.end_step(None)
decoding_batch=db1; _pp_slot_last_batch(1); last_batch=E
ingest_requests(); _pp_commit_comm_work(work:[]); _pp_send_pyobj_to_next_stage([],async_send=True); send_req_work=work:[]; get_next_batch_to_run(decoding_batch=db1,last_batch=E); _pp_debug_check_slot(1); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_proxy_send_work(); _pp_commit_send_output_work_and_preprocess_output_tensors(1,0); _pp_pop_due_batch(); iq.pop_due(); pp_outputs=out0; iq.end_step(None); _sched_idled=True; on_idle()
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=None
ingest_requests()
""",
    "pp_l2_spec_relay": """
init_pp_loop_state(); inflight_steps=IQ(1,True,True); pp_outputs=None; send_req_work=[]; rows_join_before_plan=True
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=None
ingest_requests(); _pp_commit_comm_work([]); _pp_send_pyobj_to_next_stage([],async_send=True); send_req_work=work:[]; get_next_batch_to_run(decoding_batch=db0,last_batch=None); _pp_debug_check_slot(0); iq.push(A,None); cur_batch_for_debug=A; _pp_recv_proxy_tensors(); _pp_commit_proxy_send_work(); _pp_launch_batch(0,A,proxy,[None,md],lrcq); launch_event=evA; _pp_commit_send_output_work_and_preprocess_output_tensors(0,1); _pp_pop_due_batch(); _pp_send_proxy_to_next_stage(resA); pp_outputs=out1; iq.end_step(A)
decoding_batch=db1; _pp_slot_last_batch(1); last_batch=None
ingest_requests(); _pp_commit_comm_work(work:[]); _pp_send_pyobj_to_next_stage([],async_send=True); send_req_work=work:[]; get_next_batch_to_run(decoding_batch=db1,last_batch=None); _pp_debug_check_slot(1); iq.push(D,None); cur_batch_for_debug=D; _pp_recv_proxy_tensors(); _pp_commit_send_output_work_and_preprocess_output_tensors(1,0); _pp_commit_proxy_send_work(); _pp_launch_batch(1,D,proxy,[None,md],lrcq); launch_event=evD; _pp_pop_due_batch(); iq.pop_due(); d2h_event.synchronize(); _pp_process_batch_result(A,nres0); _pp_update_rows_at_commit(0,A); _pp_send_proxy_to_next_stage(resD); pp_outputs=out0; iq.end_step(D)
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=A
ingest_requests(); _pp_commit_comm_work(work:[]); _pp_send_pyobj_to_next_stage([],async_send=True); send_req_work=work:[]; get_next_batch_to_run(decoding_batch=db0,last_batch=A); _pp_debug_check_slot(0); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_proxy_send_work(); _pp_commit_send_output_work_and_preprocess_output_tensors(0,1); _pp_pop_due_batch(); iq.pop_due(); d2h_event.synchronize(); _pp_process_batch_result(F1,nres1); _pp_update_rows_at_commit(1,D); pp_outputs=out1; iq.end_step(None)
decoding_batch=db1; _pp_slot_last_batch(1); last_batch=D
ingest_requests(); _pp_commit_comm_work(work:[]); _pp_send_pyobj_to_next_stage([],async_send=True); send_req_work=work:[]; get_next_batch_to_run(decoding_batch=db1,last_batch=D); _pp_debug_check_slot(1); iq.push(E,None); cur_batch_for_debug=E; _pp_recv_proxy_tensors(); _pp_commit_send_output_work_and_preprocess_output_tensors(1,0); _pp_commit_proxy_send_work(); _pp_launch_batch(1,E,proxy,[None,md],lrcq); launch_event=evE; _pp_pop_due_batch(); iq.pop_due(); _pp_send_proxy_to_next_stage(resE); pp_outputs=out0; iq.end_step(E)
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=None
ingest_requests(); _pp_commit_comm_work(work:[]); _pp_send_pyobj_to_next_stage([],async_send=True); send_req_work=work:[]; get_next_batch_to_run(decoding_batch=db0,last_batch=None); _pp_debug_check_slot(0); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_proxy_send_work(); _pp_commit_send_output_work_and_preprocess_output_tensors(0,1); _pp_pop_due_batch(); iq.pop_due(); d2h_event.synchronize(); _pp_process_batch_result(F1,nres1); _pp_update_rows_at_commit(1,E); pp_outputs=out1; iq.end_step(None)
decoding_batch=db1; _pp_slot_last_batch(1); last_batch=E
ingest_requests(); _pp_commit_comm_work(work:[]); _pp_send_pyobj_to_next_stage([],async_send=True); send_req_work=work:[]; get_next_batch_to_run(decoding_batch=db1,last_batch=E); _pp_debug_check_slot(1); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_proxy_send_work(); _pp_commit_send_output_work_and_preprocess_output_tensors(1,0); _pp_pop_due_batch(); iq.pop_due(); pp_outputs=out0; iq.end_step(None); _sched_idled=True; on_idle()
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=None
ingest_requests()
""",
    "pp_l3_last": """
init_pp_loop_state(); inflight_steps=IQ(2,True,True); pp_outputs=None; send_req_work=[]; rows_join_before_plan=True
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=None
ingest_requests(); get_next_batch_to_run(decoding_batch=db0,last_batch=None); _pp_debug_check_slot(0); iq.push(A,None); cur_batch_for_debug=A; _pp_recv_proxy_tensors(); _pp_commit_send_output_work_and_preprocess_output_tensors(2,1); _pp_commit_proxy_send_work(); _pp_launch_batch(0,A,proxy,[None,None,None],lrcq); launch_event=evA; _pp_pop_due_batch(); pp_outputs=out1; iq.end_step(A)
decoding_batch=db1; _pp_slot_last_batch(1); last_batch=None
ingest_requests(); get_next_batch_to_run(decoding_batch=db1,last_batch=None); _pp_debug_check_slot(1); iq.push(D,None); cur_batch_for_debug=D; _pp_recv_proxy_tensors(); _pp_commit_send_output_work_and_preprocess_output_tensors(0,2); _pp_commit_proxy_send_work(); _pp_launch_batch(1,D,proxy,[None,None,None],lrcq); launch_event=evD; _pp_pop_due_batch(); pp_outputs=out2; iq.end_step(D)
decoding_batch=db2; _pp_slot_last_batch(2); last_batch=None
ingest_requests(); get_next_batch_to_run(decoding_batch=db2,last_batch=None); _pp_debug_check_slot(2); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_send_output_work_and_preprocess_output_tensors(1,0); _pp_commit_proxy_send_work(); _pp_pop_due_batch(); iq.pop_due(); d2h_event.synchronize(); _pp_process_batch_result(A,nres0); _pp_update_rows_at_commit(0,A); pp_outputs=out0; iq.end_step(None)
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=A
ingest_requests(); get_next_batch_to_run(decoding_batch=db0,last_batch=A); _pp_debug_check_slot(0); iq.push(E,None); cur_batch_for_debug=E; _pp_recv_proxy_tensors(); _pp_commit_send_output_work_and_preprocess_output_tensors(2,1); _pp_commit_proxy_send_work(); _pp_launch_batch(0,E,proxy,[None,None,None],lrcq); launch_event=evE; _pp_pop_due_batch(); iq.pop_due(); d2h_event.synchronize(); _pp_process_batch_result(D,nres1); _pp_update_rows_at_commit(1,D); pp_outputs=out1; iq.end_step(E)
decoding_batch=db1; _pp_slot_last_batch(1); last_batch=D
ingest_requests(); get_next_batch_to_run(decoding_batch=db1,last_batch=D); _pp_debug_check_slot(1); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_send_output_work_and_preprocess_output_tensors(0,2); _pp_commit_proxy_send_work(); _pp_pop_due_batch(); iq.pop_due(); pp_outputs=out2; iq.end_step(None)
decoding_batch=db2; _pp_slot_last_batch(2); last_batch=None
ingest_requests(); get_next_batch_to_run(decoding_batch=db2,last_batch=None); _pp_debug_check_slot(2); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_send_output_work_and_preprocess_output_tensors(1,0); _pp_commit_proxy_send_work(); _pp_pop_due_batch(); iq.pop_due(); d2h_event.synchronize(); _pp_process_batch_result(E,nres0); _pp_update_rows_at_commit(0,E); pp_outputs=out0; iq.end_step(None)
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=E
ingest_requests(); get_next_batch_to_run(decoding_batch=db0,last_batch=E); _pp_debug_check_slot(0); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_send_output_work_and_preprocess_output_tensors(2,1); _pp_commit_proxy_send_work(); _pp_pop_due_batch(); iq.pop_due(); pp_outputs=out1; iq.end_step(None)
decoding_batch=db1; _pp_slot_last_batch(1); last_batch=None
ingest_requests(); get_next_batch_to_run(decoding_batch=db1,last_batch=None); _pp_debug_check_slot(1); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_send_output_work_and_preprocess_output_tensors(0,2); _pp_commit_proxy_send_work(); _pp_pop_due_batch(); iq.pop_due(); pp_outputs=out2; iq.end_step(None)
decoding_batch=db2; _pp_slot_last_batch(2); last_batch=None
ingest_requests(); get_next_batch_to_run(decoding_batch=db2,last_batch=None); _pp_debug_check_slot(2); iq.push(None,None); cur_batch_for_debug=None; _pp_commit_send_output_work_and_preprocess_output_tensors(1,0); _pp_commit_proxy_send_work(); _pp_pop_due_batch(); iq.pop_due(); pp_outputs=out0; iq.end_step(None); _sched_idled=True; on_idle()
decoding_batch=db0; _pp_slot_last_batch(0); last_batch=None
ingest_requests()
""",
}


if __name__ == "__main__":
    unittest.main()
