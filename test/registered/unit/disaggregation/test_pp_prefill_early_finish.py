"""PP PD prefill early finish (SGLANG_OPT_PP_PREFILL_EARLY_FINISH).

PP rank 0 streams a request's finish once every stage's KV sender reports
Success, from a completion thread, while the KV release still waits for the
ring consensus. Covers the switch and its checks, the emit / ring-release
contract, the thread against real stores, and the send lock.
"""

import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation import pp_early_finish
from sglang.srt.disaggregation import prefill as prefill_module
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.mode import FAKE_BOOTSTRAP_HOST
from sglang.srt.disaggregation.pp_consensus_store import PPConsensusStore
from sglang.srt.disaggregation.pp_early_finish import (
    EarlyFinishThread,
    PPPrefillEarlyFinish,
    start_pp_prefill_early_finish,
)
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.managers import scheduler_pp_mixin as pp_module
from sglang.srt.managers.req_types import FINISH_ABORT, FINISH_LENGTH
from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
    move,
)
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import FakeReq
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

S = KVPoll.Success
MASTER = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"
CHILD = "SGLANG_OPT_PP_PREFILL_EARLY_FINISH"
_HANG_S = 60  # a hang guard only: no case waits this long when it passes


class _PPGroup:
    def __init__(self):
        self._obj = None

    def broadcast_object(self, obj, src=0):
        if obj is not None:
            self._obj = obj
        return self._obj


class _FakeStore:
    """PP rank 0's view of a PPConsensusStore over ``pp`` stages."""

    def __init__(self, pp):
        self.pp = pp
        self.version = 0
        self.values = {}
        self.collects = 0

    def set(self, rank, room, status):
        self.values.setdefault(rank, {})[room] = status
        self.version += 1

    def collect(self, room):
        self.collects += 1
        return [self.values.get(rank, {}).get(room) for rank in range(self.pp)]


class _TimeStats:
    def __init__(self):
        self.calls = []

    def set_prefill_kv_transfer_finish_time(self):
        self.calls.append("kv_finish")

    def set_completion_time(self):
        self.calls.append("completion")

    def compute_and_observe_kv_transfer_metrics(self, metric):
        return None


class _Req(FakeReq):
    def __init__(self, rid, room):
        super().__init__(rid=rid)
        self.pending_bootstrap = False
        self.finished_output = None
        self.finished_reason = None
        self.bootstrap_host = "127.0.0.1"
        self.bootstrap_room = room
        self.return_routed_experts = False
        self.return_logprob = False
        self.metadata_buffer_index = -1
        self.time_stats = _TimeStats()
        self.local = KVPoll.Transferring
        self.disagg_kv_sender = SimpleNamespace(
            poll=lambda: self.local,
            clear=lambda: None,
            get_transfer_metric=lambda: None,
        )

    def finished(self):
        return self.finished_reason is not None

    def finish_outside_step(self, reason):
        self.finished_reason = reason


class _Streamer:
    """The streamer's rule for finished requests (a finished request goes out
    once), with an optional GIL yield between the check and the mark."""

    def __init__(self, yield_s=0.0):
        self.yield_s = yield_s
        self.sent = []
        self.send_to_detokenizer = SenderWrapper(None)
        self.rust_server = None

    def stream_output(self, reqs, return_logprob, skip_req=None):
        for req in reqs:
            if req.finished() and req.finished_output:
                continue
            if self.yield_s:
                time.sleep(self.yield_s)
            if req.finished():
                req.finished_output = True
                self.sent.append((req.rid, time.monotonic_ns(), threading.get_ident()))


class _KVManager:
    def __init__(self, store):
        self.request_status = store
        self.staging = {}

    def staging_holds_success(self, room):
        return self.staging.get(room, 0) > 0


def _local_polls(senders, cp_group, tp_group):
    return [sender.poll() for sender in senders]


def _scheduler(store, streamer=None):
    """The fields the ring release reads, with a tracker over ``store``."""
    sched = SchedulerDisaggregationPrefillMixin()
    sched.scheduler_stage_metrics = None
    sched.disagg_prefill_transfer_queue = IndexedReqList(ListKind.PD_SENDING)
    sched.attn_cp_cpu_group = sched.attn_tp_cpu_group = None
    sched.output_streamer = streamer or _Streamer()
    sched.req_to_metadata_buffer_idx_allocator = None
    sched.kv_mgr = _KVManager(store)
    sched.released = []
    sched.failed = []
    sched.retired = IndexedReqList(ListKind.PD_RELEASING)

    def release(req, checkpoint):
        sched.released.append((req.rid, checkpoint))
        move([req], sched.retired)

    def handle_failure(req):
        sched.failed.append(req.rid)
        req.finish_outside_step(FINISH_ABORT("transfer failed"))
        release(req, checkpoint=False)

    sched.request_retirer = SimpleNamespace(release=release)
    sched.handle_inflight_transfer_failure = handle_failure
    sched._release_mm_inputs = lambda req: None
    sched.pp_prefill_early_finish = PPPrefillEarlyFinish(
        kv_manager=sched.kv_mgr,
        transfer_queue=sched.disagg_prefill_transfer_queue,
        output_streamer=sched.output_streamer,
    )
    return sched


def _patch_polls(test):
    patcher = patch.object(
        prefill_module, "poll_and_all_reduce_attn_cp_tp_group", _local_polls
    )
    patcher.start()
    test.addCleanup(patcher.stop)


class TestEarlyFinishSwitch(CustomTestCase):
    def setUp(self):
        self.stores = [PPConsensusStore(2, rank, _PPGroup()) for rank in range(1)]
        self.addCleanup(lambda: [store.close() for store in self.stores])
        self.parallel = SimpleNamespace(pp_rank=0, attn_tp_size=1, attn_cp_size=1)
        self.disagg = SimpleNamespace(disaggregation_transfer_backend="mooncake")
        for name, value in (
            ("get_parallel", self.parallel),
            ("get_disagg", self.disagg),
        ):
            patcher = patch.object(pp_early_finish, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _start(self, *, kv_manager=None, streamer=None):
        streamer = streamer or _Streamer()
        tracker = start_pp_prefill_early_finish(
            kv_manager=kv_manager or _KVManager(self.stores[0]),
            transfer_queue=IndexedReqList(ListKind.PD_SENDING),
            output_streamer=streamer,
        )
        return tracker, streamer

    def test_engages_only_where_rank_zero_sees_every_sender(self):
        for field, value in (("pp_rank", 1), ("attn_tp_size", 2), ("attn_cp_size", 2)):
            with self.subTest(field=field), patch.object(self.parallel, field, value):
                tracker, _ = self._start()
                self.assertIsNone(tracker)
        with (
            self.subTest("not Mooncake"),
            patch.object(self.disagg, "disaggregation_transfer_backend", "nixl"),
        ):
            self.assertIsNone(self._start()[0])
        with self.subTest("sender status not shared across PP stages"):
            self.assertIsNone(self._start(kv_manager=_KVManager({}))[0])
        rust = _Streamer()
        rust.rust_server = object()
        with self.subTest("output sink not a zmq socket"):
            self.assertIsNone(self._start(streamer=rust)[0])
            self.assertIsNone(rust.send_to_detokenizer._lock)
        started = []
        real_start = EarlyFinishThread.start

        def start(thread):
            started.append(thread)
            self.addCleanup(thread.stop)
            return real_start(thread)

        with patch.object(EarlyFinishThread, "start", start):
            tracker, _ = self._start()
        self.assertIsNotNone(tracker)
        self.assertEqual([thread.tracker for thread in started], [tracker])


class _StopLoop(Exception):
    pass


class TestLoopEntry(CustomTestCase):
    def test_the_loop_starts_it_only_when_the_child_resolves_on(self):
        def stop():
            raise _StopLoop()

        for env, on in (
            ({}, False),
            ({MASTER: "1"}, True),
            ({MASTER: "1", CHILD: "0"}, False),
            ({MASTER: "0", CHILD: "1"}, True),
        ):
            with self.subTest(env=env):
                sched = SimpleNamespace(
                    init_pp_loop_state=lambda: None,
                    pp_loop_size=1,
                    increment_scheduler_iterations=stop,
                    disagg_prefill_bootstrap_queue=SimpleNamespace(kv_manager="kv"),
                    disagg_prefill_transfer_queue="queue",
                    output_streamer="streamer",
                )
                environ = {
                    **{k: v for k, v in os.environ.items() if k not in (MASTER, CHILD)},
                    **env,
                }
                with (
                    patch.dict(os.environ, environ, clear=True),
                    patch.object(
                        pp_module, "start_pp_prefill_early_finish", return_value="t"
                    ) as start,
                    self.assertRaises(_StopLoop),
                ):
                    SchedulerPPMixin.event_loop_pp_disagg_prefill(sched)
                self.assertEqual(start.called, on)
                self.assertEqual(sched.pp_prefill_early_finish, "t" if on else None)
                if on:
                    start.assert_called_once_with(
                        kv_manager="kv",
                        transfer_queue="queue",
                        output_streamer="streamer",
                    )


class TestEmitContract(CustomTestCase):
    PP = 4

    def setUp(self):
        _patch_polls(self)
        self.store = _FakeStore(self.PP)
        self.sched = _scheduler(self.store)
        self.tracker = self.sched.pp_prefill_early_finish

    def _add(self, rid, room):
        req = _Req(rid, room)
        move([req], self.sched.disagg_prefill_transfer_queue)
        return req

    def _all_success(self, req, ranks=None):
        for rank in range(self.PP) if ranks is None else ranks:
            self.store.set(rank, req.bootstrap_room, S)
        req.local = S

    def _sent(self):
        return [rid for rid, _, _ in self.sched.output_streamer.sent]

    def test_streams_only_once_every_stage_succeeded(self):
        req = self._add("a", 11)
        self._all_success(req, ranks=(0, 1, 3))
        self.store.set(2, 11, KVPoll.Transferring)
        self.assertEqual(self.tracker.emit(), 0)
        self.store.set(2, 11, S)
        self.assertEqual(self.tracker.emit(), 1)
        self.assertEqual(self._sent(), ["a"])
        self.assertIsInstance(req.finished_reason, FINISH_LENGTH)
        self.assertEqual(req.time_stats.calls, ["kv_finish", "completion"])
        # Still queued: the ring release frees the KV.
        self.assertIn(req, self.sched.disagg_prefill_transfer_queue)
        self.assertEqual(self.sched.released, [])
        # An unchanged store is not scanned again.
        collects = self.store.collects
        self.assertEqual(self.tracker.emit(), 0)
        self.assertEqual(self.store.collects, collects)

    def test_ring_release_frees_the_kv_without_a_second_output(self):
        early = self._add("a", 11)
        late = self._add("b", 12)
        self._all_success(early)
        self.tracker.emit()
        late.local = S  # concluded by the ring only
        done = self.sched.process_disagg_prefill_transfer_queue(
            {"a": False, "b": False}
        )
        self.assertEqual([req.rid for req in done], ["a", "b"])
        self.assertEqual(self.sched.released, [("a", True), ("b", True)])
        self.assertEqual(self._sent(), ["a", "b"])
        self.assertEqual(early.time_stats.calls, ["kv_finish", "completion"])
        self.assertEqual(late.time_stats.calls, ["kv_finish", "completion"])
        self.assertEqual(len(self.sched.disagg_prefill_transfer_queue), 0)
        self.assertEqual(self.tracker._emitted, set())
        self.assertEqual(self.tracker._ring_owned, set())

    def test_failure_at_the_ring_after_an_early_finish(self):
        req = self._add("a", 11)
        self._all_success(req)
        self.tracker.emit()
        self.sched.process_disagg_prefill_transfer_queue({"a": True})
        self.assertEqual(self.sched.failed, ["a"])
        self.assertEqual(self._sent(), ["a"])
        self.assertEqual(req.time_stats.calls, ["kv_finish", "completion"])

    def test_requests_left_to_the_ring(self):
        cases = {
            "parked": lambda r: setattr(r, "pending_bootstrap", True),
            "aborted": lambda r: r.finish_outside_step(FINISH_ABORT("x")),
            "warmup": lambda r: setattr(r, "bootstrap_host", FAKE_BOOTSTRAP_HOST),
            "experts": lambda r: setattr(r, "return_routed_experts", True),
            "staging": lambda r: self.sched.kv_mgr.staging.update(
                {r.bootstrap_room: 1}
            ),
        }
        for i, (rid, make) in enumerate(cases.items()):
            req = self._add(rid, 100 + i)
            make(req)
            self._all_success(req)
        self.assertEqual(self.tracker.emit(), 0)
        # A staging hold is retried by room, with no store change.
        self.sched.kv_mgr.staging.clear()
        rooms = self.tracker.take_recheck_rooms()
        self.assertEqual(rooms, {104})
        self.assertEqual(self.tracker.emit(rooms), 1)
        self.assertEqual(self._sent(), ["staging"])

    def test_ring_release_during_an_emit_streams_once(self):
        """An emit holds the tracker lock from its scan to its stream, so a ring
        release of the same request waits for it and then skips the output."""
        req = self._add("a", 11)
        self._all_success(req)
        entered, resume = threading.Event(), threading.Event()
        self.addCleanup(resume.set)
        inner = self.sched.output_streamer.stream_output

        def blocking_stream(reqs, return_logprob, skip_req=None):
            entered.set()
            resume.wait(_HANG_S)
            inner(reqs, return_logprob, skip_req)

        self.sched.output_streamer.stream_output = blocking_stream
        emitter = threading.Thread(target=self.tracker.emit, args=({11},))
        emitter.start()
        self.assertTrue(entered.wait(_HANG_S))
        ring = threading.Thread(
            target=self.sched.process_disagg_prefill_transfer_queue,
            args=({"a": False},),
        )
        ring.start()
        ring.join(0.1)
        self.assertTrue(ring.is_alive())
        resume.set()
        emitter.join(_HANG_S)
        ring.join(_HANG_S)
        self.assertEqual(self._sent(), ["a"])
        self.assertEqual(req.time_stats.calls, ["kv_finish", "completion"])
        self.assertEqual(self.sched.released, [("a", True)])

    def test_a_request_the_ring_claimed_is_never_streamed_early(self):
        req = self._add("a", 11)
        req.local = S
        self.assertFalse(self.tracker.released(req))
        self._all_success(req)
        self.assertEqual(self.tracker.emit({11}), 0)
        self.assertEqual(self.tracker.emit(), 0)
        self.tracker.forget(["a"])
        self.assertEqual(self.tracker._ring_owned, set())


class TestCompletionThread(CustomTestCase):
    PP = 2

    def setUp(self):
        _patch_polls(self)
        group = _PPGroup()
        self.stores = [PPConsensusStore(self.PP, r, group) for r in range(self.PP)]
        self.addCleanup(lambda: [store.close() for store in self.stores])

    def _start(self, sched):
        thread = EarlyFinishThread(sched.pp_prefill_early_finish).start()
        self.addCleanup(thread.stop)
        return thread

    def _add(self, sched, rid, room):
        req = _Req(rid, room)
        move([req], sched.disagg_prefill_transfer_queue)
        return req

    def _signal_sends(self, sched, count):
        """An event set once ``count`` finishes went out."""
        streamer, done = sched.output_streamer, threading.Event()
        stream_output = streamer.stream_output

        def stream_and_signal(reqs, return_logprob, skip_req=None):
            stream_output(reqs, return_logprob, skip_req)
            if len(streamer.sent) >= count:
                done.set()

        streamer.stream_output = stream_and_signal
        return done

    def test_finish_goes_out_while_the_main_loop_is_busy(self):
        """The main thread blocks and never reaches a check point, so every
        finish came from the thread."""
        sched = _scheduler(self.stores[0])
        n = 15
        done = self._signal_sends(sched, n)
        self._start(sched)
        reqs = [self._add(sched, f"r{i}", 2000 + i) for i in range(n)]
        for req in reqs:
            for store in self.stores:
                store[req.bootstrap_room] = KVPoll.Transferring

        def stage(rank):
            for req in reqs:
                time.sleep(0.002)
                self.stores[rank][req.bootstrap_room] = S

        threads = [threading.Thread(target=stage, args=(r,)) for r in range(self.PP)]
        for t in threads:
            t.start()
        self.assertTrue(done.wait(_HANG_S))
        for t in threads:
            t.join()
        sent = sched.output_streamer.sent
        self.assertEqual(sorted(rid for rid, _, _ in sent), sorted(r.rid for r in reqs))
        main = threading.get_ident()
        self.assertTrue(all(tid != main for _, _, tid in sent))

    def test_staging_hold_is_retried_without_a_store_change(self):
        sched = _scheduler(self.stores[0])
        tracker = sched.pp_prefill_early_finish
        held = threading.Event()
        polls = []
        take = tracker.take_recheck_rooms

        def counting_take():
            rooms = take()
            if 77 in rooms:
                polls.append(rooms)
                if len(polls) >= 2:
                    held.set()
            return rooms

        tracker.take_recheck_rooms = counting_take
        sent = self._signal_sends(sched, 1)
        self._start(sched)
        self._add(sched, "s", 77)
        sched.kv_mgr.staging[77] = 1
        for store in self.stores:
            store[77] = S
        # Two recheck polls of the held room send nothing.
        self.assertTrue(held.wait(_HANG_S))
        self.assertEqual(sched.output_streamer.sent, [])
        sched.kv_mgr.staging[77] = 0
        self.assertTrue(sent.wait(_HANG_S))
        self.assertEqual([rid for rid, _, _ in sched.output_streamer.sent], ["s"])

    def test_failing_emits_stop_the_thread_and_leave_the_ring(self):
        sched = _scheduler(self.stores[0])
        tracker = sched.pp_prefill_early_finish
        with patch.object(tracker, "emit", side_effect=RuntimeError("boom")):
            thread = self._start(sched)
            req = self._add(sched, "d", 88)
            for store in self.stores:
                store[88] = S
            deadline = time.monotonic() + _HANG_S
            while thread.alive() and time.monotonic() < deadline:
                time.sleep(0.005)
        self.assertFalse(thread.alive())
        req.local = S
        sched.process_disagg_prefill_transfer_queue({"d": False})
        self.assertEqual([rid for rid, _, _ in sched.output_streamer.sent], ["d"])


class TestStoreListener(CustomTestCase):
    def test_rank0_hears_local_and_replicated_puts(self):
        group = _PPGroup()
        stores = [PPConsensusStore(2, r, group) for r in range(2)]
        self.addCleanup(lambda: [store.close() for store in stores])
        heard = []
        stores[0].add_listener(lambda k, v: heard.append((k, v)))
        version = stores[0].version
        stores[0][5] = S
        stores[1][5] = S
        stores[1][6] = KVPoll.Transferring
        deadline = time.monotonic() + _HANG_S
        while len(heard) < 3 and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertEqual(
            sorted(heard), sorted([(5, S), (5, S), (6, KVPoll.Transferring)])
        )
        self.assertEqual(stores[0].version - version, 3)
        # A raising listener never breaks replication.
        stores[0].add_listener(lambda k, v: 1 / 0)
        stores[1][7] = S
        deadline = time.monotonic() + _HANG_S
        while stores[0].collect(7)[1] != S and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertEqual(stores[0].collect(7), [None, S])


class TestSenderWrapperLock(CustomTestCase):
    def test_sends_are_serialized_once_enabled(self):
        inside, overlap = [], []

        def fake_send(sock, obj, flags=0):
            inside.append(1)
            if len(inside) > 1:
                overlap.append(obj)
            time.sleep(0.0005)
            inside.pop()

        wrapper = SenderWrapper(socket=object())
        wrapper.enable_thread_safety()
        with patch(
            "sglang.srt.managers.scheduler_components.output_sender.sock_send",
            fake_send,
        ):
            threads = [
                threading.Thread(
                    target=lambda: [
                        wrapper.send_output(SimpleNamespace()) for _ in range(50)
                    ]
                )
                for _ in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(overlap, [])


if __name__ == "__main__":
    unittest.main()
