"""Run the production PD PP loop with bounded rendezvous and delayed fake GPUs.

This tests control/proxy/output ordering, not model execution or transport speed.
Every async send completes only after its peer receives it; GPU payloads also
wait for a simulated stream deadline. A cycle in the reordered loop times out.
The admission protocol itself is tested with the real coordinator separately.
"""

import collections
import contextlib
import logging
import queue
import threading
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.pp_cache_test_support import (
    PrefillControl,
    functions,
    m,
    method,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class Done(Exception):
    pass


class Event:
    def __init__(self, stop, deadline=0):
        self.stop, self.deadline = stop, deadline

    def record(self, stream):
        pass

    def synchronize(self):
        while time.monotonic() < self.deadline:
            if self.stop.wait(min(0.001, self.deadline - time.monotonic())):
                raise Done()


class Batch:
    def __init__(self, reqs=(), batch_is_full=False, step=-1):
        self.reqs, self.batch_is_full, self.step = reqs, batch_is_full, step
        self.forward_mode = NS(is_prebuilt=lambda: False)

    def is_empty(self):
        return not self.reqs


class Admission:
    def __init__(self, scheduler, *, parallel):
        self.s = scheduler
        scheduler.tree_cache.cache_controller = NS(
            layer_done_counter=NS(
                configure_overlap_capacity=lambda n: scheduler.timeline.append(
                    ("capacity", n, None)
                )
            )
        )

    def select_batch(self, running_batch):
        return self.s.get_new_batch_prefill(running_batch), m.AdmissionUpdate(
            layout=(("r", 0, self.s.step),)
        )

    def receive_forward(self, update):
        assert update.layout == (("r", 0, self.s.step),)

    def consume_return(self, update):
        assert isinstance(update, m.AdmissionUpdate)

    def has_pending_work(self):
        return self.s.pending_only


class Network:
    def __init__(self, size):
        self.stop = threading.Event()
        self.cpu = [queue.Queue() for _ in range(size)]
        self.gpu = [queue.Queue() for _ in range(size)]
        self.sent = [[] for _ in range(size)]

    def send(self, channels, rank, payload):
        ack = threading.Event()
        channels[(rank + 1) % len(channels)].put((payload, ack))

        def wait():
            while not ack.wait(0.002):
                if self.stop.is_set():
                    raise Done()

        return [NS(work=NS(wait=wait))]

    def receive(self, channels, rank):
        while not self.stop.is_set():
            try:
                value, ack = channels[rank].get(timeout=0.002)
                if isinstance(value, dict):
                    value["event"].synchronize()
                ack.set()
                return value
            except queue.Empty:
                pass
        raise Done()


def exercise(size, depth, early, sparse, *, pending_only=False):
    net = Network(size)
    current = threading.local()
    namespace = dict(
        DynamicGradMode=lambda: lambda fn: fn,
        get_parallel=lambda: current.parallel,
        pp_cache_admission_enabled=lambda: early,
        pp_spec_stable_rows_enabled=lambda: False,
        envs=NS(
            SGLANG_ENABLE_PP_SPEC=NS(get=lambda: False),
            SGLANG_PP_COMM_OVERLAP=NS(get=lambda: False),
        ),
        nullcontext=contextlib.nullcontext,
        _is_npu=False,
        ScheduleBatch=Batch,
        PPProxyTensors=lambda tensors: NS(tensors=tensors),
        torch=NS(profiler=NS(record_function=lambda *a: contextlib.nullcontext())),
        deque=collections.deque,
        defaultdict=collections.defaultdict,
        is_xpu=lambda: False,
        _pp_can_skip_output_comm=lambda _: False,
        profile_range=lambda *_: contextlib.nullcontext(),
        logger=logging.getLogger(__name__),
        PrefillControl=PrefillControl,
    )
    names = {
        "event_loop_pp_disagg_prefill",
        "init_pp_loop_state",
        "_pp_pd_post_prefill_control",
        "_pp_pd_send_prefill_return",
        "_pp_send_prefill_control",
        "_pp_recv_prefill_control",
        "_pp_commit_comm_work",
        "_pp_record_comm_event",
        "_pp_commit_proxy_send_work",
        "_pp_send_proxy_to_next_stage",
        "_pp_commit_send_output_work_and_preprocess_output_tensors",
        "_pp_send_recv_and_preprocess_output_tensors",
        "_pp_send_output_to_next_stage",
        "_pp_recv_typed_dict",
        "_pp_recv_proxy_tensors",
        "_pp_recv_dict_from_prev_stage",
    }
    functions(
        "managers/scheduler_pp_mixin.py", names, namespace, cls="SchedulerPPMixin"
    )

    class Scheduler:
        def __init__(self, rank):
            self.rank, self.step, self.gpu_deadline = rank, -1, 0
            self.pending_only = pending_only
            self.ps = NS(pp_rank=rank, pp_size=size, pp_async_batch_depth=depth)
            self.attn_tp_group = None
            self.device = "cpu"
            self.pp_group = NS(
                is_first_rank=rank == 0,
                is_last_rank=rank == size - 1,
                recv_tensor_dict=lambda **kwargs: net.receive(net.gpu, rank),
            )
            self.pp_prefetch_admission = None
            self.tree_cache = NS(cache_controller=None)
            self.dp_attn_adapter = NS(maybe_prepare_mlp_sync_batch=lambda batch: batch)
            self.request_receiver = NS(recv_requests=self.recv_requests)
            self.enable_staging = False
            self.disagg_prefill_inflight_queue = []
            self.copy_stream_ctx = contextlib.nullcontext()
            self.copy_stream = NS(wait_stream=lambda stream: None)
            self.schedule_stream = None
            self.device_module = NS(
                current_stream=lambda: NS(wait_event=lambda event: None),
                Event=lambda: Event(net.stop),
            )
            self.results = []
            self.timeline = []

        def ingest_requests(self):
            return self.recv_requests()

        def _process_hicache_events(self):
            pass

        def recv_requests(self):
            self.step += 1
            if self.rank == 0:
                if self.step == 60:
                    net.stop.set()
                    raise Done()
                return ["req", self.step]
            result = net.receive(net.cpu, self.rank)
            assert result == ["req", self.step], (self.rank, self.step, result)
            return result

        def process_input_requests(self, reqs):
            pass

        def _pp_pd_get_prefill_transferred_ids(self, previous):
            expected = ["transfer", self.step]
            assert previous == (expected if self.rank else None)
            return expected

        def process_prefill_chunk(self, **kwargs):
            pass

        def get_new_batch_prefill(self, running_batch, *, admission=None):
            batch = (
                None
                if pending_only or (sparse and self.step % 5 in (1, 2))
                else Batch([self.step], step=self.step)
            )
            return NS(batch_to_run=batch, running_batch=running_batch)

        def _pp_send_pyobj_to_next_stage(self, data, async_send):
            assert async_send
            if len(data) == 1 and isinstance(data[0], PrefillControl):
                frame = data[0]
                assert frame.admission is not None
                normalized = tuple(frame.transferred_rids)
            else:
                normalized = tuple(data)
            self.timeline.append(("send", self.step, normalized[0]))
            net.sent[self.rank].append(normalized)
            return net.send(net.cpu, self.rank, list(data))

        def _pp_recv_pyobj_from_prev_stage(self):
            return net.receive(net.cpu, self.rank)

        def process_bootstrapped_queue(self):
            pass

        def process_disagg_prefill_inflight_queue(self, data):
            assert data[0] == "transfer", (self.rank, data)

        def _pp_launch_batch(self, mb_id, batch, proxy, metadata, last_queue):
            self.timeline.append(("launch", self.step, None))
            if self.rank:
                assert proxy.tensors["step"] == self.step
            self.gpu_deadline = max(time.monotonic(), self.gpu_deadline) + 0.0003 * (
                1 + (self.rank + self.step) % 3
            )
            event = Event(net.stop, self.gpu_deadline)
            tensors = {"step": self.step, "event": event}
            if self.pp_group.is_last_rank:
                last_queue.append((event, NS(tensors=tensors)))
            return (
                NS(
                    pp_hidden_states_proxy_tensors=NS(tensors=tensors),
                    can_run_cuda_graph=False,
                ),
                event,
            )

        def _pp_send_dict_to_next_stage(
            self, tensors, *, async_send, msg_type, ready_event=None
        ):
            assert async_send
            return net.send(net.gpu, self.rank, {**tensors, "__msg_type__": msg_type})

        def _pp_prep_batch_result(self, target, metadata, outputs):
            assert target.step == outputs.tensors["step"], (
                self.rank,
                target.step,
                outputs.tensors["step"],
            )
            return NS(tensors=outputs.tensors, logits_output=None)

        def _pp_process_batch_result(self, batch, output):
            self.results.append(batch.step)

        def on_idle(self):
            self.timeline.append(("housekeeping", self.step, None))

    for name in names:
        setattr(Scheduler, name, namespace[name])
    stages = [Scheduler(rank) for rank in range(size)]
    failures = []

    def run(s):
        current.parallel = s.ps
        try:
            s.event_loop_pp_disagg_prefill()
        except Done:
            pass
        except BaseException as exc:
            failures.append((s.rank, s.step, repr(exc)))
            net.stop.set()

    with patch.dict(
        "sys.modules",
        {
            "sglang.srt.managers.pp_prefetch_scheduler": NS(PPAdmissionGate=Admission),
        },
    ):
        threads = [threading.Thread(target=run, args=(s,), daemon=True) for s in stages]
        for t in threads:
            t.start()
        deadline = time.monotonic() + 8
        while not net.stop.wait(0.01) and time.monotonic() < deadline:
            pass
        reached = stages[0].step
        net.stop.set()
        for t in threads:
            t.join(timeout=1)
    assert not failures, failures
    assert reached == 60, ("loop stalled", [s.step for s in stages])
    if pending_only:
        assert all(not s.results for s in stages)
        assert all(
            sum(kind == "housekeeping" for kind, *_ in s.timeline) > 2 for s in stages
        ), [s.timeline for s in stages]
    else:
        assert all(len(s.results) > 15 for s in stages), [s.results for s in stages]
    for stage in stages:
        assert [n for kind, n, _ in stage.timeline if kind == "capacity"] == (
            [size + depth] if early else []
        )
        launches = {
            step: index
            for index, (kind, step, _) in enumerate(stage.timeline)
            if kind == "launch"
        }
        # Verify the selected execution path really moved sends, while flag0
        # still uses the native post-launch placement (not merely same data).
        checked = 0
        send_kind = "transfer" if stage.pp_group.is_last_rank else "req"
        for index, (kind, step, label) in enumerate(stage.timeline):
            if kind == "send" and label == send_kind and step in launches:
                assert (index < launches[step]) == early
                checked += 1
        assert checked == 0 if pending_only else checked > 10
    return net.sent


class ControlBoundaryTests(CustomTestCase):
    def codec(self, enabled):
        scope = functions(
            "managers/scheduler_pp_mixin.py",
            {"_pp_recv_prefill_control", "_pp_send_prefill_control"},
            dict(PrefillControl=PrefillControl),
            cls="SchedulerPPMixin",
        )
        s = NS(
            pp_prefetch_admission=object() if enabled else None,
            _pp_recv_pyobj_from_prev_stage=Mock(),
            _pp_send_pyobj_to_next_stage=Mock(return_value=["work"]),
        )
        return s, scope

    def test_native_transfer_wire_format_is_unchanged_including_empty_status(self):
        for rids in ([], ["a", "b"]):
            with self.subTest(rids=rids):
                s, code = self.codec(False)
                s._pp_recv_pyobj_from_prev_stage.return_value = rids
                control = code["_pp_recv_prefill_control"](s)
                self.assertEqual(control, PrefillControl(rids))
                self.assertIsNone(control.admission)
                self.assertEqual(code["_pp_send_prefill_control"](s, control), ["work"])
                s._pp_send_pyobj_to_next_stage.assert_called_once_with(
                    rids, async_send=True
                )

    def test_admission_wrapper_exists_only_at_the_transport_boundary(self):
        s, code = self.codec(True)
        update = m.AdmissionUpdate(layout=(("r", 256, 1025),))
        control = PrefillControl(["r"], update)
        s._pp_recv_pyobj_from_prev_stage.return_value = [control]
        self.assertIs(code["_pp_recv_prefill_control"](s), control)
        code["_pp_send_prefill_control"](s, control)
        s._pp_send_pyobj_to_next_stage.assert_called_once_with(
            [control], async_send=True
        )

    def test_bootstrap_processing_has_no_gate_or_envelope_dependency(self):
        process = method(
            "managers/scheduler_pp_mixin.py",
            "SchedulerPPMixin",
            "process_bootstrapped_queue",
        )
        ready = NS(rid="ready")
        pop = Mock(return_value=[ready])
        s = NS(
            waiting_queue=[], disagg_prefill_bootstrap_queue=NS(pop_bootstrapped=pop)
        )
        process(s)
        self.assertEqual(s.waiting_queue, [ready])
        pop.assert_called_once_with()

    def test_transfer_fold_preserves_the_native_rid_intersection(self):
        fold = method(
            "managers/scheduler_pp_mixin.py",
            "SchedulerPPMixin",
            "_pp_pd_get_prefill_transferred_ids",
            {"KVPoll": NS(Success=1, Failed=2)},
        )
        for previous, expected in (
            (None, ["common", "local"]),
            ([], []),
            (["common", "upstream"], ["common"]),
        ):
            with self.subTest(previous=previous):
                s = NS(
                    disagg_prefill_inflight_queue=[],
                    get_rids=Mock(return_value=["common", "local"]),
                )
                self.assertCountEqual(fold(s, previous), expected)
                s.get_rids.assert_called_once_with(
                    s.disagg_prefill_inflight_queue, True, [1, 2]
                )

    def test_prefill_return_keeps_the_native_startup_slot_gate(self):
        send = method(
            "managers/scheduler_pp_mixin.py",
            "SchedulerPPMixin",
            "_pp_pd_send_prefill_return",
        )
        control = PrefillControl([])
        for last, slot, returned, should_send in (
            (True, None, None, False),
            (True, None, control, False),
            (True, control, None, True),
            (False, control, None, False),
            (False, None, control, True),
        ):
            with self.subTest(last=last, slot=slot, returned=returned):
                s = NS(
                    pp_group=NS(is_last_rank=last),
                    _pp_send_prefill_control=Mock(return_value=["work"]),
                )
                work, value = send(s, [slot], 0, returned, control)
                self.assertEqual(work, ["work"] if should_send else [])
                self.assertIs(value, control if should_send else returned)
                self.assertEqual(
                    s._pp_send_prefill_control.call_count, int(should_send)
                )


class ControlOverlapTests(CustomTestCase):
    def test_pending_only_gate_still_runs_native_housekeeping(self):
        for pp in (2, 4):
            for depth in (0, 1):
                with self.subTest(pp=pp, depth=depth):
                    exercise(pp, depth, True, False, pending_only=True)

    def test_production_loop_keeps_wire_order_and_makes_progress(self):
        for size in (2, 4, 8, 16):
            for depth in (0, 1):
                for sparse in (False, True):
                    with self.subTest(pp=size, depth=depth, sparse=sparse):
                        old = exercise(size, depth, False, sparse)
                        early = exercise(size, depth, True, sparse)
                        for a, b in zip(old, early):
                            # Threads stop when PP0 reaches the bound; peers may
                            # have a different short tail, but shared prefixes
                            # of their untagged CPU channels must be identical.
                            length = min(len(a), len(b))
                            self.assertEqual(a[:length], b[:length])


class LoadEventWindowTests(CustomTestCase):
    def counter(self):
        class ManualEvent:
            done = True

            def record(self):
                self.done = False

            def query(self):
                return self.done

        namespace = dict(device_module=NS(Event=ManualEvent))
        path = "managers/cache_controller.py"
        names = {"__init__", "complete", "finish_event"}
        functions(path, names, namespace, cls="LayerLoadingEvent")
        namespace["LayerLoadingEvent"] = type(
            "LayerLoadingEvent", (), {name: namespace[name] for name in names}
        )
        names = {
            "__init__",
            "configure_overlap_capacity",
            "update_producer",
            "set_consumer",
        }
        functions(path, names, namespace, cls="LayerDoneCounter")
        cls = type("LayerDoneCounter", (), {name: namespace[name] for name in names})
        return cls(2)

    def test_capacity_is_startup_only_and_preserves_registered_objects(self):
        counter = self.counter()
        registered_pool = NS(counter=counter)
        original = list(counter.events)
        self.assertEqual(counter.num_counters, 3)
        counter.configure_overlap_capacity(2)
        self.assertEqual(counter.events, original)
        counter.configure_overlap_capacity(8)
        self.assertIs(registered_pool.counter, counter)
        self.assertEqual(counter.events[:3], original)
        self.assertEqual(counter.num_counters, 8)
        counter.configure_overlap_capacity(4)
        self.assertEqual(counter.num_counters, 8)
        with self.assertRaises(ValueError):
            counter.configure_overlap_capacity(0)
        counter.update_producer()
        with self.assertRaises(RuntimeError):
            counter.configure_overlap_capacity(16)
        self.assertEqual(counter.num_counters, 8)

    def test_consumer_alone_also_forbids_resize(self):
        counter = self.counter()
        counter.set_consumer(0)
        with self.assertRaises(RuntimeError):
            counter.configure_overlap_capacity(4)

    def test_window_size_does_not_disable_busy_slot_guard(self):
        for capacity in (3, 4, 5, 8, 16):
            with self.subTest(capacity=capacity):
                counter = self.counter()
                counter.configure_overlap_capacity(capacity)
                for index in range(capacity):
                    self.assertEqual(counter.update_producer(), index)
                    counter.events[index].complete(1)
                with self.assertRaisesRegex(AssertionError, "Producer finish event"):
                    counter.update_producer()


if __name__ == "__main__":
    unittest.main()
