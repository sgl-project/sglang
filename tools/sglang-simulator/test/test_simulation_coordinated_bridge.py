"""Exercise the real coordination bridge with a small scheduler and loopback I/O."""

import queue
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import pytest
import zmq
from sglang_simulator.compat import (
    apply_simulator_server_args,
    validate_simulator_server_args,
)
from sglang_simulator.simulation.coordinator import (
    Cancel,
    ProtocolError,
    Ready,
    Submit,
    decode_grant,
    encode_message,
)
from sglang_simulator.simulation.manager import StateManager
from sglang_simulator.simulation.sglang import coordinated
from sglang_simulator.simulation.sglang.coordinated import CoordinatedScheduler
from sglang_simulator.simulation.sglang.req_stats_manager import request_stats_manager


@pytest.mark.parametrize("coordinated", [False, True])
@pytest.mark.parametrize("as_dict", [False, True])
def test_coordinator_server_args_are_opt_in(monkeypatch, coordinated, as_dict):
    monkeypatch.delenv("SGLANG_SIMULATOR_COORDINATOR", raising=False)
    for name in ("SGLANG_REQ_WAITING_TIMEOUT", "SGLANG_REQ_RUNNING_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(
        "SGLANG_SIMULATOR_OUTPUT_MODE", "OFFLINE" if coordinated else "BLOCKING"
    )
    if coordinated:
        monkeypatch.setenv("SGLANG_SIMULATOR_COORDINATOR", "tcp://127.0.0.1:12345")
    values = {
        "skip_server_warmup": False,
        "enable_cache_report": False,
        "enable_hierarchical_cache": not coordinated,
        "prefill_decode_interval": 0 if coordinated else 1,
    }
    args = values if as_dict else SimpleNamespace(**values)
    apply_simulator_server_args(args)
    if as_dict:
        args = SimpleNamespace(**args)
    assert args.skip_server_warmup is coordinated
    assert args.enable_cache_report is coordinated
    validate_simulator_server_args(args)


@pytest.mark.parametrize(
    "env, overrides, error",
    [
        ({"SGLANG_SIMULATOR_OUTPUT_MODE": "BLOCKING"}, {}, "requires OFFLINE"),
        ({}, {"enable_hierarchical_cache": True}, "enable_hierarchical_cache"),
        ({}, {"prefill_decode_interval": 1}, "prefill_decode_interval"),
        ({}, {"prefill_decode_interval": 22}, "prefill_decode_interval"),
        ({}, {"skip_server_warmup": False}, "requires skip_server_warmup"),
        ({"SGLANG_REQ_WAITING_TIMEOUT": "1"}, {}, "virtual deadline"),
        ({"SGLANG_REQ_RUNNING_TIMEOUT": "1"}, {}, "virtual deadline"),
    ],
)
def test_coordinator_rejects_unsupported_configuration(
    monkeypatch, env, overrides, error
):
    monkeypatch.setenv("SGLANG_SIMULATOR_COORDINATOR", "tcp://127.0.0.1:12345")
    monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_MODE", "OFFLINE")
    for name in ("SGLANG_REQ_WAITING_TIMEOUT", "SGLANG_REQ_RUNNING_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    args = SimpleNamespace()
    apply_simulator_server_args(args)
    for name, value in overrides.items():
        setattr(args, name, value)
    with pytest.raises(ValueError, match=error):
        validate_simulator_server_args(args)


def rid(index):
    return f"{'0' * 32}:{index:016x}"


class TokenizedGenerateReqInput:
    def __init__(self, request_id):
        self.rid = request_id
        self.input_ids = [10, 11, 12]
        self.sampling_params = SimpleNamespace(max_new_tokens=1)


class NativeRequest:
    def __init__(self, request_id):
        self.rid = request_id
        self.output_ids = []
        self.to_finish = None
        self.done = False

    def finished(self):
        return self.done


class AbortReq:
    def __init__(self, *, rid):
        self.rid = rid
        self.abort_all = False


class FreezeGCReq:
    pass


class ShutdownReq:
    pass


@pytest.mark.parametrize("finished", [False, True])
def test_native_shutdown_outside_an_active_run(harness, finished):
    if finished:

        def driver(control):
            control.send(encode_message(Ready("run", 0, [], finished=True)))
            return decode_grant(control.recv())

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(harness.run_driver, driver)
            assert harness.receive() == []
            assert future.result(timeout=3).status == "FINISHED"
    shutdown = ShutdownReq()
    harness.payloads.put(shutdown)
    assert harness.receive() == [shutdown]
    assert harness.bridge.clock.now_ns == 0
    assert harness.bridge._finished is finished


def test_native_shutdown_cannot_silently_end_an_active_run(harness):
    harness.bridge.clock.accept_ready(Ready("run", 0, [], next_wakeup_ns=10))
    harness.payloads.put(ShutdownReq())
    with pytest.raises(ProtocolError, match="unsupported request.*ShutdownReq"):
        harness.receive()
    assert not harness.bridge._finished


class FakeScheduler:
    def __init__(self):
        self.abort_times = []
        self.payload_aborts = []
        self.waiting_queue = []
        self.running_batch = SimpleNamespace(reqs=[])
        self.last_batch = None
        self.chunked_req = None
        self._pending_chunked_abort_req = None
        self.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=self._send_output)
        )

    def _send_output(self, output, request):
        self.payload_aborts.append((output.rid, request.rid))

    def abort_request(self, request):
        self.abort_times.append((request.rid, StateManager.get_global_clock()))
        for req in self.collect_inflight_reqs():
            if req.rid == request.rid and not req.finished():
                req.to_finish = SimpleNamespace(reason="aborted")

    def collect_inflight_reqs(self):
        return {
            req
            for batch in (self.running_batch, self.last_batch)
            if batch is not None
            for req in batch.reqs
        }

    def process_pending_chunked_abort(self):
        pass


class Harness:
    def __init__(self, timeout_s=1.0):
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        self.endpoint = f"tcp://127.0.0.1:{port}"
        self.scheduler = FakeScheduler()
        self.bridge = CoordinatedScheduler(
            scheduler=self.scheduler, endpoint=self.endpoint, timeout_s=timeout_s
        )
        self.payloads = queue.Queue()

    def send_payload(self, request_id):
        self.payloads.put(TokenizedGenerateReqInput(request_id))

    def raw_receive(self):
        result = []
        while True:
            try:
                result.append(self.payloads.get_nowait())
            except queue.Empty:
                return result

    def receive(self):
        return self.bridge.receive(self.raw_receive)

    def process(self, requests, duration_ns, events):
        StateManager.set_current_inference_dur(duration_ns / 1e9)

        def native_process():
            for request in requests:
                request.output_ids.append(99)
                request.done = True
                events.append((request.rid, self.bridge.clock.now_ns))

        self.bridge.process_batch_result(
            SimpleNamespace(reqs=requests),
            SimpleNamespace(auxiliary_host_output=None),
            process=native_process,
        )

    def run_driver(self, run):
        context = zmq.Context()
        control = context.socket(zmq.REQ)
        control.setsockopt(zmq.LINGER, 0)
        control.setsockopt(zmq.RCVTIMEO, 3000)
        control.setsockopt(zmq.SNDTIMEO, 3000)
        control.connect(self.endpoint)
        try:
            return run(control)
        finally:
            control.close(linger=0)
            context.term()

    def close(self):
        self.bridge.close()


@pytest.fixture
def harness(monkeypatch):
    # Cancellation imports these two native types. No SRT code is replaced.
    io_struct = ModuleType("sglang.srt.managers.io_struct")
    io_struct.AbortReq = AbortReq
    schedule_batch = ModuleType("sglang.srt.managers.schedule_batch")
    schedule_batch.FINISH_ABORT = lambda: SimpleNamespace(reason="aborted")
    monkeypatch.setitem(sys.modules, io_struct.__name__, io_struct)
    monkeypatch.setitem(sys.modules, schedule_batch.__name__, schedule_batch)
    StateManager.reset()
    request_stats_manager.reset()
    instance = Harness()
    try:
        yield instance
    finally:
        instance.close()
        StateManager.reset()
        request_stats_manager.reset()


@pytest.mark.parametrize("interval", [1, 22])
def test_resolved_prefill_cadence_is_rejected_before_binding(harness, interval):
    scheduler = FakeScheduler()
    scheduler.prefill_decode_interval = interval
    with pytest.raises(ValueError, match="prefill_decode_interval"):
        CoordinatedScheduler(scheduler=scheduler, endpoint=harness.endpoint)


def test_close_restores_watchdogs_and_releases_endpoint(harness):
    harness.close()
    active = lambda: True
    watchdog = SimpleNamespace(is_active=active)
    scheduler = FakeScheduler()
    scheduler.watchdog = watchdog
    scheduler.soft_watchdog = SimpleNamespace(is_active=active)
    bridge = CoordinatedScheduler(scheduler=scheduler, endpoint=harness.endpoint)
    try:
        assert watchdog.is_active()
        bridge._waiting = True
        assert not watchdog.is_active()
        assert not scheduler.soft_watchdog.is_active()
    finally:
        bridge.close()
        bridge.close()
    assert watchdog.is_active is active
    assert scheduler.soft_watchdog.is_active is active
    assert bridge._socket.closed
    assert bridge._context.closed
    replacement = CoordinatedScheduler(scheduler=scheduler, endpoint=harness.endpoint)
    replacement.close()


def test_bind_failure_releases_context(harness, monkeypatch):
    contexts = []
    context_factory = zmq.Context

    def make_context():
        context = context_factory()
        contexts.append(context)
        return context

    monkeypatch.setattr(coordinated.zmq, "Context", make_context)
    with pytest.raises(zmq.ZMQError, match="Address already in use"):
        CoordinatedScheduler(scheduler=FakeScheduler(), endpoint=harness.endpoint)
    assert len(contexts) == 1
    assert contexts[0].closed


def test_payload_timeout_excludes_pre_run_dataset_preparation(harness, monkeypatch):
    wall_time = [0.0]
    monkeypatch.setattr(
        coordinated, "time", SimpleNamespace(monotonic=lambda: wall_time[0])
    )

    def receive_after_preparation():
        wall_time[0] = 120.0
        if harness.bridge.clock.run_id is not None:
            return [TokenizedGenerateReqInput(rid(1))]
        return []

    def driver(control):
        control.send(encode_message(Ready("run", 0, [Submit(rid(1))])))
        return harness.bridge.receive(receive_after_preparation)

    admitted = harness.run_driver(driver)
    assert [request.rid for request in admitted] == [rid(1)]
    assert harness.bridge.clock.now_ns == 0


@pytest.mark.parametrize("payload_order", [(1, 2), (2, 1)])
@pytest.mark.parametrize("delay_s", [0, 0.01])
def test_batch_timer_and_payload_jitter_preserve_virtual_order(
    harness, payload_order, delay_s
):
    def driver(control):
        control.send(
            encode_message(
                Ready("run", 0, [Submit(rid(1)), Submit(rid(2))], next_wakeup_ns=12)
            )
        )
        for index in payload_order:
            time.sleep(delay_s)
            harness.send_payload(rid(index))
        timer = decode_grant(control.recv())
        assert timer.now_ns == 12
        assert not timer.terminals
        control.send(encode_message(Ready("run", 1, [Submit(rid(3))])))
        time.sleep(delay_s)
        harness.send_payload(rid(3))
        completion = decode_grant(control.recv())
        assert [(event.rid, event.at_ns) for event in completion.terminals] == [
            (rid(1), 20),
            (rid(2), 20),
        ]
        control.send(encode_message(Ready("run", 2, [])))
        child = decode_grant(control.recv())
        assert [(event.rid, event.at_ns) for event in child.terminals] == [(rid(3), 30)]
        control.send(encode_message(Ready("run", 3, [], finished=True)))
        return decode_grant(control.recv())

    events = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(harness.run_driver, driver)
        assert [request.rid for request in harness.receive()] == [rid(1), rid(2)]
        harness.process([NativeRequest(rid(1)), NativeRequest(rid(2))], 20, events)
        assert [request.rid for request in harness.receive()] == [rid(3)]
        harness.process([NativeRequest(rid(3))], 10, events)
        assert harness.receive() == []
        final = future.result(timeout=3)
    assert final.status == "FINISHED"
    assert final.now_ns == 30
    assert events == [(rid(1), 20), (rid(2), 20), (rid(3), 30)]
    assert request_stats_manager.get_req_stats(rid(3)).created_time == 12 / 1e9


@pytest.mark.parametrize("send_admin", [False, True])
def test_missing_payload_cannot_release_input_barrier(harness, send_admin):
    harness.bridge.timeout_s = 0.05

    def driver(control):
        control.send(encode_message(Ready("run", 0, [Submit(rid(1)), Submit(rid(2))])))
        harness.send_payload(rid(1))
        if send_admin:
            harness.payloads.put(FreezeGCReq())
        return decode_grant(control.recv())

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(harness.run_driver, driver)
        with pytest.raises(ProtocolError, match="driver/payload timeout"):
            # Bootstrap administration may arrive before the first Ready.
            while True:
                assert all(isinstance(req, FreezeGCReq) for req in harness.receive())
        error = future.result(timeout=3)
    assert error.status == "ERROR"
    assert rid(2) in error.error
    assert harness.bridge.clock.now_ns == 0
    assert harness.bridge.clock.pending_rids == (rid(2),)


def test_running_cancellation_does_not_shorten_batch(harness):
    def driver(control):
        control.send(
            encode_message(Ready("run", 0, [Submit(rid(1))], next_wakeup_ns=12))
        )
        harness.send_payload(rid(1))
        assert decode_grant(control.recv()).now_ns == 12
        control.send(encode_message(Ready("run", 1, [Cancel(rid(1))])))
        cancelled = decode_grant(control.recv())
        assert len(cancelled.terminals) == 1
        assert cancelled.terminals[0].kind == "CANCELLED"
        assert cancelled.terminals[0].at_ns == 12
        control.send(encode_message(Ready("run", 2, [], finished=True)))
        return decode_grant(control.recv())

    events = []
    request = NativeRequest(rid(1))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(harness.run_driver, driver)
        assert [payload.rid for payload in harness.receive()] == [rid(1)]
        harness.process([request], 20, events)
        assert harness.receive() == []
        final = future.result(timeout=3)
    assert request.to_finish is not None
    assert events == [(rid(1), 20)]
    assert harness.scheduler.abort_times == [(rid(1), 20 / 1e9)]
    assert final.now_ns == 20
    assert final.status == "FINISHED"
    assert not final.terminals


@pytest.mark.parametrize("native_batch", ["running_batch", "last_batch"])
def test_cancel_at_batch_boundary_drains_native_work_before_finished(
    harness, native_batch
):
    def driver(control):
        control.send(encode_message(Ready("run", 0, [Submit(rid(1)), Submit(rid(2))])))
        harness.send_payload(rid(1))
        harness.send_payload(rid(2))
        completion = decode_grant(control.recv())
        assert completion.now_ns == 10
        assert [event.rid for event in completion.terminals] == [rid(1)]
        control.send(encode_message(Ready("run", 1, [Cancel(rid(2))], finished=True)))
        cancelled = decode_grant(control.recv())
        assert [
            (event.rid, event.kind, event.at_ns) for event in cancelled.terminals
        ] == [(rid(2), "CANCELLED", 10)]
        control.send(encode_message(Ready("run", 2, [], finished=True)))
        return decode_grant(control.recv())

    short, long = NativeRequest(rid(1)), NativeRequest(rid(2))
    batch = SimpleNamespace(reqs=[short, long])
    setattr(harness.scheduler, native_batch, batch)

    def first_result():
        short.output_ids.append(99)
        short.done = True
        long.output_ids.append(99)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(harness.run_driver, driver)
        harness.receive()
        StateManager.set_current_inference_dur(10 / 1e9)
        harness.bridge.process_batch_result(
            batch, SimpleNamespace(auxiliary_host_output=None), process=first_result
        )
        assert harness.receive() == []
        assert not harness.bridge._finished
        assert long.to_finish is not None
        harness.process([long], 10, [])
        assert harness.receive() == []
        final = future.result(timeout=3)

    assert final.status == "FINISHED"
    assert final.now_ns == 20
    assert not final.terminals


@pytest.mark.parametrize(
    "native_state", ["waiting_queue", "chunked_req", "_pending_chunked_abort_req"]
)
def test_finished_waits_for_native_idle_cleanup(harness, native_state):
    request = NativeRequest(rid(1))
    is_queue = native_state == "waiting_queue"
    setattr(harness.scheduler, native_state, [request] if is_queue else request)

    def driver(control):
        control.send(encode_message(Ready("run", 0, [], finished=True)))
        return decode_grant(control.recv())

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(harness.run_driver, driver)
        assert harness.receive() == []
        assert not harness.bridge._finished
        setattr(harness.scheduler, native_state, [] if is_queue else None)
        harness.bridge.on_idle()
        final = future.result(timeout=3)

    assert final.status == "FINISHED"
    assert final.now_ns == 0


def test_idle_cannot_finish_with_unreleased_native_request(harness):
    harness.scheduler.running_batch.reqs.append(NativeRequest(rid(1)))

    def driver(control):
        control.send(encode_message(Ready("run", 0, [], finished=True)))
        return decode_grant(control.recv())

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(harness.run_driver, driver)
        assert harness.receive() == []
        with pytest.raises(ProtocolError, match="stalled during native cleanup"):
            harness.bridge.on_idle()
        error = future.result(timeout=3)

    assert error.status == "ERROR"
    assert not harness.bridge._finished


def test_completion_at_timer_is_in_same_grant(harness):
    def driver(control):
        control.send(
            encode_message(Ready("run", 0, [Submit(rid(1))], next_wakeup_ns=20))
        )
        harness.send_payload(rid(1))
        completion = decode_grant(control.recv())
        assert completion.now_ns == 20
        assert [(event.rid, event.at_ns) for event in completion.terminals] == [
            (rid(1), 20)
        ]
        control.send(encode_message(Ready("run", 1, [], finished=True)))
        return decode_grant(control.recv())

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(harness.run_driver, driver)
        harness.receive()
        harness.process([NativeRequest(rid(1))], 20, [])
        harness.receive()
        assert future.result(timeout=3).status == "FINISHED"


@pytest.mark.parametrize("unexpected", ["duplicate_payload", "native_abort"])
def test_unregistered_input_after_admission_fails_current_round(harness, unexpected):
    def driver(control):
        control.send(encode_message(Ready("run", 0, [Submit(rid(1))])))
        harness.send_payload(rid(1))
        return decode_grant(control.recv())

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(harness.run_driver, driver)
        assert [request.rid for request in harness.receive()] == [rid(1)]
        if unexpected == "duplicate_payload":
            harness.send_payload(rid(1))
            expected = "duplicate payload"
        else:
            harness.payloads.put(AbortReq(rid=rid(1)))
            expected = "HTTP abort without Cancel"
        with pytest.raises(ProtocolError, match=expected):
            harness.receive()
        error = future.result(timeout=3)
    assert error.status == "ERROR"
    assert expected in error.error
    assert harness.bridge.clock.now_ns == 0


def test_same_round_cancel_does_not_require_payload_and_late_payload_is_aborted(
    harness,
):
    def driver(control):
        control.send(encode_message(Ready("run", 0, [Submit(rid(1)), Cancel(rid(1))])))
        cancelled = decode_grant(control.recv())
        assert cancelled.terminals[0].kind == "CANCELLED"
        harness.send_payload(rid(1))
        control.send(encode_message(Ready("run", 1, [], finished=True)))
        return decode_grant(control.recv())

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(harness.run_driver, driver)
        assert harness.receive() == []
        final = future.result(timeout=3)
    assert final.status == "FINISHED"
    assert final.now_ns == 0
    assert not final.terminals
    assert harness.scheduler.payload_aborts == [(rid(1), rid(1))]


@pytest.mark.parametrize("duration", [0, -1, 1e-12, float("inf"), float("nan")])
def test_invalid_batch_duration_fails_without_committing_result(harness, duration):
    StateManager.set_current_inference_dur(duration)
    processed = []
    with pytest.raises(ProtocolError, match="invalid predicted batch duration"):
        harness.bridge.process_batch_result(
            SimpleNamespace(reqs=[]), None, process=lambda: processed.append(True)
        )
    assert processed == []
    assert harness.bridge.clock.now_ns == 0
