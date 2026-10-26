"""Exercise the real coordination bridge with a small scheduler and loopback I/O."""

import importlib.util
import json
import queue
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

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
        ({}, {"prefill_decode_interval": -1}, "prefill_decode_interval"),
        ({}, {"prefill_decode_interval": True}, "prefill_decode_interval"),
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


@pytest.mark.parametrize("profile", [False, True])
@pytest.mark.parametrize("algorithm", ["DSPARK", "EAGLE"])
def test_coordinator_speculation_requires_the_native_dspark_source_path(
    monkeypatch, tmp_path, profile, algorithm
):
    config = tmp_path / "simulator.json"
    predictor = {"name": "infercast_dspark"}
    if profile:
        predictor["native_segment_profile_id"] = "b" * 64
    config.write_text(json.dumps({"predictor": predictor}), encoding="utf-8")
    monkeypatch.setenv("SGLANG_SIMULATOR_CONFIG_PATH", str(config))
    monkeypatch.setenv("SGLANG_SIMULATOR_COORDINATOR", "tcp://127.0.0.1:12345")
    monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_MODE", "OFFLINE")
    args = SimpleNamespace(speculative_algorithm=algorithm)
    apply_simulator_server_args(args)
    if profile and algorithm == "DSPARK":
        validate_simulator_server_args(args)
        assert args.disable_overlap_schedule is False
    else:
        with pytest.raises(ValueError, match="source-bound native DSPark"):
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


class ProfileReq:
    pass


@pytest.mark.parametrize("finished", [False, True])
def test_native_profile_only_at_drained_run_boundaries(harness, finished):
    if finished:

        def driver(control):
            control.send(encode_message(Ready("run", 0, [], finished=True)))
            return decode_grant(control.recv())

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(harness.run_driver, driver)
            assert harness.receive() == []
            assert future.result(timeout=3).status == "FINISHED"
    profile = ProfileReq()
    harness.payloads.put(profile)
    assert harness.receive() == [profile]
    assert harness.bridge.clock.now_ns == 0
    assert harness.bridge._finished is finished


def test_native_profile_cannot_reset_an_active_run(harness):
    harness.bridge.clock.accept_ready(Ready("run", 0, [], next_wakeup_ns=10))
    harness.payloads.put(ProfileReq())
    with pytest.raises(ProtocolError, match="ProfileReq requires"):
        harness.receive()
    assert harness.bridge._admin == []
    assert harness.bridge.clock.now_ns == 0


def test_bootstrap_profile_returns_before_an_already_queued_ready(harness):
    profile = ProfileReq()
    harness.payloads.put(profile)

    def driver(control):
        control.send(encode_message(Ready("run", 0, [], finished=True)))
        return decode_grant(control.recv())

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(harness.run_driver, driver)
        assert harness.bridge._socket.poll(1000)
        assert harness.receive() == [profile]
        assert harness.bridge.clock.run_id is None
        assert not harness.bridge._finished
        assert not future.done()
        # The native caller now applies profile administration before its next
        # receive pass accepts the driver's already queued first manifest.
        assert harness.receive() == []
        assert future.result(timeout=3).status == "FINISHED"


@pytest.mark.parametrize("finished", [False, True])
@pytest.mark.parametrize(
    "pending", ["batch", "result", "reference", "native", "payload", "same_poll"]
)
def test_native_profile_cannot_skip_pending_work(harness, finished, pending):
    # Also reject inconsistent post-finish state rather than hiding a lost tail.
    if finished:
        harness.bridge._finished = True
        harness.bridge.clock.run_id = "run"
    if pending == "batch":
        harness.bridge._batch = object()
    elif pending == "result":
        harness.scheduler.result_queue = [object()]
    elif pending == "reference":
        harness.bridge._pending_work = lambda: True
    elif pending == "native":
        harness.scheduler.running_batch.reqs.append(NativeRequest(rid(1)))
    elif pending == "payload":
        harness.bridge._payloads[rid(1)] = TokenizedGenerateReqInput(rid(1))
    harness.payloads.put(ProfileReq())
    if pending == "same_poll":
        harness.send_payload(rid(1))
    with pytest.raises(ProtocolError, match="ProfileReq requires"):
        harness.receive()
    assert harness.bridge._admin == []
    assert harness.bridge.clock.now_ns == 0


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


def test_native_callback_filter_keeps_original_request_completion_owner(harness):
    """Counterfactual mutation control; this does not diagnose a historical run."""
    request_id = rid(1)
    harness.send_payload(request_id)

    def driver(control):
        control.send(encode_message(Ready("run", 0, [Submit(request_id)])))
        return decode_grant(control.recv())

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(harness.run_driver, driver)
        assert len(harness.receive()) == 1
        req = NativeRequest(request_id)
        batch = SimpleNamespace(reqs=[req])

        def process():
            req.output_ids.append(99)
            req.done = True
            batch.reqs.clear()

        harness.bridge.process_batch_result(
            batch,
            SimpleNamespace(auxiliary_host_output=None),
            process=process,
            duration=0.01,
        )
        grant = future.result(timeout=3)
    assert [(row.rid, row.kind) for row in grant.terminals] == [
        (request_id, "COMPLETED")
    ]
    assert request_stats_manager.get_req_stats(request_id).gen_token_latencies == [0.01]


@pytest.mark.parametrize("fail", [False, True])
def test_opt_in_token_ledger_only_finishes_after_native_cleanup(
    monkeypatch, tmp_path, fail
):
    path = tmp_path / "tokens.jsonl"
    monkeypatch.setenv("SGLANG_SIMULATOR_TOKEN_EVENTS", str(path))
    instance = Harness()
    instance.scheduler.running_batch.reqs.append(NativeRequest(rid(1)))

    def driver(control):
        control.send(encode_message(Ready("run", 0, [], finished=True)))
        return decode_grant(control.recv())

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(instance.run_driver, driver)
            assert instance.receive() == []
            assert "FINISHED" not in path.read_text()
            if fail:
                with pytest.raises(
                    ProtocolError, match="stalled during native cleanup"
                ):
                    instance.bridge.on_idle()
            else:
                instance.scheduler.running_batch.reqs.clear()
                instance.bridge.on_idle()
            grant = future.result(timeout=3)
        assert grant.status == ("ERROR" if fail else "FINISHED")
    finally:
        instance.close()
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert (events[-1]["event"] == "finish") is not fail


def test_token_ledger_disabled_does_not_open_a_file(monkeypatch):
    monkeypatch.delenv("SGLANG_SIMULATOR_TOKEN_EVENTS", raising=False)

    def unexpected_file(*args, **kwargs):
        raise AssertionError("disabled ledger must not create a writer")

    monkeypatch.setattr(coordinated, "TokenEventRecorder", unexpected_file)
    instance = Harness()
    instance.close()


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


@pytest.mark.parametrize("interval", [0, 1, 192])
def test_positive_cadence_arguments_wait_for_runtime_surface_validation(
    monkeypatch, interval
):
    monkeypatch.setenv("SGLANG_SIMULATOR_COORDINATOR", "tcp://127.0.0.1:12345")
    monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_MODE", "OFFLINE")
    args = SimpleNamespace(prefill_decode_interval=interval)
    apply_simulator_server_args(args)
    validate_simulator_server_args(args)


def cadence_scheduler(harness, interval):
    class NativeCadence(FakeScheduler):
        def __init__(self):
            super().__init__()
            self.prefill_decode_interval = interval
            self._prefill_decode_interval_remaining = 0
            self.next_batch = None
            self.next_error = None

        def _should_defer_prefill(self):
            if self._prefill_decode_interval_remaining == 0:
                return False
            self._prefill_decode_interval_remaining -= 1
            return True

        def _arm_prefill_decode_interval(self, batch):
            if self.prefill_decode_interval and batch is not None:
                self._prefill_decode_interval_remaining = self.prefill_decode_interval

        def get_next_batch_to_run(self, running_batch, last_batch):
            if self.next_error is not None:
                raise self.next_error
            batch = None if self._should_defer_prefill() else self.next_batch
            return SimpleNamespace(batch_to_run=batch, running_batch=running_batch)

    coordinated.install_prefill_cadence_hooks(NativeCadence, lambda: harness.bridge)
    harness.close()
    harness.scheduler = NativeCadence()
    harness.bridge = CoordinatedScheduler(
        scheduler=harness.scheduler, endpoint=harness.endpoint
    )
    return harness.scheduler


def test_cadence_hooks_without_a_coordinator_only_delegate():
    calls = []

    class Native:
        @property
        def _prefill_decode_interval_remaining(self):
            pytest.fail("non-coordinated hooks must not inspect the native counter")

        def _should_defer_prefill(self):
            calls.append("defer")
            return True

        def get_next_batch_to_run(self, running_batch, last_batch):
            calls.append("next")
            return self._should_defer_prefill()

    coordinated.install_prefill_cadence_hooks(Native, lambda: None)
    assert Native().get_next_batch_to_run(None, None) is True
    assert calls == ["next", "defer"]


@pytest.mark.parametrize("interval", [1, 2, 192])
def test_cadence_only_passes_keep_pending_work_at_the_same_virtual_time(
    harness, monkeypatch, interval
):
    scheduler = cadence_scheduler(harness, interval)
    harness.bridge.clock.accept_ready(
        Ready("run", 0, [Submit(rid(1))], next_wakeup_ns=100)
    )
    harness.send_payload(rid(1))
    assert [req.rid for req in harness.receive()] == [rid(1)]
    scheduler.waiting_queue.append(NativeRequest(rid(1)))
    scheduler._arm_prefill_decode_interval(object())
    monkeypatch.setattr(
        harness.bridge, "_grant", lambda **kwargs: pytest.fail("cadence must not Grant")
    )
    monkeypatch.setattr(
        harness.bridge, "_advance", lambda now: pytest.fail("cadence must not advance")
    )
    iteration = StateManager.get_iteration()
    for remaining in reversed(range(interval)):
        plan = scheduler.get_next_batch_to_run(scheduler.running_batch, None)
        assert plan.batch_to_run is None
        assert scheduler._prefill_decode_interval_remaining == remaining
        harness.bridge.on_idle()
        assert harness.bridge.clock.now_ns == 0
        assert StateManager.get_iteration() == iteration
        assert harness.receive() == []
    scheduler.next_batch = object()
    plan = scheduler.get_next_batch_to_run(scheduler.running_batch, None)
    assert plan.batch_to_run is scheduler.next_batch
    assert not harness.bridge._cadence_only_pass


def test_last_cadence_pass_does_not_mask_a_subsequent_native_stall(harness):
    scheduler = cadence_scheduler(harness, 1)
    harness.bridge.clock.accept_ready(Ready("run", 0, [Submit(rid(1))]))
    scheduler.waiting_queue.append(NativeRequest(rid(1)))
    scheduler._arm_prefill_decode_interval(object())
    scheduler.get_next_batch_to_run(scheduler.running_batch, None)
    harness.bridge.on_idle()  # The final 1->0 cadence pass is still not idle.
    scheduler.get_next_batch_to_run(scheduler.running_batch, None)
    with pytest.raises(ProtocolError, match="scheduler stalled with requests"):
        harness.bridge.on_idle()


def test_cadence_without_native_work_keeps_normal_driver_timer(harness, monkeypatch):
    scheduler = cadence_scheduler(harness, 1)
    harness.bridge.clock.accept_ready(Ready("run", 0, [], next_wakeup_ns=10))
    scheduler._arm_prefill_decode_interval(object())
    scheduler.get_next_batch_to_run(scheduler.running_batch, None)
    events = []
    monkeypatch.setattr(
        harness.bridge, "_advance", lambda now: events.append(("advance", now))
    )
    monkeypatch.setattr(harness.bridge, "_grant", lambda: events.append(("grant",)))
    harness.bridge.on_idle()
    assert events == [("advance", 10), ("grant",)]


def test_schedule_failure_preserves_original_error_and_clears_cadence_marker(harness):
    scheduler = cadence_scheduler(harness, 1)
    scheduler._arm_prefill_decode_interval(object())
    scheduler.get_next_batch_to_run(scheduler.running_batch, None)
    error = RuntimeError("native scheduling failed")
    scheduler.next_error = error
    with pytest.raises(RuntimeError) as caught:
        scheduler.get_next_batch_to_run(scheduler.running_batch, None)
    assert caught.value is error
    assert not harness.bridge._cadence_only_pass
    assert not harness.bridge._in_scheduler_pass


@pytest.mark.parametrize("counter", [True, -1, 193])
def test_positive_cadence_rejects_unrepresented_resolved_counter(harness, counter):
    scheduler = cadence_scheduler(harness, 192)
    harness.close()
    scheduler._prefill_decode_interval_remaining = counter
    with pytest.raises(ValueError, match="valid native counter"):
        CoordinatedScheduler(scheduler=scheduler, endpoint=harness.endpoint)


@pytest.mark.parametrize(
    "method",
    ["_arm_prefill_decode_interval", "_should_defer_prefill", "get_next_batch_to_run"],
)
def test_positive_cadence_rejects_an_unknown_native_signature(harness, method):
    scheduler = cadence_scheduler(harness, 192)
    harness.close()
    setattr(scheduler, method, lambda *args: None)
    with pytest.raises(ValueError, match="prefill_decode_interval requires native"):
        CoordinatedScheduler(scheduler=scheduler, endpoint=harness.endpoint)


@pytest.mark.parametrize("counter", [True, -1, 193])
def test_native_cadence_counter_drift_fails_closed_during_the_pass(harness, counter):
    scheduler = cadence_scheduler(harness, 192)
    scheduler._prefill_decode_interval_remaining = counter
    with pytest.raises(ProtocolError, match="unsupported native prefill cadence"):
        scheduler.get_next_batch_to_run(scheduler.running_batch, None)
    assert not harness.bridge._cadence_only_pass
    assert not harness.bridge._in_scheduler_pass


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


@pytest.fixture
def native_watchdog_module(monkeypatch):
    """Load real watchdogs without parent diagnostics or background monitoring."""
    name = "sglang.srt.utils.watchdog"
    diagnostics = ModuleType("sglang.srt.utils.cudacore_pyspy_dump_utils")
    diagnostics.pyspy_dump_schedulers = Mock()
    monkeypatch.setitem(sys.modules, diagnostics.__name__, diagnostics)
    process_inspection = ModuleType("psutil")
    process_inspection.Process = Mock()
    monkeypatch.setitem(sys.modules, process_inspection.__name__, process_inspection)
    path = Path(__file__).resolve().parents[3] / "python/sglang/srt/utils/watchdog.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "threading", SimpleNamespace(Thread=Mock()))
    return module


def test_native_noop_and_wrapped_watchdog_activation_owner(
    harness, native_watchdog_module
):
    harness.close()
    native = native_watchdog_module
    noop = native.Watchdog.create("disabled", None)
    wrapped = native.Watchdog.create("soft", 60, soft=True)
    original = wrapped._raw.is_active
    scheduler = FakeScheduler()
    scheduler.watchdog, scheduler.soft_watchdog = noop, wrapped
    bridge = CoordinatedScheduler(scheduler=scheduler, endpoint=harness.endpoint)
    try:
        assert not hasattr(noop, "is_active")
        assert not hasattr(wrapped, "is_active")
        assert wrapped._raw.is_active()
        wrapped.feed()
        assert wrapped._counter == 1
        with wrapped.disable():
            assert not wrapped._raw.is_active()
        bridge._waiting = True
        assert not wrapped._raw.is_active()
        assert wrapped._active is True
        bridge._waiting = False
        assert wrapped._raw.is_active()
    finally:
        bridge.close()
    assert wrapped._raw.is_active is original
    assert not hasattr(noop, "is_active")
    assert not hasattr(wrapped, "is_active")


def test_native_raw_watchdog_wait_and_close(harness, native_watchdog_module):
    harness.close()
    active = lambda: True
    raw = native_watchdog_module.WatchdogRaw("raw", lambda: 0, active, 60)
    scheduler = FakeScheduler()
    scheduler.watchdog = raw
    bridge = CoordinatedScheduler(scheduler=scheduler, endpoint=harness.endpoint)
    try:
        assert raw.is_active()
        bridge._waiting = True
        assert not raw.is_active()
    finally:
        bridge.close()
    assert raw.is_active is active


@pytest.mark.parametrize(
    "unsupported",
    [
        SimpleNamespace(),
        SimpleNamespace(is_active=False),
        SimpleNamespace(_raw=None),
        SimpleNamespace(_raw=SimpleNamespace(is_active=None)),
        SimpleNamespace(is_active=False, _raw=SimpleNamespace(is_active=lambda: True)),
    ],
)
def test_invalid_watchdog_restores_prior_owner_and_control_resources(
    harness, native_watchdog_module, monkeypatch, unsupported
):
    harness.close()
    active = lambda: True
    scheduler = FakeScheduler()
    scheduler.watchdog = native_watchdog_module.WatchdogRaw(
        "raw", lambda: 0, active, 60
    )
    scheduler.soft_watchdog = unsupported
    contexts = []
    factory = coordinated.zmq.Context

    def context():
        value = factory()
        contexts.append(value)
        return value

    monkeypatch.setattr(coordinated.zmq, "Context", context)
    with pytest.raises((TypeError, ValueError), match="watchdog activation"):
        CoordinatedScheduler(scheduler=scheduler, endpoint=harness.endpoint)
    assert scheduler.watchdog.is_active is active
    assert len(contexts) == 1 and contexts[0].closed
    scheduler.soft_watchdog = None
    replacement = CoordinatedScheduler(scheduler=scheduler, endpoint=harness.endpoint)
    replacement.close()


def test_native_watchdog_callback_error_is_preserved(harness, native_watchdog_module):
    harness.close()
    error = RuntimeError("native activation failed")

    def active():
        raise error

    scheduler = FakeScheduler()
    scheduler.watchdog = native_watchdog_module.WatchdogRaw(
        "raw", lambda: 0, active, 60
    )
    bridge = CoordinatedScheduler(scheduler=scheduler, endpoint=harness.endpoint)
    try:
        with pytest.raises(RuntimeError) as caught:
            scheduler.watchdog.is_active()
        assert caught.value is error
        bridge._waiting = True
        assert not scheduler.watchdog.is_active()
    finally:
        bridge.close()
    assert scheduler.watchdog.is_active is active


def test_bind_failure_releases_context(harness, monkeypatch):
    contexts = []
    context_factory = zmq.Context

    def make_context():
        context = context_factory()
        contexts.append(context)
        return context

    monkeypatch.setattr(coordinated.zmq, "Context", make_context)
    with pytest.raises(zmq.ZMQError) as error:
        CoordinatedScheduler(scheduler=FakeScheduler(), endpoint=harness.endpoint)
    assert error.value.errno == zmq.EADDRINUSE
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


@pytest.mark.parametrize("interval", [0, 192])
def test_running_cancellation_does_not_shorten_batch(harness, interval):
    if interval:
        cadence_scheduler(harness, interval)

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
        if interval:
            harness.scheduler.waiting_queue.append(request)
            harness.scheduler._arm_prefill_decode_interval(object())
            for _ in range(interval):
                plan = harness.scheduler.get_next_batch_to_run(
                    harness.scheduler.running_batch, None
                )
                assert plan.batch_to_run is None
                harness.bridge.on_idle()
                assert harness.bridge.clock.now_ns == 0
                assert not future.done()
                assert harness.receive() == []
            harness.scheduler.waiting_queue.clear()
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
