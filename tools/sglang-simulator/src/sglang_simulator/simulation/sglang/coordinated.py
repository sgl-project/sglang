"""A causal boundary around native scheduler results and external driver events."""

import inspect
import math
import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps

import zmq

from sglang_simulator.compat import validate_coordinator_endpoint
from sglang_simulator.simulation.coordinator import (
    Cancel,
    Coordinator,
    Grant,
    ProtocolError,
    Submit,
    decode_ready,
    encode_message,
)
from sglang_simulator.simulation.manager import Envs, StateManager
from sglang_simulator.simulation.sglang.req_stats_manager import request_stats_manager
from sglang_simulator.simulation.sglang.timing_metadata import attach_timing_output
from sglang_simulator.simulation.sglang.token_events import TokenEventRecorder


def install_prefill_cadence_hooks(target, coordinator_getter) -> None:
    """Observe native cadence decisions without changing their counter or result."""
    next_batch = getattr(target, "get_next_batch_to_run", None)
    should_defer = getattr(target, "_should_defer_prefill", None)
    if not callable(next_batch) or not callable(should_defer):
        return  # A resolved positive interval is rejected by the runtime check.

    @wraps(next_batch)
    def wrapped_next_batch(self, *args, **kwargs):
        bridge = coordinator_getter()
        if bridge is None:
            return next_batch(self, *args, **kwargs)
        bridge._cadence_only_pass = False
        bridge._cadence_deferred = None
        bridge._in_scheduler_pass = True
        try:
            result = next_batch(self, *args, **kwargs)
            bridge._cadence_only_pass = (
                bridge._cadence_deferred
                and getattr(result, "batch_to_run", result) is None
            )
            return result
        finally:
            bridge._in_scheduler_pass = False

    @wraps(should_defer)
    def wrapped_should_defer(self, *args, **kwargs):
        bridge = coordinator_getter()
        if bridge is None:
            return should_defer(self, *args, **kwargs)
        before = getattr(self, "_prefill_decode_interval_remaining", None)
        result = should_defer(self, *args, **kwargs)
        after = getattr(self, "_prefill_decode_interval_remaining", None)
        if (
            not bridge._in_scheduler_pass
            or bridge._cadence_deferred is not None
            or type(result) is not bool
            or type(before) is not int
            or type(after) is not int
            or before < 0
            or before > self.prefill_decode_interval
            or after != (before - 1 if result else 0)
            or (result and before == 0)
            or (not result and before != 0)
        ):
            raise ProtocolError("unsupported native prefill cadence decision")
        bridge._cadence_deferred = result
        return result

    wrapped_next_batch._simulator_cadence_hook = True
    wrapped_should_defer._simulator_cadence_hook = True
    target.get_next_batch_to_run = wrapped_next_batch
    target._should_defer_prefill = wrapped_should_defer


def _validate_prefill_cadence_runtime(scheduler) -> None:
    interval = getattr(scheduler, "prefill_decode_interval", 0)
    if type(interval) is not int or interval < 0:
        raise ValueError("unsupported resolved prefill_decode_interval")
    if interval == 0:
        return
    expected = {
        "get_next_batch_to_run": ("running_batch", "last_batch"),
        "_should_defer_prefill": (),
        "_arm_prefill_decode_interval": ("batch",),
    }
    for name, parameters in expected.items():
        method = getattr(scheduler, name, None)
        if (
            not callable(method)
            or tuple(inspect.signature(method).parameters) != parameters
        ):
            raise ValueError(f"prefill_decode_interval requires native {name} surface")
        if name != "_arm_prefill_decode_interval" and not getattr(
            method, "_simulator_cadence_hook", False
        ):
            raise ValueError(f"prefill_decode_interval requires observed native {name}")
    remaining = getattr(scheduler, "_prefill_decode_interval_remaining", None)
    if type(remaining) is not int or not 0 <= remaining <= interval:
        raise ValueError("prefill_decode_interval requires a valid native counter")


def _watchdog_activation_owner(watchdog):
    """Resolve the native raw callback without activating a disabled watchdog."""
    missing = object()
    active = getattr(watchdog, "is_active", missing)
    if active is not missing:
        owner = watchdog
    else:
        owner = getattr(watchdog, "_raw", missing)
        if owner is missing:
            from sglang.srt.utils.watchdog import _WatchdogNoop

            if type(watchdog) is _WatchdogNoop:
                return None
            raise ValueError("unsupported native watchdog activation interface")
        active = getattr(owner, "is_active", missing)
    if not callable(active):
        raise TypeError("native watchdog activation must be callable")
    return owner, active


class CoordinatedScheduler:
    """Receive payloads while time is stopped; commit results at their due time.

    One instance lives in the scheduler process. A server handles one coordinated
    run, including its warmup; restarting it gives the next run a clean cache.
    """

    def __init__(
        self,
        *,
        scheduler,
        endpoint: str,
        timeout_s: float = 30.0,
        pending_work: Callable[[], bool] | None = None,
    ):
        endpoint = validate_coordinator_endpoint(endpoint)
        # Model defaults can resolve cadence absent from the raw launch arguments.
        _validate_prefill_cadence_runtime(scheduler)
        self.scheduler = scheduler
        self._pending_work = pending_work
        self.clock = Coordinator()
        self.timeout_s = timeout_s
        self._closed = False
        self._token_events = None
        self._watchdogs = []
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        try:
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.setsockopt(zmq.IPV6, 1)
            self._socket.bind(endpoint)
            token_events_path = Envs.token_events_path()
            if token_events_path is not None:
                self._token_events = TokenEventRecorder(token_events_path)
        except Exception:
            self.close()
            raise
        self._reply_pending = False
        self._payloads = {}
        self._seen_payloads = set()
        self._pending = []
        self._aborts = set()
        self._admin = []
        self._raw_receive = None
        self._batch = None
        self._waiting = False
        self._finished = False
        self._cancel_cleanup = set()
        self._zero_time_rounds = 0
        self._rid_length = None
        self._in_scheduler_pass = False
        self._cadence_deferred = False
        self._cadence_only_pass = False
        try:
            for name in ("watchdog", "soft_watchdog"):
                watchdog = getattr(scheduler, name, None)
                if watchdog is not None:
                    activation = _watchdog_activation_owner(watchdog)
                    if activation is not None:
                        owner, active = activation
                        self._watchdogs.append((owner, active))
                        owner.is_active = lambda active=active: (
                            not self._waiting and active()
                        )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Release control resources without resetting native scheduler state."""
        if self._closed:
            return
        self._closed = True
        for watchdog, active in self._watchdogs:
            watchdog.is_active = active
        self._watchdogs.clear()
        if self._token_events is not None:
            self._token_events.close()
        self._socket.close(linger=0)
        self._context.term()

    @contextmanager
    def _guard(self):
        try:
            yield
        except Exception as exc:
            if self._reply_pending:
                self._socket.send(
                    encode_message(
                        Grant(
                            self.clock.run_id or "",
                            self.clock.next_round,
                            self.clock.now_ns,
                            [],
                            status="ERROR",
                            error=str(exc),
                        )
                    )
                )
                self._reply_pending = False
            raise

    def receive(self, receive: Callable[[], list]) -> list:
        self._raw_receive = receive
        with self._guard():
            self._settle()
            if self._batch is not None:
                return []
            result, self._pending = self._pending, []
            result.extend(self._admin)
            self._admin = []
            return result

    def _pump(self) -> None:
        profiles = []
        for req in self._raw_receive():
            kind = type(req).__name__
            if kind == "TokenizedGenerateReqInput":
                if self.clock.is_terminal(req.rid):
                    self._abort_payload(req)
                    continue
                if req.rid in self._seen_payloads:
                    raise ProtocolError(f"duplicate payload rid={req.rid}")
                self._validate_request(req)
                self._seen_payloads.add(req.rid)
                self._payloads[req.rid] = req
            elif kind == "AbortReq":
                if req.abort_all:
                    raise ProtocolError("unregistered abort_all")
                if not self.clock.is_terminal(req.rid):
                    if not self.clock.awaiting_ready:
                        raise ProtocolError(f"HTTP abort without Cancel: {req.rid}")
                    self._aborts.add(req.rid)
            elif kind in ("GetInternalStateReq", "ConfigureLoggingReq", "FreezeGCReq"):
                self._admin.append(req)
            elif kind == "ProfileReq":
                profiles.append(req)
            elif kind == "ShutdownReq" and (
                self._finished or self.clock.run_id is None
            ):
                self._admin.append(req)
            else:
                raise ProtocolError(f"unsupported request during coordination: {kind}")
        if profiles:
            # Check the entire receive pass: an earlier profile request cannot
            # reset accounting ahead of a payload from the same native poll.
            if not (
                (self.clock.run_id is None or self._finished)
                and not self._reply_pending
                and self._batch is None
                and not self._payloads
                and not self._pending
                and not self._aborts
                and not self._cancel_cleanup
                and not self.clock.pending_rids
                and not self.clock.open_rids
                and not self.clock.has_terminals
                and not self._has_native_work()
            ):
                raise ProtocolError(
                    "ProfileReq requires bootstrap or a drained finished run"
                )
            self._admin.extend(profiles)

    @staticmethod
    def _validate_request(req) -> None:
        if getattr(req, "session_params", None) is not None:
            raise ProtocolError("coordinator does not support explicit sessions")
        params = req.sampling_params
        if any(
            getattr(params, name, None)
            for name in ("json_schema", "regex", "ebnf", "structural_tag")
        ):
            raise ProtocolError("coordinator does not support asynchronous grammar")
        if getattr(req, "mm_inputs", None) is not None:
            raise ProtocolError("coordinator supports text requests only")
        if params.max_new_tokens < 1:
            raise ProtocolError("coordinator requires max_new_tokens >= 1")

    def _settle(self) -> None:
        start = time.monotonic()
        self._waiting = True
        try:
            while True:
                self._pump()
                if self._admin and self._batch is None and self.clock.run_id is None:
                    return  # Apply bootstrap administration before accepting Ready.
                if self._finished:
                    if self._payloads:
                        raise ProtocolError("run already finished")
                    self._socket.poll(5)
                    return
                if self.clock.awaiting_ready and self._socket.poll(0):
                    first_ready = self.clock.run_id is None
                    self._accept_ready()
                    if first_ready:
                        # Dataset preparation before the first Ready is unbounded.
                        start = time.monotonic()
                if not self.clock.awaiting_ready:
                    self._match_payloads()
                    if not self.clock.pending_rids:
                        if self.clock.has_terminals:
                            self._grant()
                            start = time.monotonic()
                            continue
                        if (
                            self.clock.driver_finished
                            and not self.clock.open_rids
                            and self._batch is None
                            and not self._has_native_work()
                        ):
                            if self._payloads or self._pending:
                                raise ProtocolError(
                                    "finished driver left unregistered payloads"
                                )
                            self._grant(finished=True)
                        elif self.clock.next_wakeup_ns == self.clock.now_ns:
                            self._zero_time_rounds += 1
                            if self._zero_time_rounds > 10000:
                                raise ProtocolError(
                                    "workload does not settle at current virtual time"
                                )
                            self._grant()
                            continue
                        return
                if (
                    self.clock.run_id is not None
                    and time.monotonic() - start > self.timeout_s
                ):
                    raise ProtocolError(
                        f"run={self.clock.run_id} round={self.clock.next_round}: "
                        f"driver/payload timeout; missing={self.clock.pending_rids}"
                    )
                self._socket.poll(5)
        finally:
            self._waiting = False

    def _accept_ready(self) -> None:
        self._reply_pending = True
        ready = decode_ready(self._socket.recv())
        lengths = {len(op.rid) for op in ready.operations if isinstance(op, Submit)}
        if self._rid_length is not None:
            lengths.add(self._rid_length)
        if len(lengths) > 1:
            raise ProtocolError("request IDs must have a fixed length within a run")
        self.clock.accept_ready(ready)
        if lengths:
            self._rid_length = lengths.pop()
        for op in ready.operations:
            if isinstance(op, Cancel) and self.clock.is_cancelled(op.rid):
                self._cancel(op.rid)
        unexpected = {rid for rid in self._aborts if not self.clock.is_terminal(rid)}
        if unexpected:
            raise ProtocolError(f"HTTP abort without Cancel: {sorted(unexpected)}")
        self._aborts.clear()

    def _match_payloads(self) -> None:
        for rid in self.clock.pending_rids:
            if rid not in self._payloads:
                break
            req = self._payloads.pop(rid)
            self.clock.mark_admitted(rid)
            stats = request_stats_manager.get_req_stats(rid)
            stats.input_length = len(req.input_ids)
            stats.output_length = req.sampling_params.max_new_tokens
            stats.created_time = stats.queue_start = stats.last_event_time = (
                self.clock.now_ns / 1e9
            )
            self._pending.append(req)

    def _has_native_work(self) -> bool:
        # Cancellation is logically terminal before native decode/chunked cleanup.
        return bool(
            (self._pending_work is not None and self._pending_work())
            or getattr(self.scheduler, "result_queue", ())
            or self.scheduler.waiting_queue
            or self.scheduler.chunked_req is not None
            or self.scheduler._pending_chunked_abort_req is not None
            or any(not req.finished() for req in self.scheduler.collect_inflight_reqs())
        )

    def _cancel(self, rid: str) -> None:
        payload = self._payloads.pop(rid, None)
        if payload is not None:
            self._abort_payload(payload)
        kept = []
        for req in self._pending:
            if req.rid == rid:
                self._abort_payload(req)
            else:
                kept.append(req)
        self._pending = kept
        self._cancel_cleanup.add(rid)
        if self._batch is None:
            self._cleanup_cancelled()

    def _abort_payload(self, req) -> None:
        from sglang.srt.managers.io_struct import AbortReq

        self.scheduler.ipc_channels.send_to_tokenizer.send_output(
            AbortReq(rid=req.rid), req
        )

    def _cleanup_cancelled(self) -> None:
        from sglang.srt.managers.io_struct import AbortReq
        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        for rid in self._cancel_cleanup:
            # Native abort uses startswith, so protocol request IDs must not be prefixes.
            self.scheduler.abort_request(AbortReq(rid=rid))
            if self._batch is not None:
                for req in self._batch.reqs:
                    if req.rid == rid:
                        req.to_finish = FINISH_ABORT()
        self._cancel_cleanup.clear()

    def _advance(self, now_ns: int) -> None:
        if now_ns > self.clock.now_ns:
            self._zero_time_rounds = 0
        self.clock.advance(now_ns)
        StateManager.set_global_clock(now_ns / 1e9)

    def _grant(self, *, finished: bool = False) -> None:
        grant = self.clock.grant(finished=finished)
        if finished and self._token_events is not None:
            self._token_events.finish(grant.run_id, grant.now_ns)
        self._socket.send(encode_message(grant))
        self._reply_pending = False
        self._finished = finished

    def process_batch_result(
        self, batch, result, *, process: Callable, duration: float | None = None
    ):
        with self._guard():
            finish_ns = self._prepare_result_time(batch, duration)
            attach_timing_output(result, self.clock, self._token_events)
            requests = tuple(batch.reqs)
            before = {
                id(req): (len(req.output_ids), req.finished()) for req in requests
            }
            ret = process()
            self.scheduler.process_pending_chunked_abort()
            for req in requests:
                stats = request_stats_manager.get_req_stats(req.rid)
                output_count, finished = before[id(req)]
                if (
                    not finished
                    and not self.clock.is_cancelled(req.rid)
                    and len(req.output_ids) > output_count
                ):
                    stats.gen_token_latencies.append(
                        finish_ns / 1e9 - stats.last_event_time
                    )
                    stats.last_event_time = finish_ns / 1e9
                if req.finished() and not self.clock.is_terminal(req.rid):
                    self.clock.terminal(req.rid, "COMPLETED")
            self._finish_result_time()
            return ret

    def _prepare_result_time(self, owner, duration):
        """Advance once before native output; a DP owner supplies paired ACKs."""
        if duration is None:
            duration = StateManager.get_current_inference_dur()
        if not math.isfinite(duration) or duration <= 0 or round(duration * 1e9) == 0:
            raise ProtocolError(f"invalid predicted batch duration: {duration}")
        if self._batch is not None:
            raise ProtocolError("a native result still owns the clock boundary")
        finish_ns = self.clock.now_ns + round(duration * 1e9)
        self._batch = owner
        while (
            self.clock.next_wakeup_ns is not None
            and self.clock.next_wakeup_ns < finish_ns
        ):
            self._advance(self.clock.next_wakeup_ns)
            self._grant()
            self._settle()
        self._advance(finish_ns)
        self._cleanup_cancelled()
        return finish_ns

    def _finish_result_time(self):
        """Release only after native processing and accounting consumption."""
        self._batch = None
        if self.clock.has_terminals or self.clock.next_wakeup_ns == self.clock.now_ns:
            self._grant()

    def on_idle(self) -> None:
        with self._guard():
            cadence_only, self._cadence_only_pass = self._cadence_only_pass, False
            if self._finished or self.clock.awaiting_ready:
                return
            if cadence_only and self._has_native_work():
                # Native countdown passes consume no modeled forward time. Let
                # the next receive/schedule pass run before any driver timer.
                return
            if self.clock.next_wakeup_ns is not None:
                self._advance(self.clock.next_wakeup_ns)
                self._grant()
            elif self.clock.open_rids:
                raise ProtocolError(
                    f"scheduler stalled with requests {self.clock.open_rids}"
                )
            elif self.clock.driver_finished:
                if self._has_native_work():
                    raise ProtocolError("scheduler stalled during native cleanup")
                self._grant(finished=True)
            else:
                raise ProtocolError(
                    "workload stalled: no requests, timer, or finished flag"
                )
