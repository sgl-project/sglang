"""A causal boundary around the existing, non-overlapping SGLang scheduler."""

import math
import os
import time
from collections.abc import Callable
from contextlib import contextmanager

import zmq
from sglang_simulator.simulation.client import validate_endpoint
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


def validate_server_args(args) -> None:
    validate_endpoint(Envs.coordinator_endpoint())
    if Envs.simulation_mode() != "OFFLINE":
        raise ValueError("coordinator requires OFFLINE")
    for name in (
        "enable_hierarchical_cache",
        "enable_lora",
        "enable_dp_attention",
        "speculative_algorithm",
    ):
        if getattr(args, name, None):
            raise ValueError(f"coordinator does not support {name}")
    if getattr(args, "disaggregation_mode", "null") != "null":
        raise ValueError("coordinator does not support disaggregation")
    if not getattr(args, "skip_server_warmup", False):
        raise ValueError("coordinator requires skip_server_warmup")
    if not getattr(args, "enable_cache_report", False):
        raise ValueError("coordinator requires enable_cache_report")
    for name in ("SGLANG_REQ_WAITING_TIMEOUT", "SGLANG_REQ_RUNNING_TIMEOUT"):
        if float(os.getenv(name, "-1")) > 0:
            raise ValueError(
                f"coordinator requires a virtual deadline instead of {name}"
            )


class CoordinatedScheduler:
    """Receive payloads while time is stopped; commit results at their due time.

    One instance lives in the scheduler process. A server handles one coordinated
    run, including its warmup; restarting it gives the next run a clean cache.
    """

    def __init__(self, *, scheduler, endpoint: str, timeout_s: float = 30.0):
        self.scheduler = scheduler
        self.clock = Coordinator()
        self.timeout_s = timeout_s
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.IPV6, 1)
        self._socket.bind(validate_endpoint(endpoint))
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
        for name in ("watchdog", "soft_watchdog"):
            watchdog = getattr(scheduler, name, None)
            if watchdog is not None:
                active = watchdog.is_active
                watchdog.is_active = lambda active=active: (
                    not self._waiting and active()
                )

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
            else:
                raise ProtocolError(f"unsupported request during coordination: {kind}")

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
                if self._finished:
                    if self._payloads:
                        raise ProtocolError("run already finished")
                    self._socket.poll(5)
                    return
                if self.clock.awaiting_ready and self._socket.poll(0):
                    self._reply_pending = True
                    ready = decode_ready(self._socket.recv())
                    lengths = {
                        len(op.rid) for op in ready.operations if isinstance(op, Submit)
                    }
                    if self._rid_length is not None:
                        lengths.add(self._rid_length)
                    if len(lengths) > 1:
                        raise ProtocolError(
                            "request IDs must have a fixed length within a run"
                        )
                    if lengths:
                        self._rid_length = lengths.pop()
                    if self.clock.run_id is None:
                        # Dataset preparation before the first Ready is unbounded.
                        start = time.monotonic()
                    self.clock.accept_ready(ready)
                    for op in ready.operations:
                        if isinstance(op, Cancel) and self.clock.is_cancelled(op.rid):
                            self._cancel(op.rid)
                    unexpected = {
                        rid for rid in self._aborts if not self.clock.is_terminal(rid)
                    }
                    if unexpected:
                        raise ProtocolError(
                            f"HTTP abort without Cancel: {sorted(unexpected)}"
                        )
                    self._aborts.clear()
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
                if self._admin and self._batch is None and self.clock.run_id is None:
                    return  # Bootstrap administration does not require a driver.
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
            self.scheduler.waiting_queue
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
        self._socket.send(encode_message(self.clock.grant(finished=finished)))
        self._reply_pending = False
        self._finished = finished

    def process_batch_result(self, batch, result, *, process: Callable):
        with self._guard():
            duration = StateManager.get_current_inference_dur()
            if (
                not math.isfinite(duration)
                or duration <= 0
                or round(duration * 1e9) == 0
            ):
                raise ProtocolError(f"invalid predicted batch duration: {duration}")
            finish_ns = self.clock.now_ns + round(duration * 1e9)
            self._batch = batch
            while (
                self.clock.next_wakeup_ns is not None
                and self.clock.next_wakeup_ns < finish_ns
            ):
                self._advance(self.clock.next_wakeup_ns)
                self._grant()
                self._settle()
            self._advance(finish_ns)
            self._cleanup_cancelled()
            attach_timing_output(result, self.clock)
            ret = process()
            self.scheduler.process_pending_chunked_abort()
            for req in batch.reqs:
                stats = request_stats_manager.get_req_stats(req.rid)
                if req.output_ids:
                    stats.gen_token_latencies.append(
                        finish_ns / 1e9 - stats.last_event_time
                    )
                    stats.last_event_time = finish_ns / 1e9
                if req.finished() and not self.clock.is_terminal(req.rid):
                    self.clock.terminal(req.rid, "COMPLETED")
            self._batch = None
            if self.clock.has_terminals or self.clock.next_wakeup_ns == finish_ns:
                self._grant()
            return ret

    def on_idle(self) -> None:
        with self._guard():
            if self._finished or self.clock.awaiting_ready:
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
