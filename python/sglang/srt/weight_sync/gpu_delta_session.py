"""Ordered control plane for the Miles-owned GPU delta receiver.

Miles owns these engines from startup through disposal and sends ordered controls.
Concurrent administration, retries, and arbitrary API sequences are unsupported.
Preparation owns immutable buffers; update pauses, fences readers, retracts, then mutates on the
scheduler thread. Failed or ambiguous updates require restart, never XOR retry.
"""

from __future__ import annotations

import asyncio
import os
import socket
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from sglang.srt.weight_sync import gpu_delta_io as delta_io


class GpuDeltaCommunicator:
    """Serialize delta controls and correlate replies without blocking generation."""

    def __init__(self, send: Callable, fan_out: int):
        self._send = send
        self._fan_out = fan_out
        self._lock = asyncio.Lock()
        self._rid = self._event = self._results = self._expected = None

    async def __call__(self, request):
        async with self._lock:
            request.rid = self._rid = uuid.uuid4().hex
            self._event = asyncio.Event()
            self._results = []
            self._expected = self._fan_out
            try:
                self._send(request)
                await self._event.wait()
                return self._results
            finally:
                self._rid = self._event = self._results = self._expected = None

    def handle_recv(self, reply):
        # Canceled requests can still reply after the next control was sent.
        if self._event is None or reply.rid != self._rid:
            return
        self._results.append(reply)
        if len(self._results) == self._expected:
            self._event.set()

    def set_fan_out(self, fan_out: int):
        self._fan_out = fan_out


@dataclass
class _Session:
    request: dict
    state: str = "PREPARING"
    message: str = ""
    prepared: Any = None
    result: dict = field(default_factory=dict)
    pause_started_ns: int | None = None
    reader_fence_completed_ns: int | None = None
    resumed_ns: int | None = None


class DeltaSession:
    """One original process following the Miles prepare/apply/resume sequence.

    The backend is injected so lifecycle tests do not need CUDA. ``prepare`` may
    allocate/upload on its own stream; ``apply`` must check decoder status and finish its GPU work
    and ``close`` must retain buffers until every use of that stream completes.
    """

    def __init__(self, identity: dict, backend: Any, initial_version: int = 0):
        self.identity = identity
        self.backend = backend
        self.version = initial_version
        self._session: _Session | None = None
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gpu-delta"
        )

    def status(self) -> dict:
        with self._lock:
            session = self._session
            request = session.request
            return {
                "identity": self.identity,
                "state": session.state,
                "message": session.message,
                **{
                    key: request[key]
                    for key in (
                        "session_id",
                        "manifest_sha256",
                        "stream_id",
                        "base_version",
                        "target_version",
                        "plan_digest",
                    )
                },
                "result": session.result,
                # One original process's clock, never subtract across ranks.
                # Open/failed pauses have no qualified completed duration.
                "scheduler_timing": {
                    "clock": "monotonic_ns",
                    "pause_started_ns": session.pause_started_ns,
                    "reader_fence_completed_ns": session.reader_fence_completed_ns,
                    "resumed_ns": session.resumed_ns,
                    "blocked_s": (
                        (session.resumed_ns - session.pause_started_ns) / 1e9
                        if session.resumed_ns is not None
                        and session.pause_started_ns is not None
                        else None
                    ),
                },
            }

    def prepare(self, request: dict) -> dict:
        with self._lock:
            if (
                request["base_version"] != self.version
                or request["target_version"] <= self.version
            ):
                raise ValueError(
                    "delta base version differs from committed local version"
                )
            session = self._session = _Session(request=request)
            self._executor.submit(self._prepare, session)
            return self.status()

    def _prepare(self, session: _Session) -> None:
        prepared = None
        try:
            req = session.request
            prepared = self.backend.prepare(
                req["manifest_path"], req["manifest_sha256"], req
            )
            with self._lock:
                if session.state != "ABORTED":
                    session.prepared = prepared
                    session.state = "PREPARED"
                    prepared = None
        except Exception as exc:
            with self._lock:
                if session.state != "ABORTED":
                    session.state = "FAILED"
                    session.message = f"preparation failed: {exc}"
        finally:
            if prepared is not None:
                prepared.close()

    def apply(
        self,
        fence: Callable[[], None],
        retract: Callable[[], None],
        flush: Callable[[], bool],
    ) -> dict:
        with self._lock:
            session = self._session
            # The scheduler has stopped new work. From this point a peer may
            # already be mutating. Miles does not abort after dispatching apply.
            session.state = "APPLYING"
            session.pause_started_ns = time.monotonic_ns()
        try:
            fence()
            session.reader_fence_completed_ns = time.monotonic_ns()
            # Never reclaim KV after a failed reader fence.
            retract()
            if not flush():
                raise ValueError("cache flush failed before delta mutation")
            result = session.prepared.apply()
        except Exception as exc:
            with self._lock:
                session.state = "POISONED"
                session.message = f"update failed after pause: {exc}; restart required"
            raise
        with self._lock:
            session.result = result
            session.state = "APPLIED"
            return self.status()

    def resume(self, resume: Callable[[int], None]) -> dict:
        with self._lock:
            session = self._session
            self.version = session.request["target_version"]
            session.state = "RESUMING"
            resume(self.version)
            session.resumed_ns = time.monotonic_ns()
            session.state = "RESUMED"
            prepared, session.prepared = session.prepared, None
            # Keep host release I/O off the scheduler; this FIFO executor runs
            # it before the next prepare. Miles sends resume after every rank applied.
            self._executor.submit(prepared.release_and_close)
            return self.status()

    def abort(self, session_id: str) -> dict:
        with self._lock:
            if self._session is None:
                return {
                    "identity": self.identity,
                    "state": "ABORTED",
                    "session_id": session_id,
                }
            session = self._session
            session.state = "ABORTED"
            if session.prepared is not None:
                prepared, session.prepared = session.prepared, None
                self._executor.submit(prepared.close)
            return self.status()


def with_gpu_delta_controls(scheduler, dispatcher):
    """Register only delta requests; ordinary handlers remain unchanged."""
    from sglang.utils import TypeBasedDispatcher

    control = GpuDeltaSchedulerControl(scheduler)
    dispatcher += TypeBasedDispatcher(
        [
            (delta_io.GetWeightsDeltaInfoReqInput, control.handle),
            (delta_io.PrepareWeightsFromDeltaReqInput, control.handle),
            (delta_io.GetWeightsDeltaStatusReqInput, control.handle),
            (delta_io.UpdateWeightsFromDeltaReqInput, control.handle),
            (delta_io.AbortWeightsFromDeltaReqInput, control.handle),
            (delta_io.ResumeWeightsFromDeltaReqInput, control.handle),
        ]
    )
    return dispatcher


class GpuDeltaSchedulerControl:
    """Small scheduler adapter; dependencies are lazy for ordinary disk users."""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.session: DeltaSession | None = None

    def _describe(self, engine_id: str) -> dict:
        from sglang.srt.runtime_context import get_exec, get_parallel

        scheduler = self.scheduler
        parallel = get_parallel()
        if (
            parallel.enable_dp_attention_local_control_broadcast
            or get_exec().moe.is_ep_scale_joiner
        ):
            raise ValueError(
                "GPU delta requires the global control broadcast without EP scale joiners"
            )
        # The existing reply transport emits one reply per attention-DP rank.
        # Reject hidden TP/CP/PP followers rather than claiming full coverage.
        if (
            parallel.attn_tp_size != 1
            or parallel.attn_cp_size != 1
            or parallel.pp_size != 1
        ):
            raise ValueError("GPU delta currently requires attention TP1/CP1 and PP1")
        if scheduler.enable_lora or str(scheduler.disaggregation_mode.value) != "null":
            raise ValueError(
                "GPU delta currently requires no LoRA and no disaggregation"
            )
        if scheduler.rust_server is not None:
            raise ValueError("GPU delta endpoints currently require the Python server")
        from sglang.srt.model_executor.model_runner_components.weight_updater import (
            _unsupported_derived_weight_cache_error,
        )

        runner = scheduler.tp_worker.model_runner
        # Reuse the ordinary updater's exclusions: shared IPC weights may belong
        # to other engines, and untracked derived caches would retain old values.
        runner.weight_updater._assert_weight_cache_inactive("update_weights_from_delta")
        error = _unsupported_derived_weight_cache_error(runner.model)
        if error is not None:
            raise ValueError(error)
        if self.session is None:
            from sglang.srt.weight_sync.gpu_delta_host import host_cache_id
            from sglang.srt.weight_sync.gpu_delta_layout import GpuDeltaBackend

            # proc stat starttime is the 22nd field; comm may contain spaces.
            with open("/proc/self/stat") as source:
                start_ticks = int(source.read().rsplit(")", 1)[1].split()[19])
            identity = {
                "engine_id": engine_id,
                "rank_id": uuid.uuid4().hex,
                "hostname": socket.gethostname(),
                "host_cache_id": host_cache_id(engine_id),
                "pid": os.getpid(),
                "start_ticks": start_ticks,
                "tp_rank": parallel.tp_rank,
                "dp_rank": parallel.attn_dp_rank,
                "pp_rank": parallel.pp_rank,
            }
            backend = GpuDeltaBackend(runner, identity)
            self.session = DeltaSession(identity, backend)
        return {
            "identity": self.session.identity,
            "state": "IDLE",
            "version": self.session.version,
            "plan": self.session.backend.describe(),
        }

    def handle(self, request):
        from sglang.srt.managers import io_struct as io

        try:
            if isinstance(request, delta_io.GetWeightsDeltaInfoReqInput):
                receipt = self._describe(request.engine_id)
            elif isinstance(request, delta_io.PrepareWeightsFromDeltaReqInput):
                receipt = self.session.prepare(
                    {
                        "session_id": request.session_id,
                        "manifest_path": request.manifest_path,
                        "manifest_sha256": request.manifest_sha256,
                        "stream_id": request.stream_id,
                        "base_version": request.base_version,
                        "target_version": request.target_version,
                        "plan_digest": request.plan_digest,
                        "participants": request.participants,
                        "host_tensor_names": request.host_tensor_names,
                    }
                )
            elif isinstance(request, delta_io.GetWeightsDeltaStatusReqInput):
                receipt = self.session.status()
            elif isinstance(request, delta_io.UpdateWeightsFromDeltaReqInput):
                self.scheduler._engine_paused = True
                receipt = self.session.apply(
                    self.scheduler.device_module.synchronize,
                    lambda: self.scheduler.pause_generation(
                        io.PauseGenerationReqInput(mode="retract")
                    ),
                    lambda: self.scheduler.flush_cache(empty_cache=False),
                )
            elif isinstance(request, delta_io.ResumeWeightsFromDeltaReqInput):

                def resume(version):
                    self.scheduler.record_weight_version_change(str(version))
                    self.scheduler.continue_generation(
                        io.ContinueGenerationReqInput(torch_empty_cache=False)
                    )

                receipt = self.session.resume(resume)
            elif isinstance(request, delta_io.AbortWeightsFromDeltaReqInput):
                receipt = self.session.abort(request.session_id)
            else:
                raise ValueError("unknown delta operation")
            success = receipt["state"] not in {"FAILED", "POISONED"}
            return delta_io.DeltaWeightsReqOutput(
                rid=request.rid,
                success=success,
                message=receipt.get("message", ""),
                participant=receipt,
            )
        except Exception as exc:
            receipt = {
                "identity": self.session.identity if self.session else None,
                "state": "REJECTED",
            }
            if self.session is not None and self.session._session is not None:
                receipt = self.session.status()
            return delta_io.DeltaWeightsReqOutput(
                rid=request.rid, success=False, message=str(exc), participant=receipt
            )
