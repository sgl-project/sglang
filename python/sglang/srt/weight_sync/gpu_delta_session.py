"""Fail-closed control plane for the opt-in GPU delta receiver.

One exclusive controller owns these engines from startup through disposal. Mixing
ordinary weight, memory, topology, or pause controls is unsupported. Preparation
owns immutable buffers; update pauses, fences readers, retracts, then mutates on the
scheduler thread. Failed or ambiguous updates require restart, never XOR retry.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import socket
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any


class GpuDeltaConflict(ValueError):
    """A delta control conflicts with the current engine/session state."""


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


def _identity_key(identity: dict) -> str:
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _identities(values: list[dict]) -> set[str]:
    keys = {_identity_key(value) for value in values}
    if not values or len(keys) != len(values):
        raise ValueError("participant identities must be nonempty and unique")
    return keys


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
    """One original process, one leased publication, and no automatic XOR retry.

    The backend is injected so lifecycle tests do not need CUDA. ``prepare`` may
    allocate/upload on its own stream; ``apply`` must check decoder status and finish its GPU work
    and ``close`` must retain buffers until every use of that stream completes.
    """

    def __init__(self, identity: dict, backend: Any, initial_version: int = 0):
        self.identity = copy.deepcopy(identity)
        self.backend = backend
        self.version = initial_version
        self.stream_id = None
        self._session: _Session | None = None
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gpu-delta"
        )
        self._seen_ids: set[str] = set()

    @property
    def leased(self) -> bool:
        with self._lock:
            return self._session is not None and self._session.state not in {
                "ABORTED",
                "RESUMED",
            }

    def _get(self, session_id: str) -> _Session:
        if self._session is None or self._session.request["session_id"] != session_id:
            raise ValueError(
                "unknown delta session; never retry an ambiguous update as a new session"
            )
        return self._session

    def status(self, session_id: str) -> dict:
        with self._lock:
            session = self._get(session_id)
            request = session.request
            receipt = {
                "identity": copy.deepcopy(self.identity),
                "state": session.state,
                "cohort_digest": hashlib.sha256(
                    json.dumps(sorted(_identities(request["cohort"]))).encode()
                ).hexdigest(),
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
                "result": copy.deepcopy(session.result),
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
            if session.state == "APPLIED":
                receipt["certificate"] = {
                    key: copy.deepcopy(receipt[key])
                    for key in (
                        "identity",
                        "state",
                        "cohort_digest",
                        "session_id",
                        "manifest_sha256",
                        "stream_id",
                        "base_version",
                        "target_version",
                        "plan_digest",
                    )
                }
            return receipt

    def prepare(self, request: dict) -> dict:
        request = copy.deepcopy(request)
        with self._lock:
            session_id = request["session_id"]
            if self.leased or session_id in self._seen_ids:
                raise ValueError(
                    "delta session already leased or session identity already consumed"
                )
            local = _identities(request["participants"])
            _identities(request["cohort"])
            engine_id = self.identity["engine_id"]
            cohort_local = _identities(
                [item for item in request["cohort"] if item["engine_id"] == engine_id]
            )
            if _identity_key(self.identity) not in local or local != cohort_local:
                raise ValueError(
                    "prepare does not bind this original engine's complete participants"
                )
            if (
                request["base_version"] != self.version
                or request["target_version"] <= self.version
            ):
                raise ValueError(
                    "delta base version differs from committed local version"
                )
            if self.stream_id is not None and request["stream_id"] != self.stream_id:
                raise ValueError("delta stream changed")
            session = self._session = _Session(request=request)
            self._seen_ids.add(session_id)
            self._executor.submit(self._prepare, session)
            return self.status(session_id)

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
        session_id: str,
        fence: Callable[[], None],
        retract: Callable[[], None],
        flush: Callable[[], bool],
    ) -> dict:
        with self._lock:
            session = self._get(session_id)
            if session.state != "PREPARED":
                raise ValueError(f"apply requires PREPARED, got {session.state}")
            # The scheduler has stopped new work. From this point a peer may
            # already be mutating: even a failed fence cannot make this abortable.
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
            return self.status(session_id)

    def _validate_applied(self, session: _Session, receipts: list[dict]) -> None:
        expected = _identities(session.request["cohort"])
        actual = _identities([receipt["identity"] for receipt in receipts])
        if actual != expected:
            raise ValueError(
                "certificate must contain every original rank exactly once"
            )
        cohort_digest = hashlib.sha256(
            json.dumps(sorted(expected)).encode()
        ).hexdigest()
        for receipt in receipts:
            if receipt.get("cohort_digest") != cohort_digest:
                raise ValueError("certificate cohort differs from the prepared cohort")
            if receipt["state"] != "APPLIED":
                raise ValueError("certificate requires APPLIED on every rank")
            for key in (
                "session_id",
                "manifest_sha256",
                "stream_id",
                "base_version",
                "target_version",
                "plan_digest",
            ):
                if receipt[key] != session.request[key]:
                    raise ValueError(
                        f"certificate {key} differs from prepared publication"
                    )

    def resume(
        self, session_id: str, receipts: list[dict], resume: Callable[[int], None]
    ) -> dict:
        with self._lock:
            session = self._get(session_id)
            if session.state != "APPLIED":
                raise ValueError(f"resume requires APPLIED, got {session.state}")
            self._validate_applied(session, receipts)
            self.version = session.request["target_version"]
            self.stream_id = session.request["stream_id"]
            session.state = "RESUMING"
            resume(self.version)
            session.resumed_ns = time.monotonic_ns()
            session.state = "RESUMED"
            prepared, session.prepared = session.prepared, None
            # Keep host release I/O off the scheduler; this FIFO executor runs
            # it before the next prepare. Only global APPLIED resume releases.
            self._executor.submit(prepared.release_and_close)
            return self.status(session_id)

    def abort(self, session_id: str) -> dict:
        with self._lock:
            if not self.leased and (
                self._session is None
                or self._session.request["session_id"] != session_id
            ):
                return {
                    "identity": copy.deepcopy(self.identity),
                    "state": "ABORTED",
                    "session_id": session_id,
                }
            session = self._get(session_id)
            if session.state not in {"PREPARING", "PREPARED", "FAILED", "ABORTED"}:
                raise ValueError(
                    f"cannot abort {session.state}; keep all engines paused"
                )
            session.state = "ABORTED"
            if session.prepared is not None:
                prepared, session.prepared = session.prepared, None
                self._executor.submit(prepared.close)
            return self.status(session_id)


def with_gpu_delta_controls(scheduler, dispatcher):
    """Register only delta requests; ordinary handlers remain unchanged."""
    from sglang.srt.managers import io_struct as io
    from sglang.utils import TypeBasedDispatcher

    control = GpuDeltaSchedulerControl(scheduler)
    dispatcher += TypeBasedDispatcher(
        [
            (io.GetWeightsDeltaInfoReqInput, control.handle),
            (io.PrepareWeightsFromDeltaReqInput, control.handle),
            (io.GetWeightsDeltaStatusReqInput, control.handle),
            (io.UpdateWeightsFromDeltaReqInput, control.handle),
            (io.AbortWeightsFromDeltaReqInput, control.handle),
            (io.ResumeWeightsFromDeltaReqInput, control.handle),
        ]
    )
    return dispatcher


class GpuDeltaSchedulerControl:
    """Small scheduler adapter; dependencies are lazy for ordinary disk users."""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.session: DeltaSession | None = None
        self.identity: dict | None = None
        self.backend = None

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
        if self.identity is None:
            from sglang.srt.weight_sync.gpu_delta_host import host_cache_id
            from sglang.srt.weight_sync.gpu_delta_layout import GpuDeltaBackend

            # proc stat starttime is the 22nd field; comm may contain spaces.
            with open("/proc/self/stat") as source:
                start_ticks = int(source.read().rsplit(")", 1)[1].split()[19])
            identity = {
                "engine_id": engine_id,
                "rank_id": uuid.uuid4().hex,
                "hostname": socket.gethostname(),
                "host_cache_id": host_cache_id(),
                "pid": os.getpid(),
                "start_ticks": start_ticks,
                "tp_rank": parallel.tp_rank,
                "dp_rank": parallel.attn_dp_rank,
                "pp_rank": parallel.pp_rank,
            }
            backend = GpuDeltaBackend(runner, identity)
            self.identity = identity
            self.backend = backend
            self.session = DeltaSession(identity, backend)
        elif self.identity["engine_id"] != engine_id:
            raise ValueError("engine identity is already bound")
        return {
            "identity": self.identity,
            "state": "IDLE",
            "version": self.session.version,
            "plan": self.backend.describe(),
        }

    def handle(self, request):
        from sglang.srt.managers import io_struct as io

        try:
            if isinstance(request, io.GetWeightsDeltaInfoReqInput):
                receipt = self._describe(request.engine_id)
            elif isinstance(request, io.PrepareWeightsFromDeltaReqInput):
                if (
                    self.identity is not None
                    and request.engine_id != self.identity["engine_id"]
                ):
                    raise ValueError("prepare engine identity mismatch")
                if self.session is None:
                    raise ValueError(
                        "describe original engine participants before prepare"
                    )
                if (
                    self.scheduler.weight_updater._session is not None
                    or self.scheduler.weight_updater.offload_tags
                ):
                    raise ValueError(
                        "another weight update or memory offload is active"
                    )
                receipt = self.session.prepare(
                    {
                        key: getattr(request, key)
                        for key in (
                            "session_id",
                            "manifest_path",
                            "manifest_sha256",
                            "stream_id",
                            "base_version",
                            "target_version",
                            "plan_digest",
                            "participants",
                            "cohort",
                            "host_tensor_names",
                        )
                    }
                )
            elif self.session is None:
                raise ValueError("no GPU delta session")
            elif isinstance(request, io.GetWeightsDeltaStatusReqInput):
                receipt = self.session.status(request.session_id)
            elif isinstance(request, io.UpdateWeightsFromDeltaReqInput):
                self.scheduler._engine_paused = True
                receipt = self.session.apply(
                    request.session_id,
                    self.scheduler.device_module.synchronize,
                    lambda: self.scheduler.pause_generation(
                        io.PauseGenerationReqInput(mode="retract")
                    ),
                    lambda: self.scheduler.flush_cache(empty_cache=False),
                )
            elif isinstance(request, io.ResumeWeightsFromDeltaReqInput):

                def resume(version):
                    self.scheduler.record_weight_version_change(str(version))
                    self.scheduler.continue_generation(
                        io.ContinueGenerationReqInput(torch_empty_cache=False)
                    )

                receipt = self.session.resume(
                    request.session_id, request.receipts, resume
                )
            elif isinstance(request, io.AbortWeightsFromDeltaReqInput):
                receipt = self.session.abort(request.session_id)
            else:
                raise ValueError("unknown delta operation")
            success = receipt["state"] not in {"FAILED", "POISONED"}
            return io.DeltaWeightsReqOutput(
                rid=request.rid,
                success=success,
                message=receipt.get("message", ""),
                participant=receipt,
            )
        except Exception as exc:
            receipt = {"identity": self.identity, "state": "REJECTED"}
            if self.session is not None and getattr(request, "session_id", None):
                try:
                    receipt = self.session.status(request.session_id)
                except ValueError:
                    pass
            return io.DeltaWeightsReqOutput(
                rid=request.rid, success=False, message=str(exc), participant=receipt
            )
