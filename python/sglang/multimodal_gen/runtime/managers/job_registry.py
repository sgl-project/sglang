# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side job control: identity, idempotency, cancel, status."""

from __future__ import annotations

import threading
import time
from typing import Any

import msgspec

from sglang.multimodal_gen.runtime.ipc_array import NumpyArrayFileRef

_FINISHED_RETENTION = 64
_FINISHED_TTL_S = 300.0
_PRECANCEL_CAP = 1024
_FINISHED_HARD_CAP = 4 * _FINISHED_RETENTION

QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"
_TERMINAL = (COMPLETED, FAILED, CANCELLED)


class RequestCancelledError(Exception):
    pass


def contains_file_refs(value: Any) -> bool:
    """True if the payload holds single-consumer spilled array refs."""
    if isinstance(value, NumpyArrayFileRef):
        return True
    if isinstance(value, (list, tuple)):
        return any(contains_file_refs(item) for item in value)
    return False


def _is_replayable(output: Any) -> bool:
    """Only lightweight terminal payloads are worth retaining for idempotent
    replay; bulk payloads (raw realtime frames, trajectory tensors) would pin
    GPU-scale buffers for the whole retention window."""
    return (
        not contains_file_refs(output.output)
        and output.raw_frame_batches is None
        and output.trajectory_latents is None
        and output.rollout_trajectory_data is None
    )


class CancelReq(msgspec.Struct):
    request_id: str


class JobStatusReq(msgspec.Struct):
    request_id: str


class JobHandle:
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.status = QUEUED
        self.created_ts = time.time()
        self.finished_ts: float | None = None
        self.step: int | None = None
        self.total_steps: int | None = None
        self.cancel_event = threading.Event()
        self.waiters: list[bytes] = []
        self.output: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status,
            "created_ts": self.created_ts,
            "step": self.step,
            "total_steps": self.total_steps,
        }


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, JobHandle] = {}
        self._finished: list[str] = []
        self._precancelled: dict[str, float] = {}
        self._lock = threading.Lock()

    def admit(
        self, request_id: str, identity: bytes | None
    ) -> tuple[str, JobHandle | Any | None]:
        """Return ("new"|"wait"|"replay"|"cancelled", handle_or_output)."""
        with self._lock:
            handle = self._jobs.get(request_id)
            if handle is None:
                self._expire_precancelled()
                if request_id in self._precancelled:
                    del self._precancelled[request_id]
                    handle = JobHandle(request_id)
                    handle.status = CANCELLED
                    handle.cancel_event.set()
                    handle.finished_ts = time.time()
                    self._jobs[request_id] = handle
                    self._finished.append(request_id)
                    self._trim_finished()
                    return ("cancelled", None)
                handle = JobHandle(request_id)
                self._jobs[request_id] = handle
                return ("new", handle)
            if handle.status in _TERMINAL:
                return ("replay", handle.output)
            if identity is not None:
                handle.waiters.append(identity)
            return ("wait", handle)

    def mark_running(self, request_id: str) -> JobHandle | None:
        with self._lock:
            handle = self._jobs.get(request_id)
            if handle is not None and handle.status == QUEUED:
                handle.status = RUNNING
            return handle

    def finish(self, request_id: str, output: Any) -> list[bytes]:
        """Mark terminal and return waiter identities owed a reply."""
        with self._lock:
            handle = self._jobs.get(request_id)
            if handle is None:
                return []
            if handle.status not in _TERMINAL:
                if output.cancelled:
                    handle.status = CANCELLED
                elif output.error:
                    handle.status = FAILED
                else:
                    handle.status = COMPLETED
                handle.finished_ts = time.time()
                self._finished.append(request_id)
                self._trim_finished()
            handle.output = output if _is_replayable(output) else None
            waiters, handle.waiters = handle.waiters, []
            return waiters

    def _expire_precancelled(self) -> None:
        now = time.time()
        self._precancelled = {
            request_id: expiry
            for request_id, expiry in self._precancelled.items()
            if expiry > now
        }

    def _trim_finished(self) -> None:
        now = time.time()
        while len(self._finished) > _FINISHED_RETENTION:
            oldest = self._jobs.get(self._finished[0])
            if (
                oldest is not None
                and oldest.finished_ts is not None
                and now - oldest.finished_ts < _FINISHED_TTL_S
                and len(self._finished) <= _FINISHED_HARD_CAP
            ):
                break
            self._jobs.pop(self._finished.pop(0), None)

    def cancel(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            handle = self._jobs.get(request_id)
            if handle is None:
                self._expire_precancelled()
                if len(self._precancelled) >= _PRECANCEL_CAP:
                    self._precancelled.pop(next(iter(self._precancelled)), None)
                self._precancelled[request_id] = time.time() + _FINISHED_TTL_S
                return {
                    "request_id": request_id,
                    "status": "unknown",
                    "cancelled": True,
                }
            if handle.status in _TERMINAL:
                return {**handle.to_dict(), "cancelled": False}
            handle.cancel_event.set()
            return {**handle.to_dict(), "cancelled": True}

    def is_cancelled(self, request_id: str) -> bool:
        with self._lock:
            handle = self._jobs.get(request_id)
            return handle is not None and handle.cancel_event.is_set()

    def status(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            handle = self._jobs.get(request_id)
            if handle is None:
                return {"request_id": request_id, "status": "unknown"}
            return handle.to_dict()


_current_jobs: list[JobHandle] = []


def set_current_jobs(handles: list[JobHandle]) -> None:
    global _current_jobs
    _current_jobs = list(handles)


def clear_current_jobs() -> None:
    global _current_jobs
    _current_jobs = []


def check_current_step(step_index: int, total_steps: int) -> None:
    """Between-step cancellation and progress point for denoise loops."""
    jobs = _current_jobs
    if not jobs:
        return
    for handle in jobs:
        handle.step = step_index
        handle.total_steps = total_steps
    if not all(handle.cancel_event.is_set() for handle in jobs):
        return
    raise RequestCancelledError(f"cancelled at denoise step {step_index}/{total_steps}")
