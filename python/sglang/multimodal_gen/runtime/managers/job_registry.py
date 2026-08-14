# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side job control: identity, idempotency, cancel, status."""

from __future__ import annotations

import dataclasses
import sys
import threading
import time
from typing import Any

import msgspec

from sglang.multimodal_gen.runtime.ipc_array import NumpyArrayFileRef

_FINISHED_RETENTION = 64
_FINISHED_TTL_S = 300.0
_PRECANCEL_CAP = 1024
_FINISHED_HARD_CAP = 4 * _FINISHED_RETENTION
_WAITER_CAP = 64
_REPLAY_BYTES_CAP = 128 << 20
_LIVE_JOB_CAP = 1024

QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"
_TERMINAL = (COMPLETED, FAILED, CANCELLED)


class RequestCancelledError(Exception):
    pass


class RequestConflictError(Exception):
    pass


class RequestOverloadedError(Exception):
    pass


def contains_file_refs(value: Any) -> bool:
    """True if the payload holds single-consumer spilled array refs."""
    if isinstance(value, NumpyArrayFileRef):
        return True
    if isinstance(value, (list, tuple)):
        return any(contains_file_refs(item) for item in value)
    if isinstance(value, dict):
        return any(contains_file_refs(item) for item in value.values())
    return False


def _value_nbytes(value: Any, seen: set[int] | None = None) -> int | None:
    if value is None:
        return 0
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return 0
    seen.add(value_id)
    if isinstance(value, NumpyArrayFileRef):
        return None
    dtype = getattr(value, "dtype", None)
    if getattr(dtype, "hasobject", False):
        return None
    device = getattr(value, "device", None)
    if device is not None and str(device) != "cpu":
        return None
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        size = max(sys.getsizeof(value), nbytes)
        storage = getattr(value, "untyped_storage", None)
        if callable(storage):
            try:
                size = max(size, storage().nbytes())
            except (AttributeError, RuntimeError, TypeError):
                return None
        backing = getattr(value, "base", None)
        if backing is None:
            backing = getattr(value, "_base", None)
        if backing is None and isinstance(value, memoryview):
            backing = value.obj
        if backing is not None:
            backing_size = _value_nbytes(backing, seen)
            return None if backing_size is None else size + backing_size
        return size
    size = sys.getsizeof(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        sizes = [_value_nbytes(item, seen) for item in value]
    elif isinstance(value, dict):
        sizes = [_value_nbytes(item, seen) for pair in value.items() for item in pair]
    elif isinstance(value, (str, bytes, bytearray, memoryview, bool, int, float)):
        return size
    elif hasattr(value, "__dict__"):
        sizes = [_value_nbytes(vars(value), seen)]
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        sizes = [
            _value_nbytes(getattr(value, field.name), seen)
            for field in dataclasses.fields(value)
        ]
    else:
        return None
    return None if any(item is None for item in sizes) else size + sum(sizes)


def _replay_size(output: Any) -> int | None:
    """Return total retained bytes, or None for unsafe payloads."""
    if (
        getattr(output, "output_file_paths", None)
        or getattr(output, "raw_frame_batches", None) is not None
        or getattr(output, "audio", None) is not None
        or getattr(output, "action_pred", None) is not None
        or getattr(output, "trajectory_timesteps", None) is not None
        or getattr(output, "trajectory_latents", None) is not None
        or getattr(output, "trajectory_decoded", None) is not None
        or getattr(output, "rollout_trajectory_data", None) is not None
        or getattr(output, "noise_pred", None) is not None
    ):
        return None
    return _value_nbytes(output)


class CancelReq(msgspec.Struct):
    request_id: str


class JobStatusReq(msgspec.Struct):
    request_id: str


class JobHandle:
    def __init__(self, request_id: str, fingerprint: str | None = None) -> None:
        self.request_id = request_id
        self.fingerprint = fingerprint
        self.status = QUEUED
        self.created_ts = time.time()
        self.finished_at_monotonic: float | None = None
        self.step: int | None = None
        self.total_steps: int | None = None
        self.cancel_event = threading.Event()
        self.waiters: list[bytes] = []
        self.output: Any | None = None
        self.output_bytes = 0

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
        self._live_jobs = 0
        self._replay_bytes = 0
        self._lock = threading.Lock()

    def admit(
        self,
        request_id: str,
        identity: bytes | None,
        fingerprint: str | None = None,
    ) -> tuple[str, JobHandle | Any | None]:
        """Return an admission verdict and its handle or cached output."""
        with self._lock:
            handle = self._jobs.get(request_id)
            if handle is None:
                self._expire_precancelled()
                if request_id in self._precancelled:
                    del self._precancelled[request_id]
                    handle = JobHandle(request_id, fingerprint)
                    handle.status = CANCELLED
                    handle.cancel_event.set()
                    handle.finished_at_monotonic = time.monotonic()
                    self._jobs[request_id] = handle
                    self._finished.append(request_id)
                    self._trim_finished()
                    return ("cancelled", None)
                if self._live_jobs >= _LIVE_JOB_CAP:
                    return ("capacity", None)
                handle = JobHandle(request_id, fingerprint)
                self._jobs[request_id] = handle
                self._live_jobs += 1
                return ("new", handle)
            if handle.fingerprint is None or handle.fingerprint != fingerprint:
                return ("conflict", None)
            if handle.status in _TERMINAL:
                return ("replay", handle.output)
            if identity is not None:
                if len(handle.waiters) >= _WAITER_CAP:
                    return ("overloaded", None)
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
                handle.finished_at_monotonic = time.monotonic()
                self._finished.append(request_id)
                self._live_jobs -= 1
            self._drop_replay(handle)
            replay_size = _replay_size(output)
            if handle.fingerprint is None and not output.cancelled and not output.error:
                replay_size = None
            if replay_size is not None and replay_size <= _REPLAY_BYTES_CAP:
                handle.output = output
                handle.output_bytes = replay_size
                self._replay_bytes += replay_size
            self._trim_finished()
            self._trim_replay_bytes()
            waiters, handle.waiters = handle.waiters, []
            return waiters

    def _expire_precancelled(self) -> None:
        now = time.monotonic()
        self._precancelled = {
            request_id: expiry
            for request_id, expiry in self._precancelled.items()
            if expiry > now
        }

    def _trim_finished(self) -> None:
        now = time.monotonic()
        while len(self._finished) > _FINISHED_RETENTION:
            oldest = self._jobs.get(self._finished[0])
            if (
                oldest is not None
                and oldest.finished_at_monotonic is not None
                and now - oldest.finished_at_monotonic < _FINISHED_TTL_S
                and len(self._finished) <= _FINISHED_HARD_CAP
            ):
                break
            evicted = self._jobs.pop(self._finished.pop(0), None)
            if evicted is not None:
                self._drop_replay(evicted)

    def _drop_replay(self, handle: JobHandle) -> None:
        self._replay_bytes -= handle.output_bytes
        handle.output = None
        handle.output_bytes = 0

    def _trim_replay_bytes(self) -> None:
        for request_id in self._finished:
            if self._replay_bytes <= _REPLAY_BYTES_CAP:
                break
            handle = self._jobs.get(request_id)
            if handle is not None:
                self._drop_replay(handle)

    def cancel(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            handle = self._jobs.get(request_id)
            if handle is None:
                self._expire_precancelled()
                if request_id in self._precancelled:
                    self._precancelled[request_id] = time.monotonic() + _FINISHED_TTL_S
                    return {
                        "request_id": request_id,
                        "status": "unknown",
                        "cancelled": True,
                    }
                if len(self._precancelled) >= _PRECANCEL_CAP:
                    return {
                        "request_id": request_id,
                        "status": "unknown",
                        "cancelled": False,
                        "overloaded": True,
                    }
                self._precancelled[request_id] = time.monotonic() + _FINISHED_TTL_S
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


_current_job: JobHandle | None = None


def set_current_job(handle: JobHandle | None) -> None:
    global _current_job
    _current_job = handle


def clear_current_job() -> None:
    global _current_job
    _current_job = None


def check_current_step(step_index: int, total_steps: int) -> None:
    """Between-step cancellation and progress point for denoise loops."""
    job = _current_job
    if job is None:
        return
    job.step = step_index
    job.total_steps = total_steps
    if job.cancel_event.is_set():
        raise RequestCancelledError(
            f"cancelled at denoise step {step_index}/{total_steps}"
        )
