"""A Transport that moves no bytes, so the PD protocol runs without an engine.

It keeps the registration bookkeeping a real backend keeps, checks every write
against it, and lets a test decide when a write finishes.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Sequence

import numpy as np

from sglang.srt.disaggregation.transport.base import (
    BufferKey,
    Handle,
    RemoteBuffers,
    Result,
    Status,
    Transport,
)


class NullRemoteBuffers(RemoteBuffers):
    def __init__(self, lengths: Dict[BufferKey, int]):
        # Lengths only: nothing is written, so a peer needs no addresses.
        self.lengths = lengths


class NullHandle(Handle):
    def __init__(
        self, *, src: BufferKey, dst: BufferKey, ranges: np.ndarray, result: Result
    ):
        self.src = src
        self.dst = dst
        # (N, 3) uint64 rows of (src_offset, dst_offset, length).
        self.ranges = ranges
        self.result = result


class NullTransport(Transport):
    """A write an engine would reject (unknown key, range past either buffer,
    closed transport) is FAILED with a reason. Any other write is DONE at once,
    or stays PENDING under hold() until complete() or fail()."""

    def __init__(self) -> None:
        # Protocol worker threads write and poll while a test thread finishes writes.
        self._lock = threading.Lock()
        self._lengths: Dict[BufferKey, int] = {}
        self._hold_all = False
        self._held_dsts: set[BufferKey] = set()
        self._closed = False
        # Every accepted write, in submission order.
        self.writes: List[NullHandle] = []

    def register(self, key: BufferKey, ptr: int, length: int, device: int) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("NullTransport is closed")
            self._lengths[key] = length

    def deregister(self, keys: Sequence[BufferKey]) -> None:
        with self._lock:
            for key in keys:
                self._lengths.pop(key, None)

    def export_registration(self) -> Dict[BufferKey, int]:
        with self._lock:
            return dict(self._lengths)

    def import_registration(
        self, *, lengths: Dict[BufferKey, int]
    ) -> NullRemoteBuffers:
        return NullRemoteBuffers(lengths=dict(lengths))

    def write(
        self,
        remote: NullRemoteBuffers,
        src: BufferKey,
        dst: BufferKey,
        ranges: np.ndarray,
    ) -> NullHandle:
        with self._lock:
            reason = self._reject_reason(remote=remote, src=src, dst=dst, ranges=ranges)
            if reason is not None:
                result = Result(Status.FAILED, reason=reason)
            elif self._hold_all or dst in self._held_dsts:
                result = Result(Status.PENDING)
            else:
                result = Result(Status.DONE)
            handle = NullHandle(src=src, dst=dst, ranges=ranges, result=result)
            if reason is None:
                self.writes.append(handle)
            return handle

    def status(self, handle: NullHandle) -> Result:
        with self._lock:
            return handle.result

    def close(self) -> None:
        with self._lock:
            self._lengths.clear()
            self._closed = True

    def hold(self, dst: Optional[BufferKey] = None) -> None:
        """Leave later writes PENDING: every write, or only those into `dst`."""
        with self._lock:
            if dst is None:
                self._hold_all = True
            else:
                self._held_dsts.add(dst)

    def release(self) -> None:
        """Later writes finish at once again; writes already held stay PENDING."""
        with self._lock:
            self._hold_all = False
            self._held_dsts.clear()

    def pending(self) -> List[NullHandle]:
        with self._lock:
            return [h for h in self.writes if h.result.status is Status.PENDING]

    def complete(self, handle: NullHandle) -> None:
        self._finish(handle=handle, result=Result(Status.DONE))

    def fail(self, handle: NullHandle, reason: str = "injected failure") -> None:
        self._finish(handle=handle, result=Result(Status.FAILED, reason=reason))

    def _finish(self, *, handle: NullHandle, result: Result) -> None:
        with self._lock:
            if handle.result.status is not Status.PENDING:
                raise RuntimeError(
                    f"write is already {handle.result.status.value}; "
                    "a final status cannot change"
                )
            handle.result = result

    def _reject_reason(
        self,
        *,
        remote: NullRemoteBuffers,
        src: BufferKey,
        dst: BufferKey,
        ranges: np.ndarray,
    ) -> Optional[str]:
        if self._closed:
            return "transport is closed"
        if src not in self._lengths:
            return f"source {src!r} is not registered"
        if dst not in remote.lengths:
            return f"destination {dst!r} is not in the peer registration"
        src_length = self._lengths[src]
        dst_length = remote.lengths[dst]
        # Python ints: a uint64 offset + length wraps silently at 2**64.
        for src_offset, dst_offset, length in ranges.tolist():
            if src_offset + length > src_length:
                return (
                    f"range reads {src!r} to byte {src_offset + length}, "
                    f"past its length {src_length}"
                )
            if dst_offset + length > dst_length:
                return (
                    f"range writes {dst!r} to byte {dst_offset + length}, "
                    f"past its length {dst_length}"
                )
        return None
