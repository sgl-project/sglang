"""Per-backend transport interface of PD disaggregation (RFC #46455).

The protocol (KVManager / KVSender / KVReceiver) decides which bytes move where
and when; a Transport copies bytes between registered buffers and reports
transfer status. Only the KVManager holds and calls it. Request identity,
topology, state types, protocol timeout and retry policy, staging lifecycle and
decode completion signaling stay in the protocol.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import msgspec
import numpy as np


class BufferKey(ABC):
    """Hashable buffer identity whose meaning is defined by the protocol.
    The Transport stores keys without interpreting them."""

    @abstractmethod
    def __eq__(self, other: object) -> bool: ...

    @abstractmethod
    def __hash__(self) -> int: ...


class RemoteBuffers:
    """Opaque result of import_buffers(), passed to write() and release_remote();
    compared by identity."""


class BufferExport:
    """Opaque description of registered buffers from export_buffers(); the
    protocol carries it to the peer, which passes it to import_buffers()."""


class Handle:
    """Opaque write result queried through status(); compared by identity."""


class Status(enum.Enum):
    """DONE and FAILED are final: once reported, every later status() repeats them.
    T_rep and T_err are the reporting and stop bounds of ADR #44186."""

    # Lasts at most T_rep after the write is posted.
    PENDING = "pending"
    # Every byte is in the destination and nothing is still in flight; a message
    # sent to the peer after seeing DONE arrives after the data.
    DONE = "done"
    # No claim about what was written; the write stops touching memory within
    # T_err after FAILED is reported.
    FAILED = "failed"


class Result(msgspec.Struct, frozen=True):
    status: Status
    # Set when status is FAILED.
    reason: Optional[str] = None


class Copy(msgspec.Struct, frozen=True):
    src: BufferKey
    dst: BufferKey
    # (N, 3) uint64 rows of (src_offset, dst_offset, length).
    ranges: np.ndarray


class Transport(ABC):
    """Owns registrations and any dedicated engine assigned to this Transport.

    Shared engines are borrowed. The caller owns backing memory and keeps it
    valid while registered. Before removing registrations or closing, the
    caller must ensure local and peer transfers have stopped using the buffers.
    """

    @abstractmethod
    def register(self, key: BufferKey, ptr: int, length: int, device: int) -> None:
        """Register [ptr, ptr + length) under `key` for local reads and peer writes.
        `device` is the GPU index, or -1 for page-locked host memory."""

    @abstractmethod
    def deregister(self, keys: Sequence[BufferKey]) -> None:
        """Unregister `keys`, preserving regions still used by other keys.
        Leave backing memory allocated and the Transport usable."""

    # Not abstract: a backend whose protocol still exchanges registrations
    # itself does not implement the next three.
    def export_buffers(self, keys: Sequence[BufferKey]) -> BufferExport:
        """Describe the buffers registered under `keys`, in that order, for a
        peer to write into. Covers registrations made before the call."""
        raise NotImplementedError

    def import_buffers(
        self, desc: BufferExport, keys: Sequence[BufferKey]
    ) -> RemoteBuffers:
        """Import a peer's export_buffers() result. `keys` names the peer's
        buffers for write(), in the peer's export order."""
        raise NotImplementedError

    def release_remote(self, remote: RemoteBuffers) -> None:
        """Drop the state import_buffers() created for `remote`; later imports
        of the same peer stay usable. The caller ensures no write to `remote`
        can still access memory."""
        raise NotImplementedError

    @abstractmethod
    def write(
        self, remote: RemoteBuffers, src: BufferKey, dst: BufferKey, ranges: np.ndarray
    ) -> Handle:
        """Start copying local buffer `src` into peer buffer `dst`. Each row of
        `ranges` is (src_offset, dst_offset, length).

        Transfer errors, including a rejected post, are reported as FAILED on
        the returned handle, so work already posted stays tracked by it. Ranges
        of one write, and separate writes, complete in no particular order."""

    @abstractmethod
    def status(self, handle: Handle) -> Result:
        """Return the current result without blocking; the protocol controls polling."""

    @abstractmethod
    def close(self) -> None:
        """Release this Transport's registrations and owned backend resources.
        Borrowed engines stay running; backing memory stays with the caller."""

    def write_all(self, remote: RemoteBuffers, copies: Sequence[Copy]) -> list[Handle]:
        """Submit copies across buffers. A backend may combine them into one handle."""
        return [self.write(remote, c.src, c.dst, c.ranges) for c in copies]

    def status_all(self, handles: Sequence[Handle]) -> Result:
        """FAILED as soon as any handle is FAILED, DONE when every handle is
        DONE, PENDING otherwise. FAILED may leave other handles PENDING;
        it does not establish that the buffers are safe to release."""
        results = [self.status(h) for h in handles]
        for r in results:
            if r.status is Status.FAILED:
                return r
        if all(r.status is Status.DONE for r in results):
            return Result(Status.DONE)
        return Result(Status.PENDING)
