from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import msgspec
import numpy as np

from sglang.srt.disaggregation.transport.base import (
    BufferKey,
    Copy,
    Handle,
    RemoteBuffers,
    Result,
    Status,
    Transport,
)

logger = logging.getLogger(__name__)


class Registration(msgspec.Struct, frozen=True):
    key: BufferKey
    ptr: int
    length: int
    # GPU index, or -1 for page-locked host memory.
    device: int


class MooncakeRemoteBuffers(RemoteBuffers):
    def __init__(self, session_id: str, bases: Dict[BufferKey, int]):
        self.session_id = session_id
        # Mooncake addresses a peer write by its absolute target address.
        self.bases = bases


class MooncakeHandle(Handle):
    def __init__(self, result: Result):
        self.result = result


class MooncakeTransport(Transport):
    """Transport over the process-wide Mooncake engine (or its Ascend subclass).

    write() and write_all() block until batch_transfer_sync returns, then report
    DONE or FAILED; the binding's async batch status call (v0.3.13.post1) also
    blocks. A timeout can leave transfers running after FAILED, for T_err of
    about 5 s (ADR #44186). The protocol's worker threads supply concurrency.
    """

    def __init__(self, *, engine):
        # Shared with HiCache, the encoder and elastic EP: close() never destroys it.
        self._engine = engine
        # (ptr, length) each key is registered with. Keys sharing a region share
        # one engine registration: double registration fails in the engine.
        self._registrations: Dict[BufferKey, Tuple[int, int]] = {}

    @property
    def session_id(self) -> str:
        return self._engine.get_session_id()

    def register(self, key: BufferKey, ptr: int, length: int, device: int) -> None:
        self.register_all(
            [Registration(key=key, ptr=ptr, length=length, device=device)]
        )

    def register_all(self, registrations: Sequence[Registration]) -> None:
        """Register in one batch_register call: Ascend's MemFabric merges
        overlapping 2 MiB aligned ranges only within one batch."""
        self.deregister(
            [
                r.key
                for r in registrations
                if self._registrations.get(r.key, (r.ptr, r.length))
                != (r.ptr, r.length)
            ]
        )
        registered_regions = set(self._registrations.values())
        new_regions = []
        for r in registrations:
            if r.length == 0:
                # Elided DSA index-K layers keep a 0-row buffer for alignment;
                # the engine rejects zero-length regions and nothing writes them.
                continue
            region = (r.ptr, r.length)
            if region not in registered_regions:
                registered_regions.add(region)
                new_regions.append(region)
            self._registrations[r.key] = region
        if new_regions:
            ptrs, lens = zip(*new_regions)
            self._engine.batch_register(list(ptrs), list(lens))

    def deregister(self, keys: Sequence[BufferKey]) -> None:
        """Unregister `keys`, preserving regions still held by other keys."""
        released_regions = {
            self._registrations.pop(key) for key in keys if key in self._registrations
        }
        unused_regions = released_regions - set(self._registrations.values())
        if unused_regions:
            self._engine.batch_deregister([ptr for ptr, _ in unused_regions])

    def import_registration(
        self, *, session_id: str, bases: Dict[BufferKey, int]
    ) -> MooncakeRemoteBuffers:
        """Import the current registration fields; the first write triggers
        the engine handshake with this session."""
        return MooncakeRemoteBuffers(session_id=session_id, bases=bases)

    def write(
        self,
        remote: MooncakeRemoteBuffers,
        src: BufferKey,
        dst: BufferKey,
        ranges: np.ndarray,
    ) -> MooncakeHandle:
        return self.write_all(remote, [Copy(src=src, dst=dst, ranges=ranges)])[0]

    def write_all(
        self, remote: MooncakeRemoteBuffers, copies: Sequence[Copy]
    ) -> List[MooncakeHandle]:
        """Copy all ranges in one blocking batch and return its result handle.
        An empty copy list returns no handles."""
        if not copies:
            return []
        unknown = [c.src for c in copies if c.src not in self._registrations] + [
            c.dst for c in copies if c.dst not in remote.bases
        ]
        if unknown:
            return [
                MooncakeHandle(
                    Result(Status.FAILED, reason=f"unregistered buffers {unknown}")
                )
            ]
        # uint64 throughout: XPU pointers can set bit 63, and numpy promotes
        # uint64 mixed with int64 to float64.
        ranges = [c.ranges.astype(np.uint64, copy=False) for c in copies]
        src_addrs = [
            np.uint64(self._registrations[c.src][0]) + r[:, 0]
            for c, r in zip(copies, ranges)
        ]
        dst_addrs = [
            np.uint64(remote.bases[c.dst]) + r[:, 1] for c, r in zip(copies, ranges)
        ]
        lengths = [r[:, 2] for r in ranges]
        ret = self._engine.batch_transfer_sync(
            remote.session_id,
            np.concatenate(src_addrs).tolist(),
            np.concatenate(dst_addrs).tolist(),
            np.concatenate(lengths).tolist(),
        )
        if ret == 0:
            return [MooncakeHandle(Result(Status.DONE))]
        # A timeout can return -1 while the batch still accesses memory.
        return [
            MooncakeHandle(
                Result(Status.FAILED, reason=f"batch_transfer_sync returned {ret}")
            )
        ]

    def status(self, handle: MooncakeHandle) -> Result:
        return handle.result

    def send_probe(self, session_id: str) -> int:
        """Lifts the engine's blacklist of a failed session when it answers."""
        return self._engine.send_probe(session_id)

    def close(self) -> None:
        self.deregister(list(self._registrations))
