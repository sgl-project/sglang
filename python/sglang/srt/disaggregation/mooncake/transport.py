from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import msgspec
import numpy as np

from sglang.srt.disaggregation.transport.base import (
    BufferExport,
    BufferKey,
    Copy,
    Handle,
    RemoteBuffers,
    Result,
    Status,
    Transport,
)
from sglang.srt.distributed.device_communicators.kv_transfer_engine import (
    TransferIntent,
)


class Registration(msgspec.Struct, frozen=True):
    key: BufferKey
    ptr: int
    length: int
    # GPU index, or -1 for page-locked host memory.
    device: int


def _joined(parts: List[np.ndarray]) -> np.ndarray:
    """One array of `parts`, without a copy when there is a single part."""
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


class MooncakeBufferExport(BufferExport):
    def __init__(self, session_id: str, bases: Sequence[int]):
        self.session_id = session_id
        # One base address per exported key, in export order.
        self.bases = tuple(bases)


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
            region = (r.ptr, r.length)
            # Elided DSA index-K layers keep a 0-row buffer for alignment; the
            # engine rejects zero-length regions, but the key is still exported.
            if r.length and region not in registered_regions:
                registered_regions.add(region)
                new_regions.append(region)
        if new_regions:
            ptrs, lens = zip(*new_regions)
            self._engine.batch_register(list(ptrs), list(lens))
        # Only after the engine accepted them: a failed batch leaves no key behind.
        for r in registrations:
            self._registrations[r.key] = (r.ptr, r.length)

    def deregister(self, keys: Sequence[BufferKey]) -> None:
        """Unregister `keys`, preserving regions still held by other keys."""
        released_regions = {
            self._registrations.pop(key) for key in keys if key in self._registrations
        }
        unused_regions = {
            region
            for region in released_regions - set(self._registrations.values())
            if region[1]
        }
        if unused_regions:
            self._engine.batch_deregister([ptr for ptr, _ in unused_regions])

    def export_buffers(self, keys: Sequence[BufferKey]) -> MooncakeBufferExport:
        return MooncakeBufferExport(
            session_id=self.session_id,
            bases=[self._registrations[key][0] for key in keys],
        )

    def import_buffers(
        self, desc: MooncakeBufferExport, keys: Sequence[BufferKey]
    ) -> MooncakeRemoteBuffers:
        """The first write to the result triggers the engine handshake with
        the peer's session."""
        return MooncakeRemoteBuffers(
            session_id=desc.session_id, bases=dict(zip(keys, desc.bases, strict=True))
        )

    def release_remote(self, remote: MooncakeRemoteBuffers) -> None:
        """Nothing to drop: import_buffers() leaves no engine state."""

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
        # uint64: XPU pointers can set bit 63.
        src_addrs = [
            np.uint64(self._registrations[c.src][0]) + c.ranges[:, 0] for c in copies
        ]
        dst_addrs = [np.uint64(remote.bases[c.dst]) + c.ranges[:, 1] for c in copies]
        lengths = [c.ranges[:, 2] for c in copies]
        ret = self._engine.batch_transfer_sync(
            remote.session_id,
            _joined(src_addrs).tolist(),
            _joined(dst_addrs).tolist(),
            _joined(lengths).tolist(),
            # Decode waits on every PD write, so all of them ride the foreground lane.
            TransferIntent.FOREGROUND,
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

    def send_probe(self, remote: MooncakeRemoteBuffers) -> int:
        """Lifts the engine's blacklist of a failed peer when it answers."""
        return self._engine.send_probe(remote.session_id)

    def close(self) -> None:
        self.deregister(list(self._registrations))
