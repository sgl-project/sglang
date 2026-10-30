from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from mori.cpp import TransferStatus
from mori.io import EngineDesc, MemoryDesc, MemoryLocationType, StatusCode

from sglang.srt.disaggregation.transport.base import (
    BufferKey,
    Copy,
    Handle,
    RemoteBuffers,
    Result,
    Status,
    Transport,
)

_DONE = Result(Status.DONE)
_PENDING = Result(Status.PENDING)


class MoriRemoteBuffers(RemoteBuffers):
    def __init__(self, descs: Dict[BufferKey, Optional[MemoryDesc]]):
        # Mori addresses a peer write by the peer's MemoryDesc and an offset in it.
        self.descs = descs


class MoriHandle(Handle):
    def __init__(self, transfer_status: TransferStatus):
        self.transfer_status = transfer_status


class MoriTransport(Transport):
    """Transport over a Mori IOEngine dedicated to one KVManager.

    write() and write_all() post without waiting and status() does not block;
    wait_all() and wait() block in the engine.
    """

    def __init__(self, *, engine):
        # One engine per KVManager, owned here: close() drops it.
        self._engine = engine
        self._engine_desc = engine.get_engine_desc()
        # The MemoryDesc each key is registered with; None for a zero-length key.
        self._registrations: Dict[BufferKey, Optional[MemoryDesc]] = {}

    @property
    def engine_key(self) -> str:
        return self._engine_desc.key

    @property
    def engine_desc_blob(self) -> bytes:
        return self._engine_desc.pack()

    def register(self, key: BufferKey, ptr: int, length: int, device: int) -> None:
        """Register one engine region per key: the registration message carries
        a MemoryDesc for every buffer."""
        if length == 0:
            # Elided DSA index-K; ibv_reg_mr would abort on size 0.
            # Keep the slot so descs stay index-aligned.
            self._registrations[key] = None
            return
        self._registrations[key] = self._engine.register_memory(
            ptr,
            length,
            device,
            MemoryLocationType.CPU if device < 0 else MemoryLocationType.GPU,
        )

    def deregister(self, keys: Sequence[BufferKey]) -> None:
        descs = [
            self._registrations.pop(key) for key in keys if key in self._registrations
        ]
        for desc in descs:
            if desc is not None:
                self._engine.deregister_memory(desc)

    def mem_desc_blobs(self, keys: Sequence[BufferKey]) -> List[Optional[bytes]]:
        """Packed MemoryDesc of each key; None for a zero-length key."""
        descs = [self._registrations[key] for key in keys]
        return [None if desc is None else desc.pack() for desc in descs]

    def import_registration(
        self,
        *,
        engine_desc: EngineDesc,
        descs: Dict[BufferKey, Optional[MemoryDesc]],
    ) -> MoriRemoteBuffers:
        """Register the peer's engine and keep the MemoryDesc of each peer key."""
        self._engine.register_remote_engine(engine_desc)
        return MoriRemoteBuffers(descs=descs)

    def write(
        self,
        remote: MoriRemoteBuffers,
        src: BufferKey,
        dst: BufferKey,
        ranges: np.ndarray,
    ) -> MoriHandle:
        return self.write_all(
            remote=remote, copies=[Copy(src=src, dst=dst, ranges=ranges)]
        )[0]

    def write_all(
        self, remote: MoriRemoteBuffers, copies: Sequence[Copy]
    ) -> List[MoriHandle]:
        """Post the copies in one batch_write, with a transfer uid and a handle
        per copy."""
        uids = [self._engine.allocate_transfer_uid() for _ in copies]
        statuses = self._engine.batch_write(
            [self._registrations[c.src] for c in copies],
            [c.ranges[:, 0].tolist() for c in copies],
            [remote.descs[c.dst] for c in copies],
            [c.ranges[:, 1].tolist() for c in copies],
            [c.ranges[:, 2].tolist() for c in copies],
            uids,
        )
        return [MoriHandle(status) for status in statuses]

    def status(self, handle: MoriHandle) -> Result:
        status = handle.transfer_status
        if status.InProgress():
            return _PENDING
        if status.Failed():
            return Result(
                Status.FAILED, reason=f"KV transfer failed: {status.Message()}"
            )
        return _DONE

    def wait_all(self, handles: Sequence[MoriHandle], *, timeout_ms: int) -> Result:
        """Block up to timeout_ms for every handle to settle. As with
        status_all(), FAILED can leave other handles PENDING."""
        statuses = [handle.transfer_status for handle in handles]
        rc = self._engine.wait_all(statuses, timeout_ms=timeout_ms)
        if rc == StatusCode.IN_PROGRESS:
            return _PENDING
        if rc == StatusCode.SUCCESS:
            return _DONE
        return Result(
            Status.FAILED, reason=self._collect_transfer_failure_reason(statuses)
        )

    @staticmethod
    def _collect_transfer_failure_reason(statuses: List[TransferStatus]) -> str:
        for status in statuses:
            if status.Failed():
                return f"KV transfer failed: {status.Message()}"
        return "KV transfer failed due to unknown reason"

    def wait(self, handles: Sequence[MoriHandle]) -> None:
        """Block until no handle is PENDING."""
        for handle in handles:
            if handle.transfer_status.InProgress():
                handle.transfer_status.Wait()

    def close(self) -> None:
        try:
            for desc in self._registrations.values():
                if desc is None:
                    continue
                try:
                    self._engine.deregister_memory(desc)
                except Exception:
                    pass
        finally:
            self._registrations.clear()
            self._engine = None
