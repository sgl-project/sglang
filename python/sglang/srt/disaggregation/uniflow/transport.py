from __future__ import annotations

import base64
import importlib
import logging
import threading
import time
from collections import deque
from types import ModuleType
from typing import Any, Deque, Dict, List, Optional, Sequence

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

# One per prefill connection, reconnects after a failed put included. Decode
# cannot tell a dead connection from a live one, so past the bound the oldest
# is dropped, which closes it.
MAX_ACCEPTED_CONNECTIONS = 1024

_DONE = Result(Status.DONE)
_PENDING = Result(Status.PENDING)


class Registration(msgspec.Struct, frozen=True):
    key: BufferKey
    ptr: int
    length: int
    # GPU index, or -1 for page-locked host memory.
    device: int


def import_uniflow_core() -> ModuleType:
    """Import the uniflow._core binding, naming the backend flag on failure."""
    try:
        return importlib.import_module("uniflow._core")
    except ImportError as e:
        raise ImportError(
            "--disaggregation-transfer-backend uniflow requires the uniflow "
            "package with its compiled _core extension; importing "
            f"uniflow._core failed: {e}"
        ) from e


def _value(result: Any, action: str) -> Any:
    if not result.has_value():
        error = result.error() if result.has_error() else "no value"
        raise RuntimeError(f"UniFlow {action} failed: {error}")
    return result.value()


class UniflowRemoteBuffers(RemoteBuffers):
    def __init__(
        self, *, agent_id: str, connection: Any, segments: Dict[BufferKey, Any]
    ):
        self.agent_id = agent_id
        # Every write to this peer goes through this connection.
        self.connection = connection
        # UniFlow addresses a peer write by the imported segment and an offset in it.
        self.segments = segments


class UniflowHandle(Handle):
    def __init__(self, *, future: Any = None, result: Optional[Result] = None):
        self.future = future
        # Set once final; a put the binding rejected starts final.
        self.result = result


class UniflowTransport(Transport):
    """Transport over a UniFlow agent made by create().

    write() and write_all() post one put batch without waiting; status() does
    not block, and wait_all() blocks on the batches up to a deadline. UniFlow
    picks one transport tier per put batch, using a tier only if every request
    has a handle for it on both sides, so callers keep device and host buffers
    in separate write_all() calls.
    """

    def __init__(self, *, agent: Any, core: ModuleType):
        # One agent per KVManager, owned here.
        self._agent = agent
        self._core = core
        self.agent_id: str = _value(agent.get_unique_id(), "get unique id")
        # The registered segment of each key: destroying it deregisters the
        # memory, so dropping the last reference is the deregistration.
        self._segments: Dict[BufferKey, Any] = {}
        self._export_ids: Dict[BufferKey, str] = {}
        # Connections this side opened, shut down by disconnect() or close().
        self._remotes: List[UniflowRemoteBuffers] = []
        # Connections peers opened to write here; kept alive, bounded.
        self._accepted: Deque[Any] = deque(maxlen=MAX_ACCEPTED_CONNECTIONS)
        self._accepted_lock = threading.Lock()

    @classmethod
    def create(
        cls, core: ModuleType, *, device_id: int, name: str, host: str
    ) -> UniflowTransport:
        config = core.UniflowAgentConfig(
            device_id=device_id,
            name=name,
            # UniFlow splits host:port on the last colon and rejects brackets,
            # so an IPv6 host stays unbracketed.
            listen_address=f"{host}:0",
        )
        return cls(agent=core.UniflowAgent(config), core=core)

    def register(self, key: BufferKey, ptr: int, length: int, device: int) -> None:
        self.register_all(
            [Registration(key=key, ptr=ptr, length=length, device=device)]
        )

    def register_all(self, registrations: Sequence[Registration]) -> None:
        """Register one segment per key: the REGISTER message carries an export
        id for every buffer. Host memory (device -1) is UniFlow DRAM."""
        self.deregister([r.key for r in registrations])
        for r in registrations:
            if r.length == 0:
                # An elided DSA index-K entry keeps its position; nothing writes it.
                self._export_ids[r.key] = ""
                continue
            memory_type = (
                self._core.MemoryType.DRAM
                if r.device < 0
                else self._core.MemoryType.VRAM
            )
            segment = _value(
                self._agent.register_segment(
                    self._core.Segment(r.ptr, r.length, memory_type, r.device)
                ),
                f"register segment {r.key!r}",
            )
            export_id = _value(segment.export_id(), f"export segment {r.key!r}")
            self._segments[r.key] = segment
            self._export_ids[r.key] = base64.b64encode(export_id).decode("ascii")

    def deregister(self, keys: Sequence[BufferKey]) -> None:
        for key in keys:
            self._segments.pop(key, None)
            self._export_ids.pop(key, None)

    def export_ids(self, keys: Sequence[BufferKey]) -> List[str]:
        """Export id of each key, base64; empty for a zero-length key."""
        return [self._export_ids[key] for key in keys]

    def import_registration(
        self, *, agent_id: str, export_ids: Dict[BufferKey, str]
    ) -> UniflowRemoteBuffers:
        """Import the peer's segments, then connect to its agent. An empty
        export id imports nothing."""
        segments = {
            key: _value(
                self._agent.import_segment(base64.b64decode(export_id)),
                f"import segment {key!r}",
            )
            for key, export_id in export_ids.items()
            if export_id
        }
        connection = _value(self._agent.connect(agent_id), "connect peer")
        remote = UniflowRemoteBuffers(
            agent_id=agent_id, connection=connection, segments=segments
        )
        self._remotes.append(remote)
        return remote

    def disconnect(self, remote: UniflowRemoteBuffers) -> None:
        """Shut the peer's connection down; raises what shutdown() raised.
        Only the RDMA tier waits there for in-flight puts."""
        if remote in self._remotes:
            self._remotes.remove(remote)
        remote.connection.shutdown()

    def write(
        self,
        remote: UniflowRemoteBuffers,
        src: BufferKey,
        dst: BufferKey,
        ranges: np.ndarray,
    ) -> UniflowHandle:
        handles = self.write_all(remote, [Copy(src=src, dst=dst, ranges=ranges)])
        return handles[0] if handles else UniflowHandle(result=_DONE)

    def write_all(
        self, remote: UniflowRemoteBuffers, copies: Sequence[Copy]
    ) -> List[UniflowHandle]:
        """Post every range in one put batch and return its handle. Copies
        without ranges post nothing and return no handles."""
        requests = [
            self._core.TransferRequest(
                self._segments[c.src].span(src_offset, length),
                remote.segments[c.dst].span(dst_offset, length),
            )
            for c in copies
            for src_offset, dst_offset, length in c.ranges.tolist()
        ]
        if not requests:
            return []
        try:
            future = remote.connection.put(requests=requests)
        except Exception as e:
            # The binding may have left part of the batch on the wire.
            return [UniflowHandle(result=Result(Status.FAILED, reason=str(e)))]
        return [UniflowHandle(future=future)]

    def status(self, handle: UniflowHandle) -> Result:
        if handle.result is not None:
            return handle.result
        try:
            # wait_for(0) polls without blocking.
            if not handle.future.wait_for(timeout_ms=0):
                return _PENDING
            return self._settle(handle)
        except Exception as e:
            handle.result = Result(Status.FAILED, reason=str(e))
            return handle.result

    def wait_all(self, handles: Sequence[UniflowHandle], *, timeout_ms: int) -> Result:
        """Block up to timeout_ms for every handle to settle; PENDING if the
        deadline passes first. As with status_all(), FAILED can leave other
        handles PENDING."""
        deadline = time.monotonic() + timeout_ms / 1000
        wait_ms = timeout_ms
        for handle in handles:
            if handle.result is None:
                try:
                    if not handle.future.wait_for(timeout_ms=wait_ms):
                        return _PENDING
                    self._settle(handle)
                except Exception as e:
                    handle.result = Result(Status.FAILED, reason=str(e))
            if handle.result.status is Status.FAILED:
                return handle.result
            wait_ms = max(0, int((deadline - time.monotonic()) * 1000))
        return _DONE

    @staticmethod
    def _settle(handle: UniflowHandle) -> Result:
        outcome = handle.future.get()
        if outcome.has_error():
            handle.result = Result(Status.FAILED, reason=str(outcome.error()))
        else:
            handle.result = _DONE
        return handle.result

    def start_accepting(self) -> threading.Thread:
        """Accept peer connections on a daemon thread and keep them alive, up
        to MAX_ACCEPTED_CONNECTIONS: peers write into this side over them."""

        def accept_loop() -> None:
            last_error: Optional[str] = None
            while True:
                try:
                    result = self._agent.accept()
                    if result.has_error():
                        # Warn once per distinct error; a broken listener
                        # repeats the same one every retry.
                        error = str(result.error())
                        level = (
                            logging.DEBUG if error == last_error else logging.WARNING
                        )
                        logger.log(level, "UniFlow accept failed: %s", error)
                        last_error = error
                        time.sleep(0.1)
                        continue
                    last_error = None
                    with self._accepted_lock:
                        self._accepted.append(result.value())
                except Exception:
                    # Keep the accept thread alive across unexpected errors.
                    logger.exception("UniFlow accept loop iteration failed")
                    time.sleep(0.1)

        thread = threading.Thread(target=accept_loop, daemon=True)
        thread.start()
        return thread

    def close(self) -> None:
        """Shut down the connections this side opened and drop every
        registration and accepted connection. The accept thread keeps
        running: the binding cannot interrupt accept()."""
        for remote in list(self._remotes):
            try:
                self.disconnect(remote)
            except Exception:
                logger.exception("UniFlow shutdown of peer %s failed", remote.agent_id)
        self._remotes.clear()
        self.deregister(list(self._export_ids))
        with self._accepted_lock:
            self._accepted.clear()
