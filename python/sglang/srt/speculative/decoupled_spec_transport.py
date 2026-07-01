"""Pluggable control-plane transport for the decoupled-spec IPC mesh.

The decoupled drafter and verifier talk over a small message mesh. This module
hides the wire behind one interface so the rest of the code never touches ZMQ
directly, which buys:

* a real **ZMQ** backend for production (``ZmqTransport``);
* a **fake, in-process** backend for tests (``FakeTransport`` + ``FakeTransportMesh``):
  the feature is two GPU processes talking over the network, which is hard to
  test in CI. The fake backend runs every endpoint inside one process over
  in-memory queues, so the protocol / ``DraftTailBuffer`` / scheduler logic can
  be driven deterministically, and a test can inject a dead peer on demand.

Addressing mirrors the real mesh exactly (see ``DecoupledSpecIpcConfig``): each
endpoint **binds** one inbound channel (its own ``bind_endpoint``) and
**connects** to N peers' inbound channels (``connect_endpoints``, ordered by
peer rank), so ``connect_endpoints[k]`` is peer-rank-``k``'s ``bind_endpoint``.

The transport is deliberately **message-type agnostic**: it ships
``DraftMeshMessage`` envelopes and nothing more. Validation, rank filtering, and
routing to the inbox / ``DraftTailBuffer`` stay in the caller (the IPC threads),
not here.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from typing import Optional

from sglang.srt.speculative.decoupled_spec_io import DraftMeshMessage

logger = logging.getLogger(__name__)


class TransportClosed(Exception):
    """Raised by a transport op once the underlying transport is torn down.

    Lets the IPC thread loops break cleanly without depending on a specific
    backend's shutdown exception (e.g. ZMQ's ``ContextTerminated``).
    """


class DecoupledSpecTransportKind(str, Enum):
    ZMQ = "zmq"
    FAKE = "fake"


class BaseDecoupledSpecTransport(ABC):
    """Symmetric control-plane transport for one mesh endpoint.

    One instance per IPC thread. ``send`` pushes a message to a peer by rank;
    ``try_recv`` non-blockingly pops one inbound message. Backends bind one
    inbound channel and connect N outbound channels at ``start()``.
    """

    @abstractmethod
    def start(self) -> None:
        """Open the inbound channel and the outbound channels to all peers."""

    @abstractmethod
    def send(self, dst_rank: int, message: DraftMeshMessage) -> None:
        """Push ``message`` to peer ``dst_rank`` (its inbound channel)."""

    @abstractmethod
    def try_recv(self) -> Optional[DraftMeshMessage]:
        """Pop one inbound message, or return None if none is available.

        Raises ``TransportClosed`` if the transport has been torn down.
        """

    @abstractmethod
    def wait_for_input(self, timeout_s: float) -> bool:
        """Block up to ``timeout_s`` for inbound input to (maybe) be available.

        Best-effort idle hint to avoid busy-spinning; the caller must still use
        ``try_recv`` (which may still return None). Returns True if input is
        likely available (or the endpoint is gone), False on timeout.
        """

    @abstractmethod
    def close(self) -> None:
        """Tear down all channels. Idempotent."""

    def is_peer_alive(self, rank: int) -> bool:
        """Health hook: whether peer ``rank`` is believed reachable.

        1c only wires this for the fake backend (dead-peer injection); the real
        liveness policy and the verifier's reaction land in phase 5c. Defaults
        to optimistic True.
        """
        return True


class ZmqTransport(BaseDecoupledSpecTransport):
    """Real ZMQ PUSH/PULL backend.

    Binds one inbound PULL on ``bind_endpoint`` and connects one outbound PUSH
    per peer in ``connect_endpoints``. ``zmq`` is imported lazily (only on
    ``start`` / receive) so this module stays importable without pyzmq.
    """

    def __init__(
        self,
        *,
        bind_endpoint: str,
        connect_endpoints: tuple[str, ...],
        context=None,
    ) -> None:
        self._bind_endpoint = bind_endpoint
        self._connect_endpoints = tuple(connect_endpoints)
        self._context = context
        self._recv_socket = None
        self._send_sockets: dict[int, object] = {}
        self._started = False
        self._is_closed = False

    def start(self) -> None:
        if self._started:
            return
        import zmq

        from sglang.srt.utils.network import get_zmq_socket

        if self._context is None:
            raise RuntimeError("ZmqTransport requires a zmq.Context to start.")
        self._recv_socket = get_zmq_socket(
            self._context, zmq.PULL, self._bind_endpoint, True
        )
        self._send_sockets = {
            rank: get_zmq_socket(self._context, zmq.PUSH, endpoint, False)
            for rank, endpoint in enumerate(self._connect_endpoints)
        }
        self._is_closed = False
        self._started = True

    def send(self, dst_rank: int, message: DraftMeshMessage) -> None:
        import zmq

        if self._is_closed:
            raise TransportClosed()
        socket = self._send_sockets.get(int(dst_rank))
        if socket is None:
            raise RuntimeError(
                f"ZmqTransport has no outbound channel for dst_rank={dst_rank} "
                f"(connect_endpoints has {len(self._connect_endpoints)} peers)."
            )
        try:
            socket.send_pyobj(message)
        except zmq.error.ContextTerminated as exc:
            raise TransportClosed() from exc

    def try_recv(self) -> Optional[DraftMeshMessage]:
        import zmq

        if self._is_closed:
            raise TransportClosed()
        if self._recv_socket is None:
            return None
        try:
            return self._recv_socket.recv_pyobj(zmq.NOBLOCK)
        except zmq.error.ContextTerminated as exc:
            raise TransportClosed() from exc
        except zmq.ZMQError:
            # EAGAIN / no message currently available.
            return None

    def wait_for_input(self, timeout_s: float) -> bool:
        import zmq

        if self._is_closed:
            raise TransportClosed()
        if self._recv_socket is None:
            return False
        try:
            return bool(self._recv_socket.poll(timeout=int(timeout_s * 1000)))
        except zmq.error.ContextTerminated as exc:
            raise TransportClosed() from exc

    def close(self) -> None:
        if self._recv_socket is not None:
            self._recv_socket.close(linger=0)
            self._recv_socket = None
        for socket in self._send_sockets.values():
            socket.close(linger=0)
        self._send_sockets = {}
        self._started = False
        self._is_closed = True


class FakeTransportMesh:
    """In-process switchboard shared by the ``FakeTransport`` endpoints.

    One mesh per test/process. It maps each endpoint string to an in-memory
    inbound queue and routes ``deliver`` to the target endpoint's queue, exactly
    as the real mesh would route to a peer's bound socket. A ``threading``
    condition makes it safe whether the IPC threads run on real background
    threads or are driven step-by-step from a single test thread.
    """

    def __init__(self) -> None:
        self._inboxes: dict[str, deque] = {}
        self._dead: set[str] = set()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def register(self, endpoint: str) -> None:
        with self._cond:
            self._inboxes.setdefault(endpoint, deque())

    def unregister(self, endpoint: str) -> None:
        with self._cond:
            self._inboxes.pop(endpoint, None)
            self._cond.notify_all()

    def deliver(self, endpoint: str, message: DraftMeshMessage) -> None:
        with self._cond:
            if endpoint in self._dead:
                # Drop: the target peer is dead. 5c decides how the sender reacts.
                logger.debug(
                    "FakeTransportMesh dropping message to dead endpoint %s", endpoint
                )
                return
            self._inboxes.setdefault(endpoint, deque()).append(message)
            self._cond.notify_all()

    def take(self, endpoint: str) -> Optional[DraftMeshMessage]:
        with self._cond:
            queue = self._inboxes.get(endpoint)
            if queue:
                return queue.popleft()
            return None

    def wait(self, endpoint: str, timeout_s: float) -> bool:
        with self._cond:
            if self._inboxes.get(endpoint) or endpoint in self._dead:
                return True
            self._cond.wait(timeout=timeout_s)
            return bool(self._inboxes.get(endpoint)) or endpoint in self._dead

    def kill(self, endpoint: str) -> None:
        """Mark ``endpoint`` dead: future deliveries to it are dropped."""
        with self._cond:
            self._dead.add(endpoint)
            self._cond.notify_all()

    def is_dead(self, endpoint: str) -> bool:
        with self._lock:
            return endpoint in self._dead


class FakeTransport(BaseDecoupledSpecTransport):
    """In-process backend over a shared ``FakeTransportMesh`` (no sockets)."""

    def __init__(
        self,
        *,
        mesh: FakeTransportMesh,
        bind_endpoint: str,
        connect_endpoints: tuple[str, ...],
    ) -> None:
        self._mesh = mesh
        self._bind_endpoint = bind_endpoint
        self._connect_endpoints = tuple(connect_endpoints)

    def start(self) -> None:
        self._mesh.register(self._bind_endpoint)

    def send(self, dst_rank: int, message: DraftMeshMessage) -> None:
        rank = int(dst_rank)
        if not 0 <= rank < len(self._connect_endpoints):
            raise RuntimeError(
                f"FakeTransport has no outbound channel for dst_rank={dst_rank} "
                f"(connect_endpoints has {len(self._connect_endpoints)} peers)."
            )
        self._mesh.deliver(self._connect_endpoints[rank], message)

    def try_recv(self) -> Optional[DraftMeshMessage]:
        return self._mesh.take(self._bind_endpoint)

    def wait_for_input(self, timeout_s: float) -> bool:
        return self._mesh.wait(self._bind_endpoint, timeout_s)

    def close(self) -> None:
        self._mesh.unregister(self._bind_endpoint)

    def is_peer_alive(self, rank: int) -> bool:
        return not self._mesh.is_dead(self._connect_endpoints[int(rank)])


def build_transport(
    *,
    kind: DecoupledSpecTransportKind | str,
    bind_endpoint: str,
    connect_endpoints,
    context=None,
    mesh: Optional[FakeTransportMesh] = None,
) -> BaseDecoupledSpecTransport:
    """Construct a transport for the given ``kind``.

    ``kind`` accepts either the enum or its string value (e.g. the
    ``DecoupledSpecIpcConfig.transport_kind`` str) and is coerced/validated here
    via the str-enum constructor, so callers can pass the config string directly.

    The ``zmq`` import for the ZMQ backend is deferred to ``ZmqTransport.start``,
    so callers selecting the fake backend never need pyzmq installed.
    """
    kind = DecoupledSpecTransportKind(kind)
    connect_endpoints = tuple(connect_endpoints)
    if kind == DecoupledSpecTransportKind.ZMQ:
        return ZmqTransport(
            bind_endpoint=bind_endpoint,
            connect_endpoints=connect_endpoints,
            context=context,
        )
    if kind == DecoupledSpecTransportKind.FAKE:
        if mesh is None:
            raise ValueError("FAKE transport requires a FakeTransportMesh.")
        return FakeTransport(
            mesh=mesh,
            bind_endpoint=bind_endpoint,
            connect_endpoints=connect_endpoints,
        )
    raise ValueError(f"Unknown decoupled-spec transport kind: {kind}")
