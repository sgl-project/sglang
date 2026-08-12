"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
KV caching events
"""

import atexit
import enum
import logging
import queue
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from queue import Queue
from typing import Any, Callable, Optional, Union

import msgspec
import zmq
from pydantic import BaseModel, ConfigDict, ValidationError

logger = logging.getLogger(__name__)


def select_kv_publisher_dp_rank(
    attn_dp_size: int, attn_dp_rank: int, dp_rank: Optional[int]
) -> int:
    """Index used to offset this scheduler's KV-event publisher port.

    Each independent KV cache must publish on its own port so a consumer can
    subscribe per replica. There are always ``dp_size`` such publishers; which
    rank distinguishes them depends on the parallelism mode:

    - DP-attention (``attn_dp_size > 1``): each attention-DP rank owns a KV
      cache shard, so distinguish by ``attn_dp_rank``.
    - Pure DP (``attn_dp_size == 1``): every worker has ``attn_dp_rank == 0``,
      so distinguish by ``dp_rank`` (the data-parallel replica index).

    Both span ``0..dp_size-1``, matching the ``dp_size`` advertised in
    ``/server_info`` and the per-rank ports the router subscribes to.
    """
    if attn_dp_size > 1:
        return attn_dp_rank
    return dp_rank or 0


class EventBatch(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
):
    ts: float
    events: list[Any]
    attn_dp_rank: Optional[int] = None


class KVCacheEvent(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag=True,
):
    """Base class for all KV cache-related events"""


class StorageMedium(str, enum.Enum):
    """Storage tier for KV cache events."""

    GPU = "GPU"  # L1: device HBM
    CPU = "CPU_PINNED"  # L2: host pinned memory
    DISK = "DISK"  # L3: SSD / NVMe
    EXTERNAL = "EXTERNAL"  # L4: shared / remote pool (e.g. Mooncake)


class BlockStoredMetadata(msgspec.Struct, omit_defaults=True, gc=False):
    """Typed request metadata attached to a stored KV block."""

    cache_salt: str


class OffloadedState:
    """
    OffloadedState represents the state of a KV cache block offloaded to the hicache.

    - prefill_len (int): The length of the prefill part of the KV cache block.
    - inc_len (int): The length of the incremental part of the KV cache block.
    - last_hash (Optional[str]): The hash of the last token in the KV cache block.
    """

    def __init__(
        self, prefill_len: int, inc_len: int = 0, last_hash: Optional[str] = None
    ):
        self.prefill_len = prefill_len
        self.inc_len = inc_len
        self.last_hash = last_hash


class BlockStored(KVCacheEvent):
    block_hashes: list[int]
    parent_block_hash: Optional[int]
    token_ids: list[int]
    block_size: int
    lora_id: Optional[int]
    medium: Optional[str] = None


class BlockStoredWithMetadata(BlockStored, tag="BlockStored", kw_only=True):
    """BlockStored wire extension used only when typed metadata is present.

    A separate struct keeps unsalted events at their legacy array length; an
    optional field on BlockStored would still serialize a trailing null.
    """

    metadata: BlockStoredMetadata


class BlockRemoved(KVCacheEvent):
    block_hashes: list[int]
    medium: Optional[str] = None


class AllBlocksCleared(KVCacheEvent):
    pass


class KVEventBatch(EventBatch):
    # BlockStoredWithMetadata deliberately stays out of this tagged union.
    # Existing typed consumers decode its shared "BlockStored" tag as the base
    # type and ignore the trailing metadata; adding both types would give
    # msgspec duplicate tags and make the union invalid.
    events: list[Union[BlockStored, BlockRemoved, AllBlocksCleared]]


class KVSnapshotBlock(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
):
    """One logical block edge in a placement snapshot.

    ``block_hashes`` is intentionally compatible with ``BlockStored``. The
    current provider emits one hash per record, while the wire shape leaves
    room for a future provider to coalesce contiguous chains.
    """

    parent_block_hash: Optional[int]
    block_hashes: list[int]


class KVSnapshotHeader(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
):
    """Metadata for one DP replica's atomic placement cut.

    ``epoch`` identifies that replica publisher's lifecycle, not the lifetime
    of the enclosing server instance.
    """

    version: int
    epoch: str
    replica_rank: int
    resume_seq: int
    barrier_seq: int
    barrier_id: str
    record_count: int


@dataclass(frozen=True, slots=True)
class _KVPlacementSnapshot:
    header: KVSnapshotHeader
    blocks: list[KVSnapshotBlock]


@dataclass(frozen=True, slots=True)
class _SnapshotCaptureRequest:
    response: Queue


SNAPSHOT_PROTOCOL_VERSION = 1
SNAPSHOT_REQUEST = b"snapshot-v1"
SNAPSHOT_HEADER = b"header"
SNAPSHOT_CHUNK = b"chunk"
SNAPSHOT_END = b"end"
SNAPSHOT_ERROR = b"error"
SNAPSHOT_CHUNK_RECORDS = 4096
SNAPSHOT_CAPTURE_TIMEOUT = 5.0
# Bound how long the sole snapshot service thread may wait for one client to
# drain an outbound message. Without this, a client that stops reading can
# fill the ROUTER socket's send queue and prevent every later request (and
# clean shutdown) from making progress.
SNAPSHOT_SEND_TIMEOUT_MS = 500

_EPOCH_TOPIC_MARKER = b"\x00sgl-kv-epoch="
_SNAPSHOT_BARRIER_MARKER = b"\x00sgl-kv-snapshot="


def _live_topic(topic: bytes, epoch: str) -> bytes:
    return topic + _EPOCH_TOPIC_MARKER + epoch.encode("utf-8")


def _snapshot_barrier_topic(topic: bytes, epoch: str, barrier_id: str) -> bytes:
    return (
        _live_topic(topic, epoch)
        + _SNAPSHOT_BARRIER_MARKER
        + barrier_id.encode("utf-8")
    )


class EventPublisher(ABC):
    """
    Lightweight publisher for EventBatch batches with
    support for DP attention.

    In DP attention - each rank has its own Scheduler and
    KV cache instance in order to avoid duplicate events
    and ensure proper event attribution. In our implementation

    - Each DP rank has its own EventPublisher
    - Publishers annotate events with the dp rank
    - This allows consumers to distinguish events from different DP ranks
    """

    @abstractmethod
    def publish(self, events: EventBatch) -> None:
        """Emit events in order.

        Implementations should guarantee at-least-once delivery and
        monotonic ordering (e.g., via sequence numbers).
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the publisher."""


class NullEventPublisher(EventPublisher):
    """No-op implementation (default when disabled)."""

    def publish(self, events) -> None:
        return

    def shutdown(self) -> None:
        return


class ZmqEventPublisher(EventPublisher):
    """Reliable PUB/ROUTER publisher with an in-memory replay buffer.

    Spawns a separate thread to handle publishing from a queue.

    Parameters
    ----------
    endpoint:
        PUB address. Use ``tcp://*:5557`` to bind or ``tcp://host:5557`` to
        connect.
    replay_endpoint:
        Optional ROUTER address for replay requests. When given, subscribers can
        request missed batches by sending the starting sequence number as an
        8-byte big-endian integer.
    snapshot_endpoint:
        Optional ROUTER address for chunked placement snapshots. One endpoint is
        exposed per independently routable DP replica.
    buffer_steps:
        Number of past batches to keep for replay.
    hwm:
        ZeroMQ high-water-mark for PUB socket.
    max_queue_size:
        Maximum number of events to buffer in memory.
    topic:
        Topic to publish events to.
    epoch:
        Replica-lifecycle token. Snapshot-enabled live events carry it in the
        topic metadata. A fresh publisher-local UUID is generated by default,
        so rebuilding one DP replica does not reuse the server-wide instance
        identity shared by the other replicas.
    """

    SHUTDOWN_TIMEOUT: float = 1.0
    END_SEQ = (-1).to_bytes(8, "big", signed=True)

    def __init__(
        self,
        attn_dp_rank: int,
        endpoint: str = "tcp://*:5557",
        replay_endpoint: Optional[str] = None,
        snapshot_endpoint: Optional[str] = None,
        buffer_steps: int = 10_000,
        hwm: int = 100_000,
        max_queue_size: int = 100_000,
        topic: str = "",
        epoch: Optional[str] = None,
    ) -> None:
        # Storage
        self._event_queue = Queue[Optional[EventBatch]](maxsize=max_queue_size)
        self._buffer = deque[tuple[int, bytes]](maxlen=buffer_steps)

        # ZMQ sockets
        self._ctx = zmq.Context.instance()
        self._pub: Optional[zmq.Socket] = None
        self._replay: Optional[zmq.Socket] = None
        self._dp_rank = attn_dp_rank
        self._endpoint = self.offset_endpoint_port(endpoint, self._dp_rank)
        self._replay_endpoint = self.offset_endpoint_port(
            replay_endpoint, self._dp_rank
        )
        self._snapshot_endpoint = self.offset_endpoint_port(
            snapshot_endpoint, self._dp_rank
        )
        self._hwm = hwm
        self._socket_setup()

        # Payload
        self._next_seq = 0
        self._epoch = epoch or uuid.uuid4().hex
        self._topic_bytes = topic.encode("utf-8")
        # Preserve the exact legacy topic when snapshots are disabled. A
        # snapshot-capable publisher appends backward-compatible metadata to
        # the topic frame; SUB filters are prefix based, so existing consumers
        # configured for ``topic`` continue to receive the same payloads.
        self._live_topic_bytes = (
            _live_topic(self._topic_bytes, self._epoch)
            if self._snapshot_endpoint is not None
            else self._topic_bytes
        )
        # The publisher thread is the sole writer. Keeping the logical
        # placement mirror beside sequence assignment makes snapshot cuts
        # exact without traversing the scheduler's radix tree concurrently.
        self._snapshot_blocks: dict[int, KVSnapshotBlock] = {}
        # At most one capture may wait behind the publisher thread. This keeps
        # a stalled publisher from accumulating timed-out requests forever.
        self._snapshot_requests: Queue[_SnapshotCaptureRequest] = Queue(maxsize=1)

        # Thread
        self._running = True
        logger.info("Starting ZMQ publisher thread")

        self._thread = threading.Thread(
            target=self._publisher_thread, daemon=True, name="zmq-publisher"
        )
        self._thread.start()

        self._snapshot_thread: Optional[threading.Thread] = None
        self._snapshot_started = threading.Event()
        self._snapshot_start_error: Optional[BaseException] = None
        if self._snapshot_endpoint is not None:
            self._snapshot_thread = threading.Thread(
                target=self._snapshot_server_thread,
                daemon=True,
                name="zmq-kv-snapshot",
            )
            self._snapshot_thread.start()
            if not self._snapshot_started.wait(timeout=SNAPSHOT_CAPTURE_TIMEOUT):
                self.shutdown()
                raise RuntimeError(
                    f"Timed out starting KV snapshot endpoint {self._snapshot_endpoint}"
                )
            if self._snapshot_start_error is not None:
                startup_error = self._snapshot_start_error
                self.shutdown()
                raise RuntimeError(
                    f"Failed to start KV snapshot endpoint {self._snapshot_endpoint}"
                ) from startup_error

        atexit.register(self.shutdown)

    def publish(self, events: EventBatch) -> None:
        if not self._running:
            raise RuntimeError("Publisher is closed")
        if events.attn_dp_rank is None:
            events.attn_dp_rank = self._dp_rank
        self._event_queue.put(events)

    def shutdown(self) -> None:
        """Stop the publisher thread and clean up resources."""
        if not self._running:
            return
        self._running = False
        try:
            self._event_queue.put_nowait(None)
        except queue.Full:
            # The publisher loop also exits after draining a non-empty queue
            # when `_running` is false, so a full queue needs no sentinel.
            pass

        start = time.time()
        pending_items = True
        while pending_items and (time.time() - start < self.SHUTDOWN_TIMEOUT):
            pending_items = not self._event_queue.empty()
            if pending_items:
                time.sleep(0.1)

        if pending_items:
            logger.warning(
                "Warning: Queue still has %s items after %s seconds timeout",
                self._event_queue.qsize(),
                self.SHUTDOWN_TIMEOUT,
            )

        if self._thread.is_alive():
            self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)
        if self._snapshot_thread is not None and self._snapshot_thread.is_alive():
            self._snapshot_thread.join(timeout=self.SHUTDOWN_TIMEOUT)

        # Clean up ZMQ resources
        try:
            if self._pub is not None:
                self._pub.close(linger=0)
            if self._replay is not None:
                self._replay.close(linger=0)
        finally:
            pass  # Do not terminate context; other sockets may use it

    def _socket_setup(self) -> None:
        """Initialize sockets
        https://pyzmq.readthedocs.io/en/v19.0.0/morethanbindings.html#thread-safety
        """
        if self._pub is None:
            self._pub = self._ctx.socket(zmq.PUB)
            self._pub.set_hwm(self._hwm)
            # Heuristic: bind if wildcard / * present, else connect.
            # bind stable, connect volatile convention.
            # ``0.0.0.0`` is the IPv4 bind-all wildcard alongside ``*``
            # and ``::``; ``/server_info`` advertises it as a wildcard,
            # so the publisher must bind it for the advertised endpoint
            # to actually be listening.
            if (
                "*" in self._endpoint
                or "::" in self._endpoint
                or "0.0.0.0" in self._endpoint
                or self._endpoint.startswith("ipc://")
                or self._endpoint.startswith("inproc://")
            ):
                logger.debug(
                    f"ZmqEventPublisher socket publisher_endpoint bind to {self._endpoint}"
                )
                self._pub.bind(self._endpoint)
            else:
                self._pub.connect(self._endpoint)

        # Set up replay socket: use ROUTER
        # 1) handles multiple REQ clients (identities)
        # 2) lets us send back one request → many replies (streamed events)
        # 3) works in our non‑blocking poll loop alongside PUB
        if self._replay_endpoint is not None:
            self._replay = self._ctx.socket(zmq.ROUTER)
            logger.debug(
                f"ZmqEventPublisher socket replay_endpoint bind to {self._replay_endpoint}"
            )
            self._replay.bind(self._replay_endpoint)

    def _publisher_thread(self) -> None:
        """Background thread that processes the event queue."""
        self._pack = msgspec.msgpack.Encoder()

        assert self._pub is not None  # narrows type for mypy

        while self._running or self._event_queue.qsize() > 0:
            # --- replay (non-critical) ---------------------------------
            if self._replay is not None and self._replay.poll(0):
                try:
                    self._service_replay()
                except Exception as e:
                    logger.exception("Error in replay: %s", e)

            # Snapshot capture is kept on this thread so the copied placement
            # view, its resume sequence, and the live barrier form one serial
            # cut. The separate snapshot server thread handles encoding and
            # chunked network transfer after this short in-memory copy.
            self._service_snapshot_capture()

            # --- main queue (critical) ---------------------------------
            try:
                event = self._event_queue.get(timeout=0.1)
                if event is None:
                    break  # Sentinel received, exit thread
            except queue.Empty:
                continue

            seq = self._next_seq
            try:
                payload = self._pack.encode(event)
                seq_bytes = seq.to_bytes(8, "big")

                # Keep the replay journal complete even when live publication
                # fails. Once a sequence number is assigned and encoded, a
                # consumer must be able to recover that batch from `_buffer`
                # instead of being forced to fetch a full snapshot.
                self._buffer.append((seq, payload))
                self._pub.send_multipart((self._live_topic_bytes, seq_bytes, payload))

            except Exception as e:
                # Publishing failed;  back-off a bit to avoid a tight error loop
                logger.exception("Error in publisher thread: %s", e)
                time.sleep(0.1)
            finally:
                # The cache mutation already happened before the event entered
                # this queue. Advance the placement mirror even when PUB send
                # fails. The advanced sequence exposes a live-stream gap;
                # consumers can replay it from `_buffer` or fall back to a
                # snapshot if the replay window has already expired.
                self._apply_snapshot_events(event)
                self._next_seq = seq + 1
                self._event_queue.task_done()

    def _apply_snapshot_events(self, batch: EventBatch) -> None:
        for event in batch.events:
            if isinstance(event, BlockStored):
                parent = event.parent_block_hash
                for block_hash in event.block_hashes:
                    self._snapshot_blocks[block_hash] = KVSnapshotBlock(
                        parent_block_hash=parent,
                        block_hashes=[block_hash],
                    )
                    parent = block_hash
            elif isinstance(event, BlockRemoved):
                for block_hash in event.block_hashes:
                    self._snapshot_blocks.pop(block_hash, None)
            elif isinstance(event, AllBlocksCleared):
                self._snapshot_blocks.clear()

    def _service_snapshot_capture(self) -> None:
        try:
            request = self._snapshot_requests.get_nowait()
        except queue.Empty:
            return

        try:
            barrier_id = uuid.uuid4().hex
            barrier_seq = self._next_seq
            barrier = KVEventBatch(
                ts=time.time(), events=[], attn_dp_rank=self._dp_rank
            )
            payload = self._pack.encode(barrier)
            assert self._pub is not None
            self._pub.send_multipart(
                (
                    _snapshot_barrier_topic(self._topic_bytes, self._epoch, barrier_id),
                    barrier_seq.to_bytes(8, "big"),
                    payload,
                )
            )
            self._buffer.append((barrier_seq, payload))
            self._next_seq = barrier_seq + 1
            blocks = list(self._snapshot_blocks.values())
            request.response.put(
                _KVPlacementSnapshot(
                    header=KVSnapshotHeader(
                        version=SNAPSHOT_PROTOCOL_VERSION,
                        epoch=self._epoch,
                        replica_rank=self._dp_rank,
                        resume_seq=self._next_seq,
                        barrier_seq=barrier_seq,
                        barrier_id=barrier_id,
                        record_count=len(blocks),
                    ),
                    blocks=blocks,
                )
            )
        except BaseException as exc:
            request.response.put(exc)
        finally:
            self._snapshot_requests.task_done()

    def _snapshot_server_thread(self) -> None:
        assert self._snapshot_endpoint is not None
        sock: Optional[zmq.Socket] = None
        try:
            sock = self._ctx.socket(zmq.ROUTER)
            sock.setsockopt(zmq.SNDTIMEO, SNAPSHOT_SEND_TIMEOUT_MS)
            sock.bind(self._snapshot_endpoint)
        except BaseException as exc:
            self._snapshot_start_error = exc
            self._snapshot_started.set()
            return

        self._snapshot_started.set()
        encoder = msgspec.msgpack.Encoder()
        try:
            while self._running:
                if not sock.poll(100):
                    continue
                frames = sock.recv_multipart()
                if len(frames) != 3:
                    logger.warning("Invalid snapshot request: %s", frames)
                    continue
                client_id, delimiter, command = frames
                if delimiter or command != SNAPSHOT_REQUEST:
                    self._send_snapshot_error(
                        sock, client_id, b"invalid snapshot-v1 request"
                    )
                    continue

                response: Queue = Queue(maxsize=1)
                try:
                    self._snapshot_requests.put_nowait(
                        _SnapshotCaptureRequest(response=response)
                    )
                except queue.Full:
                    self._send_snapshot_error(
                        sock, client_id, b"snapshot provider busy"
                    )
                    continue
                try:
                    snapshot = response.get(timeout=SNAPSHOT_CAPTURE_TIMEOUT)
                except queue.Empty:
                    self._send_snapshot_error(
                        sock, client_id, b"snapshot capture timeout"
                    )
                    continue
                if isinstance(snapshot, BaseException):
                    logger.exception(
                        "KV snapshot capture failed",
                        exc_info=(
                            type(snapshot),
                            snapshot,
                            snapshot.__traceback__,
                        ),
                    )
                    self._send_snapshot_error(
                        sock, client_id, b"snapshot capture failed"
                    )
                    continue

                try:
                    self._send_snapshot_response(sock, client_id, encoder, snapshot)
                except zmq.Again:
                    logger.warning(
                        "Timed out sending KV snapshot to client %r after %d ms; "
                        "aborting this response",
                        client_id,
                        SNAPSHOT_SEND_TIMEOUT_MS,
                    )
        finally:
            sock.close(linger=0)

    @staticmethod
    def _send_snapshot_response(
        sock: zmq.Socket,
        client_id: bytes,
        encoder: msgspec.msgpack.Encoder,
        snapshot: _KVPlacementSnapshot,
    ) -> None:
        sock.send_multipart(
            (
                client_id,
                b"",
                SNAPSHOT_HEADER,
                encoder.encode(snapshot.header),
            )
        )
        for start in range(0, len(snapshot.blocks), SNAPSHOT_CHUNK_RECORDS):
            sock.send_multipart(
                (
                    client_id,
                    b"",
                    SNAPSHOT_CHUNK,
                    encoder.encode(
                        snapshot.blocks[start : start + SNAPSHOT_CHUNK_RECORDS]
                    ),
                )
            )
        sock.send_multipart((client_id, b"", SNAPSHOT_END, b""))

    @staticmethod
    def _send_snapshot_error(
        sock: zmq.Socket, client_id: bytes, message: bytes
    ) -> None:
        try:
            sock.send_multipart((client_id, b"", SNAPSHOT_ERROR, message))
        except zmq.Again:
            logger.warning(
                "Timed out sending KV snapshot error to client %r after %d ms; "
                "dropping this response",
                client_id,
                SNAPSHOT_SEND_TIMEOUT_MS,
            )

    def _service_replay(self) -> None:
        """If a replay request is waiting, send buffered batches."""
        assert self._replay is not None  # narrows type for mypy

        frame = self._replay.recv_multipart()
        if len(frame) != 3:
            logger.warning("Invalid replay request: %s", frame)
            return
        client_id, _, start_seq_bytes = frame
        start_seq = int.from_bytes(start_seq_bytes, "big")

        for seq, buf in self._buffer:
            if seq >= start_seq:
                # [identity, empty_delim, seq_bytes, payload]
                # (identity, empty_delim) are stripped off by the router
                # receiving payload is (seq_bytes, payload)
                self._replay.send_multipart(
                    (client_id, b"", seq.to_bytes(8, "big"), buf)
                )
        # Send end of sequence marker
        # receiving payload is (-1, b""")
        self._replay.send_multipart((client_id, b"", self.END_SEQ, b""))

    @staticmethod
    def offset_endpoint_port(
        endpoint: Optional[str], data_parallel_rank: int
    ) -> Optional[str]:
        """Helper function to offset the port in an endpoint by
            the data parallel rank.

        Args:
            endpoint: The endpoint string
                (e.g., "tcp://*:5557" or "inproc://cache")
            data_parallel_rank: The data parallel rank to offset by

        Returns:
            The endpoint with the port offset by data_parallel_rank
                or suffix appended
        """
        # Do nothing if input is None or data_parallel_rank is 0
        if not endpoint or data_parallel_rank == 0:
            return endpoint

        if "inproc" in endpoint:
            return f"{endpoint}_dp{data_parallel_rank}"
        if "tcp" in endpoint:
            if endpoint and ":" in endpoint:
                # Get everything after the last colon (the port)
                last_colon_idx = endpoint.rfind(":")
                base_addr = endpoint[:last_colon_idx]
                base_port = int(endpoint[last_colon_idx + 1 :])
                new_port = base_port + data_parallel_rank
                return f"{base_addr}:{new_port}"
            return endpoint
        raise ValueError("Invalid endpoint: must contain 'inproc' or 'tcp'")


class _LegacyEventPublisherConfig(BaseModel):
    """Stable constructor contract for publishers using the legacy registry API.

    Do not add fields here. Publisher-specific capabilities belong in that
    publisher's own config model so extending one implementation cannot break
    constructors registered by another implementation.
    """

    model_config = ConfigDict(extra="forbid")

    endpoint: str = "tcp://*:5557"
    """The zmq endpoint to use for publishing kv events.
    """

    replay_endpoint: Optional[str] = None
    """The zmq endpoint to use for replaying kv events.
    """

    buffer_steps: int = 10_000
    """The number of steps to cache for replay endpoint. Will only save
    events from the last N steps for the replay endpoint.
    """

    hwm: int = 100_000
    """The zmq high water mark for the event publisher. After queueing N events,
    events will start dropping if the consumer is not keeping up.
    """

    max_queue_size: int = 100_000
    """The maximum number of events to queue while waiting for publishing.
    """

    topic: str = ""
    """The topic to use for the event publisher. Consumers can subscribe to
    this topic to receive events.
    """


class ZmqEventPublisherConfig(_LegacyEventPublisherConfig):
    """Configuration owned by the built-in ZMQ event publisher."""

    snapshot_endpoint: Optional[str] = None
    """The ZMQ ROUTER endpoint used to stream placement snapshots.

    Like ``endpoint`` and ``replay_endpoint``, the data-parallel replica rank
    is added to the configured base port.
    """


class NullEventPublisherConfig(BaseModel):
    """The null publisher intentionally accepts no publisher options."""

    model_config = ConfigDict(extra="forbid")


class KVEventsConfig(ZmqEventPublisherConfig):
    """Backward-compatible model for the original built-in ZMQ CLI shape.

    Publisher construction and server introspection now use the
    publisher-specific models registered in :class:`EventPublisherFactory`;
    this model remains for existing external imports of ``KVEventsConfig``.
    """

    publisher: str = "null"
    """The publisher to use for publishing kv events. Can be "null", "zmq".
    """

    @classmethod
    def from_cli(cls, cli_value: str) -> "KVEventsConfig":
        """Parse the CLI value for the event publisher config."""
        return cls.model_validate_json(cli_value)


class _EventPublisherSelection(BaseModel):
    """Parse only the registry key while preserving publisher-owned fields."""

    model_config = ConfigDict(extra="allow")

    publisher: str = "null"


@dataclass(frozen=True, slots=True)
class _PublisherSpec:
    config_model: type[BaseModel]
    factory: Callable[[int, BaseModel], EventPublisher]


def _create_null_event_publisher(
    _attn_dp_rank: int, _config: BaseModel
) -> EventPublisher:
    return NullEventPublisher()


def _create_zmq_event_publisher(attn_dp_rank: int, config: BaseModel) -> EventPublisher:
    assert isinstance(config, ZmqEventPublisherConfig)
    return ZmqEventPublisher(attn_dp_rank=attn_dp_rank, **config.model_dump())


class EventPublisherFactory:
    _registry: dict[str, _PublisherSpec] = {
        "null": _PublisherSpec(
            config_model=NullEventPublisherConfig,
            factory=_create_null_event_publisher,
        ),
        "zmq": _PublisherSpec(
            config_model=ZmqEventPublisherConfig,
            factory=_create_zmq_event_publisher,
        ),
    }

    @classmethod
    def register_publisher(cls, name: str, ctor: Callable[..., EventPublisher]) -> None:
        """Register a publisher using the stable pre-snapshot kwargs contract.

        New integrations should use :meth:`register_publisher_spec` to own and
        validate their configuration schema. This method remains source
        compatible with existing custom publishers.
        """
        if name in cls._registry:
            raise KeyError(f"publisher '{name}' already registered")

        def legacy_factory(attn_dp_rank: int, config: BaseModel) -> EventPublisher:
            return ctor(attn_dp_rank=attn_dp_rank, **config.model_dump())

        cls._registry[name] = _PublisherSpec(
            config_model=_LegacyEventPublisherConfig,
            factory=legacy_factory,
        )

    @classmethod
    def register_publisher_spec(
        cls,
        name: str,
        config_model: type[BaseModel],
        factory: Callable[[int, BaseModel], EventPublisher],
    ) -> None:
        """Register a publisher with an implementation-owned config model.

        ``factory`` is called as ``factory(attn_dp_rank, validated_config)``.
        The config model should normally reject extra fields so unsupported
        capabilities fail during validation rather than being ignored.
        """
        if name in cls._registry:
            raise KeyError(f"publisher '{name}' already registered")
        cls._registry[name] = _PublisherSpec(
            config_model=config_model,
            factory=factory,
        )

    @classmethod
    def create(cls, config: Optional[str], attn_dp_rank: int = 0) -> EventPublisher:
        """Validate config against the selected publisher's schema and build it."""
        if not config:
            return NullEventPublisher()

        _kind, spec, publisher_config = cls._parse_config(config)
        return spec.factory(attn_dp_rank, publisher_config)

    @classmethod
    def parse_config(cls, config: str) -> tuple[str, BaseModel]:
        """Return the selected kind and its validated publisher-owned config."""
        kind, _spec, publisher_config = cls._parse_config(config)
        return kind, publisher_config

    @classmethod
    def _parse_config(cls, config: str) -> tuple[str, _PublisherSpec, BaseModel]:
        selection = _EventPublisherSelection.model_validate_json(config)
        kind = selection.publisher
        try:
            spec = cls._registry[kind]
        except KeyError as exc:
            raise ValueError(f"Unknown event publisher '{kind}'") from exc
        try:
            publisher_config = spec.config_model.model_validate(
                selection.model_extra or {}
            )
        except ValidationError as exc:
            raise ValueError(
                f"Invalid config for event publisher '{kind}': {exc}"
            ) from exc
        return kind, spec, publisher_config
