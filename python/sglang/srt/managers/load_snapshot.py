"""Load snapshot: publish scheduler load metrics for DP balancing and /v1/loads.

Architecture
------------

Each scheduler periodically publishes a ``LoadSnapshot`` containing its
current load metrics (running reqs, tokens, throughput, ...).  Two
transport backends are supported:

**SHM mode** (single-node, default)::

    Scheduler  ──ShmLoadSnapshotWriter──▶  /dev/shm mmap file
                                               ▲
    TokenizerManager  ──ShmLoadSnapshotReader───┘  (for /v1/loads)
    DataParallelController  ──ShmLoadSnapshotReader─┘  (for dispatch)

**ZMQ mode** (multi-node DP attention, or ``SGLANG_LOAD_SNAPSHOT_USE_ZMQ=1``)::

    Scheduler (any node)  ──ZmqLoadSnapshotWriter (PUSH)──▶  network
                                                               │
    ZmqShmLoadSnapshotReader (PULL, node 0)  ◀─────────────────┘
        │  drains zmq, writes to SHM
        ▼
    /dev/shm mmap file (node 0)
        ▲
    TokenizerManager / DataParallelController  ──ShmLoadSnapshotReader──┘

Shared memory does not work across nodes, so multi-node DP attention
requires the ZMQ transport.  The ``ZmqShmLoadSnapshotReader`` on node 0
receives snapshots from all schedulers via zmq PUSH/PULL and writes them
into the local SHM file.  All readers (tokenizer, dp_controller) on
node 0 then read from SHM.

``zmq_reader_owner()`` decides which process on node 0 binds the zmq
PULL socket (only one can bind); the other reads plain SHM.
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import mmap
import os
import struct
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

import msgspec
import msgspec.msgpack
import msgspec.structs

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import GetLoadsReqOutput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DISAGG_MODE_TO_INT = {"null": 0, "prefill": 1, "decode": 2}
INT_TO_DISAGG_MODE = {v: k for k, v in DISAGG_MODE_TO_INT.items()}


def _native(v):
    """Coerce numpy scalars to Python int/float for msgpack encoding."""
    if hasattr(v, "item"):
        return v.item()
    return v


def should_use_zmq(server_args) -> bool:
    """Whether to use zmq PUSH/PULL instead of shared memory for load snapshots.

    Shared memory (mmap) only works within a single node.  When schedulers
    run on multiple nodes (multi-node DP attention), they cannot write to
    the SHM file on node 0, so we fall back to zmq transport.  The env var
    ``SGLANG_LOAD_SNAPSHOT_USE_ZMQ`` forces zmq mode for testing.
    """
    return (
        server_args.enable_dp_attention and server_args.nnodes > 1
    ) or envs.SGLANG_LOAD_SNAPSHOT_USE_ZMQ.get()


_LOAD_AWARE_METHODS = frozenset({"total_requests", "total_tokens"})


def zmq_reader_owner(server_args, caller: str) -> bool:
    """Decide which process owns the zmq PULL socket.

    Exactly one of ``"dp_controller"`` or ``"tokenizer"`` must return True
    when zmq mode is active.  The owner polls zmq -> SHM; the other reads SHM.

    Rules:
      - Non-zero node_rank: no tokenizer, dp_controller only launches
        schedulers and waits -> nobody owns it.
      - dp_size == 1: no dp_controller exists -> tokenizer owns it.
      - dp_size > 1, load-aware method: dp_controller polls on every
        dispatch via refresh_load_budget() -> dp_controller owns it.
      - dp_size > 1, round-robin / other: dp_controller never reads
        load data -> tokenizer owns it (polls on /v1/loads calls).
    """
    if not should_use_zmq(server_args):
        return False
    if server_args.node_rank != 0:
        return False
    if server_args.dp_size == 1:
        return caller == "tokenizer"
    if server_args.load_balance_method.lower() in _LOAD_AWARE_METHODS:
        return caller == "dp_controller"
    return caller == "tokenizer"


# ---------------------------------------------------------------------------
# LoadSnapshot data class
# ---------------------------------------------------------------------------

CORE_METRIC_FIELDS = (
    "timestamp",
    "dp_rank",
    "num_running_reqs",
    "num_waiting_reqs",
    "num_used_tokens",
    "num_total_tokens",
    "max_total_num_tokens",
    "max_running_requests",
    "token_usage",
    "gen_throughput",
    "cache_hit_rate",
    "utilization",
)
SECTION_FIELDS = (
    (
        "memory",
        "memory",
        "has_memory",
        (
            ("weight_gb", "memory_weight_gb"),
            ("kv_cache_gb", "memory_kv_cache_gb"),
            ("graph_gb", "memory_graph_gb"),
            ("token_capacity", "memory_token_capacity"),
        ),
    ),
    (
        "spec",
        "speculative",
        "has_speculative",
        (
            ("accept_length", "speculative_accept_length"),
            ("accept_rate", "speculative_accept_rate"),
        ),
    ),
    (
        "lora",
        "lora",
        "has_lora",
        (
            ("slots_used", "lora_slots_used"),
            ("slots_total", "lora_slots_total"),
            ("utilization", "lora_utilization"),
        ),
    ),
    (
        "disagg",
        "disaggregation",
        "has_disaggregation",
        (
            ("mode", "disagg_mode"),
            ("prefill_bootstrap_queue_reqs", "prefill_bootstrap_queue_reqs"),
            ("prefill_inflight_queue_reqs", "prefill_inflight_queue_reqs"),
            ("decode_prealloc_queue_reqs", "decode_prealloc_queue_reqs"),
            ("decode_transfer_queue_reqs", "decode_transfer_queue_reqs"),
            ("decode_retracted_queue_reqs", "decode_retracted_queue_reqs"),
            ("kv_transfer_speed_gb_s", "kv_transfer_speed_gb_s"),
            ("kv_transfer_latency_ms", "kv_transfer_latency_ms"),
        ),
    ),
    (
        "queues",
        "queues",
        "has_queues",
        (
            ("waiting", "queue_waiting"),
            ("grammar", "queue_grammar"),
            ("paused", "queue_paused"),
            ("retracted", "queue_retracted"),
        ),
    ),
)


class LoadSnapshot(msgspec.Struct, omit_defaults=True):
    timestamp: float = 0.0
    dp_rank: int = 0
    num_running_reqs: int = 0
    num_waiting_reqs: int = 0
    num_used_tokens: int = 0
    num_total_tokens: int = 0
    max_total_num_tokens: int = 0
    max_running_requests: int = 0
    token_usage: float = 0.0
    gen_throughput: float = 0.0
    cache_hit_rate: float = 0.0
    utilization: float = 0.0

    has_memory: int = 0
    memory_weight_gb: float = 0.0
    memory_kv_cache_gb: float = 0.0
    memory_graph_gb: float = 0.0
    memory_token_capacity: int = 0

    has_speculative: int = 0
    speculative_accept_length: float = 0.0
    speculative_accept_rate: float = 0.0

    has_lora: int = 0
    lora_slots_used: int = 0
    lora_slots_total: int = 0
    lora_utilization: float = 0.0

    has_disaggregation: int = 0
    disagg_mode: int = 0
    prefill_bootstrap_queue_reqs: int = 0
    prefill_inflight_queue_reqs: int = 0
    decode_prealloc_queue_reqs: int = 0
    decode_transfer_queue_reqs: int = 0
    decode_retracted_queue_reqs: int = 0
    kv_transfer_speed_gb_s: float = 0.0
    kv_transfer_latency_ms: float = 0.0

    has_queues: int = 0
    queue_waiting: int = 0
    queue_grammar: int = 0
    queue_paused: int = 0
    queue_retracted: int = 0

    @classmethod
    def from_get_loads_output(cls, output: GetLoadsReqOutput) -> LoadSnapshot:
        snapshot: dict = {}
        for name in CORE_METRIC_FIELDS:
            value = getattr(output, name)
            if name == "dp_rank":
                snapshot[name] = int(value) if value is not None else 0
            else:
                snapshot[name] = _native(value)

        for _, section_name, present_attr, attrs in SECTION_FIELDS:
            section = getattr(output, section_name, None)
            snapshot[present_attr] = int(section is not None)
            if section is None:
                continue
            for section_attr, snapshot_attr in attrs:
                value = getattr(section, section_attr)
                if snapshot_attr == "disagg_mode":
                    value = DISAGG_MODE_TO_INT.get(value, 0)
                else:
                    value = _native(value)
                snapshot[snapshot_attr] = value

        return cls(**snapshot)

    VALID_SECTIONS = frozenset(
        {"core", "memory", "spec", "lora", "disagg", "queues", "all"}
    )

    def to_dict(self, include: Optional[set[str]] = None) -> dict:
        load = {
            "dp_rank": self.dp_rank,
            "num_running_reqs": self.num_running_reqs,
            "num_waiting_reqs": self.num_waiting_reqs,
            "num_used_tokens": self.num_used_tokens,
            "num_total_tokens": self.num_total_tokens,
            "max_total_num_tokens": self.max_total_num_tokens,
            "max_running_requests": self.max_running_requests,
            "token_usage": self.token_usage,
            "gen_throughput": self.gen_throughput,
            "cache_hit_rate": self.cache_hit_rate,
            "utilization": self.utilization,
        }

        if include is None or "all" in include:
            include_all = True
        else:
            if not (include <= self.VALID_SECTIONS):
                raise ValueError(
                    f"Invalid include sections: {include - self.VALID_SECTIONS}. "
                    f"Valid options: {sorted(self.VALID_SECTIONS)}"
                )
            if include == {"core"}:
                return load
            include_all = False

        for include_key, section_name, present_attr, attrs in SECTION_FIELDS:
            if not getattr(self, present_attr):
                continue
            if not include_all and include_key not in include:
                continue

            section = {}
            for section_attr, snapshot_attr in attrs:
                value = getattr(self, snapshot_attr)
                if snapshot_attr == "disagg_mode":
                    value = INT_TO_DISAGG_MODE.get(value, "null")
                section[section_attr] = value
            load[section_name] = section

        return load


snapshot_encoder = msgspec.msgpack.Encoder()
snapshot_decoder = msgspec.msgpack.Decoder(LoadSnapshot)


# ---------------------------------------------------------------------------
# SHM file layout utilities
# ---------------------------------------------------------------------------

MAGIC = b"SLNS"
VERSION = 2
HEADER_STRUCT = struct.Struct("<4sHHI")
SLOT_LEN_STRUCT = struct.Struct("<I")
SLOT_SIZE = 16 * 1024


@contextmanager
def file_lock(fd: int, lock_type: int):
    fcntl.flock(fd, lock_type)
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)


def shm_path_for(ipc_name: str) -> str:
    name = os.path.basename(ipc_name.rstrip("/")) or "default"
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    digest = hashlib.blake2s(ipc_name.encode(), digest_size=4).hexdigest()
    return f"/dev/shm/sglang_loads_{safe_name}_{digest}.shm"


def file_size(dp_size: int, slot_size: int = SLOT_SIZE) -> int:
    return HEADER_STRUCT.size + dp_size * slot_size


def slot_offset(dp_rank: int, slot_size: int = SLOT_SIZE) -> int:
    return HEADER_STRUCT.size + dp_rank * slot_size


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


class ShmLoadSnapshotWriter:
    def __init__(
        self, path: str, dp_size: int, dp_rank: int, publish_interval: int = 1
    ):
        if dp_rank < 0 or dp_rank >= dp_size:
            raise ValueError(f"invalid dp_rank={dp_rank} for dp_size={dp_size}")
        self.publish_interval = max(1, publish_interval)
        self.publish_counter = 0

        self.path = path
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.slot_size = SLOT_SIZE
        self.fd = -1
        size = file_size(dp_size, self.slot_size)

        self.fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            with file_lock(self.fd, fcntl.LOCK_EX):
                os.ftruncate(self.fd, size)
                self.mmap = mmap.mmap(self.fd, size, access=mmap.ACCESS_WRITE)
                HEADER_STRUCT.pack_into(
                    self.mmap, 0, MAGIC, VERSION, dp_size, self.slot_size
                )
                self._write_payload(LoadSnapshot(dp_rank=dp_rank))
        except Exception:
            if self.fd >= 0:
                os.close(self.fd)
            raise

    def write(self, snapshot: LoadSnapshot) -> None:
        if snapshot.dp_rank != self.dp_rank:
            raise ValueError(
                f"snapshot dp_rank={snapshot.dp_rank} does not match writer dp_rank={self.dp_rank}"
            )

        with file_lock(self.fd, fcntl.LOCK_EX):
            self._write_payload(snapshot)

    def _write_payload(self, snapshot: LoadSnapshot) -> None:
        payload = snapshot_encoder.encode(snapshot)
        max_payload_size = self.slot_size - SLOT_LEN_STRUCT.size
        if len(payload) > max_payload_size:
            raise ValueError(
                f"load snapshot payload size {len(payload)} exceeds slot payload "
                f"capacity {max_payload_size}"
            )

        offset = slot_offset(self.dp_rank, self.slot_size)
        payload_start = offset + SLOT_LEN_STRUCT.size
        payload_end = payload_start + len(payload)
        slot_end = offset + self.slot_size

        SLOT_LEN_STRUCT.pack_into(self.mmap, offset, 0)
        self.mmap[payload_start:payload_end] = payload
        self.mmap[payload_end:slot_end] = b"\0" * (slot_end - payload_end)
        SLOT_LEN_STRUCT.pack_into(self.mmap, offset, len(payload))

    def close(self) -> None:
        self.mmap.close()
        os.close(self.fd)


class ZmqLoadSnapshotWriter:
    """Sends load snapshots via zmq PUSH to a ZmqShmLoadSnapshotReader.

    CONFLATE is set so only the latest message is kept in the send
    buffer when the reader is slower than the writer.
    """

    def __init__(
        self, endpoint: str, dp_size: int, dp_rank: int, publish_interval: int = 1
    ):
        import zmq as _zmq

        if dp_rank < 0 or dp_rank >= dp_size:
            raise ValueError(f"invalid dp_rank={dp_rank} for dp_size={dp_size}")
        self.publish_interval = max(1, publish_interval)
        self.publish_counter = 0
        self.dp_size = dp_size
        self.dp_rank = dp_rank

        self._zmq = _zmq
        self._ctx = _zmq.Context.instance()
        self._socket = self._ctx.socket(_zmq.PUSH)
        self._socket.setsockopt(_zmq.LINGER, 0)
        self._socket.setsockopt(_zmq.CONFLATE, 1)
        self._socket.connect(endpoint)

    def write(self, snapshot: LoadSnapshot) -> None:
        if snapshot.dp_rank != self.dp_rank:
            raise ValueError(
                f"snapshot dp_rank={snapshot.dp_rank} does not match "
                f"writer dp_rank={self.dp_rank}"
            )
        try:
            self._socket.send(snapshot_encoder.encode(snapshot), self._zmq.NOBLOCK)
        except self._zmq.Again:
            pass

    def close(self) -> None:
        self._socket.close()


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


class ShmLoadSnapshotReader:
    def __init__(self, path: str, dp_size: int):
        self.path = path
        self.dp_size = dp_size
        self.mmap: Optional[mmap.mmap] = None
        self.fd: Optional[int] = None
        self.slot_size = SLOT_SIZE
        self._header_warning_logged = False
        self._attach()

    def _attach(self) -> bool:
        if self.mmap is not None:
            return True

        try:
            fd = os.open(self.path, os.O_RDONLY)
        except FileNotFoundError:
            return False

        size = os.fstat(fd).st_size
        if size < HEADER_STRUCT.size:
            os.close(fd)
            return False

        try:
            with file_lock(fd, fcntl.LOCK_SH):
                mapped = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
                magic, version, dp_size, slot_size = HEADER_STRUCT.unpack_from(
                    mapped, 0
                )
        except (OSError, ValueError):
            os.close(fd)
            return False

        if (
            magic != MAGIC
            or version != VERSION
            or dp_size != self.dp_size
            or slot_size < SLOT_LEN_STRUCT.size
            or size < file_size(self.dp_size, slot_size)
        ):
            mapped.close()
            os.close(fd)
            if not self._header_warning_logged:
                logger.warning("load shm header mismatch at %s", self.path)
                self._header_warning_logged = True
            return False

        self.mmap = mapped
        self.fd = fd
        self.slot_size = slot_size
        return True

    def read(self, dp_rank: int) -> Optional[LoadSnapshot]:
        if dp_rank < 0 or dp_rank >= self.dp_size:
            return None
        if not self._attach():
            return None

        assert self.fd is not None
        with file_lock(self.fd, fcntl.LOCK_SH):
            return self._read_slot(dp_rank)

    def _read_slot(self, dp_rank: int) -> Optional[LoadSnapshot]:
        assert self.mmap is not None
        offset = slot_offset(dp_rank, self.slot_size)
        (payload_len,) = SLOT_LEN_STRUCT.unpack_from(self.mmap, offset)
        max_payload_size = self.slot_size - SLOT_LEN_STRUCT.size
        if payload_len == 0 or payload_len > max_payload_size:
            return None

        payload_start = offset + SLOT_LEN_STRUCT.size
        payload_end = payload_start + payload_len
        try:
            return snapshot_decoder.decode(self.mmap[payload_start:payload_end])
        except Exception as e:
            logger.debug("load snapshot decode failed for rank %s: %s", dp_rank, e)
            return None

    def read_all(self) -> list[LoadSnapshot]:
        if not self._attach():
            return []

        assert self.fd is not None
        with file_lock(self.fd, fcntl.LOCK_SH):
            loads = []
            for r in range(self.dp_size):
                load = self._read_slot(r)
                if load is not None:
                    loads.append(load)
            return loads

    def close(self) -> None:
        if self.mmap is not None:
            self.mmap.close()
            self.mmap = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class ZmqShmLoadSnapshotReader:
    """Receives snapshots via zmq PULL from writers, writes to SHM, reads from SHM.

    Transparently wraps a ShmLoadSnapshotReader.  Every read() / read_all()
    first drains the PULL socket into SHM so callers always see fresh data.
    """

    def __init__(self, endpoint: str, shm_path: str, dp_size: int):
        import zmq as _zmq

        self._zmq = _zmq
        self._ctx = _zmq.Context.instance()
        self._socket = self._ctx.socket(_zmq.PULL)
        self._socket.setsockopt(_zmq.LINGER, 0)
        self._socket.setsockopt(_zmq.CONFLATE, 1)
        self._socket.bind(endpoint)

        self._endpoint = endpoint
        self._shm_path = shm_path
        self.dp_size = dp_size
        self._shm_reader = ShmLoadSnapshotReader(shm_path, dp_size)
        self._shm_writers: dict[int, ShmLoadSnapshotWriter] = {}

    def _poll(self) -> None:
        """Drain zmq messages and write latest per dp_rank to SHM."""
        latest: dict[int, LoadSnapshot] = {}
        while True:
            try:
                data = self._socket.recv(self._zmq.NOBLOCK)
            except self._zmq.Again:
                break
            try:
                snapshot = snapshot_decoder.decode(data)
                if 0 <= snapshot.dp_rank < self.dp_size:
                    latest[snapshot.dp_rank] = snapshot
            except Exception as e:
                logger.warning("load snapshot zmq decode failed: %s", e)

        for dp_rank, snapshot in latest.items():
            if dp_rank not in self._shm_writers:
                self._shm_writers[dp_rank] = ShmLoadSnapshotWriter(
                    self._shm_path, self.dp_size, dp_rank
                )
            try:
                self._shm_writers[dp_rank].write(snapshot)
            except Exception as e:
                logger.warning(
                    "load snapshot shm write failed for rank %d: %s", dp_rank, e
                )

    def read(self, dp_rank: int) -> Optional[LoadSnapshot]:
        self._poll()
        return self._shm_reader.read(dp_rank)

    def read_all(self) -> list[LoadSnapshot]:
        self._poll()
        return self._shm_reader.read_all()

    def close(self) -> None:
        for w in self._shm_writers.values():
            w.close()
        self._shm_writers.clear()
        self._shm_reader.close()
        self._socket.close()
        if self._endpoint.startswith("ipc://"):
            try:
                os.unlink(self._endpoint[len("ipc://") :])
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def _zmq_addr_for(port_args) -> str:
    """Return the zmq PUSH/PULL address from PortArgs.

    For dp_attention (TCP mode), uses the ``load_collector_ipc_name`` field
    stored in PortArgs.  For single-node IPC (env-var override), derives
    a deterministic IPC path from ``instance_id``.
    """
    ipc_name = getattr(port_args, "load_collector_ipc_name", "")
    if ipc_name:
        return ipc_name
    safe = "".join(
        c if c.isalnum() or c in "._-" else "_" for c in port_args.instance_id
    )
    digest = hashlib.blake2s(port_args.instance_id.encode(), digest_size=4).hexdigest()
    return f"ipc:///tmp/sglang_load_collector_{safe}_{digest}.sock"


def create_load_snapshot_writer(
    server_args,
    port_args,
    dp_size: int,
    dp_rank: int,
    publish_interval: int = 1,
):
    """Return a SHM or ZMQ writer based on server configuration."""
    if should_use_zmq(server_args):
        return ZmqLoadSnapshotWriter(
            _zmq_addr_for(port_args), dp_size, dp_rank, publish_interval
        )
    return ShmLoadSnapshotWriter(
        shm_path_for(port_args.instance_id), dp_size, dp_rank, publish_interval
    )


def create_load_snapshot_reader(server_args, port_args, caller: str):
    """Create a load snapshot reader.

    Args:
        caller: ``"dp_controller"`` or ``"tokenizer"`` -- determines who
            binds the zmq PULL socket when zmq mode is active.
    """
    dp_size = server_args.dp_size
    if zmq_reader_owner(server_args, caller):
        return ZmqShmLoadSnapshotReader(
            _zmq_addr_for(port_args), shm_path_for(port_args.instance_id), dp_size
        )
    return ShmLoadSnapshotReader(shm_path_for(port_args.instance_id), dp_size)
