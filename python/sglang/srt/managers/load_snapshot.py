from __future__ import annotations

import fcntl
import hashlib
import logging
import mmap
import os
import struct
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace
from typing import Optional

logger = logging.getLogger(__name__)

MAGIC = b"SLNS"
VERSION = 1
HEADER_STRUCT = struct.Struct("<4sHHI")

DISAGG_MODE_TO_INT = {"null": 0, "prefill": 1, "decode": 2}
INT_TO_DISAGG_MODE = {v: k for k, v in DISAGG_MODE_TO_INT.items()}


def i64(default: int = 0):
    return field(default=default, metadata={"fmt": "q"})


def f64(default: float = 0.0):
    return field(default=default, metadata={"fmt": "d"})


@dataclass
class LoadSnapshot:
    timestamp: float = f64()
    dp_rank: int = i64()
    num_running_reqs: int = i64()
    num_waiting_reqs: int = i64()
    num_used_tokens: int = i64()
    num_total_tokens: int = i64()
    max_total_num_tokens: int = i64()
    max_running_requests: int = i64()
    token_usage: float = f64()
    gen_throughput: float = f64()
    cache_hit_rate: float = f64()
    utilization: float = f64()

    has_memory: int = i64()
    memory_weight_gb: float = f64()
    memory_kv_cache_gb: float = f64()
    memory_graph_gb: float = f64()
    memory_token_capacity: int = i64()

    has_speculative: int = i64()
    speculative_accept_length: float = f64()
    speculative_accept_rate: float = f64()

    has_lora: int = i64()
    lora_slots_used: int = i64()
    lora_slots_total: int = i64()
    lora_utilization: float = f64()

    has_disaggregation: int = i64()
    disagg_mode: int = i64()
    prefill_bootstrap_queue_reqs: int = i64()
    prefill_inflight_queue_reqs: int = i64()
    decode_prealloc_queue_reqs: int = i64()
    decode_transfer_queue_reqs: int = i64()
    decode_retracted_queue_reqs: int = i64()
    kv_transfer_speed_gb_s: float = f64()
    kv_transfer_latency_ms: float = f64()

    has_queues: int = i64()
    queue_waiting: int = i64()
    queue_grammar: int = i64()
    queue_paused: int = i64()
    queue_retracted: int = i64()

    @classmethod
    def from_metrics(cls, **metrics) -> "LoadSnapshot":
        snapshot = {name: metrics[name] for name in CORE_METRIC_FIELDS}

        for include_key, section_name, present_attr, attrs in SECTION_FIELDS:
            section = metrics.get(section_name)
            snapshot[present_attr] = int(section is not None)
            if section is None:
                continue

            for section_attr, snapshot_attr in attrs:
                value = getattr(section, section_attr)
                if snapshot_attr == "disagg_mode":
                    value = DISAGG_MODE_TO_INT.get(value, 0)
                snapshot[snapshot_attr] = value

        return cls(**snapshot)

    def with_sections(self, include: list[str]) -> "LoadSnapshot":
        if not include or "all" in include:
            return self

        include = set(include)
        updates = {}
        for section, defaults in SECTION_FIELD_DEFAULTS.items():
            if section not in include:
                updates.update(defaults)
        return replace(self, **updates) if updates else self

    def to_dict(self) -> dict:
        load = {
            name: getattr(self, name)
            for name in CORE_METRIC_FIELDS
            if name != "timestamp"
        }

        for include_key, section_name, present_attr, attrs in SECTION_FIELDS:
            if not getattr(self, present_attr):
                continue

            section = {}
            for section_attr, snapshot_attr in attrs:
                value = getattr(self, snapshot_attr)
                if snapshot_attr == "disagg_mode":
                    value = INT_TO_DISAGG_MODE.get(value, "null")
                section[section_attr] = value
            load[section_name] = section

        return load


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

SNAPSHOT_FIELDS = fields(LoadSnapshot)
PAYLOAD_STRUCT = struct.Struct(
    "<" + "".join(snapshot_field.metadata["fmt"] for snapshot_field in SNAPSHOT_FIELDS)
)
# 64-byte aligned so each slot starts on a cache-line boundary.
SLOT_SIZE = (PAYLOAD_STRUCT.size + 63) & ~63
FIELD_DEFAULTS = {
    snapshot_field.name: snapshot_field.default for snapshot_field in SNAPSHOT_FIELDS
}
SECTION_FIELD_DEFAULTS = {
    include_key: {
        name: FIELD_DEFAULTS[name]
        for name in (present_attr, *(snapshot_attr for _, snapshot_attr in attrs))
    }
    for include_key, section_name, present_attr, attrs in SECTION_FIELDS
}


def snapshot_values(snapshot: LoadSnapshot) -> tuple:
    return tuple(
        getattr(snapshot, snapshot_field.name) for snapshot_field in SNAPSHOT_FIELDS
    )


def snapshot_from_values(values: tuple) -> LoadSnapshot:
    return LoadSnapshot(
        **dict(
            zip(
                (snapshot_field.name for snapshot_field in SNAPSHOT_FIELDS),
                values,
                strict=True,
            )
        )
    )


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


def file_size(dp_size: int) -> int:
    return HEADER_STRUCT.size + dp_size * SLOT_SIZE


def slot_offset(dp_rank: int) -> int:
    return HEADER_STRUCT.size + dp_rank * SLOT_SIZE


class LoadSnapshotWriter:
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
        self.fd = -1
        size = file_size(dp_size)

        self.fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            with file_lock(self.fd, fcntl.LOCK_EX):
                os.ftruncate(self.fd, size)
                self.mmap = mmap.mmap(self.fd, size, access=mmap.ACCESS_WRITE)
                HEADER_STRUCT.pack_into(
                    self.mmap, 0, MAGIC, VERSION, dp_size, SLOT_SIZE
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
        PAYLOAD_STRUCT.pack_into(
            self.mmap,
            slot_offset(self.dp_rank),
            *snapshot_values(snapshot),
        )

    def close(self) -> None:
        self.mmap.close()
        os.close(self.fd)


class LoadSnapshotReader:
    def __init__(self, path: str, dp_size: int):
        self.path = path
        self.dp_size = dp_size
        self.mmap: Optional[mmap.mmap] = None
        self.fd: Optional[int] = None
        self.header_warning_logged = False

    def attach(self) -> bool:
        if self.mmap is not None:
            return True

        size = file_size(self.dp_size)
        try:
            fd = os.open(self.path, os.O_RDONLY)
        except FileNotFoundError:
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
            or slot_size != SLOT_SIZE
        ):
            mapped.close()
            os.close(fd)
            if not self.header_warning_logged:
                logger.warning("load shm header mismatch at %s", self.path)
                self.header_warning_logged = True
            return False

        self.mmap = mapped
        self.fd = fd
        return True

    def read(self, dp_rank: int) -> Optional[LoadSnapshot]:
        if dp_rank < 0 or dp_rank >= self.dp_size:
            return None
        if not self.attach():
            return None

        assert self.fd is not None
        with file_lock(self.fd, fcntl.LOCK_SH):
            return self._read_slot(dp_rank)

    def _read_slot(self, dp_rank: int) -> LoadSnapshot:
        assert self.mmap is not None
        values = PAYLOAD_STRUCT.unpack_from(self.mmap, slot_offset(dp_rank))
        return snapshot_from_values(values)

    def read_all(self) -> list[LoadSnapshot]:
        if not self.attach():
            return []

        assert self.fd is not None
        with file_lock(self.fd, fcntl.LOCK_SH):
            return [self._read_slot(r) for r in range(self.dp_size)]

    def close(self) -> None:
        if self.mmap is not None:
            self.mmap.close()
            self.mmap = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
