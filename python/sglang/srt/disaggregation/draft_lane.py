"""Versioned descriptors and safe row plans for independent DFlash PD storage.

This lane has its own head ownership and completion state. Target MLA indices,
engine sessions, and completion counters never stand in for draft ownership.
No CUDA operations or collectives are performed by the wire/state layer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal

WIRE_VERSION = 1
CONTROL_TAG = b"DFLASH_PD_V1"


@dataclass(frozen=True)
class DraftRegion:
    layer: int
    kind: Literal["K", "V"]
    pool: Literal["FULL", "WINDOW"]
    base: int
    registered_bytes: int
    row_stride: int
    rows: int

    def __post_init__(self):
        if self.kind not in ("K", "V") or self.pool not in ("FULL", "WINDOW"):
            raise ValueError("Unknown draft region type")
        if (
            min(self.base, self.registered_bytes, self.row_stride, self.rows) <= 0
            or self.layer < 0
        ):
            raise ValueError("Invalid draft region geometry")
        if self.rows * self.row_stride > self.registered_bytes:
            raise ValueError("Draft rows exceed registered memory")

    @property
    def key(self):
        return self.layer, self.kind, self.pool

    def address(self, row, count=1):
        if row < 0 or count < 1 or row + count > self.rows:
            raise ValueError("Draft write outside registered region")
        return self.base + row * self.row_stride


@dataclass(frozen=True)
class DraftPeer:
    registration_id: str
    cohort_id: str
    cohort_epoch: str
    peer: int
    engine_session: str
    control_endpoint: str
    fingerprint: str
    dtype: str
    head_begin: int
    head_end: int
    total_heads: int
    head_dim: int
    page_size: int
    block_size: int
    regions: tuple[DraftRegion, ...]
    version: int = WIRE_VERSION
    layout: str = "NHD"

    def validate(self):
        if self.version != WIRE_VERSION or self.layout != "NHD":
            raise ValueError("Unsupported draft wire version/layout")
        if not all(
            (
                self.registration_id,
                self.cohort_id,
                self.cohort_epoch,
                self.engine_session,
                self.control_endpoint,
                self.fingerprint,
            )
        ):
            raise ValueError("Incomplete draft peer registration")
        if self.dtype not in ("bfloat16", "float16"):
            raise ValueError("Initial draft PD lane requires unscaled BF16/FP16")
        if not 0 <= self.head_begin < self.head_end <= self.total_heads:
            raise ValueError("Invalid draft head interval")
        if min(self.head_dim, self.page_size, self.block_size) <= 0:
            raise ValueError("Invalid draft attention geometry")
        keys = [region.key for region in self.regions]
        if len(set(keys)) != len(keys) or not keys:
            raise ValueError("Missing or duplicate draft regions")
        expected_stride = (self.head_end - self.head_begin) * self.head_dim * 2
        if any(r.row_stride != expected_stride for r in self.regions):
            raise ValueError("Draft region is not a packed NHD head shard")
        layers = {r.layer for r in self.regions}
        for layer in layers:
            entries = [r for r in self.regions if r.layer == layer]
            if (
                len(entries) != 2
                or {r.kind for r in entries} != {"K", "V"}
                or len({r.pool for r in entries}) != 1
            ):
                raise ValueError("Incomplete draft layer K/V pair")
        return self


def validate_cohort(peers):
    peers = tuple(sorted(peers, key=lambda p: p.peer))
    if not peers or [p.peer for p in peers] != list(range(len(peers))):
        raise ValueError("Missing or duplicate draft cohort peer")
    first = peers[0]
    common = (
        "cohort_id",
        "cohort_epoch",
        "fingerprint",
        "dtype",
        "total_heads",
        "head_dim",
        "page_size",
        "block_size",
        "version",
        "layout",
    )
    head = 0
    for peer in peers:
        peer.validate()
        if any(getattr(peer, k) != getattr(first, k) for k in common):
            raise ValueError("Draft cohort registration mismatch")
        if peer.head_begin != head or {r.key for r in peer.regions} != {
            r.key for r in first.regions
        }:
            raise ValueError("Draft cohort head/layer coverage has a gap or overlap")
        head = peer.head_end
    if head != first.total_heads:
        raise ValueError("Draft cohort does not cover all KV heads")
    if len({p.registration_id for p in peers}) != len(peers):
        raise ValueError("Duplicate draft registration identity")
    return peers


@dataclass(frozen=True)
class DraftLease:
    lease_id: str
    room: int
    request_id: str
    generation: int
    owner: int
    slot: int
    logical_end: int
    context_length: int
    full_row0: int
    full_stride: int
    window_row0: int
    window_stride: int
    cohort_epoch: str
    attempt: int = 0

    def validate(self):
        if not self.lease_id or not self.request_id or not self.cohort_epoch:
            raise ValueError("Incomplete draft lease identity")
        if (
            min(self.generation, self.owner, self.attempt, self.room) < 0
            or self.slot < 1
        ):
            raise ValueError("Invalid draft lease owner/slot")
        if not 0 < self.logical_end <= self.context_length <= self.full_stride:
            raise ValueError("Draft lease exceeds model/storage context")
        if min(self.full_row0, self.window_row0, self.window_stride) <= 0:
            raise ValueError("Invalid draft lease physical mapping")
        return self

    @property
    def window_floor(self):
        return max(0, self.logical_end - self.window_stride)

    def row(self, position, pool):
        if not 0 <= position < self.logical_end:
            raise ValueError("Position outside authorized draft lease")
        if pool == "FULL":
            return self.full_row0 + position
        if pool == "WINDOW" and position >= self.window_floor:
            return self.window_row0 + position % self.window_stride
        raise ValueError("Position outside authorized draft window")


def plan_region_copy(source, destination, source_rows, positions, lease):
    """Return bounded contiguous RDMA blocks for exactly one layer and K/V shard."""
    lease.validate()
    if source.key != destination.key or source.row_stride != destination.row_stride:
        raise ValueError("Draft source/destination region mismatch")
    source_rows, positions = list(source_rows), list(positions)
    if len(source_rows) != len(positions) or positions != sorted(set(positions)):
        raise ValueError("Draft source position manifest is ambiguous")
    blocks = []
    prior_src = prior_dst = None
    for source_row, position in zip(source_rows, positions):
        if source.pool == "WINDOW" and position < lease.window_floor:
            continue
        destination_row = lease.row(position, destination.pool)
        source_address = source.address(source_row)
        destination_address = destination.address(destination_row)
        if blocks and source_row == prior_src + 1 and destination_row == prior_dst + 1:
            src, dst, size = blocks[-1]
            blocks[-1] = (src, dst, size + source.row_stride)
        else:
            blocks.append((source_address, destination_address, source.row_stride))
        prior_src, prior_dst = source_row, destination_row
    return blocks


def destination_head_shards(peers, source_head_begin, source_head_end):
    """Select complete destination shards owned by one context-encoder rank.

    Refining TP2 into TP4 gives each destination exactly one writer. Coarsening
    would require per-head completion accounting and is deliberately rejected.
    """
    peers = validate_cohort(peers)
    if not 0 <= source_head_begin < source_head_end <= peers[0].total_heads:
        raise ValueError("Invalid source draft head interval")
    selected = []
    for peer in peers:
        if peer.head_end <= source_head_begin or peer.head_begin >= source_head_end:
            continue
        if not (
            source_head_begin <= peer.head_begin < peer.head_end <= source_head_end
        ):
            raise ValueError("Destination draft head shard crosses source ownership")
        selected.append(peer)
    if (
        not selected
        or selected[0].head_begin != source_head_begin
        or selected[-1].head_end != source_head_end
    ):
        raise ValueError("Destination draft shards do not cover source heads")
    return tuple(selected)


def plan_region_head_copy(
    source,
    destination,
    source_rows,
    positions,
    lease,
    *,
    source_head_begin,
    source_head_end,
    destination_head_begin,
    destination_head_end,
    head_dim,
):
    """Bounded NHD sub-head copies, with the equal-TP contiguous fast path."""
    lease.validate()
    if source.key != destination.key or head_dim <= 0:
        raise ValueError("Draft head copy region mismatch")
    if not (
        0
        <= source_head_begin
        <= destination_head_begin
        < destination_head_end
        <= source_head_end
    ):
        raise ValueError("Destination draft heads exceed source ownership")
    head_bytes = head_dim * 2  # Wire v1 permits only unscaled BF16/FP16.
    width = (destination_head_end - destination_head_begin) * head_bytes
    if (
        source.row_stride != (source_head_end - source_head_begin) * head_bytes
        or destination.row_stride != width
    ):
        raise ValueError("Draft head copy stride does not match head geometry")
    offset = (destination_head_begin - source_head_begin) * head_bytes
    source_rows, positions = list(source_rows), list(positions)
    if len(source_rows) != len(positions) or positions != sorted(set(positions)):
        raise ValueError("Draft source position manifest is ambiguous")
    blocks = []
    for source_row, position in zip(source_rows, positions):
        if source.pool == "WINDOW" and position < lease.window_floor:
            continue
        src = source.address(source_row) + offset
        dst = destination.address(lease.row(position, destination.pool))
        if (
            blocks
            and blocks[-1][0] + blocks[-1][2] == src
            and blocks[-1][1] + blocks[-1][2] == dst
        ):
            old_src, old_dst, size = blocks[-1]
            blocks[-1] = (old_src, old_dst, size + width)
        else:
            blocks.append((src, dst, width))
    return blocks


class DraftLeaseState:
    """Scheduler-thread state. An uncertain transport never releases a slot."""

    def __init__(self, lease, peer):
        self.lease = lease.validate()
        self.peer = peer.validate()
        if lease.cohort_epoch != peer.cohort_epoch:
            raise ValueError("Stale draft cohort lease")
        self.phase = "RESERVED"
        self.coverage = {r.key: [] for r in peer.regions}
        self.chunks = set()
        self.readers_drained = False
        self.transport_drained = False

    def transferred(self, chunk_id, ranges):
        if chunk_id in self.chunks:
            return False
        if self.phase != "RESERVED":
            raise RuntimeError(
                "Cannot write a ready, retired, or quarantined draft lease"
            )
        for key, start, end in ranges:
            if (
                key not in self.coverage
                or not 0 <= start < end <= self.lease.logical_end
            ):
                raise ValueError("Invalid draft transfer coverage")
            if key[2] == "WINDOW" and start < self.lease.window_floor:
                raise ValueError("Window transfer exceeds the authorized tail")
        for key, start, end in ranges:
            self.coverage[key].append((start, end))
        self.chunks.add(chunk_id)
        return True

    def mark_ready(self):
        if self.phase == "READY":
            return
        if self.phase != "RESERVED":
            raise RuntimeError("Retired/quarantined draft lease cannot become ready")
        for key, intervals in self.coverage.items():
            cursor = self.lease.window_floor if key[2] == "WINDOW" else 0
            for start, end in sorted(intervals):
                if start > cursor:
                    raise RuntimeError("Draft context has missing transfer ranges")
                cursor = max(cursor, end)
            if cursor < self.lease.logical_end:
                raise RuntimeError("Draft context is incomplete")
        self.phase = "READY"

    def retire(self, *, readers_drained, transport_drained):
        self.readers_drained |= readers_drained
        self.transport_drained |= transport_drained
        self.phase = (
            "RELEASED"
            if self.readers_drained and self.transport_drained
            else "QUARANTINED"
        )
        return self.phase == "RELEASED"


def encode_control(kind, payload):
    return [
        CONTROL_TAG,
        json.dumps(
            {"version": WIRE_VERSION, "kind": kind, "payload": payload},
            separators=(",", ":"),
        ).encode(),
    ]


def decode_control(frames):
    if not frames or frames[0] != CONTROL_TAG:
        return None
    if len(frames) != 2:
        raise ValueError("Malformed draft control envelope")
    record = json.loads(frames[1])
    if record.get("version") != WIRE_VERSION or not isinstance(
        record.get("payload"), dict
    ):
        raise ValueError("Unsupported/malformed draft control envelope")
    if not isinstance(record.get("kind"), str):
        raise ValueError("Missing draft control message type")
    return record


def peer_to_dict(peer):
    return asdict(peer.validate())


def peer_from_dict(record):
    record = dict(record)
    record["regions"] = tuple(DraftRegion(**r) for r in record["regions"])
    return DraftPeer(**record).validate()
