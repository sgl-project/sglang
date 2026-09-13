# SPDX-License-Identifier: Apache-2.0
"""Runtime reader and VRAM cache for SGLANG-EXPERTPACK-v1."""

from __future__ import annotations

import atexit
import concurrent.futures
import hashlib
import json
import logging
import os
import re
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

MAGIC = b"SGLANG-EXPERTPACK-v1\0\0\0\0"
ROLE_NAMES = ("gate", "up", "down")
HEADER_STRUCT = struct.Struct("<24sIIIIQQQIIII32s32s32s")
ENTRY_STRUCT = struct.Struct("<HHBBH16s80sQQQQQQ32s32s32s4Q16s16sQQ")
REQUIRED_FLAGS = (1 << 0) | (1 << 1)
READ_SPLITS = 4
KIMI_FORMAT = "SGLANG-KIMI-GGMLMOEPACK-ADAPTER-v1"
GGML_PACK_MAGIC = b"GGMLMOEPACKv1\0\0\0"
GGML_PACK_HEADER = struct.Struct("<16sIIQQ")
GGML_PACK_ENTRY = struct.Struct("<128siIQQ")
KIMI_PHYSICAL_ROLES = ("up", "gate", "down")
KIMI_EXPERT_RE = re.compile(
    r"^blk\.(?P<layer>\d+)\.ffn_(?P<role>up|gate|down)_exps\.weight$"
)


def _fixed_string(value: bytes) -> str:
    return value.split(b"\0", 1)[0].decode("utf-8")


def _sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ExpertPackHeader:
    flags: int
    index_count: int
    data_start: int
    alignment: int
    num_layers: int
    num_experts: int
    top_k: int
    role_count: int
    model_identity_sha256: str
    source_blob_sha256: str
    config_sha256: str

    @classmethod
    def read(cls, stream) -> ExpertPackHeader:
        raw = stream.read(HEADER_STRUCT.size)
        if len(raw) != HEADER_STRUCT.size:
            raise ValueError("expert-pack header is truncated")
        values = HEADER_STRUCT.unpack(raw)
        if values[0] != MAGIC or values[1] != 1:
            raise ValueError("expert-pack magic or version does not match")
        if values[2] != HEADER_STRUCT.size or values[3] != ENTRY_STRUCT.size:
            raise ValueError("expert-pack struct sizes do not match")
        header = cls(
            flags=values[4],
            index_count=values[5],
            data_start=values[6],
            alignment=values[7],
            num_layers=values[8],
            num_experts=values[9],
            top_k=values[10],
            role_count=values[11],
            model_identity_sha256=values[12].hex(),
            source_blob_sha256=values[13].hex(),
            config_sha256=values[14].hex(),
        )
        expected = header.num_layers * header.num_experts * len(ROLE_NAMES)
        if header.index_count != expected or header.role_count != len(ROLE_NAMES):
            raise ValueError("expert-pack header coverage is inconsistent")
        if header.flags & REQUIRED_FLAGS != REQUIRED_FLAGS:
            raise ValueError("expert-pack is not identity triplet layout")
        if header.alignment <= 0 or header.alignment & (header.alignment - 1):
            raise ValueError("expert-pack alignment is invalid")
        minimum = HEADER_STRUCT.size + header.index_count * ENTRY_STRUCT.size
        if header.data_start < minimum or header.data_start % header.alignment:
            raise ValueError("expert-pack data offset is invalid")
        return header


@dataclass(frozen=True)
class ExpertPackEntry:
    layer: int
    expert: int
    role_id: int
    dtype_id: int
    dtype: str
    tensor_name: str
    source_slice_offset: int
    source_slice_nbytes: int
    pack_offset: int
    pack_nbytes: int
    checksum: str
    shape: tuple[int, ...]
    quant_scheme: str
    transform_id: str
    block_size: int
    generation: int

    @classmethod
    def read(cls, stream) -> ExpertPackEntry:
        raw = stream.read(ENTRY_STRUCT.size)
        if len(raw) != ENTRY_STRUCT.size:
            raise ValueError("expert-pack index is truncated")
        values = ENTRY_STRUCT.unpack(raw)
        role_id, rank = values[2], values[3]
        if role_id >= len(ROLE_NAMES) or not 1 <= rank <= 4:
            raise ValueError("expert-pack index role or rank is invalid")
        return cls(
            layer=values[0],
            expert=values[1],
            role_id=role_id,
            dtype_id=values[4],
            dtype=_fixed_string(values[5]),
            tensor_name=_fixed_string(values[6]),
            source_slice_offset=values[9],
            source_slice_nbytes=values[10],
            pack_offset=values[11],
            pack_nbytes=values[12],
            checksum=values[15].hex(),
            shape=tuple(values[16 : 16 + rank]),
            quant_scheme=_fixed_string(values[20]),
            transform_id=_fixed_string(values[21]),
            block_size=values[22],
            generation=values[23],
        )


@dataclass
class _CacheSlot:
    key: tuple[int, int] | None = None
    generation: int = 0
    frequency: int = 0
    last_use: torch.cuda.Event | None = None
    ready: torch.cuda.Event | None = None


def _initialize_runtime_state(
    store,
    *,
    cache_vram_mib: int,
    cache_vram_reserve_mib: int,
    stage_slots: int,
    read_splits: int,
    direct_io: bool,
    stats_flush_interval: int,
    stats_path: str | os.PathLike[str] | None,
) -> None:
    store.cache_vram_mib = int(cache_vram_mib)
    store.cache_vram_reserve_mib = int(cache_vram_reserve_mib)
    store.kernel_backend = "custom"
    store.stage_slot_count = int(stage_slots)
    store.read_splits = int(read_splits)
    store.direct_io = bool(direct_io)
    store.stats_flush_interval = int(stats_flush_interval)
    if (
        store.cache_vram_mib <= 0
        or store.cache_vram_reserve_mib <= 0
        or store.stage_slot_count <= 0
        or store.read_splits <= 0
    ):
        raise ValueError(
            "expert cache and staging budgets, and read splits, must be positive"
        )
    if store.stats_flush_interval < 0:
        raise ValueError("expert-pack stats flush interval cannot be negative")
    if store.direct_io and not hasattr(os, "O_DIRECT"):
        raise ValueError("expert-pack direct I/O is unavailable on this platform")
    open_flags = os.O_RDONLY | (os.O_DIRECT if store.direct_io else 0)
    store._fd = os.open(store.path, open_flags)
    store._lock = threading.RLock()
    store._cache = None
    store._cache_slots = []
    store._key_to_slot = {}
    store._key_frequency = {}
    store._lru = OrderedDict()
    store._staging = []
    store._stage_events = []
    store._stage_cursor = 0
    store._transfer_stream = None
    store._read_executor = None
    store._active_keys = set()
    store._route_calls_by_layer = [0] * store.header.num_layers
    store._route_tokens_by_layer = [0] * store.header.num_layers
    store.stats_path = Path(stats_path).resolve() if stats_path else None
    store._last_stats_flush_calls = 0
    store.stats = {
        "pack_path": str(store.path),
        "pack_entries": len(store.entries),
        "pack_reads": 0,
        "pack_read_bytes": 0,
        "pack_read_ns": 0,
        "read_splits": store.read_splits,
        "direct_io": store.direct_io,
        "cache_hits": 0,
        "cache_misses": 0,
        "cache_evictions": 0,
        "cache_policy": "reuse-lfu-lru-v2",
        "kernel_backend": "custom",
        "resident_experts": 0,
        "resident_bytes": 0,
        "h2d_bytes": 0,
        "cache_vram_reserve_mib": store.cache_vram_reserve_mib,
        "fallback_count": 0,
        "io_errors": 0,
    }
    atexit.register(store.close)


class ExpertPackStore:
    """Validated pack index plus generation-aware GPU and host caches."""

    def __init__(
        self,
        pack_path: str | os.PathLike[str],
        *,
        manifest_path: str | os.PathLike[str] | None = None,
        expected_layers: int,
        expected_experts: int,
        expected_top_k: int,
        expected_source_sha256: str | None = None,
        expected_model_identity_sha256: str | None = None,
        expected_config_sha256: str | None = None,
        cache_vram_mib: int = 20 * 1024,
        cache_vram_reserve_mib: int = 3 * 1024,
        stage_slots: int = 8,
        read_splits: int = READ_SPLITS,
        direct_io: bool = False,
        stats_flush_interval: int = 0,
        verify_pack_sha256: bool = False,
        stats_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.path = Path(pack_path).resolve()
        self.manifest_path = Path(
            manifest_path or str(self.path) + ".manifest.json"
        ).resolve()
        if not self.path.is_file() or not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"expert-pack or manifest is missing: {self.path}, {self.manifest_path}"
            )
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not self.manifest.get("complete"):
            raise ValueError("expert-pack manifest is not complete")

        with self.path.open("rb", buffering=0) as stream:
            self.header = ExpertPackHeader.read(stream)
            entries = [
                ExpertPackEntry.read(stream) for _ in range(self.header.index_count)
            ]

        expected_dimensions = (expected_layers, expected_experts, expected_top_k)
        actual_dimensions = (
            self.header.num_layers,
            self.header.num_experts,
            self.header.top_k,
        )
        if actual_dimensions != expected_dimensions:
            raise ValueError(
                f"expert-pack dimensions {actual_dimensions} != {expected_dimensions}"
            )
        expected_digests = {
            "source_blob_sha256": expected_source_sha256,
            "model_identity_sha256": expected_model_identity_sha256,
            "config_sha256": expected_config_sha256,
        }
        for field, expected in expected_digests.items():
            if expected and getattr(self.header, field) != expected:
                raise ValueError(
                    f"expert-pack {field} does not match configured digest"
                )

        self.entries: dict[tuple[int, int, int], ExpertPackEntry] = {}
        role_bytes: int | None = None
        object_generations: dict[tuple[int, int], int] = {}
        for entry in entries:
            key = (entry.layer, entry.expert, entry.role_id)
            if key in self.entries:
                raise ValueError(f"duplicate expert-pack entry {key}")
            if entry.dtype != "MXFP4" or entry.quant_scheme != "MXFP4":
                raise ValueError(f"unsupported expert dtype for {key}: {entry.dtype}")
            if entry.transform_id != "identity-v1" or entry.block_size != 32:
                raise ValueError(f"unsupported expert transform for {key}")
            if entry.pack_nbytes != entry.source_slice_nbytes:
                raise ValueError(f"non-identity expert payload size for {key}")
            if role_bytes is None:
                role_bytes = entry.pack_nbytes
            elif role_bytes != entry.pack_nbytes:
                raise ValueError("expert-pack roles are not fixed size")
            object_key = key[:2]
            generation = object_generations.setdefault(object_key, entry.generation)
            if generation != entry.generation:
                raise ValueError(f"mixed generation in expert object {object_key}")
            self.entries[key] = entry

        assert role_bytes is not None
        self.role_bytes = role_bytes
        self.object_payload_bytes = role_bytes * len(ROLE_NAMES)
        self.object_stride = int(self.manifest["object_stride"])
        if self.object_stride < self.object_payload_bytes:
            raise ValueError("expert-pack object stride is smaller than its payload")
        expected_size = self.header.data_start + (
            expected_layers * expected_experts * self.object_stride
        )
        if self.path.stat().st_size != expected_size:
            raise ValueError("expert-pack file size does not match its index")
        self.object_offsets: dict[tuple[int, int], int] = {}
        self.active_moe_layer_ids = frozenset(range(expected_layers))
        for layer in range(expected_layers):
            for expert in range(expected_experts):
                object_offset = (
                    self.header.data_start
                    + (layer * expected_experts + expert) * self.object_stride
                )
                self.object_offsets[(layer, expert)] = object_offset
                for role_id in range(len(ROLE_NAMES)):
                    entry = self.entries[(layer, expert, role_id)]
                    if entry.pack_offset != object_offset + role_id * role_bytes:
                        raise ValueError(
                            f"expert-pack object layout mismatch at {(layer, expert, role_id)}"
                        )

        manifest_pack_sha = self.manifest.get("pack_sha256")
        if verify_pack_sha256:
            actual_pack_sha = _sha256_file(self.path)
            if actual_pack_sha != manifest_pack_sha:
                raise ValueError("expert-pack SHA-256 does not match its manifest")
        self.pack_sha256 = str(manifest_pack_sha)
        self.role_offsets = {
            role: role_id * self.role_bytes for role_id, role in enumerate(ROLE_NAMES)
        }
        self.role_nbytes = {role: self.role_bytes for role in ROLE_NAMES}
        _initialize_runtime_state(
            self,
            cache_vram_mib=cache_vram_mib,
            cache_vram_reserve_mib=cache_vram_reserve_mib,
            stage_slots=stage_slots,
            read_splits=read_splits,
            direct_io=direct_io,
            stats_flush_interval=stats_flush_interval,
            stats_path=stats_path,
        )

    def initialize_device_cache(self, device: torch.device | str) -> None:
        if self._cache is not None:
            return
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("expert-pack runtime currently requires CUDA")
        requested = self.cache_vram_mib * 1024 * 1024
        free_bytes, _ = torch.cuda.mem_get_info(device)
        reserve = self.cache_vram_reserve_mib * 1024 * 1024
        budget = min(requested, max(0, free_bytes - reserve))
        slot_count = max(0, budget // self.object_payload_bytes)
        if slot_count < self.header.top_k:
            raise MemoryError(
                "insufficient free VRAM for one top-k expert working set: "
                f"free={free_bytes}, object={self.object_payload_bytes}"
            )
        cache_bytes = slot_count * self.object_payload_bytes
        self._cache = torch.empty(
            (slot_count, self.object_payload_bytes),
            dtype=torch.uint8,
            device=device,
        )
        self._cache_slots = [_CacheSlot() for _ in range(slot_count)]
        self._staging = [
            torch.empty(self.object_payload_bytes, dtype=torch.uint8, pin_memory=True)
            for _ in range(self.stage_slot_count)
        ]
        if self.direct_io:
            alignment = 4096
            ranges = self._object_read_ranges()
            if any(offset % alignment for offset in self.object_offsets.values()):
                raise ValueError("expert-pack direct I/O requires aligned objects")
            if any(start % alignment or length % alignment for start, length in ranges):
                raise ValueError("expert-pack direct I/O requires aligned read ranges")
            if any(staging.data_ptr() % alignment for staging in self._staging):
                raise ValueError("expert-pack direct I/O requires aligned staging")
        self._stage_events = [None] * self.stage_slot_count
        self._transfer_stream = torch.cuda.Stream(device=device)
        self._read_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.stage_slot_count * self.read_splits,
            thread_name_prefix="expert-pack-read",
        )
        self.stats["cache_capacity_experts"] = slot_count
        self.stats["cache_capacity_bytes"] = cache_bytes
        staged_bytes = self.stage_slot_count * self.object_payload_bytes
        self.stats["staged_bytes"] = staged_bytes
        logger.info(
            "Expert pack ready: entries=%d resident_experts=0 dense_bytes=external "
            "staged_bytes=%d cache_capacity_experts=%d cache_capacity_bytes=%d",
            len(self.entries),
            staged_bytes,
            slot_count,
            cache_bytes,
        )

    def _read_object(
        self, layer: int, expert: int, staging: torch.Tensor
    ) -> tuple[int, int]:
        return self._read_object_range(
            layer, expert, staging, start=0, length=self.object_payload_bytes
        )

    def _read_object_range(
        self,
        layer: int,
        expert: int,
        staging: torch.Tensor,
        *,
        start: int,
        length: int,
    ) -> tuple[int, int]:
        try:
            object_offset = self.object_offsets[(layer, expert)]
        except KeyError as exc:
            raise ValueError(f"no expert-pack object for {(layer, expert)}") from exc
        offset = object_offset + start
        view = memoryview(staging.numpy()).cast("B")[start : start + length]
        started = time.perf_counter_ns()
        read_bytes = os.preadv(self._fd, [view], offset)
        elapsed = time.perf_counter_ns() - started
        if read_bytes != length:
            raise OSError(
                f"short expert-pack read for {(layer, expert, start, length)}: "
                f"{read_bytes} != {length}"
            )
        return read_bytes, elapsed

    def _object_read_ranges(self) -> list[tuple[int, int]]:
        split_count = self.read_splits
        alignment = 4096 if self.object_payload_bytes >= split_count * 4096 else 1
        boundaries = [
            self.object_payload_bytes * part // split_count // alignment * alignment
            for part in range(split_count)
        ] + [self.object_payload_bytes]
        return [
            (boundaries[part], boundaries[part + 1] - boundaries[part])
            for part in range(split_count)
        ]

    def _victim_slot(
        self,
        protected: set[tuple[int, int]],
        *,
        preserve_oldest: bool = False,
    ) -> int:
        for index, slot in enumerate(self._cache_slots):
            if slot.key is None:
                return index
        keys = reversed(self._lru) if preserve_oldest else iter(self._lru)
        victim_index = None
        victim_frequency = None
        for key in keys:
            if key in protected:
                continue
            slot_index = self._key_to_slot[key]
            frequency = self._cache_slots[slot_index].frequency
            if victim_frequency is None or frequency < victim_frequency:
                victim_index = slot_index
                victim_frequency = frequency
        if victim_index is not None:
            return victim_index
        raise RuntimeError("expert cache cannot evict the active top-k working set")

    def _install_staging(
        self,
        staging: torch.Tensor,
        slot_index: int,
        stream: torch.cuda.Stream,
    ) -> torch.cuda.Event:
        """Publish one host object into the custom GPU cache."""
        with torch.cuda.stream(stream):
            assert self._cache is not None
            self._cache[slot_index].copy_(staging, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(stream)
        return ready

    def _record_read(self, read_bytes: int, elapsed: int) -> None:
        self.stats["pack_reads"] = int(self.stats["pack_reads"]) + 1
        self.stats["pack_read_bytes"] = int(self.stats["pack_read_bytes"]) + read_bytes
        self.stats["pack_read_ns"] = int(self.stats["pack_read_ns"]) + elapsed

    def acquire(
        self, layer: int, topk_ids: torch.Tensor, *, is_prefill: bool | None = None
    ) -> tuple[torch.Tensor, list[int]]:
        if (
            self._cache is None
            or self._transfer_stream is None
            or self._read_executor is None
        ):
            raise RuntimeError("expert device cache is not initialized")
        if layer not in self.active_moe_layer_ids:
            raise ValueError(f"layer {layer} is not an active routed MoE layer")
        if topk_ids.ndim != 2 or topk_ids.shape[-1] != self.header.top_k:
            raise ValueError(
                f"runtime top-k must be exactly {self.header.top_k}; "
                f"received shape {tuple(topk_ids.shape)}"
            )
        route_ids = [int(value) for value in topk_ids.detach().cpu().reshape(-1)]
        if is_prefill is None:
            is_prefill = topk_ids.shape[0] > 1
        if any(expert < 0 or expert >= self.header.num_experts for expert in route_ids):
            raise ValueError("route contains an out-of-range expert id")
        requested = {(layer, expert) for expert in route_ids}
        events: list[torch.cuda.Event] = []

        with self._lock:
            for key in requested:
                self._key_frequency[key] = min(self._key_frequency.get(key, 0) + 1, 255)
            self._active_keys.update(requested)
            self._route_calls_by_layer[layer] += 1
            self._route_tokens_by_layer[layer] += int(topk_ids.shape[0])
            pending: list[tuple[tuple[int, int], int]] = []
            requested_keys = sorted(
                dict.fromkeys((layer, expert) for expert in route_ids)
            )
            for key in requested_keys:
                slot_index = self._key_to_slot.get(key)
                generation = self.entries[(key[0], key[1], 0)].generation
                if (
                    slot_index is not None
                    and self._cache_slots[slot_index].generation == generation
                ):
                    self.stats["cache_hits"] = int(self.stats["cache_hits"]) + 1
                    self._lru.move_to_end(key)
                    slot = self._cache_slots[slot_index]
                    slot.frequency = self._key_frequency[key]
                    if slot.ready is not None:
                        events.append(slot.ready)
                    continue

                self.stats["cache_misses"] = int(self.stats["cache_misses"]) + 1
                if slot_index is None:
                    slot_index = self._victim_slot(
                        requested, preserve_oldest=is_prefill
                    )
                slot = self._cache_slots[slot_index]
                if slot.key is not None:
                    self.stats["cache_evictions"] = (
                        int(self.stats["cache_evictions"]) + 1
                    )
                    self._key_to_slot.pop(slot.key, None)
                    self._lru.pop(slot.key, None)
                slot.key = key
                slot.generation = generation
                slot.frequency = self._key_frequency[key]
                self._key_to_slot[key] = slot_index
                self._lru[key] = None
                pending.append((key, slot_index))

            for batch_start in range(0, len(pending), len(self._staging)):
                batch = pending[batch_start : batch_start + len(self._staging)]
                jobs = []
                for key, slot_index in batch:
                    stage_index = self._stage_cursor
                    self._stage_cursor = (self._stage_cursor + 1) % len(self._staging)
                    stage_event = self._stage_events[stage_index]
                    if stage_event is not None:
                        stage_event.synchronize()
                    staging = self._staging[stage_index]
                    futures = tuple(
                        self._read_executor.submit(
                            self._read_object_range,
                            key[0],
                            key[1],
                            staging,
                            start=start,
                            length=length,
                        )
                        for start, length in self._object_read_ranges()
                    )
                    jobs.append((futures, key, stage_index, staging, slot_index))

                for futures, key, stage_index, staging, slot_index in jobs:
                    if futures:
                        try:
                            results = [future.result() for future in futures]
                        except OSError:
                            self.stats["io_errors"] = int(self.stats["io_errors"]) + 1
                            raise
                        read_bytes = sum(result[0] for result in results)
                        elapsed = max(result[1] for result in results)
                        self._record_read(read_bytes, elapsed)
                    slot = self._cache_slots[slot_index]
                    if slot.ready is not None:
                        self._transfer_stream.wait_event(slot.ready)
                    if slot.last_use is not None:
                        self._transfer_stream.wait_event(slot.last_use)
                    ready = self._install_staging(
                        staging,
                        slot_index,
                        self._transfer_stream,
                    )
                    self._stage_events[stage_index] = ready
                    events.append(ready)
                    slot.last_use = None
                    slot.ready = ready
                    self.stats["h2d_bytes"] = int(self.stats["h2d_bytes"]) + (
                        self.object_payload_bytes
                    )

            cache_device = self._cache.device
            current_stream = torch.cuda.current_stream(cache_device)
            for event in events:
                current_stream.wait_event(event)
            slots = [self._key_to_slot[(layer, expert)] for expert in route_ids]
            self.stats["resident_experts"] = len(self._key_to_slot)
            self.stats["resident_bytes"] = (
                len(self._key_to_slot) * self.object_payload_bytes
            )
        return (
            torch.tensor(slots, dtype=torch.int32, device=cache_device),
            slots,
        )

    def mark_used(self, slot_indices: list[int]) -> None:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        with self._lock:
            for slot_index in set(slot_indices):
                slot = self._cache_slots[slot_index]
                slot.last_use = event
                slot.ready = None
                if slot.key is not None:
                    self._active_keys.discard(slot.key)
            route_calls = sum(self._route_calls_by_layer)
            if (
                self.stats_path is not None
                and self.stats_flush_interval
                and route_calls - self._last_stats_flush_calls
                >= self.stats_flush_interval
            ):
                self._write_stats()
                self._last_stats_flush_calls = route_calls

    @property
    def device_cache(self) -> torch.Tensor:
        if self._cache is None:
            raise RuntimeError("raw expert device cache is not initialized")
        return self._cache

    def snapshot(self) -> dict[str, Any]:
        value = dict(self.stats)
        reads = int(value["pack_reads"])
        value["mean_read_ms"] = (
            int(value["pack_read_ns"]) / reads / 1e6 if reads else 0.0
        )
        value["route_calls_by_layer"] = list(self._route_calls_by_layer)
        value["route_tokens_by_layer"] = list(self._route_tokens_by_layer)
        return value

    def _write_stats(self) -> None:
        assert self.stats_path is not None
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.stats_path.with_name(
            self.stats_path.name + f".{os.getpid()}.tmp"
        )
        temporary.write_text(
            json.dumps(self.snapshot(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.stats_path)

    def close(self) -> None:
        if self._read_executor is not None:
            self._read_executor.shutdown(wait=True)
            self._read_executor = None
        if self.stats_path is not None:
            self._write_stats()
        if getattr(self, "_fd", -1) >= 0:
            os.close(self._fd)
            self._fd = -1


class KimiGGMLExpertPackStore(ExpertPackStore):
    """Runtime cache for the audited, zero-copy Kimi GGMLMOEPACKv1 layout."""

    def __init__(
        self,
        pack_path: str | os.PathLike[str],
        *,
        manifest_path: str | os.PathLike[str],
        expected_layers: int,
        expected_experts: int,
        expected_top_k: int,
        cache_vram_mib: int = 18 * 1024,
        cache_vram_reserve_mib: int = 3 * 1024,
        stage_slots: int = 16,
        read_splits: int = READ_SPLITS,
        direct_io: bool = False,
        stats_flush_interval: int = 0,
        verify_pack_sha256: bool = False,
        stats_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.path = Path(pack_path).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        if not self.path.is_file() or not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Kimi expert-pack or manifest is missing: "
                f"{self.path}, {self.manifest_path}"
            )
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if (
            not self.manifest.get("complete")
            or self.manifest.get("format") != KIMI_FORMAT
        ):
            raise ValueError("Kimi expert-pack manifest is incomplete or unsupported")

        constraints = self.manifest.get("hard_constraints", {})
        if constraints != {
            "all_selected_experts_must_execute": True,
            "expert_pruning_allowed": False,
            "requantization_allowed": False,
            "top_k": 16,
            "top_k_is_immutable": True,
        }:
            raise ValueError(
                "Kimi manifest hard constraints do not match runtime policy"
            )
        model = self.manifest["model"]
        dimensions = (
            int(model["num_hidden_layers"]),
            int(model["num_experts"]),
            int(model["num_experts_per_token"]),
        )
        expected_dimensions = (expected_layers, expected_experts, expected_top_k)
        if dimensions != expected_dimensions or expected_top_k != 16:
            raise ValueError(
                f"Kimi expert-pack dimensions {dimensions} != {expected_dimensions}; "
                "Top-K is immutable at 16"
            )
        active_layers = tuple(int(value) for value in model["active_moe_layer_ids"])
        if active_layers != tuple(range(1, 93)):
            raise ValueError("Kimi active routed MoE layers must be exactly 1..92")

        pack_manifest = self.manifest["expert_pack"]
        if Path(pack_manifest["path"]).resolve() != self.path:
            raise ValueError("Kimi manifest expert-pack path does not match pack_path")
        if int(pack_manifest["size"]) != self.path.stat().st_size:
            raise ValueError("Kimi expert-pack size does not match its manifest")
        if pack_manifest.get("physical_role_order") != list(KIMI_PHYSICAL_ROLES):
            raise ValueError("Kimi expert-pack physical role order is unsupported")
        roles = pack_manifest["roles"]
        expected_roles = {
            "up": ("Q2_K", 10),
            "gate": ("Q2_K", 10),
            "down": ("Q3_K", 11),
        }
        for role, (dtype, dtype_id) in expected_roles.items():
            if (
                roles[role]["dtype"] != dtype
                or int(roles[role]["dtype_id"]) != dtype_id
            ):
                raise ValueError(f"Kimi expert-pack {role} quant type is unsupported")

        expected_entry_count = len(active_layers) * expected_experts * len(ROLE_NAMES)
        index_digest = hashlib.sha256()
        self.entries: dict[tuple[int, int, int], ExpertPackEntry] = {}
        self.object_offsets: dict[tuple[int, int], int] = {}
        object_payload_bytes = int(pack_manifest["object_bytes"])
        previous_end = int(pack_manifest["data_start"])
        role_offsets: dict[str, int] = {}
        role_nbytes = {
            role: int(roles[role]["expert_bytes"]) for role in KIMI_PHYSICAL_ROLES
        }
        running_role_offset = 0
        for role in KIMI_PHYSICAL_ROLES:
            role_offsets[role] = running_role_offset
            running_role_offset += role_nbytes[role]
        if running_role_offset != object_payload_bytes:
            raise ValueError("Kimi expert-pack role sizes do not match object bytes")

        with self.path.open("rb", buffering=0) as stream:
            raw_header = stream.read(GGML_PACK_HEADER.size)
            if len(raw_header) != GGML_PACK_HEADER.size:
                raise ValueError("Kimi expert-pack header is truncated")
            index_digest.update(raw_header)
            magic, version, header_size, index_count, data_start = (
                GGML_PACK_HEADER.unpack(raw_header)
            )
            if (
                magic != GGML_PACK_MAGIC
                or version != 1
                or header_size != GGML_PACK_HEADER.size
                or index_count != expected_entry_count
                or data_start != int(pack_manifest["data_start"])
            ):
                raise ValueError("Kimi expert-pack header does not match its manifest")

            for index in range(index_count):
                raw_entry = stream.read(GGML_PACK_ENTRY.size)
                if len(raw_entry) != GGML_PACK_ENTRY.size:
                    raise ValueError("Kimi expert-pack index is truncated")
                index_digest.update(raw_entry)
                name_raw, expert, reserved, offset, nbytes = GGML_PACK_ENTRY.unpack(
                    raw_entry
                )
                object_index, physical_role_id = divmod(index, len(KIMI_PHYSICAL_ROLES))
                layer_index, expected_expert = divmod(object_index, expected_experts)
                layer = active_layers[layer_index]
                physical_role = KIMI_PHYSICAL_ROLES[physical_role_id]
                name = _fixed_string(name_raw)
                match = KIMI_EXPERT_RE.fullmatch(name)
                if (
                    match is None
                    or int(match.group("layer")) != layer
                    or match.group("role") != physical_role
                    or expert != expected_expert
                    or reserved != 0
                ):
                    raise ValueError(
                        f"Kimi expert-pack identity mismatch at index {index}"
                    )
                expected_nbytes = role_nbytes[physical_role]
                if (
                    nbytes != expected_nbytes
                    or offset % int(pack_manifest["alignment"])
                    or offset < previous_end
                    or offset + nbytes > self.path.stat().st_size
                ):
                    raise ValueError(
                        f"Kimi expert-pack range mismatch at index {index}"
                    )
                previous_end = offset + nbytes
                object_key = (layer, expert)
                if physical_role_id == 0:
                    self.object_offsets[object_key] = offset
                expected_offset = (
                    self.object_offsets[object_key] + role_offsets[physical_role]
                )
                if offset != expected_offset:
                    raise ValueError(
                        f"Kimi expert object is not contiguous at index {index}"
                    )
                generation_bytes = hashlib.sha256(
                    f"{pack_manifest['index_sha256']}:{layer}:{expert}".encode("ascii")
                ).digest()
                generation = int.from_bytes(generation_bytes[:8], "little") or 1
                logical_role_id = ROLE_NAMES.index(physical_role)
                logical_shape = tuple(
                    int(value) for value in roles[physical_role]["logical_shape"]
                )
                self.entries[(layer, expert, logical_role_id)] = ExpertPackEntry(
                    layer=layer,
                    expert=expert,
                    role_id=logical_role_id,
                    dtype_id=int(roles[physical_role]["dtype_id"]),
                    dtype=str(roles[physical_role]["dtype"]),
                    tensor_name=name,
                    source_slice_offset=0,
                    source_slice_nbytes=nbytes,
                    pack_offset=offset,
                    pack_nbytes=nbytes,
                    checksum="",
                    shape=logical_shape,
                    quant_scheme=str(roles[physical_role]["dtype"]),
                    transform_id="identity-v1",
                    block_size=256,
                    generation=generation,
                )

        if previous_end != self.path.stat().st_size:
            raise ValueError("Kimi expert-pack file has trailing or missing bytes")
        actual_index_sha256 = index_digest.hexdigest()
        if actual_index_sha256 != pack_manifest["index_sha256"]:
            raise ValueError("Kimi expert-pack index SHA-256 does not match manifest")
        if verify_pack_sha256:
            expected_sha256 = pack_manifest.get("sha256")
            if not expected_sha256:
                raise ValueError(
                    "full pack verification requested, but manifest has no full SHA-256"
                )
            if _sha256_file(self.path) != expected_sha256:
                raise ValueError("Kimi expert-pack SHA-256 does not match manifest")

        self.header = ExpertPackHeader(
            flags=REQUIRED_FLAGS,
            index_count=expected_entry_count,
            data_start=int(pack_manifest["data_start"]),
            alignment=int(pack_manifest["alignment"]),
            num_layers=expected_layers,
            num_experts=expected_experts,
            top_k=expected_top_k,
            role_count=len(ROLE_NAMES),
            model_identity_sha256="0" * 64,
            source_blob_sha256=str(self.manifest["source"]["inventory_sha256"]),
            config_sha256=str(model["config_sha256"]),
        )
        self.active_moe_layer_ids = frozenset(active_layers)
        self.role_offsets = role_offsets
        self.role_nbytes = role_nbytes
        self.role_bytes = role_nbytes["gate"]
        self.object_payload_bytes = object_payload_bytes
        self.object_stride = object_payload_bytes
        self.pack_sha256 = str(pack_manifest.get("sha256") or actual_index_sha256)
        _initialize_runtime_state(
            self,
            cache_vram_mib=cache_vram_mib,
            cache_vram_reserve_mib=cache_vram_reserve_mib,
            stage_slots=stage_slots,
            read_splits=read_splits,
            direct_io=direct_io,
            stats_flush_interval=stats_flush_interval,
            stats_path=stats_path,
        )
        self.stats["pack_format"] = KIMI_FORMAT
        self.stats["active_moe_layers"] = len(active_layers)
