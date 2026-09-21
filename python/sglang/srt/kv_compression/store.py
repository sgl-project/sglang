"""Fixed-block, byte-budgeted compressed Host storage; no cache policy here.

Logical handles retain their generation. Physical blocks never move while an
object exists. CPU staging is bounded and charged to the same L2 budget.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import itertools
import json
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from sglang.srt.kv_compression.faults import TestFault
from sglang.srt.kv_compression.types import (
    BufferDrainError,
    EncodedPage,
    Lease,
    align_bytes,
)

BLOCK_BYTES = 4096
BATCH_PAGES = 64
logger = logging.getLogger(__name__)
_RECORD = np.dtype(
    [
        (n, "int64")
        for n in (
            "generation",
            "state",
            "first",
            "capacity",
            "length",
            "ref",
            "users",
            "encoding",
        )
    ]
)
_FREE, _RESERVED, _WRITING, _READY, _RETIRED = range(5)


@dataclass(frozen=True)
class HostEncodedPage:
    pool: object
    handle: int
    nbytes: int
    encoding: str
    raw_bytes: int
    raw_sha256: bytes | None


class WriteLease:
    """One immutable mapping, protected until scatter and all DMA have drained."""

    def __init__(self, pool, handles, lengths, blocks, counts):
        self.pool, self.handles, self.lengths = pool, handles, lengths
        self.blocks, self.counts = blocks, counts
        self.written = False
        self.closed = False
        self.quarantined = False

    def scatter(self, stage):
        if self.closed or self.quarantined or self.written:
            raise ValueError("Write lease cannot be reused")
        # Validate the complete mapping before touching the arena. Cancellation
        # may retire the object, but cannot change the leased physical mapping.
        with self.pool.lock:
            actual = self.pool._chains(self.handles)
            if any(not np.array_equal(a, b) for a, b in zip(actual, self.blocks)):
                raise ValueError("Write mapping changed")
        started = time.perf_counter()
        ids = np.concatenate(self.blocks)
        src = np.concatenate(
            [
                np.arange(
                    i * self.pool.blocks_per_page, i * self.pool.blocks_per_page + count
                )
                for i, count in enumerate(self.counts)
            ]
        )
        # One CPU tensor operation across all blocks; no per-block CUDA copies.
        torch.index_select(
            stage.view(-1, BLOCK_BYTES),
            0,
            torch.from_numpy(src),
            out=self.pool.write_scratch[: len(src)],
        )
        self.pool.arena.view(-1, BLOCK_BYTES).index_copy_(
            0,
            torch.from_numpy(ids.astype(np.int64)),
            self.pool.write_scratch[: len(src)],
        )
        self.pool.allocation_stats["scatter_seconds"] += time.perf_counter() - started
        self.written = True
        self.pool.test_fault.check(f"write:{self.handles[0]}", "after_scatter")

    def publish(self, refs, pages, checksums):
        if not self.written or self.closed or self.quarantined:
            raise ValueError("Cannot publish before scatter completion")
        if not (len(refs) == len(pages) == len(checksums) == len(self.handles)):
            raise ValueError("Publication count mismatch")
        self.pool.test_fault.check(f"write:{self.handles[0]}", "before_publish")
        # Validate the whole window first, including duplicate identities.
        with self.pool.lock:
            if len(set(map(int, refs))) != len(refs):
                raise ValueError("Duplicate compressed representation")
            for h, ref, page, checksum, length in zip(
                self.handles, refs, pages, checksums, self.lengths
            ):
                self.pool._validate_publish(h, ref, page.encoding, length, checksum)
            for h, ref, page, checksum, length in zip(
                self.handles, refs, pages, checksums, self.lengths
            ):
                self.pool._publish(h, ref, page.encoding, length, checksum)

    def quarantine(self):
        if not self.quarantined:
            self.quarantined = True
            with self.pool.lock:
                self.pool.quarantined.append(self)

    def close(self):
        if not self.closed and not self.quarantined:
            self.closed = True
            for h in self.handles:
                self.pool._release(h, writer=True)


class CompressedHostKVCache:
    can_use_write_back_jit = False
    requires_complete_restore = True
    page_size = logical_page_size = 1
    layout = "compressed_pages_v2"
    device = "cpu"
    pin_memory = True

    def __init__(
        self,
        raw_page_bytes,
        budget_bytes,
        *,
        pin_memory=True,
        reservation_bytes=None,
        verify=False,
    ):
        if raw_page_bytes <= 0 or budget_bytes <= 0:
            raise ValueError("KV page size and L2 budget must be positive")
        if reservation_bytes is not None and reservation_bytes < raw_page_bytes:
            raise ValueError("L2 reservation cannot be smaller than a raw page")
        self.size_per_token = raw_page_bytes
        self.raw_capacity = align_bytes(raw_page_bytes)
        self.output_bound = align_bytes(
            raw_page_bytes if reservation_bytes is None else reservation_bytes
        )
        self.blocks_per_page = (self.output_bound + BLOCK_BYTES - 1) // BLOCK_BYTES
        self.reservation_bytes = self.blocks_per_page * BLOCK_BYTES
        self.verify, self.budget_bytes, self.pin_memory = (
            verify,
            budget_bytes,
            pin_memory,
        )
        # Two pinned channels. CPU scatter scratch is also bounded and charged;
        # it is pageable and never a DMA source. Index scratch is charged here.
        lane_bytes = BATCH_PAGES * self.reservation_bytes
        self.staging_bytes = 2 * lane_bytes
        self.scratch_bytes = lane_bytes + 2 * BATCH_PAGES * self.blocks_per_page * 8
        remaining = budget_bytes - self.staging_bytes - self.scratch_bytes
        self.size = max(
            1, min(8 * (budget_bytes // self.raw_capacity), remaining // 256)
        )
        self.logical_size = self.page_num = self.size
        index_size = 1 << (2 * self.size - 1).bit_length()
        tables_bytes = (
            self.size * (_RECORD.itemsize + 8 + (32 if verify else 0)) + index_size * 16
        )
        self.block_count = (remaining - tables_bytes) // (BLOCK_BYTES + 12)
        if (
            self.size > np.iinfo(np.int32).max
            or self.block_count > np.iinfo(np.int32).max
        ):
            raise ValueError("L2 block or slot IDs exceed int32 range")
        if self.block_count < self.blocks_per_page:
            raise ValueError(
                "L2 byte budget cannot hold metadata, staging and one reserved page"
            )
        self.records = np.zeros(self.size, dtype=_RECORD)
        self.checksums = np.zeros((self.size, 32), dtype=np.uint8) if verify else None
        self.keys = np.zeros(index_size, dtype=np.int64)
        self.slots = np.zeros(index_size, dtype=np.int64)
        self.free_slots = np.arange(self.size - 1, -1, -1, dtype=np.int64)
        self.next_block = np.full(self.block_count, -1, dtype=np.int32)
        self.block_owner = np.full(self.block_count, -1, dtype=np.int32)
        self.free_blocks = np.arange(self.block_count - 1, -1, -1, dtype=np.int32)
        self.free_block_count, self.free_count = self.block_count, self.size
        self.metadata_bytes = tables_bytes + 12 * self.block_count
        self.arena = torch.empty(
            self.block_count * BLOCK_BYTES, dtype=torch.uint8, pin_memory=pin_memory
        )
        self.read_stage = torch.empty(
            (BATCH_PAGES, self.reservation_bytes),
            dtype=torch.uint8,
            pin_memory=pin_memory,
        )
        self.write_stage = torch.empty_like(self.read_stage, pin_memory=pin_memory)
        self.write_scratch = torch.empty(
            (BATCH_PAGES * self.blocks_per_page, BLOCK_BYTES), dtype=torch.uint8
        )
        self.lock = threading.RLock()
        self.read_lock, self.write_lock = threading.Lock(), threading.Lock()
        self.stage_waiters = self.stage_users = 0
        self.quarantined = []
        self.index_max_probe = 0
        self.live_payload_bytes = self.encoded_bytes = self.reserved_bytes = (
            self.retired_bytes
        ) = 0
        self.active_readers = self.active_writers = 0
        self.allocation_stats = defaultdict(float)
        self.test_fault = TestFault()
        self.trace_identities = (
            os.environ.get("SGLANG_KV_COMPRESSION_TRACE_STORE", "0") == "1"
        )
        self._operations = itertools.count(1)

    def trace_node(self, action, node_id=None, handles=(), refs=(), **details):
        """Optional identities join cache ops to P/D traces without inventing rid."""
        if not self.trace_identities:
            return
        with self.lock:
            info = dict(
                operation_id=next(self._operations),
                action=action,
                node_id=node_id,
                capacity=self.snapshot(),
                **details,
            )
            info["handles"] = list(map(int, handles))
            info["page_refs"] = list(map(int, refs))
            info["generations"] = [int(h) >> 32 for h in handles]
        logger.info("KV_COMPRESSION_STORE %s", json.dumps(info))

    def _bucket(self, ref, insert=False):
        pos = (int(ref) * 11400714819323198485) & (len(self.keys) - 1)
        for probes in range(1, len(self.keys) + 1):
            key = int(self.keys[pos])
            if key == ref or key == 0:
                self.index_max_probe = max(self.index_max_probe, probes)
                return pos if key == ref or insert else None
            pos = (pos + 1) & (len(self.keys) - 1)
        raise RuntimeError("Compressed host index has no empty bucket")

    def _erase_bucket(self, hole):
        mask = len(self.keys) - 1
        scan = (hole + 1) & mask
        while self.keys[scan]:
            home = (int(self.keys[scan]) * 11400714819323198485) & mask
            if ((scan - home) & mask) >= ((scan - hole) & mask):
                self.keys[hole], self.slots[hole] = self.keys[scan], self.slots[scan]
                hole = scan
            scan = (scan + 1) & mask
        self.keys[hole] = self.slots[hole] = 0

    def _record(self, handle):
        slot, generation = int(handle) & 0xFFFFFFFF, int(handle) >> 32
        if slot >= self.size or generation <= 0:
            raise ValueError("Invalid compressed host handle")
        r = self.records[slot]
        if r["state"] == _FREE or r["generation"] != generation:
            raise ValueError("Stale compressed host handle")
        return slot, r

    def _chains(self, handles):
        records = [self._record(h) for h in handles]
        if not records:
            return []
        counts = np.array([int(r["capacity"]) // BLOCK_BYTES for _, r in records])
        if (
            np.any(counts <= 0)
            or np.any(counts > self.blocks_per_page)
            or any(int(r["capacity"]) % BLOCK_BYTES for _, r in records)
        ):
            raise ValueError("Corrupted block capacity")
        ids = np.array([int(r["first"]) for _, r in records])
        slots = np.array([s for s, _ in records])
        result = np.full((len(records), int(counts.max())), -1, dtype=np.int32)
        for col in range(result.shape[1]):
            active = counts > col
            current = ids[active]
            if np.any(current < 0) or np.any(current >= self.block_count):
                raise ValueError("Corrupted block chain")
            if np.any(self.block_owner[current] != slots[active]):
                raise ValueError("Corrupted block ownership")
            result[active, col] = current
            ids[active] = self.next_block[current]
        if np.any(ids != -1):
            raise ValueError("Corrupted block chain termination")
        flat = result[result >= 0]
        if len(np.unique(flat)) != len(flat):
            raise ValueError("Corrupted block chain cycle or alias")
        return [row[:count].copy() for row, count in zip(result, counts)]

    def allocated_size(self):
        with self.lock:
            return self.size - self.free_count

    def available_size(self):
        with self.lock:
            return min(self.free_count, self.free_block_count // self.blocks_per_page)

    def can_reserve(self, count):
        return 0 <= count <= self.available_size()

    def alloc(self, count):
        if count < 0:
            raise ValueError("Negative page count")
        if not count:
            return torch.empty(0, dtype=torch.int64)
        started = time.perf_counter()
        with self.lock:
            acquired = time.perf_counter()
            stats = self.allocation_stats
            stats["allocation_calls"] += 1
            stats["allocation_lock_wait_seconds"] += acquired - started
            try:
                if not self.can_reserve(count):
                    stats["allocation_failures"] += 1
                    return None
                slots = self.free_slots[self.free_count - count : self.free_count][
                    ::-1
                ].copy()
                records = self.records[slots].copy()
                if np.any(records["generation"] >= 0x7FFFFFFF):
                    raise RuntimeError("Compressed host generation exhausted")
                n = count * self.blocks_per_page
                blocks = (
                    self.free_blocks[self.free_block_count - n : self.free_block_count][
                        ::-1
                    ]
                    .copy()
                    .reshape(count, -1)
                )
                if (
                    np.any(blocks < 0)
                    or np.any(blocks >= self.block_count)
                    or len(np.unique(blocks)) != n
                    or np.any(self.block_owner[blocks] != -1)
                ):
                    raise ValueError("Corrupted free block stack")
                records["generation"] += 1
                records["state"], records["first"], records["capacity"] = (
                    _RESERVED,
                    blocks[:, 0],
                    self.reservation_bytes,
                )
                handles = torch.from_numpy((records["generation"] << 32) | slots)
                owners = np.repeat(slots, self.blocks_per_page)
                # Fallible preparation precedes live ownership mutation.
                self.next_block[blocks[:, :-1]] = blocks[:, 1:]
                self.next_block[blocks[:, -1]] = -1
                self.block_owner[blocks.ravel()] = owners
                self.records[slots] = records
                self.free_count -= count
                self.free_block_count -= n
                self.reserved_bytes += n * BLOCK_BYTES
                stats["allocated_pages"] += count
                stats["allocation_blocks"] += n
                return handles
            finally:
                ended = time.perf_counter()
                stats["allocation_seconds"] += ended - started
                stats["allocation_max_lock_seconds"] = max(
                    stats["allocation_max_lock_seconds"], ended - acquired
                )

    def _return_blocks(self, blocks):
        n = len(blocks)
        self.free_blocks[self.free_block_count : self.free_block_count + n] = blocks
        self.block_owner[blocks] = -1
        self.next_block[blocks] = -1
        self.free_block_count += n

    @contextlib.contextmanager
    def _timed_lock(self, operation):
        started = time.perf_counter()
        with self.lock:
            acquired = time.perf_counter()
            self.allocation_stats[operation + "_lock_wait_seconds"] += (
                acquired - started
            )
            try:
                yield
            finally:
                ended = time.perf_counter()
                self.allocation_stats[operation + "_seconds"] += ended - started
                key = operation + "_max_lock_seconds"
                self.allocation_stats[key] = max(
                    self.allocation_stats[key], ended - acquired
                )

    def prepare_write(self, handles, lengths):
        handles, lengths = tuple(map(int, handles)), tuple(map(int, lengths))
        if (
            not 0 < len(handles) <= BATCH_PAGES
            or len(handles) != len(lengths)
            or len(set(handles)) != len(handles)
        ):
            raise ValueError("Invalid write window")
        with self._timed_lock("prepare_write"):
            chains = self._chains(handles)
            counts = [(n + BLOCK_BYTES - 1) // BLOCK_BYTES for n in lengths]
            for h, n in zip(handles, lengths):
                _, r = self._record(h)
                if r["state"] != _RESERVED or not 0 < n <= self.output_bound:
                    raise ValueError("Invalid compressed L2 reservation")
            kept = [chain[:n].copy() for chain, n in zip(chains, counts)]
            returned = np.concatenate([chain[n:] for chain, n in zip(chains, counts)])
            lease = WriteLease(self, handles, lengths, kept, counts)
            for h, n, chain in zip(handles, lengths, kept):
                _, r = self._record(h)
                r["state"], r["length"], r["capacity"] = (
                    _WRITING,
                    n,
                    len(chain) * BLOCK_BYTES,
                )
                r["users"] += 1
                self.next_block[chain[-1]] = -1
            self._return_blocks(returned)
            self.reserved_bytes -= len(returned) * BLOCK_BYTES
            self.active_writers += len(handles)
            return lease

    def _validate_publish(self, handle, ref, encoding, length, checksum):
        _, r = self._record(handle)
        if (
            r["state"] != _WRITING
            or encoding not in ("raw", "lz4")
            or int(ref) <= 0
            or int(ref) > np.iinfo(np.int64).max
        ):
            raise ValueError("Invalid compressed page publication")
        if length != r["length"] or (
            encoding == "raw" and length != self.size_per_token
        ):
            raise ValueError("Invalid compressed page size")
        if self.verify and (not isinstance(checksum, bytes) or len(checksum) != 32):
            raise ValueError("Verified L2 requires a source KV SHA256 per page")
        if self._bucket(ref) is not None:
            raise ValueError("Duplicate compressed representation")

    def _publish(self, handle, ref, encoding, length, checksum):
        slot, r = self._record(handle)
        bucket = self._bucket(ref, insert=True)
        self.reserved_bytes -= int(r["capacity"])
        self.live_payload_bytes += int(r["capacity"])
        self.encoded_bytes += length
        r["ref"], r["encoding"], r["state"] = ref, encoding == "lz4", _READY
        if self.verify:
            self.checksums[slot] = np.frombuffer(checksum, dtype=np.uint8)
        self.keys[bucket], self.slots[bucket] = ref, slot

    def acquire(self, handle, expected_ref=None):
        with self.lock:
            slot, r = self._record(handle)
            if r["state"] != _READY:
                raise ValueError("Compressed page is not ready")
            if expected_ref is not None and int(r["ref"]) != expected_ref:
                raise ValueError(
                    "Compressed handle belongs to another KV materialization"
                )
            page = HostEncodedPage(
                self,
                int(handle),
                int(r["length"]),
                "lz4" if r["encoding"] else "raw",
                self.size_per_token,
                self.checksums[slot].tobytes() if self.verify else None,
            )
            future = concurrent.futures.Future()
            future.set_result(page)
            lease = Lease(future, lambda: self._release(handle), source="host")
            r["users"] += 1
            self.active_readers += 1
            return lease

    def acquire_ref(self, ref, mode):
        with self.lock:
            bucket = self._bucket(ref)
            if bucket is None:
                return None
            slot = int(self.slots[bucket])
            r = self.records[slot]
            if mode == "passthrough" and r["encoding"]:
                return None
            return self.acquire((int(r["generation"]) << 32) | slot, ref)

    def _release(self, handle, writer=False):
        with self.lock:
            slot, r = self._record(handle)
            if r["users"] <= 0:
                raise ValueError("Lease already released")
            blocks = (
                self._chains([handle])[0]
                if r["state"] == _RETIRED and r["users"] == 1
                else None
            )
            r["users"] -= 1
            if writer:
                self.active_writers -= 1
            else:
                self.active_readers -= 1
            if blocks is not None:
                ref = int(r["ref"])
                self.retired_bytes -= int(r["capacity"])
                self._reclaim(slot, r, blocks)
                self.trace_node(
                    "lease_release",
                    handles=[handle],
                    refs=[ref],
                    released_blocks=len(blocks),
                )

    def _reclaim(self, slot, record, blocks):
        self._return_blocks(blocks)
        generation = int(record["generation"])
        self.records[slot] = 0
        self.records[slot]["generation"] = generation
        if self.verify:
            self.checksums[slot].fill(0)
        self.free_slots[self.free_count] = slot
        self.free_count += 1

    def free_matching(self, handle, expected_ref):
        with self.lock:
            try:
                _, r = self._record(handle)
            except ValueError:
                return False
            if r["state"] != _READY or int(r["ref"]) != expected_ref:
                return False
            self.free([handle])
            return True

    def free(self, indices):
        handles = list(map(int, indices))
        if len(set(handles)) != len(handles):
            raise ValueError("Duplicate compressed host free")
        with self._timed_lock("free"):
            records = [self._record(h) for h in handles]
            if any(r["state"] == _RETIRED for _, r in records):
                raise ValueError("Compressed host object already retired")
            chains = self._chains(handles)
            before_blocks = self.free_block_count
            refs = [int(r["ref"]) for _, r in records]
            for (slot, r), blocks in zip(records, chains):
                if r["state"] == _READY:
                    bucket = self._bucket(int(r["ref"]))
                    self._erase_bucket(bucket)
                    self.live_payload_bytes -= int(r["capacity"])
                    self.encoded_bytes -= int(r["length"])
                else:
                    self.reserved_bytes -= int(r["capacity"])
                if r["users"]:
                    r["state"] = _RETIRED
                    self.retired_bytes += int(r["capacity"])
                else:
                    self._reclaim(slot, r, blocks)
            self.trace_node(
                "physical_free",
                handles=handles,
                refs=refs,
                logical_evicted_pages=len(handles),
                released_blocks=self.free_block_count - before_blocks,
                free_blocks_before=before_blocks,
            )
            return len(handles)

    @contextlib.contextmanager
    def stage(self, read):
        lane_lock = self.read_lock if read else self.write_lock
        name = "read" if read else "write"
        started = time.perf_counter()
        with self.lock:
            if self.quarantined:
                raise BufferDrainError("Host staging is quarantined")
            self.stage_waiters += 1
        lane_lock.acquire()  # Worker only; never under pool/runtime lock.
        with self.lock:
            self.stage_waiters -= 1
            if self.quarantined:
                lane_lock.release()
                raise BufferDrainError("Host staging is quarantined")
            self.stage_users += 1
            self.allocation_stats[name + "_stage_wait_seconds"] += (
                time.perf_counter() - started
            )
        quarantine = False
        try:
            stage = self.read_stage if read else self.write_stage
            stage.zero_()
            yield stage
        except BufferDrainError as exc:
            quarantine = True
            with self.lock:
                self.quarantined.append((name, exc))
            raise
        finally:
            with self.lock:
                if not quarantine:
                    self.stage_users -= 1
            # Quarantined storage remains allocated and new accesses rejected.
            # Unlock so already-waiting workers wake and fail instead of hanging.
            lane_lock.release()

    def has_readers(self):
        with self.lock:
            return bool(
                self.active_readers
                or self.active_writers
                or self.stage_users
                or self.stage_waiters
                or self.quarantined
            )

    def allocator_snapshot(self):
        with self.lock:
            return dict(self.allocation_stats)

    def snapshot(self):
        with self.lock:
            return {
                "budget_bytes": self.budget_bytes,
                "reservation_bytes": self.reservation_bytes,
                "verify": self.verify,
                "active_readers": self.active_readers,
                "active_writers": self.active_writers,
                "metadata_bytes": self.metadata_bytes,
                "staging_bytes": self.staging_bytes,
                "scratch_bytes": self.scratch_bytes,
                "payload_bytes": self.live_payload_bytes,
                "encoded_bytes": self.encoded_bytes,
                "internal_padding_bytes": self.live_payload_bytes - self.encoded_bytes,
                "reserved_bytes": self.reserved_bytes,
                "retired_bytes": self.retired_bytes,
                "arena_bytes": self.arena.numel(),
                "used_pages": self.size - self.free_count,
                "block_count": self.block_count,
                "free_blocks": self.free_block_count,
                "reserved_blocks": self.reserved_bytes // BLOCK_BYTES,
                "live_blocks": self.live_payload_bytes // BLOCK_BYTES,
                "retired_blocks": self.retired_bytes // BLOCK_BYTES,
                "stage_waiters": self.stage_waiters,
                "stage_users": self.stage_users,
                "quarantined_objects": len(self.quarantined),
            }

    def clear(self):
        with self.lock:
            if self.has_readers():
                raise RuntimeError(
                    "Cannot clear L2 with active payload readers or writers"
                )
            self.free(
                [
                    (int(r["generation"]) << 32) | i
                    for i, r in enumerate(self.records)
                    if r["state"] != _FREE
                ]
            )

    def destroy(self):
        self.clear()

    def get_data_page(self, *_args, **_kwargs):
        raise RuntimeError("Compressed L2 cannot use raw HostKVCache/L3 page I/O")

    get_dummy_flat_data_page = set_from_flat_data_page = (
        backup_from_device_all_layer
    ) = load_to_device_per_layer = get_data_page


@contextlib.contextmanager
def materialize_pages(pages):
    """Worker-only. Caller retains object leases and drains DMA inside scope."""
    host = [p for p in pages if isinstance(p, HostEncodedPage)]
    if not host:
        yield pages
        return
    pool = host[0].pool
    if len(pages) > BATCH_PAGES or any(p.pool is not pool for p in host):
        raise ValueError("Invalid host materialization window")
    with pool.stage(read=True) as stage:
        started = time.perf_counter()
        with pool.lock:
            chains = pool._chains([p.handle for p in host])
            for p in host:
                _, r = pool._record(p.handle)
                if (
                    r["state"] not in (_READY, _RETIRED)
                    or not r["users"]
                    or r["length"] != p.nbytes
                ):
                    raise ValueError("Host materialization needs a live read lease")
        # Gather each object's blocks with a batched CPU index_select directly
        # into its contiguous row. Padding is zero, never transmitted as data.
        for i, (p, blocks) in enumerate(zip(host, chains)):
            torch.index_select(
                pool.arena.view(-1, BLOCK_BYTES),
                0,
                torch.from_numpy(blocks.astype(np.int64)),
                out=stage[i, : len(blocks) * BLOCK_BYTES].view(-1, BLOCK_BYTES),
            )
            stage[i, p.nbytes : len(blocks) * BLOCK_BYTES].zero_()
        pool.allocation_stats["gather_seconds"] += time.perf_counter() - started
        result, i = [], 0
        for page in pages:
            if isinstance(page, HostEncodedPage):
                result.append(
                    EncodedPage(
                        stage[i, : page.nbytes],
                        page.encoding,
                        page.raw_bytes,
                        page.raw_sha256,
                    )
                )
                i += 1
            else:
                result.append(page)
        yield result
