import asyncio
import logging
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, auto
from queue import Empty, Queue
from typing import List, Optional, Tuple

import torch

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.mem_cache.embedding_store import EmbeddingStore

logger = logging.getLogger(__name__)

TARGET_PAGE_BYTES = 256 * 1024
VISION_POOL_RATIO = 0.8


def _dtype_element_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def compute_page_size(dim: int, element_size: int = 4) -> int:
    return max(TARGET_PAGE_BYTES // (dim * element_size), 1)


class EntryState(Enum):
    FILLING = auto()
    READY = auto()


@dataclass(frozen=True)
class PageRun:
    start: int
    length: int

    @property
    def end(self) -> int:
        return self.start + self.length

    def page_ids(self) -> List[int]:
        return list(range(self.start, self.end))


class RangePageAllocator:
    """Range-aware page allocator that prefers contiguous physical page runs."""

    def __init__(self, num_pages: int):
        self.free_ranges: List[Tuple[int, int]] = (
            [(0, num_pages)] if num_pages > 0 else []
        )

    def allocate(self, num_tokens: int, page_size: int) -> Optional[List[PageRun]]:
        required_pages = math.ceil(num_tokens / page_size)
        if required_pages <= 0:
            return []
        if self.free_pages < required_pages:
            return None

        for idx, (start, length) in enumerate(self.free_ranges):
            if length >= required_pages:
                run = PageRun(start, required_pages)
                if length == required_pages:
                    self.free_ranges.pop(idx)
                else:
                    self.free_ranges[idx] = (
                        start + required_pages,
                        length - required_pages,
                    )
                return [run]

        runs: List[PageRun] = []
        remaining = required_pages
        while remaining > 0 and self.free_ranges:
            start, length = self.free_ranges.pop(0)
            take = min(length, remaining)
            runs.append(PageRun(start, take))
            remaining -= take
            if take < length:
                self.free_ranges.insert(0, (start + take, length - take))

        if remaining != 0:
            self.free(runs)
            return None
        return runs

    def free(self, runs: List[PageRun]):
        if not runs:
            return
        for run in runs:
            if run.length > 0:
                self.free_ranges.append((run.start, run.length))
        self.free_ranges.sort()

        merged: List[Tuple[int, int]] = []
        for start, length in self.free_ranges:
            if not merged:
                merged.append((start, length))
                continue
            prev_start, prev_length = merged[-1]
            prev_end = prev_start + prev_length
            if prev_end >= start:
                merged[-1] = (prev_start, max(prev_end, start + length) - prev_start)
            else:
                merged.append((start, length))
        self.free_ranges = merged

    @property
    def free_pages(self) -> int:
        return sum(length for _, length in self.free_ranges)


class EvictableLRU:
    """Per-pool eviction candidate queue.

    Only entries that can be freed immediately (READY, zero read pins)
    belong in this queue.  The controller is responsible for
    adding and removing entries at the right state transitions; this
    class does not inspect entry state.
    """

    def __init__(self):
        self._lru: OrderedDict[str, float] = OrderedDict()

    def touch(self, mm_hash: str):
        """Mark as recently used (move to tail of the queue)."""
        self._lru.pop(mm_hash, None)
        self._lru[mm_hash] = time.time()

    def remove(self, mm_hash: str):
        """Remove from candidates."""
        self._lru.pop(mm_hash, None)

    def pop_oldest(self) -> Optional[str]:
        """Pop the least-recently-used candidate. Returns None if empty."""
        if not self._lru:
            return None
        mm_hash, _ = self._lru.popitem(last=False)
        return mm_hash

    def __len__(self) -> int:
        return len(self._lru)

    def __contains__(self, mm_hash: str) -> bool:
        return mm_hash in self._lru

    def keys(self):
        return self._lru.keys()


@dataclass
class EmbeddingPool:
    modality: str
    dim: int
    dtype: torch.dtype
    page_size: int
    tensor: torch.Tensor
    num_pages: int
    allocator: RangePageAllocator
    page_bytes: int
    pool_size_bytes: int
    pin_memory: bool = True

    @classmethod
    def create(
        cls,
        modality: str,
        dim: int,
        pool_size_bytes: int,
        dtype: torch.dtype = torch.float32,
        pin_memory: bool = True,
    ) -> "EmbeddingPool":
        element_size = _dtype_element_size(dtype)
        page_size = compute_page_size(dim, element_size)
        capacity_tokens = pool_size_bytes // (dim * element_size)
        num_pages = capacity_tokens // page_size
        total_tokens = num_pages * page_size
        tensor = torch.empty(
            (total_tokens, dim),
            dtype=dtype,
            pin_memory=pin_memory,
        )
        page_bytes = page_size * dim * element_size
        return cls(
            modality=modality,
            dim=dim,
            dtype=dtype,
            page_size=page_size,
            tensor=tensor,
            num_pages=num_pages,
            allocator=RangePageAllocator(num_pages),
            page_bytes=page_bytes,
            pool_size_bytes=pool_size_bytes,
            pin_memory=pin_memory,
        )


@dataclass
class EmbeddingCacheEntry:
    hash: str
    modality: object
    num_tokens: int
    dim: int
    page_runs: List[PageRun]
    state: EntryState
    ref_count: int = 0

    @property
    def page_ids(self) -> List[int]:
        return [page_id for run in self.page_runs for page_id in run.page_ids()]

    def pin(self):
        self.ref_count += 1

    def unpin(self):
        if self.ref_count <= 0:
            logger.warning("unpin called with ref_count=0 for %s", self.hash)
            return
        self.ref_count -= 1

    def is_evictable(self) -> bool:
        return self.state == EntryState.READY and self.ref_count == 0


def build_transfer_buffers(
    entry: EmbeddingCacheEntry, pool: EmbeddingPool
) -> Tuple[List[int], List[int]]:
    """Build one pointer/size pair per physical page run."""
    if not entry.page_runs:
        return [], []

    ptrs: List[int] = []
    sizes: List[int] = []
    remaining_tokens = entry.num_tokens
    element_size = _dtype_element_size(pool.dtype)

    for run in entry.page_runs:
        if remaining_tokens <= 0:
            break
        valid_tokens = min(pool.page_size * run.length, remaining_tokens)
        ptr = pool.tensor[run.start * pool.page_size].data_ptr()
        size_bytes = valid_tokens * entry.dim * element_size
        ptrs.append(ptr)
        sizes.append(size_bytes)
        remaining_tokens -= valid_tokens
    return ptrs, sizes


@dataclass
class AsyncCopyHandle:
    event: object
    entry_hash: str
    device: Optional[torch.device] = None
    _src_ref: object = None

    def is_complete(self) -> bool:
        if self.event is None:
            return True
        return bool(self.event.query())

    def wait(self):
        if self.event is not None:
            self.event.synchronize()
        self._src_ref = None


class EmbeddingPrefetchOperation:
    """Groups all missing images of a request for a single batch GET."""

    def __init__(
        self,
        req_id: str,
        keys: List[str],
        ptrs: List[List[int]],
        sizes: List[List[int]],
    ):
        self.req_id = req_id
        self.keys = keys
        self.ptrs = ptrs
        self.sizes = sizes
        self.is_finished = False
        self.success = False
        self._lock = threading.Lock()

    def mark_done(self, success: bool):
        with self._lock:
            self.success = success
            self.is_finished = True


class EmbeddingInsertOperation:
    """Groups all newly computed images of a request for a single batch PUT."""

    def __init__(self, keys: List[str], ptrs: List[List[int]], sizes: List[List[int]]):
        self.keys = keys
        self.ptrs = ptrs
        self.sizes = sizes


class EmbeddingCacheController:
    def __init__(
        self,
        tp_rank,
        tp_size,
        embedding_store: EmbeddingStore,
        max_pool_size_gb=4.0,
        hidden_dims: dict = None,
        tp_group=None,
        all_rank_get=False,
        enable_eviction: bool = True,
        max_eviction_batch: int = 100,
        dtype: torch.dtype = torch.float32,
    ):
        self.tp_world_size = tp_size
        self.tp_group = tp_group
        self.tp_rank = tp_rank
        self.all_rank_get = all_rank_get
        self.hidden_dims = hidden_dims or {}
        # Pool dtype must match the model's embedding dtype so that pool views,
        # ViT output, and the final send buffer share one dtype — assembly then
        # copies without any cast. Defaults to float32 for backward compat.
        self.dtype = dtype
        self.element_size = _dtype_element_size(self.dtype)
        self.enable_eviction = enable_eviction
        self.max_eviction_batch = max_eviction_batch

        self.embedding_store = embedding_store
        self.total_pool_size_bytes = int(max_pool_size_gb * 1024**3)
        self.vision_pool, self.audio_pool = self._create_pools(pin_memory=True)
        self.pools = {
            "vision": self.vision_pool,
            "audio": self.audio_pool,
        }
        self._register_pool_buffer(self.vision_pool)
        self._register_pool_buffer(self.audio_pool)

        self.entries = {}
        # self.lock protects entries, pool allocators, entry state/pins,
        # and per-pool evictable LRUs. Do not mutate without holding self.lock.
        self.vision_pool.evictable = EvictableLRU()
        self.audio_pool.evictable = EvictableLRU()

        self.stats = {
            "total_allocated": 0,
            "total_evicted": 0,
            "eviction_count": 0,
            "allocation_failures": 0,
        }

        self._copy_streams = {}

        self.ongoing_prefetch = {}
        self.prefetch_queue = Queue()
        self.insert_queue = Queue()

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.io_thread = threading.Thread(target=self._io_loop, daemon=True)
        self.io_thread.start()

        if self.tp_world_size > 1:
            if self.tp_group is None:
                raise ValueError("tp_group must be provided when tp_size > 1")
            from sglang.srt.distributed.parallel_state import (
                create_custom_parallel_group,
            )

            group_ranks = torch.distributed.get_process_group_ranks(self.tp_group)
            self.prefetch_tp_group = create_custom_parallel_group(
                group_ranks=group_ranks, backend="gloo"
            )
        else:
            self.prefetch_tp_group = None

    def _create_pools(self, pin_memory: bool) -> Tuple[EmbeddingPool, EmbeddingPool]:
        # vision pool uses IMAGE dim (IMAGE == VIDEO dim in all supported models)
        vision_dim = self.hidden_dims.get(Modality.IMAGE) or self.hidden_dims.get(
            Modality.VIDEO
        )
        audio_dim = self.hidden_dims.get(Modality.AUDIO) or vision_dim
        vision_bytes = int(self.total_pool_size_bytes * VISION_POOL_RATIO)
        audio_bytes = self.total_pool_size_bytes - vision_bytes
        return (
            EmbeddingPool.create(
                "vision", vision_dim, vision_bytes, self.dtype, pin_memory
            ),
            EmbeddingPool.create(
                "audio", audio_dim, audio_bytes, self.dtype, pin_memory
            ),
        )

    def _register_pool_buffer(self, pool: EmbeddingPool):
        if pool.tensor.numel() == 0:
            logger.warning(
                f"[Rank {self.tp_rank}] {pool.modality} embedding pool has zero pages; "
                f"dim={pool.dim}, budget={pool.pool_size_bytes} bytes"
            )
            return
        self.embedding_store.register_buffer(pool.tensor)
        logger.info(
            f"[Rank {self.tp_rank}] Registered {pool.modality} embedding pool: "
            f"dim={pool.dim}, pages={pool.num_pages}, "
            f"page_tokens={pool.page_size}, "
            f"capacity={pool.num_pages * pool.page_bytes / 1024**2:.2f} MB"
        )

    def _get_pool(self, modality: Modality) -> Optional[EmbeddingPool]:
        if modality == Modality.AUDIO:
            return self.audio_pool
        if modality in (Modality.IMAGE, Modality.VIDEO):
            return self.vision_pool
        return None

    # --- LRU and state helpers (caller must hold self.lock) ---

    def _lru_touch(self, mm_hash: str):
        """Mark an evictable entry as recently used in its pool's LRU."""
        entry = self.entries.get(mm_hash)
        if entry is None or not entry.is_evictable():
            return
        pool = self._get_pool(entry.modality)
        if pool is not None:
            pool.evictable.touch(mm_hash)

    def _mark_ready(self, entry: EmbeddingCacheEntry):
        """Transition a FILLING entry to READY."""
        entry.state = EntryState.READY
        pool = self._get_pool(entry.modality)
        if pool is not None and entry.is_evictable():
            pool.evictable.touch(entry.hash)

    def _pin_read(self, entry: EmbeddingCacheEntry):
        """Pin a READY entry for a read transfer."""
        if entry.ref_count == 0:
            pool = self._get_pool(entry.modality)
            if pool is not None:
                pool.evictable.remove(entry.hash)
        entry.pin()

    def _unpin_read(self, entry: EmbeddingCacheEntry):
        """Release a read transfer pin."""
        entry.unpin()
        if entry.is_evictable():
            pool = self._get_pool(entry.modality)
            if pool is not None:
                pool.evictable.touch(entry.hash)

    # --- Eviction ---

    def _evict_entry(self, mm_hash: str, remove_lru: bool = True):
        """Free one entry and remove metadata.

        Caller must hold self.lock.
        """
        entry = self.entries.get(mm_hash)
        if entry is None:
            return
        pool = self._get_pool(entry.modality)
        if remove_lru and pool is not None:
            pool.evictable.remove(mm_hash)
        pool.allocator.free(entry.page_runs)
        self.stats["total_evicted"] += entry.num_tokens * entry.dim * self.element_size
        del self.entries[mm_hash]

    def _evict_for_pool(self, pool: EmbeddingPool, required_pages: int):
        """Evict oldest entries from this pool until enough pages are free.

        Caller must hold self.lock.
        """
        if pool.allocator.free_pages >= required_pages:
            return

        evicted = 0
        while pool.allocator.free_pages < required_pages:
            if evicted >= self.max_eviction_batch:
                break
            mm_hash = pool.evictable.pop_oldest()
            if mm_hash is None:
                break
            entry = self.entries.get(mm_hash)
            if entry is None:
                continue
            self._evict_entry(mm_hash, remove_lru=False)
            evicted += 1

        if evicted > 0:
            self.stats["eviction_count"] += 1
            logger.info(
                f"[Rank {self.tp_rank}] Evicted {evicted} embeddings from "
                f"{pool.modality} pool"
            )

    def _allocate_with_eviction(
        self, pool: EmbeddingPool, num_tokens: int
    ) -> Optional[List[PageRun]]:
        """Allocate pages, evicting LRU entries from the same pool if needed.

        Caller must hold self.lock.
        """
        required_pages = math.ceil(num_tokens / pool.page_size)
        if required_pages > pool.num_pages:
            self.stats["allocation_failures"] += 1
            return None

        if self.enable_eviction:
            self._evict_for_pool(pool, required_pages)

        page_runs = pool.allocator.allocate(num_tokens, pool.page_size)
        if page_runs is not None:
            self.stats["total_allocated"] += num_tokens * pool.dim * self.element_size
        else:
            self.stats["allocation_failures"] += 1
            logger.warning(
                f"[Rank {self.tp_rank}] Cannot allocate {required_pages} pages "
                f"in {pool.modality} pool: free={pool.allocator.free_pages}"
            )
        return page_runs

    def prefetch(
        self,
        req_id: str,
        mm_hashes: List[str],
        expected_tokens: List[int],
        modality=None,
    ):
        """Issues ONE batch GET for cache-hit embeddings that are not local yet."""
        pool = self._get_pool(modality)
        if pool is None:
            logger.warning(f"prefetch: unknown modality {modality}; skipping.")
            return

        keys, all_ptrs, all_sizes = [], [], []

        with self.lock:
            for mm_hash, num_tokens in zip(mm_hashes, expected_tokens):
                entry = self.entries.get(mm_hash)
                if entry is not None:
                    if entry.state == EntryState.READY:
                        self._lru_touch(mm_hash)
                    else:
                        logger.debug(
                            f"Req {req_id}: {mm_hash} is FILLING; " f"treating as miss."
                        )
                    continue

                page_runs = self._allocate_with_eviction(pool, int(num_tokens))
                if page_runs is None:
                    logger.warning(
                        f"Req {req_id}: Failed to allocate {num_tokens} tokens "
                        f"in {pool.modality} pool; falling back to encoder."
                    )
                    continue

                entry = EmbeddingCacheEntry(
                    hash=mm_hash,
                    modality=modality,
                    num_tokens=int(num_tokens),
                    dim=pool.dim,
                    page_runs=page_runs,
                    state=EntryState.FILLING,
                )
                self.entries[mm_hash] = entry
                keys.append(mm_hash)
                entry_ptrs, entry_sizes = build_transfer_buffers(entry, pool)
                all_ptrs.append(entry_ptrs)
                all_sizes.append(entry_sizes)

            if not keys:
                return

            logger.info(
                f"Req {req_id}: Starting global fetch for {len(keys)} "
                f"embeddings from Mooncake."
            )

            op = EmbeddingPrefetchOperation(req_id, keys, all_ptrs, all_sizes)
            self.ongoing_prefetch[req_id] = op
            self.prefetch_queue.put(op)

    def insert_batch(
        self,
        mm_hashes: List[str],
        modality: Modality = None,
    ):
        """Issues ONE batch PUT for embeddings already in the host pool.

        Only READY entries are pushed to Mooncake for multi-node sharing.
        If an entry was never stored (e.g. store_to_pool_async allocation failed),
        it is silently skipped.
        """
        pool = self._get_pool(modality)
        if pool is None:
            logger.warning(f"insert_batch: unknown modality {modality}; skipping.")
            return

        keys, all_ptrs, all_sizes = [], [], []
        skipped_count = 0

        with self.lock:
            for mm_hash in mm_hashes:
                entry = self.entries.get(mm_hash)
                if entry is None or entry.state != EntryState.READY:
                    skipped_count += 1
                    continue

                self._pin_read(entry)
                keys.append(mm_hash)
                entry_ptrs, entry_sizes = build_transfer_buffers(entry, pool)
                all_ptrs.append(entry_ptrs)
                all_sizes.append(entry_sizes)

            if keys:
                logger.info(
                    f"Global Cache: Inserting {len(keys)} embeddings into "
                    f"Mooncake cluster ({skipped_count} skipped)"
                )
                self.insert_queue.put(
                    EmbeddingInsertOperation(keys, all_ptrs, all_sizes)
                )

    def _finish_get(self, op: EmbeddingPrefetchOperation, results: List[bool]):
        with self.lock:
            for mm_hash, success in zip(op.keys, results):
                entry = self.entries.get(mm_hash)
                if entry is None:
                    continue
                if success:
                    if entry.state == EntryState.FILLING:
                        self._mark_ready(entry)
                else:
                    pool = self._get_pool(entry.modality)
                    pool.evictable.remove(mm_hash)
                    pool.allocator.free(entry.page_runs)
                    del self.entries[mm_hash]
        op.mark_done(all(results))

    def _finish_put(self, op: EmbeddingInsertOperation, results: List[bool]):
        with self.lock:
            for mm_hash, success in zip(op.keys, results):
                entry = self.entries.get(mm_hash)
                if entry is None:
                    continue
                if not success:
                    logger.warning(
                        f"[Rank {self.tp_rank}] Mooncake PUT failed for "
                        f"{mm_hash}; keeping local cache entry."
                    )
                self._unpin_read(entry)

    def _io_loop(self):
        """Asynchronous worker handling both Batch GET and Batch PUT."""
        while not self.stop_event.is_set():
            processed_any = False

            try:
                op = self.prefetch_queue.get_nowait()
                try:
                    results = self.embedding_store.batch_get_into_multi_buffers(
                        op.keys, op.ptrs, op.sizes
                    )
                except Exception:
                    logger.exception("Mooncake multi-buffer GET failed")
                    results = [False] * len(op.keys)
                success_count = sum(results)
                logger.info(
                    f"Mooncake GET Finished: Req {op.req_id}, "
                    f"Successfully fetched {success_count}/{len(op.keys)} embeddings."
                )
                self._finish_get(op, results)
                self.prefetch_queue.task_done()
                processed_any = True
            except Empty:
                pass

            try:
                op = self.insert_queue.get_nowait()
                try:
                    results = self.embedding_store.batch_put_from_multi_buffers(
                        op.keys, op.ptrs, op.sizes
                    )
                except Exception:
                    logger.exception("Mooncake multi-buffer PUT failed")
                    results = [False] * len(op.keys)
                self._finish_put(op, results)
                logger.info(
                    f"Mooncake PUT Finished: Stored {sum(results)}/{len(op.keys)} "
                    f"embeddings in cluster."
                )
                self.insert_queue.task_done()
                processed_any = True
            except Empty:
                pass

            if not processed_any:
                time.sleep(0.001)

    def check_prefetch_progress(self, req_id: str) -> bool:
        """TP-Group barrier: ensures all cards have the request batch ready."""
        local_ready = False
        with self.lock:
            if req_id not in self.ongoing_prefetch:
                local_ready = True
            else:
                op = self.ongoing_prefetch[req_id]
                if op.is_finished:
                    local_ready = True

        if self.all_rank_get and self.tp_world_size > 1:
            ready_tensor = torch.tensor(
                [1 if local_ready else 0], dtype=torch.int, device="cpu"
            )
            torch.distributed.all_reduce(
                ready_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.prefetch_tp_group,
            )
            local_ready = ready_tensor.item() == 1

        if local_ready:
            with self.lock:
                self.ongoing_prefetch.pop(req_id, None)
            return True
        return False

    def load_to_device_async(
        self, mm_hash: str, dst_tensor: torch.Tensor, dst_token_offset: int
    ) -> Optional[AsyncCopyHandle]:
        """Async host-pool → device copy for a single READY entry.

        Returns an AsyncCopyHandle on success, or None if the entry is
        missing/not ready.  Pass returned handles to wait_load_to_device().
        """
        with self.lock:
            entry = self.entries.get(mm_hash)
            if entry is None:
                logger.warning(f"Hash {mm_hash} not found in local cache")
                return None
            if entry.state != EntryState.READY:
                logger.warning(f"Hash {mm_hash} is not ready; state={entry.state.name}")
                return None
            self._pin_read(entry)

        pool = self._get_pool(entry.modality)
        try:
            device = dst_tensor.device
            copy_stream = self._get_copy_stream(device)
            event = torch.cuda.Event()
            copied = 0
            with torch.cuda.stream(copy_stream):
                for run in entry.page_runs:
                    valid_tokens = min(
                        pool.page_size * run.length, entry.num_tokens - copied
                    )
                    if valid_tokens <= 0:
                        break
                    src_start = run.start * pool.page_size
                    dst_start = dst_token_offset + copied
                    dst_tensor[dst_start : dst_start + valid_tokens].copy_(
                        pool.tensor[src_start : src_start + valid_tokens],
                        non_blocking=True,
                    )
                    copied += valid_tokens
                event.record(copy_stream)
            return AsyncCopyHandle(event, mm_hash, device=torch.device(device))
        except Exception:
            with self.lock:
                self._unpin_read(entry)
            raise

    def wait_load_to_device(self, handles: List[AsyncCopyHandle]):
        """Wait for async GPU copies and release pool pins."""
        for handle in handles:
            handle.wait()
        with self.lock:
            for handle in handles:
                entry = self.entries.get(handle.entry_hash)
                if entry is not None:
                    self._unpin_read(entry)

    def get_pool_views(
        self, mm_hashes: List[str]
    ) -> List[Optional[List[torch.Tensor]]]:
        """Return zero-copy slice lists into the host pool for READY entries.

        Each element is a list of tensor views (one per page run) or None
        if the entry is missing/not ready. The caller should flatten these
        into a single list and do one torch.cat at the end.
        Call release_pool_views() when done.
        """
        results: List[Optional[List[torch.Tensor]]] = []
        with self.lock:
            for mm_hash in mm_hashes:
                entry = self.entries.get(mm_hash)
                if entry is None or entry.state != EntryState.READY:
                    results.append(None)
                    continue
                self._pin_read(entry)
                pool = self._get_pool(entry.modality)
                slices = []
                copied = 0
                for run in entry.page_runs:
                    valid = min(pool.page_size * run.length, entry.num_tokens - copied)
                    if valid <= 0:
                        break
                    start = run.start * pool.page_size
                    slices.append(pool.tensor[start : start + valid])
                    copied += valid
                results.append(slices)
        return results

    def release_pool_views(self, mm_hashes: List[str]):
        """Release pins acquired by get_pool_views."""
        with self.lock:
            for mm_hash in mm_hashes:
                entry = self.entries.get(mm_hash)
                if entry is not None:
                    self._unpin_read(entry)

    def _get_copy_stream(self, device: torch.device) -> "torch.cuda.Stream":
        key = str(device)
        stream = self._copy_streams.get(key)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._copy_streams[key] = stream
        return stream

    def has_local_embedding(self, mm_hash: str) -> bool:
        with self.lock:
            entry = self.entries.get(mm_hash)
            return entry is not None and entry.state == EntryState.READY

    def store_to_pool_async(
        self,
        mm_hashes: List[str],
        tensors: List[torch.Tensor],
        modality=None,
    ) -> List[Tuple["EmbeddingCacheEntry", "AsyncCopyHandle"]]:
        """Launch async D2H copies into host paged pool.

        Allocates pages and launches D2H copies on a side stream but does
        NOT wait for completion. Entries remain in FILLING state.

        Items that cannot be stored (FILLING conflict, allocation failure,
        dim mismatch) are silently skipped — the caller's assembly step
        falls back to the GPU tensor for those items.

        Returns: pending D2H handles; pass to wait_store_to_pool().
        """
        for tensor in tensors:
            if tensor.device.type != "cuda":
                raise ValueError(
                    f"store_to_pool_async expects CUDA tensors, "
                    f"got device={tensor.device}"
                )

        pool = self._get_pool(modality)
        if pool is None:
            return []

        pending: List[Tuple[torch.Tensor, EmbeddingCacheEntry]] = []
        with self.lock:
            for mm_hash, tensor in zip(mm_hashes, tensors):
                if tensor.ndim != 2:
                    tensor = tensor.reshape(-1, tensor.shape[-1])
                num_tokens, actual_dim = int(tensor.shape[0]), int(tensor.shape[1])

                if actual_dim != pool.dim:
                    logger.warning(
                        f"[Rank {self.tp_rank}] Embedding dim mismatch for "
                        f"{mm_hash}: pool.dim={pool.dim}, tensor.dim={actual_dim}; "
                        f"skipping pool store"
                    )
                    continue

                entry = self.entries.get(mm_hash)
                if entry is not None:
                    if entry.state == EntryState.READY:
                        self._lru_touch(mm_hash)
                        continue
                    if entry.state == EntryState.FILLING:
                        continue
                    self._evict_entry(mm_hash)

                page_runs = self._allocate_with_eviction(pool, num_tokens)
                if page_runs is None:
                    continue

                entry = EmbeddingCacheEntry(
                    hash=mm_hash,
                    modality=modality,
                    num_tokens=num_tokens,
                    dim=actual_dim,
                    page_runs=page_runs,
                    state=EntryState.FILLING,
                )
                self.entries[mm_hash] = entry
                pending.append((tensor, entry))

        handles: List[Tuple[EmbeddingCacheEntry, AsyncCopyHandle]] = []
        for tensor, entry in pending:
            handle = self._copy_tensor_to_pool(tensor, entry, pool)
            handles.append((entry, handle))

        return handles

    def wait_store_to_pool(
        self,
        handles: List[Tuple["EmbeddingCacheEntry", "AsyncCopyHandle"]],
    ):
        """Wait for async D2H copies and mark entries READY."""
        for entry, handle in handles:
            handle.wait()
        with self.lock:
            for entry, handle in handles:
                current = self.entries.get(entry.hash)
                if current is entry and current.state == EntryState.FILLING:
                    self._mark_ready(current)

    def _copy_tensor_to_pool(
        self, tensor: torch.Tensor, entry: EmbeddingCacheEntry, pool: EmbeddingPool
    ) -> AsyncCopyHandle:
        """Async D2H copy of a CUDA tensor into pool pages."""
        src = tensor.detach()
        if src.ndim != 2:
            src = src.reshape(-1, src.shape[-1])
        if not src.is_contiguous():
            src = src.contiguous()

        device = src.device
        producer_stream = torch.cuda.current_stream(device)
        copy_stream = self._get_copy_stream(device)
        copy_stream.wait_stream(producer_stream)
        src.record_stream(copy_stream)
        event = torch.cuda.Event()
        copied = 0
        with torch.cuda.stream(copy_stream):
            for run in entry.page_runs:
                valid_tokens = min(
                    pool.page_size * run.length, entry.num_tokens - copied
                )
                if valid_tokens <= 0:
                    break
                start = run.start * pool.page_size
                pool.tensor[start : start + valid_tokens].copy_(
                    src[copied : copied + valid_tokens],
                    non_blocking=True,
                )
                copied += valid_tokens
            event.record(copy_stream)
        return AsyncCopyHandle(
            event=event,
            entry_hash=entry.hash,
            device=torch.device(device),
            _src_ref=src,
        )

    def get_embedding_dim(self, modality=None) -> int:
        return self._get_pool(modality).dim

    def get_stats(self) -> dict:
        """Return cache statistics."""
        with self.lock:
            allocated_bytes = sum(
                sum(run.length for run in entry.page_runs)
                * self._get_pool(entry.modality).page_bytes
                for entry in self.entries.values()
            )
            free_bytes = sum(
                pool.allocator.free_pages * pool.page_bytes
                for pool in self.pools.values()
            )
            return {
                **self.stats,
                "num_cached": len(self.entries),
                "num_pinned": sum(
                    1 for entry in self.entries.values() if entry.ref_count > 0
                ),
                "allocated_mb": allocated_bytes / 1024**2,
                "free_mb": free_bytes / 1024**2,
                "total_mb": sum(
                    pool.num_pages * pool.page_bytes for pool in self.pools.values()
                )
                / 1024**2,
                "vision_free_pages": self.vision_pool.allocator.free_pages,
                "audio_free_pages": self.audio_pool.allocator.free_pages,
            }

    async def batch_is_exist(self, mm_hashes: List[str]) -> List[bool]:
        with self.lock:
            local_results = []
            for h in mm_hashes:
                entry = self.entries.get(h)
                if entry is not None and entry.state == EntryState.READY:
                    self._lru_touch(h)
                    local_results.append(True)
                else:
                    local_results.append(False)
        local_hit_count = sum(local_results)

        global_hit_count = 0
        if not all(local_results):
            missing_indices = [i for i, res in enumerate(local_results) if not res]
            missing_hashes = [mm_hashes[i] for i in missing_indices]

            global_exists = await asyncio.to_thread(
                self.embedding_store.batch_is_exist, missing_hashes
            )
            global_hit_count = sum(global_exists)

            for i, exists in zip(missing_indices, global_exists):
                local_results[i] = exists

        total = len(mm_hashes)
        miss_count = total - local_hit_count - global_hit_count
        logger.info(
            f"=== Multi-Level Cache Check === "
            f"Total: {total} | "
            f"Local Hits: {local_hit_count} | "
            f"Global Hits: {global_hit_count} | "
            f"Misses (GPU Work): {miss_count}"
        )
        return local_results
