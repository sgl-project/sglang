import asyncio
import logging
import threading
import time
from queue import Empty, Queue
from typing import List, Optional

import torch

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_embedding_store import (
    MooncakeEmbeddingStore,
)

logger = logging.getLogger(__name__)


class ContiguousMemoryAllocator:
    """
    A simple allocator to manage variable-sized contiguous blocks
    within a large pre-allocated flat buffer.
    """

    def __init__(self, total_size_bytes: int):
        self.total_size = total_size_bytes
        # List of (offset, size) for free blocks
        self.free_blocks = [(0, total_size_bytes)]
        self.allocated_map = {}  # {handle: (offset, size)}
        self.lock = threading.Lock()

    def allocate(self, size_bytes: int) -> Optional[int]:
        with self.lock:
            # Simple First-Fit allocation
            for i, (offset, block_size) in enumerate(self.free_blocks):
                if block_size >= size_bytes:
                    # Allocate from this block
                    remaining_size = block_size - size_bytes
                    if remaining_size > 0:
                        self.free_blocks[i] = (offset + size_bytes, remaining_size)
                    else:
                        self.free_blocks.pop(i)
                    return offset
            return None

    def free(self, offset: int, size_bytes: int):
        with self.lock:
            # Return block and merge adjacent free blocks
            self.free_blocks.append((offset, size_bytes))
            self.free_blocks.sort()

            merged = []
            if not self.free_blocks:
                return

            curr_offset, curr_size = self.free_blocks[0]
            for next_offset, next_size in self.free_blocks[1:]:
                if curr_offset + curr_size == next_offset:
                    curr_size += next_size
                else:
                    merged.append((curr_offset, curr_size))
                    curr_offset, curr_size = next_offset, next_size
            merged.append((curr_offset, curr_size))
            self.free_blocks = merged


class EmbeddingPrefetchOperation:
    """Groups all missing images of a request for a single batch GET."""

    def __init__(self, req_id: str, keys: List[str], ptrs: List[int], sizes: List[int]):
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

    def __init__(self, keys: List[str], ptrs: List[int], sizes: List[int]):
        self.keys = keys
        self.ptrs = ptrs
        self.sizes = sizes


class EmbeddingCacheController:
    def __init__(
        self,
        tp_rank,
        tp_size,
        max_pool_size_gb=4.0,
        hidden_dim=1024,
        tp_group=None,
        all_rank_get=False,
    ):
        self.tp_world_size = tp_size
        self.tp_group = tp_group
        self.all_rank_get = all_rank_get
        self.hidden_dim = hidden_dim
        self.element_size = torch.float32.itemsize

        # 1. Mooncake Backend & Pinned Buffer
        self.mooncake_store = MooncakeEmbeddingStore()
        self.total_pool_size_bytes = int(max_pool_size_gb * 1024**3)
        self.cpu_pool = torch.empty(
            self.total_pool_size_bytes, dtype=torch.uint8, pin_memory=True
        )
        self.mooncake_store.register_buffer(self.cpu_pool)

        # 2. Variable Size Memory Management
        self.allocator = ContiguousMemoryAllocator(self.total_pool_size_bytes)
        self.hash_to_metadata = {}  # {image_hash: (offset, num_tokens, size_bytes)}

        # 3. Task Tracking
        self.ongoing_prefetch = {}  # {req_id: EmbeddingPrefetchOperation}
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

    def prefetch(
        self, req_id: str, image_hashes: List[str], expected_tokens: List[int]
    ):
        """Issues ONE batch GET for all missing images in the request."""
        keys, ptrs, sizes = [], [], []

        with self.lock:
            for h, num_tokens in zip(image_hashes, expected_tokens):
                if h in self.hash_to_metadata:
                    logger.debug(
                        f"Req {req_id}: Hash  already in local metadata, skipping prefetch."
                    )
                    continue

                size_bytes = num_tokens * self.hidden_dim * self.element_size
                offset = self.allocator.allocate(size_bytes)
                if offset is None:
                    continue

                self.hash_to_metadata[h] = (offset, num_tokens, size_bytes)
                keys.append(h)
                ptrs.append(self.cpu_pool.data_ptr() + offset)
                sizes.append(size_bytes)

            if not keys:
                return

            logger.info(
                f"Req {req_id}: Starting global fetch for {len(keys)} images from Mooncake."
            )

            op = EmbeddingPrefetchOperation(req_id, keys, ptrs, sizes)
            self.ongoing_prefetch[req_id] = op
            self.prefetch_queue.put(op)

    def insert_batch(
        self, image_hashes: List[str], embedding_tensors: List[torch.Tensor]
    ):
        """Issues ONE batch PUT for all embeddings computed by this request."""
        keys, ptrs, sizes = [], [], []

        with self.lock:
            for h, tensor in zip(image_hashes, embedding_tensors):
                if h in self.hash_to_metadata:
                    continue

                num_tokens = tensor.shape[0]
                size_bytes = num_tokens * self.hidden_dim * self.element_size
                offset = self.allocator.allocate(size_bytes)
                if offset is None:
                    continue

                # Copy to pinned pool for RDMA
                self.hash_to_metadata[h] = (offset, num_tokens, size_bytes)
                target_view = (
                    self.cpu_pool[offset : offset + size_bytes]
                    .view(torch.float32)
                    .view(num_tokens, self.hidden_dim)
                )
                target_view.copy_(tensor.cpu())

                keys.append(h)
                ptrs.append(self.cpu_pool.data_ptr() + offset)
                sizes.append(size_bytes)

            if keys:
                logger.info(
                    f"Global Cache: Inserting {len(keys)} new embeddings into Mooncake cluster."
                )
                self.insert_queue.put(EmbeddingInsertOperation(keys, ptrs, sizes))

    def _io_loop(self):
        """Asynchronous worker handling both Batch GET and Batch PUT."""
        while not self.stop_event.is_set():
            processed_any = False

            try:
                op = self.prefetch_queue.get_nowait()
                results = self.mooncake_store.batch_get(op.keys, op.ptrs, op.sizes)
                success_count = sum(results)
                logger.info(
                    f"Mooncake GET Finished: Req {op.req_id}, Successfully fetched {success_count}/{len(op.keys)} images."
                )
                op.mark_done(all(results))
                self.prefetch_queue.task_done()
                processed_any = True
            except Empty:
                pass

            try:
                op = self.insert_queue.get_nowait()
                self.mooncake_store.batch_put(op.keys, op.ptrs, op.sizes)
                logger.info(
                    f"Mooncake PUT Finished: Successfully stored {len(op.keys)} keys in cluster."
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
                    local_ready = op.success

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

    def get_embeddings(self, image_hashes: List[str]) -> List[torch.Tensor]:
        """Final reconstruction for model input."""
        with self.lock:
            tensors = []
            for h in image_hashes:
                offset, num_tokens, size_bytes = self.hash_to_metadata[h]
                tensors.append(
                    self.cpu_pool[offset : offset + size_bytes]
                    .view(torch.float32)
                    .view(num_tokens, self.hidden_dim)
                )
            return tensors

    async def batch_is_exist(self, image_hashes: List[str]) -> List[bool]:
        with self.lock:
            local_results = [h in self.hash_to_metadata for h in image_hashes]
        local_hit_count = sum(local_results)

        global_hit_count = 0
        if not all(local_results):
            missing_indices = [i for i, res in enumerate(local_results) if not res]
            missing_hashes = [image_hashes[i] for i in missing_indices]

            global_exists = await asyncio.to_thread(
                self.mooncake_store.batch_is_exist, missing_hashes
            )
            global_hit_count = sum(global_exists)

            for i, exists in zip(missing_indices, global_exists):
                local_results[i] = exists

        total = len(image_hashes)
        miss_count = total - local_hit_count - global_hit_count
        logger.info(
            f"=== Multi-Level Cache Check === "
            f"Total: {total} | "
            f"Local Hits: {local_hit_count} | "
            f"Global Hits: {global_hit_count} | "
            f"Misses (GPU Work): {miss_count}"
        )
        return local_results
