"""Owner-preserving CPU L2 for page-sharded DSA KV and indexer pools.

The radix tree and controller queues retain mirrored logical indices. Only
the resolved L2 transfers use this rank's physical rows. The ordinary DSA
host declarations pack target/draft storage and the L2 engine publishes a
layer-ready event after all KV and indexer transfers for that layer.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.mem_cache.allocator.page_interleave import PageInterleavePoolAllocator
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    CacheOperation,
    HybridCacheController,
)
from sglang.srt.mem_cache.pool_host import HostPoolGroup

logger = logging.getLogger(__name__)


def local_transfer_indices(host_indices, device_indices, *, page_size, rank, size):
    """Filter paired logical pages before collapsing their owner dimension."""
    host = host_indices.to(device="cpu", dtype=torch.int64)
    device = device_indices.to(device="cpu", dtype=torch.int64)
    if host.ndim != 1 or host.shape != device.shape:
        raise ValueError("Sharded HiCache source/destination lengths differ")
    if host.numel() % page_size:
        raise ValueError("Sharded HiCache transfers require whole pages")
    for indices in (host, device):
        blocks = indices.reshape(-1, page_size)
        if not torch.equal(blocks, blocks[:, :1] + torch.arange(page_size)) or bool(
            (blocks[:, 0] % page_size != 0).any()
            or (blocks[:, 0] < size * page_size).any()
        ):
            raise ValueError("Sharded HiCache requires aligned, nonreserved pages")
    owners = host // page_size % size
    if not torch.equal(owners, device // page_size % size):
        raise ValueError("Sharded HiCache source/destination owners differ")
    mask = owners == rank

    def physical(indices):
        indices = indices[mask]
        return indices // (size * page_size) * page_size + indices % page_size

    return physical(host), physical(device)


class ShardedDSAHostPoolGroup(HostPoolGroup):
    """Logical allocation facade over native, owner-local packed host pools."""

    def __init__(self, entries, device_pool):
        super().__init__(entries)
        self.shard_rank = device_pool.shard_rank
        self.shard_size = device_pool.shard_size
        self.physical_size = self.size
        if self.layout not in ("layer_first", "page_first"):
            raise ValueError("Sharded DSA HiCache supports layer_first or page_first")
        for entry in entries:
            if entry.name not in (PoolName.KV, PoolName.INDEXER):
                raise ValueError(
                    "Sharded DSA HiCache requires KV-derived indexer pools"
                )
            if entry.host_pool.size != self.physical_size:
                raise ValueError(
                    "Sharded HiCache host pools must have identical capacity"
                )
            for pool in (entry.device_pool, *entry.packed_draft_device_pools):
                if (pool.shard_rank, pool.shard_size, pool.page_size) != (
                    self.shard_rank,
                    self.shard_size,
                    self.page_size,
                ):
                    raise ValueError("Target and draft must share CP page ownership")
        if self.physical_size < self.page_size:
            raise ValueError("Sharded HiCache needs at least its reserved host page")
        # Physical page zero is padding, just as in the device allocator.
        self.logical_allocator = PageInterleavePoolAllocator(
            size=self.physical_size - self.page_size,
            physical_page_size=self.page_size,
            shard_size=self.shard_size,
            dtype=self.anchor_entry.host_pool.dtype,
            device="cpu",
            kvcache=None,
            need_sort=True,
        )
        self.size = self.logical_size = self.logical_allocator.size
        logger.info(
            "CP-sharded HiCache: rank=%d/%d physical_host_tokens=%d "
            "logical_allocatable_tokens=%d",
            self.shard_rank,
            self.shard_size,
            self.physical_size,
            self.logical_size,
        )

    def clear(self):
        super().clear()
        self.logical_allocator.clear()

    def alloc(self, need_size, *, pool=None, reclaim=None):
        raise RuntimeError("Sharded HiCache allocation requires page ownership")

    def alloc_matching(self, device_indices, *, reclaim=None):
        indices = self.logical_allocator.alloc_matching(device_indices)
        if indices is None and reclaim is not None:
            reclaim(len(device_indices))
            indices = self.logical_allocator.alloc_matching(device_indices)
        return indices

    def free(self, indices, *, pool=None):
        if pool not in (None, PoolName.KV):
            raise ValueError("Sharded indexer pages follow the KV allocation")
        self.logical_allocator.free(indices)
        return len(indices)

    def available_size(self, pool=None):
        if pool not in (None, PoolName.KV, PoolName.INDEXER):
            raise ValueError("Unknown sharded host pool")
        return self.logical_allocator.available_size()


class ShardedDSAHiCacheController(HybridCacheController):
    def __init__(self, token_to_kv_pool_allocator, mem_pool_host, *args, **kwargs):
        if not isinstance(token_to_kv_pool_allocator, PageInterleavePoolAllocator):
            raise ValueError("Sharded HiCache requires the page-interleave allocator")
        if kwargs.get("storage_backend") is not None:
            raise ValueError("CP-sharded HiCache supports CPU L2 only")
        if kwargs.get("host_memory_mode", "cache") != "cache":
            raise ValueError("CP-sharded HiCache requires host memory mode cache")
        if kwargs.get("io_backend", "kernel") != "kernel":
            raise ValueError("CP-sharded HiCache requires the kernel I/O backend")
        super().__init__(token_to_kv_pool_allocator, mem_pool_host, *args, **kwargs)
        # Native packed draft transfers finish before target layer zero is
        # published. Draft gather streams must wait on the same generation.
        registered = set()
        for entry in mem_pool_host.entries:
            for draft in entry.packed_draft_device_pools:
                if id(draft) not in registered:
                    draft.register_layer_transfer_counter(self.layer_done_counter)
                    registered.add(id(draft))

    def attach_storage_backend(self, *args, **kwargs):
        raise ValueError("CP-sharded HiCache supports CPU L2 only")

    @staticmethod
    def _validate_sidecars(transfers):
        for transfer in transfers or ():
            if (
                transfer.name != PoolName.INDEXER
                or transfer.indices_from_pool != PoolName.KV
            ):
                raise ValueError("Sharded DSA sidecar indices must follow KV")

    def allocate_host_transfers(
        self, device_indices, extra_pools=None, *, reclaim=None
    ):
        self._validate_sidecars(extra_pools)
        host_indices = self.mem_pool_host.alloc_matching(
            device_indices, reclaim=reclaim
        )
        if host_indices is None:
            return None
        transfers = self.mem_pool_host.resolve_host_transfers(
            extra_pools,
            primary_device_indices=device_indices,
            primary_host_indices=host_indices,
        )
        if transfers is None and extra_pools:
            self.mem_pool_host.free(host_indices)
            return None
        return host_indices, transfers

    def load(self, host_indices, priority=None, node_id=-1, extra_pools=None):
        self._validate_sidecars(extra_pools)
        allocator = self.mem_pool_device_allocator
        device_indices = allocator.alloc_matching(host_indices)
        if device_indices is None:
            return None
        transfers = self._resolve_device_transfers(
            extra_pools, kv_device_indices=device_indices, kv_host_indices=host_indices
        )
        if transfers is None and extra_pools:
            allocator.free(device_indices)
            return None
        self.load_queue.append(
            CacheOperation(
                host_indices,
                device_indices,
                node_id,
                priority,
                pool_transfers=transfers or None,
            )
        )
        return device_indices

    def _move_op_indices(self, op):
        # Delay localization to _l2_transfers: retraction backup/restore calls
        # that interface directly, without going through the queued path.
        return op.host_indices, op.device_indices, op.pool_transfers

    _move_write_operation = _move_op_indices

    def _local_indices(self, host_indices, device_indices):
        return local_transfer_indices(
            host_indices,
            device_indices,
            page_size=self.page_size,
            rank=self.mem_pool_host.shard_rank,
            size=self.mem_pool_host.shard_size,
        )

    def _l2_transfers(self, host_indices, device_indices, pool_transfers=None):
        transfers = super()._l2_transfers(host_indices, device_indices, pool_transfers)
        local = []
        for transfer in transfers:
            host, device = self._local_indices(
                transfer.host_indices, transfer.device_indices
            )
            if not host.numel():
                continue
            host, device = self._move_pool_indices(
                transfer.host_pool, host, device.to(self.device), write_back_jit=True
            )
            local.append(transfer._replace(host_indices=host, device_indices=device))
        # An empty owner still submits the operation: ACKs and layer-ready
        # events participate in rank consensus even when this rank copies zero bytes.
        return local

    def _l2_load_transfers(self, host_indices, device_indices, pool_transfers=None):
        transfers = super()._l2_load_transfers(
            host_indices, device_indices, pool_transfers
        )
        result = []
        for transfer in transfers:
            host, device = self._move_pool_indices(
                transfer.host_pool,
                transfer.host_indices,
                transfer.device_indices,
                write_back_jit=False,
            )
            result.append(transfer._replace(host_indices=host, device_indices=device))
        return result

    def _transfer_num_bytes(self, op):
        host, _ = self._local_indices(op.host_indices, op.device_indices)
        names = [self.mem_pool_host.anchor_entry.name]
        names.extend(t.name for t in op.pool_transfers or ())
        return len(host) * sum(
            self.mem_pool_host.entry_map[name].host_pool.size_per_token
            for name in names
        )
