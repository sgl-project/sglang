from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


class MiniCPMCompressedCache:
    def __init__(
        self,
        pool: ReqToTokenPool,
        allocator: BaseTokenToKVPoolAllocator,
        *,
        kernel_size: int,
        kernel_stride: int,
        enable_memory_saver: bool,
    ):
        self.pool = pool
        self.allocator = allocator
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        saver = TorchMemorySaverAdapter.create(enable=enable_memory_saver)
        with saver.region(GPU_MEMORY_TYPE_KV_CACHE):
            k1_size = (pool.max_context_len - kernel_size) // kernel_stride + 1
            k2_size = (pool.max_context_len - kernel_size * 4) // (
                kernel_stride * 4
            ) + 1
            pool.req_to_sparse_k1_token = torch.zeros(
                (pool._alloc_size, k1_size), dtype=torch.int32, device=pool.device
            )
            pool.req_to_sparse_k2_token = torch.zeros(
                (pool._alloc_size, k2_size), dtype=torch.int32, device=pool.device
            )
        self.allocated_lens = [
            [0] * pool._alloc_size,
            [0] * pool._alloc_size,
        ]
        self.reserved_slots = torch.empty(0, dtype=torch.int64, device=pool.device)
        self.free_slots = self.reserved_slots
        self.reset_allocator()

    def reset_allocator(self) -> None:
        """Reserve K1/K2 capacity after the backing allocator is cleared."""
        if self.allocator.page_size != 1:
            raise ValueError("MiniCPM sparse attention requires page_size=1")

        total_slots = self.allocator.available_size()
        # K1 and K2 consume at most 1/s and 1/(4s) slots per dense token.
        denominator = 4 * self.kernel_stride + 5
        reserve_size = (5 * total_slots + denominator - 1) // denominator
        reserved_slots = self.allocator.alloc(reserve_size)
        if reserved_slots is None:
            raise RuntimeError(
                f"Unable to reserve {reserve_size} MiniCPM compressed-cache slots"
            )

        self.dense_capacity = total_slots - reserve_size
        self.reserved_slots = reserved_slots
        self.free_slots = reserved_slots
        self.clear()

    def _alloc_reserved(self, size: int) -> torch.Tensor:
        if size > len(self.free_slots):
            raise RuntimeError(
                "MiniCPM compressed cache is out of reserved slots: "
                f"requested={size}, available={len(self.free_slots)}"
            )
        slots = self.free_slots[:size]
        self.free_slots = self.free_slots[size:]
        return slots

    def _free_reserved(self, slots: torch.Tensor) -> None:
        self.free_slots = torch.cat(
            (self.free_slots, slots.to(self.reserved_slots.dtype))
        )

    def _sparse_len(self, length: int, scale: int) -> int:
        kernel_size = self.kernel_size * scale
        if length < kernel_size:
            return 0
        return (length - kernel_size) // (self.kernel_stride * scale) + 1

    def alloc_to_lengths(
        self,
        *,
        req_pool_indices_cpu: torch.Tensor,
        target_seq_lens_cpu: torch.Tensor,
    ) -> None:
        req_indices = req_pool_indices_cpu.tolist()
        seq_lens = target_seq_lens_cpu.tolist()
        tables = (
            self.pool.req_to_sparse_k1_token,
            self.pool.req_to_sparse_k2_token,
        )
        plans = []
        for level, (table, scale) in enumerate(zip(tables, (1, 4))):
            targets = {
                req_idx: self._sparse_len(seq_len, scale)
                for req_idx, seq_len in zip(req_indices, seq_lens)
            }
            rows = [
                (req_idx, self.allocated_lens[level][req_idx], target)
                for req_idx, target in targets.items()
                if target > self.allocated_lens[level][req_idx]
            ]
            plans.append((table, rows, sum(end - start for _, start, end in rows)))

        allocated = []
        try:
            for _, _, size in plans:
                allocated.append(self._alloc_reserved(size) if size > 0 else None)

            for (table, rows, _), locs in zip(plans, allocated):
                if locs is None:
                    continue
                offset = 0
                for req_idx, start, end in rows:
                    count = end - start
                    table[req_idx, start:end] = locs[offset : offset + count].to(
                        torch.int32
                    )
                    offset += count
            for level, (_, rows, _) in enumerate(plans):
                for req_idx, _, end in rows:
                    self.allocated_lens[level][req_idx] = end
        except Exception:
            for locs in allocated:
                if locs is not None:
                    self._free_reserved(locs)
            raise

    def free(self, req_pool_idx: int) -> None:
        allocated = []
        for table, lengths in zip(
            (
                self.pool.req_to_sparse_k1_token,
                self.pool.req_to_sparse_k2_token,
            ),
            self.allocated_lens,
        ):
            length = lengths[req_pool_idx]
            if length > 0:
                allocated.append(table[req_pool_idx, :length].clone())
                table[req_pool_idx, :length].zero_()
                lengths[req_pool_idx] = 0

        if allocated:
            self._free_reserved(torch.cat(allocated))

    def clear(self) -> None:
        self.pool.req_to_sparse_k1_token.zero_()
        self.pool.req_to_sparse_k2_token.zero_()
        for lengths in self.allocated_lens:
            lengths[:] = [0] * len(lengths)
        self.free_slots = self.reserved_slots


def attach_compressed_cache(
    pool: ReqToTokenPool,
    allocator: BaseTokenToKVPoolAllocator,
    *,
    kernel_size: int,
    kernel_stride: int,
    enable_memory_saver: bool,
) -> ReqToTokenPool:
    if isinstance(pool._aux_cache, MiniCPMCompressedCache):
        return pool

    pool.attach_aux_cache(
        MiniCPMCompressedCache(
            pool,
            allocator,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            enable_memory_saver=enable_memory_saver,
        )
    )
    return pool
