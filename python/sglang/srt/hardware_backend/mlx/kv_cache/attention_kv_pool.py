"""Flat attention KV pool for the MLX backend.

Each layer buffer has shape ``(pool_size, n_kv_heads, head_dim)``.
This v1 pool is intentionally uniform: every wrapped softmax-attention
layer must share the same KV shape and full-context KV semantics.
Heterogeneous KV shapes and sliding-window KV need per-layer/window-aware
pools before they can use MLX radix reuse.

Slot 0 is reserved as padding (1-based indexing).
"""

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


class MlxAttentionKVPool:
    """Pre-allocated attention KV pool indexed by integer slot IDs."""

    def __init__(
        self,
        pool_size: int,
        num_layers: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float16,
    ):
        self.pool_size = pool_size
        self.num_layers = num_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Per-attention-layer buffers: (pool_size, n_kv_heads, head_dim)
        self.k_buffer: list[mx.array] = [
            mx.zeros((pool_size, n_kv_heads, head_dim), dtype=dtype)
            for _ in range(num_layers)
        ]
        self.v_buffer: list[mx.array] = [
            mx.zeros((pool_size, n_kv_heads, head_dim), dtype=dtype)
            for _ in range(num_layers)
        ]

        mem_mb = (pool_size * n_kv_heads * head_dim * 2 * num_layers * dtype.size) / (
            1024 * 1024
        )
        logger.info(
            f"MlxAttentionKVPool: {pool_size} slots x {num_layers} layers "
            f"x {n_kv_heads} heads x {head_dim} dim, "
            f"dtype={dtype}, ~{mem_mb:.1f} MB"
        )

    def set_kv(self, layer_id: int, slots: mx.array, k: mx.array, v: mx.array) -> None:
        """Scatter K/V into *slots* for one layer."""
        self.k_buffer[layer_id][slots] = k
        self.v_buffer[layer_id][slots] = v

    def get_kv(self, layer_id: int, slots: mx.array) -> tuple[mx.array, mx.array]:
        """Gather K/V from *slots* for one layer."""
        return self.k_buffer[layer_id][slots], self.v_buffer[layer_id][slots]

    def get_kv_all_layers(self, slots: mx.array) -> tuple[mx.array, mx.array]:
        """Gather K/V from *slots* across all layers."""
        k_all = mx.stack([self.k_buffer[i][slots] for i in range(self.num_layers)])
        v_all = mx.stack([self.v_buffer[i][slots] for i in range(self.num_layers)])
        return k_all, v_all

    def set_kv_all_layers(
        self, slots: mx.array, k_all: mx.array, v_all: mx.array
    ) -> None:
        """Scatter K/V into *slots* across all layers."""
        for i in range(self.num_layers):
            self.set_kv(i, slots, k_all[i], v_all[i])

    def all_buffers(self) -> list[mx.array]:
        """Return all buffer arrays (for ``mx.eval``)."""
        return self.k_buffer + self.v_buffer

    def clear(self) -> None:
        """Zero all buffers."""
        shape = (self.pool_size, self.n_kv_heads, self.head_dim)
        for i in range(self.num_layers):
            self.k_buffer[i] = mx.zeros(shape, dtype=self.dtype)
            self.v_buffer[i] = mx.zeros(shape, dtype=self.dtype)


class MlxBlockAttentionKVPool:
    """Pre-allocated block-table attention KV pool."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float16,
        req_capacity: int | None = None,
    ):
        # Block 0 is reserved for padding/invalid use. Allocated request blocks start
        # at physical block 1.
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.req_capacity = max(
            1, req_capacity if req_capacity is not None else num_blocks
        )
        self._free_blocks: list[int] = list(range(1, num_blocks))
        self._req_block_tables: dict[str, list[int]] = {}
        self._req_pool_rows: dict[str, int] = {}
        self.req_to_block = (
            mx.zeros((self.req_capacity, num_blocks), dtype=mx.int32) - 1
        )

        shape = (num_blocks, block_size, n_kv_heads, head_dim)
        self.k_buffer: list[mx.array] = [
            mx.zeros(shape, dtype=dtype) for _ in range(num_layers)
        ]
        self.v_buffer: list[mx.array] = [
            mx.zeros(shape, dtype=dtype) for _ in range(num_layers)
        ]

        mem_mb = (
            num_blocks
            * block_size
            * n_kv_heads
            * head_dim
            * 2
            * num_layers
            * dtype.size
        ) / (1024 * 1024)
        metadata_mem_mb = (self.req_capacity * num_blocks * mx.int32.size) / (
            1024 * 1024
        )
        logger.info(
            f"MlxBlockAttentionKVPool: {num_blocks} blocks x block_size={block_size} "
            f"x {num_layers} layers x {n_kv_heads} heads x {head_dim} dim, "
            f"dtype={dtype}, ~{mem_mb:.1f} MB KV + "
            f"~{metadata_mem_mb:.1f} MB req_to_block"
        )

    def _check_req_pool_idx(self, req_pool_idx: int) -> int:
        row = int(req_pool_idx)
        if row < 0 or row >= self.req_capacity:
            raise ValueError(
                f"req_pool_idx must be in [0, {self.req_capacity}), got {row}"
            )
        return row

    def _set_req_to_block_row(self, req_id: str, req_pool_idx: int) -> None:
        row = self._check_req_pool_idx(req_pool_idx)
        existing = self._req_pool_rows.get(req_id)
        if existing is not None and existing != row:
            self.req_to_block[existing] = (
                mx.zeros((self.num_blocks,), dtype=mx.int32) - 1
            )
        self._req_pool_rows[req_id] = row
        blocks = self._req_block_tables.get(req_id, [])
        row_values = blocks + [-1] * (self.num_blocks - len(blocks))
        self.req_to_block[row] = mx.array(row_values, dtype=mx.int32)

    def alloc_blocks(
        self, req_id: str, num_blocks: int, req_pool_idx: int | None = None
    ) -> list[int] | None:
        """Allocate physical blocks for a request, or return None if full."""
        if num_blocks < 0:
            raise ValueError(f"num_blocks must be non-negative, got {num_blocks}")
        if req_id in self._req_block_tables:
            raise ValueError(f"request {req_id!r} already has allocated blocks")
        if num_blocks > len(self._free_blocks):
            return None
        blocks = self._free_blocks[:num_blocks]
        del self._free_blocks[:num_blocks]
        self._req_block_tables[req_id] = blocks
        if req_pool_idx is not None:
            self._set_req_to_block_row(req_id, req_pool_idx)
        return list(blocks)

    def ensure_blocks(
        self, req_id: str, num_blocks: int, req_pool_idx: int | None = None
    ) -> list[int] | None:
        """Ensure a request owns at least ``num_blocks`` physical blocks."""
        if num_blocks < 0:
            raise ValueError(f"num_blocks must be non-negative, got {num_blocks}")
        existing = self._req_block_tables.get(req_id, [])
        needed = num_blocks - len(existing)
        if needed <= 0:
            if req_pool_idx is not None:
                self._set_req_to_block_row(req_id, req_pool_idx)
            return list(existing)
        if needed > len(self._free_blocks):
            return None
        new_blocks = self._free_blocks[:needed]
        del self._free_blocks[:needed]
        if req_id not in self._req_block_tables:
            existing = []
            self._req_block_tables[req_id] = existing
        existing.extend(new_blocks)
        if req_pool_idx is not None:
            self._set_req_to_block_row(req_id, req_pool_idx)
        return list(existing)

    def block_table(self, req_id: str) -> list[int]:
        """Return the physical block table for a request."""
        return list(self._req_block_tables[req_id])

    def free_request(self, req_id: str) -> None:
        """Release a request's blocks back to the free list."""
        blocks = self._req_block_tables.pop(req_id, None)
        row = self._req_pool_rows.pop(req_id, None)
        if row is not None:
            self.req_to_block[row] = mx.zeros((self.num_blocks,), dtype=mx.int32) - 1
        if not blocks:
            return
        self._free_blocks = sorted(blocks + self._free_blocks)

    def set_kv(
        self,
        layer_id: int,
        block_ids: mx.array,
        block_offsets: mx.array,
        k: mx.array,
        v: mx.array,
    ) -> None:
        """Scatter K/V into ``[block_id, block_offset]`` locations."""
        if block_ids.ndim != 1 or block_offsets.ndim != 1:
            raise ValueError("block_ids and block_offsets must be 1-D")
        if tuple(block_ids.shape) != tuple(block_offsets.shape):
            raise ValueError("block_ids and block_offsets must have matching shape")
        if tuple(k.shape) != (block_ids.shape[0], self.n_kv_heads, self.head_dim):
            raise ValueError(
                "k must have shape [num_tokens, n_kv_heads, head_dim], "
                f"got {k.shape}"
            )
        if tuple(v.shape) != tuple(k.shape):
            raise ValueError(f"v shape must match k shape, got {v.shape} vs {k.shape}")
        self.k_buffer[layer_id][block_ids, block_offsets] = k
        self.v_buffer[layer_id][block_ids, block_offsets] = v

    def get_kv(self, layer_id: int) -> tuple[mx.array, mx.array]:
        """Return full block K/V buffers for one layer."""
        return self.k_buffer[layer_id], self.v_buffer[layer_id]

    def all_buffers(self) -> list[mx.array]:
        """Return all buffer arrays (for ``mx.eval``)."""
        return self.k_buffer + self.v_buffer + [self.req_to_block]

    def materialize(self) -> None:
        """Force lazy block-pool buffers to be allocated before serving decode."""
        mx.eval(*self.all_buffers())

    def clear(self) -> None:
        """Zero all buffers and clear request ownership."""
        shape = (self.num_blocks, self.block_size, self.n_kv_heads, self.head_dim)
        for i in range(self.num_layers):
            self.k_buffer[i] = mx.zeros(shape, dtype=self.dtype)
            self.v_buffer[i] = mx.zeros(shape, dtype=self.dtype)
        self.req_to_block = (
            mx.zeros((self.req_capacity, self.num_blocks), dtype=mx.int32) - 1
        )
        self._free_blocks = list(range(1, self.num_blocks))
        self._req_block_tables.clear()
        self._req_pool_rows.clear()
