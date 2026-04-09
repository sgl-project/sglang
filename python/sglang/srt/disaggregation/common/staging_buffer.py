"""
GPU Staging Buffer for heterogeneous TP KV cache transfer.

When prefill attn_tp_size != decode attn_tp_size, the per-token RDMA approach
generates O(tokens * layers) small RDMA requests. This module provides a staging
buffer mechanism that gathers scattered head slices into contiguous GPU memory,
enabling bulk RDMA transfers that reduce request count to O(layers) or O(1).

Usage:
    Activated by setting SGLANG_DISAGG_STAGING_BUFFER=1.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# TODO(yangminl): remove torch fallback implementations once the Triton kernels
# have been validated in production across all configurations.
_USE_TRITON_STAGING = not bool(os.environ.get("SGLANG_STAGING_USE_TORCH", ""))


@triton.jit
def _fused_gather_to_staging_kernel(
    layer_ptrs,
    page_indices,
    staging,
    num_tokens,
    stride_pool_token,
    head_offset,
    per_layer_elems,
    ELEMS_PER_TOKEN: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    layer_id = tl.program_id(0)
    block_id = tl.program_id(1)

    layer_ptr = tl.load(layer_ptrs + layer_id).to(staging.dtype)

    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < per_layer_elems

    t_idx = offsets // ELEMS_PER_TOKEN
    e_idx = offsets % ELEMS_PER_TOKEN

    page_id = t_idx // PAGE_SIZE
    intra_page = t_idx % PAGE_SIZE
    page_val = tl.load(page_indices + page_id, mask=mask, other=0)
    pool_token = page_val * PAGE_SIZE + intra_page

    src_offsets = (
        pool_token * stride_pool_token.to(tl.int64) + head_offset.to(tl.int64) + e_idx
    )
    vals = tl.load(layer_ptr + src_offsets, mask=mask)

    dst_offsets = tl.program_id(0).to(tl.int64) * per_layer_elems.to(tl.int64) + offsets
    tl.store(staging + dst_offsets, vals, mask=mask)


@triton.jit
def _fused_scatter_from_staging_kernel(
    layer_ptrs,
    page_indices,
    staging,
    writer_head_offsets,
    num_tokens,
    stride_pool_token,
    per_layer_elems,
    ELEMS_PER_TOKEN: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_LAYERS_X2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    prog_id = tl.program_id(0)
    block_id = tl.program_id(1)

    writer_id = prog_id // NUM_LAYERS_X2
    layer_kv_id = prog_id % NUM_LAYERS_X2

    layer_ptr = tl.load(layer_ptrs + layer_kv_id).to(staging.dtype)
    head_offset = tl.load(writer_head_offsets + writer_id)

    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < per_layer_elems

    t_idx = offsets // ELEMS_PER_TOKEN
    e_idx = offsets % ELEMS_PER_TOKEN

    page_id = t_idx // PAGE_SIZE
    intra_page = t_idx % PAGE_SIZE
    page_val = tl.load(page_indices + page_id, mask=mask, other=0)
    pool_token = page_val * PAGE_SIZE + intra_page

    per_rank_elems = per_layer_elems.to(tl.int64) * NUM_LAYERS_X2
    src_offsets = (
        writer_id.to(tl.int64) * per_rank_elems
        + layer_kv_id.to(tl.int64) * per_layer_elems.to(tl.int64)
        + offsets
    )
    vals = tl.load(staging + src_offsets, mask=mask)

    dst_offsets = (
        pool_token * stride_pool_token.to(tl.int64) + head_offset.to(tl.int64) + e_idx
    )
    tl.store(layer_ptr + dst_offsets, vals, mask=mask)


class StagingBuffer:
    """Pre-allocated GPU staging buffer for bulk KV transfer.

    When a custom_mem_pool is provided (e.g., mooncake NVLink allocator),
    the buffer is allocated within that pool so it's compatible with
    NVLink/MNNVL transport (requires cuMemCreate-backed memory).
    """

    def __init__(
        self,
        size_bytes: int,
        device: str,
        gpu_id: int,
        custom_mem_pool=None,
    ):
        self.size_bytes = size_bytes
        self.device = device
        self.gpu_id = gpu_id

        torch.cuda.set_device(gpu_id)
        if custom_mem_pool is not None:
            with torch.cuda.use_mem_pool(custom_mem_pool):
                self.buffer = torch.empty(size_bytes, dtype=torch.uint8, device=device)
            alloc_method = "custom_mem_pool (cuMemCreate)"
        else:
            self.buffer = torch.empty(size_bytes, dtype=torch.uint8, device=device)
            alloc_method = "cudaMalloc"
        self.data_ptr = self.buffer.data_ptr()

        logger.info(
            f"StagingBuffer allocated: {size_bytes / (1024*1024):.1f} MB "
            f"on {device}, method={alloc_method}, ptr=0x{self.data_ptr:x}"
        )

    def get_ptr(self) -> int:
        return self.data_ptr

    def get_size(self) -> int:
        return self.size_bytes

    def fits(self, required_bytes: int) -> bool:
        return required_bytes <= self.size_bytes


class StagingAllocator:
    """Decode-side dynamic staging ring buffer allocator with overcommit.

    One large pre-allocated GPU buffer used as a ring buffer. Each request
    gets a (alloc_id, offset, round) triple based on its actual byte
    requirement. Allocation (assign) is overcommit — it always succeeds
    as long as the request fits in the buffer. Overlap safety is enforced
    on the prefill side before RDMA, using a watermark that tracks the
    oldest un-freed allocation.

    The watermark (round, tail_offset) is periodically sent to prefill.
    Prefill transfer workers wait before writing if their target region
    overlaps with not-yet-freed data from a previous round.
    """

    # Permanent alloc failure: chunk exceeds ring buffer total size.
    ALLOC_OVERSIZED = -2

    def __init__(
        self,
        total_size_bytes: int,
        device: str,
        gpu_id: int,
        custom_mem_pool=None,
    ):
        self.buffer = StagingBuffer(total_size_bytes, device, gpu_id, custom_mem_pool)
        self.total_size = total_size_bytes
        self.base_ptr = self.buffer.data_ptr
        self.head = 0
        self.round = 0
        self.allocations: dict = {}  # alloc_id -> (offset, size, round)
        self.alloc_order: List[int] = []
        self.next_alloc_id = 0
        self.watermark_round = 0
        self.watermark_tail = 0
        self.lock = threading.Lock()

        logger.info(
            f"StagingAllocator (ring+overcommit): "
            f"{total_size_bytes / (1024*1024):.1f} MB "
            f"on {device}, ptr=0x{self.base_ptr:x}"
        )

    def assign(self, required_bytes: int) -> Optional[Tuple[int, int, int]]:
        """Allocate a region. Returns (alloc_id, offset, round) or None."""
        with self.lock:
            if required_bytes > self.total_size:
                return None

            space_at_end = self.total_size - self.head
            if required_bytes <= space_at_end:
                offset = self.head
                self.head += required_bytes
            else:
                self.round += 1
                offset = 0
                self.head = required_bytes

            alloc_id = self.next_alloc_id
            self.next_alloc_id += 1
            self.allocations[alloc_id] = (offset, required_bytes, self.round)
            self.alloc_order.append(alloc_id)
            return (alloc_id, offset, self.round)

    def free(self, alloc_id: int):
        """Free an allocation and advance watermark past consecutive freed entries."""
        with self.lock:
            if alloc_id not in self.allocations:
                return
            self.allocations.pop(alloc_id)

            while self.alloc_order and self.alloc_order[0] not in self.allocations:
                self.alloc_order.pop(0)

            if not self.allocations:
                self.watermark_round = self.round
                self.watermark_tail = self.head
            elif self.alloc_order:
                off, _, rnd = self.allocations[self.alloc_order[0]]
                self.watermark_round = rnd
                self.watermark_tail = off

    def get_watermark(self) -> Tuple[int, int]:
        """Return (round, tail_offset). Everything before this is safe to write."""
        with self.lock:
            return (self.watermark_round, self.watermark_tail)

    def get_ptr(self, alloc_id: int) -> int:
        offset, _, _ = self.allocations[alloc_id]
        return self.base_ptr + offset

    def get_offset(self, alloc_id: int) -> int:
        offset, _, _ = self.allocations[alloc_id]
        return offset

    def get_round(self, alloc_id: int) -> int:
        _, _, rnd = self.allocations[alloc_id]
        return rnd

    def get_base_ptr(self) -> int:
        return self.base_ptr

    def get_total_size(self) -> int:
        return self.total_size


def gather_kv_head_slices(
    kv_buffer_tensor: torch.Tensor,
    gather_idx: torch.Tensor,
    head_start: int,
    num_heads: int,
    staging_tensor: torch.Tensor,
):
    """Gather KV head slices from scattered pages into contiguous staging buffer.

    Uses torch.gather(out=) to write directly into staging_tensor without
    allocating temporary tensors (avoids CUDA caching allocator stalls).

    Args:
        kv_buffer_tensor: [pool_size, head_num, head_dim], one layer.
        gather_idx: [num_tokens, num_heads, head_dim] int64, pre-computed
            token indices expanded for gather on dim=0.
        head_start: Starting head index for the slice.
        num_heads: Number of heads to gather.
        staging_tensor: Output tensor, shape [num_tokens, num_heads, head_dim].
    """
    src = kv_buffer_tensor[:, head_start : head_start + num_heads, :]
    torch.gather(src, 0, gather_idx, out=staging_tensor)


def scatter_kv_head_slices(
    staging_tensor: torch.Tensor,
    kv_buffer_tensor: torch.Tensor,
    page_indices: torch.Tensor,
    head_start: int,
    num_heads: int,
    page_size: int = 1,
):
    """Scatter KV head slices from contiguous staging buffer to KV cache.

    Args:
        staging_tensor: Input tensor from staging buffer (contiguous packed data).
        kv_buffer_tensor: The KV buffer for one layer, shape [pool_size, head_num, head_dim].
        page_indices: [num_pages] int32/int64 tensor of page indices.
        head_start: Starting head index for the slice.
        num_heads: Number of heads to scatter.
        page_size: Number of tokens per page.
    """
    head_dim = kv_buffer_tensor.shape[-1]
    if page_size == 1:
        num_tokens = page_indices.shape[0]
        data = staging_tensor.reshape(num_tokens, num_heads, head_dim)
        kv_buffer_tensor[page_indices, head_start : head_start + num_heads, :] = data
    else:
        num_tokens = page_indices.shape[0] * page_size
        offsets = torch.arange(page_size, device=page_indices.device)
        token_indices = (page_indices.unsqueeze(1) * page_size + offsets).reshape(-1)
        data = staging_tensor.reshape(num_tokens, num_heads, head_dim)
        kv_buffer_tensor[token_indices, head_start : head_start + num_heads, :] = data


def _gather_all_layers_torch(
    k_buffers: list,
    v_buffers: list,
    page_indices_np,
    staging_buffer: StagingBuffer,
    src_head_start: int,
    num_heads: int,
    page_size: int,
    gpu_id: int,
) -> int:
    """torch.gather path: zero per-layer allocation, one kernel per layer."""
    import numpy as np

    num_layers = len(k_buffers)
    head_dim = k_buffers[0].shape[-1]
    dtype_size = k_buffers[0].element_size()
    num_tokens = len(page_indices_np) * page_size
    per_layer_bytes = num_tokens * num_heads * head_dim * dtype_size

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    page_idx_tensor = torch.from_numpy(page_indices_np.astype(np.int64)).to(device)

    if page_size == 1:
        token_indices = page_idx_tensor
    else:
        offsets = torch.arange(page_size, device=device)
        token_indices = (page_idx_tensor.unsqueeze(1) * page_size + offsets).reshape(-1)

    gather_idx = token_indices.view(-1, 1, 1).expand(num_tokens, num_heads, head_dim)

    if not hasattr(staging_buffer, "_gather_stream"):
        staging_buffer._gather_stream = torch.cuda.Stream(device=device)

    staging_buffer._gather_stream.wait_stream(
        torch.cuda.default_stream(torch.device(device))
    )

    staging_view = staging_buffer.buffer
    offset = 0
    with torch.cuda.stream(staging_buffer._gather_stream):
        for layer_id in range(num_layers):
            dst = (
                staging_view[offset : offset + per_layer_bytes]
                .view(k_buffers[layer_id].dtype)
                .reshape(num_tokens, num_heads, head_dim)
            )
            gather_kv_head_slices(
                k_buffers[layer_id],
                gather_idx,
                src_head_start,
                num_heads,
                dst,
            )
            offset += per_layer_bytes
        for layer_id in range(num_layers):
            dst = (
                staging_view[offset : offset + per_layer_bytes]
                .view(v_buffers[layer_id].dtype)
                .reshape(num_tokens, num_heads, head_dim)
            )
            gather_kv_head_slices(
                v_buffers[layer_id],
                gather_idx,
                src_head_start,
                num_heads,
                dst,
            )
            offset += per_layer_bytes

    staging_buffer._gather_stream.synchronize()
    return offset


def _gather_all_layers_triton(
    k_buffers: list,
    v_buffers: list,
    page_indices_np,
    staging_buffer: StagingBuffer,
    src_head_start: int,
    num_heads: int,
    page_size: int,
    gpu_id: int,
) -> int:
    """Triton fused kernel path: single kernel launch for all layers."""
    import numpy as np

    num_layers = len(k_buffers)
    head_dim = k_buffers[0].shape[-1]
    total_heads = k_buffers[0].shape[1]
    dtype_size = k_buffers[0].element_size()
    num_tokens = len(page_indices_np) * page_size
    elems_per_token = num_heads * head_dim
    per_layer_elems = num_tokens * elems_per_token
    per_layer_bytes = per_layer_elems * dtype_size
    total_bytes = per_layer_bytes * num_layers * 2

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    page_idx_tensor = torch.from_numpy(page_indices_np.astype(np.int64)).to(device)

    layer_ptrs = torch.tensor(
        [buf.data_ptr() for buf in k_buffers] + [buf.data_ptr() for buf in v_buffers],
        dtype=torch.int64,
        device=device,
    )
    # Use integer dtype matching element size for bit-preserving copy
    int_dtype_map = {1: torch.int8, 2: torch.int16, 4: torch.int32}
    int_dtype = int_dtype_map.get(dtype_size, torch.int16)
    staging_typed = staging_buffer.buffer[:total_bytes].view(int_dtype)

    if not hasattr(staging_buffer, "_gather_stream"):
        staging_buffer._gather_stream = torch.cuda.Stream(device=device)

    staging_buffer._gather_stream.wait_stream(
        torch.cuda.default_stream(torch.device(device))
    )

    BLOCK_SIZE = 1024
    grid = (2 * num_layers, triton.cdiv(per_layer_elems, BLOCK_SIZE))

    with torch.cuda.stream(staging_buffer._gather_stream):
        _fused_gather_to_staging_kernel[grid](
            layer_ptrs,
            page_idx_tensor,
            staging_typed,
            num_tokens,
            total_heads * head_dim,
            src_head_start * head_dim,
            per_layer_elems,
            elems_per_token,
            page_size,
            BLOCK_SIZE,
        )

    staging_buffer._gather_stream.synchronize()
    return total_bytes


def gather_all_layers_to_staging(
    k_buffers: list,
    v_buffers: list,
    page_indices_np,
    staging_buffer: StagingBuffer,
    src_head_start: int,
    num_heads: int,
    page_size: int,
    gpu_id: int,
) -> int:
    """Gather all layers' K and V head slices into a staging buffer.

    Returns total bytes written.
    Dispatches to Triton fused kernel when available, falls back to torch.gather.
    """
    if _USE_TRITON_STAGING:
        return _gather_all_layers_triton(
            k_buffers,
            v_buffers,
            page_indices_np,
            staging_buffer,
            src_head_start,
            num_heads,
            page_size,
            gpu_id,
        )
    return _gather_all_layers_torch(
        k_buffers,
        v_buffers,
        page_indices_np,
        staging_buffer,
        src_head_start,
        num_heads,
        page_size,
        gpu_id,
    )


def _scatter_staging_to_kv_torch(
    staging_buffer_view: torch.Tensor,
    k_buffers: list,
    v_buffers: list,
    page_idx_tensor: torch.Tensor,
    page_size: int,
    prefill_attn_tp_size: int,
    decode_attn_tp_size: int,
    dst_tp_rank: int,
    total_kv_heads: int,
) -> None:
    """torch path for scatter."""
    num_layers = len(k_buffers)
    head_dim = k_buffers[0].shape[-1]
    dtype_size = k_buffers[0].element_size()
    num_tokens = page_idx_tensor.shape[0] * page_size

    if prefill_attn_tp_size > decode_attn_tp_size:
        num_writers = prefill_attn_tp_size // max(1, decode_attn_tp_size)
    else:
        num_writers = 1

    for writer_rank in range(num_writers):
        _, num_heads, dst_head_start, _ = compute_head_slice_params(
            prefill_attn_tp_size,
            decode_attn_tp_size,
            writer_rank,
            dst_tp_rank,
            total_kv_heads,
        )
        per_layer_bytes = num_tokens * num_heads * head_dim * dtype_size
        per_rank_bytes = per_layer_bytes * num_layers * 2
        rank_base = writer_rank * per_rank_bytes

        offset = rank_base
        for layer_id in range(num_layers):
            layer_data = (
                staging_buffer_view[offset : offset + per_layer_bytes]
                .view(k_buffers[layer_id].dtype)
                .reshape(num_tokens, num_heads, head_dim)
            )
            scatter_kv_head_slices(
                layer_data,
                k_buffers[layer_id],
                page_idx_tensor,
                dst_head_start,
                num_heads,
                page_size,
            )
            offset += per_layer_bytes
        for layer_id in range(num_layers):
            layer_data = (
                staging_buffer_view[offset : offset + per_layer_bytes]
                .view(v_buffers[layer_id].dtype)
                .reshape(num_tokens, num_heads, head_dim)
            )
            scatter_kv_head_slices(
                layer_data,
                v_buffers[layer_id],
                page_idx_tensor,
                dst_head_start,
                num_heads,
                page_size,
            )
            offset += per_layer_bytes


def _scatter_staging_to_kv_triton(
    staging_buffer_view: torch.Tensor,
    k_buffers: list,
    v_buffers: list,
    page_idx_tensor: torch.Tensor,
    page_size: int,
    prefill_attn_tp_size: int,
    decode_attn_tp_size: int,
    dst_tp_rank: int,
    total_kv_heads: int,
) -> None:
    """Triton fused kernel path for scatter."""
    num_layers = len(k_buffers)
    head_dim = k_buffers[0].shape[-1]
    total_heads = k_buffers[0].shape[1]
    dtype_size = k_buffers[0].element_size()
    num_tokens = page_idx_tensor.shape[0] * page_size
    device = page_idx_tensor.device

    if prefill_attn_tp_size > decode_attn_tp_size:
        num_writers = prefill_attn_tp_size // max(1, decode_attn_tp_size)
    else:
        num_writers = 1

    # All writers share the same num_heads; only dst_head_start differs
    _, num_heads, _, _ = compute_head_slice_params(
        prefill_attn_tp_size,
        decode_attn_tp_size,
        0,
        dst_tp_rank,
        total_kv_heads,
    )
    elems_per_token = num_heads * head_dim
    per_layer_elems = num_tokens * elems_per_token

    layer_ptrs = torch.tensor(
        [buf.data_ptr() for buf in k_buffers] + [buf.data_ptr() for buf in v_buffers],
        dtype=torch.int64,
        device=device,
    )

    writer_head_offsets = torch.tensor(
        [
            compute_head_slice_params(
                prefill_attn_tp_size,
                decode_attn_tp_size,
                wr,
                dst_tp_rank,
                total_kv_heads,
            )[2]
            * head_dim
            for wr in range(num_writers)
        ],
        dtype=torch.int64,
        device=device,
    )

    int_dtype_map = {1: torch.int8, 2: torch.int16, 4: torch.int32}
    int_dtype = int_dtype_map.get(dtype_size, torch.int16)
    total_staging_bytes = (
        num_tokens * elems_per_token * dtype_size * num_layers * 2 * num_writers
    )
    staging_typed = staging_buffer_view[:total_staging_bytes].view(int_dtype)

    BLOCK_SIZE = 1024
    num_layers_x2 = 2 * num_layers
    grid = (num_writers * num_layers_x2, triton.cdiv(per_layer_elems, BLOCK_SIZE))

    _fused_scatter_from_staging_kernel[grid](
        layer_ptrs,
        page_idx_tensor,
        staging_typed,
        writer_head_offsets,
        num_tokens,
        total_heads * head_dim,
        per_layer_elems,
        elems_per_token,
        page_size,
        num_layers_x2,
        BLOCK_SIZE,
    )


def scatter_staging_to_kv(
    staging_buffer_view: torch.Tensor,
    k_buffers: list,
    v_buffers: list,
    page_idx_tensor: torch.Tensor,
    page_size: int,
    prefill_attn_tp_size: int,
    decode_attn_tp_size: int,
    dst_tp_rank: int,
    total_kv_heads: int,
) -> None:
    """Scatter data from a contiguous staging region into KV cache buffers."""
    if _USE_TRITON_STAGING:
        return _scatter_staging_to_kv_triton(
            staging_buffer_view,
            k_buffers,
            v_buffers,
            page_idx_tensor,
            page_size,
            prefill_attn_tp_size,
            decode_attn_tp_size,
            dst_tp_rank,
            total_kv_heads,
        )
    return _scatter_staging_to_kv_torch(
        staging_buffer_view,
        k_buffers,
        v_buffers,
        page_idx_tensor,
        page_size,
        prefill_attn_tp_size,
        decode_attn_tp_size,
        dst_tp_rank,
        total_kv_heads,
    )


def compute_head_slice_params(
    src_attn_tp_size: int,
    dst_attn_tp_size: int,
    src_tp_rank: int,
    dst_tp_rank: int,
    total_kv_heads: int,
) -> Tuple[int, int, int, int]:
    """Compute head slicing parameters for heterogeneous TP transfer.

    Returns:
        (src_head_start, num_heads_to_send, dst_head_start, num_heads_to_send)
    """
    src_heads_per_rank = max(1, total_kv_heads // src_attn_tp_size)
    dst_heads_per_rank = max(1, total_kv_heads // dst_attn_tp_size)

    local_tp_rank = src_tp_rank % src_attn_tp_size
    dst_tp_rank_in_group = dst_tp_rank % dst_attn_tp_size

    if src_attn_tp_size > dst_attn_tp_size:
        src_head_start = 0
        num_heads_to_send = src_heads_per_rank
        src_replication = max(1, src_attn_tp_size // total_kv_heads)
        unique_head_idx = local_tp_rank // src_replication
        dst_head_start = (unique_head_idx * src_heads_per_rank) % dst_heads_per_rank
    else:
        src_head_start = (
            dst_tp_rank_in_group * dst_heads_per_rank
        ) % src_heads_per_rank
        num_heads_to_send = dst_heads_per_rank
        dst_head_start = 0

    return src_head_start, num_heads_to_send, dst_head_start, num_heads_to_send


def compute_staging_layout(
    src_attn_tp_size: int,
    dst_attn_tp_size: int,
    dst_tp_rank: int,
    total_kv_heads: int,
    num_tokens: int,
    bytes_per_head_token: int,
    num_layers: int,
) -> Tuple[int, List[int], int]:
    """Compute per-writer byte layout for a staging region.

    Returns:
        (num_writers, writer_bytes_list, total_bytes)
        where writer_bytes_list[i] = bytes for writer i covering all layers (K+V).
    """
    if src_attn_tp_size > dst_attn_tp_size:
        num_writers = src_attn_tp_size // max(1, dst_attn_tp_size)
    else:
        num_writers = 1

    writer_bytes = []
    for wr in range(num_writers):
        _, nh, _, _ = compute_head_slice_params(
            src_attn_tp_size,
            dst_attn_tp_size,
            wr,
            dst_tp_rank,
            total_kv_heads,
        )
        writer_bytes.append(num_tokens * nh * bytes_per_head_token * num_layers * 2)
    return num_writers, writer_bytes, sum(writer_bytes)


def resolve_total_kv_heads(
    kv_args,
    attn_tp_size: int,
) -> int:
    """Resolve the global total KV head count from kv_args metadata."""
    total = getattr(kv_args, "total_kv_head_num", 0)
    if total > 0:
        return total
    per_rank = getattr(kv_args, "kv_head_num", 0)
    if per_rank > 0:
        return per_rank * attn_tp_size
    raise ValueError(
        "Cannot resolve total_kv_heads: kv_args has neither total_kv_head_num "
        "nor kv_head_num. "
        "Ensure DecodePreallocQueue._init_kv_manager sets kv_args.kv_head_num."
    )
