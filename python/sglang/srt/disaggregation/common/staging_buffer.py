"""
GPU staging buffer for CP and heterogeneous TP KV cache transfer.

When prefill attn_tp_size != decode attn_tp_size, the per-token RDMA approach
generates O(tokens * layers) small RDMA requests. This module provides a staging
buffer mechanism that gathers scattered head slices into contiguous GPU memory,
enabling bulk RDMA transfers that reduce request count to O(layers) or O(1).

Usage:
    Activated by setting SGLANG_DISAGG_STAGING_BUFFER=1.
"""

from __future__ import annotations

import logging
import threading
import time
from functools import cache
from itertools import accumulate
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.memory.ptr_table import make_ptr_table
from sglang.srt.environ import envs
from sglang.srt.layers.cp.utils import compute_kv_head_overlap
from sglang.srt.utils import create_device_stream, device_stream_context

logger = logging.getLogger(__name__)

# TODO(yangminl): remove torch fallback implementations once the Triton kernels
# have been validated in production across all configurations.
_USE_TRITON_STAGING = not envs.SGLANG_STAGING_USE_TORCH.get()

# CUDA (including ROCm, which reports "cuda") is the original staging target;
# XPU is the only other device validated on this path.
_STAGING_SUPPORTED_DEVICE_TYPES = ("cuda", "xpu")


def _device_ptr_type_str(ptr: int, device_type: str) -> str:
    """XPU kernel-space pointers set bit 63; empty string off XPU."""
    if device_type != "xpu":
        return ""
    if ptr & (1 << 63):
        return ", ptr_type=XPU-kernel-space"
    else:
        return ", ptr_type=XPU-user-space"


def _check_kv_on_staging_device(
    k_buffers: list, v_buffers: list, staging_device: torch.device
) -> None:
    assert k_buffers, "k_buffers must not be empty"
    assert v_buffers, "v_buffers must not be empty"
    device = k_buffers[0].device
    # The kernels read KV pointers and write staging in one launch, so a
    # cross-device pair would fault inside the kernel rather than here.
    assert device == staging_device, (
        f"KV buffers are on {device} but staging is on {staging_device}; "
        f"both must be the same device."
    )


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


@triton.jit
def _scatter_from_staging_persistent_kernel(
    layer_ptrs,
    page_indices,
    staging,
    writer_head_offsets,
    stride_pool_token,
    per_layer_elems,
    n_blocks,
    n_items,
    ELEMS_PER_TOKEN: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_LAYERS_X2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NPROG: tl.constexpr,
):
    """Scatter strided blocks using at most NPROG CTAs."""
    per_rank_elems = per_layer_elems.to(tl.int64) * NUM_LAYERS_X2
    for item in range(tl.program_id(0), n_items, NPROG):
        prog_id = item // n_blocks
        block_id = item % n_blocks
        writer_id = prog_id // NUM_LAYERS_X2
        layer_kv_id = prog_id % NUM_LAYERS_X2
        layer_ptr = tl.load(layer_ptrs + layer_kv_id).to(staging.dtype)
        head_offset = tl.load(writer_head_offsets + writer_id)
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < per_layer_elems
        t_idx = offsets // ELEMS_PER_TOKEN
        e_idx = offsets % ELEMS_PER_TOKEN
        page_val = tl.load(page_indices + t_idx // PAGE_SIZE, mask=mask, other=0)
        pool_token = page_val * PAGE_SIZE + t_idx % PAGE_SIZE
        src = (
            writer_id.to(tl.int64) * per_rank_elems
            + layer_kv_id.to(tl.int64) * per_layer_elems.to(tl.int64)
            + offsets
        )
        vals = tl.load(staging + src, mask=mask)
        dst = (
            pool_token * stride_pool_token.to(tl.int64)
            + head_offset.to(tl.int64)
            + e_idx
        )
        tl.store(layer_ptr + dst, vals, mask=mask)


@triton.jit
def _scatter_staging_pages_kernel(
    layer_ptrs,
    page_indices,
    staging,
    writer_head_offsets,
    stride_pool_token,
    per_layer_elems,
    ELEMS_PER_TOKEN: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_LAYERS_X2: tl.constexpr,
):
    """Move one page for one writer and layer K/V per program."""
    page_id = tl.program_id(0)
    program_id = tl.program_id(1)
    writer_id = program_id // NUM_LAYERS_X2
    layer_kv_id = program_id % NUM_LAYERS_X2

    layer_ptr = tl.load(layer_ptrs + layer_kv_id).to(staging.dtype)
    head_offset = tl.load(writer_head_offsets + writer_id).to(tl.int64)
    page = tl.load(page_indices + page_id).to(tl.int64)
    token_offset = tl.arange(0, PAGE_SIZE)[:, None]
    element_offset = tl.arange(0, ELEMS_PER_TOKEN)[None, :]
    source = (
        (writer_id.to(tl.int64) * NUM_LAYERS_X2 + layer_kv_id)
        * per_layer_elems.to(tl.int64)
        + (page_id.to(tl.int64) * PAGE_SIZE + token_offset) * ELEMS_PER_TOKEN
        + element_offset
    )
    destination = (
        (page * PAGE_SIZE + token_offset) * stride_pool_token.to(tl.int64)
        + head_offset
        + element_offset
    )
    tl.store(layer_ptr + destination, tl.load(staging + source))


def _launch_scatter(
    layer_ptrs,
    page_idx,
    staging_typed,
    head_offsets,
    num_tokens,
    stride_pool_token,
    per_layer_elems,
    elems_per_token,
    page_size,
    num_layers_x2,
    num_writers,
    ctas,
):
    """Select the default, bounded-CTA, or per-page scatter kernel."""
    if (
        ctas < 0
        and page_size > 0
        and page_size & (page_size - 1) == 0
        and elems_per_token > 0
        and elems_per_token & (elems_per_token - 1) == 0
    ):
        _scatter_staging_pages_kernel[
            (num_tokens // page_size, num_writers * num_layers_x2)
        ](
            layer_ptrs,
            page_idx,
            staging_typed,
            head_offsets,
            stride_pool_token,
            per_layer_elems,
            ELEMS_PER_TOKEN=elems_per_token,
            PAGE_SIZE=page_size,
            NUM_LAYERS_X2=num_layers_x2,
            num_warps=8,
        )
        return
    if ctas <= 0:
        grid = (num_writers * num_layers_x2, triton.cdiv(per_layer_elems, 1024))
        _fused_scatter_from_staging_kernel[grid](
            layer_ptrs,
            page_idx,
            staging_typed,
            head_offsets,
            num_tokens,
            stride_pool_token,
            per_layer_elems,
            elems_per_token,
            page_size,
            num_layers_x2,
            1024,
        )
        return
    BLOCK = 16384
    n_blocks = triton.cdiv(per_layer_elems, BLOCK)
    n_items = num_writers * num_layers_x2 * n_blocks
    _scatter_from_staging_persistent_kernel[(min(ctas, n_items),)](
        layer_ptrs,
        page_idx,
        staging_typed,
        head_offsets,
        stride_pool_token,
        per_layer_elems,
        n_blocks,
        n_items,
        ELEMS_PER_TOKEN=elems_per_token,
        PAGE_SIZE=page_size,
        NUM_LAYERS_X2=num_layers_x2,
        BLOCK_SIZE=BLOCK,
        NPROG=min(ctas, n_items),
        num_warps=16,
    )


_SCATTER_I64_CACHE: dict[tuple, torch.Tensor] = {}


def _cached_scatter_i64(key: tuple, values: list[int], device) -> torch.Tensor:
    tensor = _SCATTER_I64_CACHE.get(key)
    if tensor is None:
        # Keep pageable pointer-table copies off the in-flight forward stream.
        with device_stream_context(create_device_stream(device)):
            tensor = make_ptr_table(values, device=device)
        _SCATTER_I64_CACHE[key] = tensor
    return tensor


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
        if isinstance(device, str):
            device = torch.device(device)

        # An explicit index on the device string wins over gpu_id, so a caller
        # passing "xpu:3" cannot end up allocating on a different device.
        if device.index is not None and device.index != gpu_id:
            logger.warning(
                f"Device index mismatch: device={device} (index={device.index}) "
                f"but gpu_id={gpu_id}. Using device.index={device.index}."
            )
            gpu_id = device.index
        elif device.index is None:
            device = torch.device(device.type, gpu_id)

        self.device = device
        self.gpu_id = gpu_id

        if device.type not in _STAGING_SUPPORTED_DEVICE_TYPES:
            raise RuntimeError(
                f"Staging buffer is not supported on device type '{device.type}' "
                f"(supported: {', '.join(_STAGING_SUPPORTED_DEVICE_TYPES)}). "
                f"Unset SGLANG_DISAGG_STAGING_BUFFER to disable it."
            )

        if device.type != "cuda" and custom_mem_pool is not None:
            logger.warning(
                f"Custom memory pools are a CUDA construct and are ignored on "
                f"{device.type}. Using the default allocator instead; memory may "
                f"not be RDMA-compatible."
            )

        self.device_module = torch.get_device_module(device)
        self.device_module.set_device(gpu_id)

        if device.type == "cuda" and custom_mem_pool is not None:
            with torch.cuda.use_mem_pool(custom_mem_pool):
                self.buffer = torch.empty(size_bytes, dtype=torch.uint8, device=device)
            alloc_method = "custom_mem_pool (cuMemCreate)"
        elif device.type == "cuda":
            self.buffer = torch.empty(size_bytes, dtype=torch.uint8, device=device)
            alloc_method = "cudaMalloc"
        else:
            # XPU has no equivalent of use_mem_pool, so its default caching
            # allocator backs the buffer.
            self.buffer = torch.empty(size_bytes, dtype=torch.uint8, device=device)
            alloc_method = "default allocator"

        self.data_ptr = self.buffer.data_ptr()

        # Created on first gather; the decode-side allocator's buffer never
        # gathers, so eager creation would leak one device stream per rank.
        self._gather_stream: Optional[torch.Stream] = None

        ptr_type_str = _device_ptr_type_str(self.data_ptr, device.type)
        logger.info(
            f"StagingBuffer allocated: {size_bytes / (1024 * 1024):.1f} MB "
            f"on {device}, method={alloc_method}, ptr=0x{self.data_ptr:x}{ptr_type_str}"
        )

    def producer_stream(self) -> torch.Stream:
        """The calling thread's current stream, for the gather to order against.

        Resolved per call: the current stream is thread-local and the transfer
        workers run on their own threads. This edge does not order the gather
        after the prefill forward that wrote the KV rows. Under overlap, the
        early-send path queues a chunk while that forward may still be running,
        and TransferKVChunk.wait_event, synchronized before the gather, provides
        the ordering; every other send runs after the forward has resolved.
        """
        return self.device_module.current_stream(self.device)

    def get_ptr(self) -> int:
        return self.data_ptr

    def get_size(self) -> int:
        return self.size_bytes

    def fits(self, required_bytes: int) -> bool:
        return required_bytes <= self.size_bytes

    def create_stream(self) -> torch.Stream:
        return create_device_stream(self.device)

    def get_gather_stream(self) -> torch.Stream:
        if self._gather_stream is None:
            self._gather_stream = self.create_stream()
        return self._gather_stream


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
    # Chunk already scattered; a late duplicate STAGING_REQ must not reallocate.
    CHUNK_SCATTERED = -3

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
        # Lazily created on the decode side by the first scatter; stays None
        # until then so release_room can drain it without a defensive check.
        self._scatter_stream = None

        ptr_type_str = _device_ptr_type_str(self.base_ptr, self.buffer.device.type)
        logger.info(
            f"StagingAllocator (ring+overcommit): "
            f"{total_size_bytes / (1024 * 1024):.1f} MB "
            f"on {device}, ptr=0x{self.base_ptr:x}{ptr_type_str}"
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
                # An empty ring makes the entire prior round reusable. Start a
                # fresh round at offset zero so the watermark cannot stay stale.
                self.round += 1
                self.head = 0
                self.watermark_round = self.round
                self.watermark_tail = 0
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
) -> int:
    """torch.gather path: zero per-layer allocation, one kernel per layer."""
    import numpy as np

    num_layers = len(k_buffers)
    head_dim = k_buffers[0].shape[-1]
    dtype_size = k_buffers[0].element_size()
    num_tokens = len(page_indices_np) * page_size
    per_layer_bytes = num_tokens * num_heads * head_dim * dtype_size

    device = k_buffers[0].device
    device_module = staging_buffer.device_module

    device_module.set_device(device.index)
    page_idx_tensor = torch.from_numpy(page_indices_np.astype(np.int64)).to(device)

    if page_size == 1:
        token_indices = page_idx_tensor
    else:
        offsets = torch.arange(page_size, device=device)
        token_indices = (page_idx_tensor.unsqueeze(1) * page_size + offsets).reshape(-1)

    gather_idx = token_indices.view(-1, 1, 1).expand(num_tokens, num_heads, head_dim)

    gather_stream = staging_buffer.get_gather_stream()
    gather_stream.wait_stream(staging_buffer.producer_stream())

    staging_view = staging_buffer.buffer
    offset = 0
    with device_stream_context(gather_stream):
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

    gather_stream.synchronize()
    return offset


def _gather_all_layers_triton(
    k_buffers: list,
    v_buffers: list,
    page_indices_np,
    staging_buffer: StagingBuffer,
    src_head_start: int,
    num_heads: int,
    page_size: int,
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

    device = k_buffers[0].device
    device_module = staging_buffer.device_module

    device_module.set_device(device.index)
    page_idx_tensor = torch.from_numpy(page_indices_np.astype(np.int64)).to(device)

    layer_ptrs = make_ptr_table(
        [buf.data_ptr() for buf in k_buffers] + [buf.data_ptr() for buf in v_buffers],
        device=device,
    )
    # Use integer dtype matching element size for bit-preserving copy
    int_dtype_map = {1: torch.int8, 2: torch.int16, 4: torch.int32}
    int_dtype = int_dtype_map.get(dtype_size, torch.int16)
    staging_typed = staging_buffer.buffer[:total_bytes].view(int_dtype)

    gather_stream = staging_buffer.get_gather_stream()
    gather_stream.wait_stream(staging_buffer.producer_stream())

    BLOCK_SIZE = 1024
    grid = (2 * num_layers, triton.cdiv(per_layer_elems, BLOCK_SIZE))

    with device_stream_context(gather_stream):
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

    gather_stream.synchronize()
    return total_bytes


def gather_all_layers_to_staging(
    k_buffers: list,
    v_buffers: list,
    page_indices_np,
    staging_buffer: StagingBuffer,
    src_head_start: int,
    num_heads: int,
    page_size: int,
) -> int:
    """Gather all layers' K and V head slices into a staging buffer.

    Returns total bytes written.
    Dispatches to Triton fused kernel when available, falls back to torch.gather.
    """
    _check_kv_on_staging_device(k_buffers, v_buffers, staging_buffer.device)
    buffers = k_buffers + v_buffers
    head_bytes = get_kv_bytes_per_head(buffers)
    if (
        page_size <= 0
        or num_heads <= 0
        or src_head_start < 0
        or src_head_start + num_heads > buffers[0].shape[1]
    ):
        raise ValueError("Invalid staging gather geometry")
    tokens = len(page_indices_np) * page_size
    offsets = compute_staging_entry_byte_offsets(
        [num_heads * b for b in head_bytes], tokens
    )
    if not staging_buffer.fits(offsets[-1]):
        raise ValueError("Staging gather exceeds allocation")
    if not tokens:
        return 0
    if len({(b.dtype, b.shape[1:]) for b in buffers}) == 1:
        gather = (
            _gather_all_layers_triton
            if _USE_TRITON_STAGING
            else _gather_all_layers_torch
        )
        return gather(
            k_buffers,
            v_buffers,
            page_indices_np,
            staging_buffer,
            src_head_start,
            num_heads,
            page_size,
        )
    # CUDA's current device is thread-local. Transfer workers must bind the KV
    # owner before StreamContext captures the stream it will restore on exit.
    staging_buffer.device_module.set_device(buffers[0].device.index)
    stream = staging_buffer.get_gather_stream()
    # Source KV completion is owned by the transfer worker, not the scheduler.
    stream.wait_stream(staging_buffer.producer_stream())
    with device_stream_context(stream):
        idx = torch.as_tensor(
            page_indices_np, device=buffers[0].device, dtype=torch.int64
        )
        if _USE_TRITON_STAGING:
            import triton

            ptrs = torch.tensor(
                [b.data_ptr() for b in buffers], device=idx.device, dtype=torch.int64
            )
            meta = torch.tensor(
                [
                    [
                        offsets[i],
                        num_heads * hb,
                        buf.stride(0) * buf.element_size(),
                        src_head_start * hb,
                        1,
                    ]
                    for i, (buf, hb) in enumerate(zip(buffers, head_bytes))
                ],
                device=idx.device,
                dtype=torch.int64,
            )
            _get_kv_rearrange_kernel()[
                (len(buffers), triton.cdiv(tokens * num_heads * max(head_bytes), 1024))
            ](
                ptrs,
                meta,
                idx,
                staging_buffer.buffer,
                None,
                len(buffers),
                page_size,
                1,
                tokens,
                False,
                False,
                1024,
            )
        else:
            rows = (
                idx[:, None] * page_size + torch.arange(page_size, device=idx.device)
            ).flatten()
            for i, buf in enumerate(buffers):
                # View bytes before indexing: Torch lacks float8 index kernels.
                value = buf.view(torch.uint8)[
                    rows, src_head_start : src_head_start + num_heads, :
                ]
                staging_buffer.buffer[offsets[i] : offsets[i + 1]].copy_(
                    value.reshape(-1)
                )
    # RDMA consumes the packed allocation on the host transfer worker.
    stream.synchronize()
    return offsets[-1]


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

    pointers = [buf.data_ptr() for buf in k_buffers] + [
        buf.data_ptr() for buf in v_buffers
    ]
    head_offsets = [
        compute_head_slice_params(
            prefill_attn_tp_size,
            decode_attn_tp_size,
            writer,
            dst_tp_rank,
            total_kv_heads,
        )[2]
        * head_dim
        for writer in range(num_writers)
    ]
    if envs.SGLANG_PLB_SCATTER_DEFER.get():
        device_key = (device.type, device.index)
        layer_ptrs = _cached_scatter_i64(
            ("pointers", device_key, *pointers), pointers, device
        )
        writer_head_offsets = _cached_scatter_i64(
            ("offsets", device_key, *head_offsets), head_offsets, device
        )
    else:
        layer_ptrs = make_ptr_table(pointers, device=device)
        writer_head_offsets = torch.tensor(
            head_offsets, dtype=torch.int64, device=device
        )

    int_dtype_map = {1: torch.int8, 2: torch.int16, 4: torch.int32}
    int_dtype = int_dtype_map.get(dtype_size, torch.int16)
    total_staging_bytes = (
        num_tokens * elems_per_token * dtype_size * num_layers * 2 * num_writers
    )
    staging_typed = staging_buffer_view[:total_staging_bytes].view(int_dtype)

    _launch_scatter(
        layer_ptrs,
        page_idx_tensor,
        staging_typed,
        writer_head_offsets,
        num_tokens,
        total_heads * head_dim,
        per_layer_elems,
        elems_per_token,
        page_size,
        2 * num_layers,
        num_writers,
        envs.SGLANG_PLB_SCATTER_CTAS.get(),
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
    cp_size: int = 1,
    writer_slot_mask: Optional[torch.Tensor] = None,
) -> None:
    """Scatter data from a contiguous staging region into KV cache buffers."""
    _check_kv_on_staging_device(k_buffers, v_buffers, staging_buffer_view.device)
    buffers = k_buffers + v_buffers
    head_bytes = get_kv_bytes_per_head(buffers)
    if (
        cp_size < 1
        or page_size % cp_size
        or (cp_size > 1 and prefill_attn_tp_size != 1)
    ):
        raise ValueError("Invalid CP staging scatter geometry")
    if (
        cp_size == 1
        and writer_slot_mask is None
        and prefill_attn_tp_size <= total_kv_heads
        and len({(b.dtype, b.shape[1:]) for b in buffers}) == 1
    ):
        scatter = (
            _scatter_staging_to_kv_triton
            if _USE_TRITON_STAGING
            else _scatter_staging_to_kv_torch
        )
        return scatter(
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
    local_page = page_size // cp_size
    tokens = page_idx_tensor.numel() * local_page
    writers = (
        cp_size if cp_size > 1 else max(1, prefill_attn_tp_size // decode_attn_tp_size)
    )
    if writer_slot_mask is not None and writer_slot_mask.shape != (
        writers,
        len(buffers),
    ):
        raise ValueError("Staging writer/slot completion mask has the wrong shape")
    if not tokens:
        return
    meta, base, max_width = [], 0, 0
    for writer in range(writers):
        # Packed writer IDs are local to this receiver. Convert to the
        # attention-TP rank before intersecting its global KV-head range.
        src_rank = (
            (dst_tp_rank % decode_attn_tp_size)
            * (prefill_attn_tp_size // decode_attn_tp_size)
            + writer
            if prefill_attn_tp_size > decode_attn_tp_size
            else (dst_tp_rank % decode_attn_tp_size)
            // (decode_attn_tp_size // prefill_attn_tp_size)
        )
        _, heads, start = compute_kv_head_overlap(
            prefill_attn_tp_size,
            decode_attn_tp_size,
            src_rank,
            dst_tp_rank,
            total_kv_heads,
        )
        if start + heads > buffers[0].shape[1]:
            raise ValueError("Staging head slice exceeds destination")
        offsets = compute_staging_entry_byte_offsets(
            [heads * hb for hb in head_bytes], tokens
        )
        # Replicated source GQA heads have multiple completion senders, but
        # only one may write a destination row (avoid GPU write/write races).
        replicas = max(1, prefill_attn_tp_size // total_kv_heads)
        enabled = cp_size > 1 or writer % replicas == 0
        for i, (buf, hb) in enumerate(zip(buffers, head_bytes)):
            meta.append(
                [
                    base + offsets[i],
                    heads * hb,
                    buf.stride(0) * buf.element_size(),
                    start * hb,
                    int(enabled),
                ]
            )
            max_width = max(max_width, heads * hb)
        base += offsets[-1]
    if base > staging_buffer_view.numel():
        raise ValueError("Staging scatter exceeds allocation")
    if _USE_TRITON_STAGING:
        import triton

        ptrs = torch.tensor(
            [b.data_ptr() for b in buffers],
            device=page_idx_tensor.device,
            dtype=torch.int64,
        )
        metadata = torch.tensor(meta, device=page_idx_tensor.device, dtype=torch.int64)
        _get_kv_rearrange_kernel()[(len(meta), triton.cdiv(tokens * max_width, 1024))](
            ptrs,
            metadata,
            page_idx_tensor,
            staging_buffer_view,
            writer_slot_mask,
            len(buffers),
            local_page,
            cp_size,
            tokens,
            True,
            writer_slot_mask is not None,
            1024,
        )
    else:
        for task, (offset, width, stride, head, enabled) in enumerate(meta):
            if not enabled:
                continue
            writer, slot = divmod(task, len(buffers))
            rows = (
                page_idx_tensor[:, None] * page_size
                + torch.arange(local_page, device=page_idx_tensor.device) * cp_size
                + (writer if cp_size > 1 else 0)
            ).flatten()
            dst = buffers[slot].view(torch.uint8).reshape(-1, stride)
            value = staging_buffer_view[offset : offset + tokens * width].reshape(
                tokens, width
            )
            # where keeps masked slots untouched without a device->host wait.
            if writer_slot_mask is not None:
                value = torch.where(
                    writer_slot_mask[writer, slot],
                    value,
                    dst[rows, head : head + width],
                )
            dst[rows, head : head + width] = value


def compute_staging_layout(
    src_attn_tp_size: int,
    dst_attn_tp_size: int,
    dst_tp_rank: int,
    total_kv_heads: int,
    num_tokens: int,
    bytes_per_head_token: int,
    num_layers: int,
    cp_size: int = 1,
    slot_head_bytes: Optional[List[int]] = None,
) -> Tuple[int, List[int], int]:
    """Compute per-writer byte layout for a staging region.

    Returns:
        (num_writers, writer_bytes_list, total_bytes)
        where writer_bytes_list[i] = bytes for writer i covering all layers (K+V).
    """
    byte_sum = (
        sum(slot_head_bytes)
        if slot_head_bytes is not None
        else bytes_per_head_token * num_layers * 2
    )
    if byte_sum <= 0 or (
        slot_head_bytes is not None
        and (
            len(slot_head_bytes) != num_layers * 2
            or any(n <= 0 for n in slot_head_bytes)
        )
    ):
        raise ValueError("Invalid staging slot byte metadata")
    if cp_size > 1:
        if src_attn_tp_size != 1 or num_tokens % cp_size:
            raise ValueError(
                "CP staging requires aligned tokens and prefill attention TP1"
            )
        num_heads = max(1, total_kv_heads // dst_attn_tp_size)
        per_writer = (num_tokens // cp_size) * num_heads * byte_sum
        return cp_size, [per_writer] * cp_size, per_writer * cp_size
    if src_attn_tp_size > dst_attn_tp_size:
        num_writers = src_attn_tp_size // max(1, dst_attn_tp_size)
    else:
        num_writers = 1

    # Every connected writer contributes the smaller local head shard. Rank
    # offsets affect placement, not the size of the packed staging allocation.
    num_heads = max(1, total_kv_heads // max(src_attn_tp_size, dst_attn_tp_size))
    writer_bytes = [num_tokens * num_heads * byte_sum] * num_writers
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


def staging_grid_tokens(chunked_prefill_size: Optional[int], page_size: int) -> int:
    """Token width of one staging grid slot; shared by prefetch and the
    sender's grid alignment."""
    cps = chunked_prefill_size or 8192
    return max(1, cps // page_size) * page_size


def compute_grid_segments(
    start_idx: int, end_idx: int, base: int, grid_tokens: int
) -> List[Tuple[int, int]]:
    """Split [start_idx, end_idx) at grid boundaries base + k * grid_tokens
    so each segment maps to exactly one staging slot. An empty range yields
    one empty segment (a metadata-only last chunk still needs a send).
    """
    segments: List[Tuple[int, int]] = []
    seg_start = start_idx
    while seg_start < end_idx:
        next_boundary = base + ((seg_start - base) // grid_tokens + 1) * grid_tokens
        seg_end = min(next_boundary, end_idx)
        segments.append((seg_start, seg_end))
        seg_start = seg_end
    if not segments:
        segments = [(start_idx, end_idx)]
    return segments


def build_staging_kv_entry_ids(num_layers, num_target_layers, start_layer=0):
    """Identify [all K, all V] buffer entries independently of pointer order.

    Target indices use the full-attention layer ordering across PP stages.
    Draft indices live in a separate namespace and never alias target layers.
    """
    if not 0 < num_target_layers <= num_layers or start_layer < 0:
        raise ValueError("Invalid target/draft staging layer counts")
    layers = [("target", start_layer + i) for i in range(num_target_layers)]
    layers += [("draft", i) for i in range(num_layers - num_target_layers)]
    return [(model, kind, layer) for kind in ("k", "v") for model, layer in layers]


def validate_staging_kv_entries(entry_ids, entry_page_bytes=None):
    """Validate a version-1 staging layout before using any remote offsets."""
    if not entry_ids or len(entry_ids) % 2:
        raise ValueError("Staging layout requires paired K/V slots")
    ids = []
    for entry in entry_ids:
        if (
            not isinstance(entry, (list, tuple))
            or len(entry) != 3
            or entry[0] not in ("target", "draft")
            or entry[1] not in ("k", "v")
            or type(entry[2]) is not int
            or entry[2] < 0
        ):
            raise ValueError("Invalid staging entry identity")
        ids.append(tuple(entry))
    half = len(ids) // 2
    if (
        len(set(ids)) != len(ids)
        or any(kind != "k" for _, kind, _ in ids[:half])
        or ids[half:] != [(model, "v", layer) for model, _, layer in ids[:half]]
    ):
        raise ValueError("Staging slots must be unique and ordered [all K, all V]")
    if entry_page_bytes is not None and (
        len(entry_page_bytes) != len(ids)
        or any(type(n) is not int or n <= 0 for n in entry_page_bytes)
    ):
        raise ValueError("Invalid per-entry staging byte strides")
    return ids


def map_src_to_dst_kv_entries(source_ids, destination_ids):
    """Return (src_entry_index, dst_entry_index) pairs by model, K/V and layer.

    These are buffer entries, not token slots. Receiver-only NEXTN entries
    remain untouched.
    """
    source_ids = validate_staging_kv_entries(source_ids)
    destination_ids = validate_staging_kv_entries(destination_ids)
    destination = {slot: i for i, slot in enumerate(destination_ids)}
    missing = [slot for slot in source_ids if slot not in destination]
    if missing:
        raise ValueError(f"Decode staging layout is missing source slots: {missing}")
    return [(i, destination[slot]) for i, slot in enumerate(source_ids)]


def compute_staging_entry_byte_offsets(entry_token_bytes, num_tokens):
    """Prefix byte offsets of packed K/V entries, including the final end offset."""
    if num_tokens < 0 or any(n <= 0 for n in entry_token_bytes):
        raise ValueError("Invalid KV slot byte geometry")
    return list(accumulate((num_tokens * n for n in entry_token_bytes), initial=0))


def get_kv_bytes_per_head(buffers):
    """Return bytes per head for each [token, head, dim] KV buffer."""
    if not buffers:
        raise ValueError("Empty KV staging buffer list")
    for buf in buffers:
        if buf.ndim != 3 or not buf.is_contiguous():
            raise ValueError("Staging requires contiguous [token, head, dim] buffers")
    if len({buf.shape[1] for buf in buffers}) != 1:
        raise ValueError("KV entries must share the attention head partition")
    return [buf.shape[2] * buf.element_size() for buf in buffers]


@cache
def _get_kv_rearrange_kernel():
    import triton
    import triton.language as tl

    @triton.jit(
        do_not_specialize=["TOKENS"],
        do_not_specialize_on_alignment=["pages"],
    )
    def _rearrange_kv_cache_kernel(
        ptrs,
        metadata,
        pages,
        packed,
        writer_mask,
        NUM_SLOTS: tl.constexpr,
        PAGE: tl.constexpr,
        CP: tl.constexpr,
        TOKENS,
        SCATTER: tl.constexpr,
        HAS_MASK: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        task = tl.program_id(0)
        slot = task % NUM_SLOTS
        writer = task // NUM_SLOTS
        ptr = tl.load(ptrs + slot).to(tl.pointer_type(tl.uint8))
        # packed offset, token_slot bytes, pool stride, head offset, writer enabled
        base = tl.load(metadata + task * 5)
        width = tl.load(metadata + task * 5 + 1)
        stride = tl.load(metadata + task * 5 + 2)
        head = tl.load(metadata + task * 5 + 3)
        enabled = tl.load(metadata + task * 5 + 4) != 0
        x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = (x < TOKENS * width) & enabled
        if HAS_MASK:
            mask = mask & tl.load(writer_mask + task)
        token, byte = x // width, x % width
        page = tl.load(pages + token // PAGE, mask=mask, other=0).to(tl.int64)
        token_slot = page * PAGE + token % PAGE
        if SCATTER:
            token_slot = page * PAGE * CP + (token % PAGE) * CP
            if CP > 1:
                token_slot += writer
        address = token_slot * stride + head + byte
        if SCATTER:
            value = tl.load(packed + base + x, mask=mask, other=0)
            tl.store(ptr + address, value, mask=mask)
        else:
            value = tl.load(ptr + address, mask=mask, other=0)
            tl.store(packed + base + x, value, mask=mask)

    return _rearrange_kv_cache_kernel


@cache
def warmup_staging_kernels(*, device, num_slots, page_size, scatter):
    """Compile and load legal source CP page geometries before serving requests.

    Byte widths, head offsets and strides are already read from metadata. Only
    slot count, page geometry and direction specialize the kernel. Decode may
    receive any source CP divisor of its logical page size, so cover each one.
    Mock tensor arguments compile without allocating or reading live KV state.
    """
    started = time.monotonic()
    variants = 0
    cp_sizes = (
        [cp for cp in range(1, page_size + 1) if page_size % cp == 0]
        if scatter
        else [1]
    )
    with torch.cuda.device(device):
        for cp_size in cp_sizes:
            for index_dtype in (torch.int32, torch.int64):
                for has_mask in (False, True) if scatter else (False,):
                    compiled = _get_kv_rearrange_kernel().warmup(
                        torch.int64,
                        torch.int64,
                        index_dtype,
                        torch.uint8,
                        torch.bool if has_mask else None,
                        num_slots,
                        page_size // cp_size,
                        cp_size,
                        1,
                        scatter,
                        has_mask,
                        1024,
                        grid=(num_slots * cp_size, 1),
                    )
                    # Triton warmup leaves module and launcher handles lazy.
                    # Accessing run loads both without executing the kernel.
                    _ = compiled.run
                    variants += 1
    logger.info(
        "KV bulk kernel warmup complete: device=%s direction=%s slots=%d "
        "logical_page=%d variants=%d dynamic_token_lengths=True elapsed=%.3fs",
        device,
        "scatter" if scatter else "gather",
        num_slots,
        page_size,
        variants,
        time.monotonic() - started,
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
        # GQA replication: consecutive decode ranks share a KV head
        # (tp_rank // num_kv_head_replicas), so map by integer division not modulo.
        dst_replication = max(1, dst_attn_tp_size // total_kv_heads)
        unique_dst_head_idx = dst_tp_rank_in_group // dst_replication
        src_head_start = (unique_dst_head_idx * dst_heads_per_rank) % src_heads_per_rank
        num_heads_to_send = dst_heads_per_rank
        dst_head_start = 0

    return src_head_start, num_heads_to_send, dst_head_start, num_heads_to_send
