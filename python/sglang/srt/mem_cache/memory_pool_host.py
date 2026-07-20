from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

import numpy as np
import psutil
import torch

from sglang.jit_kernel.hicache import (
    can_use_write_back_jit_kernel,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_all_layer_mla_staged_lf_pf as jit_transfer_hicache_all_layer_mla_staged_lf_pf,
)
from sglang.jit_kernel.hisparse import transfer_cache_dsv4_mla
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, MambaPool
from sglang.srt.utils import is_cuda, is_hip, is_mps, is_npu, is_xpu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_xpu = is_xpu()
_is_mps = is_mps()
if _is_cuda or _is_hip:
    from sgl_kernel.kvcacheio import (
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_all_layer_mla,
        transfer_kv_all_layer_mla_lf_pf,
        transfer_kv_direct,
        transfer_kv_per_layer_direct_pf_lf,
        transfer_kv_per_layer_mla,
        transfer_kv_per_layer_mla_pf_lf,
    )
if _is_cuda:
    from sglang.jit_kernel.transfer_mamba import (
        transfer_kv_mamba_lf_pf,
        transfer_kv_mamba_pf_lf,
    )
if _is_npu:
    pass

logger = logging.getLogger(__name__)


from sglang.srt.mem_cache.pool_host import HostKVCache
from sglang.srt.mem_cache.pool_host.base import (
    _WRITE_BACK_STAGING_PAGE_CHUNK,
    HICACHE_HOST_MEMORY_RESERVE_BYTES,
    sync_fixed_hicache_size,
    synchronized,
)
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    get_allocator_from_storage,
)
from sglang.srt.mem_cache.pool_host.hisparse import HiSparseHostPoolMixin


class MambaPoolHost(HostKVCache):

    def __init__(
        self,
        device_pool: MambaPool,
        host_to_device_ratio: float,
        host_size: int,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        layout: str = "layer_first",
    ):
        self.device_pool = device_pool
        self.page_size = 1

        assert layout in [
            "page_first",
            "page_first_direct",
        ], f"Unsupported layout: {layout}"

        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)
        self.num_mamba_layers = device_pool.num_mamba_layers

        self.conv_state_shapes = [
            conv_state.shape[2:] for conv_state in device_pool.mamba_cache.conv
        ]
        self.temporal_state_shape = device_pool.mamba_cache.temporal.shape[2:]
        self.temporal_state_elem_size = int(np.prod(self.temporal_state_shape))
        self.conv_state_elem_sizes = [
            int(np.prod(conv_shape)) for conv_shape in self.conv_state_shapes
        ]
        self.conv_dtype = device_pool.mamba_cache.conv[0].dtype
        self.temporal_dtype = device_pool.mamba_cache.temporal.dtype
        self.dtype = self.conv_dtype
        self.size_per_token = self.get_size_per_token()

        if host_size > 0:
            self.size = sync_fixed_hicache_size(
                int(host_size * 1e9 // self.size_per_token), host_size
            )
        else:
            self.size = int(device_pool.size * host_to_device_ratio)

        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size

        if self.size <= device_pool.size:
            logger.warning(
                "HiCache host KV pool (%d tokens) is smaller than the device pool (%d tokens);"
                "L2 cache effectiveness is reduced."
                "Consider increasing --hicache-ratio (or --hicache-size) for higher L2 cache hit rate.",
                self.size,
                device_pool.size,
            )

        host_mem = psutil.virtual_memory()
        requested_bytes = self.size * self.size_per_token
        available_bytes = host_mem.available - HICACHE_HOST_MEMORY_RESERVE_BYTES
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        logger.info(
            "Allocating %.2f GB host memory for hierarchical Mamba cache (layout=%s).",
            requested_bytes / 1e9,
            self.layout,
        )

        self.temporal_device_ptrs = torch.tensor(
            [
                device_pool.mamba_cache.temporal[i].data_ptr()
                for i in range(self.num_mamba_layers)
            ],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self.conv_device_ptrs = [
            torch.tensor(
                [conv_state[i].data_ptr() for i in range(self.num_mamba_layers)],
                dtype=torch.uint64,
                device=self.device_pool.device,
            )
            for conv_state in device_pool.mamba_cache.conv
        ]

        self.init_kv_buffer()
        self._init_write_back_staging_buffers()
        self.lock = threading.RLock()
        self.clear()

    def init_kv_buffer(self):
        _host_alloc = ALLOC_MEMORY_FUNCS[self.device_pool.device]

        def alloc_func(dims, *, dtype, device, pin_memory, allocator):
            # conv-only linear attention has no ssm state: mmap can't map the
            # 0-element temporal buffer, so hand back a plain empty tensor.
            if np.prod(dims) == 0:
                return torch.empty(dims, dtype=dtype, device=device)
            return _host_alloc(
                dims,
                dtype=dtype,
                device=device,
                pin_memory=pin_memory,
                allocator=allocator,
            )

        if self.layout in ["page_first", "page_first_direct"]:
            # page-first: (page_num, num_layers, 1, *shape) — per-page data is contiguous
            temporal_dims = (
                self.size,
                self.num_mamba_layers,
                1,
            ) + self.temporal_state_shape
            self.temporal_buffer = alloc_func(
                temporal_dims,
                dtype=self.temporal_dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.conv_buffer = []
            for conv_shape in self.conv_state_shapes:
                conv_dims = (self.size, self.num_mamba_layers, 1) + conv_shape
                self.conv_buffer.append(
                    alloc_func(
                        conv_dims,
                        dtype=self.conv_dtype,
                        device=self.device,
                        pin_memory=self.pin_memory,
                        allocator=self.allocator,
                    )
                )
        else:
            # layer-first: (num_layers, size, *shape)
            temporal_dims = (
                self.num_mamba_layers,
                self.size,
            ) + self.temporal_state_shape
            self.temporal_buffer = alloc_func(
                temporal_dims,
                dtype=self.temporal_dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.conv_buffer = []
            for conv_shape in self.conv_state_shapes:
                conv_dims = (self.num_mamba_layers, self.size) + conv_shape
                self.conv_buffer.append(
                    alloc_func(
                        conv_dims,
                        dtype=self.conv_dtype,
                        device=self.device,
                        pin_memory=self.pin_memory,
                        allocator=self.allocator,
                    )
                )

    def _init_write_back_staging_buffers(self):
        self.temporal_staging_buffer = None
        self.conv_staging_buffers = [None] * len(self.conv_buffer)
        # Must be True: HostPoolGroup computes can_use_write_back_jit as AND of
        # all pools. When True, start_writing() keeps indices on CPU, which MLA's
        # staged write-back kernel requires. MambaPoolHost's own backup path does
        # not check this flag — it routes by layout + io_backend instead.
        self.can_use_write_back_jit = True
        self._temporal_can_use_jit = False
        self._conv_can_use_jit = [False] * len(self.conv_buffer)

    def get_hybrid_pool_buffer(self):
        # Expose all mamba host tensors that need Mooncake buffer registration.
        return [self.temporal_buffer, *self.conv_buffer]

    def _iter_page_tensors(self, index: int):
        if self.layout in ["page_first", "page_first_direct"]:
            yield self.temporal_buffer[index]
            for conv_buf in self.conv_buffer:
                yield conv_buf[index]
        else:
            yield self.temporal_buffer[:, index : index + self.page_size]
            for conv_buf in self.conv_buffer:
                yield conv_buf[:, index : index + self.page_size]

    @staticmethod
    def _flatten_tensor_bytes(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.contiguous().view(torch.uint8).reshape(-1)

    @synchronized
    def clear(self):
        self.mem_state = torch.zeros(
            (self.size,), dtype=torch.uint8, device=self.device
        )
        self.free_slots = torch.arange(self.size, dtype=torch.int64)
        self.release_slots = []
        self.num_release_slots = 0

    def available_size(self):
        return len(self.free_slots) + self.num_release_slots

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        assert (
            need_size % self.page_size == 0
        ), "The requested size should be a multiple of the page size."
        if need_size > self.available_size():
            return None

        if need_size > len(self.free_slots):
            self._merge_release_slots()

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        indices_cpu = indices.cpu()
        if indices_cpu.numel() == 0:
            return 0

        self.release_slots.append(indices_cpu)
        self.num_release_slots += len(indices_cpu)
        return len(indices)

    def get_size_per_token(self):
        conv_total_size = sum(
            conv_elem_size * self.conv_dtype.itemsize
            for conv_elem_size in self.conv_state_elem_sizes
        )
        temporal_size = self.temporal_state_elem_size * self.temporal_dtype.itemsize
        return (conv_total_size + temporal_size) * self.num_mamba_layers

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    @staticmethod
    def _item_size_per_index(tensor: torch.Tensor) -> int:
        if tensor.shape[0] == 0:
            return 0
        return int(tensor[0].numel() * tensor.element_size())

    @staticmethod
    def _copy_tensor(
        src: torch.Tensor,
        dst: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        io_backend: str,
    ) -> None:
        if src_indices.numel() == 0:
            return
        if io_backend == "kernel":
            # TODO: Rename the interface for clarity.
            # Here, transfer_kv_per_layer_mla is reused to transfer the Mamba state.
            # This has nothing to do with MLA; it's only reused because this interface happens to transfer a single Pool.
            transfer_kv_per_layer_mla(
                src=src,
                dst=dst,
                src_indices=src_indices,
                dst_indices=dst_indices,
                item_size=MambaPoolHost._item_size_per_index(src),
            )
        elif io_backend == "direct":
            transfer_kv_direct(
                src_layers=[src],
                dst_layers=[dst],
                src_indices=src_indices,
                dst_indices=dst_indices,
                page_size=1,
            )
        else:
            raise ValueError(f"Unsupported io_backend: {io_backend}")

    @staticmethod
    def _copy_tensor_pf_lf(
        src: torch.Tensor,
        dst: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        layer_id: int,
        num_layers: int,
        io_backend: str,
    ) -> None:
        if src_indices.numel() == 0:
            return
        if io_backend == "kernel":
            item_size = MambaPoolHost._item_size_per_index(dst)
            # Mamba JIT kernel expects all index tensors on CUDA.
            # host_indices may be on CPU (kept there by start_writing when
            # can_use_write_back_jit is True on the HostPoolGroup).
            if src_indices.device.type != "cuda":
                src_indices = src_indices.to(dst_indices.device, non_blocking=True)
            transfer_kv_mamba_pf_lf(
                src=src,
                dst=dst,
                src_indices=src_indices,
                dst_indices=dst_indices,
                layer_id=layer_id,
                item_size=item_size,
                src_layout_dim=item_size * num_layers,
            )
        elif io_backend == "direct":
            transfer_kv_per_layer_direct_pf_lf(
                src_ptrs=[src],
                dst_ptrs=[dst],
                src_indices=src_indices,
                dst_indices=dst_indices,
                layer_id=layer_id,
                page_size=1,
            )
        else:
            raise ValueError(f"Unsupported io_backend: {io_backend}")

    @staticmethod
    def _copy_tensor_all_layers_lf_pf(
        src_layers: torch.Tensor,
        dst: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        num_layers: int,
        io_backend: str,
        src_ptrs: torch.Tensor,
        staging: Optional[torch.Tensor] = None,
        can_use_jit: bool = False,
    ) -> None:
        if src_indices.numel() == 0:
            return
        if io_backend == "kernel":
            item_size = MambaPoolHost._item_size_per_index(src_layers[0])
            # Mamba JIT kernel expects all index tensors on CUDA.
            # When can_use_write_back_jit is True on the HostPoolGroup,
            # start_writing() keeps host_indices on CPU (for MLA staged kernel).
            # Move dst_indices to CUDA here to satisfy the kernel's requirement.
            if dst_indices.device.type != "cuda":
                dst_indices = dst_indices.to(src_indices.device, non_blocking=True)
            transfer_kv_mamba_lf_pf(
                src_ptrs=src_ptrs,
                dst=dst,
                src_indices=src_indices,
                dst_indices=dst_indices,
                item_size=item_size,
                dst_layout_dim=item_size * num_layers,
                num_layers=num_layers,
            )
        elif io_backend == "direct":
            src_ptrs = [src_layers[i] for i in range(num_layers)]
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=src_ptrs,
                dst_ptrs=[dst],
                src_indices=src_indices,
                dst_indices=dst_indices,
                page_size=1,
            )
        else:
            raise ValueError(f"Unsupported io_backend: {io_backend}")

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend="kernel",
    ):
        if self.layout in ["page_first", "page_first_direct"]:
            # no ssm state on conv-only models: nothing to transfer
            if self.temporal_state_elem_size > 0:
                self._copy_tensor_pf_lf(
                    src=self.temporal_buffer,
                    dst=device_pool.mamba_cache.temporal[layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    num_layers=self.num_mamba_layers,
                    io_backend=io_backend,
                )
            for conv_idx in range(len(self.conv_state_shapes)):
                self._copy_tensor_pf_lf(
                    src=self.conv_buffer[conv_idx],
                    dst=device_pool.mamba_cache.conv[conv_idx][layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    num_layers=self.num_mamba_layers,
                    io_backend=io_backend,
                )
        else:
            self._copy_tensor(
                self.temporal_buffer[layer_id],
                device_pool.mamba_cache.temporal[layer_id],
                host_indices,
                device_indices,
                io_backend,
            )
            for conv_idx in range(len(self.conv_state_shapes)):
                self._copy_tensor(
                    self.conv_buffer[conv_idx][layer_id],
                    device_pool.mamba_cache.conv[conv_idx][layer_id],
                    host_indices,
                    device_indices,
                    io_backend,
                )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend="kernel"
    ):
        if self.layout in ["page_first", "page_first_direct"]:
            # no ssm state on conv-only models: a 0-size batched memcpy errors
            if self.temporal_state_elem_size > 0:
                self._copy_tensor_all_layers_lf_pf(
                    src_layers=device_pool.mamba_cache.temporal,
                    dst=self.temporal_buffer,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    num_layers=self.num_mamba_layers,
                    io_backend=io_backend,
                    staging=self.temporal_staging_buffer,
                    can_use_jit=self._temporal_can_use_jit,
                    src_ptrs=self.temporal_device_ptrs,
                )
            for conv_idx in range(len(self.conv_state_shapes)):
                self._copy_tensor_all_layers_lf_pf(
                    src_layers=device_pool.mamba_cache.conv[conv_idx],
                    dst=self.conv_buffer[conv_idx],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    num_layers=self.num_mamba_layers,
                    io_backend=io_backend,
                    staging=self.conv_staging_buffers[conv_idx],
                    can_use_jit=self._conv_can_use_jit[conv_idx],
                    src_ptrs=self.conv_device_ptrs[conv_idx],
                )
        else:
            for layer_id in range(self.num_mamba_layers):
                self._copy_tensor(
                    device_pool.mamba_cache.temporal[layer_id],
                    self.temporal_buffer[layer_id],
                    device_indices,
                    host_indices,
                    io_backend,
                )
                for conv_idx in range(len(self.conv_state_shapes)):
                    self._copy_tensor(
                        device_pool.mamba_cache.conv[conv_idx][layer_id],
                        self.conv_buffer[conv_idx][layer_id],
                        device_indices,
                        host_indices,
                        io_backend,
                    )

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        data_page = torch.cat(
            [
                self._flatten_tensor_bytes(tensor)
                for tensor in self._iter_page_tensors(index)
            ]
        )
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            self.page_size * self.size_per_token,
            dtype=torch.uint8,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def set_from_flat_data_page(
        self,
        index: int,
        data_page: torch.Tensor,
    ) -> None:
        flat_bytes = data_page.contiguous().view(torch.uint8).reshape(-1)
        start = 0
        for tensor in self._iter_page_tensors(index):
            num_bytes = tensor.numel() * tensor.element_size()
            tensor_bytes = flat_bytes[start : start + num_bytes]
            start += num_bytes
            restored = tensor_bytes.view(dtype=tensor.dtype).reshape(tensor.shape)
            tensor.copy_(restored)

    def get_page_buffer_meta(self, indices):
        """Meta data for zero-copy storage I/O.

        Only page-first layouts are supported for mamba storage zero-copy because
        each page slot in temporal/conv buffers is directly addressable.
        """
        assert len(indices) % self.page_size == 0
        if self.layout not in ["page_first", "page_first_direct"]:
            raise ValueError(
                f"Mamba storage zero-copy requires page_first layout, got {self.layout}"
            )
        indices = indices.tolist()
        ptr_list = []
        element_size_list = []

        # Compute base pointers once; each page pointer is offset from these bases.
        temporal_base_ptr = self.temporal_buffer.data_ptr()
        conv_base_ptrs = [buf.data_ptr() for buf in self.conv_buffer]
        # Component sizes are constant across pages, so precompute once as well.
        temporal_element_size = (
            self.page_size
            * self.num_mamba_layers
            * self.temporal_dtype.itemsize
            * self.temporal_state_elem_size
        )
        conv_element_sizes = [
            (
                self.page_size
                * self.num_mamba_layers
                * self.conv_dtype.itemsize
                * self.conv_state_elem_sizes[i]
            )
            for i in range(len(self.conv_state_shapes))
        ]

        for i in range(0, len(indices), self.page_size):
            # Emit component pointers in stable order: temporal first (dropped
            # for conv-only models with no ssm state), then conv_0..conv_n.
            # _get_hybrid_page_component_keys drops the temporal key under the
            # same condition, keeping keys and buffers aligned.
            if self.temporal_state_elem_size > 0:
                temporal_ptr = (
                    temporal_base_ptr
                    + indices[i]
                    * self.num_mamba_layers
                    * self.temporal_state_elem_size
                    * self.temporal_dtype.itemsize
                )
                ptr_list.append(temporal_ptr)
                element_size_list.append(temporal_element_size)
            for j in range(len(self.conv_buffer)):
                conv_ptr = (
                    conv_base_ptrs[j]
                    + indices[i]
                    * self.num_mamba_layers
                    * self.conv_state_elem_sizes[j]
                    * self.conv_dtype.itemsize
                )
                ptr_list.append(conv_ptr)
                element_size_list.append(conv_element_sizes[j])
        return ptr_list, element_size_list

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        if self.layout not in ["page_first", "page_first_direct"]:
            return False
        temporal_stride = (
            self.num_mamba_layers
            * self.temporal_state_elem_size
            * self.temporal_dtype.itemsize
        )
        if self.temporal_buffer.data_ptr() % page_size_bytes != 0:
            return False
        if temporal_stride % page_size_bytes != 0:
            return False
        for buf, elem_size in zip(self.conv_buffer, self.conv_state_elem_sizes):
            conv_stride = self.num_mamba_layers * elem_size * self.conv_dtype.itemsize
            if buf.data_ptr() % page_size_bytes != 0:
                return False
            if conv_stride % page_size_bytes != 0:
                return False
        return True


# ---- V4 Compressed KV Host Pools ----


class LogicalHostPool:
    """Pure-logical anchor pool for V4 HiCache.

    The pool manages page-aligned token slots but holds no KV tensor. V4
    compressed side pools use these logical FULL indices as stable page anchors.
    """

    def __init__(self, size: int, page_size: int, layout: str = "layer_first"):
        if size % page_size != 0:
            raise ValueError(
                "LogicalHostPool size must be page-aligned, "
                f"got size={size}, page_size={page_size}"
            )
        self.size = size
        self.page_size = page_size
        self.device = "cpu"
        self.layout = layout
        self.dtype = torch.uint8
        self.layer_num = 0
        self.start_layer = 0
        self.end_layer = 0
        self.kv_buffer = None
        self.size_per_token = 0
        self.allocator = None
        self.can_use_write_back_jit = True
        self.lock = threading.RLock()
        self.clear()

    @synchronized
    def clear(self):
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size % self.page_size != 0:
            raise ValueError(
                "LogicalHostPool allocation must be page-aligned, "
                f"got need_size={need_size}, page_size={self.page_size}"
            )
        if need_size > self.available_size():
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        if len(indices) % self.page_size != 0:
            raise ValueError(
                "LogicalHostPool free must be page-aligned, "
                f"got len(indices)={len(indices)}, page_size={self.page_size}"
            )
        self.free_slots = torch.cat(
            [self.free_slots, indices.to(dtype=torch.int64, device="cpu").flatten()]
        )
        return len(indices)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        pass

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        pass

    def get_data_page(self, index, flat=True):
        return torch.empty(0, dtype=torch.uint8)

    def get_dummy_flat_data_page(self):
        return torch.empty(0, dtype=torch.uint8)

    def set_from_flat_data_page(self, index, data_page):
        pass

    def get_page_buffer_meta(self, indices):
        return None

    def get_ksize_per_token(self):
        return 0


class DeepSeekV4PagedHostPool(HiSparseHostPoolMixin, HostKVCache):
    """Host mirror for a DeepSeek V4 paged KV/indexer sub-pool."""

    def __init__(
        self,
        pool_name: str,
        device_buffers: list[torch.Tensor],
        item_bytes: int,
        num_host_pages: int,
        slot_page_size: int,
        layout: str = "layer_first",
        device: str = "cpu",
        pin_memory: bool = True,
        allocator_type: str = "default",
    ):
        self.pool_name = pool_name
        self.layer_num = len(device_buffers)
        self.item_bytes = item_bytes
        self.num_host_pages = num_host_pages
        self.slot_page_size = slot_page_size
        self.dtype = torch.uint8
        self.device = device
        self.pin_memory = pin_memory
        self.allocator = get_allocator_from_storage(allocator_type)
        self.page_size = slot_page_size
        self.size = num_host_pages * slot_page_size
        self.layout = layout
        self.size_per_token = item_bytes
        self.start_layer = 0
        self.end_layer = self.layer_num
        self.lock = threading.RLock()

        self.device_buffers = device_buffers
        self.gpu_device = device_buffers[0].device if device_buffers else device

        requested_bytes = self.layer_num * num_host_pages * self.item_bytes
        host_mem = psutil.virtual_memory()
        available_bytes = host_mem.available - HICACHE_HOST_MEMORY_RESERVE_BYTES
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory for V4 paged pool {pool_name}. "
                f"Requesting {requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free."
            )

        alloc_func = ALLOC_MEMORY_FUNCS[self.gpu_device]
        self.data_refs = []
        if self.layout == "layer_first":
            self.kv_buffer = [
                alloc_func(
                    (num_host_pages, self.item_bytes),
                    dtype=self.dtype,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    allocator=self.allocator,
                )
                for _ in range(self.layer_num)
            ]
            self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        elif self.layout == "page_first":
            self.kv_buffer = alloc_func(
                (num_host_pages, self.layer_num, self.item_bytes),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
        elif self.layout == "page_first_direct":
            self.kv_buffer = alloc_func(
                (num_host_pages, self.layer_num, 1, self.item_bytes),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

        logger.info(
            "Allocating %.2f GB host memory for V4 paged pool '%s' "
            "(layers=%d, pages=%d, item_bytes=%d, layout=%s).",
            requested_bytes / 1e9,
            self.pool_name,
            self.layer_num,
            num_host_pages,
            self.item_bytes,
            self.layout,
        )

        self.device_ptrs = torch.tensor(
            [x.data_ptr() for x in self.device_buffers],
            dtype=torch.uint64,
            device=self.gpu_device,
        )
        self.data_ptrs = (
            torch.tensor(
                [x.data_ptr() for x in self.data_refs],
                dtype=torch.uint64,
                device=self.gpu_device,
            )
            if self.data_refs
            else None
        )
        self.can_use_jit = False
        self.can_use_write_back_jit = False
        self._init_write_back_staging_buffers()
        self.clear()

    def _init_write_back_staging_buffers(self):
        self.staging_buffer = None
        if self.layout != "page_first" or (_is_npu or _is_xpu or _is_mps):
            return

        self.can_use_write_back_jit = _is_cuda and can_use_write_back_jit_kernel(
            element_size=self.item_bytes * self.dtype.itemsize,
        )
        staging_page_capacity = min(self.num_host_pages, _WRITE_BACK_STAGING_PAGE_CHUNK)
        self.staging_buffer = torch.empty(
            (staging_page_capacity, self.layer_num, self.item_bytes),
            dtype=self.dtype,
            device=self.gpu_device,
        )

    def get_contiguous_buf_infos(self):
        """Return per-layer page-row buffers for PD direct-to-host transfer."""
        data_ptrs = [int(self.data_ptrs[i].item()) for i in range(self.layer_num)]
        data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        item_lens = [self.item_bytes * self.dtype.itemsize] * self.layer_num
        return data_ptrs, data_lens, item_lens

    def _to_page_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return indices.reshape(-1, self.slot_page_size)[:, 0] // self.slot_page_size

    def _has_transfer_indices(
        self, host_indices: torch.Tensor | None, device_indices: torch.Tensor | None
    ) -> bool:
        if host_indices is None or device_indices is None:
            return False
        if host_indices.numel() != device_indices.numel():
            raise ValueError(
                f"{self.pool_name} transfer index size mismatch: "
                f"host={host_indices.numel()}, device={device_indices.numel()}"
            )
        return host_indices.numel() > 0

    def get_size_per_token(self):
        return self.item_bytes

    def get_ksize_per_token(self):
        return self.item_bytes

    def init_kv_buffer(self):
        return self.kv_buffer

    def get_hybrid_pool_buffer(self):
        return self.kv_buffer if isinstance(self.kv_buffer, list) else [self.kv_buffer]

    def clear(self):
        self.free_slots = torch.arange(self.size, dtype=torch.int64)
        self.release_slots = []
        self.num_release_slots = 0

    def available_size(self):
        return len(self.free_slots) + self.num_release_slots

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        need_size = (
            (need_size + self.slot_page_size - 1) // self.slot_page_size
        ) * self.slot_page_size
        if need_size > self.available_size():
            return None

        if need_size > len(self.free_slots):
            self._merge_release_slots()

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        indices_cpu = indices.cpu()
        if indices_cpu.numel() == 0:
            return 0

        self.release_slots.append(indices_cpu)
        self.num_release_slots += len(indices_cpu)
        return len(indices)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if not self._has_transfer_indices(host_indices, device_indices):
            return
        if (
            host_indices.numel() % self.slot_page_size != 0
            or device_indices.numel() % self.slot_page_size != 0
        ):
            # Whole C4 pages can use the normal HiCache page-row copy below.
            # Token-granular DSV4 C4 copy needs this helper because a token is
            # not one contiguous byte range in the paged row:
            # [value0..value63][scale0..scale63].
            transfer_cache_dsv4_mla(
                src_ptrs=self.device_ptrs,
                dst_ptrs=self.data_ptrs,
                src_indices=device_indices.to(dtype=torch.int64),
                dst_indices=host_indices.to(dtype=torch.int64),
            )
            return
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)
        if io_backend == "kernel" and self.layout == "layer_first":
            transfer_kv_all_layer_mla(
                src_layers=self.device_ptrs,
                dst_layers=self.data_ptrs,
                src_indices=device_rows,
                dst_indices=host_rows,
                item_size=self.item_bytes,
                num_layers=self.layer_num,
            )
        elif io_backend == "kernel" and self.layout == "page_first":
            if self.can_use_write_back_jit:
                jit_transfer_hicache_all_layer_mla_staged_lf_pf(
                    ptr_src=self.device_ptrs,
                    src_indices=device_rows,
                    dst_indices=host_rows,
                    staging=self.staging_buffer,
                    dst=self.kv_buffer,
                    page_size=1,
                    element_size=self.item_bytes,
                )
            else:
                transfer_kv_all_layer_mla_lf_pf(
                    src_layers=self.device_ptrs,
                    dst=self.kv_buffer,
                    src_indices=device_rows,
                    dst_indices=host_rows,
                    item_size=self.item_bytes,
                    dst_layout_dim=self.layer_num * self.item_bytes,
                    num_layers=self.layer_num,
                )
        elif io_backend == "direct" and self.layout == "layer_first":
            transfer_kv_direct(
                src_layers=self.device_buffers,
                dst_layers=self.data_refs,
                src_indices=device_rows,
                dst_indices=host_rows,
                page_size=1,
            )
        elif io_backend == "direct" and self.layout == "page_first_direct":
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=self.device_buffers,
                dst_ptrs=[self.kv_buffer],
                src_indices=device_rows,
                dst_indices=host_rows,
                page_size=1,
            )
        else:
            raise ValueError(
                f"Unsupported V4 paged host layout/backend: {self.layout}/{io_backend}"
            )

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        if not self._has_transfer_indices(host_indices, device_indices):
            return
        if (
            host_indices.numel() % self.slot_page_size != 0
            or device_indices.numel() % self.slot_page_size != 0
        ):
            # Same DSV4 C4 layout issue as backup: this is token-granular
            # preload, so it cannot use the normal HiCache page-row copy.
            transfer_cache_dsv4_mla(
                src_ptrs=self.data_ptrs[layer_id : layer_id + 1],
                dst_ptrs=self.device_ptrs[layer_id : layer_id + 1],
                src_indices=host_indices.to(dtype=torch.int64),
                dst_indices=device_indices.to(dtype=torch.int64),
            )
            return
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)

        if io_backend == "kernel" and self.layout == "layer_first":
            transfer_kv_per_layer_mla(
                src=self.data_refs[layer_id],
                dst=self.device_buffers[layer_id],
                src_indices=host_rows,
                dst_indices=device_rows,
                item_size=self.item_bytes,
            )
        elif io_backend == "kernel" and self.layout == "page_first":
            transfer_kv_per_layer_mla_pf_lf(
                src=self.kv_buffer,
                dst=self.device_buffers[layer_id],
                src_indices=host_rows,
                dst_indices=device_rows,
                layer_id=layer_id,
                item_size=self.item_bytes,
                src_layout_dim=self.layer_num * self.item_bytes,
            )
        elif io_backend == "direct" and self.layout == "layer_first":
            transfer_kv_direct(
                src_layers=[self.data_refs[layer_id]],
                dst_layers=[self.device_buffers[layer_id]],
                src_indices=host_rows,
                dst_indices=device_rows,
                page_size=1,
            )
        elif io_backend == "direct" and self.layout == "page_first_direct":
            transfer_kv_per_layer_direct_pf_lf(
                src_ptrs=[self.kv_buffer],
                dst_ptrs=[self.device_buffers[layer_id]],
                src_indices=host_rows,
                dst_indices=device_rows,
                layer_id=layer_id,
                page_size=1,
            )
        else:
            raise ValueError(
                f"Unsupported V4 paged host layout/backend: {self.layout}/{io_backend}"
            )

    def get_data_page(self, index, flat=True):
        index = int(index) // self.slot_page_size
        if self.layout == "layer_first":
            data_page = torch.stack(
                [self.kv_buffer[i][index] for i in range(self.layer_num)]
            )
        elif self.layout in ["page_first", "page_first_direct"]:
            data_page = self.kv_buffer[index]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self):
        return torch.zeros(
            (self.layer_num, self.item_bytes),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index, data_page):
        index = int(index) // self.slot_page_size
        if self.layout == "layer_first":
            data = data_page.view(self.dtype).reshape(self.layer_num, self.item_bytes)
            for i in range(self.layer_num):
                self.kv_buffer[i][index].copy_(data[i])
        elif self.layout == "page_first":
            self.kv_buffer[index].copy_(
                data_page.view(self.dtype).reshape(self.layer_num, self.item_bytes)
            )
        elif self.layout == "page_first_direct":
            self.kv_buffer[index].copy_(
                data_page.view(self.dtype).reshape(self.layer_num, 1, self.item_bytes)
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        ptr_list = []
        rows = self._to_page_indices(indices).tolist()
        if self.layout == "layer_first":
            for row in rows:
                page_index = int(row)
                for layer_id in range(self.layer_num):
                    ptr = (
                        self.kv_buffer[layer_id].data_ptr()
                        + page_index * self.item_bytes * self.dtype.itemsize
                    )
                    ptr_list.append(ptr)
            element_size = self.item_bytes * self.dtype.itemsize
            return ptr_list, [element_size] * len(ptr_list)
        if self.layout in ["page_first", "page_first_direct"]:
            page_bytes = self.layer_num * self.item_bytes * self.dtype.itemsize
            for row in rows:
                ptr_list.append(self.kv_buffer[int(row)].data_ptr())
            return ptr_list, [page_bytes] * len(ptr_list)
        raise ValueError(f"Unsupported layout: {self.layout}")

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        if self.layout not in ["page_first", "page_first_direct"]:
            return False
        page_bytes = self.layer_num * self.item_bytes * self.dtype.itemsize
        return (
            self.kv_buffer.data_ptr() % page_size_bytes == 0
            and page_bytes % page_size_bytes == 0
        )


class DeepSeekV4StateHostPool(HostKVCache):
    """Host pool for V4 CompressStatePool page rows."""

    def __init__(
        self,
        pool_name: str,
        state_pools: list,
        num_host_pages: int,
        swa_page_size: int,
        layout: str = "layer_first",
        device: str = "cpu",
        pin_memory: bool = True,
        allocator_type: str = "default",
    ):
        if any(pool is None for pool in state_pools):
            raise ValueError(f"{pool_name} state_pools must not contain None")

        self.pool_name = pool_name
        self.state_pools = state_pools
        self.layer_num = len(state_pools)
        self.num_host_pages = num_host_pages
        self.swa_page_size = swa_page_size
        self.dtype = torch.uint8
        self.device = device
        self.pin_memory = pin_memory
        self.allocator = get_allocator_from_storage(allocator_type)
        self.page_size = swa_page_size
        self.size = num_host_pages * swa_page_size
        self.layout = layout
        self.start_layer = 0
        self.end_layer = self.layer_num
        self.lock = threading.RLock()

        self.ring_size = 0
        self.state_page_bytes = 0
        self.device_page_views = []
        self.gpu_device = device
        self._init_device_page_views()
        self.size_per_token = self.state_page_bytes

        requested_bytes = self.layer_num * num_host_pages * self.state_page_bytes
        host_mem = psutil.virtual_memory()
        available_bytes = host_mem.available - HICACHE_HOST_MEMORY_RESERVE_BYTES
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory for V4 state pool {pool_name}. "
                f"Requesting {requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free."
            )

        alloc_func = ALLOC_MEMORY_FUNCS[self.gpu_device]
        self.data_refs = []
        if self.layout == "layer_first":
            self.kv_buffer = [
                alloc_func(
                    (num_host_pages, self.state_page_bytes),
                    dtype=self.dtype,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    allocator=self.allocator,
                )
                for _ in range(self.layer_num)
            ]
            self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        elif self.layout == "page_first":
            self.kv_buffer = alloc_func(
                (num_host_pages, self.layer_num, self.state_page_bytes),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
        elif self.layout == "page_first_direct":
            self.kv_buffer = alloc_func(
                (num_host_pages, self.layer_num, 1, self.state_page_bytes),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        logger.info(
            "Allocating %.2f GB host memory for V4 state pool '%s' "
            "(layers=%d, pages=%d, state_page_bytes=%d, layout=%s).",
            requested_bytes / 1e9,
            self.pool_name,
            self.layer_num,
            num_host_pages,
            self.state_page_bytes,
            self.layout,
        )
        self.device_ptrs = torch.tensor(
            [x.data_ptr() for x in self.device_page_views],
            dtype=torch.uint64,
            device=self.gpu_device,
        )
        self.data_ptrs = (
            torch.tensor(
                [x.data_ptr() for x in self.data_refs],
                dtype=torch.uint64,
                device=self.gpu_device,
            )
            if self.data_refs
            else None
        )
        self.can_use_jit = False
        self.can_use_write_back_jit = False
        self._init_write_back_staging_buffers()

    def _init_device_page_views(self) -> None:
        expected_ring_size = None
        expected_state_page_bytes = None
        for pool in self.state_pools:
            state_tensor = pool.kv_score_buffer.kv_score
            if not state_tensor.is_contiguous():
                raise ValueError(f"{self.pool_name} state tensor must be contiguous")
            ring_size = pool.ring_size
            slot_bytes = state_tensor[0].nbytes
            state_page_bytes = ring_size * slot_bytes
            if expected_ring_size is None:
                expected_ring_size = ring_size
                expected_state_page_bytes = state_page_bytes
                self.gpu_device = state_tensor.device
            elif (
                expected_ring_size != ring_size
                or expected_state_page_bytes != state_page_bytes
            ):
                raise ValueError(
                    f"{self.pool_name} state pools must share ring size and slot bytes"
                )

            state_bytes = state_tensor.view(torch.uint8).reshape(
                state_tensor.shape[0], -1
            )
            usable_slots = (state_tensor.shape[0] // ring_size) * ring_size
            self.device_page_views.append(
                state_bytes[:usable_slots].reshape(-1, state_page_bytes)
            )

        self.ring_size = expected_ring_size or 0
        self.state_page_bytes = expected_state_page_bytes or 0

    def _init_write_back_staging_buffers(self):
        self.staging_buffer = None
        if self.layout != "page_first" or (_is_npu or _is_xpu or _is_mps):
            return

        self.can_use_write_back_jit = _is_cuda and can_use_write_back_jit_kernel(
            element_size=self.state_page_bytes * self.dtype.itemsize,
        )
        staging_page_capacity = min(self.num_host_pages, _WRITE_BACK_STAGING_PAGE_CHUNK)
        self.staging_buffer = torch.empty(
            (staging_page_capacity, self.layer_num, self.state_page_bytes),
            dtype=self.dtype,
            device=self.gpu_device,
        )

    def _to_page_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() % self.swa_page_size != 0:
            raise ValueError(
                f"{self.pool_name} transfer indices must be SWA-page-aligned, "
                f"got numel={indices.numel()}, swa_page_size={self.swa_page_size}"
            )
        return indices.reshape(-1, self.swa_page_size)[:, 0] // self.swa_page_size

    def get_size_per_token(self):
        return self.state_page_bytes

    def get_ksize_per_token(self):
        return self.state_page_bytes

    def init_kv_buffer(self):
        return self.kv_buffer

    def get_hybrid_pool_buffer(self):
        return self.kv_buffer if isinstance(self.kv_buffer, list) else [self.kv_buffer]

    def clear(self):
        pass

    def available_size(self):
        raise NotImplementedError(
            f"{self.pool_name} reuses SWA transfer indices and has no allocator"
        )

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        raise NotImplementedError(
            f"{self.pool_name} reuses SWA transfer indices and has no allocator"
        )

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        raise NotImplementedError(
            f"{self.pool_name} reuses SWA transfer indices and has no free list"
        )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if host_indices is None or device_indices is None:
            return
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)
        if io_backend == "kernel" and self.layout == "layer_first":
            assert self.data_ptrs is not None
            transfer_kv_all_layer_mla(
                src_layers=self.device_ptrs,
                dst_layers=self.data_ptrs,
                src_indices=device_rows,
                dst_indices=host_rows,
                item_size=self.state_page_bytes,
                num_layers=self.layer_num,
            )
        elif io_backend == "kernel" and self.layout == "page_first":
            if self.can_use_write_back_jit:
                jit_transfer_hicache_all_layer_mla_staged_lf_pf(
                    ptr_src=self.device_ptrs,
                    src_indices=device_rows,
                    dst_indices=host_rows,
                    staging=self.staging_buffer,
                    dst=self.kv_buffer,
                    page_size=1,
                    element_size=self.state_page_bytes,
                )
            else:
                transfer_kv_all_layer_mla_lf_pf(
                    src_layers=self.device_ptrs,
                    dst=self.kv_buffer,
                    src_indices=device_rows,
                    dst_indices=host_rows,
                    item_size=self.state_page_bytes,
                    dst_layout_dim=self.layer_num * self.state_page_bytes,
                    num_layers=self.layer_num,
                )
        elif io_backend == "direct" and self.layout == "layer_first":
            transfer_kv_direct(
                src_layers=self.device_page_views,
                dst_layers=self.data_refs,
                src_indices=device_rows,
                dst_indices=host_rows,
                page_size=1,
            )
        elif io_backend == "direct" and self.layout == "page_first_direct":
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=self.device_page_views,
                dst_ptrs=[self.kv_buffer],
                src_indices=device_rows,
                dst_indices=host_rows,
                page_size=1,
            )
        else:
            raise ValueError(
                f"Unsupported V4 state host layout/backend: {self.layout}/{io_backend}"
            )

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        if host_indices is None or device_indices is None:
            return
        host_rows = self._to_page_indices(host_indices)
        device_rows = self._to_page_indices(device_indices)
        if io_backend == "kernel" and self.layout == "layer_first":
            transfer_kv_per_layer_mla(
                src=self.data_refs[layer_id],
                dst=self.device_page_views[layer_id],
                src_indices=host_rows,
                dst_indices=device_rows,
                item_size=self.state_page_bytes,
            )
        elif io_backend == "kernel" and self.layout == "page_first":
            transfer_kv_per_layer_mla_pf_lf(
                src=self.kv_buffer,
                dst=self.device_page_views[layer_id],
                src_indices=host_rows,
                dst_indices=device_rows,
                layer_id=layer_id,
                item_size=self.state_page_bytes,
                src_layout_dim=self.layer_num * self.state_page_bytes,
            )
        elif io_backend == "direct" and self.layout == "layer_first":
            transfer_kv_direct(
                src_layers=[self.data_refs[layer_id]],
                dst_layers=[self.device_page_views[layer_id]],
                src_indices=host_rows,
                dst_indices=device_rows,
                page_size=1,
            )
        elif io_backend == "direct" and self.layout == "page_first_direct":
            transfer_kv_per_layer_direct_pf_lf(
                src_ptrs=[self.kv_buffer],
                dst_ptrs=[self.device_page_views[layer_id]],
                src_indices=host_rows,
                dst_indices=device_rows,
                layer_id=layer_id,
                page_size=1,
            )
        else:
            raise ValueError(
                f"Unsupported V4 state host layout/backend: {self.layout}/{io_backend}"
            )

    def get_data_page(self, index, flat=True):
        index = int(index) // self.swa_page_size
        if self.layout == "layer_first":
            data_page = torch.stack(
                [self.kv_buffer[i][index] for i in range(self.layer_num)]
            )
        elif self.layout in ["page_first", "page_first_direct"]:
            data_page = self.kv_buffer[index]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self):
        return torch.zeros(
            (self.layer_num, self.state_page_bytes),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index, data_page):
        index = int(index) // self.swa_page_size
        if self.layout == "layer_first":
            data = data_page.view(self.dtype).reshape(
                self.layer_num, self.state_page_bytes
            )
            for i in range(self.layer_num):
                self.kv_buffer[i][index].copy_(data[i])
        elif self.layout == "page_first":
            self.kv_buffer[index].copy_(
                data_page.view(self.dtype).reshape(
                    self.layer_num, self.state_page_bytes
                )
            )
        elif self.layout == "page_first_direct":
            self.kv_buffer[index].copy_(
                data_page.view(self.dtype).reshape(
                    self.layer_num, 1, self.state_page_bytes
                )
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        ptr_list = []
        rows = self._to_page_indices(indices).tolist()
        if self.layout == "layer_first":
            for row in rows:
                page_index = int(row)
                for layer_id in range(self.layer_num):
                    ptr = (
                        self.kv_buffer[layer_id].data_ptr()
                        + page_index * self.state_page_bytes * self.dtype.itemsize
                    )
                    ptr_list.append(ptr)
            element_size = self.state_page_bytes * self.dtype.itemsize
            return ptr_list, [element_size] * len(ptr_list)
        if self.layout in ["page_first", "page_first_direct"]:
            page_bytes = self.layer_num * self.state_page_bytes * self.dtype.itemsize
            for row in rows:
                ptr_list.append(self.kv_buffer[int(row)].data_ptr())
            return ptr_list, [page_bytes] * len(ptr_list)
        raise ValueError(f"Unsupported layout: {self.layout}")

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        if self.layout not in ["page_first", "page_first_direct"]:
            return False
        page_bytes = self.layer_num * self.state_page_bytes * self.dtype.itemsize
        return (
            self.kv_buffer.data_ptr() % page_size_bytes == 0
            and page_bytes % page_size_bytes == 0
        )


@dataclass
class PoolEntry:
    name: PoolName
    host_pool: Any
    device_pool: Any
    layer_mapper: Callable[[int], Optional[int]]
    is_primary_index_anchor: bool = False
    # Optional eviction callbacks for auto-alloc in HybridCacheController.
    # host_evict_fn(n): evict n slots from the host pool (used by write()).
    # device_evict_fn(n): evict n slots from the device pool (used by load()).
    host_evict_fn: Optional[Callable] = None
    device_evict_fn: Optional[Callable] = None
    # Optional alloc/free overrides for the device side, used by
    # _resolve_pool_transfers_allocation. Set when entry.device_pool is the
    # raw KV/state pool (layout) rather than an allocator (e.g. SWA/Mamba,
    # where alloc lives on a separate allocator object).
    # When None, fall back to entry.device_pool.alloc/free.
    device_alloc_fn: Optional[Callable] = None
    device_free_fn: Optional[Callable] = None


class HostPoolGroup:
    def __init__(self, entries: list[PoolEntry]):
        if not entries:
            raise ValueError("HostPoolGroup requires at least one pool entry.")
        self.entries = entries
        self.entry_map = {entry.name: entry for entry in entries}
        self.anchor_entry = next(
            (entry for entry in entries if entry.is_primary_index_anchor),
            entries[0],
        )

        self.layout = self.anchor_entry.host_pool.layout
        self.page_size = self.anchor_entry.host_pool.page_size
        self.device = self.anchor_entry.host_pool.device
        self.size = self.anchor_entry.host_pool.size
        self.can_use_write_back_jit = all(
            getattr(entry.host_pool, "can_use_write_back_jit", False)
            for entry in entries
        )

    @property
    def kv_buffer(self):
        return self.anchor_entry.host_pool.kv_buffer

    @property
    def size_per_token(self):
        return self.anchor_entry.host_pool.size_per_token

    @property
    def allocator(self):
        return self.anchor_entry.host_pool.allocator

    @property
    def dtype(self):
        return self.anchor_entry.host_pool.dtype

    @property
    def start_layer(self):
        return self.anchor_entry.host_pool.start_layer

    @property
    def end_layer(self):
        return self.anchor_entry.host_pool.end_layer

    def get_ksize_per_token(self):
        return self.anchor_entry.host_pool.get_ksize_per_token()

    def get_pool(self, name: PoolName):
        return self.entry_map[name].host_pool

    def get_page_buffer_meta(self, indices):
        return self.anchor_entry.host_pool.get_page_buffer_meta(indices)

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        return self.anchor_entry.host_pool.is_stride_page_aligned(page_size_bytes)

    def clear(self) -> None:
        for entry in self.entries:
            entry.host_pool.clear()

    def available_size(self):
        return self.anchor_entry.host_pool.available_size()

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        return self.anchor_entry.host_pool.alloc(need_size)

    def free(self, indices: torch.Tensor) -> int:
        return self.anchor_entry.host_pool.free(indices)

    def get_data_page(self, index, flat: bool = True):
        return self.anchor_entry.host_pool.get_data_page(index, flat)

    def get_dummy_flat_data_page(self):
        return self.anchor_entry.host_pool.get_dummy_flat_data_page()

    def set_from_flat_data_page(self, index: int, data_page) -> None:
        return self.anchor_entry.host_pool.set_from_flat_data_page(index, data_page)

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        pool_transfers: Optional[list] = None,
    ) -> None:
        # 1. Anchor (KV) transfer
        anchor = self.anchor_entry
        local_layer_id = anchor.layer_mapper(layer_id)
        if local_layer_id is not None and host_indices.numel() > 0:
            anchor.host_pool.load_to_device_per_layer(
                anchor.device_pool,
                host_indices,
                device_indices,
                local_layer_id,
                io_backend,
            )

        # 2. Extra pool transfers
        for transfer in pool_transfers or []:
            entry = self.entry_map.get(transfer.name)
            if entry is None or transfer.host_indices is None:
                continue
            local_layer_id = entry.layer_mapper(layer_id)
            if local_layer_id is None:
                continue
            entry.host_pool.load_to_device_per_layer(
                entry.device_pool,
                transfer.host_indices,
                transfer.device_indices,
                local_layer_id,
                io_backend,
            )

    def backup_from_device_all_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        io_backend,
        pool_transfers: Optional[list] = None,
    ) -> None:
        # 1. Anchor (KV) backup
        self.anchor_entry.host_pool.backup_from_device_all_layer(
            self.anchor_entry.device_pool,
            host_indices,
            device_indices,
            io_backend,
        )
        # 2. Extra pool backup
        for transfer in pool_transfers or []:
            entry = self.entry_map.get(transfer.name)
            if entry is None or transfer.host_indices is None:
                continue
            entry.host_pool.backup_from_device_all_layer(
                entry.device_pool,
                transfer.host_indices,
                transfer.device_indices,
                io_backend,
            )


class DSAIndexerPoolHost(HostKVCache):
    """Host-side DSA index buffers only. Slot layout matches the anchor MLA host pool."""

    device_pool: DSATokenToKVPool

    def __init__(
        self,
        device_pool: DSATokenToKVPool,
        anchor_host: MLATokenToKVPoolHost,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ):
        self.device_pool = device_pool
        self.page_size = anchor_host.page_size
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)
        self.dtype = device_pool.store_dtype
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer
        self.layer_num = self._effective_host_layer_num()

        self.index_head_dim = device_pool.index_head_dim
        self.indexer_quant_block_size = device_pool.quant_block_size
        self.indexer_dtype = DSATokenToKVPool.index_k_with_scale_buffer_dtype
        self.indexer_size_per_token = (
            self.index_head_dim
            + self.index_head_dim // self.indexer_quant_block_size * 4
        )
        self.size = anchor_host.size
        self.page_num = anchor_host.page_num

        self.indexer_page_stride_size = (
            self.indexer_size_per_token * self.page_size * self.indexer_dtype.itemsize
        )
        self.indexer_layout_dim = self.indexer_page_stride_size * self.layer_num
        self.indexer_page_num = (self.size + self.page_size + 1) // self.page_size
        self.size_per_token = (
            self.indexer_size_per_token * self.layer_num * self.indexer_dtype.itemsize
        )

        buf_elem_size = self.page_num * self.layer_num * self.indexer_page_stride_size
        requested_bytes = buf_elem_size * self.indexer_dtype.itemsize
        host_mem = psutil.virtual_memory()
        available_bytes = host_mem.available - HICACHE_HOST_MEMORY_RESERVE_BYTES
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory for DSA indexer hierarchical cache. "
                f"Requesting {requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free."
            )
        logger.info(
            "Allocating %.2f GB host memory for DSA indexer (layout=%s).",
            requested_bytes / 1e9,
            layout,
        )
        self.init_kv_buffer()
        self.can_use_jit = False
        self.can_use_write_back_jit = False
        self._init_write_back_staging_buffers()
        self.lock = threading.RLock()
        self.clear()

    def get_size_per_token(self):
        return (
            self.indexer_size_per_token * self.layer_num * self.indexer_dtype.itemsize
        )

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    def init_kv_buffer(self):
        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        self.index_k_device_ptrs = torch.tensor(
            [x.data_ptr() for x in self.device_pool.index_k_with_scale_buffer],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        if self.layout == "layer_first":
            self.index_k_with_scale_buffer = alloc_func(
                (self.layer_num, self.indexer_page_num, self.indexer_page_stride_size),
                dtype=self.indexer_dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.index_k_data_refs = [
                self.index_k_with_scale_buffer[i] for i in range(self.layer_num)
            ]
            self.index_k_data_ptrs = torch.tensor(
                [x.data_ptr() for x in self.index_k_data_refs],
                dtype=torch.uint64,
                device=self.device_pool.device,
            )
        elif self.layout in ["page_first", "page_first_direct"]:
            self.index_k_with_scale_buffer = alloc_func(
                (
                    self.indexer_page_num,
                    self.layer_num,
                    1,
                    self.indexer_page_stride_size,
                ),
                dtype=self.indexer_dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def _init_write_back_staging_buffers(self):
        self.staging_buffer = None
        if self.layout != "page_first" or (_is_npu or _is_xpu or _is_mps):
            return

        self.can_use_write_back_jit = _is_cuda and can_use_write_back_jit_kernel(
            element_size=self.indexer_page_stride_size * self.indexer_dtype.itemsize,
        )
        staging_page_capacity = min(
            self.indexer_page_num, _WRITE_BACK_STAGING_PAGE_CHUNK
        )
        self.staging_buffer = torch.empty(
            (
                staging_page_capacity,
                self.layer_num,
                1,
                self.indexer_page_stride_size,
            ),
            dtype=self.indexer_dtype,
            device=self.device_pool.device,
        )

    def get_hybrid_pool_buffer(self):
        return [self.index_k_with_scale_buffer]

    def _get_indexer_page_indices(self, host_indices, device_indices):
        if host_indices.numel() == 0:
            return host_indices, device_indices
        if host_indices.numel() % self.page_size != 0:
            raise ValueError(
                "Index buffer transfer expects page-aligned indices for DSA."
            )
        host_page_indices = (
            host_indices.reshape(-1, self.page_size)[:, 0] // self.page_size
        )
        device_page_indices = (
            device_indices.reshape(-1, self.page_size)[:, 0] // self.page_size
        )
        return host_page_indices, device_page_indices

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        if not self._is_device_layer_owned(device_pool, layer_id):
            return
        host_layer = self._host_layer_index(layer_id)

        host_page_indices, device_page_indices = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
        if use_kernel:
            if self.layout == "layer_first":
                transfer_kv_per_layer_mla(
                    src=self.index_k_with_scale_buffer[host_layer],
                    dst=device_pool.index_k_with_scale_buffer[layer_id],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    item_size=self.indexer_page_stride_size,
                )
            elif self.layout == "page_first":
                transfer_kv_per_layer_mla_pf_lf(
                    src=self.index_k_with_scale_buffer,
                    dst=device_pool.index_k_with_scale_buffer[layer_id],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    layer_id=host_layer,
                    item_size=self.indexer_page_stride_size,
                    src_layout_dim=self.indexer_layout_dim,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.index_k_with_scale_buffer[host_layer]],
                    dst_layers=[device_pool.index_k_with_scale_buffer[layer_id]],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    page_size=1,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.index_k_with_scale_buffer],
                    dst_ptrs=[device_pool.index_k_with_scale_buffer[layer_id]],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    layer_id=host_layer,
                    page_size=1,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def _backup_from_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        host_layer = self._host_layer_index(layer_id)
        host_page_indices, device_page_indices = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
        if use_kernel:
            if self.layout == "layer_first":
                transfer_kv_per_layer_mla(
                    src=device_pool.index_k_with_scale_buffer[layer_id],
                    dst=self.index_k_with_scale_buffer[host_layer],
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    item_size=self.indexer_page_stride_size,
                )
            elif self.layout == "page_first":
                raise ValueError(
                    "Layer-sharded DSA indexer HiCache backup with page_first "
                    "layout is not supported without a per-layer LF->PF kernel."
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[device_pool.index_k_with_scale_buffer[layer_id]],
                    dst_layers=[self.index_k_with_scale_buffer[host_layer]],
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    page_size=1,
                )
            else:
                raise ValueError(
                    "Layer-sharded direct DSA indexer backup only supports "
                    f"layer_first layout, got {self.layout}"
                )
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if self._is_device_layer_sharded(device_pool):
            for layer_id in self._owned_device_layer_ids(device_pool):
                self._backup_from_device_per_layer(
                    device_pool, host_indices, device_indices, layer_id, io_backend
                )
            return

        host_page_indices, device_page_indices = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
        if use_kernel:
            if self.layout == "layer_first":
                transfer_kv_all_layer_mla(
                    src_layers=self.index_k_device_ptrs,
                    dst_layers=self.index_k_data_ptrs,
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    item_size=self.indexer_page_stride_size,
                    num_layers=self.layer_num,
                )
            elif self.layout == "page_first":
                if self.can_use_write_back_jit:
                    jit_transfer_hicache_all_layer_mla_staged_lf_pf(
                        ptr_src=self.index_k_device_ptrs,
                        src_indices=device_page_indices,
                        dst_indices=host_page_indices,
                        staging=self.staging_buffer,
                        dst=self.index_k_with_scale_buffer,
                        page_size=1,
                        element_size=self.indexer_page_stride_size,
                    )
                else:
                    transfer_kv_all_layer_mla_lf_pf(
                        src_layers=self.index_k_device_ptrs,
                        dst=self.index_k_with_scale_buffer,
                        src_indices=device_page_indices,
                        dst_indices=host_page_indices,
                        item_size=self.indexer_page_stride_size,
                        dst_layout_dim=self.indexer_layout_dim,
                        num_layers=self.layer_num,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_pool.index_k_with_scale_buffer,
                    dst_layers=self.index_k_data_refs,
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    page_size=1,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=device_pool.index_k_with_scale_buffer,
                    dst_ptrs=[self.index_k_with_scale_buffer],
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    page_size=1,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        page_idx = int(index) // self.page_size
        if self.layout == "layer_first":
            data_page = self.index_k_with_scale_buffer[:, page_idx : page_idx + 1, :]
        elif self.layout in ["page_first", "page_first_direct"]:
            data_page = self.index_k_with_scale_buffer[page_idx : page_idx + 1, :, :, :]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        if flat:
            data_page = data_page.flatten()
        return data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            (self.layer_num, self.indexer_page_stride_size),
            dtype=self.indexer_dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        page_idx = int(index) // self.page_size
        if self.layout == "layer_first":
            self.index_k_with_scale_buffer[:, page_idx : page_idx + 1, :] = (
                data_page.reshape(
                    self.layer_num,
                    1,
                    self.indexer_page_stride_size,
                )
            )
        elif self.layout in ["page_first", "page_first_direct"]:
            self.index_k_with_scale_buffer[page_idx : page_idx + 1, :, :, :] = (
                data_page.reshape(
                    1,
                    self.layer_num,
                    1,
                    self.indexer_page_stride_size,
                )
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        """Meta data for zero-copy storage I/O."""
        assert len(indices) % self.page_size == 0
        if self.layout not in ["page_first", "page_first_direct"]:
            raise ValueError(f"Unsupported layout: {self.layout}")
        ptr_list = []
        indices = indices.tolist()
        page_stride_bytes = (
            self.layer_num * self.indexer_page_stride_size * self.indexer_dtype.itemsize
        )
        base_ptr = self.index_k_with_scale_buffer.data_ptr()
        for i in range(0, len(indices), self.page_size):
            page_index = int(indices[i]) // self.page_size
            ptr_list.append(base_ptr + page_index * page_stride_bytes)
        return ptr_list, [page_stride_bytes] * len(ptr_list)

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        if self.layout not in ["page_first", "page_first_direct"]:
            return False
        page_stride_bytes = (
            self.layer_num * self.indexer_page_stride_size * self.indexer_dtype.itemsize
        )
        return (
            self.index_k_with_scale_buffer.data_ptr() % page_size_bytes == 0
            and page_stride_bytes % page_size_bytes == 0
        )
