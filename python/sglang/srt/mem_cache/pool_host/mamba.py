from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
import psutil
import torch

from sglang.jit_kernel.hicache import (
    can_use_write_back_jit_kernel,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_all_layer_mla_staged_lf_pf as jit_transfer_hicache_all_layer_mla_staged_lf_pf,
)
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.mem_cache.pool_host.base import (
    _WRITE_BACK_STAGING_PAGE_CHUNK,
    HICACHE_HOST_MEMORY_RESERVE_BYTES,
    HostKVCache,
    sync_fixed_hicache_size,
    synchronized,
)
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    get_allocator_from_storage,
)
from sglang.srt.utils import is_cuda, is_hip, is_mps, is_npu, is_xpu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_xpu = is_xpu()
_is_mps = is_mps()
if _is_cuda or _is_hip:
    from sgl_kernel.kvcacheio import (
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_all_layer_mla_lf_pf,
        transfer_kv_direct,
        transfer_kv_per_layer_direct_pf_lf,
        transfer_kv_per_layer_mla,
        transfer_kv_per_layer_mla_pf_lf,
    )

logger = logging.getLogger(__name__)


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

        # TODO: Mamba pool is currently incompatible with write-back staging
        # kernel; only allow 'page_first_direct' + 'direct' for now.
        # Relax this restriction once the staging bug is fixed.
        if layout != "page_first_direct":
            raise ValueError(
                f"MambaPoolHost only supports layout='page_first_direct', "
                f"got '{layout}'."
            )

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
        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]

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
        self.can_use_write_back_jit = False
        self._temporal_can_use_jit = False
        self._conv_can_use_jit = [False] * len(self.conv_buffer)
        if self.layout != "page_first" or (_is_npu or _is_xpu or _is_mps):
            return

        self._temporal_can_use_jit = _is_cuda and can_use_write_back_jit_kernel(
            element_size=self._item_size_per_index(self.temporal_buffer[0]),
        )
        self._conv_can_use_jit = [
            _is_cuda
            and can_use_write_back_jit_kernel(
                element_size=self._item_size_per_index(buf[0]),
            )
            for buf in self.conv_buffer
        ]
        self.can_use_write_back_jit = self._temporal_can_use_jit and all(
            self._conv_can_use_jit
        )
        self.staging_page_capacity = min(self.page_num, _WRITE_BACK_STAGING_PAGE_CHUNK)
        self.staging_token_capacity = self.staging_page_capacity * self.page_size
        self.temporal_staging_buffer = torch.empty(
            (
                self.staging_token_capacity,
                self.num_mamba_layers,
                1,
                *self.temporal_state_shape,
            ),
            dtype=self.temporal_dtype,
            device=self.device_pool.device,
        )
        self.conv_staging_buffers = [
            torch.empty(
                (
                    self.staging_token_capacity,
                    self.num_mamba_layers,
                    1,
                    *conv_shape,
                ),
                dtype=self.conv_dtype,
                device=self.device_pool.device,
            )
            for conv_shape in self.conv_state_shapes
        ]

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

    def available_size(self):
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        assert (
            need_size % self.page_size == 0
        ), "The requested size should be a multiple of the page size."
        if need_size > self.available_size():
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        self.free_slots = torch.cat([self.free_slots, indices])
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
            transfer_kv_per_layer_mla_pf_lf(
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
            if can_use_jit:
                jit_transfer_hicache_all_layer_mla_staged_lf_pf(
                    ptr_src=src_ptrs,
                    src_indices=src_indices,
                    dst_indices=dst_indices,
                    staging=staging,
                    dst=dst,
                    page_size=1,
                    element_size=item_size,
                )
            else:
                transfer_kv_all_layer_mla_lf_pf(
                    src_layers=src_ptrs,
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
        if io_backend != "direct":
            raise ValueError(
                f"MambaPoolHost only supports io_backend='direct', "
                f"got '{io_backend}'."
            )
        if self.layout in ["page_first", "page_first_direct"]:
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
        if io_backend != "direct":
            raise ValueError(
                f"MambaPoolHost only supports io_backend='direct', "
                f"got '{io_backend}'."
            )
        if self.layout in ["page_first", "page_first_direct"]:
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
            # Emit component pointers in stable order:
            # temporal first, then conv_0..conv_n for this page.
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
