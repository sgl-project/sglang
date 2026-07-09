from __future__ import annotations

import logging
import threading
from typing import Optional

import psutil
import torch

from sglang.jit_kernel.hicache import (
    can_use_hicache_jit_kernel,
    can_use_write_back_jit_kernel,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_all_layer as jit_transfer_hicache_all_layer,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_all_layer_mla as jit_transfer_hicache_all_layer_mla,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_all_layer_staged_lf_pf as jit_transfer_hicache_all_layer_staged_lf_pf,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_one_layer as jit_transfer_hicache_one_layer,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_one_layer_mla as jit_transfer_hicache_one_layer_mla,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKOnlyPool, MHATokenToKVPool
from sglang.srt.mem_cache.pool_host.base import (
    _WRITE_BACK_STAGING_PAGE_CHUNK,
    HICACHE_HOST_MEMORY_RESERVE_BYTES,
    HostKVCache,
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
        transfer_kv_all_layer,
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_all_layer_lf_pf,
        transfer_kv_all_layer_lf_ph,
        transfer_kv_all_layer_mla_lf_pf,
        transfer_kv_direct,
        transfer_kv_per_layer,
        transfer_kv_per_layer_direct_pf_lf,
        transfer_kv_per_layer_mla,
        transfer_kv_per_layer_mla_pf_lf,
        transfer_kv_per_layer_pf_lf,
        transfer_kv_per_layer_ph_lf,
    )
if _is_npu:
    from sgl_kernel_npu.kvcacheio import TransferDirection, transfer_kv_dim_exchange

logger = logging.getLogger(__name__)


class MHATokenToKVPoolHost(HostKVCache):
    device_pool: MHATokenToKVPool

    def __init__(
        self,
        device_pool: MHATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        host_page_num: Optional[int] = None,
    ):
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
            host_page_num=host_page_num,
        )
        self.element_dim = self.device_pool.head_num * self.device_pool.head_dim
        self.can_use_jit = _is_cuda and can_use_hicache_jit_kernel(
            element_size=self.element_dim * self.dtype.itemsize
        )

        if self.layout == "page_first":
            # Transpose [page, layer, ...] -> [layer, page, ...] to get per-layer views
            # This swaps strides without copying data
            k_transposed = self.k_buffer.transpose(0, 1)
            v_transposed = self.v_buffer.transpose(0, 1)
            self.k_data_refs = [k_transposed[i] for i in range(self.layer_num)]
            self.v_data_refs = [v_transposed[i] for i in range(self.layer_num)]
        else:
            self.k_data_refs = [self.k_buffer[i] for i in range(self.layer_num)]
            self.v_data_refs = [self.v_buffer[i] for i in range(self.layer_num)]
        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self._init_write_back_staging_buffers()

    def get_size_per_token(self):
        self.head_num = self.device_pool.head_num
        self.head_dim = self.device_pool.head_dim
        self.layer_num = self.device_pool.layer_num
        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2

    def get_ksize_per_token(self):
        return self.get_size_per_token() // 2

    def init_kv_buffer(self):
        if self.layout == "layer_first":
            dims = (2, self.layer_num, self.size, self.head_num, self.head_dim)
        elif self.layout == "page_first":
            dims = (2, self.size, self.layer_num, self.head_num, self.head_dim)
        elif self.layout == "page_first_direct":
            dims = (
                2,
                self.page_num,
                self.layer_num,
                self.page_size,
                self.head_num,
                self.head_dim,
            )
        elif self.layout == "page_head":
            dims = (
                2,
                self.page_num,
                self.head_num,
                self.page_size,
                self.layer_num,
                self.head_dim,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        self.token_stride_size = self.head_num * self.head_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num

        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        buffer = alloc_func(
            dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )
        return buffer

    def _init_write_back_staging_buffers(self):
        self.staging_page_capacity = 0
        self.staging_token_capacity = 0
        self.staging_k_buffer = None
        self.staging_v_buffer = None
        self.can_use_write_back_jit = False
        if self.layout != "page_first" or (_is_npu or _is_xpu or _is_mps):
            return

        self.can_use_write_back_jit = _is_cuda and can_use_write_back_jit_kernel(
            element_size=self.element_dim * self.dtype.itemsize,
        )
        if not self.can_use_write_back_jit:
            return

        self.staging_page_capacity = min(self.page_num, _WRITE_BACK_STAGING_PAGE_CHUNK)
        self.staging_token_capacity = self.staging_page_capacity * self.page_size
        self.staging_k_buffer = torch.empty(
            (
                self.staging_token_capacity,
                self.layer_num,
                self.head_num,
                self.head_dim,
            ),
            dtype=self.dtype,
            device=self.device_pool.device,
        )
        self.staging_v_buffer = torch.empty_like(self.staging_k_buffer)

    @property
    def k_buffer(self):
        return self.kv_buffer[0]

    @property
    def v_buffer(self):
        return self.kv_buffer[1]

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer(
                        k_cache_dst=device_pool.k_buffer[layer_id],
                        v_cache_dst=device_pool.v_buffer[layer_id],
                        k_cache_src=self.k_buffer[layer_id],
                        v_cache_src=self.v_buffer[layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.element_dim,
                    )
                else:
                    transfer_kv_per_layer(
                        src_k=self.k_buffer[layer_id],
                        dst_k=device_pool.k_buffer[layer_id],
                        src_v=self.v_buffer[layer_id],
                        dst_v=device_pool.v_buffer[layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        item_size=self.token_stride_size,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    # Transpose [page, layer, ...] -> [layer, page, ...] then
                    # index by layer_id to get a per-layer view with strided layout.
                    # The kernel handles different src/dst strides automatically.
                    jit_transfer_hicache_one_layer(
                        k_cache_dst=device_pool.k_buffer[layer_id],
                        v_cache_dst=device_pool.v_buffer[layer_id],
                        k_cache_src=self.k_data_refs[layer_id],
                        v_cache_src=self.v_data_refs[layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.element_dim,
                    )
                else:
                    transfer_kv_per_layer_pf_lf(
                        src_k=self.k_buffer,
                        dst_k=device_pool.k_buffer[layer_id],
                        src_v=self.v_buffer,
                        dst_v=device_pool.v_buffer[layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        layer_id=layer_id,
                        item_size=self.token_stride_size,
                        src_layout_dim=self.layout_dim,
                    )
            elif self.layout == "page_head":
                transfer_kv_per_layer_ph_lf(
                    src_k=self.k_buffer,
                    dst_k=device_pool.k_buffer[layer_id],
                    src_v=self.v_buffer,
                    dst_v=device_pool.v_buffer[layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    item_size=self.token_stride_size,
                    src_layout_dim=self.layout_dim,
                    page_size=self.page_size,
                    head_num=self.head_num,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.k_buffer[layer_id], self.v_buffer[layer_id]],
                    dst_layers=[
                        device_pool.k_buffer[layer_id],
                        device_pool.v_buffer[layer_id],
                    ],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.k_buffer, self.v_buffer],
                    dst_ptrs=[
                        device_pool.k_buffer[layer_id],
                        device_pool.v_buffer[layer_id],
                    ],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_direct":
                # Ascend-specific: transfer KV data for all layers when layer_id == 0
                if layer_id == 0:
                    transfer_kv_dim_exchange(
                        device_indices=device_indices,
                        host_indices=host_indices,
                        device_k=device_pool.k_buffer,
                        host_k=self.k_buffer,
                        device_v=device_pool.v_buffer,
                        host_v=self.v_buffer,
                        page_size=self.page_size,
                        direction=TransferDirection.H2D,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_all_layer(
                        k_ptr_dst=self.k_data_ptrs,
                        v_ptr_dst=self.v_data_ptrs,
                        indices_dst=host_indices,
                        k_ptr_src=device_pool.k_data_ptrs,
                        v_ptr_src=device_pool.v_data_ptrs,
                        indices_src=device_indices,
                        kv_cache_dst_stride_bytes=self.token_stride_size,
                        kv_cache_src_stride_bytes=self.token_stride_size,
                        element_size=self.element_dim * self.dtype.itemsize,
                    )
                else:
                    transfer_kv_all_layer(
                        src_k_layers=device_pool.k_data_ptrs,
                        dst_k_layers=self.k_data_ptrs,
                        src_v_layers=device_pool.v_data_ptrs,
                        dst_v_layers=self.v_data_ptrs,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                        num_layers=self.layer_num,
                    )
            elif self.layout == "page_first":
                if self.can_use_write_back_jit:
                    jit_transfer_hicache_all_layer_staged_lf_pf(
                        k_ptr_src=device_pool.k_data_ptrs,
                        v_ptr_src=device_pool.v_data_ptrs,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        staging_k=self.staging_k_buffer,
                        staging_v=self.staging_v_buffer,
                        dst_k=self.k_buffer,
                        dst_v=self.v_buffer,
                        page_size=self.page_size,
                    )
                else:
                    transfer_kv_all_layer_lf_pf(
                        src_k_layers=device_pool.k_data_ptrs,
                        dst_k=self.k_buffer,
                        src_v_layers=device_pool.v_data_ptrs,
                        dst_v=self.v_buffer,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                        dst_layout_dim=self.layout_dim,
                        num_layers=self.layer_num,
                    )
            elif self.layout == "page_head":
                transfer_kv_all_layer_lf_ph(
                    src_k_layers=device_pool.k_data_ptrs,
                    dst_k=self.k_buffer,
                    src_v_layers=device_pool.v_data_ptrs,
                    dst_v=self.v_buffer,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    item_size=self.token_stride_size,
                    dst_layout_dim=self.layout_dim,
                    num_layers=self.layer_num,
                    page_size=self.page_size,
                    head_num=self.head_num,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_pool.k_buffer + device_pool.v_buffer,
                    dst_layers=self.k_data_refs + self.v_data_refs,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=device_pool.k_buffer + device_pool.v_buffer,
                    dst_ptrs=[self.k_buffer, self.v_buffer],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_direct":
                transfer_kv_dim_exchange(
                    device_indices=device_indices,
                    host_indices=host_indices,
                    device_k=device_pool.k_buffer,
                    host_k=self.k_buffer,
                    device_v=device_pool.v_buffer,
                    host_v=self.v_buffer,
                    page_size=self.page_size,
                    direction=TransferDirection.D2H,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        if self.layout == "layer_first":
            data_page = self.kv_buffer[:, :, index : index + self.page_size, :, :]
        elif self.layout == "page_first":
            data_page = self.kv_buffer[:, index : index + self.page_size, :, :, :]
        elif self.layout in ["page_first_direct", "page_head"]:
            real_index = index // self.page_size
            data_page = self.kv_buffer[:, real_index : real_index + 1, :, :, :, :]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        if flat:
            data_page = data_page.flatten()
        return data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            (2, self.layer_num, self.page_size, self.head_num, self.head_dim),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        if self.layout == "layer_first":
            self.kv_buffer[:, :, index : index + self.page_size, :, :] = (
                data_page.reshape(
                    2,
                    self.layer_num,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                )
            )
        elif self.layout == "page_first":
            self.kv_buffer[:, index : index + self.page_size, :, :, :] = (
                data_page.reshape(
                    2, self.page_size, self.layer_num, self.head_num, self.head_dim
                )
            )
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            self.kv_buffer[:, real_index : real_index + 1, :, :, :, :] = (
                data_page.reshape(
                    2, 1, self.layer_num, self.page_size, self.head_num, self.head_dim
                )
            )
        elif self.layout == "page_head":
            real_index = index // self.page_size
            self.kv_buffer[:, real_index : real_index + 1, :, :, :, :] = (
                data_page.reshape(
                    2, 1, self.head_num, self.page_size, self.layer_num, self.head_dim
                )
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_split_heads_page_buffer_meta(
        self, indices: torch.Tensor, split_factor: int
    ):
        """
        get meta data for zero copy of heterogeneous ranks' KVCache
        """
        assert self.layout == "page_head"
        assert len(indices) % self.page_size == 0
        assert self.head_num % split_factor == 0
        ptr_list = []
        kv_buffer_data_ptr = self.kv_buffer.data_ptr()
        indices = indices.tolist()
        v_offset = (
            self.layer_num
            * self.size
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        for index in range(0, len(indices), self.page_size):
            for head_id in range(0, self.head_num, self.head_num // split_factor):
                k_ptr = (
                    kv_buffer_data_ptr
                    + indices[index]
                    * self.layer_num
                    * self.head_num
                    * self.head_dim
                    * self.dtype.itemsize
                    + head_id
                    * self.page_size
                    * self.layer_num
                    * self.head_dim
                    * self.dtype.itemsize
                )
                v_ptr = k_ptr + v_offset
                ptr_list.append(k_ptr)
                ptr_list.append(v_ptr)
        element_size = (
            self.layer_num
            * self.dtype.itemsize
            * self.page_size
            * self.head_num
            * self.head_dim
            // split_factor
        )
        element_size_list = [element_size] * len(ptr_list)
        return ptr_list, element_size_list

    def get_page_buffer_meta(self, indices):
        """
        meta data for zero copy
        """
        assert len(indices) % self.page_size == 0
        ptr_list = []
        kv_buffer_data_ptr = self.kv_buffer.data_ptr()
        indices = indices.tolist()
        v_offset = (
            self.layer_num
            * self.size
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        if self.layout == "layer_first":
            for index in range(0, len(indices), self.page_size):
                for layer_id in range(self.layer_num):
                    k_ptr = (
                        kv_buffer_data_ptr
                        + indices[index]
                        * self.head_num
                        * self.head_dim
                        * self.dtype.itemsize
                        + layer_id
                        * self.size
                        * self.head_num
                        * self.head_dim
                        * self.dtype.itemsize
                    )
                    v_ptr = k_ptr + v_offset
                    ptr_list.append(k_ptr)
                    ptr_list.append(v_ptr)
            element_size = (
                self.dtype.itemsize * self.page_size * self.head_num * self.head_dim
            )
            element_size_list = [element_size] * len(ptr_list)
        elif self.layout in ["page_first", "page_first_direct", "page_head"]:
            for index in range(0, len(indices), self.page_size):
                k_ptr = (
                    kv_buffer_data_ptr
                    + indices[index]
                    * self.layer_num
                    * self.head_num
                    * self.head_dim
                    * self.dtype.itemsize
                )
                v_ptr = k_ptr + v_offset
                ptr_list.append(k_ptr)
                ptr_list.append(v_ptr)
            element_size = (
                self.layer_num
                * self.dtype.itemsize
                * self.page_size
                * self.head_num
                * self.head_dim
            )
            element_size_list = [element_size] * len(ptr_list)
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        return ptr_list, element_size_list

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        """Return True if per-page strides are multiples of *page_size_bytes*.

        When O_DIRECT is used with any file-based NIXL backend, every data pointer
        passed to the kernel must be page-aligned.  In zero-copy mode the
        pointer for KV page ``p`` is:

            base_ptr + p * page_size * layer_num * head_num * head_dim * itemsize

        For this to be page-aligned (given a page-aligned ``base_ptr``) the per-page
        stride must itself be a multiple of the OS page size.
        """
        if self.layout not in ("page_first", "page_first_direct", "page_head"):
            return False
        stride = (
            self.page_size
            * self.layer_num
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        base_aligned = self.kv_buffer.data_ptr() % page_size_bytes == 0
        return base_aligned and stride % page_size_bytes == 0


class MHATokenToKOnlyPoolHost(HostKVCache):
    """Host pool for MiniMax sparse index-K buffers (no index V)."""

    device_pool: MHATokenToKOnlyPool

    def __init__(
        self,
        device_pool: MHATokenToKOnlyPool,
        anchor_host: MHATokenToKVPoolHost,
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

        self.head_num = device_pool.head_num
        self.head_dim = device_pool.head_dim
        self.layer_num = device_pool.layer_num
        self.element_dim = self.head_num * self.head_dim
        self.token_stride_size = self.element_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num

        self.size = anchor_host.size
        self.page_num = anchor_host.page_num
        self.size_per_token = self.get_size_per_token()

        host_mem = psutil.virtual_memory()
        requested_bytes = self.size * self.size_per_token
        available_bytes = host_mem.available - HICACHE_HOST_MEMORY_RESERVE_BYTES
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory for MiniMax index-K hierarchical cache. "
                f"Requesting {requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free."
            )
        logger.info(
            "Allocating %.2f GB host memory for MiniMax sparse index-K (layout=%s).",
            requested_bytes / 1e9,
            layout,
        )

        self.init_kv_buffer()
        self.lock = threading.RLock()
        self.clear()

        self.can_use_jit = _is_cuda and can_use_hicache_jit_kernel(
            element_size=self.token_stride_size
        )
        self.k_device_ptrs = torch.tensor(
            [x.data_ptr() for x in self.device_pool.k_buffer],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        if self.layout == "page_first":
            transposed = self.k_buffer.transpose(0, 1)
            self.k_data_refs = [transposed[i] for i in range(self.layer_num)]
        elif self.layout == "layer_first":
            self.k_data_refs = [self.k_buffer[i] for i in range(self.layer_num)]
        else:
            self.k_data_refs = []
        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

    def get_size_per_token(self):
        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    def init_kv_buffer(self):
        if self.layout == "layer_first":
            dims = (self.layer_num, self.size, self.head_num, self.head_dim)
        elif self.layout == "page_first":
            dims = (self.size, self.layer_num, self.head_num, self.head_dim)
        elif self.layout == "page_first_direct":
            dims = (
                self.page_num,
                self.layer_num,
                self.page_size,
                self.head_num,
                self.head_dim,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        self.k_buffer = alloc_func(
            dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )

    def get_hybrid_pool_buffer(self):
        return [self.k_buffer]

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer_mla(
                        cache_dst=device_pool.k_buffer[layer_id],
                        cache_src=self.k_buffer[layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.element_dim,
                    )
                else:
                    transfer_kv_per_layer_mla(
                        src=self.k_buffer[layer_id],
                        dst=device_pool.k_buffer[layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        item_size=self.token_stride_size,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer_mla(
                        cache_dst=device_pool.k_buffer[layer_id],
                        cache_src=self.k_data_refs[layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.element_dim,
                    )
                else:
                    transfer_kv_per_layer_mla_pf_lf(
                        src=self.k_buffer,
                        dst=device_pool.k_buffer[layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        layer_id=layer_id,
                        item_size=self.token_stride_size,
                        src_layout_dim=self.layout_dim,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.k_buffer[layer_id]],
                    dst_layers=[device_pool.k_buffer[layer_id]],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.k_buffer],
                    dst_ptrs=[device_pool.k_buffer[layer_id]],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    for layer_id in range(self.layer_num):
                        jit_transfer_hicache_one_layer_mla(
                            cache_dst=self.k_buffer[layer_id],
                            cache_src=device_pool.k_buffer[layer_id],
                            indices_dst=host_indices,
                            indices_src=device_indices,
                            element_dim=self.element_dim,
                        )
                else:
                    for layer_id in range(self.layer_num):
                        transfer_kv_per_layer_mla(
                            src=device_pool.k_buffer[layer_id],
                            dst=self.k_buffer[layer_id],
                            src_indices=device_indices,
                            dst_indices=host_indices,
                            item_size=self.token_stride_size,
                        )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    jit_transfer_hicache_all_layer_mla(
                        ptr_dst=self.k_data_ptrs,
                        indices_dst=host_indices,
                        ptr_src=self.k_device_ptrs,
                        indices_src=device_indices,
                        cache_dst_stride_bytes=self.layout_dim,
                        cache_src_stride_bytes=self.token_stride_size,
                        element_size=self.element_dim * self.dtype.itemsize,
                    )
                else:
                    transfer_kv_all_layer_mla_lf_pf(
                        src_layers=self.k_device_ptrs,
                        dst=self.k_buffer,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                        dst_layout_dim=self.layout_dim,
                        num_layers=self.layer_num,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_pool.k_buffer,
                    dst_layers=self.k_data_refs,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=device_pool.k_buffer,
                    dst_ptrs=[self.k_buffer],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        if self.layout == "layer_first":
            data_page = self.k_buffer[:, index : index + self.page_size, :, :]
        elif self.layout == "page_first":
            data_page = self.k_buffer[index : index + self.page_size, :, :, :]
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            data_page = self.k_buffer[real_index : real_index + 1, :, :, :, :]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        if flat:
            return data_page.flatten()
        return data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            (self.layer_num, self.page_size, self.head_num, self.head_dim),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        if self.layout == "layer_first":
            self.k_buffer[:, index : index + self.page_size, :, :] = data_page.reshape(
                self.layer_num, self.page_size, self.head_num, self.head_dim
            )
        elif self.layout == "page_first":
            self.k_buffer[index : index + self.page_size, :, :, :] = data_page.reshape(
                self.page_size, self.layer_num, self.head_num, self.head_dim
            )
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            self.k_buffer[real_index : real_index + 1, :, :, :, :] = data_page.reshape(
                1, self.layer_num, self.page_size, self.head_num, self.head_dim
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        """Meta data for zero-copy storage I/O."""
        assert len(indices) % self.page_size == 0
        if self.layout not in ["page_first", "page_first_direct"]:
            raise ValueError(f"Unsupported layout: {self.layout}")

        ptr_list = []
        k_buffer_data_ptr = self.k_buffer.data_ptr()
        indices = indices.tolist()
        for index in range(0, len(indices), self.page_size):
            k_ptr = (
                k_buffer_data_ptr
                + indices[index]
                * self.layer_num
                * self.head_num
                * self.head_dim
                * self.dtype.itemsize
            )
            ptr_list.append(k_ptr)
        element_size = (
            self.layer_num
            * self.dtype.itemsize
            * self.page_size
            * self.head_num
            * self.head_dim
        )
        element_size_list = [element_size] * len(ptr_list)
        return ptr_list, element_size_list


class AsymmetricMHATokenToKVPoolHost(MHATokenToKVPoolHost):
    """Host KV pool for MHA models whose K and V have different head dims
    (``head_dim != v_head_dim``), e.g. MiMo-V2.

    K and V are stored in two independent host buffers (``self.k_buffer`` and
    ``self.v_buffer``) instead of a single ``(2, ...)`` tensor, so each side
    keeps its native stride. The kernel transfer path dispatches K and V as
    independent single-buffer copies so each side uses its own ``item_size``.
    K/V direct transfers must be dispatched separately because the direct
    kernels derive copy sizes from each call's first tensor.
    """

    def get_size_per_token(self):
        self.head_num = self.device_pool.head_num
        self.head_dim = self.device_pool.head_dim
        self.layer_num = self.device_pool.layer_num
        self.v_head_dim = self.device_pool.v_head_dim
        return (
            (self.head_dim + self.v_head_dim)
            * self.head_num
            * self.layer_num
            * self.dtype.itemsize
        )

    def get_ksize_per_token(self):
        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize

    def init_kv_buffer(self):
        if self.layout == "page_first":
            k_dims = (self.size, self.layer_num, self.head_num, self.head_dim)
            v_dims = (self.size, self.layer_num, self.head_num, self.v_head_dim)
        elif self.layout == "page_first_direct":
            k_dims = (
                self.page_num,
                self.layer_num,
                self.page_size,
                self.head_num,
                self.head_dim,
            )
            v_dims = (
                self.page_num,
                self.layer_num,
                self.page_size,
                self.head_num,
                self.v_head_dim,
            )
        else:
            raise ValueError(
                f"Unsupported layout for models with head_dim != v_head_dim: "
                f"{self.layout}; expected 'page_first' or 'page_first_direct'."
            )

        # token_stride_size / layout_dim are intentionally NOT set: K and V
        # have different strides, so any caller that reaches for a single
        # shared stride is a bug. Such callers will fail loudly with
        # AttributeError rather than silently use the K stride for V copies.

        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        k_buffer = alloc_func(
            k_dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )
        v_buffer = alloc_func(
            v_dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )
        return (k_buffer, v_buffer)

    def _k_token_stride_size(self) -> int:
        return self.head_num * self.head_dim * self.dtype.itemsize

    def _v_token_stride_size(self) -> int:
        return self.head_num * self.v_head_dim * self.dtype.itemsize

    def _k_layout_dim(self) -> int:
        return self._k_token_stride_size() * self.layer_num

    def _v_layout_dim(self) -> int:
        return self._v_token_stride_size() * self.layer_num

    def _flat_page_unsupported(self) -> NotImplementedError:
        return NotImplementedError(
            "Models with head_dim != v_head_dim do not support the flat-page "
            "interface used by HiCache L3 storage backends {hf3fs, eic, nixl}. "
            "Use a backend that does not use this interface (e.g. mooncake, simm)."
        )

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
    ):
        if io_backend == "kernel":
            if self.layout != "page_first":
                raise ValueError(
                    f"Unsupported layout for models with head_dim != v_head_dim "
                    f"and io_backend='kernel': {self.layout}; expected 'page_first'."
                )
            transfer_kv_per_layer_mla_pf_lf(
                src=self.k_buffer,
                dst=device_pool.k_buffer[layer_id],
                src_indices=host_indices,
                dst_indices=device_indices,
                layer_id=layer_id,
                item_size=self._k_token_stride_size(),
                src_layout_dim=self._k_layout_dim(),
            )
            transfer_kv_per_layer_mla_pf_lf(
                src=self.v_buffer,
                dst=device_pool.v_buffer[layer_id],
                src_indices=host_indices,
                dst_indices=device_indices,
                layer_id=layer_id,
                item_size=self._v_token_stride_size(),
                src_layout_dim=self._v_layout_dim(),
            )
        elif io_backend == "direct":
            if self.layout != "page_first_direct":
                raise ValueError(
                    f"Unsupported layout for models with head_dim != v_head_dim "
                    f"and io_backend='direct': {self.layout}; expected "
                    "'page_first_direct'."
                )
            transfer_kv_per_layer_direct_pf_lf(
                src_ptrs=[self.k_buffer],
                dst_ptrs=[device_pool.k_buffer[layer_id]],
                src_indices=host_indices,
                dst_indices=device_indices,
                layer_id=layer_id,
                page_size=self.page_size,
            )
            transfer_kv_per_layer_direct_pf_lf(
                src_ptrs=[self.v_buffer],
                dst_ptrs=[device_pool.v_buffer[layer_id]],
                src_indices=host_indices,
                dst_indices=device_indices,
                layer_id=layer_id,
                page_size=self.page_size,
            )
        else:
            raise ValueError(
                f"Unsupported IO backend for models with head_dim != v_head_dim: "
                f"{io_backend}; expected 'kernel' or 'direct'."
            )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if io_backend == "kernel":
            if self.layout != "page_first":
                raise ValueError(
                    f"Unsupported layout for models with head_dim != v_head_dim "
                    f"and io_backend='kernel': {self.layout}; expected 'page_first'."
                )
            transfer_kv_all_layer_mla_lf_pf(
                src_layers=device_pool.k_data_ptrs,
                dst=self.k_buffer,
                src_indices=device_indices,
                dst_indices=host_indices,
                item_size=self._k_token_stride_size(),
                dst_layout_dim=self._k_layout_dim(),
                num_layers=self.layer_num,
            )
            transfer_kv_all_layer_mla_lf_pf(
                src_layers=device_pool.v_data_ptrs,
                dst=self.v_buffer,
                src_indices=device_indices,
                dst_indices=host_indices,
                item_size=self._v_token_stride_size(),
                dst_layout_dim=self._v_layout_dim(),
                num_layers=self.layer_num,
            )
        elif io_backend == "direct":
            if self.layout != "page_first_direct":
                raise ValueError(
                    f"Unsupported layout for models with head_dim != v_head_dim "
                    f"and io_backend='direct': {self.layout}; expected "
                    "'page_first_direct'."
                )
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=device_pool.k_buffer,
                dst_ptrs=[self.k_buffer],
                src_indices=device_indices,
                dst_indices=host_indices,
                page_size=self.page_size,
            )
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=device_pool.v_buffer,
                dst_ptrs=[self.v_buffer],
                src_indices=device_indices,
                dst_indices=host_indices,
                page_size=self.page_size,
            )
        else:
            raise ValueError(
                f"Unsupported IO backend for models with head_dim != v_head_dim: "
                f"{io_backend}; expected 'kernel' or 'direct'."
            )

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        raise self._flat_page_unsupported()

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        raise self._flat_page_unsupported()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        raise self._flat_page_unsupported()

    def get_split_heads_page_buffer_meta(
        self, indices: torch.Tensor, split_factor: int
    ):
        raise NotImplementedError(
            "get_split_heads_page_buffer_meta requires layout='page_head', "
            "which is not supported for models with head_dim != v_head_dim."
        )

    def get_page_buffer_meta(self, indices):
        assert len(indices) % self.page_size == 0
        if self.layout not in ("page_first", "page_first_direct"):
            raise ValueError(
                f"Unsupported layout for models with head_dim != v_head_dim: "
                f"{self.layout}"
            )
        indices = indices.tolist()
        k_base_ptr = self.k_buffer.data_ptr()
        v_base_ptr = self.v_buffer.data_ptr()
        k_element_size = (
            self.layer_num
            * self.dtype.itemsize
            * self.page_size
            * self.head_num
            * self.head_dim
        )
        v_element_size = (
            self.layer_num
            * self.dtype.itemsize
            * self.page_size
            * self.head_num
            * self.v_head_dim
        )
        ptr_list = []
        element_size_list = []
        if self.layout == "page_first_direct":
            k_index_stride = (
                self.layer_num * self.page_size * self.head_num * self.head_dim
            )
            v_index_stride = (
                self.layer_num * self.page_size * self.head_num * self.v_head_dim
            )
        else:
            k_index_stride = self.layer_num * self.head_num * self.head_dim
            v_index_stride = self.layer_num * self.head_num * self.v_head_dim
        for index in range(0, len(indices), self.page_size):
            buffer_index = (
                indices[index] // self.page_size
                if self.layout == "page_first_direct"
                else indices[index]
            )
            k_ptr = k_base_ptr + buffer_index * k_index_stride * self.dtype.itemsize
            v_ptr = v_base_ptr + buffer_index * v_index_stride * self.dtype.itemsize
            ptr_list.extend([k_ptr, v_ptr])
            element_size_list.extend([k_element_size, v_element_size])
        return ptr_list, element_size_list

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        if self.layout not in ("page_first", "page_first_direct"):
            return False
        k_stride = (
            self.page_size
            * self.layer_num
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        v_stride = (
            self.page_size
            * self.layer_num
            * self.head_num
            * self.v_head_dim
            * self.dtype.itemsize
        )
        base_aligned = (
            self.k_buffer.data_ptr() % page_size_bytes == 0
            and self.v_buffer.data_ptr() % page_size_bytes == 0
        )
        return (
            base_aligned
            and k_stride % page_size_bytes == 0
            and v_stride % page_size_bytes == 0
        )


def get_mha_host_pool_cls(device_pool: MHATokenToKVPool) -> type:
    """Pick the right MHA host-pool class based on the device pool's K/V dims.

    Returns ``AsymmetricMHATokenToKVPoolHost`` when ``head_dim != v_head_dim``
    (e.g. MiMo-V2), else the default ``MHATokenToKVPoolHost``.
    """
    if device_pool.head_dim != device_pool.v_head_dim:
        return AsymmetricMHATokenToKVPoolHost
    return MHATokenToKVPoolHost
