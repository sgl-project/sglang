from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

import torch

from sglang.kernels.ops.kvcache.hicache import (
    can_use_write_back_jit_kernel,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_all_layer_mla_staged_lf_pf as jit_transfer_hicache_all_layer_mla_staged_lf_pf,
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.pool_host.base import (
    _WRITE_BACK_STAGING_PAGE_CHUNK,
    HostKVCache,
    host_memory_budget_bytes,
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
        transfer_kv_all_layer_mla,
        transfer_kv_all_layer_mla_lf_pf,
        transfer_kv_direct,
        transfer_kv_per_layer_direct_pf_lf,
        transfer_kv_per_layer_mla,
        transfer_kv_per_layer_mla_pf_lf,
    )

logger = logging.getLogger(__name__)


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
        self.target_layer_num = self._effective_host_layer_num()
        self.mtp_draft_device_pools = anchor_host.mtp_draft_device_pools
        self.layer_num = self.target_layer_num + len(self.mtp_draft_device_pools)

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
        available_bytes = host_memory_budget_bytes()
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory for DSA indexer hierarchical cache. "
                f"Requesting {requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free."
            )
        draft_layer_num = self.layer_num - self.target_layer_num
        if draft_layer_num > 0:
            logger.info(
                "Allocating %.2f GB host memory for DSA indexer (layout=%s), "
                "packed MTP layers: "
                "target_layers=%d, draft_layers=%d, total_layers=%d.",
                requested_bytes / 1e9,
                layout,
                self.target_layer_num,
                draft_layer_num,
                self.layer_num,
            )
        else:
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
        device_pools = (self.device_pool, *self.mtp_draft_device_pools)
        self.packed_device_index_buffers = [
            buffer for pool in device_pools for buffer in pool.index_k_with_scale_buffer
        ]
        self.index_k_device_ptrs = torch.tensor(
            [x.data_ptr() for x in self.packed_device_index_buffers],
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
                registration_granularity_bytes=self.indexer_layout_dim,
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
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        if not is_draft and not self._is_device_layer_owned(device_pool, layer_id):
            return
        # MTP draft layers do not participate in CP layer sharding.
        host_layer_id = layer_id if is_draft else self._host_layer_index(layer_id)
        device_layer_id = 0 if is_draft else layer_id

        host_page_indices, device_page_indices = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
        if use_kernel:
            if self.layout == "layer_first":
                transfer_kv_per_layer_mla(
                    src=self.index_k_with_scale_buffer[host_layer_id],
                    dst=device_pool.index_k_with_scale_buffer[device_layer_id],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    item_size=self.indexer_page_stride_size,
                )
            elif self.layout == "page_first":
                transfer_kv_per_layer_mla_pf_lf(
                    src=self.index_k_with_scale_buffer,
                    dst=device_pool.index_k_with_scale_buffer[device_layer_id],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    layer_id=host_layer_id,
                    item_size=self.indexer_page_stride_size,
                    src_layout_dim=self.indexer_layout_dim,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.index_k_with_scale_buffer[host_layer_id]],
                    dst_layers=[device_pool.index_k_with_scale_buffer[device_layer_id]],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    page_size=1,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.index_k_with_scale_buffer],
                    dst_ptrs=[device_pool.index_k_with_scale_buffer[device_layer_id]],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    layer_id=host_layer_id,
                    page_size=1,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def _backup_from_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        # MTP draft layers do not participate in CP layer sharding.
        host_layer_id = layer_id if is_draft else self._host_layer_index(layer_id)
        device_layer_id = 0 if is_draft else layer_id

        host_page_indices, device_page_indices = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
        if use_kernel:
            if self.layout == "layer_first":
                transfer_kv_per_layer_mla(
                    src=device_pool.index_k_with_scale_buffer[device_layer_id],
                    dst=self.index_k_with_scale_buffer[host_layer_id],
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
                    src_layers=[device_pool.index_k_with_scale_buffer[device_layer_id]],
                    dst_layers=[self.index_k_with_scale_buffer[host_layer_id]],
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
            for draft_layer_id, draft_device_pool in enumerate(
                self.mtp_draft_device_pools
            ):
                self._backup_from_device_per_layer(
                    draft_device_pool,
                    host_indices,
                    device_indices,
                    self.device_pool.layer_num + draft_layer_id,
                    io_backend,
                    is_draft=True,
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
                    src_layers=self.packed_device_index_buffers,
                    dst_layers=self.index_k_data_refs,
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    page_size=1,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=self.packed_device_index_buffers,
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
