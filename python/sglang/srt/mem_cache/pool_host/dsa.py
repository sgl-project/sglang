from __future__ import annotations

import logging
import threading

import torch

from sglang.kernels.ops.kvcache.hicache import (
    can_use_write_back_jit_kernel,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_all_layer_mla_staged_lf_pf as jit_transfer_hicache_all_layer_mla_staged_lf_pf,
)
from sglang.srt.mem_cache.device_pool_info import IndexKeyBufferInfo
from sglang.srt.mem_cache.pool_host._kvcacheio import (
    MLA_KVCACHEIO_OPS,
    require_kvcacheio_ops,
)
from sglang.srt.mem_cache.pool_host.base import (
    _WRITE_BACK_STAGING_PAGE_CHUNK,
    HostKVCache,
    host_memory_budget_bytes,
)
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    _cuda_host_unregister,
    get_allocator_from_storage,
    make_kernel_ptr_table,
)
from sglang.srt.platforms import current_platform
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
elif _is_xpu:
    from sglang.srt.mem_cache.pool_host._kvcacheio import (
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
    """Host index pages with precomputed device-layer to compact-host-layer binding."""

    def __init__(
        self,
        buffer_info: IndexKeyBufferInfo,
        *,
        layer_mapping: dict[int, int],
        target_device_layer_num: int,
        num_host_pages: int,
        transfer_page_size: int,
        layout: str,
        buffer_device: str,
        dtype: torch.dtype,
        layer_sharded: bool = False,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        is_dummy: bool = False,
    ):
        buffers = buffer_info.buffers
        if buffers:
            buffer_info.validate()
        if len(layer_mapping) != len(buffers) or set(layer_mapping.values()) != set(
            range(len(buffers))
        ):
            raise ValueError(
                f"index layer_mapping={layer_mapping} must cover {len(buffers)} buffers"
            )
        if (
            target_device_layer_num < 1
            or num_host_pages < 1
            or any(layer < 0 for layer in layer_mapping)
        ):
            raise ValueError(
                f"invalid index host geometry: target_device_layer_num={target_device_layer_num}, num_host_pages={num_host_pages}, layer_mapping={layer_mapping}"
            )
        if _is_xpu and not is_dummy:
            require_kvcacheio_ops(
                pool=type(self).__name__,
                layout=layout,
                ops=self._kvcacheio_ops,
            )
        self._is_dummy = is_dummy
        self._destroyed = False
        self.device_pool = None
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)
        self._layer_sharded = layer_sharded
        self._buffer_device = torch.device(buffer_device)
        self.page_size = transfer_page_size
        self.layout = layout
        self.dtype = dtype
        self.indexer_dtype = torch.uint8
        self.layer_num = len(buffers)
        self.start_layer = 0
        self.end_layer = target_device_layer_num
        self._device_to_host_layer = dict(layer_mapping)
        self._live_target_layers = [
            layer for layer in layer_mapping if layer < target_device_layer_num
        ]
        self._draft_device_layer_ids = [
            layer for layer in layer_mapping if layer >= target_device_layer_num
        ]
        self.target_layer_num = len(self._live_target_layers)
        self.packed_device_index_buffers = list(buffers)
        self.page_num = num_host_pages
        self.size = num_host_pages * self.page_size
        self.indexer_page_stride_size = (
            buffer_info.kernel_page_bytes // buffer_info.page_size
        ) * transfer_page_size
        self._initialize_host_buffers()

    def _initialize_host_buffers(self):
        if self.indexer_page_stride_size % self.page_size:
            raise ValueError("DSA index page bytes must split into whole token bytes")
        self.indexer_layout_dim = self.indexer_page_stride_size * self.layer_num
        self.indexer_page_num = (self.size + self.page_size + 1) // self.page_size
        self.size_per_token = self.indexer_layout_dim // self.page_size
        self.can_use_jit = False
        self.can_use_write_back_jit = False
        if self._is_dummy:
            self.index_k_with_scale_buffer = None
            self.index_k_device_ptrs = None
            logger.info(
                "DSAIndexerPoolHost dummy mode: allocator-only, size=%d tokens, "
                "skipping indexer buffer allocation",
                self.size,
            )
            self.lock = threading.RLock()
            self.clear()
            return

        requested_bytes = self.size * self.size_per_token
        available_bytes = host_memory_budget_bytes(requested_bytes)
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
                self.layout,
                self.target_layer_num,
                draft_layer_num,
                self.layer_num,
            )
        else:
            logger.info(
                "Allocating %.2f GB host memory for DSA indexer (layout=%s).",
                requested_bytes / 1e9,
                self.layout,
            )
        self.init_kv_buffer()
        self._init_write_back_staging_buffers()
        self.lock = threading.RLock()
        self.clear()

    def destroy(self):
        if self._destroyed:
            return
        if not self._is_dummy and self.pin_memory and (_is_cuda or _is_hip):
            _cuda_host_unregister(self.index_k_with_scale_buffer)
        self.index_k_with_scale_buffer = None
        self.index_k_data_refs = []
        super().destroy()

    def _is_device_layer_sharded(self, device_pool=None) -> bool:
        return self._layer_sharded

    def get_size_per_token(self):
        return self.size_per_token

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    def _is_device_layer_owned(self, device_pool, layer_id: int) -> bool:
        return layer_id in self._device_to_host_layer

    def _host_layer_index(self, layer_id: int, device_pool=None) -> int:
        return self._device_to_host_layer[layer_id]

    def _owned_device_layer_ids(self, device_pool) -> list[int]:
        return list(self._live_target_layers)

    def _draft_host_layer(self, layer_id: int) -> int:
        return self._device_to_host_layer[layer_id]

    def init_kv_buffer(self):
        alloc_func = ALLOC_MEMORY_FUNCS[self._buffer_device.type]
        self.index_k_device_ptrs = torch.tensor(
            [x.data_ptr() for x in self.packed_device_index_buffers],
            dtype=torch.uint64,
            device=self._buffer_device,
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
            self.index_k_data_ptrs = make_kernel_ptr_table(
                self.index_k_data_refs,
                self._buffer_device,
                host_memory_registered=self.pin_memory,
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
        if (
            self.layout != "page_first"
            or not current_platform.capabilities.hicache_device_kernels
        ):
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
            device=self._buffer_device,
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

    if _is_xpu:
        # {io_backend: {layout: kvcacheio ops}} the transfer methods below call,
        # checked at construction; keep in step with their branches.
        _kvcacheio_ops = MLA_KVCACHEIO_OPS

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
        assert not getattr(self, "_is_dummy", False), (
            "load on a dummy (non-src DSA) host pool"
        )
        # MTP draft layers do not participate in CP layer sharding.
        host_layer_id = (
            self._draft_host_layer(layer_id)
            if is_draft
            else self._host_layer_index(layer_id)
        )

        host_page_indices, device_page_indices = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
        if use_kernel:
            if self.layout == "layer_first":
                transfer_kv_per_layer_mla(
                    src=self.index_k_with_scale_buffer[host_layer_id],
                    dst=self.packed_device_index_buffers[host_layer_id],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    item_size=self.indexer_page_stride_size,
                )
            elif self.layout == "page_first":
                transfer_kv_per_layer_mla_pf_lf(
                    src=self.index_k_with_scale_buffer,
                    dst=self.packed_device_index_buffers[host_layer_id],
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
                    dst_layers=[self.packed_device_index_buffers[host_layer_id]],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    page_size=1,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.index_k_with_scale_buffer],
                    dst_ptrs=[self.packed_device_index_buffers[host_layer_id]],
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
        assert not getattr(self, "_is_dummy", False), (
            "backup on a dummy (non-src DSA) host pool"
        )
        # MTP draft layers do not participate in CP layer sharding.
        host_layer_id = (
            self._draft_host_layer(layer_id)
            if is_draft
            else self._host_layer_index(layer_id)
        )

        host_page_indices, device_page_indices = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
        if use_kernel:
            if self.layout == "layer_first":
                transfer_kv_per_layer_mla(
                    src=self.packed_device_index_buffers[host_layer_id],
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
                    src_layers=[self.packed_device_index_buffers[host_layer_id]],
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
        assert not getattr(self, "_is_dummy", False), (
            "backup on a dummy (non-src DSA) host pool"
        )
        if self._is_device_layer_sharded(device_pool):
            for layer_id in self._owned_device_layer_ids(device_pool):
                self._backup_from_device_per_layer(
                    device_pool, host_indices, device_indices, layer_id, io_backend
                )
            for draft_layer_id in self._draft_device_layer_ids:
                self._backup_from_device_per_layer(
                    None,
                    host_indices,
                    device_indices,
                    draft_layer_id,
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
