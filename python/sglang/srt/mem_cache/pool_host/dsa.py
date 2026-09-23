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
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.pool_host.base import (
    _WRITE_BACK_STAGING_PAGE_CHUNK,
    HostKVCache,
    host_memory_budget_bytes,
)
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    get_allocator_from_storage,
    make_kernel_ptr_table,
)
from sglang.srt.mem_cache.pool_host.host_pool_decl import (
    HostPoolDecl,
    HostPoolStorageInfo,
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


def dsa_indexer_bytes_per_token_per_layer(
    *, index_head_dim: int, quant_block_size: int
) -> int:
    # packed index keys plus one fp32 scale per quant block, stored as uint8
    elems = index_head_dim + index_head_dim // quant_block_size * 4
    return elems * DSATokenToKVPool.index_k_with_scale_buffer_dtype.itemsize


class DSAIndexerHostPoolBuilder:
    def validate(
        self,
        *,
        decl: HostPoolDecl,
        page_size: int,
        packed_draft_device_pools: tuple[DSATokenToKVPool, ...],
    ) -> None:
        target = decl.device_pool
        target_format = (target.index_head_dim, target.quant_block_size)
        for draft in packed_draft_device_pools:
            draft_format = (draft.index_head_dim, draft.quant_block_size)
            if draft_format != target_format:
                raise ValueError(
                    f"{decl.pool_name}: packed index key format {draft_format} "
                    f"differs from target {target_format}"
                )

    def build(
        self,
        *,
        decl: HostPoolDecl,
        anchor_host: MLATokenToKVPoolHost,
        allocator_type: str,
        packed_draft_device_pools: tuple[DSATokenToKVPool, ...],
    ) -> DSAIndexerPoolHost:
        return DSAIndexerPoolHost(
            decl=decl,
            anchor_host=anchor_host,
            packed_draft_device_pools=packed_draft_device_pools,
            allocator_type=allocator_type,
        )


def make_dsa_indexer_pool_decl(
    pool: DSATokenToKVPool, *, name: PoolName = PoolName.INDEXER
) -> HostPoolDecl:
    """Index key buffers riding on the full-KV pages: indices and layout both follow KV."""
    return HostPoolDecl(
        pool_name=name,
        device_pool=pool,
        indices_from_pool=PoolName.KV,
        layout_source=PoolName.KV,
        storage_info=HostPoolStorageInfo(
            bytes_per_token_per_layer=dsa_indexer_bytes_per_token_per_layer(
                index_head_dim=pool.index_head_dim,
                quant_block_size=pool.quant_block_size,
            ),
            dtype=DSATokenToKVPool.index_k_with_scale_buffer_dtype,
        ),
        host_pool_builder=DSAIndexerHostPoolBuilder(),
        # Shared-topk layers own a 0-row placeholder buffer; they get no host
        # layer and must not reach the transfer kernels.
        owned_device_layers=tuple(
            i for i, skip in enumerate(pool.skip_topk_layers) if not skip
        ),
    )


class DSAIndexerPoolHost(HostKVCache):
    """Host-side DSA index buffers only. Slot layout matches the anchor MLA host pool."""

    device_pool: DSATokenToKVPool

    def __init__(
        self,
        decl: HostPoolDecl,
        anchor_host: MLATokenToKVPoolHost,
        *,
        packed_draft_device_pools: tuple[DSATokenToKVPool, ...] = (),
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        is_dummy: bool = False,
    ):
        self._is_dummy = is_dummy
        self.decl = decl
        storage_info = decl.storage_info
        device_pool = decl.device_pool
        self.device_pool = device_pool
        self.page_size = anchor_host.page_size
        self.layout = anchor_host.layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)
        self.dtype = device_pool.store_dtype
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer
        # Host layers are compact: only owned device layers that hold index
        # buffers, then one tail layer per packed draft pool.
        owned_start, owned_end = self._device_owned_layer_range()
        declared = decl.owned_device_layers
        self._live_target_layers = [
            layer
            for layer in range(owned_start, owned_end)
            if declared is None or layer in declared
        ]
        self._device_to_host_layer = {
            layer: i for i, layer in enumerate(self._live_target_layers)
        }
        self.target_layer_num = len(self._live_target_layers)
        self.mtp_draft_device_pools = tuple(packed_draft_device_pools)
        self.layer_num = self.target_layer_num + len(self.mtp_draft_device_pools)

        self.indexer_dtype = storage_info.dtype
        self.size = anchor_host.size
        self.page_num = anchor_host.page_num

        # uint8 storage, so element counts below are byte counts
        self.indexer_page_stride_size = storage_info.page_bytes(self.page_size)
        self.indexer_layout_dim = self.indexer_page_stride_size * self.layer_num
        self.indexer_page_num = (self.size + self.page_size + 1) // self.page_size
        self.size_per_token = storage_info.bytes_per_token_per_layer * self.layer_num

        self.can_use_jit = False
        self.can_use_write_back_jit = False
        if is_dummy:
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

        requested_bytes = storage_info.host_bytes(
            page_num=self.page_num, layer_num=self.layer_num, page_size=self.page_size
        )
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

    def get_size_per_token(self):
        return self.decl.storage_info.bytes_per_token_per_layer * self.layer_num

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    def _is_device_layer_owned(self, device_pool, layer_id: int) -> bool:
        return layer_id in self._device_to_host_layer

    def _host_layer_index(self, layer_id: int, device_pool=None) -> int:
        return self._device_to_host_layer[layer_id]

    def _owned_device_layer_ids(self, device_pool) -> list[int]:
        return list(self._live_target_layers)

    def _draft_host_layer(self, layer_id: int) -> int:
        # The controller hands packed drafts ``target_device_layer_num + depth``.
        return self.target_layer_num + (layer_id - self.device_pool.layer_num)

    def init_kv_buffer(self):
        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        self.packed_device_index_buffers = [
            self.device_pool.index_k_with_scale_buffer[layer]
            for layer in self._live_target_layers
        ] + [
            buffer
            for pool in self.mtp_draft_device_pools
            for buffer in pool.index_k_with_scale_buffer
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
            self.index_k_data_ptrs = make_kernel_ptr_table(
                self.index_k_data_refs,
                self.device_pool.device,
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
        assert not getattr(self, "_is_dummy", False), (
            "load on a dummy (non-src DSA) host pool"
        )
        # MTP draft layers do not participate in CP layer sharding.
        host_layer_id = (
            self._draft_host_layer(layer_id)
            if is_draft
            else self._host_layer_index(layer_id)
        )
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
        assert not getattr(self, "_is_dummy", False), (
            "backup on a dummy (non-src DSA) host pool"
        )
        # MTP draft layers do not participate in CP layer sharding.
        host_layer_id = (
            self._draft_host_layer(layer_id)
            if is_draft
            else self._host_layer_index(layer_id)
        )
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
        assert not getattr(self, "_is_dummy", False), (
            "backup on a dummy (non-src DSA) host pool"
        )
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
