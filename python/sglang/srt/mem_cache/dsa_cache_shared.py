# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Page-sharded DSA KV and indexer cache for context-parallel prefill."""

from __future__ import annotations

import ctypes
import logging
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.kernels.ops.attention.dsa import index_buf_accessor
from sglang.kernels.ops.kvcache.dsa_shared import set_mla_kv_buffer_owner_triton
from sglang.srt.distributed.device_communicators.vmm_utils import (
    _get_cuda_driver,
    all_ranks_ok,
    check_drv,
    exchange_posix_fds,
    export_shareable_handles,
    import_peer_handle,
    make_rw_access_desc,
)
from sglang.srt.mem_cache.memory_pool import (
    GPU_MEMORY_TYPE_KV_CACHE,
    DSATokenToKVPool,
    RadixAttention,
    maybe_detect_oob,
)

logger = logging.getLogger(__name__)

_shared_vmm_use_fabric: Optional[bool] = None


def _validate_same_host_group(cpu_group: ProcessGroup) -> None:
    hosts = [None] * dist.get_world_size(group=cpu_group)
    dist.all_gather_object(hosts, os.uname().nodename, group=cpu_group)
    if len(set(hosts)) != 1:
        raise ValueError(
            "DSA shared KV cache requires every attention CP rank on the same host."
        )


def _export_dsa_shareable_handles(retained_handles, group, rank):
    global _shared_vmm_use_fabric
    first_attempt = _shared_vmm_use_fabric is None
    result = export_shareable_handles(
        retained_handles,
        group,
        rank,
        try_fabric=_shared_vmm_use_fabric is not False,
        log_fallback=first_attempt and rank == 0,
    )
    _shared_vmm_use_fabric = result[2]
    return result


def _shareable_allocation_handle_types(drv):
    handle_types = drv.CUmemAllocationHandleType
    posix_fd = handle_types.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    if _shared_vmm_use_fabric is False:
        return posix_fd
    return posix_fd | handle_types.CU_MEM_HANDLE_TYPE_FABRIC


def _synchronize_vmm_stage(
    cpu_group: ProcessGroup,
    rank: int,
    stage: str,
    local_error: Optional[BaseException],
) -> None:
    errors = [None] * dist.get_world_size(group=cpu_group)
    dist.all_gather_object(
        errors,
        None if local_error is None else str(local_error),
        group=cpu_group,
    )
    for failed_rank, error in enumerate(errors):
        if error is not None:
            message = f"DSA shared VMM {stage} failed on rank {failed_rank}: {error}"
            if failed_rank == rank:
                raise RuntimeError(message) from local_error
            raise RuntimeError(message)


def _release_partial_vmm_mapping(
    drv,
    *,
    base_va: Optional[int],
    total_bytes: int,
    mapped_addresses: list[int],
    segment_bytes: int,
) -> None:
    while mapped_addresses:
        drv.cuMemUnmap(mapped_addresses.pop(), segment_bytes)
    if base_va is not None:
        drv.cuMemAddressFree(base_va, total_bytes)


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DELETER_FN = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DELETER_FN),
]

# PyTorch storage may outlive the pool through a temporary tensor view. Keep the
# tiny DLPack callback objects alive until process exit after releasing VMM.
_CLOSED_DLPACK_REFS: list[list] = []


def _dtype_to_dlpack(dtype: torch.dtype) -> tuple[int, int]:
    mapping = {
        torch.uint8: (1, 8),
        torch.int8: (0, 8),
        torch.int32: (0, 32),
        torch.float16: (2, 16),
        torch.bfloat16: (4, 16),
        torch.float32: (2, 32),
    }
    if hasattr(torch, "float8_e4m3fn"):
        mapping[torch.float8_e4m3fn] = (2, 8)
    if dtype not in mapping:
        raise TypeError(f"Unsupported DSA shared VMM dtype: {dtype}")
    return mapping[dtype]


def _tensor_from_cuda_ptr(
    ptr: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device_id: int,
    refs: list,
) -> torch.Tensor:
    dl_code, dl_bits = _dtype_to_dlpack(dtype)
    shape_array = (ctypes.c_int64 * len(shape))(*shape)
    managed = _DLManagedTensor()
    managed.dl_tensor.data = ctypes.c_void_p(ptr)
    managed.dl_tensor.device = _DLDevice(2, device_id)
    managed.dl_tensor.ndim = len(shape)
    managed.dl_tensor.dtype = _DLDataType(dl_code, dl_bits, 1)
    managed.dl_tensor.shape = shape_array
    managed.dl_tensor.strides = None
    managed.dl_tensor.byte_offset = 0
    managed.manager_ctx = None

    @_DELETER_FN
    def deleter(_):
        return None

    managed.deleter = deleter
    refs.extend([managed, shape_array, deleter])
    ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
    ctypes.pythonapi.PyCapsule_New.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]
    capsule = ctypes.pythonapi.PyCapsule_New(ctypes.byref(managed), b"dltensor", None)
    return torch.from_dlpack(capsule)


def _align_first_dim(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    granularity: int,
    first_dim_multiple: int,
) -> tuple[int, int]:
    row_bytes = math.prod(shape[1:]) * torch.empty((), dtype=dtype).element_size()
    rows_per_granularity = granularity // math.gcd(granularity, row_bytes)
    row_alignment = math.lcm(rows_per_granularity, first_dim_multiple)
    rows = ((shape[0] + row_alignment - 1) // row_alignment) * row_alignment
    return rows, rows * row_bytes


@dataclass
class RankMajorSharedTensor:
    global_view: torch.Tensor
    rank_local_view: Optional[torch.Tensor]
    local_view: torch.Tensor
    local_rows: int
    aligned_bytes_per_rank: int
    _base_va: int
    _rank_local_base_va: Optional[int]
    _total_bytes: int
    _world_size: int
    _refs: list
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        torch.cuda.synchronize()
        del self.local_view
        if self.rank_local_view is not None:
            del self.rank_local_view
        del self.global_view
        drv = _get_cuda_driver()
        base_vas = [self._base_va]
        if self._rank_local_base_va is not None:
            base_vas.append(self._rank_local_base_va)
        for base_va in base_vas:
            for segment in range(self._world_size):
                address = base_va + segment * self.aligned_bytes_per_rank
                check_drv(
                    drv.cuMemUnmap(address, self.aligned_bytes_per_rank),
                    f"cuMemUnmap(segment={segment})",
                )
            check_drv(
                drv.cuMemAddressFree(base_va, self._total_bytes),
                "cuMemAddressFree",
            )
        _CLOSED_DLPACK_REFS.append(self._refs)


@dataclass
class RankMajorSharedSlab:
    allocation: RankMajorSharedTensor
    layer_rows: int
    global_views: list[torch.Tensor]
    rank_local_views: list[torch.Tensor]
    local_views: list[torch.Tensor]

    @property
    def rank_stride_rows(self) -> int:
        return self.allocation.local_rows

    def close(self) -> None:
        self.global_views.clear()
        self.rank_local_views.clear()
        self.local_views.clear()
        self.allocation.close()


def create_rank_major_shared_tensor(
    local_shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    cpu_group: ProcessGroup,
    first_dim_multiple: int = 1,
    map_rank_local: bool = True,
) -> RankMajorSharedTensor:
    global _shared_vmm_use_fabric

    _validate_same_host_group(cpu_group)
    drv = _get_cuda_driver()
    rank = dist.get_rank(group=cpu_group)
    world_size = dist.get_world_size(group=cpu_group)
    device_id = torch.cuda.current_device()
    posix_fd = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR

    prop = drv.CUmemAllocationProp()
    prop.requestedHandleTypes = _shareable_allocation_handle_types(drv)
    prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location = drv.CUmemLocation()
    prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id
    if hasattr(prop, "allocFlags") and hasattr(prop.allocFlags, "gpuDirectRDMACapable"):
        prop.allocFlags.gpuDirectRDMACapable = 1

    granularity = int(
        check_drv(
            drv.cuMemGetAllocationGranularity(
                prop,
                drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            ),
            "cuMemGetAllocationGranularity",
        )
    )
    local_rows, aligned_bytes = _align_first_dim(
        local_shape,
        dtype=dtype,
        granularity=granularity,
        first_dim_multiple=first_dim_multiple,
    )
    err, local_handle = drv.cuMemCreate(aligned_bytes, prop, 0)
    if not all_ranks_ok(cpu_group, err == drv.CUresult.CUDA_SUCCESS):
        if err == drv.CUresult.CUDA_SUCCESS:
            drv.cuMemRelease(local_handle)
        _shared_vmm_use_fabric = False
        prop.requestedHandleTypes = posix_fd
        err, local_handle = drv.cuMemCreate(aligned_bytes, prop, 0)
        create_error = (
            None
            if err == drv.CUresult.CUDA_SUCCESS
            else RuntimeError(f"cuMemCreate(POSIX_FD): {err}")
        )
        try:
            _synchronize_vmm_stage(cpu_group, rank, "allocation", create_error)
        except BaseException:
            if err == drv.CUresult.CUDA_SUCCESS:
                drv.cuMemRelease(local_handle)
            raise
    elif err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuMemCreate(FABRIC): {err}")

    local_fds: list[int] = []
    peer_fds: dict[tuple[int, int], int] = {}
    imported_handles = []
    base_va = None
    rank_local_base_va = None
    mapped_global_addresses: list[int] = []
    mapped_local_addresses: list[int] = []
    total_bytes = aligned_bytes * world_size
    try:
        fabric_handles, local_fds, use_fabric = _export_dsa_shareable_handles(
            [local_handle], cpu_group, rank
        )
        gathered_handles = [None] * world_size
        if use_fabric:
            dist.all_gather_object(gathered_handles, fabric_handles[0], group=cpu_group)
        else:
            peer_fds = exchange_posix_fds(
                cpu_group, rank, world_size, local_fds, [1] * world_size
            )

        mapping_error = None
        try:
            base_va = int(
                check_drv(
                    drv.cuMemAddressReserve(total_bytes, granularity, 0, 0),
                    "cuMemAddressReserve",
                )
            )
            if map_rank_local:
                rank_local_base_va = int(
                    check_drv(
                        drv.cuMemAddressReserve(total_bytes, granularity, 0, 0),
                        "cuMemAddressReserve(rank_local)",
                    )
                )
            for peer_rank in range(world_size):
                handle = local_handle
                if peer_rank != rank:
                    handle = import_peer_handle(
                        gathered_handles[peer_rank],
                        None if use_fabric else peer_fds[(peer_rank, 0)],
                        use_fabric=use_fabric,
                        peer_rank=peer_rank,
                    )
                    imported_handles.append(handle)
                address = base_va + peer_rank * aligned_bytes
                check_drv(
                    drv.cuMemMap(address, aligned_bytes, 0, handle, 0),
                    f"cuMemMap(rank={peer_rank})",
                )
                mapped_global_addresses.append(address)
                check_drv(
                    drv.cuMemSetAccess(
                        address, aligned_bytes, [make_rw_access_desc(device_id)], 1
                    ),
                    f"cuMemSetAccess(rank={peer_rank})",
                )

                if map_rank_local:
                    local_segment = (peer_rank - rank) % world_size
                    local_address = rank_local_base_va + local_segment * aligned_bytes
                    check_drv(
                        drv.cuMemMap(local_address, aligned_bytes, 0, handle, 0),
                        f"cuMemMap(rank_local={peer_rank})",
                    )
                    mapped_local_addresses.append(local_address)
                    check_drv(
                        drv.cuMemSetAccess(
                            local_address,
                            aligned_bytes,
                            [make_rw_access_desc(device_id)],
                            1,
                        ),
                        f"cuMemSetAccess(rank_local={peer_rank})",
                    )
        except BaseException as e:
            mapping_error = e

        _synchronize_vmm_stage(cpu_group, rank, "mapping", mapping_error)
        for handle in imported_handles:
            check_drv(drv.cuMemRelease(handle), "cuMemRelease(peer)")
        imported_handles.clear()
        check_drv(drv.cuMemRelease(local_handle), "cuMemRelease(local)")
        local_handle = None

        refs = []
        global_view = _tensor_from_cuda_ptr(
            base_va,
            (world_size * local_rows, *local_shape[1:]),
            dtype,
            device_id,
            refs,
        )
        rank_local_view = None
        if rank_local_base_va is not None:
            rank_local_view = _tensor_from_cuda_ptr(
                rank_local_base_va,
                (world_size * local_rows, *local_shape[1:]),
                dtype,
                device_id,
                refs,
            )
        local_view = (
            rank_local_view.narrow(0, 0, local_rows)
            if rank_local_view is not None
            else global_view.narrow(0, rank * local_rows, local_rows)
        )
        return RankMajorSharedTensor(
            global_view=global_view,
            rank_local_view=rank_local_view,
            local_view=local_view,
            local_rows=local_rows,
            aligned_bytes_per_rank=aligned_bytes,
            _base_va=base_va,
            _rank_local_base_va=rank_local_base_va,
            _total_bytes=total_bytes,
            _world_size=world_size,
            _refs=refs,
        )
    except BaseException:
        _release_partial_vmm_mapping(
            drv,
            base_va=base_va,
            total_bytes=total_bytes,
            mapped_addresses=mapped_global_addresses,
            segment_bytes=aligned_bytes,
        )
        _release_partial_vmm_mapping(
            drv,
            base_va=rank_local_base_va,
            total_bytes=total_bytes,
            mapped_addresses=mapped_local_addresses,
            segment_bytes=aligned_bytes,
        )
        for handle in imported_handles:
            drv.cuMemRelease(handle)
        if local_handle is not None:
            drv.cuMemRelease(local_handle)
        raise
    finally:
        for fd in local_fds:
            os.close(fd)
        for fd in peer_fds.values():
            os.close(fd)


def create_rank_major_shared_slab(
    layer_shape: tuple[int, ...],
    *,
    layer_num: int,
    dtype: torch.dtype,
    cpu_group: ProcessGroup,
    first_dim_multiple: int = 1,
    map_rank_local: bool = True,
) -> RankMajorSharedSlab:
    layer_rows = layer_shape[0]
    allocation = create_rank_major_shared_tensor(
        (layer_num * layer_rows, *layer_shape[1:]),
        dtype=dtype,
        cpu_group=cpu_group,
        first_dim_multiple=first_dim_multiple,
        map_rank_local=map_rank_local,
    )
    rank_stride_rows = allocation.local_rows
    global_span = (allocation._world_size - 1) * rank_stride_rows + layer_rows

    def layer_views(base: torch.Tensor) -> list[torch.Tensor]:
        return [
            base.narrow(0, layer_id * layer_rows, global_span)
            for layer_id in range(layer_num)
        ]

    local_views = [
        allocation.local_view.narrow(0, layer_id * layer_rows, layer_rows)
        for layer_id in range(layer_num)
    ]
    return RankMajorSharedSlab(
        allocation=allocation,
        layer_rows=layer_rows,
        global_views=layer_views(allocation.global_view),
        rank_local_views=(
            layer_views(allocation.rank_local_view)
            if allocation.rank_local_view is not None
            else []
        ),
        local_views=local_views,
    )


@dataclass(frozen=True)
class SharedDSAPageLayout:
    cp_size: int
    page_size: int
    pages_per_rank: int
    local_pages_per_layer: Optional[int] = None
    padding_value: int = -1

    def translate_pages(self, page_indices: torch.Tensor) -> torch.Tensor:
        valid = page_indices != self.padding_value
        safe_pages = torch.where(valid, page_indices, 0)
        owner = safe_pages % self.cp_size
        local_page = torch.div(safe_pages, self.cp_size, rounding_mode="floor")
        shared_page = owner * self.pages_per_rank + local_page
        return torch.where(valid, shared_page, page_indices)

    def translate_pages_for_rank(
        self, page_indices: torch.Tensor, *, rank: int
    ) -> torch.Tensor:
        valid = page_indices != self.padding_value
        safe_pages = torch.where(valid, page_indices, 0)
        owner = safe_pages % self.cp_size
        local_page = torch.div(safe_pages, self.cp_size, rounding_mode="floor")
        rank_local_owner = (owner - rank) % self.cp_size
        shared_page = rank_local_owner * self.pages_per_rank + local_page
        return torch.where(valid, shared_page, page_indices)

    def translate_slots(self, slot_indices: torch.Tensor) -> torch.Tensor:
        valid = slot_indices != self.padding_value
        safe_slots = torch.where(valid, slot_indices, 0)
        logical_page = torch.div(safe_slots, self.page_size, rounding_mode="floor")
        page_offset = safe_slots % self.page_size
        shared_slot = self.translate_pages(logical_page) * self.page_size + page_offset
        return torch.where(valid, shared_slot, slot_indices)

    def translate_slots_for_rank(
        self, slot_indices: torch.Tensor, *, rank: int
    ) -> torch.Tensor:
        valid = slot_indices != self.padding_value
        safe_slots = torch.where(valid, slot_indices, 0)
        logical_page = torch.div(safe_slots, self.page_size, rounding_mode="floor")
        page_offset = safe_slots % self.page_size
        shared_slot = (
            self.translate_pages_for_rank(logical_page, rank=rank) * self.page_size
            + page_offset
        )
        return torch.where(valid, shared_slot, slot_indices)

    def translate_local_slots(self, slot_indices: torch.Tensor) -> torch.Tensor:
        valid = slot_indices != self.padding_value
        safe_slots = torch.where(valid, slot_indices, 0)
        logical_page = torch.div(safe_slots, self.page_size, rounding_mode="floor")
        page_offset = safe_slots % self.page_size
        local_page = torch.div(logical_page, self.cp_size, rounding_mode="floor")
        local_slot = local_page * self.page_size + page_offset
        return torch.where(valid, local_slot, slot_indices)

    def owned_slot_mask(
        self, slot_indices: torch.Tensor, *, owner_rank: int
    ) -> torch.Tensor:
        valid = slot_indices >= 0
        safe_slots = torch.where(valid, slot_indices, 0)
        logical_page = torch.div(safe_slots, self.page_size, rounding_mode="floor")
        return valid & ((logical_page % self.cp_size) == owner_rank)


class SharedDSATokenToKVPool(DSATokenToKVPool):
    """DSA pool with page-sharded Main KV and indexer cache across CP ranks."""

    def __init__(
        self,
        *args,
        shared_rank: int,
        shared_size: int,
        **kwargs,
    ):
        assert shared_size > 1
        self.shared_rank = shared_rank
        self.shared_size = shared_size
        self.shared_cp_group = None
        self.main_layout: Optional[SharedDSAPageLayout] = None
        self.index_layout: Optional[SharedDSAPageLayout] = None
        self.shared_kv_slab: Optional[RankMajorSharedSlab] = None
        self.shared_index_slab: Optional[RankMajorSharedSlab] = None
        self.local_kv_buffer: list[torch.Tensor] = []
        self.local_index_k_with_scale_buffer: list[torch.Tensor] = []
        self.rank_local_index_k_with_scale_buffer: list[torch.Tensor] = []
        self.shared_write_fence = None
        super().__init__(*args, **kwargs)

    def _get_cp_group(self):
        if self.shared_cp_group is None:
            from sglang.srt.runtime_context import get_parallel

            self.shared_cp_group = get_parallel().attn_cp_group
        return self.shared_cp_group

    def _create_buffers(self) -> None:
        logical_pages = (self.size + 2 * self.page_size - 1) // self.page_size
        requested_pages = (logical_pages + self.shared_size - 1) // self.shared_size + 1
        local_shape = (requested_pages * self.page_size, 1, self.kv_cache_dim)
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.shared_kv_slab = create_rank_major_shared_slab(
                local_shape,
                layer_num=self.layer_num,
                dtype=self.store_dtype,
                cpu_group=self._get_cp_group().cpu_group,
                first_dim_multiple=self.page_size,
                map_rank_local=False,
            )
        self.kv_buffer = self.shared_kv_slab.global_views
        self.local_kv_buffer = self.shared_kv_slab.local_views
        pages_per_rank = self.shared_kv_slab.rank_stride_rows // self.page_size
        self.main_layout = SharedDSAPageLayout(
            cp_size=self.shared_size,
            page_size=self.page_size,
            pages_per_rank=pages_per_rank,
            local_pages_per_layer=requested_pages,
        )
        self.shared_write_fence = torch.ones(
            (1,), dtype=torch.int32, device=self.device
        )
        logger.info(
            "DSA shared Main KV: rank=%s size=%s pages_per_layer=%s "
            "rank_stride_pages=%s",
            self.shared_rank,
            self.shared_size,
            requested_pages,
            pages_per_rank,
        )

    def _create_index_buffers(self) -> None:
        logical_pages = (self.index_buf_size + self.page_size + 1) // self.page_size
        requested_pages = (logical_pages + self.shared_size - 1) // self.shared_size + 1
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            # DeepGEMM resolves the cache device from its base pointer, so the
            # paged read alias maps this rank's segment first over the same HBM.
            self.shared_index_slab = create_rank_major_shared_slab(
                self._index_buffer_shape(requested_pages),
                layer_num=self.layer_num,
                dtype=self.index_k_with_scale_buffer_dtype,
                cpu_group=self._get_cp_group().cpu_group,
            )
        self.index_k_with_scale_buffer = self.shared_index_slab.global_views
        self.local_index_k_with_scale_buffer = self.shared_index_slab.local_views
        self.rank_local_index_k_with_scale_buffer = (
            self.shared_index_slab.rank_local_views
        )
        self.index_layout = SharedDSAPageLayout(
            cp_size=self.shared_size,
            page_size=self.page_size,
            pages_per_rank=self.shared_index_slab.rank_stride_rows,
            local_pages_per_layer=requested_pages,
        )
        logger.info(
            "DSA shared Indexer: rank=%s size=%s pages_per_layer=%s "
            "rank_stride_pages=%s",
            self.shared_rank,
            self.shared_size,
            requested_pages,
            self.index_layout.pages_per_rank,
        )

    def _clear_buffers(self) -> None:
        kv_slab = self.shared_kv_slab
        index_slab = self.shared_index_slab
        self.kv_buffer = []
        self.local_kv_buffer = []
        self.index_k_with_scale_buffer = []
        self.local_index_k_with_scale_buffer = []
        self.rank_local_index_k_with_scale_buffer = []
        self.shared_kv_slab = None
        self.shared_index_slab = None
        if kv_slab is not None:
            kv_slab.close()
        if index_slab is not None:
            index_slab.close()
        self.shared_write_fence = None

    def translate_index_pages(self, page_indices: torch.Tensor) -> torch.Tensor:
        assert self.index_layout is not None
        return self.index_layout.translate_pages(page_indices)

    def translate_index_slots(self, slot_indices: torch.Tensor) -> torch.Tensor:
        assert self.index_layout is not None
        return self.index_layout.translate_slots(slot_indices)

    def prepare_paged_index_page_table(self, page_table: torch.Tensor) -> torch.Tensor:
        assert self.index_layout is not None
        return self.index_layout.translate_pages_for_rank(
            page_table, rank=self.shared_rank
        ).to(torch.int32)

    def translate_main_slots(self, slot_indices: torch.Tensor) -> torch.Tensor:
        assert self.main_layout is not None
        return self.main_layout.translate_slots(slot_indices)

    def synchronize_shared_writes(self) -> None:
        assert self.shared_write_fence is not None
        self.shared_write_fence.fill_(1)
        cp_group = self.shared_cp_group or self._get_cp_group()
        cp_group._all_reduce_in_place(self.shared_write_fence)

    def get_kv_size_bytes(self) -> int:
        assert self.shared_kv_slab is not None
        assert self.shared_index_slab is not None
        return (
            self.shared_kv_slab.allocation.aligned_bytes_per_rank
            + self.shared_index_slab.allocation.aligned_bytes_per_rank
        )

    def get_contiguous_buf_infos(self):
        buffers = self.local_kv_buffer
        data_ptrs = [buf.data_ptr() for buf in buffers]
        data_lens = [buf.nbytes for buf in buffers]
        item_lens = [buf[0].nbytes * self.page_size for buf in buffers]
        return data_ptrs, data_lens, item_lens

    def get_state_buf_infos(self):
        buffers = self.local_index_k_with_scale_buffer
        data_ptrs = [buf.data_ptr() for buf in buffers]
        data_lens = [buf.nbytes for buf in buffers]
        item_lens = [buf[0].nbytes for buf in buffers]
        return data_ptrs, data_lens, item_lens

    def get_pd_transfer_tensors(self):
        return self.local_kv_buffer, [self.local_index_k_with_scale_buffer]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.get_key_buffer(layer_id)[..., : self.kv_lora_rank]

    def _write_owned_mla_kv_buffer(
        self,
        kv_buffer: torch.Tensor,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ) -> None:
        set_mla_kv_buffer_owner_triton(
            kv_buffer,
            loc,
            cache_k_nope,
            cache_k_rope,
            owner_rank=self.shared_rank,
            owner_size=self.shared_size,
            page_size=self.page_size,
        )

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ) -> None:
        maybe_detect_oob(
            loc, 0, self.size + self.page_size, "set_mla_kv_buffer (DSA shared)"
        )
        self._write_mla_kv_buffer(
            self.local_kv_buffer[layer.layer_id - self.start_layer],
            loc,
            cache_k_nope,
            cache_k_rope,
            write_fn=self._write_owned_mla_kv_buffer,
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
        size_limit = self.size + self.page_size
        maybe_detect_oob(tgt_loc, 0, size_limit, "move_kv_cache tgt_loc")
        maybe_detect_oob(src_loc, 0, size_limit, "move_kv_cache src_loc")
        if tgt_loc.numel() == 0:
            return
        assert self.main_layout is not None
        owned = self.main_layout.owned_slot_mask(tgt_loc, owner_rank=self.shared_rank)
        local_targets = self.main_layout.translate_local_slots(tgt_loc[owned])
        shared_sources = self.main_layout.translate_slots(src_loc[owned])
        for local_kv, shared_kv in zip(self.local_kv_buffer, self.kv_buffer):
            local_kv[local_targets] = shared_kv[shared_sources]

        local_pages = local_targets // self.page_size
        shared_pages = shared_sources // self.page_size
        for local_index, shared_index in zip(
            self.local_index_k_with_scale_buffer,
            self.index_k_with_scale_buffer,
        ):
            local_index[local_pages] = shared_index[shared_pages]

    def get_index_k_write_owner(self) -> tuple[int, int]:
        return self.shared_rank, self.shared_size

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.local_index_k_with_scale_buffer[layer_id - self.start_layer]

    def get_broadcastable_index_k_with_scale_buffer(
        self, layer_id: int
    ) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.index_k_with_scale_buffer[layer_id - self.start_layer]

    def get_paged_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.rank_local_index_k_with_scale_buffer[layer_id - self.start_layer]

    def get_index_k_continuous(
        self, layer_id: int, seq_len: int, page_indices: torch.Tensor
    ):
        buffer = self.get_broadcastable_index_k_with_scale_buffer(layer_id)
        return index_buf_accessor.GetK.execute(
            self,
            buffer,
            seq_len=seq_len,
            page_indices=self.translate_index_pages(page_indices),
        )

    def get_index_k_scale_continuous(
        self, layer_id: int, seq_len: int, page_indices: torch.Tensor
    ):
        buffer = self.get_broadcastable_index_k_with_scale_buffer(layer_id)
        return index_buf_accessor.GetS.execute(
            self,
            buffer,
            seq_len=seq_len,
            page_indices=self.translate_index_pages(page_indices),
        )

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        buffer = self.get_broadcastable_index_k_with_scale_buffer(layer_id)
        return index_buf_accessor.GetKAndS.execute(
            self,
            buffer,
            page_indices=self.translate_index_pages(page_indices),
            seq_len_tensor=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        index_buf_accessor.SetKAndS.execute(
            pool=self,
            buf=self.local_index_k_with_scale_buffer[layer_id - self.start_layer],
            loc=loc,
            index_k=index_k,
            index_k_scale=index_k_scale,
            owner_rank=self.shared_rank,
            owner_size=self.shared_size,
        )
