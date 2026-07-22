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

"""Rank-major CUDA VMM storage shared by owner-sharded cache families."""

from __future__ import annotations

import ctypes
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.vmm_utils import (
    _get_cuda_driver,
    all_ranks_ok,
    check_drv,
    exchange_posix_fds,
    export_shareable_handles,
    import_peer_handle,
    make_rw_access_desc,
)

_shared_vmm_use_fabric: Optional[bool] = None


def _validate_same_host_group(cpu_group: ProcessGroup) -> None:
    rank = dist.get_rank(group=cpu_group)
    hosts = [None] * dist.get_world_size(group=cpu_group)
    hostname = None
    host_error = None
    try:
        hostname = os.uname().nodename
    except BaseException as error:
        host_error = error
    _synchronize_vmm_stage(cpu_group, rank, "host query", host_error)
    dist.all_gather_object(hosts, hostname, group=cpu_group)
    if len(set(hosts)) != 1:
        raise ValueError(
            "Shared KV VMM requires every attention CP rank on the same host."
        )


def _export_shareable_handles(retained_handles, group, rank):
    global _shared_vmm_use_fabric
    result = export_shareable_handles(retained_handles, group, rank)
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
            message = f"shared KV VMM {stage} failed on rank {failed_rank}: {error}"
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


def _release_vmm_handles_synchronized(
    drv,
    *,
    retained_handles: list,
    cpu_group: ProcessGroup,
    rank: int,
) -> None:
    """Release setup handles and publish the result before the next family."""

    release_error = None
    try:
        while retained_handles:
            handle = retained_handles[-1]
            check_drv(drv.cuMemRelease(handle), "cuMemRelease(shared setup handle)")
            retained_handles.pop()
    except BaseException as error:
        release_error = error
    _synchronize_vmm_stage(cpu_group, rank, "handle release", release_error)


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

# Tensor storage may outlive its pool through a temporary view. Retain the tiny
# callback objects after releasing the mapping so no stale DLPack callback is
# collected while PyTorch still owns an alias.
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
        raise TypeError(f"Unsupported shared KV VMM dtype: {dtype}")
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


def _construct_rank_major_views(
    *,
    cpu_group: ProcessGroup,
    rank: int,
    base_va: int,
    rank_local_base_va: Optional[int],
    world_size: int,
    local_rows: int,
    row_shape: tuple[int, ...],
    dtype: torch.dtype,
    device_id: int,
    refs: list,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Construct all DLPack aliases, then publish one symmetric setup result."""

    global_view = None
    rank_local_view = None
    local_view = None
    view_error = None
    try:
        global_view = _tensor_from_cuda_ptr(
            base_va,
            (world_size * local_rows, *row_shape),
            dtype,
            device_id,
            refs,
        )
        if rank_local_base_va is not None:
            rank_local_view = _tensor_from_cuda_ptr(
                rank_local_base_va,
                (world_size * local_rows, *row_shape),
                dtype,
                device_id,
                refs,
            )
        local_view = (
            rank_local_view.narrow(0, 0, local_rows)
            if rank_local_view is not None
            else global_view.narrow(0, rank * local_rows, local_rows)
        )
    except BaseException as error:
        view_error = error

    _synchronize_vmm_stage(cpu_group, rank, "tensor view construction", view_error)
    assert global_view is not None and local_view is not None
    return global_view, rank_local_view, local_view


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
    _closed: bool = field(default=False, init=False)

    @property
    def rank_stride_rows(self) -> int:
        return self.allocation.local_rows

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
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

    rank = dist.get_rank(group=cpu_group)
    world_size = dist.get_world_size(group=cpu_group)
    _validate_same_host_group(cpu_group)

    drv = None
    device_id = None
    posix_fd = None
    prop = None
    granularity = None
    local_rows = None
    aligned_bytes = None
    preflight_error = None
    try:
        drv = _get_cuda_driver()
        device_id = torch.cuda.current_device()
        posix_fd = (
            drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
        prop = drv.CUmemAllocationProp()
        prop.requestedHandleTypes = _shareable_allocation_handle_types(drv)
        prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location = drv.CUmemLocation()
        prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device_id
        if hasattr(prop, "allocFlags") and hasattr(
            prop.allocFlags, "gpuDirectRDMACapable"
        ):
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
    except BaseException as error:
        preflight_error = error
    _synchronize_vmm_stage(cpu_group, rank, "preflight", preflight_error)
    assert drv is not None
    assert device_id is not None
    assert posix_fd is not None
    assert prop is not None
    assert granularity is not None
    assert local_rows is not None
    assert aligned_bytes is not None

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
    retained_handles: list = []
    total_bytes = aligned_bytes * world_size
    try:
        fabric_handles, local_fds, use_fabric = _export_shareable_handles(
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
                        address,
                        aligned_bytes,
                        [make_rw_access_desc(device_id)],
                        1,
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
        except BaseException as error:
            mapping_error = error

        _synchronize_vmm_stage(cpu_group, rank, "mapping", mapping_error)
        refs = []
        global_view, rank_local_view, local_view = _construct_rank_major_views(
            cpu_group=cpu_group,
            rank=rank,
            base_va=base_va,
            rank_local_base_va=rank_local_base_va,
            world_size=world_size,
            local_rows=local_rows,
            row_shape=local_shape[1:],
            dtype=dtype,
            device_id=device_id,
            refs=refs,
        )
        retained_handles.extend(imported_handles)
        imported_handles.clear()
        retained_handles.append(local_handle)
        local_handle = None
        _release_vmm_handles_synchronized(
            drv,
            retained_handles=retained_handles,
            cpu_group=cpu_group,
            rank=rank,
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
        for handle in retained_handles:
            drv.cuMemRelease(handle)
        retained_handles.clear()
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
