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

"""Same-host CUDA VMM buffers with rank-major local and peer views.

Setup exchanges one owner-local allocation handle per rank and maps the
segments into one rank-major virtual-address span. Runtime users access the
result with ordinary device loads/stores; no collective or copy is issued by
the view itself.
"""

from __future__ import annotations

import ctypes
import math
import os
import socket
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.vmm_utils import (
    _get_cuda_driver,
    check_drv,
    exchange_posix_fds,
    import_peer_handle,
    make_rw_access_desc,
)


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

# A tensor alias may outlive its VMM owner. The address is invalid after close,
# but PyTorch must still be able to call the DLPack deleter later.
_RETIRED_DLPACK_REFS: list[list[Any]] = []


def _synchronize_stage(
    group: ProcessGroup,
    rank: int,
    stage: str,
    local_error: BaseException | None,
) -> None:
    errors: list[str | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(
        errors,
        None if local_error is None else f"{type(local_error).__name__}: {local_error}",
        group=group,
    )
    for failed_rank, error in enumerate(errors):
        if error is None:
            continue
        message = f"CUDA VMM {stage} failed on group rank {failed_rank}: {error}"
        if failed_rank == rank:
            raise RuntimeError(message) from local_error
        raise RuntimeError(message)


def _validate_group(
    group: ProcessGroup,
    requested_bytes: int,
    device_id: int,
    require_native_atomics: bool,
) -> tuple[int, int]:
    if requested_bytes <= 0:
        raise ValueError(f"requested_bytes must be positive, got {requested_bytes}")
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    if dist.get_backend(group) == dist.Backend.NCCL:
        raise ValueError("CUDA VMM setup requires a CPU-capable process group")

    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    requests: list[int | None] = [None] * world_size
    dist.all_gather_object(requests, requested_bytes, group=group)
    if any(peer_request != requested_bytes for peer_request in requests):
        raise ValueError(
            "CUDA VMM requested_bytes must match on every rank: "
            + ", ".join(
                f"rank {peer_rank}={peer_request}"
                for peer_rank, peer_request in enumerate(requests)
            )
        )

    hosts: list[str | None] = [None] * world_size
    dist.all_gather_object(hosts, socket.gethostname(), group=group)
    if len(set(hosts)) != 1:
        raise ValueError(
            "CUDA VMM peer buffers require every group rank on one host: "
            + ", ".join(
                f"rank {peer_rank}={host!r}" for peer_rank, host in enumerate(hosts)
            )
        )

    device_ids: list[int | None] = [None] * world_size
    dist.all_gather_object(device_ids, device_id, group=group)
    if len(set(device_ids)) != world_size:
        raise ValueError(
            "CUDA VMM peer buffers require one distinct CUDA device per rank: "
            f"{device_ids}"
        )

    driver = _get_cuda_driver()
    local_error = None
    try:
        source_device = check_drv(
            driver.cuDeviceGet(device_id), f"cuDeviceGet({device_id})"
        )
        atomic_attribute = (
            driver.CUdevice_P2PAttribute.CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED
        )
        for peer_rank, peer_device_id in enumerate(device_ids):
            if peer_rank == rank:
                continue
            assert peer_device_id is not None
            peer_device = check_drv(
                driver.cuDeviceGet(peer_device_id),
                f"cuDeviceGet({peer_device_id})",
            )
            can_access = int(
                check_drv(
                    driver.cuDeviceCanAccessPeer(source_device, peer_device),
                    f"cuDeviceCanAccessPeer({device_id}, {peer_device_id})",
                )
            )
            if not can_access:
                raise NotImplementedError(
                    f"CUDA device {device_id} cannot access peer rank "
                    f"{peer_rank} device {peer_device_id}"
                )
            if require_native_atomics:
                native_atomics = int(
                    check_drv(
                        driver.cuDeviceGetP2PAttribute(
                            atomic_attribute, source_device, peer_device
                        ),
                        "cuDeviceGetP2PAttribute(NATIVE_ATOMIC_SUPPORTED)",
                    )
                )
                if not native_atomics:
                    raise NotImplementedError(
                        "Peer publication requires native system-scope atomics, "
                        f"but device {device_id} cannot issue them to peer rank "
                        f"{peer_rank} device {peer_device_id}"
                    )
    except BaseException as error:  # noqa: BLE001 - synchronize failures cross-rank
        local_error = error
    _synchronize_stage(group, rank, "peer capability validation", local_error)
    return rank, world_size


def _make_allocation_property(driver, device_id: int, handle_types):
    prop = driver.CUmemAllocationProp()
    prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location = driver.CUmemLocation()
    prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id
    prop.requestedHandleTypes = handle_types
    return prop


def _tensor_from_cuda_bytes(
    ptr: int,
    num_bytes: int,
    device_id: int,
    refs: list[Any],
) -> torch.Tensor:
    shape_array = (ctypes.c_int64 * 1)(num_bytes)
    managed = _DLManagedTensor()
    managed.dl_tensor.data = ctypes.c_void_p(ptr)
    managed.dl_tensor.device = _DLDevice(2, device_id)
    managed.dl_tensor.ndim = 1
    managed.dl_tensor.dtype = _DLDataType(1, 8, 1)  # uint8
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


@dataclass
class RankMajorPeerBuffer:
    global_view: torch.Tensor
    local_view: torch.Tensor
    requested_bytes: int
    bytes_per_rank: int
    rank: int
    world_size: int
    group: ProcessGroup
    _base_va: int
    _refs: list[Any]
    _closed: bool = False

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        del self.local_view
        del self.global_view
        driver = _get_cuda_driver()
        for peer_rank in range(self.world_size):
            address = self._base_va + peer_rank * self.bytes_per_rank
            check_drv(
                driver.cuMemUnmap(address, self.bytes_per_rank),
                f"cuMemUnmap(rank={peer_rank})",
            )
        check_drv(
            driver.cuMemAddressFree(
                self._base_va, self.bytes_per_rank * self.world_size
            ),
            "cuMemAddressFree",
        )
        _RETIRED_DLPACK_REFS.append(self._refs)


def create_rank_major_peer_buffer(
    requested_bytes: int,
    *,
    group: ProcessGroup,
    device: torch.device,
    require_native_atomics: bool = False,
) -> RankMajorPeerBuffer:
    """Collectively allocate one physical byte segment per group rank."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"CUDA VMM requires a CUDA device, got {device}")
    device_id = (
        torch.cuda.current_device() if device.index is None else int(device.index)
    )
    if torch.cuda.current_device() != device_id:
        raise RuntimeError(
            "CUDA VMM current device differs from the requested device: "
            f"current={torch.cuda.current_device()}, requested={device_id}"
        )

    rank, world_size = _validate_group(
        group, requested_bytes, device_id, require_native_atomics
    )
    driver = _get_cuda_driver()
    handle_types = driver.CUmemAllocationHandleType
    posix = handle_types.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR

    # This allocator is intentionally same-host, so POSIX file descriptors are
    # sufficient. CUDA Python 13 models requestedHandleTypes as a strict enum
    # and rejects combined bitmasks such as POSIX_FD | FABRIC.
    prop = _make_allocation_property(driver, device_id, posix)
    granularity_flag = (
        driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
    )
    local_error = None
    granularity = 0
    try:
        granularity = int(
            check_drv(
                driver.cuMemGetAllocationGranularity(prop, granularity_flag),
                "cuMemGetAllocationGranularity",
            )
        )
    except BaseException as error:  # noqa: BLE001 - synchronize failures cross-rank
        local_error = error
    _synchronize_stage(group, rank, "allocation planning", local_error)
    bytes_per_rank = math.ceil(requested_bytes / granularity) * granularity

    plans: list[Any] = [None] * world_size
    dist.all_gather_object(plans, (bytes_per_rank, granularity), group=group)
    if any(plan != plans[0] for plan in plans):
        raise RuntimeError(f"CUDA VMM allocation plans differ across ranks: {plans}")

    err, local_handle = driver.cuMemCreate(bytes_per_rank, prop, 0)
    allocation_error = (
        None
        if err == driver.CUresult.CUDA_SUCCESS
        else RuntimeError(f"cuMemCreate(POSIX_FD): {err}")
    )
    _synchronize_stage(group, rank, "allocation", allocation_error)

    local_fds: list[int] = []
    peer_fds: dict[tuple[int, int], int] = {}
    imported_handles: list[Any] = []
    mapped_addresses: list[int] = []
    base_va: int | None = None
    total_bytes = bytes_per_rank * world_size
    try:
        export_error = None
        try:
            fd = check_drv(
                driver.cuMemExportToShareableHandle(local_handle, posix, 0),
                "cuMemExportToShareableHandle(POSIX_FD)",
            )
            local_fds = [int(fd)]
        except BaseException as error:  # noqa: BLE001 - synchronize failures cross-rank
            export_error = error
        _synchronize_stage(group, rank, "POSIX handle export", export_error)
        peer_fds = exchange_posix_fds(
            group, rank, world_size, local_fds, [1] * world_size
        )

        mapping_error = None
        try:
            base_va = int(
                check_drv(
                    driver.cuMemAddressReserve(total_bytes, granularity, 0, 0),
                    "cuMemAddressReserve",
                )
            )
            for peer_rank in range(world_size):
                handle = local_handle
                if peer_rank != rank:
                    handle = import_peer_handle(
                        None,
                        peer_fds[(peer_rank, 0)],
                        use_fabric=False,
                        peer_rank=peer_rank,
                    )
                    imported_handles.append(handle)
                address = base_va + peer_rank * bytes_per_rank
                check_drv(
                    driver.cuMemMap(address, bytes_per_rank, 0, handle, 0),
                    f"cuMemMap(rank={peer_rank})",
                )
                mapped_addresses.append(address)
                check_drv(
                    driver.cuMemSetAccess(
                        address,
                        bytes_per_rank,
                        [make_rw_access_desc(device_id)],
                        1,
                    ),
                    f"cuMemSetAccess(rank={peer_rank})",
                )
        except BaseException as error:  # noqa: BLE001 - synchronize failures cross-rank
            mapping_error = error
        _synchronize_stage(group, rank, "mapping", mapping_error)
        assert base_va is not None

        refs: list[Any] = []
        global_view = None
        local_view = None
        view_error = None
        try:
            global_view = _tensor_from_cuda_bytes(base_va, total_bytes, device_id, refs)
            local_view = global_view.narrow(0, rank * bytes_per_rank, requested_bytes)
        except BaseException as error:  # noqa: BLE001 - synchronize failures cross-rank
            view_error = error
        _synchronize_stage(group, rank, "tensor view construction", view_error)
        assert global_view is not None and local_view is not None

        for handle in imported_handles:
            check_drv(driver.cuMemRelease(handle), "cuMemRelease(peer)")
        imported_handles.clear()
        check_drv(driver.cuMemRelease(local_handle), "cuMemRelease(local)")
        local_handle = None
        return RankMajorPeerBuffer(
            global_view=global_view,
            local_view=local_view,
            requested_bytes=requested_bytes,
            bytes_per_rank=bytes_per_rank,
            rank=rank,
            world_size=world_size,
            group=group,
            _base_va=base_va,
            _refs=refs,
        )
    except BaseException:
        while mapped_addresses:
            driver.cuMemUnmap(mapped_addresses.pop(), bytes_per_rank)
        if base_va is not None:
            driver.cuMemAddressFree(base_va, total_bytes)
        for handle in imported_handles:
            driver.cuMemRelease(handle)
        if local_handle is not None:
            driver.cuMemRelease(local_handle)
        raise
    finally:
        for fd in local_fds:
            os.close(fd)
        for fd in peer_fds.values():
            os.close(fd)


def make_rank_major_tensor_view(
    allocation: RankMajorPeerBuffer,
    local_tensor: torch.Tensor,
) -> torch.Tensor:
    """Mirror a tensor view across every rank segment without copying."""
    if allocation.closed:
        raise RuntimeError("CUDA VMM peer buffer is closed")

    element_size = local_tensor.element_size()
    if allocation.bytes_per_rank % element_size:
        raise ValueError("Peer rank stride is not divisible by the tensor element size")
    local_offset_bytes = local_tensor.data_ptr() - allocation.local_view.data_ptr()
    view_span = 0 if local_tensor.numel() == 0 else 1
    for size, stride in zip(local_tensor.shape, local_tensor.stride()):
        if size > 0:
            view_span += (size - 1) * stride
    view_span_bytes = view_span * element_size
    if (
        local_offset_bytes < 0
        or local_offset_bytes + view_span_bytes > allocation.requested_bytes
    ):
        raise ValueError("Tensor view lies outside its local peer allocation")
    if local_offset_bytes % element_size:
        raise ValueError("Tensor view is not aligned to its element size")

    global_typed = allocation.global_view.view(local_tensor.dtype)
    rank_stride = allocation.bytes_per_rank // element_size
    return torch.as_strided(
        global_typed,
        size=(allocation.world_size, *local_tensor.shape),
        stride=(rank_stride, *local_tensor.stride()),
        storage_offset=local_offset_bytes // element_size,
    )
