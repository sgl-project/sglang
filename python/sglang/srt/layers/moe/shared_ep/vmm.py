"""Private rank-major byte VMM allocation for SharedEP.

Initialization uses a CPU process group to exchange shareable CUDA allocation
handles. The resulting tensor views are stable CUDA virtual addresses and do
not require collectives in the forward path.
"""

from __future__ import annotations

import ctypes
import os

import msgspec
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

_CLOSED_DLPACK_REFS: list[list[object]] = []


def round_up_to_granularity(value: int, granularity: int) -> int:
    if value <= 0:
        raise ValueError(f"value must be positive, got {value}")
    if granularity <= 0:
        raise ValueError(f"granularity must be positive, got {granularity}")
    return ((value + granularity - 1) // granularity) * granularity


def _synchronize_vmm_stage(
    cpu_group: ProcessGroup,
    rank: int,
    stage: str,
    local_error: BaseException | None,
) -> None:
    errors: list[str | None] = [None] * dist.get_world_size(group=cpu_group)
    dist.all_gather_object(
        errors,
        None if local_error is None else str(local_error),
        group=cpu_group,
    )
    for failed_rank, error in enumerate(errors):
        if error is not None:
            message = f"SharedEP VMM {stage} failed on rank {failed_rank}: {error}"
            if failed_rank == rank:
                raise RuntimeError(message) from local_error
            raise RuntimeError(message)


def _validate_same_host_group(cpu_group: ProcessGroup) -> None:
    rank = dist.get_rank(group=cpu_group)
    hosts: list[str | None] = [None] * dist.get_world_size(group=cpu_group)
    hostname = None
    host_error = None
    try:
        hostname = os.uname().nodename
    except BaseException as error:
        host_error = error
    _synchronize_vmm_stage(cpu_group, rank, "host query", host_error)
    dist.all_gather_object(hosts, hostname, group=cpu_group)
    if len(set(hosts)) != 1:
        raise ValueError("SharedEP VMM requires every EP rank to be on the same host")


def _release_partial_vmm_mapping(
    driver,
    *,
    base_va: int | None,
    total_bytes: int,
    mapped_addresses: list[int],
    segment_bytes: int,
) -> None:
    while mapped_addresses:
        driver.cuMemUnmap(mapped_addresses.pop(), segment_bytes)
    if base_va is not None:
        driver.cuMemAddressFree(base_va, total_bytes)


def _release_vmm_handles_synchronized(
    driver,
    *,
    retained_handles: list,
    cpu_group: ProcessGroup,
    rank: int,
) -> None:
    """Release setup handles and publish the result before returning."""

    release_error = None
    try:
        while retained_handles:
            handle = retained_handles[-1]
            check_drv(
                driver.cuMemRelease(handle),
                "cuMemRelease(shared setup handle)",
            )
            retained_handles.pop()
    except BaseException as error:
        release_error = error
    _synchronize_vmm_stage(cpu_group, rank, "handle release", release_error)


class SharedEpVmmAllocation(msgspec.Struct, kw_only=True):
    local_storage: torch.Tensor
    global_storage: torch.Tensor
    rank: int
    world_size: int
    logical_rank_bytes: int
    mapped_rank_bytes: int
    granularity: int
    _base_va: int = 0
    _total_bytes: int = 0
    _dlpack_refs: list[object] = msgspec.field(default_factory=list)
    _closed: bool = False

    def rank_offset(self, rank: int) -> int:
        if not 0 <= rank < self.world_size:
            raise IndexError(
                f"rank {rank} is outside SharedEP world size {self.world_size}"
            )
        return rank * self.mapped_rank_bytes

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.local_storage = torch.empty(0, dtype=torch.uint8)
        self.global_storage = torch.empty(0, dtype=torch.uint8)
        if self._base_va:
            torch.cuda.synchronize()
            driver = _get_cuda_driver()
            for segment in range(self.world_size):
                address = self._base_va + segment * self.mapped_rank_bytes
                check_drv(
                    driver.cuMemUnmap(address, self.mapped_rank_bytes),
                    f"cuMemUnmap(segment={segment})",
                )
            check_drv(
                driver.cuMemAddressFree(self._base_va, self._total_bytes),
                "cuMemAddressFree",
            )
            self._base_va = 0
        if self._dlpack_refs:
            _CLOSED_DLPACK_REFS.append(self._dlpack_refs)
            self._dlpack_refs = []


def _make_allocation_prop(driver, device_id: int, requested_handle_types):
    prop = driver.CUmemAllocationProp()
    prop.requestedHandleTypes = requested_handle_types
    prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location = driver.CUmemLocation()
    prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id
    if hasattr(prop, "allocFlags") and hasattr(prop.allocFlags, "gpuDirectRDMACapable"):
        prop.allocFlags.gpuDirectRDMACapable = 1
    return prop


def _select_handle_transport(
    group: ProcessGroup,
    *,
    rank: int,
    world_size: int,
    local_handle,
) -> tuple[bool, list[bytes | None], list[int], dict[tuple[int, int], int]]:
    fabric_handles, local_fds, use_fabric = export_shareable_handles(
        [local_handle],
        group,
        rank,
    )
    gathered: list[bytes | None] = [None] * world_size
    if use_fabric:
        dist.all_gather_object(gathered, fabric_handles[0], group=group)
        return True, gathered, [], {}
    try:
        peer_fds = exchange_posix_fds(
            group,
            rank,
            world_size,
            local_fds,
            [1] * world_size,
        )
    except BaseException:
        for fd in local_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        raise
    return False, gathered, local_fds, peer_fds


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


def _uint8_tensor_from_cuda_ptr(
    ptr: int,
    length: int,
    device_id: int,
    refs: list[object],
) -> torch.Tensor:
    shape = (ctypes.c_int64 * 1)(length)
    managed = _DLManagedTensor()
    managed.dl_tensor.data = ctypes.c_void_p(ptr)
    managed.dl_tensor.device = _DLDevice(2, device_id)
    managed.dl_tensor.ndim = 1
    managed.dl_tensor.dtype = _DLDataType(1, 8, 1)
    managed.dl_tensor.shape = shape
    managed.dl_tensor.strides = None
    managed.dl_tensor.byte_offset = 0
    managed.manager_ctx = None

    @_DELETER_FN
    def deleter(_):
        return None

    managed.deleter = deleter
    refs.extend([managed, shape, deleter])
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
    total_bytes: int,
    mapped_rank_bytes: int,
    logical_rank_bytes: int,
    device_id: int,
    refs: list[object],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct both DLPack aliases, then publish one symmetric result."""

    global_storage = None
    local_storage = None
    view_error = None
    try:
        global_storage = _uint8_tensor_from_cuda_ptr(
            base_va,
            total_bytes,
            device_id,
            refs,
        )
        local_storage = global_storage.narrow(
            0,
            rank * mapped_rank_bytes,
            logical_rank_bytes,
        )
    except BaseException as error:
        view_error = error

    _synchronize_vmm_stage(
        cpu_group,
        rank,
        "tensor view construction",
        view_error,
    )
    assert global_storage is not None and local_storage is not None
    return global_storage, local_storage


def allocate_rank_major_vmm(
    *,
    cpu_group: ProcessGroup,
    device: torch.device,
    logical_rank_bytes: int,
) -> SharedEpVmmAllocation:
    """Allocate one physical byte segment per rank and map them rank-major."""

    if logical_rank_bytes <= 0:
        raise ValueError(
            f"logical_rank_bytes must be positive, got {logical_rank_bytes}"
        )
    if device.type != "cuda":
        raise ValueError(f"SharedEP VMM requires a CUDA device, got {device}")

    rank = dist.get_rank(group=cpu_group)
    world_size = dist.get_world_size(group=cpu_group)
    _validate_same_host_group(cpu_group)

    driver = None
    device_id = None
    posix_type = None
    prop = None
    granularity = None
    mapped_rank_bytes = None
    preflight_error = None
    try:
        driver = _get_cuda_driver()
        device_id = device.index
        if device_id is None:
            device_id = torch.cuda.current_device()
        handle_types = driver.CUmemAllocationHandleType
        combined_types = (
            handle_types.CU_MEM_HANDLE_TYPE_FABRIC
            | handle_types.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
        posix_type = handle_types.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        prop = _make_allocation_prop(driver, device_id, combined_types)
        granularity = int(
            check_drv(
                driver.cuMemGetAllocationGranularity(
                    prop,
                    driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
                ),
                "cuMemGetAllocationGranularity",
            )
        )
        mapped_rank_bytes = round_up_to_granularity(
            logical_rank_bytes,
            granularity,
        )
    except BaseException as error:
        preflight_error = error
    _synchronize_vmm_stage(
        cpu_group,
        rank,
        "preflight",
        preflight_error,
    )
    assert driver is not None
    assert device_id is not None
    assert posix_type is not None
    assert prop is not None
    assert granularity is not None
    assert mapped_rank_bytes is not None

    error, local_handle = driver.cuMemCreate(mapped_rank_bytes, prop, 0)
    if not all_ranks_ok(cpu_group, error == driver.CUresult.CUDA_SUCCESS):
        if error == driver.CUresult.CUDA_SUCCESS:
            driver.cuMemRelease(local_handle)
        prop = _make_allocation_prop(driver, device_id, posix_type)
        error, local_handle = driver.cuMemCreate(mapped_rank_bytes, prop, 0)
        create_error = (
            None
            if error == driver.CUresult.CUDA_SUCCESS
            else RuntimeError(f"cuMemCreate(POSIX_FD): {error}")
        )
        try:
            _synchronize_vmm_stage(
                cpu_group,
                rank,
                "allocation",
                create_error,
            )
        except BaseException:
            if error == driver.CUresult.CUDA_SUCCESS:
                driver.cuMemRelease(local_handle)
            raise
    elif error != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuMemCreate(FABRIC|POSIX_FD): {error}")

    local_fds: list[int] = []
    peer_fds: dict[tuple[int, int], int] = {}
    imported_handles = []
    retained_handles = []
    base_va: int | None = None
    mapped_addresses: list[int] = []
    total_bytes = mapped_rank_bytes * world_size
    try:
        use_fabric, fabric_handles, local_fds, peer_fds = _select_handle_transport(
            cpu_group,
            rank=rank,
            world_size=world_size,
            local_handle=local_handle,
        )

        mapping_error = None
        try:
            base_va = int(
                check_drv(
                    driver.cuMemAddressReserve(total_bytes, granularity, 0, 0),
                    "cuMemAddressReserve",
                )
            )
            access = make_rw_access_desc(device_id)
            for peer_rank in range(world_size):
                handle = local_handle
                if peer_rank != rank:
                    handle = import_peer_handle(
                        fabric_handles[peer_rank],
                        None if use_fabric else peer_fds[(peer_rank, 0)],
                        use_fabric=use_fabric,
                        peer_rank=peer_rank,
                    )
                    imported_handles.append(handle)
                address = base_va + peer_rank * mapped_rank_bytes
                check_drv(
                    driver.cuMemMap(
                        address,
                        mapped_rank_bytes,
                        0,
                        handle,
                        0,
                    ),
                    f"cuMemMap(rank={peer_rank})",
                )
                mapped_addresses.append(address)
                check_drv(
                    driver.cuMemSetAccess(
                        address,
                        mapped_rank_bytes,
                        [access],
                        1,
                    ),
                    f"cuMemSetAccess(rank={peer_rank})",
                )
        except BaseException as error:
            mapping_error = error

        _synchronize_vmm_stage(
            cpu_group,
            rank,
            "mapping",
            mapping_error,
        )
        assert base_va is not None

        refs: list[object] = []
        global_storage, local_storage = _construct_rank_major_views(
            cpu_group=cpu_group,
            rank=rank,
            base_va=base_va,
            total_bytes=total_bytes,
            mapped_rank_bytes=mapped_rank_bytes,
            logical_rank_bytes=logical_rank_bytes,
            device_id=device_id,
            refs=refs,
        )

        retained_handles.extend(imported_handles)
        imported_handles.clear()
        retained_handles.append(local_handle)
        local_handle = None
        _release_vmm_handles_synchronized(
            driver,
            retained_handles=retained_handles,
            cpu_group=cpu_group,
            rank=rank,
        )
        return SharedEpVmmAllocation(
            local_storage=local_storage,
            global_storage=global_storage,
            rank=rank,
            world_size=world_size,
            logical_rank_bytes=logical_rank_bytes,
            mapped_rank_bytes=mapped_rank_bytes,
            granularity=granularity,
            _base_va=base_va,
            _total_bytes=total_bytes,
            _dlpack_refs=refs,
        )
    except BaseException:
        _release_partial_vmm_mapping(
            driver,
            base_va=base_va,
            total_bytes=total_bytes,
            mapped_addresses=mapped_addresses,
            segment_bytes=mapped_rank_bytes,
        )
        for handle in imported_handles:
            driver.cuMemRelease(handle)
        for handle in retained_handles:
            driver.cuMemRelease(handle)
        retained_handles.clear()
        if local_handle is not None:
            driver.cuMemRelease(local_handle)
        raise
    finally:
        for fd in local_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        for fd in peer_fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
