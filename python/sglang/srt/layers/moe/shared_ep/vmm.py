"""Private rank-major byte VMM allocation for SharedEP.

Initialization uses a CPU process group to exchange platform shareable
allocation handles. The resulting tensor views are stable GPU virtual addresses
and do not require collectives in the forward path.
"""

from __future__ import annotations

import ctypes
import os

import msgspec
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.vmm_utils import (
    VmmBackend,
    all_ranks_ok,
    exchange_posix_fds,
    get_vmm_backend,
    require_vmm_backend,
)


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
    backend: VmmBackend,
    *,
    base_va: int | None,
    total_bytes: int,
    mapped_addresses: list[int],
    segment_bytes: int,
) -> None:
    while mapped_addresses:
        backend.unmap(mapped_addresses.pop(), segment_bytes)
    if base_va is not None:
        backend.address_free(base_va, total_bytes)


def _release_vmm_handles_synchronized(
    backend: VmmBackend,
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
            backend.release(handle)
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
    _vmm_backend: object | None = None
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
            backend = self._vmm_backend or get_vmm_backend()
            for segment in range(self.world_size):
                address = self._base_va + segment * self.mapped_rank_bytes
                backend.unmap(address, self.mapped_rank_bytes)
            backend.address_free(self._base_va, self._total_bytes)
            self._base_va = 0


def _select_handle_transport(
    group: ProcessGroup,
    *,
    backend: VmmBackend,
    rank: int,
    world_size: int,
    local_handle,
) -> tuple[bool, list[bytes | None], list[int], dict[tuple[int, int], int]]:
    gathered: list[bytes | None] = [None] * world_size
    fabric_error = None
    if backend.supports_fabric:
        try:
            local_fabric_handle = backend.export_fabric(local_handle)
            fabric_ok = True
        except BaseException as error:
            local_fabric_handle = None
            fabric_error = error
            fabric_ok = False
        if all_ranks_ok(group, fabric_ok):
            dist.all_gather_object(gathered, local_fabric_handle, group=group)
            return True, gathered, [], {}

    local_fds = []
    posix_error = None
    try:
        local_fds.append(backend.export_posix_fd(local_handle))
        posix_ok = True
    except BaseException as error:
        posix_error = error
        posix_ok = False
    try:
        posix_all_ok = all_ranks_ok(group, posix_ok)
    except BaseException:
        for fd in local_fds:
            os.close(fd)
        raise
    if not posix_all_ok:
        for fd in local_fds:
            os.close(fd)
        local_detail = posix_error or fabric_error
        message = "SharedEP VMM POSIX-fd export failed on at least one rank"
        if local_detail is not None:
            message += f"; local rank {rank}: {local_detail}"
        raise RuntimeError(message) from posix_error

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


class _DLManagedTensorOwner(ctypes.Structure):
    _fields_ = [
        ("managed", _DLManagedTensor),
        ("shape", ctypes.c_int64 * 1),
    ]


_C_RUNTIME = ctypes.CDLL(None)
_C_RUNTIME.calloc.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
_C_RUNTIME.calloc.restype = ctypes.c_void_p
_C_RUNTIME.free.argtypes = [ctypes.c_void_p]
_C_RUNTIME.free.restype = None

# The managed tensor is the first field in the calloc-owned wrapper, so the C
# runtime's native ``free`` is itself a valid DLPack deleter. Keeping teardown
# native avoids a Python callback during interpreter shutdown.
_delete_dlmanaged_tensor = ctypes.cast(_C_RUNTIME.free, _DELETER_FN)


ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
ctypes.pythonapi.PyCapsule_New.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_void_p,
]
ctypes.pythonapi.PyCapsule_IsValid.restype = ctypes.c_int
ctypes.pythonapi.PyCapsule_IsValid.argtypes = [
    ctypes.py_object,
    ctypes.c_char_p,
]


def _uint8_tensor_from_cuda_ptr(
    ptr: int,
    length: int,
    device_id: int,
    refs: list[object],
    *,
    dlpack_device_type: int | None = None,
) -> torch.Tensor:
    """Wrap a VMM pointer with DLPack-owned metadata.

    PyTorch owns the allocated ``DLManagedTensor`` after consuming the capsule,
    so its deleter remains valid even if tensor views outlive this function.
    ``refs`` is retained in the signature for compatibility with the original
    CUDA helper; metadata lifetime no longer depends on that Python list.
    """

    del refs
    if ptr == 0:
        raise ValueError("cannot wrap a null VMM pointer")
    if length <= 0:
        raise ValueError(f"DLPack tensor length must be positive, got {length}")
    if dlpack_device_type is None:
        dlpack_device_type = 10 if torch.version.hip else 2
    if dlpack_device_type not in (2, 10):
        raise ValueError(f"unsupported DLPack GPU device type {dlpack_device_type}")

    raw_owner = _C_RUNTIME.calloc(1, ctypes.sizeof(_DLManagedTensorOwner))
    if not raw_owner:
        raise MemoryError("failed to allocate DLPack tensor metadata")
    owner = ctypes.cast(raw_owner, ctypes.POINTER(_DLManagedTensorOwner))
    managed = ctypes.pointer(owner.contents.managed)
    owner.contents.shape[0] = int(length)
    managed.contents.dl_tensor.data = ctypes.c_void_p(ptr)
    managed.contents.dl_tensor.device = _DLDevice(dlpack_device_type, device_id)
    managed.contents.dl_tensor.ndim = 1
    managed.contents.dl_tensor.dtype = _DLDataType(1, 8, 1)
    managed.contents.dl_tensor.shape = ctypes.cast(
        owner.contents.shape,
        ctypes.POINTER(ctypes.c_int64),
    )
    managed.contents.dl_tensor.strides = None
    managed.contents.dl_tensor.byte_offset = 0
    managed.contents.manager_ctx = None
    managed.contents.deleter = _delete_dlmanaged_tensor
    capsule = None
    try:
        capsule = ctypes.pythonapi.PyCapsule_New(managed, b"dltensor", None)
        return torch.from_dlpack(capsule)
    except BaseException:
        # A consumer that renamed the capsule to ``used_dltensor`` owns the
        # managed tensor even if it subsequently raised.
        if capsule is None or ctypes.pythonapi.PyCapsule_IsValid(capsule, b"dltensor"):
            _delete_dlmanaged_tensor(managed)
        raise


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
    dlpack_device_type: int | None = None,
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
            dlpack_device_type=dlpack_device_type,
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
    """Allocate one physical byte segment per rank and map them rank-major.

    CUDA prefers FABRIC handles and falls back collectively to POSIX fds. HIP
    uses uncached POSIX-fd allocations exclusively.
    """

    if logical_rank_bytes <= 0:
        raise ValueError(
            f"logical_rank_bytes must be positive, got {logical_rank_bytes}"
        )
    if device.type != "cuda":
        raise ValueError(f"SharedEP VMM requires a CUDA/HIP device, got {device}")

    rank = dist.get_rank(group=cpu_group)
    world_size = dist.get_world_size(group=cpu_group)
    _validate_same_host_group(cpu_group)

    backend = None
    device_id = None
    granularity = None
    mapped_rank_bytes = None
    preflight_error = None
    try:
        backend = require_vmm_backend(device)
        device_id = device.index
        if device_id is None:
            device_id = torch.cuda.current_device()
        granularity = backend.get_allocation_granularity(
            device_id,
            allow_fabric=backend.supports_fabric,
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
    assert backend is not None
    assert device_id is not None
    assert granularity is not None
    assert mapped_rank_bytes is not None

    local_handle = None
    create_error = None
    try:
        local_handle = backend.create_allocation(
            mapped_rank_bytes,
            device_id,
            allow_fabric=backend.supports_fabric,
        )
    except BaseException as error:
        create_error = error

    if backend.supports_fabric:
        try:
            combined_all_ok = all_ranks_ok(cpu_group, create_error is None)
        except BaseException:
            if local_handle is not None:
                backend.release(local_handle)
                local_handle = None
            raise
        if not combined_all_ok:
            fallback_cleanup_error = None
            if local_handle is not None:
                try:
                    backend.release(local_handle)
                except BaseException as error:
                    fallback_cleanup_error = error
                local_handle = None
            _synchronize_vmm_stage(
                cpu_group,
                rank,
                "FABRIC allocation fallback cleanup",
                fallback_cleanup_error,
            )
            create_error = None
            try:
                local_handle = backend.create_allocation(
                    mapped_rank_bytes,
                    device_id,
                    allow_fabric=False,
                )
            except BaseException as error:
                create_error = error
            try:
                _synchronize_vmm_stage(
                    cpu_group,
                    rank,
                    "allocation",
                    create_error,
                )
            except BaseException:
                if local_handle is not None:
                    backend.release(local_handle)
                    local_handle = None
                raise
    else:
        try:
            _synchronize_vmm_stage(
                cpu_group,
                rank,
                "allocation",
                create_error,
            )
        except BaseException:
            if local_handle is not None:
                backend.release(local_handle)
                local_handle = None
            raise
    assert local_handle is not None

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
            backend=backend,
            rank=rank,
            world_size=world_size,
            local_handle=local_handle,
        )

        mapping_error = None
        try:
            base_va = backend.reserve(total_bytes, granularity)
            for peer_rank in range(world_size):
                handle = local_handle
                if peer_rank != rank:
                    handle = (
                        backend.import_fabric(fabric_handles[peer_rank])
                        if use_fabric
                        else backend.import_posix_fd(peer_fds[(peer_rank, 0)])
                    )
                    imported_handles.append(handle)
                address = base_va + peer_rank * mapped_rank_bytes
                backend.map(address, mapped_rank_bytes, handle)
                mapped_addresses.append(address)
                backend.set_access(address, mapped_rank_bytes, device_id)
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
            dlpack_device_type=backend.dlpack_device_type,
        )

        retained_handles.extend(imported_handles)
        imported_handles.clear()
        retained_handles.append(local_handle)
        local_handle = None
        _release_vmm_handles_synchronized(
            backend,
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
            _vmm_backend=backend,
        )
    except BaseException:
        _release_partial_vmm_mapping(
            backend,
            base_va=base_va,
            total_bytes=total_bytes,
            mapped_addresses=mapped_addresses,
            segment_bytes=mapped_rank_bytes,
        )
        for handle in imported_handles:
            backend.release(handle)
        for handle in retained_handles:
            backend.release(handle)
        retained_handles.clear()
        if local_handle is not None:
            backend.release(local_handle)
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
