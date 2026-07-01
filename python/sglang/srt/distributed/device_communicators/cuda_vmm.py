from __future__ import annotations

import ctypes
import math
import os
import socket
import struct
import tempfile
import threading
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.utils import get_cuda_driver_bindings

_FD_HEADER_BYTES = 24
_FD_SEND_TIMEOUT_S = 120.0


@dataclass
class PeerMappedSharedTensor:
    global_view: torch.Tensor
    local_view: torch.Tensor
    local_pages: int
    aligned_bytes_per_rank: int
    handles: List[object]
    refs: list


def _check_drv(result_tuple, label):
    if not isinstance(result_tuple, tuple):
        result_tuple = (result_tuple,)
    err = result_tuple[0]
    drv = get_cuda_driver_bindings()
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{label}: {err}")
    return result_tuple[1] if len(result_tuple) > 1 else None


def _ceil_align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _all_ranks_ok(ok: bool, group: ProcessGroup) -> bool:
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.BAND, group=group)
    return flag.item() == 1


def _send_fd(sock, fd: int, src_rank: int, base_idx: int) -> None:
    import array

    fds = array.array("i", [int(fd)])
    header = struct.pack("<QQQ", int(src_rank), int(base_idx), 1)
    sent = sock.sendmsg(
        [header],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds.tobytes())],
    )
    if sent != len(header):
        raise RuntimeError(f"sendmsg sent {sent} bytes, expected {len(header)}")


def _recv_fd(sock):
    import array

    fd_item_size = array.array("i").itemsize
    data, ancdata, _, _ = sock.recvmsg(
        _FD_HEADER_BYTES, socket.CMSG_SPACE(fd_item_size)
    )
    if not data:
        return None
    if len(data) != _FD_HEADER_BYTES:
        raise RuntimeError(f"received truncated fd header: {len(data)}")
    src_rank, base_idx, fd_count = struct.unpack("<QQQ", data)
    fds = array.array("i")
    for level, cmsg_type, cmsg_data in ancdata:
        if level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            fds.frombytes(cmsg_data[: len(cmsg_data) - (len(cmsg_data) % fd_item_size)])
    if fd_count != 1 or len(fds) != 1:
        for fd in fds:
            os.close(fd)
        raise RuntimeError(
            f"expected one fd, got header={fd_count}, ancillary={len(fds)}"
        )
    return int(src_rank), int(base_idx), int(fds[0])


def _exchange_posix_fds(local_fd: int, group: ProcessGroup) -> dict[int, int]:
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    sock_kind = getattr(socket, "SOCK_SEQPACKET", socket.SOCK_STREAM)
    sock_dir = tempfile.mkdtemp(prefix="sgl_kv_fd_")
    sock_path = os.path.join(sock_dir, f"rank_{rank}.sock")
    server = socket.socket(socket.AF_UNIX, sock_kind)
    server.settimeout(_FD_SEND_TIMEOUT_S)
    received_fds: dict[int, int] = {}
    errors = []

    def recv_loop():
        try:
            for _ in range(world_size - 1):
                conn, _ = server.accept()
                with conn:
                    conn.settimeout(_FD_SEND_TIMEOUT_S)
                    while True:
                        packet = _recv_fd(conn)
                        if packet is None:
                            break
                        src_rank, base_idx, fd = packet
                        if base_idx != 0:
                            os.close(fd)
                            raise RuntimeError(f"unexpected base_idx={base_idx}")
                        if src_rank in received_fds:
                            os.close(fd)
                            raise RuntimeError(f"duplicate fd for rank {src_rank}")
                        received_fds[src_rank] = fd
        except BaseException as exc:
            errors.append(exc)

    try:
        server.bind(sock_path)
        server.listen(world_size)
        paths = [None] * world_size
        dist.all_gather_object(paths, sock_path, group=group)

        thread = threading.Thread(target=recv_loop, daemon=True)
        thread.start()
        try:
            for peer_rank, peer_path in enumerate(paths):
                if peer_rank == rank:
                    continue
                with socket.socket(socket.AF_UNIX, sock_kind) as sock:
                    sock.settimeout(_FD_SEND_TIMEOUT_S)
                    sock.connect(peer_path)
                    _send_fd(sock, local_fd, rank, 0)
        finally:
            thread.join(_FD_SEND_TIMEOUT_S)

        if thread.is_alive():
            raise RuntimeError("timed out waiting for POSIX fd exchange")
        if errors:
            raise RuntimeError("POSIX fd exchange receive failed") from errors[0]
        missing = set(range(world_size)).difference({rank}, received_fds.keys())
        if missing:
            for fd in received_fds.values():
                os.close(fd)
            raise RuntimeError(f"missing POSIX fds from ranks {sorted(missing)}")
        return received_fds
    finally:
        server.close()
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass
        try:
            os.rmdir(sock_dir)
        except OSError:
            pass


def _dtype_to_dlpack(dtype: torch.dtype) -> Tuple[int, int]:
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
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise TypeError(f"unsupported VMM tensor dtype: {dtype}") from exc


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


def _tensor_from_cuda_ptr(
    ptr: int,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device_id: int,
    refs: list,
) -> torch.Tensor:
    dl_code, dl_bits = _dtype_to_dlpack(dtype)
    ndim = len(shape)
    shape_arr = (ctypes.c_int64 * ndim)(*shape)

    managed = _DLManagedTensor()
    managed.dl_tensor.data = ctypes.c_void_p(ptr)
    managed.dl_tensor.device = _DLDevice(2, device_id)
    managed.dl_tensor.ndim = ndim
    managed.dl_tensor.dtype = _DLDataType(dl_code, dl_bits, 1)
    managed.dl_tensor.shape = shape_arr
    managed.dl_tensor.strides = None
    managed.dl_tensor.byte_offset = 0
    managed.manager_ctx = None

    @_DELETER_FN
    def _deleter(_):
        return None

    managed.deleter = _deleter
    refs.extend([managed, shape_arr, _deleter])

    ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
    ctypes.pythonapi.PyCapsule_New.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]
    capsule = ctypes.pythonapi.PyCapsule_New(ctypes.byref(managed), b"dltensor", None)
    tensor = torch.from_dlpack(capsule)
    try:
        tensor._sglang_vmm_refs = refs
    except Exception:
        pass
    return tensor


def create_peer_mapped_shared_tensor(
    local_shape: Tuple[int, ...],
    *,
    dtype: torch.dtype,
    cpu_group: ProcessGroup,
) -> PeerMappedSharedTensor:
    """Create a peer-mapped shared VMM tensor across a process group.

    Each rank owns one physical VMM allocation with ``local_shape``. All ranks
    export the allocation, import peer handles, and map them into one virtual
    range with the current rank's segment first.
    """
    if len(local_shape) == 0:
        raise ValueError("local_shape must have at least one dimension")

    drv = get_cuda_driver_bindings()
    rank = dist.get_rank(group=cpu_group)
    world_size = dist.get_world_size(group=cpu_group)
    device_id = torch.cuda.current_device()

    FABRIC = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    POSIX_FD = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR

    handle_type = FABRIC
    prop = drv.CUmemAllocationProp()
    prop.requestedHandleTypes = handle_type
    prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location = drv.CUmemLocation()
    prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id
    if hasattr(prop, "allocFlags") and hasattr(prop.allocFlags, "gpuDirectRDMACapable"):
        prop.allocFlags.gpuDirectRDMACapable = 1

    element_size = int(torch.empty((), dtype=dtype).element_size())
    row_bytes = element_size
    for dim in local_shape[1:]:
        row_bytes *= int(dim)
    requested_rows = int(local_shape[0])
    granularity = _check_drv(
        drv.cuMemGetAllocationGranularity(
            prop,
            drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        ),
        "cuMemGetAllocationGranularity",
    )
    rows_per_granule = int(granularity) // math.gcd(int(granularity), row_bytes)
    local_rows = _ceil_align(requested_rows, rows_per_granule)
    local_shape = (local_rows, *local_shape[1:])
    aligned_bytes = local_rows * row_bytes

    err, maybe_handle = drv.cuMemCreate(aligned_bytes, prop, 0)
    fabric_ok = err == drv.CUresult.CUDA_SUCCESS
    use_fabric = _all_ranks_ok(fabric_ok, cpu_group)
    if use_fabric:
        local_handle = maybe_handle
        handle_type = FABRIC
    else:
        if fabric_ok:
            drv.cuMemRelease(maybe_handle)
        handle_type = POSIX_FD
        prop.requestedHandleTypes = handle_type
        local_handle = _check_drv(
            drv.cuMemCreate(aligned_bytes, prop, 0),
            "cuMemCreate(POSIX_FD)",
        )

    try:
        posix_fds: dict[int, int] = {}
        local_fd = None
        if use_fabric:
            fabric_handle = _check_drv(
                drv.cuMemExportToShareableHandle(local_handle, handle_type, 0),
                "cuMemExportToShareableHandle(FABRIC)",
            )
            gathered_handles = [None for _ in range(world_size)]
            dist.all_gather_object(
                gathered_handles,
                bytes(fabric_handle.data),
                group=cpu_group,
            )
        else:
            local_fd = int(
                _check_drv(
                    drv.cuMemExportToShareableHandle(local_handle, handle_type, 0),
                    "cuMemExportToShareableHandle(POSIX_FD)",
                )
            )
            posix_fds = _exchange_posix_fds(local_fd, cpu_group)
            gathered_handles = [None for _ in range(world_size)]

        total_bytes = aligned_bytes * world_size
        base_va = _check_drv(
            drv.cuMemAddressReserve(total_bytes, int(granularity), 0, 0),
            "cuMemAddressReserve",
        )

        imported_handles: List[object] = []
        mapped_handles: List[object] = []
        try:
            segment_to_peer_rank = [
                (rank + segment) % world_size for segment in range(world_size)
            ]
            local_segment = 0

            for segment, peer_rank in enumerate(segment_to_peer_rank):
                peer_fabric_handle = gathered_handles[peer_rank]
                if peer_rank == rank:
                    handle = local_handle
                else:
                    shareable_handle = (
                        peer_fabric_handle
                        if use_fabric
                        else os.dup(posix_fds[peer_rank])
                    )
                    handle = _check_drv(
                        drv.cuMemImportFromShareableHandle(
                            shareable_handle, handle_type
                        ),
                        f"cuMemImportFromShareableHandle(rank={peer_rank})",
                    )
                    if not use_fabric:
                        os.close(shareable_handle)
                    imported_handles.append(handle)

                offset = segment * aligned_bytes
                _check_drv(
                    drv.cuMemMap(int(base_va) + offset, aligned_bytes, 0, handle, 0),
                    f"cuMemMap(rank={peer_rank}, segment={segment})",
                )
                mapped_handles.append(handle)

                access = drv.CUmemAccessDesc()
                access.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
                access.location.id = device_id
                access.flags = drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                _check_drv(
                    drv.cuMemSetAccess(
                        int(base_va) + offset,
                        aligned_bytes,
                        [access],
                        1,
                    ),
                    f"cuMemSetAccess(rank={peer_rank}, segment={segment})",
                )

            global_shape = (world_size * int(local_shape[0]), *local_shape[1:])
            refs: list = [local_handle, imported_handles, base_va]
            global_view = _tensor_from_cuda_ptr(
                int(base_va), global_shape, dtype, device_id, refs
            )
            local_view = global_view.narrow(
                0, local_segment * int(local_shape[0]), int(local_shape[0])
            )
            return PeerMappedSharedTensor(
                global_view=global_view,
                local_view=local_view,
                local_pages=local_rows,
                aligned_bytes_per_rank=aligned_bytes,
                handles=mapped_handles,
                refs=refs,
            )
        except Exception:
            for handle in imported_handles:
                drv.cuMemRelease(handle)
            raise
        finally:
            for fd in posix_fds.values():
                os.close(fd)
            if local_fd is not None:
                os.close(local_fd)
    except Exception:
        drv.cuMemRelease(local_handle)
        raise
