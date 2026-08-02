# SPDX-License-Identifier: Apache-2.0
"""Pluggable tensor transport backends for weight_cache."""

from __future__ import annotations

import array
import logging
import os
import socket
import struct
from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from sglang.srt.utils import MultiprocessingSerializer

from .protocol import send_msg

logger = logging.getLogger(__name__)

TORCH_IPC_BACKEND = "torch_ipc"
VMM_FD_BACKEND = "vmm_fd"

_FD_INDEX_STRUCT = struct.Struct("<Q")


def _send_fd(sock: socket.socket, fd: int, index: int) -> None:
    payload = _FD_INDEX_STRUCT.pack(index)
    fds = array.array("i", [int(fd)])
    sent = sock.sendmsg(
        [payload], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds.tobytes())]
    )
    if sent != len(payload):
        raise RuntimeError(f"sendmsg sent {sent} bytes, expected {len(payload)}")


def _recv_fd(sock: socket.socket) -> Tuple[int, int]:
    fd_item_size = array.array("i").itemsize
    data, ancdata, _, _ = sock.recvmsg(
        _FD_INDEX_STRUCT.size, socket.CMSG_SPACE(fd_item_size)
    )
    if len(data) != _FD_INDEX_STRUCT.size:
        raise RuntimeError(
            f"received truncated fd header: {len(data)} < {_FD_INDEX_STRUCT.size}"
        )
    index = _FD_INDEX_STRUCT.unpack(data)[0]
    fds = array.array("i")
    for level, cmsg_type, cmsg_data in ancdata:
        if level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            fds.frombytes(cmsg_data[: len(cmsg_data) - (len(cmsg_data) % fd_item_size)])
    if len(fds) != 1:
        for fd in fds:
            os.close(fd)
        raise RuntimeError(f"expected one fd, got {len(fds)}")
    return int(index), int(fds[0])


class WeightCacheTransportBackend(ABC):
    name: str

    @abstractmethod
    def prepare_export(
        self, state_tensors: Mapping[str, Tuple[torch.Tensor, bool]]
    ) -> Dict[str, Dict[str, Any]]:
        """Prepare daemon-side entries for all tensors."""

    @abstractmethod
    def send_fetch_state_response(
        self,
        conn: socket.socket,
        *,
        config: Dict[str, Any],
        entries: Dict[str, Dict[str, Any]],
        pid: int,
    ) -> None:
        """Send a successful fetch_state response."""

    @abstractmethod
    def recv_fetch_state_response(
        self, sock: socket.socket, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Client-side receive hook after recv_msg."""

    @abstractmethod
    def import_tensor(self, entry: Dict[str, Any]) -> torch.Tensor:
        """Import a single tensor from one entry."""


class TorchIpcTransportBackend(WeightCacheTransportBackend):
    name = TORCH_IPC_BACKEND

    def prepare_export(
        self, state_tensors: Mapping[str, Tuple[torch.Tensor, bool]]
    ) -> Dict[str, Dict[str, Any]]:
        entries: Dict[str, Dict[str, Any]] = {}
        for name, (tensor, is_param) in state_tensors.items():
            entries[name] = {
                "handle": MultiprocessingSerializer.serialize(tensor.data, output_str=True),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "is_param": is_param,
            }
        return entries

    def send_fetch_state_response(
        self,
        conn: socket.socket,
        *,
        config: Dict[str, Any],
        entries: Dict[str, Dict[str, Any]],
        pid: int,
    ) -> None:
        send_msg(
            conn,
            {
                "status": "ok",
                "config": config,
                "entries": entries,
                "pid": pid,
                "transport_backend": self.name,
            },
        )

    def recv_fetch_state_response(
        self, sock: socket.socket, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        return result

    def import_tensor(self, entry: Dict[str, Any]) -> torch.Tensor:
        return MultiprocessingSerializer.deserialize(entry["handle"])


class VmmFdTransportBackend(WeightCacheTransportBackend):
    name = VMM_FD_BACKEND

    def __init__(self):
        self._state_tensors: Dict[str, torch.Tensor] = {}
        self._mappings: list[Tuple[int, int]] = []

    @staticmethod
    def _load_cuda_helpers():
        from cuda.bindings import driver as drv

        from sglang.srt.distributed.device_communicators.vmm_utils import (
            check_drv,
            is_vmm_pointer,
            make_rw_access_desc,
        )
        from sglang.srt.layers.moe.dwdp.vmm import tensor_from_ptr

        return drv, check_drv, is_vmm_pointer, make_rw_access_desc, tensor_from_ptr

    @classmethod
    def can_export_state(
        cls, state_tensors: Mapping[str, Tuple[torch.Tensor, bool]]
    ) -> bool:
        if not state_tensors:
            return False
        try:
            _drv, _check_drv, is_vmm_pointer, _mk_access, _tensor_from_ptr = (
                cls._load_cuda_helpers()
            )
        except Exception:
            return False
        for tensor, _ in state_tensors.values():
            if tensor.device.type != "cuda":
                return False
            if not is_vmm_pointer(int(tensor.data_ptr())):
                return False
        return True

    def prepare_export(
        self, state_tensors: Mapping[str, Tuple[torch.Tensor, bool]]
    ) -> Dict[str, Dict[str, Any]]:
        drv, check_drv, _is_vmm_pointer, _mk_access, _tensor_from_ptr = (
            self._load_cuda_helpers()
        )
        entries: Dict[str, Dict[str, Any]] = {}
        self._state_tensors.clear()
        for name, (tensor, is_param) in state_tensors.items():
            err, base_ptr, alloc_size = drv.cuMemGetAddressRange(int(tensor.data_ptr()))
            if err != drv.CUresult.CUDA_SUCCESS:
                raise RuntimeError(
                    f"cuMemGetAddressRange failed for {name}: {err}. "
                    "Cannot export weight cache tensor over VMM FD."
                )
            alloc_offset = int(tensor.data_ptr()) - int(base_ptr)
            entries[name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "is_param": is_param,
                "alloc_size": int(alloc_size),
                "alloc_offset": int(alloc_offset),
            }
            self._state_tensors[name] = tensor
        return entries

    def _export_fd_for_tensor(self, tensor: torch.Tensor, name: str) -> int:
        drv, check_drv, _is_vmm_pointer, _mk_access, _tensor_from_ptr = (
            self._load_cuda_helpers()
        )
        err, base_ptr, _alloc_size = drv.cuMemGetAddressRange(int(tensor.data_ptr()))
        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemGetAddressRange failed for {name}: {err}")
        alloc_h = check_drv(
            drv.cuMemRetainAllocationHandle(int(base_ptr)),
            f"cuMemRetainAllocationHandle({name})",
        )
        try:
            posix_fd = check_drv(
                drv.cuMemExportToShareableHandle(
                    alloc_h,
                    drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                    0,
                ),
                f"cuMemExportToShareableHandle(POSIX_FD, {name})",
            )
            return int(posix_fd)
        finally:
            check_drv(drv.cuMemRelease(alloc_h), f"cuMemRelease({name})")

    def send_fetch_state_response(
        self,
        conn: socket.socket,
        *,
        config: Dict[str, Any],
        entries: Dict[str, Dict[str, Any]],
        pid: int,
    ) -> None:
        fd_order = list(entries.keys())
        send_msg(
            conn,
            {
                "status": "ok",
                "config": config,
                "entries": entries,
                "pid": pid,
                "transport_backend": self.name,
                "fd_order": fd_order,
            },
        )
        for index, name in enumerate(fd_order):
            fd = self._export_fd_for_tensor(self._state_tensors[name], name=name)
            try:
                _send_fd(conn, fd, index)
            finally:
                os.close(fd)

    def recv_fetch_state_response(
        self, sock: socket.socket, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        entries = result["entries"]
        fd_order = result.get("fd_order", [])
        fds_by_name: Dict[str, int] = {}
        for expected_index in range(len(fd_order)):
            recv_index, fd = _recv_fd(sock)
            if recv_index != expected_index:
                os.close(fd)
                raise RuntimeError(
                    f"received out-of-order VMM FD packet: got {recv_index}, "
                    f"expected {expected_index}"
                )
            fds_by_name[fd_order[recv_index]] = fd
        for name in fd_order:
            entry = entries[name]
            entry["_vmm_fd"] = fds_by_name[name]
        return result

    def import_tensor(self, entry: Dict[str, Any]) -> torch.Tensor:
        drv, check_drv, _is_vmm_pointer, make_rw_access_desc, tensor_from_ptr = (
            self._load_cuda_helpers()
        )
        fd = int(entry.pop("_vmm_fd"))
        imp_h = None
        dup_fd = os.dup(fd)
        try:
            imp_h = check_drv(
                drv.cuMemImportFromShareableHandle(
                    dup_fd,
                    drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                ),
                "cuMemImportFromShareableHandle(POSIX_FD)",
            )
            prop = check_drv(
                drv.cuMemGetAllocationPropertiesFromHandle(imp_h),
                "cuMemGetAllocationPropertiesFromHandle",
            )
            gran = check_drv(
                drv.cuMemGetAllocationGranularity(
                    prop,
                    drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
                ),
                "cuMemGetAllocationGranularity",
            )
            alloc_size = int(entry["alloc_size"])
            va = check_drv(
                drv.cuMemAddressReserve(alloc_size, int(gran), 0, 0),
                "cuMemAddressReserve",
            )
            check_drv(drv.cuMemMap(int(va), alloc_size, 0, imp_h, 0), "cuMemMap")
            device_id = torch.cuda.current_device()
            access = make_rw_access_desc(device_id)
            check_drv(
                drv.cuMemSetAccess(int(va), alloc_size, [access], 1), "cuMemSetAccess"
            )
            dtype = getattr(torch, entry["dtype"])
            shape = tuple(entry["shape"])
            offset = int(entry["alloc_offset"])
            tensor = tensor_from_ptr(
                ptr=int(va) + offset, shape=shape, dtype=dtype, device_id=device_id
            )
            self._mappings.append((int(va), alloc_size))
            return tensor
        finally:
            if imp_h is not None:
                try:
                    check_drv(drv.cuMemRelease(imp_h), "cuMemRelease(import)")
                except Exception:
                    pass
            try:
                os.close(dup_fd)
            except OSError:
                pass
            os.close(fd)

    def __del__(self):
        if not self._mappings:
            return
        try:
            drv, check_drv, _is_vmm_pointer, _mk_access, _tensor_from_ptr = (
                self._load_cuda_helpers()
            )
        except Exception:
            return
        while self._mappings:
            va, alloc_size = self._mappings.pop()
            try:
                check_drv(drv.cuMemUnmap(int(va), int(alloc_size)), "cuMemUnmap")
                check_drv(
                    drv.cuMemAddressFree(int(va), int(alloc_size)), "cuMemAddressFree"
                )
            except Exception:
                pass


def choose_daemon_transport_backend(
    state_tensors: Mapping[str, Tuple[torch.Tensor, bool]]
) -> WeightCacheTransportBackend:
    if VmmFdTransportBackend.can_export_state(state_tensors):
        logger.info("[weight_cache] Using transport backend: %s", VMM_FD_BACKEND)
        return VmmFdTransportBackend()
    logger.info("[weight_cache] Using transport backend: %s", TORCH_IPC_BACKEND)
    return TorchIpcTransportBackend()


def get_client_transport_backend(name: Optional[str]) -> WeightCacheTransportBackend:
    if name in (None, "", TORCH_IPC_BACKEND):
        return TorchIpcTransportBackend()
    if name == VMM_FD_BACKEND:
        return VmmFdTransportBackend()
    raise RuntimeError(f"Unknown weight cache transport backend {name!r}")
