# SPDX-License-Identifier: Apache-2.0
"""Pluggable tensor transport backends for weight_cache."""

from __future__ import annotations

import array
import logging
import math
import os
import socket
import struct
from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, NoReturn, Optional, Tuple

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
    if sent <= 0:
        raise RuntimeError("sendmsg failed to send the FD header")
    if sent < len(payload):
        sock.sendall(payload[sent:])


def _recv_fd(sock: socket.socket) -> Tuple[int, int]:
    fd_item_size = array.array("i").itemsize
    data, ancdata, flags, _ = sock.recvmsg(
        _FD_INDEX_STRUCT.size,
        socket.CMSG_SPACE(fd_item_size),
        getattr(socket, "MSG_CMSG_CLOEXEC", 0),
    )
    fds = array.array("i")
    for level, cmsg_type, cmsg_data in ancdata:
        if level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            fds.frombytes(cmsg_data[: len(cmsg_data) - (len(cmsg_data) % fd_item_size)])
    try:
        if flags & socket.MSG_CTRUNC or len(fds) != 1:
            raise RuntimeError(f"expected one complete fd, got {len(fds)}")
        while len(data) < _FD_INDEX_STRUCT.size:
            chunk = sock.recv(_FD_INDEX_STRUCT.size - len(data))
            if not chunk:
                raise RuntimeError("received truncated fd header")
            data += chunk
        index = _FD_INDEX_STRUCT.unpack(data)[0]
        os.set_inheritable(fds[0], False)
        return int(index), int(fds[0])
    except Exception:
        for fd in fds:
            os.close(fd)
        raise


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
        preloaded_weights_bytes: int = 0,
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

    def __init__(self):
        self._host_entries = {}
        self._host_export_fds = []
        self._host_received_fds = {}
        self._host_descriptors = {}
        self._host_tables = {}

    def configure_host_table_exports(self, entries, fds):
        self._host_entries = entries
        self._host_export_fds = list(fds)  # borrowed; daemon model owns them

    def prepare_export(
        self, state_tensors: Mapping[str, Tuple[torch.Tensor, bool]]
    ) -> Dict[str, Dict[str, Any]]:
        entries: Dict[str, Dict[str, Any]] = {}
        for name, (tensor, is_param) in state_tensors.items():
            if name in self._host_entries:
                entries[name] = self._host_entries[name]
                continue
            entries[name] = {
                "handle": MultiprocessingSerializer.serialize(
                    tensor.data, output_str=True
                ),
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
        preloaded_weights_bytes: int = 0,
    ) -> None:
        send_msg(
            conn,
            {
                "status": "ok",
                "config": config,
                "entries": entries,
                "pid": pid,
                "transport_backend": self.name,
                "preloaded_weights_bytes": preloaded_weights_bytes,
                "host_table_count": len(self._host_export_fds),
            },
        )
        for index, fd in enumerate(self._host_export_fds):
            _send_fd(conn, fd, index)

    def recv_fetch_state_response(
        self, sock: socket.socket, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        count = result.get("host_table_count", 0)
        if type(count) is not int or count < 0:
            raise ValueError("Invalid host table FD count")
        descriptors = {}
        for name, entry in result["entries"].items():
            desc = entry.get("host_table")
            if desc is None:
                continue
            index = desc["index"]
            if (
                type(index) is not int
                or not 0 <= index < count
                or desc["layout"] not in ("shared", "per_rank")
                or type(desc["nbytes"]) is not int
                or desc["nbytes"] <= 0
                or desc.get("backend", "memfd") not in ("memfd", "cuda_vmm")
                or name not in (desc["module"] + ".weight", desc["module"] + ".scale")
            ):
                raise ValueError("Invalid host table descriptor")
            if desc.get("backend") == "cuda_vmm" and (
                type(desc.get("allocation_nbytes")) is not int
                or desc["allocation_nbytes"] < desc["nbytes"]
                or type(desc.get("numa_id")) is not int
                or desc["numa_id"] < 0
            ):
                raise ValueError("Invalid CUDA host VMM descriptor")
            if index in descriptors and descriptors[index] != desc:
                raise ValueError("Conflicting host table descriptors")
            descriptors[index] = desc
        if set(descriptors) != set(range(count)):
            raise ValueError("Host table descriptor/FD count mismatch")
        received = {}
        try:
            for expected in range(count):
                index, fd = _recv_fd(sock)
                received[expected] = fd
                if index != expected or (
                    descriptors[expected].get("backend", "memfd") == "memfd"
                    and os.fstat(fd).st_size != descriptors[expected]["nbytes"]
                ):
                    raise ValueError("Host table FD index/size mismatch")
            self._host_received_fds = received
            self._host_descriptors = descriptors
        except Exception:
            for fd in received.values():
                os.close(fd)
            raise
        return result

    def import_tensor(self, entry: Dict[str, Any]) -> torch.Tensor:
        if "host_table" in entry:
            return self._import_host_tensor(entry)
        return MultiprocessingSerializer.deserialize(entry["handle"])

    def _import_host_tensor(self, entry):
        from sglang.srt.layers.engram import _HostTable

        desc = entry["host_table"]
        index = desc["index"]
        if self._host_descriptors.get(index) != desc:
            raise ValueError("Host table tensor descriptor mismatch")
        shape, offset = entry["shape"], entry["host_offset"]
        dtype = getattr(torch, entry["dtype"])
        itemsize = torch.empty((), dtype=dtype, device="cpu").element_size()
        if (
            any(type(n) is not int or n < 0 for n in shape)
            or type(offset) is not int
            or offset < 0
            or offset % itemsize
        ):
            raise ValueError("Invalid host table tensor shape/offset")
        size = math.prod(shape) * itemsize
        if offset + size > desc["nbytes"]:
            raise ValueError("Host table tensor exceeds its backing")
        if index not in self._host_tables:
            fd = self._host_received_fds.pop(index)
            if desc.get("backend", "memfd") == "cuda_vmm":
                self._host_tables[index] = _HostTable.from_vmm_fd(
                    desc["layout"],
                    desc["nbytes"],
                    fd,
                    desc["allocation_nbytes"],
                    desc["numa_id"],
                )
            else:
                self._host_tables[index] = _HostTable.from_fd(
                    desc["layout"], desc["nbytes"], fd
                )
        return (
            self._host_tables[index]
            .bytes.narrow(0, offset, size)
            .view(dtype)
            .view(shape)
        )

    def bind_host_tables(self, model, entries):
        for index, table in self._host_tables.items():
            desc = self._host_descriptors[index]
            module = model.get_submodule(desc["module"])
            if module.host_table is not None:
                raise ValueError("Client allocated a host table before IPC recovery")
            weight = entries[desc["module"] + ".weight"]
            scale = entries[desc["module"] + ".scale"]
            if (
                weight["host_offset"] != 0
                or scale["host_offset"]
                != module.weight.numel() * module.weight.element_size()
            ):
                raise ValueError("Host Engram weight/scale layout mismatch")
            module.host_table = table
            module._host_table_ptrs = (
                table.device_ptr,
                table.device_ptr + scale["host_offset"],
            )

    def __del__(self):
        for fd in getattr(self, "_host_received_fds", {}).values():
            try:
                os.close(fd)
            except OSError:
                pass


class VmmFdTransportBackend(WeightCacheTransportBackend):
    """Placeholder for the CUDA VMM + fd-passing transport.

    The backend is not wired up yet: can_export_state reports False so the
    daemon keeps selecting torch_ipc, and every other entry point fails loudly
    instead of silently returning None.
    """

    name = VMM_FD_BACKEND

    def __init__(self):
        self._raise_not_implemented()

    @staticmethod
    def _raise_not_implemented() -> NoReturn:
        raise NotImplementedError(
            f"weight cache transport backend {VMM_FD_BACKEND!r} is not "
            f"implemented in this build"
        )

    @classmethod
    def can_export_state(
        cls, state_tensors: Mapping[str, Tuple[torch.Tensor, bool]]
    ) -> bool:
        return False

    def prepare_export(
        self, state_tensors: Mapping[str, Tuple[torch.Tensor, bool]]
    ) -> Dict[str, Dict[str, Any]]:
        self._raise_not_implemented()

    def send_fetch_state_response(
        self,
        conn: socket.socket,
        *,
        config: Dict[str, Any],
        entries: Dict[str, Dict[str, Any]],
        pid: int,
        preloaded_weights_bytes: int = 0,
    ) -> None:
        self._raise_not_implemented()

    def recv_fetch_state_response(
        self, sock: socket.socket, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        self._raise_not_implemented()

    def import_tensor(self, entry: Dict[str, Any]) -> torch.Tensor:
        self._raise_not_implemented()


def choose_daemon_transport_backend(
    state_tensors: Mapping[str, Tuple[torch.Tensor, bool]],
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
