# SPDX-License-Identifier: Apache-2.0
"""Pluggable tensor transport backends for weight_cache."""

from __future__ import annotations

import array
import logging
import os
import socket
import struct
from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, NoReturn, Optional, Tuple

import torch

from sglang.srt.platforms import current_platform
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

    @classmethod
    def can_export_state(
        cls, state_tensors: Mapping[str, Tuple[torch.Tensor, bool]]
    ) -> bool:
        """Whether this backend can export every one of these tensors."""
        return True

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

    def prepare_export(
        self, state_tensors: Mapping[str, Tuple[torch.Tensor, bool]]
    ) -> Dict[str, Dict[str, Any]]:
        entries: Dict[str, Dict[str, Any]] = {}
        for name, (tensor, is_param) in state_tensors.items():
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
            },
        )

    def recv_fetch_state_response(
        self, sock: socket.socket, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        return result

    def import_tensor(self, entry: Dict[str, Any]) -> torch.Tensor:
        return MultiprocessingSerializer.deserialize(entry["handle"])


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


TRANSPORT_BACKENDS: Dict[str, type[WeightCacheTransportBackend]] = {
    TORCH_IPC_BACKEND: TorchIpcTransportBackend,
    VMM_FD_BACKEND: VmmFdTransportBackend,
}


def resolve_transport_backend_cls(name: str) -> type[WeightCacheTransportBackend]:
    backend_cls = TRANSPORT_BACKENDS.get(name)
    if backend_cls is None:
        raise RuntimeError(
            f"Unknown weight cache transport backend {name!r}; registered "
            f"backends are {sorted(TRANSPORT_BACKENDS)}"
        )
    return backend_cls


def choose_daemon_transport_backend(
    state_tensors: Mapping[str, Tuple[torch.Tensor, bool]],
) -> WeightCacheTransportBackend:
    """Pick the first transport this platform declares that can export these tensors."""
    declared = current_platform.capabilities.weight_cache_transports
    if not declared:
        raise RuntimeError(
            f"Platform {current_platform.device_type!r} declares no weight cache "
            f"transport, so the daemon cannot export weights. Disable the weight "
            f"cache (--weight-cache-mode off)."
        )
    for name in declared:
        backend_cls = resolve_transport_backend_cls(name)
        if not backend_cls.can_export_state(state_tensors):
            continue
        logger.info("[weight_cache] Using transport backend: %s", name)
        return backend_cls()
    raise RuntimeError(
        f"None of the transports declared by platform "
        f"{current_platform.device_type!r} ({list(declared)}) can export these "
        f"weights."
    )


def get_client_transport_backend(name: Optional[str]) -> WeightCacheTransportBackend:
    if name in (None, ""):
        return TorchIpcTransportBackend()
    return resolve_transport_backend_cls(name)()
