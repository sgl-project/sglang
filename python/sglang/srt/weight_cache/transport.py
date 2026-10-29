# SPDX-License-Identifier: Apache-2.0
"""Pluggable tensor transport backends for weight_cache."""

from __future__ import annotations

import array
import logging
import os
import socket
import struct
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

import torch

from sglang.srt.utils import MultiprocessingSerializer

from .protocol import send_msg
from .vmm_layout import pack_slot, plan_vmm_pack, validate_vmm_layout, view_from_entry

logger = logging.getLogger(__name__)

AUTO_BACKEND = "auto"
TORCH_IPC_BACKEND = "torch_ipc"
VMM_FD_BACKEND = "vmm_fd"

# vmm_fd packs the daemon's storages into allocations of at most this size (a
# larger storage gets an allocation of its own). Packing fills one allocation
# at a time and frees the originals it replaced before creating the next one,
# so the extra memory needed while packing is about one allocation.
VMM_FD_MAX_ALLOC_BYTES = 1 << 30

_FD_INDEX_STRUCT = struct.Struct("<Q")

# Receive fds with close-on-exec set, so that a program another thread execs
# while they are open does not inherit them (Linux only).
_RECV_FD_FLAGS = getattr(socket, "MSG_CMSG_CLOEXEC", 0)


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
    data, ancdata, msg_flags, _ = sock.recvmsg(
        _FD_INDEX_STRUCT.size, socket.CMSG_SPACE(fd_item_size), _RECV_FD_FLAGS
    )
    # The kernel has already installed every fd in this message in our process,
    # so take them all before any check: each error below must close them.
    fds = array.array("i")
    for level, cmsg_type, cmsg_data in ancdata:
        if level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            fds.frombytes(cmsg_data[: len(cmsg_data) - (len(cmsg_data) % fd_item_size)])
    try:
        if msg_flags & socket.MSG_CTRUNC:
            raise RuntimeError(
                "fd message truncated: more ancillary data than one fd was sent"
            )
        # A stream socket may return the header in pieces; the fd arrives with
        # the first piece.
        while 0 < len(data) < _FD_INDEX_STRUCT.size:
            more = sock.recv(_FD_INDEX_STRUCT.size - len(data))
            if not more:
                break
            data += more
        if len(data) != _FD_INDEX_STRUCT.size:
            raise RuntimeError(
                f"received truncated fd header: {len(data)} < {_FD_INDEX_STRUCT.size}"
            )
        if len(fds) != 1:
            raise RuntimeError(f"expected one fd, got {len(fds)}")
    except BaseException:
        for fd in fds:
            os.close(fd)
        raise
    return int(_FD_INDEX_STRUCT.unpack(data)[0]), int(fds[0])


def send_fds(sock: socket.socket, fds: Sequence[int]) -> None:
    """Send ``fds`` in order, one SCM_RIGHTS message each. The caller keeps
    (and must close) its own copies."""
    for index, fd in enumerate(fds):
        _send_fd(sock, fd, index)


def recv_fds(sock: socket.socket, count: int) -> List[int]:
    """Receive ``count`` fds sent by ``send_fds``, in order. On any error every
    fd received so far is closed before the exception propagates."""
    fds: List[int] = []
    try:
        for expected in range(count):
            index, fd = _recv_fd(sock)
            fds.append(fd)
            if index != expected:
                raise RuntimeError(f"received fd #{index}, expected #{expected}")
        return fds
    except BaseException:
        for fd in fds:
            os.close(fd)
        raise


def _prefix_offsets(sizes: Sequence[int]) -> List[int]:
    offsets, total = [], 0
    for size in sizes:
        offsets.append(total)
        total += size
    return offsets


class WeightCacheTransportBackend(ABC):
    name: str

    @abstractmethod
    def prepare_export(
        self, state_tensors: MutableMapping[str, Tuple[torch.Tensor, bool]]
    ) -> Dict[str, Dict[str, Any]]:
        """Prepare daemon-side entries; the backend may consume the mapping."""

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

    def close(self) -> None:
        """Release what the backend mapped or exported. Idempotent."""


class TorchIpcTransportBackend(WeightCacheTransportBackend):
    name = TORCH_IPC_BACKEND

    def prepare_export(
        self, state_tensors: MutableMapping[str, Tuple[torch.Tensor, bool]]
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
    """Share weights as CUDA VMM allocations passed as POSIX file descriptors.

    Daemon (``for_export`` + ``prepare_export``): copy every storage into a
    few large ``cuMemCreate`` allocations that can be exported as POSIX fds
    (layout in ``vmm_layout``) and repoint the daemon's own tensors at the
    copies. Each ``fetch_state`` gets the layout, then one freshly exported fd
    per allocation over ``SCM_RIGHTS``.

    Engine (``recv_fetch_state_response`` + ``import_tensor``): reserve one
    virtual address range, import and map every allocation into it, and build
    the tensors on top. CUDA frees a ``cuMemCreate`` allocation only after all
    of its mappings and handles are released, so the engine's mappings keep
    its weights valid even if the daemon exits. ``close`` unmaps them and
    frees the address range; the IPC loader calls it only when a load fails,
    since the tensors of a loaded model do not own the mappings (they stay
    until the process exits).
    """

    name = VMM_FD_BACKEND

    def __init__(
        self,
        *,
        device: Optional[torch.device] = None,
        prop: Any = None,
        granularity: int = 0,
        max_alloc_bytes: int = VMM_FD_MAX_ALLOC_BYTES,
    ):
        # Daemon-side allocation parameters, filled in by for_export().
        self._device = device
        self._prop = prop
        self._granularity = granularity
        self._max_alloc_bytes = max_alloc_bytes
        self._reservation = None  # VmmReservation holding every allocation
        self._handles: List[Any] = []  # daemon: retained handles to export
        self._alloc_sizes: List[int] = []
        self._alloc_offsets: List[int] = []  # allocation starts in the reservation
        # Engine: one storage per slot, so aliases share it.
        self._storages: Dict[Tuple[int, int, int], torch.UntypedStorage] = {}
        self.exported_bytes = 0

    @classmethod
    def for_export(cls, device_id: int) -> VmmFdTransportBackend:
        """Daemon-side backend for ``device_id``.

        Creates and exports one granule as a probe, so a GPU or driver that
        cannot share POSIX-fd VMM memory fails before the model is loaded.
        """
        from sglang.srt.utils.cuda_vmm_utils import (
            _get_cuda_driver,
            check_drv,
            get_allocation_granularity,
            make_device_allocation_prop,
        )

        drv = _get_cuda_driver()
        posix_fd = (
            drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
        prop = make_device_allocation_prop(device_id, handle_types=int(posix_fd))
        granularity = get_allocation_granularity(prop)
        handle = check_drv(
            drv.cuMemCreate(granularity, prop, 0), "cuMemCreate(vmm_fd probe)"
        )
        try:
            fd = check_drv(
                drv.cuMemExportToShareableHandle(handle, posix_fd, 0),
                "cuMemExportToShareableHandle(vmm_fd probe)",
            )
            os.close(int(fd))
        finally:
            check_drv(drv.cuMemRelease(handle), "cuMemRelease(vmm_fd probe)")
        return cls(
            device=torch.device("cuda", device_id),
            prop=prop,
            granularity=granularity,
        )

    def prepare_export(
        self, state_tensors: MutableMapping[str, Tuple[torch.Tensor, bool]]
    ) -> Dict[str, Dict[str, Any]]:
        """Pack ``state_tensors`` into exportable allocations.

        Consumes the source mapping; each slot retains its source storages until
        copied. The caller releases the model and other source owners first so
        copied originals can be reclaimed before the next allocation. Partial
        packing failures leave the daemon unusable and it exits.
        """
        from sglang.srt.utils.cuda_vmm_utils import VmmReservation, tensor_from_pointer

        if self._reservation is not None:
            raise RuntimeError("vmm_fd state was already exported")
        tic = time.perf_counter()
        plan = plan_vmm_pack(
            state_tensors,
            device=self._device,
            granularity=self._granularity,
            max_alloc_bytes=self._max_alloc_bytes,
        )
        state_tensors.clear()
        if not plan.alloc_sizes:
            return plan.entries

        offsets = _prefix_offsets(plan.alloc_sizes)
        self._reservation = VmmReservation(
            sum(plan.alloc_sizes),
            self._prop,
            self._device.index,
            alignment=self._granularity,
        )
        slots_by_alloc = [[] for _ in plan.alloc_sizes]
        for slot in plan.slots:
            slots_by_alloc[slot.alloc_index].append(slot)
        for alloc_index, size in enumerate(plan.alloc_sizes):
            self._handles.append(
                self._reservation.map(offsets[alloc_index], size, retain_handle=True)
            )
            alloc_base = self._reservation.base + offsets[alloc_index]
            for slot in slots_by_alloc[alloc_index]:
                dst = tensor_from_pointer(
                    alloc_base + slot.offset,
                    slot.nbytes,
                    device_id=self._device.index,
                )
                pack_slot(slot, dst)
            # Wait for the copies, then hand the replaced originals back to
            # the driver so that they can back the next allocation.
            torch.cuda.synchronize(self._device)
            torch.cuda.empty_cache()

        self._alloc_sizes = list(plan.alloc_sizes)
        self._alloc_offsets = offsets
        self.exported_bytes = sum(plan.alloc_sizes)
        logger.info(
            "[weight_cache] vmm_fd: packed %d tensors (%d storages) into %d "
            "allocation(s), %.2f GiB, in %.2fs",
            len(plan.entries),
            len(plan.slots),
            len(plan.alloc_sizes),
            self.exported_bytes / (1 << 30),
            time.perf_counter() - tic,
        )
        return plan.entries

    def _export_fds(self) -> List[int]:
        """Export a fresh POSIX fd per allocation; the caller closes them."""
        from sglang.srt.utils.cuda_vmm_utils import _get_cuda_driver, check_drv

        drv = _get_cuda_driver()
        posix_fd = (
            drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
        fds: List[int] = []
        try:
            for handle in self._handles:
                fd = check_drv(
                    drv.cuMemExportToShareableHandle(handle, posix_fd, 0),
                    "cuMemExportToShareableHandle(vmm_fd)",
                )
                fds.append(int(fd))
        except BaseException:
            for fd in fds:
                os.close(fd)
            raise
        return fds

    def send_fetch_state_response(
        self,
        conn: socket.socket,
        *,
        config: Dict[str, Any],
        entries: Dict[str, Dict[str, Any]],
        pid: int,
        preloaded_weights_bytes: int = 0,
    ) -> None:
        fds = self._export_fds()
        try:
            send_msg(
                conn,
                {
                    "status": "ok",
                    "config": config,
                    "entries": entries,
                    "pid": pid,
                    "transport_backend": self.name,
                    "preloaded_weights_bytes": preloaded_weights_bytes,
                    "allocation_sizes": list(self._alloc_sizes),
                },
            )
            send_fds(conn, fds)
        finally:
            for fd in fds:
                os.close(fd)

    def recv_fetch_state_response(
        self, sock: socket.socket, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        sizes = result.get("allocation_sizes")
        # Reject a bad layout before reading any fd. Closing the socket makes
        # the kernel drop the fds still queued on it.
        validate_vmm_layout(result["entries"], sizes)
        fds = recv_fds(sock, len(sizes))
        try:
            self._map_imported(fds, sizes)
        finally:
            for fd in fds:
                os.close(fd)
        return result

    def _map_imported(self, fds: Sequence[int], sizes: Sequence[int]) -> None:
        """Import every allocation and map them back to back, in order, into
        one reserved address range on the current device."""
        from sglang.srt.utils.cuda_vmm_utils import (
            VmmReservation,
            _get_cuda_driver,
            check_drv,
            get_allocation_granularity,
        )

        if self._reservation is not None:
            raise RuntimeError("vmm_fd backend already holds a mapping")
        device_id = torch.cuda.current_device()
        self._device = torch.device("cuda", device_id)
        if not fds:
            return
        drv = _get_cuda_driver()
        posix_fd = (
            drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
        offsets = _prefix_offsets(sizes)
        handles = []
        try:
            for index, fd in enumerate(fds):
                handles.append(
                    check_drv(
                        drv.cuMemImportFromShareableHandle(fd, posix_fd),
                        f"cuMemImportFromShareableHandle(vmm_fd allocation {index})",
                    )
                )
            prop = check_drv(
                drv.cuMemGetAllocationPropertiesFromHandle(handles[0]),
                "cuMemGetAllocationPropertiesFromHandle(vmm_fd)",
            )
            granularity = get_allocation_granularity(prop)
            if any(size % granularity for size in sizes):
                raise RuntimeError(
                    f"vmm_fd allocation sizes {list(sizes)} are not multiples of "
                    f"the {granularity}-byte allocation granularity"
                )
            reservation = VmmReservation(
                sum(sizes), prop, device_id, alignment=granularity
            )
            try:
                for handle, offset, size in zip(handles, offsets, sizes):
                    reservation.map_existing(offset, size, handle)
            except BaseException:
                reservation.close()
                raise
        finally:
            # Each mapping holds its own reference to the memory, so the
            # imported handles can go whether or not mapping succeeded.
            for handle in handles:
                err = drv.cuMemRelease(handle)
                err = err[0] if isinstance(err, tuple) else err
                if err != drv.CUresult.CUDA_SUCCESS:
                    logger.warning("cuMemRelease(vmm_fd import) -> %s", err)
        self._reservation = reservation
        self._alloc_sizes = list(sizes)
        self._alloc_offsets = offsets
        logger.info(
            "[weight_cache] vmm_fd: mapped %d allocation(s), %.2f GiB, at 0x%x",
            len(sizes),
            sum(sizes) / (1 << 30),
            reservation.base,
        )

    def import_tensor(self, entry: Dict[str, Any]) -> torch.Tensor:
        from sglang.srt.utils.cuda_vmm_utils import tensor_from_pointer

        if self._reservation is None:
            raise RuntimeError("vmm_fd: no allocation is mapped")
        key = (entry["alloc_index"], entry["offset"], entry["storage_nbytes"])
        storage = self._storages.get(key)
        if storage is None:
            alloc_index, offset, nbytes = key
            pointer = self._reservation.base + self._alloc_offsets[alloc_index]
            storage = tensor_from_pointer(
                pointer + offset, nbytes, device_id=self._device.index
            ).untyped_storage()
            self._storages[key] = storage
        return view_from_entry(storage, entry, self._device)

    def close(self) -> None:
        """Unmap every allocation and free the address range; the daemon also
        releases its allocation handles. Tensors built on these mappings must
        not be used afterwards."""
        self._storages.clear()
        self._handles = []
        reservation, self._reservation = self._reservation, None
        if reservation is not None:
            reservation.close()


def resolve_daemon_transport(requested: Optional[str]) -> str:
    """The backend a daemon serves for ``--weight-cache-transport``."""
    if requested in (None, "", AUTO_BACKEND):
        # auto stays on torch_ipc until vmm_fd has GPU parity results.
        return TORCH_IPC_BACKEND
    if requested in (TORCH_IPC_BACKEND, VMM_FD_BACKEND):
        return requested
    raise ValueError(f"Unknown weight cache transport {requested!r}")


def create_daemon_transport_backend(
    name: str, device_id: int
) -> WeightCacheTransportBackend:
    if name == VMM_FD_BACKEND:
        backend = VmmFdTransportBackend.for_export(device_id)
    elif name == TORCH_IPC_BACKEND:
        backend = TorchIpcTransportBackend()
    else:
        raise ValueError(f"Unknown weight cache transport {name!r}")
    logger.info("[weight_cache] Using transport backend: %s", backend.name)
    return backend


def check_served_transport(requested: Optional[str], served: str) -> None:
    """Refuse a daemon whose transport differs from the one the engine named."""
    if requested in (None, "", AUTO_BACKEND) or requested == served:
        return
    raise RuntimeError(
        f"[weight_cache] This engine requires the {requested!r} transport, but "
        f"the daemon serves {served!r}. Start the daemon with "
        f"--weight-cache-transport {requested}, or start the engine with "
        f"--weight-cache-transport auto."
    )


def get_client_transport_backend(name: Optional[str]) -> WeightCacheTransportBackend:
    if name in (None, "", TORCH_IPC_BACKEND):
        return TorchIpcTransportBackend()
    if name == VMM_FD_BACKEND:
        return VmmFdTransportBackend()
    raise RuntimeError(f"Unknown weight cache transport backend {name!r}")
