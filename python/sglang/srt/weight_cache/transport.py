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

from sglang.srt.platforms import current_platform
from sglang.srt.utils import MultiprocessingSerializer, is_npu

from .protocol import send_msg
from .vmm_layout import (
    LAYOUT_VERSION,
    check_layout_device,
    check_layout_version,
    pack_slot,
    plan_vmm_pack,
    validate_vmm_layout,
    view_from_entry,
)

logger = logging.getLogger(__name__)

AUTO_BACKEND = "auto"
TORCH_IPC_BACKEND = "torch_ipc"
VMM_FD_BACKEND = "vmm_fd"

# vmm_fd packs the daemon's storages into allocations of at most this size; a
# slot larger than that (one storage, or overlapping storages that share one)
# gets an allocation of its own. Packing fills one allocation at a time, then
# drops its references to the originals it copied and calls empty_cache(). The
# caching allocator returns memory only in whole segments, and a segment can
# also hold storages that a later allocation packs or other tensors that are
# still alive, so some copied originals stay reserved until the rest of their
# segment is freed. The extra memory needed while packing is the allocation
# being filled, up to about max(this cap, the largest slot), plus those
# originals.
#
# The cap trades that transient against attach time: on every fetch, each
# allocation is exported by the daemon, then received, imported and mapped by
# the engine. A smaller cap tends to mean more allocations and a slower
# attach, a larger cap a larger packing transient. 1 GiB sits in between.
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


def _check_npu_zero_copy(
    state_tensors: MutableMapping[str, Tuple[torch.Tensor, bool]],
) -> None:
    # torch_npu must be imported before the first ForkingPickler dump of an
    # NPU tensor so its reducer is registered; otherwise the tensor falls
    # back to a CPU copy instead of zero-copy IPC. Probe once and warn (rather
    # than error) if zero-copy is unavailable, so weight caching still works
    # by copying the weights through the CPU.
    import torch.multiprocessing.reductions as _reductions
    import torch_npu  # noqa: F401

    first = next(iter(state_tensors.values()), None)
    if first is not None:
        rebuild_fn, _ = _reductions.reduce_tensor(first[0].data)
        if getattr(rebuild_fn, "__name__", None) != "rebuild_npu_tensor":
            logger.warning(
                "NPU weight-cache zero-copy IPC is unavailable: tensor "
                "reduced to %r instead of torch_npu's rebuild_npu_tensor; "
                "falling back to a CPU copy. Zero-copy requires Ascend HDK "
                ">= 25.3.RC1 and CANN >= 8.3.RC1, and is unsupported on "
                "Ascend 950DT.",
                rebuild_fn,
            )


class TorchIpcTransportBackend(WeightCacheTransportBackend):
    name = TORCH_IPC_BACKEND

    def prepare_export(
        self, state_tensors: MutableMapping[str, Tuple[torch.Tensor, bool]]
    ) -> Dict[str, Dict[str, Any]]:
        if is_npu():
            _check_npu_zero_copy(state_tensors)
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


class VmmFdTransportBase(WeightCacheTransportBackend):
    """Share weights as VMM allocations passed as POSIX file descriptors.

    One wire contract with one implementation per platform: the packing layout,
    the fd framing and the engine-side tensor rebuild live here, and a subclass
    supplies the driver calls that create, export, import and map memory.

    Daemon (``for_export`` + ``prepare_export``): copy every storage into a few
    large exportable allocations (layout in ``vmm_layout``), dropping its
    reference to each original once it is copied. Each ``fetch_state`` gets the
    layout, then one freshly exported fd per allocation over ``SCM_RIGHTS``.

    Engine (``recv_fetch_state_response`` + ``import_tensor``): reserve one
    virtual address range, import and map every allocation into it, and build
    the tensors on top. A driver frees such an allocation only once every
    mapping and handle is released, so the engine's mappings keep its weights
    valid even if the daemon exits. ``close`` unmaps them and frees the address
    range; the IPC loader calls it only when a load fails, since the tensors of
    a loaded model do not own the mappings (they stay until the process exits).
    """

    name = VMM_FD_BACKEND
    # torch device type this implementation drives. It travels with the layout,
    # so an engine refuses allocation handles that came from another driver.
    device_type: str

    def __init__(
        self,
        *,
        device: Optional[torch.device] = None,
        granularity: int = 0,
        max_alloc_bytes: int = VMM_FD_MAX_ALLOC_BYTES,
    ):
        # Daemon-side allocation parameters, filled in by for_export().
        self._device = device
        self._granularity = granularity
        self._max_alloc_bytes = max_alloc_bytes
        self._reservation = None  # the reservation holding every allocation
        # Daemon: allocation handles retained for export. Engine: the handles
        # imported from the daemon's fds, on a platform that must keep them.
        self._handles: List[Any] = []
        self._alloc_sizes: List[int] = []
        self._alloc_offsets: List[int] = []  # allocation starts in the reservation
        # Engine: one storage per slot, so aliases share it.
        self._storages: Dict[Tuple[int, int, int], torch.UntypedStorage] = {}
        self.exported_bytes = 0

    @property
    def num_allocations(self) -> int:
        """Allocations packed (daemon) or mapped (engine)."""
        return len(self._alloc_sizes)

    # ------------------------------------------------------------------
    # Driver hooks, implemented once per platform
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def _reservation_base(self) -> int:
        """Address the reservation starts at."""
        raise NotImplementedError

    @abstractmethod
    def _empty_cache(self) -> None:
        """Hand back what the torch allocator caches for this device."""
        raise NotImplementedError

    @abstractmethod
    def _synchronize(self) -> None:
        """Wait for this device's queued work to retire."""
        raise NotImplementedError

    @abstractmethod
    def _free_device_bytes(self) -> int:
        """Device memory free right now, for the pack-plan log."""
        raise NotImplementedError

    @abstractmethod
    def _reserve(self, total_bytes: int):
        """Reserve one address range that holds every allocation."""
        raise NotImplementedError

    @abstractmethod
    def _create_and_map(self, offset: int, size: int):
        """Create one exportable allocation, map it at ``offset`` in the
        reservation, and return the handle to retain for export."""
        raise NotImplementedError

    @abstractmethod
    def _tensor_from_pointer(self, pointer: int, nbytes: int) -> torch.Tensor:
        """A uint8 tensor over ``nbytes`` at ``pointer``, on non-owning storage."""
        raise NotImplementedError

    @abstractmethod
    def _export_fds(self) -> List[int]:
        """Export a fresh fd per allocation; the caller closes them."""
        raise NotImplementedError

    @abstractmethod
    def _map_imported(self, fds: Sequence[int], sizes: Sequence[int]) -> None:
        """Import every allocation and map them back to back, in order, into
        one reserved address range on the current device."""
        raise NotImplementedError

    def _release_handles(self, handles: Sequence[Any]) -> None:
        """Release allocation handles once the reservation is unmapped.

        The default suits a driver whose reservation releases the handles it
        retained; a platform that owns them separately overrides this.
        """

    # ------------------------------------------------------------------
    # Daemon
    # ------------------------------------------------------------------

    def prepare_export(
        self, state_tensors: MutableMapping[str, Tuple[torch.Tensor, bool]]
    ) -> Dict[str, Dict[str, Any]]:
        """Pack ``state_tensors`` into exportable allocations.

        Consumes the source mapping; each slot retains its source storages until
        copied. The caller releases the model and other source owners first so
        copied originals can be reclaimed before the next allocation. Partial
        packing failures leave the daemon unusable and it exits.
        """
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
        # Hand back what torch still caches (e.g. from the released model), so
        # that the free memory read here is what the first allocation can use.
        self._empty_cache()
        self._log_pack_plan(plan, self._free_device_bytes())

        offsets = _prefix_offsets(plan.alloc_sizes)
        self._reservation = self._reserve(sum(plan.alloc_sizes))
        slots_by_alloc = [[] for _ in plan.alloc_sizes]
        for slot in plan.slots:
            slots_by_alloc[slot.alloc_index].append(slot)
        for alloc_index, size in enumerate(plan.alloc_sizes):
            self._handles.append(self._create_and_map(offsets[alloc_index], size))
            alloc_base = self._reservation_base + offsets[alloc_index]
            for slot in slots_by_alloc[alloc_index]:
                dst = self._tensor_from_pointer(alloc_base + slot.offset, slot.nbytes)
                pack_slot(slot, dst)
            # Wait for the copies, then hand the replaced originals back to
            # the driver so that they can back the next allocation.
            self._synchronize()
            self._empty_cache()

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

    def _log_pack_plan(self, plan, free_bytes: int) -> None:
        """Log the sizes ``plan`` will allocate and the free device memory,
        before anything is allocated or copied.

        Free memory below the largest allocation only gets a warning: packing
        returns originals to the driver as it copies them, so the free memory
        before packing is not the exact need, and an allocation that does not
        fit fails the driver call before any engine attaches.
        """
        largest = max(plan.slots, key=lambda slot: slot.nbytes)
        slot_key = (largest.alloc_index, largest.offset, largest.nbytes)
        name = next(
            (
                name
                for name, entry in plan.entries.items()
                if (entry["alloc_index"], entry["offset"], entry["storage_nbytes"])
                == slot_key
            ),
            None,
        )
        largest_alloc = max(plan.alloc_sizes)
        logger.info(
            "[weight_cache] vmm_fd: pack plan: %d slot(s) in %d allocation(s), "
            "cap %d bytes, largest slot %d bytes%s, largest allocation %d bytes, "
            "%d bytes in total, %d bytes free",
            len(plan.slots),
            len(plan.alloc_sizes),
            self._max_alloc_bytes,
            largest.nbytes,
            f" ({name})" if name is not None else "",
            largest_alloc,
            sum(plan.alloc_sizes),
            free_bytes,
        )
        if free_bytes < largest_alloc:
            logger.warning(
                "[weight_cache] vmm_fd: %d bytes of device memory are free, less "
                "than the largest allocation (%d bytes); packing may not have "
                "enough memory.",
                free_bytes,
                largest_alloc,
            )

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
                    "layout_version": LAYOUT_VERSION,
                    "device_type": self.device_type,
                },
            )
            send_fds(conn, fds)
        finally:
            for fd in fds:
                os.close(fd)

    # ------------------------------------------------------------------
    # Engine
    # ------------------------------------------------------------------

    def recv_fetch_state_response(
        self, sock: socket.socket, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        sizes = result.get("allocation_sizes")
        # Reject a bad layout before reading any fd. Closing the socket makes
        # the kernel drop the fds still queued on it. A daemon that predates
        # either field sends neither, and read as version 1 on this device.
        check_layout_version(result.get("layout_version"))
        check_layout_device(result.get("device_type"), self.device_type)
        validate_vmm_layout(result["entries"], sizes)
        fds = recv_fds(sock, len(sizes))
        try:
            self._map_imported(fds, sizes)
        finally:
            for fd in fds:
                os.close(fd)
        return result

    def import_tensor(self, entry: Dict[str, Any]) -> torch.Tensor:
        if self._reservation is None:
            raise RuntimeError("vmm_fd: no allocation is mapped")
        key = (entry["alloc_index"], entry["offset"], entry["storage_nbytes"])
        storage = self._storages.get(key)
        if storage is None:
            alloc_index, offset, nbytes = key
            pointer = self._reservation_base + self._alloc_offsets[alloc_index]
            storage = self._tensor_from_pointer(
                pointer + offset, nbytes
            ).untyped_storage()
            self._storages[key] = storage
        return view_from_entry(storage, entry, self._device)

    def close(self) -> None:
        """Unmap every allocation and free the address range; the daemon also
        releases its allocation handles. Tensors built on these mappings must
        not be used afterwards."""
        self._storages.clear()
        handles, self._handles = self._handles, []
        reservation, self._reservation = self._reservation, None
        try:
            if reservation is not None:
                reservation.close()
        finally:
            # Always, so a driver that refuses one unmap does not strand every
            # handle: a platform that owns them must still get its chance.
            self._release_handles(handles)


class CudaVmmFdTransportBackend(VmmFdTransportBase):
    """``vmm_fd`` over the CUDA driver.

    Allocations come from ``cuMemCreate`` with the POSIX-fd handle type, are
    mapped with ``cuMemMap`` into a ``cuMemAddressReserve`` range, and travel as
    the fd ``cuMemExportToShareableHandle`` hands out. CUDA frees the memory only
    after every mapping and handle is released, and ``VmmReservation.close``
    releases the handles it retained, so this class owns no handle separately.
    """

    device_type = "cuda"

    def __init__(
        self,
        *,
        device: Optional[torch.device] = None,
        prop: Any = None,
        granularity: int = 0,
        max_alloc_bytes: int = VMM_FD_MAX_ALLOC_BYTES,
    ):
        super().__init__(
            device=device, granularity=granularity, max_alloc_bytes=max_alloc_bytes
        )
        self._prop = prop

    @classmethod
    def for_export(cls, device_id: int) -> CudaVmmFdTransportBackend:
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

    @property
    def _reservation_base(self) -> int:
        return self._reservation.base

    def _empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def _synchronize(self) -> None:
        torch.cuda.synchronize(self._device)

    def _free_device_bytes(self) -> int:
        free_bytes, _ = torch.cuda.mem_get_info(self._device)
        return int(free_bytes)

    def _reserve(self, total_bytes: int):
        from sglang.srt.utils.cuda_vmm_utils import VmmReservation

        return VmmReservation(
            total_bytes,
            self._prop,
            self._device.index,
            alignment=self._granularity,
        )

    def _create_and_map(self, offset: int, size: int):
        return self._reservation.map(offset, size, retain_handle=True)

    def _tensor_from_pointer(self, pointer: int, nbytes: int) -> torch.Tensor:
        from sglang.srt.utils.cuda_vmm_utils import tensor_from_pointer

        return tensor_from_pointer(pointer, nbytes, device_id=self._device.index)

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

    def _map_imported(self, fds: Sequence[int], sizes: Sequence[int]) -> None:
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


def vmm_fd_backend_class() -> type:
    """The ``vmm_fd`` implementation for the active platform.

    One ``--weight-cache-transport vmm_fd`` names one wire contract, and the
    platform decides which driver carries it; see ``VmmFdTransportBase``.
    """
    if current_platform.is_xpu():
        from .vmm_fd_xpu import XpuVmmFdTransportBackend

        return XpuVmmFdTransportBackend
    if not current_platform.is_cuda_alike():
        raise RuntimeError(
            f"[weight_cache] The vmm_fd transport has no implementation for "
            f"{current_platform.device_type!r}: it needs CUDA VMM or Level Zero "
            f"VMM. Use --weight-cache-transport torch_ipc."
        )
    return CudaVmmFdTransportBackend


def resolve_daemon_transport(requested: Optional[str]) -> str:
    """The backend a daemon serves for ``--weight-cache-transport``.

    ``auto`` keeps torch_ipc wherever the platform serves it, so CUDA stays on
    torch_ipc until vmm_fd has GPU parity results, and a platform that serves
    only vmm_fd gets the one transport it has. A platform that declares none is
    left to the daemon's own gate, which names the problem properly.
    """
    transports = current_platform.weight_cache_transports()
    if requested in (None, "", AUTO_BACKEND):
        if TORCH_IPC_BACKEND in transports or not transports:
            return TORCH_IPC_BACKEND
        return transports[0]
    if requested not in (TORCH_IPC_BACKEND, VMM_FD_BACKEND):
        raise ValueError(f"Unknown weight cache transport {requested!r}")
    if transports and requested not in transports:
        raise ValueError(
            f"[weight_cache] The {requested!r} transport is not available on "
            f"{current_platform.device_type!r}, which serves "
            f"{list(transports)}. Start the daemon with "
            f"--weight-cache-transport {transports[0]} or auto."
        )
    return requested


def create_daemon_transport_backend(
    name: str, device_id: int
) -> WeightCacheTransportBackend:
    if name == VMM_FD_BACKEND:
        backend = vmm_fd_backend_class().for_export(device_id)
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
        return vmm_fd_backend_class()()
    raise RuntimeError(f"Unknown weight cache transport backend {name!r}")
