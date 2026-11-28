"""Share Intel GPU (XPU) device memory between processes over Level Zero IPC.

PyTorch has no XPU tensor IPC: ``UntypedStorage`` has ``_share_cuda_`` but no
``_share_xpu_``, and the interprocess ``torch.xpu.Event`` needs a SYCL 2026.2
build no published wheel uses. This module talks to Level Zero directly through
``libze_loader.so.1``, the way ``cuda_vmm_transport_utils`` talks to the CUDA
driver when torch is not enough.

The exporter allocates exportable device memory in a private context
(``LevelZeroIpcContext.allocate_exportable``), exports it as an IPC handle whose
first four bytes are a dma_buf fd, and serves the handle and fd to importers over
an abstract ``AF_UNIX`` ``SOCK_SEQPACKET`` socket with ``SCM_RIGHTS``
(``XpuIpcHandleServer``). An importer fetches both (``fetch_xpu_ipc_handle``),
maps the allocation into its own context (``open_peer_segment``) and wraps the
pointer as a torch tensor (``tensor_from_device_pointer``). Passing the fd over a
socket needs no ptrace permission, unlike the driver's own pidfd import, which
fails in containers without ``CAP_SYS_PTRACE``.

``XpuIpcExportedPool`` and ``XpuIpcPeerPools`` put those steps together for the
usual case, one pool exported per process and mapped once by each importer, the
way ``CudaVmmMemoryPool`` and its imported-pool cache do for CUDA.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import ctypes
import fcntl
import hmac
import logging
import os
import secrets
import selectors
import socket
import struct
import threading
import time
from functools import lru_cache

import msgspec
import torch

logger = logging.getLogger(__name__)

XPU_IPC_HANDLE_BYTES = 64

_LEVEL_ZERO_LOADER = "libze_loader.so.1"
_ZE_RESULT_SUCCESS = 0
_ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES = 0x1
_ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES = 0xC
_ZE_STRUCTURE_TYPE_CONTEXT_DESC = 0xD
_ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC = 0x15
_ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC = 0x18
_ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF = 0x2
_DRIVER_DEFAULT_ALIGNMENT = 0
# compute-runtime reports driverVersion as this base plus its build number.
_NEO_DRIVER_VERSION_BASE = 0x01030000
# 26.18 (build 38308) is the first release that imports a handle whose fd was
# rewritten as a local fd. Earlier ones with opaque handles, Xe2 included, hand
# the rewritten number to pidfd_getfd on the exporter, which can map whatever
# the exporter has open at that number.
_MIN_DRIVER_BUILD = 38308

_SERVER_BACKLOG = 16
_SERVER_POLL_S = 0.2
_SERVER_MAX_WAITING_PEERS = 64
_HANDLE_REQUEST_TIMEOUT_S = 5.0
_SEGMENT_ID_BYTES = 16
# A request is one segment id, which the server mints at a fixed length.
_HANDLE_REQUEST_LIMIT_BYTES = 64
# struct ucred from SO_PEERCRED: a signed pid, then the unsigned uid and gid.
_PEER_CREDENTIALS = struct.Struct("iII")


class XpuIpcError(RuntimeError):
    """A Level Zero call or an IPC handle exchange failed."""


class _ZeDriverProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("uuid", ctypes.c_ubyte * 16),
        ("driverVersion", ctypes.c_uint32),
    ]


class _ZeContextDesc(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
    ]


class _ZeExternalMemoryExportDesc(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
    ]


class _ZeDeviceMemAllocDesc(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("ordinal", ctypes.c_uint32),
    ]


class _ZeDeviceExternalMemoryProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("memoryAllocationImportTypes", ctypes.c_uint32),
        ("memoryAllocationExportTypes", ctypes.c_uint32),
        ("imageImportTypes", ctypes.c_uint32),
        ("imageExportTypes", ctypes.c_uint32),
    ]


class _ZeIpcMemHandle(ctypes.Structure):
    # c_ubyte, not c_char: ctypes cuts a c_char array at its first NUL byte.
    _fields_ = [("data", ctypes.c_ubyte * XPU_IPC_HANDLE_BYTES)]


class _LevelZeroDriver(msgspec.Struct, frozen=True):
    handle: int
    devices: tuple[int, ...]
    build: int


@lru_cache(maxsize=1)
def _load_level_zero() -> ctypes.CDLL:
    # torch's XPU runtime comes up first: a newer loader that already served a
    # pre-init device count rejects a later zeInit.
    torch.xpu.init()
    library = ctypes.CDLL(_LEVEL_ZERO_LOADER)
    void_p = ctypes.c_void_p
    count_p = ctypes.POINTER(ctypes.c_uint32)
    handle_p = ctypes.POINTER(ctypes.c_void_p)
    # Without argtypes, pointer arguments are truncated to 32 bits.
    signatures = {
        "zeInit": [ctypes.c_uint32],
        "zeDriverGet": [count_p, handle_p],
        "zeDriverGetProperties": [void_p, ctypes.POINTER(_ZeDriverProperties)],
        "zeDeviceGet": [void_p, count_p, handle_p],
        "zeContextCreate": [void_p, ctypes.POINTER(_ZeContextDesc), handle_p],
        "zeContextDestroy": [void_p],
        "zeDeviceGetExternalMemoryProperties": [
            void_p,
            ctypes.POINTER(_ZeDeviceExternalMemoryProperties),
        ],
        "zeMemAllocDevice": [
            void_p,
            ctypes.POINTER(_ZeDeviceMemAllocDesc),
            ctypes.c_size_t,
            ctypes.c_size_t,
            void_p,
            handle_p,
        ],
        "zeMemFree": [void_p, void_p],
        "zeMemGetAddressRange": [
            void_p,
            void_p,
            handle_p,
            ctypes.POINTER(ctypes.c_size_t),
        ],
        "zeMemGetIpcHandle": [void_p, void_p, ctypes.POINTER(_ZeIpcMemHandle)],
        "zeMemPutIpcHandle": [void_p, _ZeIpcMemHandle],
        "zeMemOpenIpcHandle": [
            void_p,
            void_p,
            _ZeIpcMemHandle,
            ctypes.c_uint32,
            handle_p,
        ],
        "zeMemCloseIpcHandle": [void_p, void_p],
    }
    for name, argtypes in signatures.items():
        entry_point = getattr(library, name)
        entry_point.argtypes = argtypes
        entry_point.restype = ctypes.c_int
    _check("zeInit", library.zeInit(0))
    return library


def _check(entry_point: str, status: int) -> None:
    if status != _ZE_RESULT_SUCCESS:
        raise XpuIpcError(f"{entry_point} failed with status 0x{status & 0xFFFFFFFF:x}")


@lru_cache(maxsize=1)
def _level_zero_driver() -> _LevelZeroDriver | None:
    library = _load_level_zero()
    driver_count = ctypes.c_uint32(0)
    _check("zeDriverGet", library.zeDriverGet(ctypes.byref(driver_count), None))
    if driver_count.value == 0:
        return None
    drivers = (ctypes.c_void_p * driver_count.value)()
    _check("zeDriverGet", library.zeDriverGet(ctypes.byref(driver_count), drivers))
    device_count = ctypes.c_uint32(0)
    _check(
        "zeDeviceGet",
        library.zeDeviceGet(drivers[0], ctypes.byref(device_count), None),
    )
    devices = (ctypes.c_void_p * device_count.value)()
    _check(
        "zeDeviceGet",
        library.zeDeviceGet(drivers[0], ctypes.byref(device_count), devices),
    )
    properties = _ZeDriverProperties(_ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES, None)
    _check(
        "zeDriverGetProperties",
        library.zeDriverGetProperties(drivers[0], ctypes.byref(properties)),
    )
    return _LevelZeroDriver(
        handle=int(drivers[0]),
        devices=tuple(int(devices[i]) for i in range(device_count.value)),
        build=properties.driverVersion - _NEO_DRIVER_VERSION_BASE,
    )


def _exports_dma_buf(device: int) -> bool:
    library = _load_level_zero()
    properties = _ZeDeviceExternalMemoryProperties(
        _ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES, None, 0, 0, 0, 0
    )
    _check(
        "zeDeviceGetExternalMemoryProperties",
        library.zeDeviceGetExternalMemoryProperties(
            ctypes.c_void_p(device), ctypes.byref(properties)
        ),
    )
    return bool(
        properties.memoryAllocationExportTypes & _ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF
    )


@lru_cache(maxsize=1)
def is_xpu_ipc_available() -> bool:
    """Whether the Level Zero driver sees torch's XPUs and each exports a dma_buf.

    Device indices are passed between processes as torch indices, so the driver
    must expose exactly the devices torch does. The driver must also be new
    enough to import a rewritten handle fd as local (see ``_MIN_DRIVER_BUILD``).
    """
    if torch.version.xpu is None:
        return False
    try:
        driver = _level_zero_driver()
        torch_device_count = torch.xpu.device_count()
        all_export = driver is not None and all(
            _exports_dma_buf(device) for device in driver.devices
        )
    except (AttributeError, OSError, RuntimeError) as error:
        logger.info(f"XPU IPC unavailable: {error}")
        return False
    driver_device_count = 0 if driver is None else len(driver.devices)
    if driver_device_count == 0:
        logger.info("XPU IPC unavailable: the Level Zero driver exposes no devices")
        return False
    if driver_device_count != torch_device_count:
        logger.warning(
            f"XPU IPC disabled: the Level Zero driver exposes {driver_device_count} "
            f"devices but torch sees {torch_device_count}"
        )
        return False
    if not all_export:
        logger.info("XPU IPC unavailable: a device cannot export a dma_buf")
        return False
    if driver.build < _MIN_DRIVER_BUILD:
        logger.warning(
            f"XPU IPC disabled: compute-runtime build {driver.build} predates "
            f"26.18 ({_MIN_DRIVER_BUILD}), whose IPC import can map the wrong "
            f"memory for a handle passed over a socket"
        )
        return False
    return True


class AddressRange(msgspec.Struct, frozen=True):
    """The allocation, or peer mapping, a device pointer lies in."""

    base_pointer: int
    segment_bytes: int


class ExportedSegment(msgspec.Struct, frozen=True):
    """One exported allocation: its IPC handle and the dma_buf fd it carries."""

    base_pointer: int
    segment_bytes: int
    handle_blob: bytes
    file_descriptor: int


class LevelZeroIpcContext:
    """A private Level Zero context that owns exported memory and peer mappings.

    Torch reads and writes both through pointers wrapped by
    ``tensor_from_device_pointer``: the Intel driver resolves a device pointer
    through a driver-wide allocation map, not per context.
    """

    def __init__(self) -> None:
        self._library = _load_level_zero()
        driver = _level_zero_driver()
        if driver is None:
            raise XpuIpcError("no Level Zero driver is available")
        self._devices = driver.devices
        descriptor = _ZeContextDesc(_ZE_STRUCTURE_TYPE_CONTEXT_DESC, None, 0)
        context = ctypes.c_void_p()
        _check(
            "zeContextCreate",
            self._library.zeContextCreate(
                ctypes.c_void_p(driver.handle),
                ctypes.byref(descriptor),
                ctypes.byref(context),
            ),
        )
        self._context: ctypes.c_void_p | None = context

    def _device(self, device_index: int) -> ctypes.c_void_p:
        if not 0 <= device_index < len(self._devices):
            raise XpuIpcError(
                f"XPU {device_index} is outside the {len(self._devices)} devices "
                f"this process can see"
            )
        return ctypes.c_void_p(self._devices[device_index])

    def allocate_exportable(self, device_index: int, num_bytes: int) -> int:
        """Allocate device memory that exports as a dma_buf; return its pointer.

        Asking for export at allocation time also keeps the memory uncompressed,
        which another card has to be able to read. Memory torch allocated itself
        is not a substitute: under expandable segments a handle covers one chunk,
        and in a process that sees one card it is compressed.
        """
        export = _ZeExternalMemoryExportDesc(
            _ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC,
            None,
            _ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF,
        )
        descriptor = _ZeDeviceMemAllocDesc(
            _ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            ctypes.cast(ctypes.pointer(export), ctypes.c_void_p),
            0,
            0,
        )
        pointer = ctypes.c_void_p()
        _check(
            "zeMemAllocDevice",
            self._library.zeMemAllocDevice(
                self._context,
                ctypes.byref(descriptor),
                num_bytes,
                _DRIVER_DEFAULT_ALIGNMENT,
                self._device(device_index),
                ctypes.byref(pointer),
            ),
        )
        return int(pointer.value)

    def free(self, device_pointer: int) -> None:
        _check(
            "zeMemFree",
            self._library.zeMemFree(self._context, ctypes.c_void_p(device_pointer)),
        )

    def address_range(self, device_pointer: int) -> AddressRange:
        base = ctypes.c_void_p()
        segment_bytes = ctypes.c_size_t(0)
        _check(
            "zeMemGetAddressRange",
            self._library.zeMemGetAddressRange(
                self._context,
                ctypes.c_void_p(device_pointer),
                ctypes.byref(base),
                ctypes.byref(segment_bytes),
            ),
        )
        return AddressRange(
            base_pointer=int(base.value or 0), segment_bytes=int(segment_bytes.value)
        )

    def export_segment(self, device_pointer: int) -> ExportedSegment:
        """Export the allocation a device pointer belongs to."""
        allocation = self.address_range(device_pointer)
        handle = _ZeIpcMemHandle()
        _check(
            "zeMemGetIpcHandle",
            self._library.zeMemGetIpcHandle(
                self._context,
                ctypes.c_void_p(allocation.base_pointer),
                ctypes.byref(handle),
            ),
        )
        handle_blob = bytes(bytearray(handle.data))
        return ExportedSegment(
            base_pointer=allocation.base_pointer,
            segment_bytes=allocation.segment_bytes,
            handle_blob=handle_blob,
            file_descriptor=struct.unpack_from("<i", handle_blob)[0],
        )

    def release_segment(self, segment: ExportedSegment) -> None:
        handle = _ZeIpcMemHandle()
        handle.data = (ctypes.c_ubyte * XPU_IPC_HANDLE_BYTES)(*segment.handle_blob)
        _check(
            "zeMemPutIpcHandle", self._library.zeMemPutIpcHandle(self._context, handle)
        )

    def open_peer_segment(
        self, handle_blob: bytes, file_descriptor: int, device_index: int
    ) -> int:
        """Map a peer's exported allocation onto ``device_index``; return its pointer.

        The driver imports the fd in the handle's first four bytes only when it
        differs from the exporter's own number; an unchanged handle makes it fetch
        the fd through pidfd instead, which needs ptrace permission. So the
        received fd is written in, or a duplicate when the two numbers collide.
        The caller still owns ``file_descriptor``.
        """
        device = self._device(device_index)
        local_descriptor = importable_descriptor(
            file_descriptor, struct.unpack_from("<i", handle_blob)[0]
        )
        try:
            local_blob = bytearray(handle_blob)
            struct.pack_into("<i", local_blob, 0, local_descriptor)
            handle = _ZeIpcMemHandle()
            handle.data = (ctypes.c_ubyte * XPU_IPC_HANDLE_BYTES)(*local_blob)
            peer_pointer = ctypes.c_void_p()
            _check(
                "zeMemOpenIpcHandle",
                self._library.zeMemOpenIpcHandle(
                    self._context, device, handle, 0, ctypes.byref(peer_pointer)
                ),
            )
        finally:
            if local_descriptor != file_descriptor:
                os.close(local_descriptor)
        return int(peer_pointer.value)

    def close_peer_segment(self, peer_pointer: int) -> None:
        _check(
            "zeMemCloseIpcHandle",
            self._library.zeMemCloseIpcHandle(
                self._context, ctypes.c_void_p(peer_pointer)
            ),
        )

    def destroy(self) -> None:
        if self._context is None:
            return
        _check("zeContextDestroy", self._library.zeContextDestroy(self._context))
        self._context = None


def importable_descriptor(received_descriptor: int, exporter_descriptor: int) -> int:
    """The received fd, or a duplicate if its number equals the exporter's.

    The lowest free number cannot collide while the received fd holds it, and
    asking for one above the exporter's number would fail with EINVAL or EMFILE
    whenever the exporter has more fds open than this process may.
    """
    if received_descriptor != exporter_descriptor:
        return received_descriptor
    return fcntl.fcntl(received_descriptor, fcntl.F_DUPFD_CLOEXEC, 0)


def tensor_from_device_pointer(
    device_pointer: int, num_bytes: int, device_index: int
) -> torch.Tensor:
    """Wrap XPU device memory as a non-owning uint8 tensor on ``device_index``.

    For a peer mapping, ``device_index`` is the device the mapping was opened
    against, so copies run on this process' card. The caller keeps the memory
    alive for as long as the tensor or any view of it is used.
    """
    # XPU device pointers set bit 63, and the storage constructor takes an int64.
    signed_pointer = (
        device_pointer - (1 << 64) if device_pointer >= (1 << 63) else device_pointer
    )
    device = torch.device("xpu", device_index)
    storage = torch._C._construct_storage_from_data_pointer(
        signed_pointer, device, num_bytes
    )
    return torch.empty(0, dtype=torch.uint8, device=device).set_(
        storage, 0, (num_bytes,)
    )


class XpuIpcHandleServer:
    """Serve one exported segment's handle and fd to importing processes.

    The socket lives in the abstract namespace, so it leaves no file behind, and
    keeps message boundaries, so a request and a reply each arrive whole. A peer
    of another user is turned away on connect, and a peer is answered only once
    its request has arrived: one that never sends it waits beside the others
    until its deadline.

    Any process in the same network namespace can read the socket name from
    /proc/net/unix, and a root exporter's uid check admits root in any container
    that shares it. So ``segment_id``, minted here from ``secrets``, is what
    keeps the fd from them: hand it only to the importers that should map the
    segment.
    """

    def __init__(self, segment: ExportedSegment) -> None:
        self.segment_id = secrets.token_hex(_SEGMENT_ID_BYTES)
        self.segment = segment
        self.socket_name = f"sglang-xpu-ipc-{os.getpid()}-{secrets.token_hex(6)}"
        self._accept_failing = False
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self._listener.bind("\0" + self.socket_name)
        self._listener.listen(_SERVER_BACKLOG)
        self._listener.setblocking(False)
        self._stopped = threading.Event()
        self.thread = threading.Thread(
            target=self._serve, name=f"xpu-ipc-{self.socket_name}", daemon=True
        )
        self.thread.start()

    def _serve(self) -> None:
        # Accepted peers whose request has not arrived, and when it is due by.
        waiting: dict[socket.socket, float] = {}
        with selectors.DefaultSelector() as selector:
            selector.register(self._listener, selectors.EVENT_READ)
            try:
                while not self._stopped.is_set():
                    for key, _ in selector.select(timeout=_SERVER_POLL_S):
                        if key.fileobj is self._listener:
                            self._admit_peer(selector, waiting)
                        elif key.fileobj in waiting:
                            connection = key.fileobj
                            _forget_peer(selector, waiting, connection)
                            self._answer_peer(connection)
                        # Otherwise admitting a peer in this batch already dropped it.
                    now = time.monotonic()
                    for connection, deadline in list(waiting.items()):
                        if deadline <= now:
                            _forget_peer(selector, waiting, connection)
                            connection.close()
            finally:
                for connection in list(waiting):
                    _forget_peer(selector, waiting, connection)
                    connection.close()

    def _admit_peer(
        self, selector: selectors.BaseSelector, waiting: dict[socket.socket, float]
    ) -> None:
        try:
            connection, _ = self._listener.accept()
        except BlockingIOError:
            return
        except OSError as error:
            if self._stopped.is_set():
                return
            # Out of fds, most often. Keep serving, and log once until it clears.
            if not self._accept_failing:
                logger.warning(
                    f"XPU IPC handle server {self.socket_name} cannot accept a "
                    f"peer: {error}"
                )
            self._accept_failing = True
            self._stopped.wait(_SERVER_POLL_S)
            return
        self._accept_failing = False
        if not self._is_own_user(connection):
            connection.close()
            return
        if len(waiting) >= _SERVER_MAX_WAITING_PEERS:
            oldest = next(iter(waiting))
            _forget_peer(selector, waiting, oldest)
            oldest.close()
            logger.warning(
                f"XPU IPC handle server {self.socket_name} dropped a peer that had "
                f"not sent its request"
            )
        connection.settimeout(_HANDLE_REQUEST_TIMEOUT_S)
        selector.register(connection, selectors.EVENT_READ)
        waiting[connection] = time.monotonic() + _HANDLE_REQUEST_TIMEOUT_S

    def _is_own_user(self, connection: socket.socket) -> bool:
        try:
            _, peer_uid, _ = _PEER_CREDENTIALS.unpack(
                connection.getsockopt(
                    socket.SOL_SOCKET, socket.SO_PEERCRED, _PEER_CREDENTIALS.size
                )
            )
        except OSError as error:
            logger.warning(
                f"XPU IPC handle server {self.socket_name} could not identify a "
                f"peer: {error}"
            )
            return False
        if peer_uid != os.geteuid():
            logger.warning(
                f"XPU IPC handle server {self.socket_name} turned away a peer "
                f"running as uid {peer_uid}"
            )
            return False
        return True

    def _answer_peer(self, connection: socket.socket) -> None:
        with connection:
            try:
                self._send_handle(connection)
            except (OSError, XpuIpcError) as error:
                # One importer giving up must not stop the segment reaching the rest.
                logger.warning(
                    f"XPU IPC handle server {self.socket_name} could not serve a "
                    f"peer: {error}"
                )

    def _send_handle(self, connection: socket.socket) -> None:
        request = connection.recv(_HANDLE_REQUEST_LIMIT_BYTES + 1)
        if len(request) > _HANDLE_REQUEST_LIMIT_BYTES:
            raise XpuIpcError(
                f"XPU IPC request exceeds the {_HANDLE_REQUEST_LIMIT_BYTES} byte limit"
            )
        if not hmac.compare_digest(request, self.segment_id.encode("ascii")):
            raise XpuIpcError(
                f"XPU IPC handle server {self.socket_name} was asked for another "
                f"segment"
            )
        sent = socket.send_fds(
            connection, [self.segment.handle_blob], [self.segment.file_descriptor]
        )
        if sent != len(self.segment.handle_blob):
            raise XpuIpcError(
                f"XPU IPC handle server {self.socket_name} sent {sent} of "
                f"{len(self.segment.handle_blob)} handle bytes"
            )

    def close(self) -> None:
        """Stop serving; once this returns nothing can still send the fd."""
        self._stopped.set()
        try:
            self._listener.shutdown(socket.SHUT_RDWR)
        except OSError:
            # A listener that cannot shut down also fails the thread's next accept.
            pass
        self.thread.join()
        self._listener.close()


def _forget_peer(
    selector: selectors.BaseSelector,
    waiting: dict[socket.socket, float],
    connection: socket.socket,
) -> None:
    selector.unregister(connection)
    waiting.pop(connection)


def fetch_xpu_ipc_handle(socket_name: str, segment_id: str) -> tuple[bytes, int]:
    """Collect an exported segment's handle and its fd; the caller owns the fd.

    ``segment_id`` is the server's secret, so no error message repeats it.
    """
    request = segment_id.encode("utf-8")
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
        connection.settimeout(_HANDLE_REQUEST_TIMEOUT_S)
        connection.connect("\0" + socket_name)
        if connection.send(request) != len(request):
            raise XpuIpcError(
                f"XPU IPC handle server {socket_name} took part of the request"
            )
        handle_blob, descriptors, flags, _ = socket.recv_fds(
            connection, XPU_IPC_HANDLE_BYTES, 1
        )
    truncated = flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC)
    if truncated or len(descriptors) != 1 or len(handle_blob) != XPU_IPC_HANDLE_BYTES:
        for descriptor in descriptors:
            os.close(descriptor)
        raise XpuIpcError(
            f"XPU IPC handle server {socket_name} returned {len(handle_blob)} handle "
            f"bytes and {len(descriptors)} fds (truncated: {bool(truncated)})"
        )
    return handle_blob, descriptors[0]


def _run_every_step(steps, description: str) -> None:
    """Run each teardown step even after one fails; raise the first failure."""
    first_error: Exception | None = None
    for step in steps:
        try:
            step()
        except Exception as error:
            if first_error is None:
                first_error = error
            else:
                logger.warning(f"{description}: another step also failed: {error}")
    if first_error is not None:
        raise first_error


class XpuIpcPoolReference(msgspec.Struct, frozen=True):
    """What an importer needs to map an exported pool.

    Plain fields, so it pickles and fits any control message.
    """

    socket_name: str
    segment_id: str
    segment_offset: int
    pool_bytes: int
    device_index: int


class XpuIpcExportedPool:
    """``num_bytes`` of device memory on one XPU, served to importers until close().

    The memory and its export belong to ``context``, which the caller destroys
    after close(). A pool that fails to come up gives back what it had made
    before the error propagates, so a caller can simply try again.
    """

    def __init__(
        self, context: LevelZeroIpcContext, device_index: int, num_bytes: int
    ) -> None:
        self._context = context
        pointer = context.allocate_exportable(device_index, num_bytes)
        segment: ExportedSegment | None = None
        try:
            tensor = tensor_from_device_pointer(pointer, num_bytes, device_index)
            segment = context.export_segment(pointer)
            server = XpuIpcHandleServer(segment)
        except BaseException:
            try:
                if segment is not None:
                    context.release_segment(segment)
            finally:
                context.free(pointer)
            raise
        self.tensor: torch.Tensor | None = tensor
        self.reference = XpuIpcPoolReference(
            socket_name=server.socket_name,
            segment_id=server.segment_id,
            segment_offset=pointer - segment.base_pointer,
            pool_bytes=num_bytes,
            device_index=device_index,
        )
        self._pointer = pointer
        self._segment = segment
        self._server: XpuIpcHandleServer | None = server

    def close(self) -> None:
        """Stop serving, release the handle and free the memory.

        Every step is attempted even after one fails. A peer that has mapped the
        pool keeps the memory alive until it unmaps.
        """
        server, self._server = self._server, None
        if server is None:
            return
        self.tensor = None
        _run_every_step(
            (
                server.close,
                lambda: self._context.release_segment(self._segment),
                lambda: self._context.free(self._pointer),
            ),
            "XPU IPC pool teardown",
        )


class XpuIpcPeerPools:
    """Pools exported by other processes, each mapped once into ``context``.

    open() blocks on the exporter's socket the first time a pool is seen and is
    safe to call from several threads. close() unmaps every pool; destroy the
    context only after it.
    """

    def __init__(self, context: LevelZeroIpcContext) -> None:
        self._context = context
        self._lock = threading.Lock()
        # segment id -> (reference, importing device, peer pointer, tensor)
        self._pools: dict[str, tuple[XpuIpcPoolReference, int, int, torch.Tensor]] = {}

    def open(self, reference: XpuIpcPoolReference, device_index: int) -> torch.Tensor:
        """The pool's bytes as a uint8 tensor on ``device_index``, mapped on first use."""
        with self._lock:
            cached = self._pools.get(reference.segment_id)
            if cached is not None:
                cached_reference, cached_device, _, tensor = cached
                if cached_reference != reference or cached_device != device_index:
                    raise ValueError(
                        f"XPU IPC pool at {reference.socket_name} is already mapped "
                        f"with another window or onto another device"
                    )
                return tensor
            peer_pointer, tensor = self._map(reference, device_index)
            self._pools[reference.segment_id] = (
                reference,
                device_index,
                peer_pointer,
                tensor,
            )
            return tensor

    def _map(
        self, reference: XpuIpcPoolReference, device_index: int
    ) -> tuple[int, torch.Tensor]:
        handle_blob, descriptor = fetch_xpu_ipc_handle(
            reference.socket_name, reference.segment_id
        )
        try:
            peer_pointer = self._context.open_peer_segment(
                handle_blob, descriptor, device_index
            )
        finally:
            # The driver holds its own reference once the mapping is open.
            os.close(descriptor)
        try:
            # The window comes from the exporter's message; it must fit the mapping.
            mapping = self._context.address_range(peer_pointer)
            mapped_bytes = mapping.base_pointer + mapping.segment_bytes - peer_pointer
            window_end = reference.segment_offset + reference.pool_bytes
            if (
                reference.segment_offset < 0
                or reference.pool_bytes <= 0
                or window_end > mapped_bytes
            ):
                raise ValueError(
                    f"XPU IPC pool at {reference.socket_name} claims bytes "
                    f"[{reference.segment_offset}, {window_end}) of a peer mapping "
                    f"that holds {mapped_bytes}"
                )
            tensor = tensor_from_device_pointer(
                peer_pointer + reference.segment_offset,
                reference.pool_bytes,
                device_index,
            )
        except BaseException:
            self._context.close_peer_segment(peer_pointer)
            raise
        return peer_pointer, tensor

    def close(self) -> None:
        """Unmap every pool, attempting each even after one fails."""
        with self._lock:
            pointers = [pool[2] for pool in self._pools.values()]
            self._pools.clear()
        _run_every_step(
            [
                lambda pointer=pointer: self._context.close_peer_segment(pointer)
                for pointer in pointers
            ],
            "XPU IPC peer pool teardown",
        )


_PEER_VISIBILITY_WARNED: set[tuple[int, int, int]] = set()


def ensure_xpu_peer_access(source_device_index: int, device_index: int) -> None:
    """Refuse a copy from ``source_device_index`` that ``device_index`` cannot do.

    A source outside this process' visible devices (another ZE_AFFINITY_MASK
    namespace) cannot be checked and is let through with a warning.
    """
    if source_device_index == device_index:
        return
    device_count = torch.xpu.device_count()
    if source_device_index >= device_count:
        warn_key = (device_index, source_device_index, device_count)
        if warn_key not in _PEER_VISIBILITY_WARNED:
            _PEER_VISIBILITY_WARNED.add(warn_key)
            logger.warning(
                f"XPU {source_device_index} is outside this process' visible range "
                f"[0, {device_count}), so peer access from XPU {device_index} was not "
                f"checked; expected only across ZE_AFFINITY_MASK namespaces"
            )
        return
    if not torch.xpu.can_device_access_peer(device_index, source_device_index):
        raise XpuIpcError(
            f"XPU {device_index} cannot access XPU {source_device_index} as a peer"
        )


def _synchronize_xpu_event(event: torch.xpu.Event, device_index: int) -> None:
    with torch.xpu.device(device_index):
        event.synchronize()


async def wait_for_xpu_event(
    event: torch.xpu.Event,
    *,
    device_index: int,
    executor: concurrent.futures.Executor,
    timeout: float,
) -> None:
    """Wait for an XPU event on ``executor`` without blocking the event loop.

    With no interprocess XPU event, an exporter waits for its own copy into a
    pool before telling an importer the bytes are there.
    """
    if event.query():
        return
    await asyncio.wait_for(
        asyncio.get_running_loop().run_in_executor(
            executor, _synchronize_xpu_event, event, device_index
        ),
        timeout=timeout,
    )
