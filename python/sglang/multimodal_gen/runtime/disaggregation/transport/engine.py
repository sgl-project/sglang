# SPDX-License-Identifier: Apache-2.0
"""Transfer engine abstraction for tensor transfer between role instances."""

import ctypes
import logging
import socket
import struct
import threading
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

_MOONCAKE_AVAILABLE = None


def _check_mooncake() -> bool:
    """Probe whether the real Mooncake native extension loads on this host.

    This must import ``mooncake.engine`` itself, not just sglang's wrapper
    module (which always imports cleanly): the wrapper only raises once it
    tries the real import inside ``__init__``. Importing the wrapper here
    would make "auto" always look like Mooncake is available, defeating the
    fallback. This also means a future Mooncake release with a working
    non-CUDA (e.g. Intel XPU) build is picked up automatically the moment the
    import succeeds, with no change needed here.
    """
    global _MOONCAKE_AVAILABLE
    if _MOONCAKE_AVAILABLE is None:
        try:
            import mooncake.engine  # noqa: F401

            _MOONCAKE_AVAILABLE = True
        except ImportError:
            _MOONCAKE_AVAILABLE = False
    return _MOONCAKE_AVAILABLE


class BaseTransferEngine(ABC):
    """Abstract transfer engine for data movement between roles."""

    @property
    def supports_gpu_direct(self) -> bool:
        return False

    @property
    @abstractmethod
    def session_id(self) -> str: ...

    @abstractmethod
    def register_buffer(self, ptr: int, length: int) -> None: ...

    @abstractmethod
    def deregister_buffer(self, ptr: int) -> None: ...

    @abstractmethod
    def transfer_sync(
        self, dst_session_id: str, src_addr: int, dst_addr: int, length: int
    ) -> int:
        """Returns 0 on success, negative on failure."""

    @abstractmethod
    def batch_transfer_sync(
        self,
        dst_session_id: str,
        src_addrs: list[int],
        dst_addrs: list[int],
        lengths: list[int],
    ) -> int: ...


class MooncakeDiffusionEngine(BaseTransferEngine):
    """Production engine backed by MooncakeTransferEngine (RDMA)."""

    @property
    def supports_gpu_direct(self) -> bool:
        return True

    def __init__(
        self,
        hostname: str,
        gpu_id: int = 0,
        ib_device: str | None = None,
    ):
        from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
            MooncakeTransferEngine,
        )

        self._engine = MooncakeTransferEngine(
            hostname=hostname,
            gpu_id=gpu_id,
            ib_device=ib_device,
        )
        logger.info(
            "MooncakeDiffusionEngine initialized: session_id=%s",
            self._engine.session_id,
        )

    @property
    def session_id(self) -> str:
        return self._engine.session_id

    def register_buffer(self, ptr: int, length: int) -> None:
        self._engine.register(ptr, length)

    def deregister_buffer(self, ptr: int) -> None:
        self._engine.deregister(ptr)

    def transfer_sync(
        self, dst_session_id: str, src_addr: int, dst_addr: int, length: int
    ) -> int:
        return self._engine.transfer_sync(dst_session_id, src_addr, dst_addr, length)

    def batch_transfer_sync(
        self,
        dst_session_id: str,
        src_addrs: list[int],
        dst_addrs: list[int],
        lengths: list[int],
    ) -> int:
        return self._engine.batch_transfer_sync(
            dst_session_id, src_addrs, dst_addrs, lengths
        )


def _memoryview_from_ptr(ptr: int, length: int) -> memoryview:
    if length == 0:
        return memoryview(bytearray())
    return memoryview((ctypes.c_ubyte * length).from_address(ptr))


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ConnectionError(
                "MockTransferEngine: peer closed before sending all data"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_into_exact(conn: socket.socket, view: memoryview) -> None:
    offset = 0
    remaining = len(view)
    while remaining > 0:
        n = conn.recv_into(view[offset:], remaining)
        if n == 0:
            raise ConnectionError(
                "MockTransferEngine: peer closed before sending all data"
            )
        offset += n
        remaining -= n


class MockTransferEngine(BaseTransferEngine):
    """Non-RDMA fallback: moves bytes over a plain TCP loopback/LAN socket.

    Used when Mooncake is unavailable (e.g. its wheel links libcudart and
    there is no CUDA runtime, as on Intel XPU hosts) or when the backend is
    explicitly set to "mock". ``supports_gpu_direct`` stays False (the base
    class default) so callers stage through a CPU pinned buffer instead of a
    device buffer.
    """

    _HEADER = struct.Struct("!QQ")  # dst_addr, length

    def __init__(self, hostname: str = "127.0.0.1", port: int = 0):
        self._lock = threading.Lock()
        self._local_ptr = 0
        self._local_len = 0

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((hostname, port))
        self._server.listen(16)
        bound_port = self._server.getsockname()[1]
        self._session_id = f"{hostname}:{bound_port}"

        self._stop = threading.Event()
        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="mock-transfer-engine-accept"
        )
        self._accept_thread.start()
        logger.info("MockTransferEngine initialized: session_id=%s", self._session_id)

    @property
    def session_id(self) -> str:
        return self._session_id

    def register_buffer(self, ptr: int, length: int) -> None:
        with self._lock:
            self._local_ptr = ptr
            self._local_len = length

    def deregister_buffer(self, ptr: int) -> None:
        with self._lock:
            self._local_ptr = 0
            self._local_len = 0
        self._stop.set()
        self._server.close()

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            threading.Thread(
                target=self._serve_conn, args=(conn,), daemon=True
            ).start()

    def _serve_conn(self, conn: socket.socket) -> None:
        try:
            dst_addr, length = self._HEADER.unpack(_recv_exact(conn, self._HEADER.size))
            with self._lock:
                ptr, size = self._local_ptr, self._local_len
            if dst_addr < ptr or dst_addr + length > ptr + size:
                raise ValueError(
                    f"dst_addr {dst_addr}+{length} outside registered buffer "
                    f"[{ptr}, {ptr + size})"
                )
            _recv_into_exact(conn, _memoryview_from_ptr(dst_addr, length))
            conn.sendall(b"\x01")
        except Exception:
            logger.exception("MockTransferEngine: failed to serve incoming transfer")
        finally:
            conn.close()

    def transfer_sync(
        self, dst_session_id: str, src_addr: int, dst_addr: int, length: int
    ) -> int:
        try:
            host, port_str = dst_session_id.rsplit(":", 1)
            with socket.create_connection((host, int(port_str))) as conn:
                conn.sendall(self._HEADER.pack(dst_addr, length))
                conn.sendall(_memoryview_from_ptr(src_addr, length))
                if _recv_exact(conn, 1) != b"\x01":
                    raise ValueError("peer did not ack the transfer")
            return 0
        except Exception:
            logger.exception(
                "MockTransferEngine: transfer_sync to %s failed", dst_session_id
            )
            return -1

    def batch_transfer_sync(
        self,
        dst_session_id: str,
        src_addrs: list[int],
        dst_addrs: list[int],
        lengths: list[int],
    ) -> int:
        for src_addr, dst_addr, length in zip(src_addrs, dst_addrs, lengths):
            ret = self.transfer_sync(dst_session_id, src_addr, dst_addr, length)
            if ret != 0:
                return ret
        return 0


def create_transfer_engine(
    hostname: str = "127.0.0.1",
    gpu_id: int = 0,
    ib_device: str | None = None,
    backend: str = "auto",
) -> BaseTransferEngine:
    """Factory: builds the transfer engine for ``backend``.

    "mooncake" requires the mooncake package and raises if it is missing.
    "mock" is the non-RDMA TCP fallback, usable on any device/host. "auto"
    prefers mooncake and falls back to the mock engine when mooncake is
    unavailable.
    """
    if backend == "mooncake":
        if not _check_mooncake():
            raise RuntimeError(
                "Mooncake transfer engine is required for disaggregated diffusion "
                "but is not installed. Please install mooncake first."
            )
        return MooncakeDiffusionEngine(
            hostname=hostname, gpu_id=gpu_id, ib_device=ib_device
        )

    if backend == "mock":
        return MockTransferEngine(hostname=hostname)

    if backend == "auto":
        if _check_mooncake():
            return MooncakeDiffusionEngine(
                hostname=hostname, gpu_id=gpu_id, ib_device=ib_device
            )
        logger.warning(
            "Mooncake transfer engine not installed; falling back to the "
            "non-RDMA mock transfer engine. Install mooncake-transfer-engine "
            "for RDMA/GPUDirect transfers."
        )
        return MockTransferEngine(hostname=hostname)

    raise ValueError(f"Unknown disagg transfer backend: {backend!r}")
