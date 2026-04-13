# SPDX-License-Identifier: Apache-2.0
"""Transfer engine abstraction for tensor transfer between role instances."""

import logging
import threading
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

_MOONCAKE_AVAILABLE = None


def _check_mooncake() -> bool:
    global _MOONCAKE_AVAILABLE
    if _MOONCAKE_AVAILABLE is None:
        try:
            from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (  # noqa: F401
                MooncakeTransferEngine as _MTE,
            )

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


class MockTransferEngine(BaseTransferEngine):
    """In-process mock for unit testing. Simulates RDMA via ctypes memmove."""

    # Shared registry so mock instances can "see" each other's buffers
    _registry: dict[str, dict[int, tuple[object, int]]] = {}
    _lock = threading.Lock()
    _counter = 0

    def __init__(self, session_id: str | None = None):
        with MockTransferEngine._lock:
            if session_id is None:
                MockTransferEngine._counter += 1
                session_id = f"mock-session-{MockTransferEngine._counter}"
            self._session_id = session_id
            MockTransferEngine._registry[session_id] = {}

    @property
    def session_id(self) -> str:
        return self._session_id

    def register_buffer(self, ptr: int, length: int) -> None:
        with MockTransferEngine._lock:
            self._registry[self._session_id][ptr] = (None, length)

    def deregister_buffer(self, ptr: int) -> None:
        with MockTransferEngine._lock:
            self._registry[self._session_id].pop(ptr, None)

    def transfer_sync(
        self, dst_session_id: str, src_addr: int, dst_addr: int, length: int
    ) -> int:
        import ctypes

        try:
            ctypes.memmove(dst_addr, src_addr, length)
            return 0
        except Exception as e:
            logger.error("MockTransferEngine transfer failed: %s", e)
            return -1

    def batch_transfer_sync(
        self,
        dst_session_id: str,
        src_addrs: list[int],
        dst_addrs: list[int],
        lengths: list[int],
    ) -> int:
        for src, dst, length in zip(src_addrs, dst_addrs, lengths):
            ret = self.transfer_sync(dst_session_id, src, dst, length)
            if ret != 0:
                return ret
        return 0

    @classmethod
    def reset(cls):
        """Reset global registry (for test cleanup)."""
        with cls._lock:
            cls._registry.clear()
            cls._counter = 0


def create_transfer_engine(
    hostname: str = "127.0.0.1",
    gpu_id: int = 0,
    ib_device: str | None = None,
    force_mock: bool = False,
) -> BaseTransferEngine:
    """Factory: returns MooncakeDiffusionEngine if available, else MockTransferEngine."""
    if not force_mock and _check_mooncake():
        return MooncakeDiffusionEngine(
            hostname=hostname, gpu_id=gpu_id, ib_device=ib_device
        )
    logger.info(
        "Using MockTransferEngine (mooncake not available or force_mock=%s)",
        force_mock,
    )
    return MockTransferEngine()
