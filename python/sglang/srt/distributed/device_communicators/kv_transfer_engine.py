"""Transport-agnostic RDMA engine interface shared by PD, encoder and diffusion disaggregation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional


class TransferIntent(str, Enum):
    """What a transfer is for; engines that support QoS map it to an isolated lane."""

    # On a request's critical path, e.g. PD KV cache the decode side waits for.
    FOREGROUND = "foreground"
    # Throughput traffic nobody waits on now, e.g. storage write-back or prefetch.
    BACKGROUND = "background"


class KVTransferEngine(ABC):
    """One engine per process; peers address each other by `session_id` ("host:port").

    Transfer methods return 0 on success and a negative value on failure, so
    connectors can treat every backend the same way.
    """

    session_id: str
    ib_device: Optional[str]

    @abstractmethod
    def register(self, ptr: int, length: int) -> None: ...

    @abstractmethod
    def deregister(self, ptr: int) -> None: ...

    @abstractmethod
    def batch_register(self, ptrs: List[int], lengths: List[int]) -> int: ...

    @abstractmethod
    def batch_deregister(self, ptrs: List[int]) -> int: ...

    @abstractmethod
    def batch_transfer_sync(
        self,
        session_id: str,
        buffers: List[int],
        peer_buffer_addresses: List[int],
        lengths: List[int],
        intent: Optional[TransferIntent] = None,
    ) -> int:
        """RDMA-write local `buffers` to the peer's addresses; blocks until done.

        `intent` is a hint; engines without QoS support ignore it. A negative
        return does not mean nothing was written: some writes may still land
        on the peer, so the caller must treat the target pages as unsettled.
        """

    def transfer_sync(
        self,
        session_id: str,
        buffer: int,
        peer_buffer_address: int,
        length: int,
        intent: Optional[TransferIntent] = None,
    ) -> int:
        return self.batch_transfer_sync(
            session_id, [buffer], [peer_buffer_address], [length], intent
        )

    @abstractmethod
    def send_probe(self, peer_session_id: str) -> int:
        """Return 0 if the peer is reachable."""

    def get_session_id(self) -> str:
        return self.session_id

    def get_ib_device(self) -> Optional[str]:
        return self.ib_device
