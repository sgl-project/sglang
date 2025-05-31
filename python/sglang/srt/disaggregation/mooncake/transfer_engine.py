import json
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class MooncakeTransferEngine:

    def __init__(self, hostname: str, gpu_id: int, ib_device: Optional[str] = None):
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run SGLang with MooncakeTransferEngine."
            ) from e

        self.engine = TransferEngine()
        self.hostname = hostname
        self.gpu_id = gpu_id
        self.ib_device = ib_device

        self.initialize(
            hostname=self.hostname,
            device_name=self.ib_device,
        )
        self.session_id = f"{self.hostname}:{self.engine.get_rpc_port()}"

    def register(self, ptr, length):
        try:
            ret_value = self.engine.register_memory(ptr, length)
        except Exception:
            # Mark register as failed
            ret_value = -1

        if ret_value != 0:
            logger.debug("Mooncake memory registration %s failed.", ptr)

    def deregister(self, ptr):
        try:
            ret_value = self.engine.unregister_memory(ptr)
        except Exception:
            # Mark deregister as failed
            ret_value = -1

        if ret_value != 0:
            logger.debug("Mooncake memory deregistration %s failed.", ptr)

    def initialize(
        self,
        hostname: str,
        device_name: Optional[str],
    ) -> None:
        """Initialize the mooncake instance."""
        ret_value = self.engine.initialize(
            hostname,
            "P2PHANDSHAKE",
            "rdma",
            device_name if device_name is not None else "",
        )
        if ret_value != 0:
            logger.error("Mooncake Transfer Engine initialization failed.")
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

    def transfer_sync(
        self, session_id: str, buffer: int, peer_buffer_address: int, length: int
    ) -> int:
        """Synchronously transfer data to the specified address."""
        try:
            # the first time: based on session_id (which contains remote_ip) to construct a queue pair, and cache the queue pair
            # later: based on the cached queue pair to send data
            ret = self.engine.transfer_sync_write(
                session_id, buffer, peer_buffer_address, length
            )
        except Exception:
            # Mark transfer request as failed
            ret = -1

        if ret < 0:
            # Do not raise an exception here, since some transfer requests fail should be accepted and the execution thread should not be stopped.
            logger.debug(
                "Failed to transfer data from %s to %s - %s.",
                buffer,
                session_id,
                peer_buffer_address,
            )

        return ret

    def get_session_id(self):
        return self.session_id
