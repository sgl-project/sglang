import logging
from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.utils import get_free_port
from typing import List, Optional
import torch_npu

logger = logging.getLogger(__name__)


class AscendTransferEngine(MooncakeTransferEngine):

    def __init__(self, hostname: str, ascend_url: str, npu_id: int, disaggregation_mode: DisaggregationMode,
                 ib_device: str, ascend_mooncake: bool):
        if not ascend_mooncake:
            try:
                from mf_adapter import TransferEngine
            except ImportError as e:
                raise ImportError(
                    "Please install AscendTransferEngine, run SGLang with AscendTransferEngine"
                ) from e
        else:
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
        self.npu_id = npu_id
        self.ib_device = ib_device
        self.ascend_mooncake = ascend_mooncake

        # Centralized storage address of the AscendTransferEngine
        self.store_url = ascend_url
        if disaggregation_mode == DisaggregationMode.PREFILL:
            self.role = "Prefill"
        elif disaggregation_mode == DisaggregationMode.DECODE:
            self.role = "Decode"
        else:
            logger.error(f"Unsupported DisaggregationMode: {disaggregation_mode}")
            raise ValueError(
                f"Unsupported DisaggregationMode: {disaggregation_mode}"
            )
        if not self.ascend_mooncake:
            self.session_id = f"{self.hostname}:{self.engine.get_rpc_port()}"
        self.initialize(
            hostname=self.hostname,
            device_name=self.ib_device
        )
        if self.ascend_mooncake:
            self.session_id = f"{self.hostname}:{self.engine.get_rpc_port()}"

    def initialize(
        self,
        hostname: str,
        device_name: Optional[str]
    ) -> None:
        """Initialize the ascend transfer instance."""
        if self.ascend_mooncake:
            hostname += f":{get_free_port()}:npu_{self.npu_id}"
            ret_value = self.engine.initialize(
                hostname,
                "P2PHANDSHAKE",
                "ascend",
                device_name if device_name is not None else "",
            )
        else:
            ret_value = self.engine.initialize(
                self.store_url,
                self.session_id,
                self.role,
                self.npu_id,
            )
        if ret_value != 0:
            logger.error("Ascend Transfer Engine initialization failed.")
            raise RuntimeError("Ascend Transfer Engine initialization failed.")

    def batch_register(self, ptr, length):
        try:
            ret_value = self.engine.batch_register_memory(ptr, length)
        except Exception:
            # Mark register as failed
            ret_value = -1
        if ret_value != 0:
            logger.debug(f"Ascend memory  registration %s failed.", ptr)

    def transfer_sync(
        self, session_id: str, buffer: int, peer_buffer_address: int, length: int
    ) -> int:
        """Synchronously transfer data to the specified address."""
        try:
            # the first time: based on session_id (which contains remote_ip) to construct a queue pair, and cache the queue pair
            # later: based on the cached queue pair to send data
            torch_npu.npu.set_device(self.npu_id)
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

    def batch_transfer_sync(
        self,
        session_id: str,
        buffers: List[int],
        peer_buffer_addresses: List[int],
        lengths: List[int],
    ) -> int:
        """Synchronously transfer data to the specified addresses in batches."""
        try:
            torch_npu.npu.set_device(self.npu_id)
            ret = self.engine.batch_transfer_sync_write(
                session_id, buffers, peer_buffer_addresses, lengths
            )
        except Exception:
            ret = -1
            # Inform user to upgrade mooncake-transfer-engine >= 0.3.4.post2

        if ret < 0:
            logger.debug(
                "Failed to batch transfer data. Buffers: %s, Session: %s, Peer addresses: %s",
                buffers,
                session_id,
                peer_buffer_addresses,
            )
        return ret
