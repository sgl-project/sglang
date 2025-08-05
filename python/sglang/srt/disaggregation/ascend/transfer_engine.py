import logging
import os
from typing import List, Optional

from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.disaggregation.utils import DisaggregationMode

logger = logging.getLogger(__name__)


class AscendTransferEngine(MooncakeTransferEngine):

    def __init__(
        self, hostname: str, npu_id: int, disaggregation_mode: DisaggregationMode
    ):
        try:
            from mf_adapter import TransferEngine
        except ImportError as e:
            raise ImportError(
                "Please install mf_adapter, for details, see docs/backend/pd_disaggregation.md"
            ) from e

        self.engine = TransferEngine()
        self.hostname = hostname
        self.npu_id = npu_id

        # Centralized storage address of the AscendTransferEngine
        self.store_url = os.getenv("ASCEND_MF_STORE_URL")
        if disaggregation_mode == DisaggregationMode.PREFILL:
            self.role = "Prefill"
        elif disaggregation_mode == DisaggregationMode.DECODE:
            self.role = "Decode"
        else:
            logger.error(f"Unsupported DisaggregationMode: {disaggregation_mode}")
            raise ValueError(f"Unsupported DisaggregationMode: {disaggregation_mode}")
        self.session_id = f"{self.hostname}:{self.engine.get_rpc_port()}"
        self.initialize()

    def initialize(self) -> None:
        """Initialize the ascend transfer instance."""
        ret_value = self.engine.initialize(
            self.store_url,
            self.session_id,
            self.role,
            self.npu_id,
        )
        if ret_value != 0:
            logger.error("Ascend Transfer Engine initialization failed.")
            raise RuntimeError("Ascend Transfer Engine initialization failed.")

    def batch_register(self, ptrs: List[int], lengths: List[int]):
        try:
            ret_value = self.engine.batch_register_memory(ptrs, lengths)
        except Exception:
            # Mark register as failed
            ret_value = -1
        if ret_value != 0:
            logger.debug(f"Ascend memory registration for ptr {ptrs} failed.")
