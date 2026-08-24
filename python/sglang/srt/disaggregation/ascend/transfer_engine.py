import logging
import os
from typing import List

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from sglang.srt.utils.network import NetworkAddress

try:
    from memfabric_hybrid import TransferEngine

    import_error = None
except ImportError as e:
    import_error = e
    pass

logger = logging.getLogger(__name__)


class AscendTransferEngine(MooncakeTransferEngine):

    def __init__(
        self,
        hostname: str,
        npu_id: int,
        disaggregation_mode: DisaggregationMode,
    ):
        if import_error is not None:
            logger.warning(
                "Please install memfabric_hybrid, for details, see docs/docs/advanced_features/pd_disaggregation.mdx"
            )
            raise import_error

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
        rpc_port = self.engine.get_rpc_port()
        self.session_id = NetworkAddress(self.hostname, rpc_port).to_host_port_str()
        self.initialize()
        if rpc_port == 0:
            rpc_port = self.engine.get_rpc_port()
            self.session_id = NetworkAddress(self.hostname, rpc_port).to_host_port_str()

    def initialize(self) -> None:
        from sglang.srt.distributed.parallel_state import (
            get_world_group,
            get_world_size,
        )

        transfer_protocol = self._get_transfer_protocol()
        if transfer_protocol is None or transfer_protocol == "sdma":
            trans_op_type = TransferEngine.TransDataOpType.SDMA
        else:
            trans_op_type = TransferEngine.TransDataOpType.DEVICE_RDMA
            """with device RDMA for PD transfer"""
            tmp_tensor = torch.zeros(1, device="npu")
            output_tensor_list = [
                torch.empty_like(tmp_tensor) for _ in range(get_world_size())
            ]
            # Initialize hccl in advance through all_gather to avoid conflicts with rdma initialization.
            torch.distributed.all_gather(
                output_tensor_list, tmp_tensor, group=get_world_group().device_group
            )
        """Initialize the ascend transfer instance."""
        ret_value = self.engine.initialize(
            self.store_url, self.session_id, self.role, self.npu_id, trans_op_type
        )
        if ret_value != 0:
            logger.error("Ascend Transfer Engine initialization failed.")
            raise RuntimeError("Ascend Transfer Engine initialization failed.")

    def batch_register(self, ptrs: List[int], lengths: List[int]):
        # Regions the ENGINE confirmed failed, for the caller's bookkeeping
        # (strict exclusion, VA confrontation in logs). Reset per call.
        self.last_failed_regions: List[tuple] = []
        try:
            ret_value = self.engine.batch_register_memory(ptrs, lengths)
        except Exception as e:
            # Mark register as failed
            logger.error(
                "Ascend memory registration raised (%s); %d regions, %d bytes "
                "total. Any later transfer touching them faults with SDMA "
                "smmu-terminate.",
                e,
                len(ptrs),
                sum(lengths),
            )
            ret_value = -1
        if ret_value != 0:
            logger.error(
                "Ascend memory registration failed ret=%s; %d regions, %d "
                "bytes total. Transfers into unregistered regions fault "
                "with SDMA smmu-terminate.",
                ret_value,
                len(ptrs),
                sum(lengths),
            )
            # Bisect the failure: a batch register is all-or-nothing in the
            # engine, so re-register every region individually. Succeeded
            # ones are now explicitly in; the failures are pinned to exact
            # VA ranges -- the [mf-reg] lines let the offline analyzer
            # confront them with the dirty-row addresses / r2t pool VA.
            for p, l in zip(ptrs, lengths):
                try:
                    r = self.engine.batch_register_memory([p], [l])
                except Exception:
                    r = -1
                if r != 0:
                    self.last_failed_regions.append((p, l))
                    logger.error(
                        "[mf-reg] FAIL ptr=0x%x len=%d end=0x%x", p, l, p + l
                    )
                else:
                    logger.info(
                        "[mf-reg] ok(ptr) ptr=0x%x len=%d end=0x%x", p, l, p + l
                    )
            if self.last_failed_regions:
                logger.error(
                    "[mf-reg] summary failed=%d/%d bytes_failed=%d",
                    len(self.last_failed_regions),
                    len(ptrs),
                    sum(l for _p, l in self.last_failed_regions),
                )
        else:
            logger.info(
                "Ascend memory registration ok: %d regions, %d bytes total, "
                "VA span [0x%x, 0x%x).",
                len(ptrs),
                sum(lengths),
                min(ptrs),
                max(p + l for p, l in zip(ptrs, lengths)),
            )

    @staticmethod
    def _get_transfer_protocol():
        protocol = os.getenv("ASCEND_MF_TRANSFER_PROTOCOL")
        allowed_protocols = {"device_rdma", "sdma"}
        if protocol and protocol.lower() in allowed_protocols:
            return protocol.lower()
        else:
            logger.warning(
                "Invalid or no transfer protocol specified, using default protocol."
            )
            return None
