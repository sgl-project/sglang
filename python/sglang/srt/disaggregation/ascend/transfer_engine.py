import logging
import os
from typing import List

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from sglang.srt.environ import envs
from sglang.srt.utils.network import NetworkAddress

try:
    from memfabric_hybrid import TransferEngine

    import_error = None
except ImportError as e:
    import_error = e
    pass

logger = logging.getLogger(__name__)

_DEFAULT_PROTOCOL = "sdma"

# MemFabric derives its data-plane port from the worker nic base port, so each
# worker owns a stride of ports. Decode and Prefill get disjoint strides so
# both roles can run on the same host.
_WORKER_PORT_STRIDE = 8
_ROLE_PORT_STRIDE = 128


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
        from sglang.srt.runtime_context import get_parallel

        transfer_protocol = self._get_transfer_protocol()
        nic = ""
        if transfer_protocol == "device_rdma":
            # with device RDMA for PD transfer: initialize hccl in advance
            # through all_gather to avoid conflicts with rdma initialization.
            tmp_tensor = torch.zeros(1, device="npu")
            output_tensor_list = [
                torch.empty_like(tmp_tensor)
                for _ in range(get_parallel().launch_world_size)
            ]
            torch.distributed.all_gather(
                output_tensor_list,
                tmp_tensor,
                group=get_parallel().world_group.device_group,
            )
        elif transfer_protocol == "host_rdma":
            # host_rdma binds its data plane on the nic endpoint; memfabric
            # would otherwise fall back to a loopback port peers cannot reach.
            nic = self._resolve_worker_hcom_url(
                envs.ASCEND_MF_HCOM_URL.get(),
                self.role,
                get_parallel().world_group.rank_in_group,
            )
        trans_op_type = self._resolve_trans_op_type(transfer_protocol)
        """Initialize the ascend transfer instance."""
        initialize_kwargs = {}
        if nic:
            # The nic argument exists only in memfabric_hybrid > v1.2.1
            # (added with host rdma); older engines reject the keyword.
            initialize_kwargs["nic"] = nic
        ret_value = self.engine.initialize(
            self.store_url,
            self.session_id,
            self.role,
            self.npu_id,
            trans_op_type,
            **initialize_kwargs,
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

    @staticmethod
    def _get_transfer_protocol() -> str:
        protocol = os.getenv("ASCEND_MF_TRANSFER_PROTOCOL")
        return protocol.strip().lower() if protocol else _DEFAULT_PROTOCOL

    @staticmethod
    def _resolve_worker_hcom_url(hcom_url: str, role: str, world_rank: int) -> str:
        if not hcom_url:
            raise ValueError(
                "ASCEND_MF_HCOM_URL (tcp://<rdma-nic-ip>:<port>) is required "
                "for host_rdma; memfabric would otherwise bind a loopback "
                "endpoint that peers cannot reach."
            )

        address, separator, port_str = hcom_url.rpartition(":")
        if not separator or not address.startswith("tcp://"):
            raise ValueError(f"Invalid port in ASCEND_MF_HCOM_URL: {hcom_url!r}")

        try:
            base_port = int(port_str)
        except ValueError as exc:
            raise ValueError(
                f"Invalid port in ASCEND_MF_HCOM_URL: {hcom_url!r}"
            ) from exc

        role_offset = 0 if role == "Decode" else _ROLE_PORT_STRIDE
        worker_port = base_port + role_offset + world_rank * _WORKER_PORT_STRIDE
        if not (1024 <= worker_port and worker_port + _WORKER_PORT_STRIDE - 1 <= 65535):
            raise ValueError(
                "Resolved ASCEND_MF_HCOM_URL port is out of range: "
                f"base_port={base_port}, world_rank={world_rank}, "
                f"role={role}, "
                f"resolved_port_range={worker_port}-{worker_port + _WORKER_PORT_STRIDE - 1}"
            )

        worker_hcom_url = f"{address}:{worker_port}"
        logger.info(
            "Resolved Ascend Host RDMA endpoint: role=%s, world_rank=%d, "
            "base=%s, endpoint=%s",
            role,
            world_rank,
            hcom_url,
            worker_hcom_url,
        )
        return worker_hcom_url

    @staticmethod
    def _resolve_trans_op_type(protocol: str):
        op_type = getattr(TransferEngine.TransDataOpType, protocol.upper(), None)
        if op_type is None:
            logger.warning(
                "Transfer protocol %r is not supported by the installed "
                "memfabric_hybrid, falling back to %r.",
                protocol,
                _DEFAULT_PROTOCOL,
            )
            op_type = TransferEngine.TransDataOpType.SDMA
        return op_type
