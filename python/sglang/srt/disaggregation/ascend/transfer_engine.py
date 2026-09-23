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

_DEFAULT_PROTOCOL = "sdma"

# Under host_rdma, MemFabric derives each rank's listen port from the base
# port it is given (it binds base_port + its group rank, 0..7), so every
# worker must own a disjoint stride of ports. Decode and Prefill get
# disjoint strides so both roles can run on the same host (up to 16
# workers per role per host).
_HOST_RDMA_PORTS_PER_WORKER = 8
_HOST_RDMA_ROLE_PORT_STRIDE = 128


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
        if transfer_protocol == "device_rdma":
            # with device RDMA for PD transfer: initialize hccl in advance
            # through all_gather to avoid conflicts with rdma initialization.
            tmp_tensor = torch.zeros(1, device="npu")
            output_tensor_list = [
                torch.empty_like(tmp_tensor) for _ in range(get_world_size())
            ]
            torch.distributed.all_gather(
                output_tensor_list, tmp_tensor, group=get_world_group().device_group
            )

        trans_op_type = self._resolve_trans_op_type(transfer_protocol)

        # Under host_rdma, the framework derives each worker's listen port from
        # the user-provided NIC endpoint: MemFabric binds base_port + group
        # rank within a worker's stride, and the multi-rank topology is only
        # known here, so the per-worker base ports are assigned in the
        # framework rather than by the engine.
        nic = os.getenv("ASCEND_MF_NIC")
        if transfer_protocol == "host_rdma":
            if not nic:
                raise ValueError(
                    "ASCEND_MF_NIC (IP:PORT or tcp://IP:PORT) must be set for "
                    "host_rdma; otherwise the engine binds a loopback endpoint "
                    "that peers cannot reach."
                )
            nic = self._derive_worker_nic(nic)

        """Initialize the ascend transfer instance."""
        ret_value = self.engine.initialize(
            self.store_url,
            self.session_id,
            self.role,
            self.npu_id,
            trans_op_type,
            nic=nic,
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

    def _derive_worker_nic(self, nic: str) -> str:
        """Derive this worker's HOST_RDMA endpoint from the configured base.

        The user configures one base endpoint in ASCEND_MF_NIC; each worker
        gets base_port + role offset + npu_id * stride. MemFabric then binds
        ports within [worker_port, worker_port + stride).
        """
        host, sep, port_str = nic.rpartition(":")
        try:
            base_port = int(port_str)
        except ValueError:
            raise ValueError(
                f"Invalid port in ASCEND_MF_NIC: {nic!r} (expected IP:PORT)"
            ) from None
        if not sep or not base_port:
            raise ValueError(
                f"Invalid ASCEND_MF_NIC: {nic!r} (expected IP:PORT)"
            )

        role_offset = (
            _HOST_RDMA_ROLE_PORT_STRIDE if self.role == "Prefill" else 0
        )
        worker_port = (
            base_port + role_offset + self.npu_id * _HOST_RDMA_PORTS_PER_WORKER
        )
        if not (
            1024
            <= worker_port
            and worker_port + _HOST_RDMA_PORTS_PER_WORKER - 1 <= 65535
        ):
            raise ValueError(
                f"Resolved HOST_RDMA port out of range: base_port={base_port}, "
                f"npu_id={self.npu_id}, role={self.role}, "
                f"resolved_range={worker_port}-"
                f"{worker_port + _HOST_RDMA_PORTS_PER_WORKER - 1}"
            )

        endpoint = f"{host}:{worker_port}"
        logger.info(
            "HOST_RDMA endpoint: role=%s, npu_id=%s, base=%s, endpoint=%s",
            self.role,
            self.npu_id,
            nic,
            endpoint,
        )
        return endpoint

    @staticmethod
    def _get_transfer_protocol() -> str:
        protocol = os.getenv("ASCEND_MF_TRANSFER_PROTOCOL")
        return protocol.strip().lower() if protocol else _DEFAULT_PROTOCOL

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
