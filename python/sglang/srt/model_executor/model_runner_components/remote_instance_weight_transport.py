from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto

logger = logging.getLogger(__name__)


# Lifecycle fields (engine / session_id / weight_info / _nixl_manager)
# are written across multiple methods after construction — explicit R5
# exception, hence `slots=True, kw_only=True` without `frozen=True`.
@dataclass(slots=True, kw_only=True)
class RemoteInstanceWeightTransport:
    server_args: ServerArgs
    get_model: Callable[[], torch.nn.Module]
    tp_rank: int
    gpu_id: int
    engine: Optional[Any] = None
    session_id: str = ""
    weight_info: Optional[dict[str, tuple[int, int, int]]] = None
    _nixl_manager: Optional[Any] = None

    @property
    def model(self) -> torch.nn.Module:
        # Always read through the getter — ModelRunner may swap ``self.model``
        # during weight reload, so a captured object reference would go stale.
        return self.get_model()

    def init_engine(self):
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            logger.warning(
                "Please install mooncake for using remote instance transfer engine: pip install mooncake"
            )
            return
        self.engine = TransferEngine()
        local_ip = get_local_ip_auto()
        self.engine.initialize(
            local_ip, "P2PHANDSHAKE", "rdma", envs.MOONCAKE_DEVICE.get()
        )
        self.session_id = NetworkAddress(
            local_ip, self.engine.get_rpc_port()
        ).to_host_port_str()

    def _register_to_engine_info_bootstrap(self):
        """Register transfer engine info with the EngineInfoBootstrapServer via HTTP PUT.

        The bootstrap server runs on node_rank==0. For multi-node setups, the
        host is derived from dist_init_addr. For single-node, use 127.0.0.1.
        """
        import requests as http_requests

        if self.server_args.dist_init_addr:
            # Multi-node: bootstrap server is on the head node (node_rank==0).
            # Derive host from dist_init_addr (shared across all nodes).
            bootstrap_host = (
                NetworkAddress.parse(self.server_args.dist_init_addr).resolved().host
            )
        else:
            bootstrap_host = "127.0.0.1"

        bootstrap_port = self.server_args.engine_info_bootstrap_port
        bootstrap_na = NetworkAddress(bootstrap_host, bootstrap_port)
        url = f"{bootstrap_na.to_url()}/register_transfer_engine_info"

        payload = {
            "tp_rank": self.tp_rank,
            "transfer_engine_info": {
                "session_id": self.session_id,
                "weights_info_dict": self.weight_info,
            },
        }

        try:
            resp = http_requests.put(url, json=payload, timeout=5)
            if resp.status_code == 200:
                logger.info(
                    f"Registered transfer engine info for tp_rank={self.tp_rank} "
                    f"with bootstrap server at {bootstrap_na}"
                )
            else:
                logger.error(
                    f"Failed to register transfer engine info for tp_rank={self.tp_rank}: "
                    f"{resp.status_code}, {resp.text}"
                )
        except Exception as e:
            logger.error(
                f"Failed to register transfer engine info for tp_rank={self.tp_rank}: {e}"
            )

    def _publish_modelexpress_metadata(self):
        """Publish metadata to ModelExpress server (seed mode).

        Supports two transport backends:
        - transfer_engine: publishes TransferEngine session_id (Mooncake)
        - nixl: creates NIXL agent, registers tensors, publishes nixl_metadata
        """
        try:
            from modelexpress import p2p_pb2
            from modelexpress.client import MxClient
        except ImportError as exc:
            raise ImportError(
                "ModelExpress support requires the 'modelexpress' package. "
                "Install it with: pip install modelexpress"
            ) from exc

        model_name = (
            self.server_args.modelexpress_model_name or self.server_args.model_path
        )
        mx_url = self.server_args.modelexpress_url
        transport = self.server_args.modelexpress_transport

        # Build SourceIdentity for this instance
        identity = p2p_pb2.SourceIdentity(
            model_name=model_name,
            backend_framework=p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
            tensor_parallel_size=self.server_args.tp_size,
            pipeline_parallel_size=self.server_args.pp_size,
            expert_parallel_size=self.server_args.ep_size,
            dtype=self.server_args.dtype or "",
            quantization=self.server_args.quantization or "",
        )

        if transport == "nixl":
            worker, tensor_count = self._build_nixl_worker_metadata(p2p_pb2)
        else:
            worker, tensor_count = self._build_transfer_engine_worker_metadata(p2p_pb2)
            if worker is None:
                return

        # Generate a unique worker_id for this running instance
        worker_id = str(uuid.uuid4())

        mx_client = MxClient(server_url=mx_url)
        try:
            logger.info(
                "ModelExpress source [%s]: publishing metadata for model=%s, "
                "tp_rank=%d, %d tensors, worker_id=%s",
                transport,
                model_name,
                self.tp_rank,
                tensor_count,
                worker_id,
            )
            mx_source_id = mx_client.publish_metadata(identity, worker, worker_id)
            mx_client.update_status(
                mx_source_id=mx_source_id,
                worker_id=worker_id,
                worker_rank=self.tp_rank,
                status=p2p_pb2.SOURCE_STATUS_READY,
            )
            logger.info(
                "ModelExpress source: published ready for model=%s, "
                "tp_rank=%d, mx_source_id=%s",
                model_name,
                self.tp_rank,
                mx_source_id,
            )
        finally:
            mx_client.close()

    def _build_transfer_engine_worker_metadata(
        self: "RemoteInstanceWeightTransport", p2p_pb2
    ):
        """Build WorkerMetadata using TransferEngine session_id."""
        session_id = self.session_id
        weight_info = self.weight_info

        if not session_id or weight_info is None:
            logger.warning(
                "ModelExpress source: skipping publish -- "
                "TransferEngine not initialized or no weight info"
            )
            return None, 0

        tensors = []
        for name, (addr, numel, element_size) in weight_info.items():
            tensors.append(
                p2p_pb2.TensorDescriptor(
                    name=name,
                    addr=addr,
                    size=numel * element_size,
                    device_id=self.gpu_id,
                )
            )

        worker = p2p_pb2.WorkerMetadata(
            worker_rank=self.tp_rank,
            transfer_engine_session_id=session_id,
            tensors=tensors,
        )
        return worker, len(tensors)

    def _build_nixl_worker_metadata(self, p2p_pb2):
        """Build WorkerMetadata using NIXL agent for RDMA transfers."""
        from modelexpress.nixl_transfer import NixlTransferManager

        agent_name = f"sglang-seed-rank{self.tp_rank}-{uuid.uuid4().hex[:8]}"
        nixl_mgr = NixlTransferManager(agent_name, self.gpu_id)
        nixl_mgr.initialize()

        # Collect model tensors for NIXL registration
        model_tensors = {}
        for name, param in self.model.named_parameters():
            t = param.data
            if t.is_contiguous():
                model_tensors[name] = t
            else:
                # Non-contiguous tensors: register underlying storage as byte view
                sv = torch.empty(0, dtype=torch.uint8, device=t.device).set_(
                    t.untyped_storage()
                )
                if sv.data_ptr() not in {v.data_ptr() for v in model_tensors.values()}:
                    model_tensors[f"{name}.__storage"] = sv

        nixl_metadata = nixl_mgr.register_tensors(model_tensors)

        # Build tensor descriptors from registered tensors
        tensors = []
        for td in nixl_mgr.tensor_descriptors:
            tensors.append(
                p2p_pb2.TensorDescriptor(
                    name=td.name,
                    addr=td.addr,
                    size=td.size,
                    device_id=td.device_id,
                    dtype=td.dtype,
                )
            )

        worker = p2p_pb2.WorkerMetadata(
            worker_rank=self.tp_rank,
            nixl_metadata=nixl_metadata,
            tensors=tensors,
        )

        # Keep reference alive so NIXL agent isn't garbage collected
        self._nixl_manager = nixl_mgr

        return worker, len(tensors)
