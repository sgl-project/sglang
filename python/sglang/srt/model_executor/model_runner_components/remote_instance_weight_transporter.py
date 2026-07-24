from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
    register_memory_region,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto

logger = logging.getLogger(__name__)


@dataclass(slots=True, kw_only=True)
class RemoteInstanceWeightTransporter:
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
        return self.get_model()

    def init_engine(self):
        """Initialize the Mooncake ``TransferEngine`` for remote-instance P2P
        weight loading.

        ``MOONCAKE_DEVICE`` selects the RDMA NIC(s) for the transfer and accepts
        either:

        * a shared spec applied to every rank, e.g. ``"mlx5_0"`` or
          ``"mlx5_0,mlx5_1"``; or
        * a per-GPU mapping so each TP rank binds to its NUMA-local NIC, given
          as inline JSON ``{"0": "mlx5_0", "1": "mlx5_1"}`` or a path to a
          ``.json`` file with the same mapping.

        The per-GPU device is resolved here and passed to
        ``TransferEngine.initialize()``. Under the TENT runtime it reaches
        ``topology/rdma_whitelist`` (deterministic per-rank NIC binding); with
        the classic engine the device filter is honored directly. When
        ``MOONCAKE_DEVICE`` is unset, the engine auto-selects NICs.
        """
        try:
            from mooncake.engine import TransferEngine
        except ImportError:
            logger.warning(
                "Please install mooncake for using remote instance transfer engine: pip install mooncake-transfer-engine"
            )
            return
        self.engine = TransferEngine()
        local_ip = get_local_ip_auto()
        from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
            get_ib_devices_for_gpu,
        )

        ib_device = (
            get_ib_devices_for_gpu(envs.MOONCAKE_DEVICE.get(), self.gpu_id)
            or envs.MOONCAKE_DEVICE.get()
        )
        self.engine.initialize(
            local_ip,
            "P2PHANDSHAKE",
            envs.MOONCAKE_PROTOCOL.get(),
            ib_device,
        )
        self.session_id = NetworkAddress(
            local_ip, self.engine.get_rpc_port()
        ).to_host_port_str()

    def maybe_register_and_publish_weight_info(self) -> None:
        if (
            self.server_args.remote_instance_weight_loader_use_transfer_engine()
            # ModelExpress owns TransferEngine memory registration and metadata
            # publishing for backend=modelexpress. Re-registering here would
            # overlap the same weight buffers.
            and self.server_args.remote_instance_weight_loader_backend
            != RemoteInstanceWeightLoaderBackend.MODELEXPRESS
            and self.engine is not None
            and self.weight_info is None
        ):
            # Register memory and upstream the transfer engine info to the bootstrap server
            self.weight_info = register_memory_region(self.model, self.engine)
            self._register_to_engine_info_bootstrap()

    def _register_to_engine_info_bootstrap(self: RemoteInstanceWeightTransporter):
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
