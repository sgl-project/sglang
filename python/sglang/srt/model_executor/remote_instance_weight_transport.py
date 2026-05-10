from __future__ import annotations

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto

logger = logging.getLogger(__name__)


class RemoteInstanceWeightTransport:

    def __init__(
        self,
        *,
        server_args: ServerArgs,
        model: torch.nn.Module,
        tp_rank: int,
        gpu_id: int,
    ):
        self.server_args = server_args
        self.model = model
        self.tp_rank = tp_rank
        self.gpu_id = gpu_id
        self.remote_instance_transfer_engine = None
        self.remote_instance_transfer_engine_session_id = ""
        self.remote_instance_transfer_engine_weight_info = None
        self._nixl_manager = None

    def remote_instance_init_transfer_engine(self):
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            logger.warning(
                "Please install mooncake for using remote instance transfer engine: pip install mooncake"
            )
            return
        self.remote_instance_transfer_engine = TransferEngine()
        local_ip = get_local_ip_auto()
        self.remote_instance_transfer_engine.initialize(
            local_ip, "P2PHANDSHAKE", "rdma", envs.MOONCAKE_DEVICE.get()
        )
        self.remote_instance_transfer_engine_session_id = NetworkAddress(
            local_ip, self.remote_instance_transfer_engine.get_rpc_port()
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
                "session_id": self.remote_instance_transfer_engine_session_id,
                "weights_info_dict": self.remote_instance_transfer_engine_weight_info,
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
