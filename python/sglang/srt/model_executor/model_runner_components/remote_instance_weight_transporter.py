from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

from sglang.srt.distributed.parallel_state import RankParallelismConfig
from sglang.srt.environ import envs
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
    describe_weight_locations,
    register_weight_memory,
)
from sglang.srt.runtime_context import (
    get_model,
    get_parallel,
    remote_instance_transfer_engine_enabled,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto

logger = logging.getLogger(__name__)


@dataclass(slots=True, kw_only=True)
class RemoteInstanceWeightTransporter:
    server_args: ServerArgs
    get_model: Callable[[], torch.nn.Module]
    # Registration may run after the runner's construction scope has exited.
    tp_rank: int = field(init=False, default_factory=lambda: get_parallel().tp_rank)
    gpu_id: int
    engine: Optional[Any] = None
    session_id: str = ""
    # name -> (address, numel, element size), once this runner registered with or loaded through a transfer engine
    weight_locations: Optional[dict[str, tuple[int, int, int]]] = None
    parallelism_config: Optional[RankParallelismConfig] = None
    _nixl_manager: Optional[Any] = None

    @property
    def model(self) -> torch.nn.Module:
        return self.get_model()

    def init_engine(self):
        try:
            from mooncake.engine import TransferEngine
        except ImportError:
            logger.warning(
                "Please install mooncake for using remote instance transfer engine: pip install mooncake-transfer-engine"
            )
            return
        self.engine = TransferEngine()
        local_ip = get_local_ip_auto()
        self.engine.initialize(
            local_ip,
            "P2PHANDSHAKE",
            envs.MOONCAKE_PROTOCOL.get(),
            envs.MOONCAKE_DEVICE.get(),
        )
        self.session_id = NetworkAddress(
            local_ip, self.engine.get_rpc_port()
        ).to_host_port_str()

    def maybe_init_parallelism_config(self) -> None:
        if self.server_args.publishes_rank_info():
            self.parallelism_config = RankParallelismConfig.from_parallel_state(
                self.tp_rank
            )

    def publish_rank_info(self, role: str) -> None:
        """Publishes this runner's rank info under `role` (``target``, or the draft role `weight_update_runners()`
        names): its parallelism config, which the P2P weight-update client needs to map training-side parameters
        onto this rank's shards, and, once its weights are registered with a transfer engine, their locations and the
        session to reach them."""
        transfer_session_id = None
        if (
            remote_instance_transfer_engine_enabled()
            # ModelExpress owns TransferEngine memory registration and metadata
            # publishing for backend=modelexpress. Re-registering here would
            # overlap the same weight buffers.
            and get_model().remote_instance_weight_loader_backend
            != RemoteInstanceWeightLoaderBackend.MODELEXPRESS
            and self.engine is not None
            and self.weight_locations is None
        ):
            register_weight_memory(self.model, self.engine)
            self.weight_locations = describe_weight_locations(self.model)
            transfer_session_id = self.session_id
        parallelism_config = (
            self.parallelism_config.to_dict()
            if self.server_args.publishes_rank_info()
            and self.parallelism_config is not None
            else None
        )
        if transfer_session_id is None and parallelism_config is None:
            return
        self._put_rank_info(
            role,
            {
                "parallelism_config": parallelism_config,
                "weight_locations": (
                    self.weight_locations if transfer_session_id is not None else None
                ),
                "transfer_session_id": transfer_session_id,
            },
        )

    def _bootstrap_url(self) -> str:
        if get_parallel().dist_init_addr:
            bootstrap_host = (
                NetworkAddress.parse(get_parallel().dist_init_addr).resolved().host
            )
        else:
            bootstrap_host = "127.0.0.1"
        bootstrap_port = get_model().engine_info_bootstrap_port
        return NetworkAddress(bootstrap_host, bootstrap_port).to_url()

    def _put_rank_info(self, role: str, rank_info: dict) -> None:
        """Register this rank's info with the EngineInfoBootstrapServer, which runs on node_rank==0."""
        import requests as http_requests

        url = f"{self._bootstrap_url()}/register_rank_info"
        payload = {"role": role, "tp_rank": self.tp_rank, "rank_info": rank_info}
        try:
            resp = http_requests.put(url, json=payload, timeout=5)
            if resp.status_code == 200:
                logger.info(
                    f"Registered rank info for role={role} tp_rank={self.tp_rank} with the bootstrap server at {url}"
                )
            else:
                logger.error(
                    f"Failed to register rank info for role={role} tp_rank={self.tp_rank}: "
                    f"{resp.status_code}, {resp.text}"
                )
        except Exception as e:
            logger.error(
                f"Failed to register rank info for role={role} tp_rank={self.tp_rank}: {e}"
            )
