from __future__ import annotations

import base64
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

from sglang.srt.distributed.parallel_state import RankParallelismConfig
from sglang.srt.environ import envs
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
    register_memory_region,
    register_memory_region_nixl,
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
    nixl_agent: Optional[Any] = None
    weight_info: Optional[dict[str, tuple[int, int, int]]] = None
    parallelism_config: Optional[RankParallelismConfig] = None
    _nixl_registration: Optional[Any] = None

    @property
    def model(self) -> torch.nn.Module:
        return self.get_model()

    def init_engine(self):
        if get_model().remote_instance_weight_loader_start_seed_via_nixl:
            self._init_nixl_agent()
        else:
            self._init_mooncake_engine()

    def _init_nixl_agent(self) -> None:
        from nixl._api import nixl_agent, nixl_agent_config

        backend = envs.SGLANG_REMOTE_INSTANCE_NIXL_BACKEND.get()
        # target and draft runners of a rank each publish their own agent
        agent = nixl_agent(
            f"sglang-weights-tp{self.tp_rank}-{uuid.uuid4()}",
            nixl_agent_config(backends=[backend]),
        )
        if backend not in agent.get_plugin_list():
            raise ValueError(
                f"NIXL has no {backend} plugin (SGLANG_REMOTE_INSTANCE_NIXL_BACKEND); installed: {agent.get_plugin_list()}"
            )
        self.nixl_agent = agent

    def _init_mooncake_engine(self) -> None:
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
        if self.server_args.registers_parallelism_config():
            self.parallelism_config = RankParallelismConfig.from_parallel_state(
                self.tp_rank
            )

    def maybe_register_and_publish_weight_info(self, role: str) -> None:
        """Registers this runner's weights with its transfer engine and publishes them, with its parallelism
        config, under `role`: ``target``, or the draft role `weight_update_runners()` names."""
        if (
            remote_instance_transfer_engine_enabled()
            # ModelExpress owns TransferEngine memory registration and metadata
            # publishing for backend=modelexpress. Re-registering here would
            # overlap the same weight buffers.
            and get_model().remote_instance_weight_loader_backend
            != RemoteInstanceWeightLoaderBackend.MODELEXPRESS
            and (self.engine is not None or self.nixl_agent is not None)
            and self.weight_info is None
        ):
            # Register memory and upstream the transfer engine info to the bootstrap server
            self._register_to_engine_info_bootstrap(
                role, self._register_weights_for_peer_writes()
            )

        # The P2P weight-update client needs each rank's parallelism layout to
        # map training-side parameters onto this rank's shards.
        if (
            self.server_args.registers_parallelism_config()
            and self.parallelism_config is not None
        ):
            self._register_parallelism_config_to_bootstrap(role)

    def _register_weights_for_peer_writes(self) -> dict:
        """Registers the weights with this runner's transport; returns what a peer needs to write them."""
        if self.nixl_agent is not None:
            self.weight_info, self._nixl_registration = register_memory_region_nixl(
                self.model, self.nixl_agent, self.gpu_id
            )
            return {
                "transport": "nixl",
                "agent_name": self.nixl_agent.name,
                # taken after registration: it carries the registered memory's remote keys
                "agent_metadata": base64.b64encode(
                    self.nixl_agent.get_agent_metadata()
                ).decode("ascii"),
                "gpu_id": self.gpu_id,
                "weights_info_dict": self.weight_info,
            }
        self.weight_info = register_memory_region(self.model, self.engine)
        return {
            "transport": "mooncake",
            "session_id": self.session_id,
            "weights_info_dict": self.weight_info,
        }

    def _bootstrap_url(self) -> str:
        if get_parallel().dist_init_addr:
            bootstrap_host = (
                NetworkAddress.parse(get_parallel().dist_init_addr).resolved().host
            )
        else:
            bootstrap_host = "127.0.0.1"
        bootstrap_port = get_model().engine_info_bootstrap_port
        return NetworkAddress(bootstrap_host, bootstrap_port).to_url()

    def _register_parallelism_config_to_bootstrap(self, role: str) -> None:
        """Register this rank's parallelism config with the EngineInfoBootstrapServer."""
        import requests as http_requests

        bootstrap_url = self._bootstrap_url()
        url = f"{bootstrap_url}/register_parallelism_config"
        payload = {
            "role": role,
            "tp_rank": self.tp_rank,
            "parallelism_config": self.parallelism_config.to_dict(),
        }
        try:
            resp = http_requests.put(url, json=payload, timeout=5)
            if resp.status_code == 200:
                logger.info(
                    f"Registered parallelism config for tp_rank={self.tp_rank} "
                    f"with bootstrap server at {bootstrap_url}"
                )
            else:
                logger.error(
                    f"Failed to register parallelism config for tp_rank={self.tp_rank}: "
                    f"{resp.status_code}, {resp.text}"
                )
        except Exception as e:
            logger.error(
                f"Failed to register parallelism config for tp_rank={self.tp_rank}: {e}"
            )

    def _register_to_engine_info_bootstrap(
        self: RemoteInstanceWeightTransporter, role: str, transfer_engine_info: dict
    ):
        """Register transfer engine info with the EngineInfoBootstrapServer via HTTP PUT.

        The bootstrap server runs on node_rank==0. For multi-node setups, the
        host is derived from dist_init_addr. For single-node, use 127.0.0.1.
        """
        import requests as http_requests

        if get_parallel().dist_init_addr:
            # Multi-node: bootstrap server is on the head node (node_rank==0).
            # Derive host from dist_init_addr (shared across all nodes).
            bootstrap_host = (
                NetworkAddress.parse(get_parallel().dist_init_addr).resolved().host
            )
        else:
            bootstrap_host = "127.0.0.1"

        bootstrap_port = get_model().engine_info_bootstrap_port
        bootstrap_na = NetworkAddress(bootstrap_host, bootstrap_port)
        url = f"{bootstrap_na.to_url()}/register_transfer_engine_info"

        payload = {
            "role": role,
            "tp_rank": self.tp_rank,
            "transfer_engine_info": transfer_engine_info,
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
