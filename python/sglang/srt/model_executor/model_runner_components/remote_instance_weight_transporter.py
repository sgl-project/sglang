from __future__ import annotations

import logging
from dataclasses import dataclass
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
    tp_rank: int
    gpu_id: int
    engine: Optional[Any] = None
    session_id: str = ""
    weight_info: Optional[dict[str, tuple[int, int, int]]] = None
    parallelism_config: Optional[RankParallelismConfig] = None
    _nixl_agent: Optional[Any] = None
    _nixl_agent_metadata: Optional[bytes] = None

    @property
    def model(self) -> torch.nn.Module:
        return self.get_model()

    def init_engine(self):
        use_nixl = self.server_args.remote_instance_weight_loader_start_seed_via_nixl
        if use_nixl:
            self._init_nixl()
        else:
            self._init_mooncake()
        if self.engine is None and self._nixl_agent is None:
            return
        self.parallelism_config = RankParallelismConfig.from_parallel_state(
            self.tp_rank
        )

    def _init_mooncake(self):
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

    def _init_nixl(self):
        """Initialize a NIXL agent on the worker (NIXL weight-transfer backend).

        SGLang is the passive, export-only target: it creates a ``nixl_agent`` and
        captures its opaque ``agent_metadata`` blob so the metadata can be published
        for the external peer (Miles). SGLang does NOT call ``add_remote_agent`` and
        does NOT issue ``transfer()`` calls itself -- the peer registers SGLang's
        agent metadata on its side and performs the RDMA WRITEs.
        """
        import uuid

        try:
            from nixl._api import nixl_agent, nixl_agent_config
        except ImportError:
            logger.warning(
                "Please install NIXL for using the NIXL remote instance weight "
                "transfer backend. See "
                "https://github.com/ai-dynamo/nixl/blob/main/README.md"
            )
            return

        backend = envs.SGLANG_REMOTE_INSTANCE_NIXL_BACKEND.get()
        agent_config = nixl_agent_config(backends=[backend])
        agent_name = f"nixl_weight_tp{self.tp_rank}_{uuid.uuid4()}"
        agent = nixl_agent(agent_name, agent_config)

        available_plugins = agent.get_plugin_list()
        if backend not in available_plugins:
            raise ValueError(
                f"NIXL backend '{backend}' not found. Available: {available_plugins}. "
                f"Please install the required NIXL plugin or choose from: {available_plugins}"
            )

        self._nixl_agent = agent
        # _nixl_agent_metadata is NOT set here. It must be set after
        # register_memory_region_nixl() so the blob includes the VRAM rkeys
        # the Miles peer needs to RDMA-WRITE into the weight buffers.
        logger.info(
            f"NIXL weight-transfer agent initialized (agent_name={agent_name}, "
            f"backend={backend}) for tp_rank={self.tp_rank}"
        )

    def maybe_register_and_publish_weight_info(self) -> None:
        if (
            remote_instance_transfer_engine_enabled()
            # ModelExpress owns TransferEngine memory registration and metadata
            # publishing for backend=modelexpress. Re-registering here would
            # overlap the same weight buffers.
            and get_model().remote_instance_weight_loader_backend
            != RemoteInstanceWeightLoaderBackend.MODELEXPRESS
            and self.engine is not None
            and self.weight_info is None
        ):
            # Register memory and upstream the transfer engine info to the bootstrap server
            self.weight_info = register_memory_region(self.model, self.engine)
            self._register_to_engine_info_bootstrap()

        if self._nixl_agent is not None and self.weight_info is None:
            # Register VRAM, then capture agent metadata. Metadata must be
            # captured AFTER registration so the blob includes the rkeys the
            # peer needs to RDMA-WRITE into the weight buffers.
            self.weight_info = register_memory_region_nixl(
                self.model, self._nixl_agent, self.gpu_id
            )
            self._nixl_agent_metadata = self._nixl_agent.get_agent_metadata()
            self._register_to_engine_info_bootstrap()

        # The P2P weight-update client needs each rank's parallelism layout to
        # map training-side parameters onto this rank's shards.
        if (
            self.server_args.registers_parallelism_config()
            and self.parallelism_config is not None
        ):
            self._register_parallelism_config_to_bootstrap()

    def _bootstrap_url(self) -> str:
        if self.server_args.dist_init_addr:
            bootstrap_host = (
                NetworkAddress.parse(self.server_args.dist_init_addr).resolved().host
            )
        else:
            bootstrap_host = "127.0.0.1"
        bootstrap_port = self.server_args.engine_info_bootstrap_port
        return NetworkAddress(bootstrap_host, bootstrap_port).to_url()

    def _register_parallelism_config_to_bootstrap(self) -> None:
        """Register this rank's parallelism config with the EngineInfoBootstrapServer."""
        import requests as http_requests

        bootstrap_url = self._bootstrap_url()
        url = f"{bootstrap_url}/register_parallelism_config"
        payload = {
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

    def _register_to_engine_info_bootstrap(self: RemoteInstanceWeightTransporter):
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

        # NIXL needs the peer to call add_remote_agent() before any transfer,
        # so we publish agent_name + the opaque agent_metadata (base64-encoded
        # for JSON transport) alongside the weight addresses.
        if self._nixl_agent is not None:
            import base64

            transfer_engine_info = {
                "backend": "nixl",
                "agent_name": self._nixl_agent.name,
                "agent_metadata": base64.b64encode(self._nixl_agent_metadata).decode(
                    "ascii"
                ),
                "weights_info_dict": self.weight_info,
            }
        else:
            transfer_engine_info = {
                "backend": "mooncake",
                "session_id": self.session_id,
                "weights_info_dict": self.weight_info,
            }

        payload = {
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
