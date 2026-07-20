from __future__ import annotations

import logging
from bisect import bisect_right
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
    dp_rank: int = 0
    pp_rank: int = 0
    ep_rank: int = 0
    engine: Optional[Any] = None
    session_id: str = ""
    weight_info: Optional[dict[str, tuple[int, int, int]]] = None
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

    def maybe_register_and_publish_weight_info(self) -> None:
        if (
            self.server_args.remote_instance_weight_loader_use_transfer_engine()
            and self.server_args.remote_instance_weight_loader_start_seed_via_transfer_engine
            # ModelExpress owns TransferEngine memory registration and metadata
            # publishing for backend=modelexpress. Re-registering here would
            # overlap the same weight buffers.
            and self.server_args.remote_instance_weight_loader_backend
            != RemoteInstanceWeightLoaderBackend.MODELEXPRESS
            and self.engine is not None
        ):
            # Register memory and upstream the transfer engine info to the bootstrap server
            if self.weight_info is None:
                self.weight_info = register_memory_region(self.model, self.engine)
            self._register_to_engine_info_bootstrap()

    @property
    def worker_id(self) -> str:
        return (
            f"{self.session_id}/dp{self.dp_rank}-pp{self.pp_rank}-"
            f"ep{self.ep_rank}-tp{self.tp_rank}"
        )

    def validate_runtime_manifest_addresses(self, manifest: Any) -> None:
        """Fail if model storage moved after TransferEngine registration."""
        if not self.weight_info:
            raise RuntimeError(
                "source model weights are not registered with TransferEngine"
            )
        ranges = sorted(
            (
                int(address),
                int(address) + int(numel) * int(itemsize),
            )
            for address, numel, itemsize in self.weight_info.values()
        )
        starts = [start for start, _ in ranges]
        tensors = (
            manifest.get("tensors", ())
            if isinstance(manifest, dict)
            else manifest.tensors
        )
        for tensor in tensors:
            address = int(
                tensor["address"] if isinstance(tensor, dict) else tensor.address
            )
            nbytes = int(
                tensor["nbytes"] if isinstance(tensor, dict) else tensor.nbytes
            )
            index = bisect_right(starts, address) - 1
            if index < 0 or address + nbytes > ranges[index][1]:
                raise RuntimeError(
                    "runtime manifest references storage outside registered "
                    "TransferEngine memory"
                )

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
        except Exception as e:
            raise RuntimeError(
                "Failed to register transfer engine info for "
                f"tp_rank={self.tp_rank}: {e}"
            ) from e
        if resp.status_code != 200:
            raise RuntimeError(
                "Failed to register transfer engine info for "
                f"tp_rank={self.tp_rank}: {resp.status_code}, {resp.text}"
            )
        logger.info(
            f"Registered transfer engine info for tp_rank={self.tp_rank} "
            f"with bootstrap server at {bootstrap_na}"
        )
