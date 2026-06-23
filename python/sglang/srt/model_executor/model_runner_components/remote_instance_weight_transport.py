from __future__ import annotations

import logging
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
            local_ip,
            "P2PHANDSHAKE",
            envs.MOONCAKE_PROTOCOL.get(),
            envs.MOONCAKE_DEVICE.get(),
        )
        self.session_id = NetworkAddress(
            local_ip, self.engine.get_rpc_port()
        ).to_host_port_str()
