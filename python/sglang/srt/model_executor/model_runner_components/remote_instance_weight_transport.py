from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from sglang.srt.server_args import ServerArgs

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
