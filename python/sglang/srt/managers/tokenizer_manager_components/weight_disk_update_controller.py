from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional

from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.aio_rwlock import RWLock


@dataclass(slots=True, kw_only=True)
class WeightDiskUpdateController:
    """update_weights_from_disk endpoint + UpdateWeightFromDiskReqOutput dispatcher handler."""

    send_to_scheduler: Any
    abort_request: Callable[..., None]
    is_pause_getter: Callable[[], bool]
    is_pause_cond: asyncio.Condition
    model_update_lock: RWLock
    server_args: ServerArgs
    auto_create_handle_loop: Callable[[], None]
    initial_weights_loaded: bool = True
    model_update_result: Optional[Awaitable[Any]] = None
    model_update_tmp: List[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.server_args.checkpoint_engine_wait_weights_before_ready:
            self.initial_weights_loaded = False
