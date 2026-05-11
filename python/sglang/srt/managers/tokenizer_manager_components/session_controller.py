from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

from sglang.srt.server_args import ServerArgs


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionController:
    """open_session / close_session endpoints + OpenSessionReqOutput dispatcher handler."""

    send_to_scheduler: Any
    auto_create_handle_loop: Callable[[], None]
    server_args: ServerArgs
    session_futures: Dict[str, asyncio.Future] = field(default_factory=dict)
