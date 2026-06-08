from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

import fastapi

from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,
    OpenSessionReqInput,
)

logger = logging.getLogger(__name__)
from typing import Any, Callable, Dict

from sglang.srt.server_args import ServerArgs


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionController:
    send_to_scheduler: Any
    auto_create_handle_loop: Callable[[], None]
    server_args: ServerArgs
    session_futures: Dict[str, asyncio.Future] = field(default_factory=dict)

    async def open_session(
        self,
        obj: OpenSessionReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        if obj.streaming:
            if not self.server_args.enable_streaming_session:
                raise ValueError(
                    "Streaming sessions are disabled. "
                    "Please relaunch with --enable-streaming-session."
                )

        if obj.session_id is None:
            obj.session_id = uuid.uuid4().hex
        elif obj.session_id in self.session_futures:
            return None

        future = asyncio.Future()
        self.session_futures[obj.session_id] = future
        self.send_to_scheduler.send_pyobj(obj)

        try:
            return await future
        finally:
            self.session_futures.pop(obj.session_id, None)

    async def close_session(
        self,
        obj: CloseSessionReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        await self.send_to_scheduler.send_pyobj(obj)

    def handle_open_session_req_output(self, recv_obj):
        future = self.session_futures.get(recv_obj.session_id)
        if future is None:
            logger.warning(
                "Open session response arrived after waiter cleanup: %s",
                recv_obj.session_id,
            )
            return
        if not future.done():
            future.set_result(recv_obj.session_id if recv_obj.success else None)
