# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import asyncio
import os
from typing import Optional, Type

import setproctitle

from sglang.srt.disaggregation.utils import TransferBackend
from sglang.srt.managers.io_struct import (
    ContinueGenerationReqInput,
    PauseContinueBroadcastReq,
    PauseGenerationReqInput,
    TokenizerWorkerRegistrationReq,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_disagg
from sglang.srt.server_args import PortArgs, ServerArgs


class TokenizerWorker(TokenizerManager):
    """Tokenizer Worker in multi-http-worker mode"""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        setproctitle.setproctitle(f"sglang::tokenizer_worker:{os.getpid()}")
        import torch

        torch.set_num_threads(1)
        super().__init__(
            server_args,
            port_args,
            start_pd_bootstrap_service=False,
        )

        self.worker_id = os.getpid()
        self.tokenizer_ipc_name = port_args.tokenizer_ipc_name

        # For PD disaggregation
        self.disaggregation_transfer_backend = TransferBackend(
            get_disagg().disaggregation_transfer_backend
        )

        # Register this worker with the router for pause/continue broadcasting
        reg = TokenizerWorkerRegistrationReq(worker_ipc_name=self.tokenizer_ipc_name)
        self._dispatch_to_scheduler(reg)

        # Future for awaiting pause/continue broadcast confirmation
        self._pause_continue_future: Optional[asyncio.Future] = None

        # Register PauseContinueBroadcastReq in the result dispatcher so
        # handle_loop routes it to _handle_pause_continue_broadcast
        from sglang.utils import TypeBasedDispatcher

        self._result_dispatcher += TypeBasedDispatcher(
            [(PauseContinueBroadcastReq, self._handle_pause_continue_broadcast)]
        )

    async def pause_generation(self, obj: PauseGenerationReqInput):
        loop = asyncio.get_event_loop()
        self._pause_continue_future = loop.create_future()
        # Send to router which will broadcast to all workers
        # (router also handles forwarding to scheduler for non-abort modes)
        self._dispatch_to_scheduler(obj)
        await self._pause_continue_future

        if obj.mode == "abort":
            # Abort polling: only the originator checks its own lock state
            while True:
                self.abort_request(abort_all=True)
                is_locked = await self.model_update_lock.is_locked()
                if not is_locked:
                    break
                await asyncio.sleep(1.0)

    async def continue_generation(self, obj: ContinueGenerationReqInput):
        loop = asyncio.get_event_loop()
        self._pause_continue_future = loop.create_future()
        self._dispatch_to_scheduler(obj)
        await self._pause_continue_future

    def _handle_pause_continue_broadcast(self, obj: PauseContinueBroadcastReq):
        """Called from handle_loop when a broadcast arrives from the router."""
        loop = asyncio.get_event_loop()
        loop.create_task(self._apply_pause_continue_broadcast(obj))

    async def _apply_pause_continue_broadcast(self, obj: PauseContinueBroadcastReq):
        """Apply pause/continue state under the condition lock."""
        async with self.is_pause_cond:
            if obj.is_pause:
                self.is_pause = True
            else:
                self.is_pause = False
                self.is_pause_cond.notify_all()

        # Resolve the pending future if this worker initiated the pause/continue
        if self._pause_continue_future and not self._pause_continue_future.done():
            self._pause_continue_future.set_result(True)
            self._pause_continue_future = None


def get_tokenizer_worker_class(server_args: ServerArgs) -> Type[TokenizerWorker]:
    worker_class = server_args.get_tokenizer_worker_class()
    if not isinstance(worker_class, type) or not issubclass(
        worker_class, TokenizerWorker
    ):
        raise TypeError(
            "ServerArgs.get_tokenizer_worker_class() must return a TokenizerWorker "
            f"subclass, got {worker_class!r}"
        )

    return worker_class
