# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from fastapi import WebSocket

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
    RealtimeFrameSendStats,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
        GenerateSession,
        RealtimeChunkContext,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
        OutputBatch,
        Req,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


@dataclass(slots=True)
class RealtimeChunkInputs:
    prompt: str
    condition_inputs: dict[str, Any] = field(default_factory=dict)


class RealtimeModelAdapter(Protocol):
    name: str

    def create_state(self) -> Any: ...

    async def on_init(
        self,
        session: GenerateSession,
        request: RealtimeVideoGenerationsRequest,
    ) -> None: ...

    async def wait_for_next_chunk(self, session: GenerateSession) -> None: ...

    def ingest_event(
        self,
        session: GenerateSession,
        event: RealtimeEvent,
    ) -> str: ...

    def prepare_next_request(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
    ) -> Req: ...

    async def send_output(
        self,
        ws: WebSocket,
        session: GenerateSession,
        result: OutputBatch,
        batch: Req,
    ) -> RealtimeFrameSendStats: ...

    def on_chunk_complete(
        self, session: GenerateSession, result: OutputBatch
    ) -> None: ...

    def dispose(self, session: GenerateSession) -> None: ...
