# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.realtime.session import (
    RealtimeSession,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
        BaseRealtimeModelAdapter,
    )


@dataclass(frozen=True, slots=True)
class RealtimeChunkContext:
    session_id: str
    index: int
    request_id: str


class GenerateSession:
    """A realtime generation session"""

    def __init__(self):
        self.id = uuid4().hex
        self.request: RealtimeVideoGenerationsRequest | None = None
        self.input_temp_dir: str | None = None
        self.generate_chunk_cnt = 0
        self.current_chunk: RealtimeChunkContext | None = None
        self.realtime_session = RealtimeSession()
        self.adapter: BaseRealtimeModelAdapter | None = None
        self.adapter_state: Any = None
        self.output_pace_next_send_at: float | None = None
        self.output_pace_last_event_id: int | None = None

    def set_adapter(self, adapter: BaseRealtimeModelAdapter):
        self.adapter = adapter
        self.adapter_state = adapter.create_state()

    def set_request(self, request: RealtimeVideoGenerationsRequest):
        self.request = request

    def dispose(self):
        if self.adapter is not None:
            self.adapter.dispose(self)
        self.request = None
        self.input_temp_dir = None
        self.generate_chunk_cnt = 0
        self.current_chunk = None
        self.adapter = None
        self.adapter_state = None
        self.output_pace_next_send_at = None
        self.output_pace_last_event_id = None
        self.realtime_session.dispose()

    def new_chunk(self) -> RealtimeChunkContext:
        if self.current_chunk is not None:
            raise RuntimeError("previous realtime chunk is still active")
        chunk = RealtimeChunkContext(
            session_id=self.id,
            index=self.generate_chunk_cnt,
            request_id=f"{self.id}_{uuid4().hex}",
        )
        self.current_chunk = chunk
        return chunk

    def generate_chunk_completed(self):
        self.generate_chunk_cnt += 1
        self.current_chunk = None

    def reached_max_chunks(self) -> bool:
        return (
            self.request is not None
            and self.request.max_chunks is not None
            and self.generate_chunk_cnt >= self.request.max_chunks
        )
