# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.utils.realtime_trace import (
    compact_client_trace,
    normalize_trace_id,
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
    generation_id: str
    index: int
    request_id: str
    action_version: int = 0
    prompt_version: int = 0


class GenerateSession:
    """A realtime generation session"""

    def __init__(self, *, max_inflight_chunks: int = 1):
        if max_inflight_chunks < 1:
            raise ValueError("max_inflight_chunks must be positive")
        self.id = uuid4().hex
        self.generation_id = uuid4().hex
        self.trace_id = self.id
        self.trace_started_at = time.perf_counter()
        self.trace_started_epoch_ms = int(time.time() * 1000)
        self.created_at = time.monotonic()
        self.last_client_activity_at = self.created_at
        self.client_activity_version = 0
        self.action_version = 0
        self.prompt_version = 0
        self.denoise_intervals: dict[int, tuple[float, float]] = {}
        self.vae_intervals: dict[int, tuple[float, float]] = {}
        self.client_trace: dict[str, Any] | None = None
        self.request: RealtimeVideoGenerationsRequest | None = None
        self.input_temp_dir: str | None = None
        self.generate_chunk_cnt = 0
        self.next_chunk_index = 0
        self.max_inflight_chunks = max_inflight_chunks
        self.active_chunks: dict[int, RealtimeChunkContext] = {}
        self._completed_chunks: set[int] = set()
        self.realtime_session = RealtimeSession()
        self.adapter: BaseRealtimeModelAdapter | None = None
        self.adapter_state: Any = None
        self.output_pace_next_send_at: float | None = None
        self.output_pace_last_event_id: int | None = None
        self.vae_client: Any = None

    def set_adapter(self, adapter: BaseRealtimeModelAdapter):
        self.adapter = adapter
        self.adapter_state = adapter.create_state()

    def bind_trace(self, request: RealtimeVideoGenerationsRequest):
        self.trace_id = normalize_trace_id(request.trace_id, fallback=self.trace_id)
        self.client_trace = compact_client_trace(request.client_trace)

    def set_request(self, request: RealtimeVideoGenerationsRequest):
        self.bind_trace(request)
        self.request = request

    def dispose(self):
        if self.adapter is not None:
            self.adapter.dispose(self)
        self.request = None
        self.client_trace = None
        self.input_temp_dir = None
        self.generate_chunk_cnt = 0
        self.next_chunk_index = 0
        self.active_chunks.clear()
        self._completed_chunks.clear()
        self.adapter = None
        self.adapter_state = None
        self.output_pace_next_send_at = None
        self.output_pace_last_event_id = None
        self.vae_client = None
        self.denoise_intervals.clear()
        self.vae_intervals.clear()
        self.realtime_session.dispose()

    def mark_client_activity(self) -> None:
        self.last_client_activity_at = time.monotonic()
        self.client_activity_version += 1

    def mark_event_version(self, kind: str) -> None:
        if kind in {"camera_actions", "action_labels", "action_weights"}:
            self.action_version += 1
        elif kind in {"prompt", "scene_cut"}:
            self.prompt_version += 1

    @property
    def current_chunk(self) -> RealtimeChunkContext | None:
        if not self.active_chunks:
            return None
        return self.active_chunks[min(self.active_chunks)]

    def can_schedule_chunk(self) -> bool:
        if len(self.active_chunks) >= self.max_inflight_chunks:
            return False
        if self.request is None or self.request.max_chunks is None:
            return True
        return self.next_chunk_index < self.request.max_chunks

    def new_chunk(
        self,
        *,
        action_version: int | None = None,
        prompt_version: int | None = None,
    ) -> RealtimeChunkContext:
        if len(self.active_chunks) >= self.max_inflight_chunks:
            if self.max_inflight_chunks == 1:
                raise RuntimeError("previous realtime chunk is still active")
            raise RuntimeError("realtime chunk in-flight limit reached")
        if not self.can_schedule_chunk():
            raise RuntimeError("realtime session reached max chunks")
        chunk = RealtimeChunkContext(
            session_id=self.id,
            generation_id=self.generation_id,
            index=self.next_chunk_index,
            request_id=f"{self.id}_{uuid4().hex}",
            action_version=(
                self.action_version if action_version is None else action_version
            ),
            prompt_version=(
                self.prompt_version if prompt_version is None else prompt_version
            ),
        )
        self.next_chunk_index += 1
        self.active_chunks[chunk.index] = chunk
        return chunk

    def generate_chunk_completed(
        self, chunk: RealtimeChunkContext | None = None
    ) -> None:
        if chunk is None:
            if not self.active_chunks:
                raise RuntimeError("no active realtime chunk to complete")
            chunk = self.active_chunks[min(self.active_chunks)]
        active = self.active_chunks.pop(chunk.index, None)
        if active != chunk:
            raise RuntimeError(f"realtime chunk {chunk.index} is not active")
        self._completed_chunks.add(chunk.index)
        while self.generate_chunk_cnt in self._completed_chunks:
            self._completed_chunks.remove(self.generate_chunk_cnt)
            self.generate_chunk_cnt += 1

    def reached_max_chunks(self) -> bool:
        return (
            self.request is not None
            and self.request.max_chunks is not None
            and self.generate_chunk_cnt >= self.request.max_chunks
        )
