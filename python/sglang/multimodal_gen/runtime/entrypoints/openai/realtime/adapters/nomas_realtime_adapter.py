# SPDX-License-Identifier: Apache-2.0
"""Nomas realtime adapter: one event segment per WebSocket chunk.

Events (msgpack maps with ``type="event"``):

* ``kind="segment"``: ``payload={"caption": str, "frames": int,
  "norm_extrinsics": [[4x4]...] | None}``. ``frames`` is the event's pixel
  frame count; the chunk generates ``(frames-1)//4+1`` latent frames and emits
  4 pixel frames per latent. ``norm_extrinsics`` holds ``(latents-1)*4+1``
  already-normalized matrices for this event.
* ``kind="end"``: no more events; one flush chunk emits the withheld tail
  frames and the session reaches ``max_chunks``.

The init request may carry ``condition_inputs={"prompt_format": "json"|"plain",
"segments": [...]}``; without ``segments`` the request prompt and num_frames
form the first event. The PE (VLM) is called by the client; it is not part of
the server.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING, Any

from msgspec import Struct

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
    BaseRealtimeModelAdapter,
    RealtimeChunkInputs,
    build_realtime_sampling_params,
)
from sglang.multimodal_gen.runtime.realtime.nomas import (
    NOMAS_FLUSH_CAPTION,
    NOMAS_FLUSH_CONDITION,
    NOMAS_SEGMENT_CONDITION,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
        GenerateSession,
        RealtimeChunkContext,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

SEGMENT_EVENT_KIND = "segment"
END_EVENT_KIND = "end"
NOMAS_REALTIME_DEFAULT_NUM_INFERENCE_STEPS = 8
NOMAS_REALTIME_DEFAULT_SEGMENT_FRAMES = 40
PROMPT_FORMATS = ("json", "plain")


class NomasSegmentEvent(Struct, frozen=True):
    caption: str
    frames: int
    norm_extrinsics: list | None = None
    event_id: int | None = None

    @property
    def latent_frames(self) -> int:
        return (self.frames - 1) // 4 + 1


def parse_segment_payload(payload: Any, event_id: int | None) -> NomasSegmentEvent:
    if not isinstance(payload, dict):
        raise ValueError("segment event payload must be a map")
    caption = payload.get("caption")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("segment caption must be a nonempty string")
    frames = payload.get("frames")
    if type(frames) is not int or frames < 1:
        raise ValueError("segment frames must be a positive integer")
    camera = payload.get("norm_extrinsics")
    if camera is not None:
        expected = ((frames - 1) // 4) * 4 + 1
        if not isinstance(camera, list) or len(camera) != expected:
            raise ValueError(
                f"segment norm_extrinsics must list {expected} 4x4 matrices"
            )
    return NomasSegmentEvent(caption, frames, camera, event_id)


class NomasRealtimeState:
    """HTTP-side control state: queued events and end-of-session flag."""

    def __init__(self):
        self.pending: deque[NomasSegmentEvent] = deque()
        self.available = asyncio.Event()
        self.ended = False
        self.prompt_format = "plain"
        self.latest_sampled_event_id: int | None = None

    def clear(self) -> None:
        self.pending.clear()
        self.available.clear()
        self.ended = False
        self.latest_sampled_event_id = None

    def push(self, segment: NomasSegmentEvent) -> None:
        if self.ended:
            raise ValueError("segment received after end")
        self.pending.append(segment)
        self.available.set()

    def end(self) -> None:
        self.ended = True
        self.available.set()

    def ready(self) -> bool:
        return bool(self.pending) or self.ended


class NomasRealtimeAdapter(BaseRealtimeModelAdapter):
    def create_state(self) -> NomasRealtimeState:
        return NomasRealtimeState()

    def _state(self, session: GenerateSession) -> NomasRealtimeState:
        state = session.adapter_state
        if not isinstance(state, NomasRealtimeState):
            raise TypeError("Nomas realtime adapter state is not initialized")
        return state

    async def on_init(
        self,
        session: GenerateSession,
        request: RealtimeVideoGenerationsRequest,
    ) -> None:
        state = self._state(session)
        state.clear()
        condition_inputs = request.condition_inputs or {}
        prompt_format = condition_inputs.get("prompt_format", "plain")
        if prompt_format not in PROMPT_FORMATS:
            raise ValueError("prompt_format must be json or plain")
        state.prompt_format = prompt_format
        segments = condition_inputs.get("segments")
        if segments is None:
            segments = [
                {
                    "caption": request.prompt,
                    "frames": request.num_frames
                    or NOMAS_REALTIME_DEFAULT_SEGMENT_FRAMES,
                }
            ]
        if not isinstance(segments, list):
            raise ValueError("condition_inputs.segments must be a list")
        for payload in segments:
            state.push(parse_segment_payload(payload, None))

    def ingest_event(self, session: GenerateSession, event: RealtimeEvent) -> str:
        state = self._state(session)
        if event.kind == SEGMENT_EVENT_KIND:
            segment = parse_segment_payload(event.payload, event.event_id)
            state.push(segment)
            return (
                f"kind=segment, frames={segment.frames}, "
                f"latents={segment.latent_frames}, camera={segment.norm_extrinsics is not None}"
            )
        if event.kind == END_EVENT_KIND:
            state.end()
            return "kind=end"
        raise ValueError(f"unsupported event kind: {event.kind}")

    async def wait_for_next_chunk(self, session: GenerateSession) -> None:
        state = self._state(session)
        while not state.ready():
            state.available.clear()
            await state.available.wait()

    def get_chunk_size(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
    ) -> int:
        state = self._state(session)
        if state.pending:
            return state.pending[0].latent_frames
        return 1

    def sample_chunk_inputs(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
        chunk_size: int,
    ) -> RealtimeChunkInputs:
        state = self._state(session)
        if state.pending:
            segment = state.pending.popleft()
            state.latest_sampled_event_id = segment.event_id
            return RealtimeChunkInputs(
                prompt=segment.caption,
                condition_inputs={
                    NOMAS_SEGMENT_CONDITION: {
                        "frames": segment.frames,
                        "norm_extrinsics": segment.norm_extrinsics,
                    }
                },
            )
        if not state.ended:
            raise RuntimeError("no Nomas event is pending; wait_for_next_chunk first")
        # One flush chunk re-decodes the withheld tail, then the loop stops.
        if session.request is not None:
            session.request.max_chunks = session.generate_chunk_cnt + 1
        return RealtimeChunkInputs(
            prompt=NOMAS_FLUSH_CAPTION,
            condition_inputs={
                NOMAS_FLUSH_CONDITION: True,
                NOMAS_SEGMENT_CONDITION: {"frames": 4, "norm_extrinsics": None},
            },
        )

    def build_sampling_params(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
        chunk_inputs: RealtimeChunkInputs,
        chunk_size: int,
    ):
        request = session.request
        if request is None:
            raise ValueError("realtime request is not initialized")
        state = self._state(session)
        segment = chunk_inputs.condition_inputs[NOMAS_SEGMENT_CONDITION]
        flush = bool(chunk_inputs.condition_inputs.get(NOMAS_FLUSH_CONDITION))
        sampling_params = build_realtime_sampling_params(
            chunk.request_id,
            request=request,
            chunk_inputs=chunk_inputs,
            num_frames=segment["frames"],
            num_inference_steps=(
                request.num_inference_steps
                or NOMAS_REALTIME_DEFAULT_NUM_INFERENCE_STEPS
            ),
            chunk_size=chunk_size,
        )
        # Model-specific request fields; the shared builder has no slot for them.
        sampling_params.segments = [
            {"frames": segment["frames"], "caption": chunk_inputs.prompt}
        ]
        sampling_params.continuation_prompts = None
        sampling_params.prompt_format = "plain" if flush else state.prompt_format
        sampling_params.norm_extrinsics = segment.get("norm_extrinsics")
        sampling_params.generator_device = "cpu"
        return sampling_params

    def get_realtime_event_id(self, session: GenerateSession) -> int | None:
        return self._state(session).latest_sampled_event_id

    def clear_state(self, session: GenerateSession) -> None:
        state = session.adapter_state
        if isinstance(state, NomasRealtimeState):
            state.clear()
