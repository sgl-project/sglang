# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
    BaseRealtimeModelAdapter,
    RealtimeChunkInputs,
    build_realtime_sampling_params,
    save_realtime_first_frame,
)
from sglang.multimodal_gen.runtime.realtime.control_signals import ControlSignalQueue
from sglang.multimodal_gen.runtime.realtime.states import (
    RealtimeCameraControlState,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
        GenerateSession,
        RealtimeChunkContext,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


LINGBOT_REALTIME_DEFAULT_NUM_INFERENCE_STEPS = 4
LINGBOT_REALTIME_MIN_CONDITION_CHUNKS = 2


class LingBotWorldRealtimeState(RealtimeCameraControlState):
    def __init__(self):
        super().__init__(
            min_pulse_items=1,
            script_maxlen=512,
            max_transitions=512,
        )
        self.prompt_queue = ControlSignalQueue(max_events={"prompt": 1})

    def clear(self) -> None:
        super().clear()
        self.prompt_queue.clear()

    def receive_prompt(self, prompt: str, *, event_id: int | None = None) -> None:
        self.prompt_queue.push("prompt", prompt, event_id=event_id)

    def receive_camera_control_event_payload(
        self,
        payload: Any,
        *,
        event_id: int | None,
    ) -> str:
        return super().receive_camera_control_event_payload(
            payload,
            event_id=event_id,
            validate_camera_actions=LingBotWorldRealtimeAdapter._validate_camera_actions,
        )

    def sample_prompt(self) -> str:
        prompt = self.prompt_queue.pop_latest("prompt")
        if not isinstance(prompt, str):
            raise ValueError("prompt event payload must be a string")
        self.latest_sampled_event_id = self.prompt_queue.last_sampled_seq_id("prompt")
        return prompt

    def has_prompt(self) -> bool:
        return self.prompt_queue.has_events("prompt")


class LingBotWorldRealtimeAdapter(BaseRealtimeModelAdapter):
    def create_state(self) -> LingBotWorldRealtimeState:
        return LingBotWorldRealtimeState()

    def _state(self, session: GenerateSession) -> LingBotWorldRealtimeState:
        state = session.adapter_state
        if not isinstance(state, LingBotWorldRealtimeState):
            raise TypeError("LingBot realtime adapter state is not initialized")
        return state

    async def on_init(
        self,
        session: GenerateSession,
        request: RealtimeVideoGenerationsRequest,
    ) -> None:
        condition_inputs = request.condition_inputs or {}
        camera_actions = condition_inputs.get("camera_actions")
        if camera_actions is not None:
            state = self._state(session)
            state.receive_camera_action_script(
                self._validate_camera_actions(camera_actions)
            )

        await save_realtime_first_frame(session, request)

    @staticmethod
    def _validate_camera_actions(payload: Any) -> list[list[str]]:
        if not isinstance(payload, list):
            raise ValueError("camera_actions event payload must be list[list[str]]")
        normalized = []
        for frame_actions in payload:
            if not isinstance(frame_actions, list):
                raise ValueError("camera_actions event payload must be list[list[str]]")
            normalized.append(list(frame_actions))
        return normalized

    def ingest_event(
        self,
        session: GenerateSession,
        event: RealtimeEvent,
    ) -> str:
        state = self._state(session)
        if event.kind == "camera_actions":
            return state.receive_camera_control_event_payload(
                event.payload,
                event_id=event.event_id,
            )
        elif event.kind == "prompt":
            if not isinstance(event.payload, str) or not event.payload:
                raise ValueError("prompt event payload must be a non-empty string")
            state.receive_prompt(event.payload, event_id=event.event_id)
            return f"kind=prompt, prompt_len={len(event.payload)}"
        raise ValueError(f"unsupported event kind: {event.kind}")

    def sample_chunk_inputs(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
        chunk_size: int,
    ) -> RealtimeChunkInputs:
        """Samples user inputs (conditions) for the current RealtimeChunk from RealtimeStates"""
        state = self._state(session)
        request = session.request
        if request is None:
            raise ValueError("realtime request is not initialized")

        if chunk.index == 0:
            prompt = request.prompt
        elif state.has_prompt():
            prompt = state.sample_prompt()
            request.prompt = prompt
        else:
            prompt = request.prompt

        condition_inputs = {}
        camera_actions = state.sample_camera_actions(chunk_size)
        if camera_actions is not None:
            condition_inputs["camera_actions"] = camera_actions
        return RealtimeChunkInputs(prompt=prompt, condition_inputs=condition_inputs)

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

        num_frames = self._condition_num_frames(
            request=request,
            server_args=server_args,
            chunk_size=chunk_size,
        )

        return build_realtime_sampling_params(
            chunk.request_id,
            request=request,
            chunk_inputs=chunk_inputs,
            num_frames=num_frames,
            num_inference_steps=(
                request.num_inference_steps
                or LINGBOT_REALTIME_DEFAULT_NUM_INFERENCE_STEPS
            ),
            chunk_size=chunk_size,
        )

    @staticmethod
    def _condition_num_frames(
        *,
        request: RealtimeVideoGenerationsRequest,
        server_args: ServerArgs | None,
        chunk_size: int,
    ) -> int:
        if server_args is None:
            return int(request.num_frames or 0)

        # encode one extra blank condition chunk so repeat-last never reuses
        # the first-frame image mask on later realtime chunks
        temporal_ratio = int(
            server_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        )
        required_latent_frames = chunk_size * LINGBOT_REALTIME_MIN_CONDITION_CHUNKS
        required_num_frames = (required_latent_frames - 1) * temporal_ratio + 1
        return max(int(request.num_frames or 0), required_num_frames)

    def get_realtime_event_id(self, session: GenerateSession) -> int | None:
        return self._state(session).latest_sampled_event_id

    def clear_state(self, session: GenerateSession) -> None:
        state = session.adapter_state
        if isinstance(state, LingBotWorldRealtimeState):
            state.clear()
