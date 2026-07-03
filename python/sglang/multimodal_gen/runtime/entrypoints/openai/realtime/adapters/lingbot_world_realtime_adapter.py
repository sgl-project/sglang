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
from sglang.multimodal_gen.runtime.lingbot_world.constants import (
    LINGBOT_CAMERA_ACTIONS_CONDITION,
    LINGBOT_PROMPT_UPDATED_CONDITION,
)
from sglang.multimodal_gen.runtime.realtime.control_signals import (
    ControlSignalQueue,
    ParsedControlEventPayload,
    parse_control_event_payload,
)
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
COMPOSITE_INPUT_EVENT_KIND = "composite_input"


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

    def parse_camera_control_event_payload(
        self,
        payload: Any,
        *,
        event_id: int | None,
    ) -> ParsedControlEventPayload:
        return parse_control_event_payload(
            payload,
            event_id=event_id,
            kind="camera_actions",
            normalize_state_payload=self._normalize_state_actions,
            validate_script_payload=LingBotWorldRealtimeAdapter._validate_camera_actions,
        )

    def receive_parsed_camera_control_event_payload(
        self,
        parsed: ParsedControlEventPayload,
        *,
        event_id: int | None,
    ) -> str:
        if parsed.mode == "state":
            transitions = parsed.payload
            self.receive_camera_state_transitions(transitions)
            return f"kind=camera_actions, mode=state, transitions={len(transitions)}"

        camera_actions = parsed.payload
        self.receive_camera_action_script(camera_actions, event_id=event_id)
        return f"kind=camera_actions, mode=script, frames={len(camera_actions)}"

    def receive_camera_control_event_payload(
        self,
        payload: Any,
        *,
        event_id: int | None,
    ) -> str:
        parsed = self.parse_camera_control_event_payload(payload, event_id=event_id)
        return self.receive_parsed_camera_control_event_payload(
            parsed, event_id=event_id
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
        camera_actions = condition_inputs.get(LINGBOT_CAMERA_ACTIONS_CONDITION)
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
            return self._ingest_camera_actions(state, event.payload, event.event_id)
        elif event.kind == "prompt":
            return self._ingest_prompt(state, event.payload, event.event_id)
        elif event.kind == COMPOSITE_INPUT_EVENT_KIND:
            return self._ingest_composite_input(state, event.payload, event.event_id)
        raise ValueError(f"unsupported event kind: {event.kind}")

    def _ingest_camera_actions(
        self,
        state: LingBotWorldRealtimeState,
        payload: Any,
        event_id: int | None,
    ) -> str:
        return state.receive_camera_control_event_payload(
            payload,
            event_id=event_id,
        )

    def _ingest_prompt(
        self,
        state: LingBotWorldRealtimeState,
        payload: Any,
        event_id: int | None,
    ) -> str:
        prompt = self._validate_prompt_payload(payload)
        state.receive_prompt(prompt, event_id=event_id)
        return f"kind=prompt, prompt_len={len(prompt)}"

    @staticmethod
    def _validate_prompt_payload(payload: Any) -> str:
        if not isinstance(payload, str) or not payload:
            raise ValueError("prompt event payload must be a non-empty string")
        return payload

    def _ingest_composite_input(
        self,
        state: LingBotWorldRealtimeState,
        payload: Any,
        event_id: int | None,
    ) -> str:
        if not isinstance(payload, dict):
            raise ValueError("composite_input event payload must be a map")
        input_types = payload.get("input_types")
        if not isinstance(input_types, list) or not input_types:
            raise ValueError(
                "composite_input event payload requires non-empty input_types"
            )

        parsed_inputs = []
        for input_type in input_types:
            if not isinstance(input_type, str) or not input_type:
                raise ValueError(
                    "composite_input input_types must contain non-empty strings"
                )
            if input_type not in payload:
                raise ValueError(f"composite_input event payload requires {input_type}")
            parsed_inputs.append(
                (
                    input_type,
                    self._parse_composite_input_item(
                        state,
                        input_type,
                        payload[input_type],
                        event_id,
                    ),
                )
            )

        input_logs = []
        for input_type, parsed_payload in parsed_inputs:
            input_logs.append(
                self._ingest_parsed_composite_input_item(
                    state,
                    input_type,
                    parsed_payload,
                    event_id,
                )
            )
        return f"kind=composite_input, inputs={input_logs}"

    def _parse_composite_input_item(
        self,
        state: LingBotWorldRealtimeState,
        input_type: str,
        payload: Any,
        event_id: int | None,
    ) -> Any:
        if input_type == "camera_actions":
            return state.parse_camera_control_event_payload(
                payload,
                event_id=event_id,
            )
        if input_type == "prompt":
            return self._validate_prompt_payload(payload)
        raise ValueError(f"unsupported composite_input type: {input_type}")

    def _ingest_parsed_composite_input_item(
        self,
        state: LingBotWorldRealtimeState,
        input_type: str,
        parsed_payload: Any,
        event_id: int | None,
    ) -> str:
        if input_type == "camera_actions":
            return state.receive_parsed_camera_control_event_payload(
                parsed_payload,
                event_id=event_id,
            )
        if input_type == "prompt":
            state.receive_prompt(parsed_payload, event_id=event_id)
            return f"kind=prompt, prompt_len={len(parsed_payload)}"
        raise ValueError(f"unsupported composite_input type: {input_type}")

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

        prompt_updated = False
        if chunk.index == 0:
            prompt = request.prompt
        elif state.has_prompt():
            prompt = state.sample_prompt()
            request.prompt = prompt
            prompt_updated = True
        else:
            prompt = request.prompt

        condition_inputs = {}
        if prompt_updated:
            condition_inputs[LINGBOT_PROMPT_UPDATED_CONDITION] = True
        camera_actions = state.sample_camera_actions(chunk_size)
        if camera_actions is not None:
            condition_inputs[LINGBOT_CAMERA_ACTIONS_CONDITION] = camera_actions
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
