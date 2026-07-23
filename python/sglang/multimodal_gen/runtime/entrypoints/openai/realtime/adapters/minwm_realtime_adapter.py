# SPDX-License-Identifier: Apache-2.0
"""Realtime control adapter for MinWM's discrete primitive action labels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sglang.multimodal_gen.configs.pipeline_configs.minwm import (
    MINWM_ACTION_LABELS_CONDITION,
    MINWM_ACTION_WEIGHTS_CONDITION,
    MINWM_PROMPT_UPDATED_CONDITION,
    MINWM_TOTAL_CHUNKS_CONDITION,
)
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
from sglang.multimodal_gen.runtime.models.dits.minwm_action import (
    key_state_to_action_label,
    validate_action_labels,
    validate_action_weights,
)
from sglang.multimodal_gen.runtime.realtime.control_signals import (
    ControlScriptQueue,
    ControlSignalQueue,
)
from sglang.multimodal_gen.runtime.realtime.states import RealtimeCameraControlState

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
        GenerateSession,
        RealtimeChunkContext,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

MINWM_DEFAULT_DMD_STEPS = 4


class MinWMRealtimeState(RealtimeCameraControlState):
    def __init__(self) -> None:
        super().__init__(min_pulse_items=1, script_maxlen=4096, max_transitions=512)
        self.action_label_queue = ControlScriptQueue(
            "action_labels", max_events=4096, default_item=0
        )
        self.action_weight_queue = ControlScriptQueue(
            "action_weights", max_events=32768, default_item=[0.0] * 8
        )
        self.prompt_queue = ControlSignalQueue(max_events={"prompt": 1})
        self.label_event_id: int | None = None
        self.weight_event_id: int | None = None
        self.action_mode = "camera"

    def clear(self) -> None:
        super().clear()
        self.action_label_queue.clear()
        self.action_weight_queue.clear()
        self.prompt_queue.clear()
        self.label_event_id = None
        self.weight_event_id = None
        self.action_mode = "camera"

    def receive_action_labels(
        self, labels: list[int], *, event_id: int | None = None
    ) -> None:
        self.camera_state_queue.clear()
        self.camera_script_queue.clear()
        self.action_weight_queue.clear()
        self.label_event_id = None
        self.weight_event_id = None
        self.action_mode = "labels"
        self.action_label_queue.push_script(labels, event_id=event_id)

    def receive_action_weights(
        self, weights: list[list[float]], *, event_id: int | None = None
    ) -> None:
        self.camera_state_queue.clear()
        self.camera_script_queue.clear()
        self.action_label_queue.clear()
        self.label_event_id = None
        self.weight_event_id = None
        self.action_mode = "weights"
        self.action_weight_queue.push_script(weights, event_id=event_id)

    def receive_camera_action_script(
        self, camera_actions: list[list[str]], *, event_id: int | None = None
    ) -> None:
        self.action_label_queue.clear()
        self.action_weight_queue.clear()
        self.label_event_id = None
        self.weight_event_id = None
        self.action_mode = "camera"
        super().receive_camera_action_script(camera_actions, event_id=event_id)

    def receive_camera_state_transitions(self, transitions) -> None:
        self.action_label_queue.clear()
        self.action_weight_queue.clear()
        self.label_event_id = None
        self.weight_event_id = None
        self.action_mode = "camera"
        super().receive_camera_state_transitions(transitions)

    def receive_prompt(self, prompt: str, *, event_id: int | None = None) -> None:
        self.prompt_queue.replace("prompt", prompt, event_id=event_id)

    def sample_action_labels(self, chunk_size: int) -> list[int]:
        if self.action_label_queue.has_script():
            labels = self.action_label_queue.sample_script(chunk_size)
            self.label_event_id = self.action_label_queue.last_sampled_seq_id()
            return validate_action_labels(labels, expected_frames=chunk_size)
        camera_actions = self.sample_camera_actions(chunk_size)
        if camera_actions is None:
            return [0] * chunk_size
        return [key_state_to_action_label(keys) for keys in camera_actions]

    def sample_action_weights(self, frame_count: int) -> list[list[float]]:
        weights = self.action_weight_queue.sample_script(frame_count)
        self.weight_event_id = self.action_weight_queue.last_sampled_seq_id()
        return validate_action_weights(weights, expected_frames=frame_count)


class MinWMRealtimeAdapter(BaseRealtimeModelAdapter):
    def create_state(self) -> MinWMRealtimeState:
        return MinWMRealtimeState()

    @staticmethod
    def _state(session: GenerateSession) -> MinWMRealtimeState:
        state = session.adapter_state
        if not isinstance(state, MinWMRealtimeState):
            raise TypeError("MinWM realtime adapter state is not initialized")
        return state

    @staticmethod
    def _validate_camera_actions(payload: Any) -> list[list[str]]:
        if not isinstance(payload, list):
            raise ValueError("camera_actions must be list[list[str]]")
        result = []
        for frame in payload:
            if not isinstance(frame, list):
                raise ValueError("camera_actions must be list[list[str]]")
            keys = [str(key).lower().strip() for key in frame]
            key_state_to_action_label(keys)
            result.append(keys)
        return result

    @staticmethod
    def _validate_prompt(payload: Any) -> str:
        if not isinstance(payload, str) or not payload:
            raise ValueError("prompt event payload must be a non-empty string")
        return payload

    async def on_init(
        self,
        session: GenerateSession,
        request: RealtimeVideoGenerationsRequest,
    ) -> None:
        state = self._state(session)
        if request.num_inference_steps not in (None, MINWM_DEFAULT_DMD_STEPS):
            raise ValueError("MinWM DMD checkpoint requires exactly 4 inference steps")
        # The distilled student has no CFG lane. The realtime protocol defaults
        # to guidance_scale=1 for other models, so normalize it explicitly.
        request.num_inference_steps = MINWM_DEFAULT_DMD_STEPS
        request.guidance_scale = 0.0
        request.guidance_scale_2 = None
        inputs = request.condition_inputs or {}
        label_values = [
            value
            for key in ("action_labels", MINWM_ACTION_LABELS_CONDITION)
            if (value := inputs.get(key)) is not None
        ]
        weight_values = [
            value
            for key in ("action_weights", MINWM_ACTION_WEIGHTS_CONDITION)
            if (value := inputs.get(key)) is not None
        ]
        camera_actions = inputs.get("camera_actions")
        if len(label_values) + len(weight_values) + int(camera_actions is not None) > 1:
            raise ValueError(
                "pass exactly one MinWM action form: action_labels, action_weights, or camera_actions"
            )
        if label_values:
            state.receive_action_labels(validate_action_labels(label_values[0]))
        elif weight_values:
            state.receive_action_weights(validate_action_weights(weight_values[0]))
        elif camera_actions is not None:
            state.receive_camera_action_script(
                self._validate_camera_actions(camera_actions)
            )
        await save_realtime_first_frame(
            session,
            request,
            required_error="MinWM realtime inference requires first_frame",
            cache_remote_urls=True,
        )

    def ingest_event(self, session: GenerateSession, event: RealtimeEvent) -> str:
        state = self._state(session)
        if event.kind == "action_labels":
            labels = validate_action_labels(event.payload)
            state.receive_action_labels(labels, event_id=event.event_id)
            return f"kind=action_labels, frames={len(labels)}"
        if event.kind == "action_weights":
            weights = validate_action_weights(event.payload)
            state.receive_action_weights(weights, event_id=event.event_id)
            return f"kind=action_weights, frames={len(weights)}"
        if event.kind == "camera_actions":
            return state.receive_camera_control_event_payload(
                event.payload,
                event_id=event.event_id,
                validate_camera_actions=self._validate_camera_actions,
            )
        if event.kind == "prompt":
            prompt = self._validate_prompt(event.payload)
            state.receive_prompt(prompt, event_id=event.event_id)
            return f"kind=prompt, prompt_len={len(prompt)}"
        raise ValueError(f"unsupported MinWM event kind: {event.kind}")

    def sample_chunk_inputs(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
        chunk_size: int,
    ) -> RealtimeChunkInputs:
        state = self._state(session)
        request = session.request
        if request is None:
            raise ValueError("realtime request is not initialized")
        prompt_updated = False
        prompt = request.prompt
        if chunk.index > 0 and state.prompt_queue.has_events("prompt"):
            prompt = state.prompt_queue.pop_latest("prompt")
            request.prompt = prompt
            prompt_updated = True
        condition_inputs = {}
        if state.action_mode == "weights":
            temporal_factor = int(
                server_args.pipeline_config.vae_config.arch_config.scale_factor_temporal
            )
            rows = state.sample_action_weights(chunk_size * temporal_factor)
            condition_inputs[MINWM_ACTION_WEIGHTS_CONDITION] = [
                rows[start : start + temporal_factor]
                for start in range(0, len(rows), temporal_factor)
            ]
        else:
            condition_inputs[MINWM_ACTION_LABELS_CONDITION] = (
                state.sample_action_labels(chunk_size)
            )
        if request.max_chunks is not None:
            condition_inputs[MINWM_TOTAL_CHUNKS_CONDITION] = int(request.max_chunks)
        if prompt_updated:
            condition_inputs[MINWM_PROMPT_UPDATED_CONDITION] = True
        return RealtimeChunkInputs(prompt=prompt, condition_inputs=condition_inputs)

    def build_sampling_params(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
        chunk_inputs: RealtimeChunkInputs,
        chunk_size: int,
    ):
        del server_args
        request = session.request
        if request is None:
            raise ValueError("realtime request is not initialized")
        return build_realtime_sampling_params(
            chunk.request_id,
            request=request,
            chunk_inputs=chunk_inputs,
            num_frames=1,
            num_inference_steps=request.num_inference_steps or MINWM_DEFAULT_DMD_STEPS,
            chunk_size=chunk_size,
        )

    def get_realtime_event_id(self, session: GenerateSession) -> int | None:
        state = self._state(session)
        if state.label_event_id is not None:
            return state.label_event_id
        if state.weight_event_id is not None:
            return state.weight_event_id
        return state.latest_sampled_event_id

    def clear_state(self, session: GenerateSession) -> None:
        state = session.adapter_state
        if isinstance(state, MinWMRealtimeState):
            state.clear()
