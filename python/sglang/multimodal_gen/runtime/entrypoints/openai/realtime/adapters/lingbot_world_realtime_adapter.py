# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import tempfile
from collections import deque
from typing import TYPE_CHECKING, Any

from fastapi import WebSocket

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
    RealtimeChunkInputs,
    RealtimeModelAdapter,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
    RawRGBRealtimeOutputAdapter,
    RealtimeFrameSendStats,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    build_sampling_params,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    prepare_request,
)
from sglang.multimodal_gen.runtime.realtime.condition_events import (
    ConditionEvent,
    ConditionEventQueue,
    ControlSignal,
    ControlStateSamplingQueue,
    ControlStateTransition,
)
from sglang.multimodal_gen.runtime.server_args import get_global_server_args

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


LINGBOT_REALTIME_DEFAULT_NUM_INFERENCE_STEPS = 4
LINGBOT_REALTIME_MIN_CONDITION_CHUNKS = 2


class LingBotWorldRealtimeState:
    def __init__(self):
        self.events = ConditionEventQueue(max_events={"prompt": 1})
        self.camera_state = ControlStateSamplingQueue(
            default_item=[],
            min_pulse_items=1,
            max_transitions=512,
        )
        self.camera_script_queue: deque[ControlSignal] = deque(maxlen=512)
        self.latest_sampled_event_id: int | None = None

    def clear(self) -> None:
        self.events.clear()
        self.camera_state.clear()
        self.camera_script_queue.clear()
        self.latest_sampled_event_id = None

    def receive_prompt(self, prompt: str, *, event_id: int | None = None) -> None:
        self.events.push(
            ConditionEvent(
                kind="prompt",
                payload=ControlSignal(
                    kind="prompt",
                    payload=prompt,
                    seq_id=event_id,
                ),
            )
        )

    def receive_camera_script(
        self,
        camera_actions: list[list[str]],
        *,
        event_id: int | None = None,
    ) -> None:
        self.camera_script_queue.clear()
        self.camera_state.clear()
        for actions in camera_actions:
            self.camera_script_queue.append(
                ControlSignal(
                    kind="camera_actions",
                    payload=list(actions),
                    seq_id=event_id,
                )
            )

    def receive_camera_state_transitions(
        self,
        transitions: list[ControlStateTransition],
    ) -> None:
        self.camera_script_queue.clear()
        self.camera_state.push_many(transitions)

    def receive_camera_actions(
        self,
        camera_actions: list[list[str]],
        *,
        event_id: int | None = None,
    ) -> None:
        self.receive_camera_script(camera_actions, event_id=event_id)

    def receive_camera_state(
        self,
        actions: list[str],
        *,
        event_id: int | None = None,
        timestamp_ms: int | None = None,
    ) -> None:
        self.receive_camera_state_transitions(
            [
                ControlStateTransition(
                    payload=list(actions),
                    seq_id=event_id,
                    timestamp_ms=timestamp_ms,
                )
            ]
        )

    def _sample_camera_script(self, chunk_size: int) -> list[list[str]]:
        chunk: list[list[str]] = []
        latest_event_id = self.latest_sampled_event_id
        while self.camera_script_queue and len(chunk) < chunk_size:
            signal = self.camera_script_queue.popleft()
            chunk.append(list(signal.payload))
            latest_event_id = signal.seq_id
        while len(chunk) < chunk_size:
            chunk.append([])
        self.latest_sampled_event_id = latest_event_id
        return chunk

    def _camera_state_transition(
        self,
        actions: list[str],
        *,
        event_id: int | None,
        timestamp_ms: int | None,
    ) -> ControlStateTransition:
        return ControlStateTransition(
            payload=list(actions),
            seq_id=event_id,
            timestamp_ms=timestamp_ms,
        )

    def _camera_transitions_from_event_payload(
        self,
        payload: dict[str, Any],
        *,
        event_id: int | None,
    ) -> list[ControlStateTransition]:
        transitions = payload.get("transitions")
        if not isinstance(transitions, list):
            raise ValueError("camera_actions state payload requires transitions")
        result = []
        for transition in transitions:
            if not isinstance(transition, dict):
                raise ValueError("camera_actions transition must be a map")
            actions = transition.get("actions")
            if not isinstance(actions, list):
                raise ValueError("camera_actions transition actions must be a list")
            timestamp_ms = transition.get("client_ts_ms")
            if timestamp_ms is not None:
                timestamp_ms = int(timestamp_ms)
            result.append(
                self._camera_state_transition(
                    list(actions),
                    event_id=event_id,
                    timestamp_ms=timestamp_ms,
                )
            )
        return result

    def receive_camera_event_payload(
        self,
        payload: Any,
        *,
        event_id: int | None,
    ) -> str:
        if isinstance(payload, dict) and payload.get("mode") == "state":
            transitions = self._camera_transitions_from_event_payload(
                payload,
                event_id=event_id,
            )
            self.receive_camera_state_transitions(transitions)
            return f"kind=camera_actions, mode=state, transitions={len(transitions)}"

        camera_actions = LingBotWorldRealtimeAdapter._validate_camera_actions(payload)
        self.receive_camera_script(camera_actions, event_id=event_id)
        return f"kind=camera_actions, mode=script, frames={len(camera_actions)}"

    def sample_prompt(self) -> str:
        prompt = self.events.pop_latest("prompt")
        if not isinstance(prompt, str):
            raise ValueError("prompt event payload must be a string")
        self.latest_sampled_event_id = self.events.last_sampled_seq_id("prompt")
        return prompt

    def sample_camera_actions(self, chunk_size: int) -> list[list[str]] | None:
        """samples a sequence of camera actions for the chunk with chunk_size frames

        Args:
            chunk_size: number of frames
        """
        if self.camera_script_queue:
            return self._sample_camera_script(chunk_size)
        action_list = self.camera_state.sample_chunk(chunk_size)
        if action_list is None:
            return None
        self.latest_sampled_event_id = self.camera_state.latest_sampled_seq_id()
        return [list(actions) for actions in action_list]

    def has_prompt(self) -> bool:
        return self.events.has_events("prompt")


class LingBotWorldRealtimeAdapter(RealtimeModelAdapter):
    name = "lingbot_world"

    def __init__(self):
        self.output_adapter = RawRGBRealtimeOutputAdapter()

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
            state.receive_camera_script(self._validate_camera_actions(camera_actions))

        if request.first_frame is None:
            return

        server_args = get_global_server_args()
        if server_args.input_save_path is not None:
            uploads_dir = server_args.input_save_path
            os.makedirs(uploads_dir, exist_ok=True)
        else:
            if session.input_temp_dir is None:
                session.input_temp_dir = tempfile.mkdtemp(prefix="sglang_input_")
            uploads_dir = session.input_temp_dir

        target_path = os.path.join(uploads_dir, f"{session.id}_first_frame")
        image_path = await save_image_to_path(request.first_frame, target_path)
        request.first_frame = image_path

    async def wait_for_next_chunk(self, session: GenerateSession) -> None:
        del session

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
            return state.receive_camera_event_payload(
                event.payload,
                event_id=event.event_id,
            )
        elif event.kind == "prompt":
            if not isinstance(event.payload, str) or not event.payload:
                raise ValueError("prompt event payload must be a non-empty string")
            state.receive_prompt(event.payload, event_id=event.event_id)
            return f"kind=prompt, prompt_len={len(event.payload)}"
        raise ValueError(f"unsupported event kind: {event.kind}")

    def _sample_chunk_inputs(
        self,
        session: GenerateSession,
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

    def _build_sampling_params(
        self,
        session: GenerateSession,
        chunk: RealtimeChunkContext,
        chunk_inputs: RealtimeChunkInputs,
        chunk_size: int,
        server_args: ServerArgs,
    ):
        request = session.request
        if request is None:
            raise ValueError("realtime request is not initialized")

        num_frames = self._condition_num_frames(
            request=request,
            server_args=server_args,
            chunk_size=chunk_size,
        )

        return build_sampling_params(
            chunk.request_id,
            prompt=chunk_inputs.prompt,
            size=request.size,
            num_frames=num_frames,
            fps=request.fps,
            image_path=request.first_frame,
            output_file_name=chunk.request_id,
            save_output=False,
            seed=request.seed,
            generator_device=request.generator_device,
            num_inference_steps=(
                request.num_inference_steps
                or LINGBOT_REALTIME_DEFAULT_NUM_INFERENCE_STEPS
            ),
            guidance_scale=request.guidance_scale,
            guidance_scale_2=request.guidance_scale_2,
            negative_prompt=request.negative_prompt,
            enable_teacache=request.enable_teacache,
            enable_frame_interpolation=request.enable_frame_interpolation,
            frame_interpolation_exp=request.frame_interpolation_exp,
            frame_interpolation_scale=request.frame_interpolation_scale,
            frame_interpolation_model_path=request.frame_interpolation_model_path,
            enable_upscaling=request.enable_upscaling,
            upscaling_model_path=request.upscaling_model_path,
            upscaling_scale=request.upscaling_scale,
            diffusers_kwargs=request.diffusers_kwargs,
            profile=request.profile,
            num_profiled_timesteps=request.num_profiled_timesteps,
            profile_all_stages=request.profile_all_stages,
            perf_dump_path=request.perf_dump_path,
            output_path=request.output_path,
            output_compression=request.output_compression,
            output_quality=request.output_quality,
            condition_inputs=chunk_inputs.condition_inputs,
            realtime_chunk_size=chunk_size,
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

    def prepare_next_request(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
    ) -> Req:
        """build a new request for the next chunk"""
        pipeline_config = server_args.pipeline_config
        chunk_size = int(pipeline_config.dit_config.arch_config.num_frames_per_block)
        chunk_inputs = self._sample_chunk_inputs(session, chunk, chunk_size)
        sampling_params = self._build_sampling_params(
            session,
            chunk,
            chunk_inputs,
            chunk_size,
            server_args,
        )
        batch = prepare_request(
            server_args=server_args,
            sampling_params=sampling_params,
        )
        batch.session = session.realtime_session
        batch.realtime_session_id = session.id
        batch.return_raw_frames = True
        batch.block_idx = chunk.index
        batch.realtime_event_id = self._state(session).latest_sampled_event_id
        if session.request is not None:
            batch.realtime_output_format = session.request.realtime_output_format
            batch.realtime_preview_max_width = (
                session.request.realtime_preview_max_width
            )
            batch.realtime_output_pacing = bool(session.request.realtime_output_pacing)
            batch.realtime_causal_sink_size = session.request.realtime_causal_sink_size
            batch.realtime_causal_kv_cache_num_frames = (
                session.request.realtime_causal_kv_cache_num_frames
            )
        return batch

    async def send_output(
        self,
        ws: WebSocket,
        session: GenerateSession,
        result: OutputBatch,
        batch: Req,
    ) -> RealtimeFrameSendStats:
        return await self.output_adapter.send(ws, session, result, batch)

    def on_chunk_complete(self, session: GenerateSession, result: OutputBatch) -> None:
        del result
        session.generate_chunk_completed()

    def dispose(self, session: GenerateSession) -> None:
        state = session.adapter_state
        if isinstance(state, LingBotWorldRealtimeState):
            state.clear()
        self.output_adapter.reset()
