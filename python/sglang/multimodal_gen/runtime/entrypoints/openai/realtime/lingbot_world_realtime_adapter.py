# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING, Any

from fastapi import WebSocket

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
    RawRGBRealtimeOutputAdapter,
    RealtimeFrameSendStats,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    build_sampling_params,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.pipelines_core.condition_events import (
    ConditionEvent,
    ConditionEventQueue,
    ConditionSamplingParams,
    ControlSignal,
)
from sglang.multimodal_gen.runtime.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
        GenerateSession,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
        OutputBatch,
        Req,
    )


class LingBotWorldRealtimeState:
    def __init__(self):
        self.events = ConditionEventQueue(
            max_events={"prompt": 1, "camera_actions": 512}
        )

    def clear(self) -> None:
        self.events.clear()

    def append_prompt(self, prompt: str) -> None:
        self.events.push(
            ConditionEvent(
                kind="prompt",
                payload=ControlSignal(kind="prompt", payload=prompt),
            )
        )

    def append_camera_actions(self, camera_actions: list[list[str]]) -> None:
        signals = [
            ControlSignal(kind="camera_actions", payload=list(actions))
            for actions in camera_actions
        ]
        self.events.push(ConditionEvent(kind="camera_actions", payload=signals))

    def sample_prompt(self) -> str:
        prompt = self.events.pop_latest("prompt")
        if not isinstance(prompt, str):
            raise ValueError("prompt event payload must be a string")
        return prompt

    def sample_camera_actions(self, chunk_size: int) -> list[list[str]] | None:
        signal_payloads = self.events.sample_chunk(
            "camera_actions",
            ConditionSamplingParams(chunk_size=chunk_size, default_item=[]),
        )
        if signal_payloads is None:
            return None
        return [list(actions) for actions in signal_payloads]

    def has_prompt(self) -> bool:
        return self.events.has_events("prompt")


class LingBotWorldRealtimeAdapter:
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
            camera_actions = self._validate_camera_actions(event.payload)
            state.append_camera_actions(camera_actions)
            return f"kind=camera_actions, frames={len(camera_actions)}"
        if event.kind == "prompt":
            if not isinstance(event.payload, str) or not event.payload:
                raise ValueError("prompt event payload must be a non-empty string")
            state.append_prompt(event.payload)
            return f"kind=prompt, prompt_len={len(event.payload)}"
        raise ValueError(f"unsupported event kind: {event.kind}")

    def build_sampling_params(self, session: GenerateSession):
        state = self._state(session)
        request = session.request
        if request is None:
            raise ValueError("realtime request is not initialized")

        if session.generate_chunk_cnt == 0:
            prompt = request.prompt
        elif state.has_prompt():
            prompt = state.sample_prompt()
            request.prompt = prompt
        else:
            prompt = request.prompt

        return build_sampling_params(
            session.request_id,
            prompt=prompt,
            size=request.size,
            num_frames=request.num_frames,
            fps=request.fps,
            image_path=request.first_frame,
            output_file_name=session.request_id,
            save_output=False,
            seed=request.seed,
            generator_device=request.generator_device,
            num_inference_steps=request.num_inference_steps,
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
        )

    def prepare_request(self, session: GenerateSession, batch: Req) -> Req:
        state = self._state(session)
        batch.session = session.realtime_session
        batch.realtime_session_id = session.id
        batch.return_raw_frames = True
        batch.block_idx = session.generate_chunk_cnt
        pipeline_config = get_global_server_args().pipeline_config
        chunk_size = int(pipeline_config.dit_config.arch_config.num_frames_per_block)
        batch.realtime_chunk_size = chunk_size
        camera_actions = state.sample_camera_actions(chunk_size)
        if camera_actions is not None:
            batch.condition_inputs["camera_actions"] = camera_actions
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
