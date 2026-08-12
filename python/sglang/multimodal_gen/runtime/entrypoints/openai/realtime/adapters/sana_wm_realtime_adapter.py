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
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.base import (
    normalize_sana_wm_camera_actions,
    parse_sana_wm_action_string,
    snap_sana_wm_num_frames,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.self_forcing import (
    SanaWMSelfForcingSampler,
)
from sglang.multimodal_gen.runtime.realtime.states import (
    RealtimeCameraControlState,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
        GenerateSession,
        RealtimeChunkContext,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
        OutputBatch,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


SANA_WM_DEFAULT_SIZE = "1280x704"
SANA_WM_DEFAULT_NUM_FRAMES = 1081
SANA_WM_DEFAULT_FPS = 16
SANA_WM_DEFAULT_STEPS = 4
SANA_WM_DEFAULT_GUIDANCE = 1.0
SANA_WM_CONTROL_PULSE_FRAMES = 8


def _normalize_sana_wm_state_actions(actions: list[Any]) -> list[str]:
    return [str(action).lower() for action in actions]


class SanaWMRealtimeAdapterState(RealtimeCameraControlState):
    def __init__(self):
        super().__init__(
            min_pulse_items=SANA_WM_CONTROL_PULSE_FRAMES,
            script_maxlen=2048,
            max_transitions=512,
            normalize_state_actions=_normalize_sana_wm_state_actions,
        )
        self.base_condition_inputs: dict[str, Any] = {}

    def clear(self) -> None:
        super().clear()
        self.base_condition_inputs.clear()

    def receive_camera_control_event_payload(
        self,
        payload: Any,
        *,
        event_id: int | None,
    ) -> str:
        return super().receive_camera_control_event_payload(
            payload,
            event_id=event_id,
            validate_camera_actions=SanaWMRealtimeAdapter._validate_camera_actions,
        )


class SanaWMRealtimeAdapter(BaseRealtimeModelAdapter):
    def create_state(self) -> SanaWMRealtimeAdapterState:
        return SanaWMRealtimeAdapterState()

    def _state(self, session: GenerateSession) -> SanaWMRealtimeAdapterState:
        state = session.adapter_state
        if not isinstance(state, SanaWMRealtimeAdapterState):
            raise TypeError("SANA-WM realtime adapter state is not initialized")
        return state

    @staticmethod
    def _validate_camera_actions(payload: Any) -> list[list[str]]:
        return normalize_sana_wm_camera_actions(
            payload, error_label="camera_actions event payload"
        )

    @staticmethod
    def _raw_frame_count(result: OutputBatch) -> int | None:
        if result.raw_frame_batches is None:
            return None
        return sum(len(frames) for frames in result.raw_frame_batches)

    async def on_init(
        self,
        session: GenerateSession,
        request: RealtimeVideoGenerationsRequest,
    ) -> None:
        request.size = request.size or SANA_WM_DEFAULT_SIZE
        if request.num_frames is not None:
            request.num_frames = int(request.num_frames)
        else:
            # Open-ended session: keep num_frames unset so prepare_next_request
            # samples uniform action chunks (no front-loaded segmentation), and
            # flag the stage explicitly via condition_inputs —
            # build_sampling_params strips None fields, so the per-chunk batch
            # would otherwise carry the SamplingParams default num_frames.
            request.condition_inputs = {
                **(request.condition_inputs or {}),
                "sana_wm_open_ended": True,
            }
        request.fps = int(request.fps or SANA_WM_DEFAULT_FPS)
        request.num_inference_steps = int(
            request.num_inference_steps or SANA_WM_DEFAULT_STEPS
        )
        request.guidance_scale = float(
            request.guidance_scale or SANA_WM_DEFAULT_GUIDANCE
        )
        if request.negative_prompt is None:
            request.negative_prompt = ""
        if request.generator_device is None:
            request.generator_device = "cuda"

        state = self._state(session)
        condition_inputs = dict(request.condition_inputs or {})
        camera_actions = condition_inputs.pop("camera_actions", None)
        action = condition_inputs.pop("action", None)
        if camera_actions is not None and action is not None:
            raise ValueError("pass only one of camera_actions or action")
        if camera_actions is not None:
            state.receive_camera_control_event_payload(camera_actions, event_id=None)
        if action is not None:
            if not isinstance(action, str) or not action:
                raise ValueError("action condition input must be a non-empty string")
            state.receive_camera_action_script(
                parse_sana_wm_action_string(action), event_id=None
            )
        state.base_condition_inputs = condition_inputs

        await save_realtime_first_frame(
            session,
            request,
            required_error="SANA-WM realtime requires first_frame",
            cache_remote_urls=True,
        )

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
        if event.kind == "action":
            if not isinstance(event.payload, str) or not event.payload:
                raise ValueError("action event payload must be a non-empty string")
            camera_actions = parse_sana_wm_action_string(event.payload)
            state.receive_camera_action_script(camera_actions, event_id=event.event_id)
            return f"kind=action, frames={len(camera_actions)}"
        raise ValueError(f"unsupported event kind: {event.kind}")

    def sample_chunk_inputs(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
        chunk_size: int,
    ) -> RealtimeChunkInputs:
        action_chunk_size = self._action_chunk_size(
            session,
            server_args,
            chunk,
            chunk_size,
        )
        state = self._state(session)
        request = session.request
        if request is None:
            raise ValueError("realtime request is not initialized")

        condition_inputs = dict(state.base_condition_inputs) if chunk.index == 0 else {}
        camera_actions = state.sample_camera_actions(action_chunk_size)
        if camera_actions is not None:
            condition_inputs["camera_actions"] = camera_actions
        return RealtimeChunkInputs(
            prompt=request.prompt,
            condition_inputs=condition_inputs,
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

        return build_realtime_sampling_params(
            chunk.request_id,
            request=request,
            chunk_inputs=chunk_inputs,
            num_frames=request.num_frames,
            num_inference_steps=request.num_inference_steps,
            chunk_size=chunk_size,
        )

    def _action_chunk_size(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
        chunk_size: int,
    ) -> int:
        temporal_compression = int(
            server_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        )
        # Match action sampling to the latent span used by the batch path. Chunk
        # 0 may carry a front-loaded remainder, so a fixed nfpb*tc action count
        # would read static-padded camera poses and drift from batch output.
        action_chunk_size = chunk_size * temporal_compression
        req_num_frames = (
            session.request.num_frames if session.request is not None else None
        )
        if req_num_frames is not None:
            snapped = snap_sana_wm_num_frames(
                int(req_num_frames), stride=temporal_compression
            )
            latent_t = (snapped - 1) // temporal_compression + 1
            segments = SanaWMSelfForcingSampler.create_autoregressive_segments(
                latent_t, chunk_size
            )
            idx = int(chunk.index)
            if 0 <= idx and idx + 1 < len(segments):
                action_chunk_size = (
                    segments[idx + 1] - segments[idx]
                ) * temporal_compression
        return action_chunk_size

    def get_realtime_event_id(self, session: GenerateSession) -> int | None:
        return self._state(session).latest_sampled_event_id

    def on_chunk_complete(self, session: GenerateSession, result: OutputBatch) -> None:
        if session.request is not None and self._raw_frame_count(result) == 0:
            session.request.max_chunks = session.generate_chunk_cnt + 1
        session.generate_chunk_completed()

    def clear_state(self, session: GenerateSession) -> None:
        state = session.adapter_state
        if isinstance(state, SanaWMRealtimeAdapterState):
            state.clear()
