# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
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
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.realtime.condition_events import (
    ControlSignal,
    ControlStateSamplingQueue,
    ControlStateTransition,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.utils import (
    normalize_camera_actions,
    parse_action_string,
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


SANA_WM_DEFAULT_SIZE = "1280x704"
SANA_WM_DEFAULT_NUM_FRAMES = 1081
SANA_WM_DEFAULT_FPS = 16
SANA_WM_DEFAULT_STEPS = 4
SANA_WM_DEFAULT_GUIDANCE = 1.0
SANA_WM_CONTROL_PULSE_FRAMES = 8


class SanaWMRealtimeAdapterState:
    def __init__(self):
        self.camera_state = ControlStateSamplingQueue(
            default_item=[],
            min_pulse_items=SANA_WM_CONTROL_PULSE_FRAMES,
            max_transitions=512,
        )
        self.camera_script_queue: deque[ControlSignal] = deque(maxlen=2048)
        self.base_condition_inputs: dict[str, Any] = {}
        self.latest_sampled_event_id: int | None = None

    def clear(self) -> None:
        self.camera_state.clear()
        self.camera_script_queue.clear()
        self.base_condition_inputs.clear()
        self.latest_sampled_event_id = None

    def receive_camera_script(
        self,
        camera_actions: list[list[str]],
        *,
        event_id: int | None = None,
    ) -> None:
        self.camera_state.clear()
        self.camera_script_queue.clear()
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
                    [str(action).lower() for action in actions],
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

        camera_actions = SanaWMRealtimeAdapter._validate_camera_actions(payload)
        self.receive_camera_script(camera_actions, event_id=event_id)
        return f"kind=camera_actions, mode=script, frames={len(camera_actions)}"

    def sample_camera_actions(self, chunk_size: int) -> list[list[str]] | None:
        if self.camera_script_queue:
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

        action_list = self.camera_state.sample_chunk(chunk_size)
        if action_list is None:
            return None
        self.latest_sampled_event_id = self.camera_state.latest_sampled_seq_id()
        return [list(actions) for actions in action_list]


class SanaWMRealtimeAdapter(RealtimeModelAdapter):
    name = "sana_wm_realtime"

    def __init__(self):
        self.output_adapter = RawRGBRealtimeOutputAdapter()

    def create_state(self) -> SanaWMRealtimeAdapterState:
        return SanaWMRealtimeAdapterState()

    def _state(self, session: GenerateSession) -> SanaWMRealtimeAdapterState:
        state = session.adapter_state
        if not isinstance(state, SanaWMRealtimeAdapterState):
            raise TypeError("SANA-WM realtime adapter state is not initialized")
        return state

    @staticmethod
    def _validate_camera_actions(payload: Any) -> list[list[str]]:
        return normalize_camera_actions(
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
        if request.first_frame is None:
            raise ValueError("SANA-WM realtime requires first_frame")

        request.size = SANA_WM_DEFAULT_SIZE
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
            state.receive_camera_event_payload(camera_actions, event_id=None)
        if action is not None:
            if not isinstance(action, str) or not action:
                raise ValueError("action condition input must be a non-empty string")
            state.receive_camera_script(parse_action_string(action), event_id=None)
        state.base_condition_inputs = condition_inputs

        server_args = get_global_server_args()
        if server_args.input_save_path is not None:
            uploads_dir = server_args.input_save_path
            os.makedirs(uploads_dir, exist_ok=True)
        else:
            if session.input_temp_dir is None:
                session.input_temp_dir = tempfile.mkdtemp(prefix="sglang_input_")
            uploads_dir = session.input_temp_dir

        if isinstance(request.first_frame, str) and request.first_frame.lower().startswith(
            ("http://", "https://")
        ):
            suffix = os.path.splitext(request.first_frame.split("?", 1)[0])[1]
            digest = hashlib.sha256(request.first_frame.encode("utf-8")).hexdigest()[:16]
            target_path = os.path.join(uploads_dir, f"realtime_ref_{digest}{suffix}")
            if os.path.exists(target_path):
                request.first_frame = target_path
                return
        else:
            target_path = os.path.join(uploads_dir, f"{session.id}_first_frame")
        image_path = await save_image_to_path(request.first_frame, target_path)
        request.first_frame = image_path

    async def wait_for_next_chunk(self, session: GenerateSession) -> None:
        del session

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
        if event.kind == "action":
            if not isinstance(event.payload, str) or not event.payload:
                raise ValueError("action event payload must be a non-empty string")
            camera_actions = parse_action_string(event.payload)
            state.receive_camera_script(camera_actions, event_id=event.event_id)
            return f"kind=action, frames={len(camera_actions)}"
        raise ValueError(f"unsupported event kind: {event.kind}")

    def _sample_chunk_inputs(
        self,
        session: GenerateSession,
        chunk: RealtimeChunkContext,
        action_chunk_size: int,
    ) -> RealtimeChunkInputs:
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

    def _build_sampling_params(
        self,
        session: GenerateSession,
        chunk: RealtimeChunkContext,
        chunk_inputs: RealtimeChunkInputs,
        chunk_size: int,
    ):
        request = session.request
        if request is None:
            raise ValueError("realtime request is not initialized")

        return build_sampling_params(
            chunk.request_id,
            prompt=chunk_inputs.prompt,
            size=request.size,
            num_frames=request.num_frames,
            fps=request.fps,
            image_path=request.first_frame,
            output_file_name=chunk.request_id,
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
            condition_inputs=chunk_inputs.condition_inputs,
            realtime_chunk_size=chunk_size,
        )

    def prepare_next_request(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
    ) -> Req:
        arch_config = server_args.pipeline_config.dit_config.arch_config
        chunk_size = int(getattr(arch_config, "num_frames_per_block", 3))
        temporal_compression = int(
            server_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        )
        # Sample action frames in lockstep with the session's FRONT-LOADED autoregressive
        # segmentation (chunk 0 carries the latent_t % nfpb remainder, matching the batch
        # path). A uniform nfpb*tc count leaves chunk 0's extra front-loaded latent frames
        # reading static-padded camera poses, so the windowed camera_conditions/chunk_plucker
        # diverge from batch and the error compounds over the clip (see realtime-vs-batch
        # audit). Match each chunk's latent span instead.
        action_chunk_size = chunk_size * temporal_compression
        req_num_frames = (
            session.request.num_frames if session.request is not None else None
        )
        if req_num_frames is not None:
            from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.self_forcing import (
                SanaWMSelfForcingSampler,
            )
            from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.utils import (
                snap_num_frames,
            )

            snapped = snap_num_frames(int(req_num_frames), stride=temporal_compression)
            latent_t = (snapped - 1) // temporal_compression + 1
            segments = SanaWMSelfForcingSampler.create_autoregressive_segments(
                latent_t, chunk_size
            )
            idx = int(chunk.index)
            if 0 <= idx and idx + 1 < len(segments):
                action_chunk_size = (
                    segments[idx + 1] - segments[idx]
                ) * temporal_compression
        chunk_inputs = self._sample_chunk_inputs(session, chunk, action_chunk_size)
        sampling_params = self._build_sampling_params(
            session,
            chunk,
            chunk_inputs,
            chunk_size,
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
            # Forward the full transport config like the LingBot adapter does —
            # the shared RawRGB output adapter / realtime_video_api consume
            # preview width + pacing too; dropping them silently disabled both
            # features for SANA-WM sessions.
            batch.realtime_output_format = session.request.realtime_output_format
            batch.realtime_preview_max_width = (
                session.request.realtime_preview_max_width
            )
            batch.realtime_output_pacing = bool(
                session.request.realtime_output_pacing
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
        if session.request is not None and self._raw_frame_count(result) == 0:
            session.request.max_chunks = session.generate_chunk_cnt + 1
        session.generate_chunk_completed()

    def dispose(self, session: GenerateSession) -> None:
        state = session.adapter_state
        if isinstance(state, SanaWMRealtimeAdapterState):
            state.clear()
        self.output_adapter.reset()
