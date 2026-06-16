# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass, field
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
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    prepare_request,
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


@dataclass(slots=True)
class RealtimeChunkInputs:
    """Sampled from realtime control state, consumed by the Req"""

    prompt: str
    condition_inputs: dict[str, Any] = field(default_factory=dict)


async def save_realtime_first_frame(
    session: GenerateSession,
    request: RealtimeVideoGenerationsRequest,
    *,
    required_error: str | None = None,
    cache_remote_urls: bool = False,
) -> None:
    first_frame = request.first_frame
    if first_frame is None:
        if required_error is not None:
            raise ValueError(required_error)
        return

    server_args = get_global_server_args()
    if server_args.input_save_path is not None:
        uploads_dir = server_args.input_save_path
        os.makedirs(uploads_dir, exist_ok=True)
    else:
        if session.input_temp_dir is None:
            session.input_temp_dir = tempfile.mkdtemp(prefix="sglang_input_")
        uploads_dir = session.input_temp_dir

    if (
        cache_remote_urls
        and isinstance(first_frame, str)
        and first_frame.lower().startswith(("http://", "https://"))
    ):
        suffix = os.path.splitext(first_frame.split("?", 1)[0])[1]
        digest = hashlib.sha256(first_frame.encode("utf-8")).hexdigest()[:16]
        target_path = os.path.join(uploads_dir, f"realtime_ref_{digest}{suffix}")
        if os.path.exists(target_path):
            request.first_frame = target_path
            return
    else:
        target_path = os.path.join(uploads_dir, f"{session.id}_first_frame")

    request.first_frame = await save_image_to_path(first_frame, target_path)


def build_realtime_sampling_params(
    request_id: str,
    *,
    request: RealtimeVideoGenerationsRequest,
    chunk_inputs: RealtimeChunkInputs,
    num_frames: int | None,
    num_inference_steps: int | None,
    chunk_size: int,
):
    return build_sampling_params(
        request_id,
        prompt=chunk_inputs.prompt,
        size=request.size,
        num_frames=num_frames,
        fps=request.fps,
        image_path=request.first_frame,
        output_file_name=request_id,
        save_output=False,
        seed=request.seed,
        generator_device=request.generator_device,
        num_inference_steps=num_inference_steps,
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


class BaseRealtimeModelAdapter:
    def __init__(self):
        self.output_adapter = RawRGBRealtimeOutputAdapter()

    async def on_init(
        self,
        session: GenerateSession,
        request: RealtimeVideoGenerationsRequest,
    ) -> None:
        raise NotImplementedError

    def create_state(self) -> Any:
        """create a state for managing runtime states"""
        raise NotImplementedError

    def ingest_event(
        self,
        session: GenerateSession,
        event: RealtimeEvent,
    ) -> str:
        """
        Ingest a realtime endpoint event and install it into the model's realtime control queues
        """
        raise NotImplementedError

    async def wait_for_next_chunk(self, session: GenerateSession) -> None:
        del session

    def get_chunk_size(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
    ) -> int:
        del session, chunk
        arch_config = server_args.pipeline_config.dit_config.arch_config
        return int(getattr(arch_config, "num_frames_per_block", 3))

    def sample_chunk_inputs(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
        chunk_size: int,
    ) -> RealtimeChunkInputs:
        raise NotImplementedError

    def build_sampling_params(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
        chunk_inputs: RealtimeChunkInputs,
        chunk_size: int,
    ):
        raise NotImplementedError

    def get_realtime_event_id(self, session: GenerateSession) -> int | None:
        del session
        return None

    def prepare_next_request(
        self,
        session: GenerateSession,
        server_args: ServerArgs,
        chunk: RealtimeChunkContext,
    ) -> Req:
        chunk_size = self.get_chunk_size(session, server_args, chunk)
        chunk_inputs = self.sample_chunk_inputs(
            session,
            server_args,
            chunk,
            chunk_size,
        )
        sampling_params = self.build_sampling_params(
            session,
            server_args,
            chunk,
            chunk_inputs,
            chunk_size,
        )
        batch = prepare_request(
            server_args=server_args,
            sampling_params=sampling_params,
        )
        self.apply_realtime_request_fields(
            batch,
            session,
            chunk,
            event_id=self.get_realtime_event_id(session),
        )
        return batch

    def apply_realtime_request_fields(
        self,
        batch: Req,
        session: GenerateSession,
        chunk: RealtimeChunkContext,
        *,
        event_id: int | None,
    ) -> None:
        batch.realtime_session_id = session.id
        batch.return_raw_frames = True
        batch.block_idx = chunk.index
        batch.realtime_event_id = event_id
        if session.request is None:
            return
        batch.realtime_output_format = session.request.realtime_output_format
        batch.realtime_preview_max_width = session.request.realtime_preview_max_width
        batch.realtime_output_pacing = bool(session.request.realtime_output_pacing)
        batch.realtime_causal_sink_size = session.request.realtime_causal_sink_size
        batch.realtime_causal_kv_cache_num_frames = (
            session.request.realtime_causal_kv_cache_num_frames
        )

    async def send_output(
        self,
        ws: WebSocket,
        session: GenerateSession,
        result: OutputBatch,
        batch: Req,
    ) -> RealtimeFrameSendStats:
        """send the generate output (usually frames) back via websocket"""
        return await self.output_adapter.send(ws, session, result, batch)

    def on_chunk_complete(self, session: GenerateSession, result: OutputBatch) -> None:
        del result
        session.generate_chunk_completed()

    def clear_state(self, session: GenerateSession) -> None:
        del session

    def dispose(self, session: GenerateSession) -> None:
        self.clear_state(session)
        self.output_adapter.reset()
