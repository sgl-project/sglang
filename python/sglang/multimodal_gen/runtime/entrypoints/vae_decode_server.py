# SPDX-License-Identifier: Apache-2.0
"""Standalone exact causal VAE decoder for realtime pipeline overlap."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, Request, Response

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.utils import post_process_sample
from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import VAELoader
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime.vae import (
    CausalVaeDecodingStage,
    RealtimeVAEDecodeState,
)
from sglang.multimodal_gen.runtime.remote.vae_decode_protocol import (
    RAW_RGB_CONTENT_TYPE,
    SCHEMA_VERSION,
    build_raw_transport_batches,
    packb,
    payload_to_tensor,
    store_raw_transport_batches_in_shared_memory,
    unpackb,
)
from sglang.multimodal_gen.runtime.server_args import (
    ServerArgs,
    set_global_server_args,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.realtime_video import (
    build_raw_rgb_frame_batches,
)

logger = init_logger(__name__)


@dataclass
class _SessionState:
    decode_state: RealtimeVAEDecodeState
    reset_causal_decode_state: Any

    def dispose(self) -> None:
        self.decode_state.reset_causal_decode_state = self.reset_causal_decode_state
        self.decode_state.dispose()


class ExactRealtimeVAEDecoder:
    """One active causal session backed by the original model VAE."""

    def __init__(self, server_args: ServerArgs, vae_path: str) -> None:
        self.server_args = server_args
        loader = VAELoader()
        self.vae, _ = loader.load(
            vae_path,
            server_args,
            component_name="vae",
            transformers_or_diffusers=loader.expected_library,
        )
        stage_cls = CausalVaeDecodingStage
        if server_args.pipeline_class_name in {
            "MinWMCausalDMDPipeline",
            "MinWMCausalUniPCPipeline",
        }:
            from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minwm import (
                MinWMCausalVaeDecodingStage,
            )

            stage_cls = MinWMCausalVaeDecodingStage
        self.stage = stage_cls(vae=self.vae)
        self.sessions: dict[str, _SessionState] = {}
        self.active_session_id: str | None = None
        logger.info("exact realtime VAE decoder loaded %s", vae_path)

    def _get_session(self, session_id: str, *, first_chunk: bool) -> _SessionState:
        if self.active_session_id not in (None, session_id):
            raise RuntimeError(
                "exact causal VAE decoder supports one active session; "
                f"active={self.active_session_id} requested={session_id}"
            )
        self.active_session_id = session_id
        state = self.sessions.get(session_id)
        if state is None:
            state = _SessionState(
                decode_state=RealtimeVAEDecodeState(),
                reset_causal_decode_state=self.stage._get_causal_decode_reset_fn(),
            )
            self.sessions[session_id] = state
        if first_chunk and callable(state.reset_causal_decode_state):
            state.reset_causal_decode_state()
        return state

    def close_session(self, session_id: str) -> None:
        state = self.sessions.pop(session_id, None)
        if state is not None:
            state.dispose()
        if self.active_session_id == session_id:
            self.active_session_id = None

    @torch.no_grad()
    def decode(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"schema mismatch: {payload.get('schema_version')}")
        session_id = str(payload["session_id"])
        block_idx = int(payload["block_idx"])
        output_block_idx = int(payload.get("output_block_idx", block_idx))
        first_chunk = bool(payload["first_chunk"])
        is_final_chunk = bool(payload.get("is_final_chunk", False))
        state = self._get_session(session_id, first_chunk=first_chunk)

        total_start = time.monotonic()
        latents = payload_to_tensor(payload["latents"])
        unpack_ms = (time.monotonic() - total_start) * 1000.0

        decode_start = time.monotonic()
        frames = self.stage.decode_causal(
            latents,
            self.server_args,
            first_chunk=first_chunk,
            decode_state=state.decode_state,
        )
        frames = self.server_args.pipeline_config.post_decoding(
            frames, self.server_args
        )
        trim_leading_frames = int(payload.get("trim_leading_frames", 0))
        if trim_leading_frames:
            frames = frames[:, :, trim_leading_frames:]
        decode_ms = (time.monotonic() - decode_start) * 1000.0

        raw_start = time.monotonic()
        sampling_params = SamplingParams(
            width=payload.get("width"),
            height=payload.get("height"),
            fps=int(payload.get("fps") or 24),
        )
        batch = Req(
            sampling_params=sampling_params,
            request_id=str(payload.get("request_id") or ""),
            block_idx=output_block_idx,
            return_raw_frames=True,
        )
        output_batch = OutputBatch(output=frames)
        raw_frame_batches, raw_frame_metadata = build_raw_rgb_frame_batches(
            frames,
            batch,
            output_batch,
            post_process_sample,
        )
        raw_ms = (time.monotonic() - raw_start) * 1000.0
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "block_idx": output_block_idx,
            "raw_frame_content_type": RAW_RGB_CONTENT_TYPE,
            "raw_frame_metadata": raw_frame_metadata,
            "stats": {
                "server_unpack_ms": unpack_ms,
                "server_decode_ms": decode_ms,
                "server_raw_rgb_ms": raw_ms,
            },
        }
        if payload.get("realtime_output_format") == "raw":
            transport_start = time.monotonic()
            transport_batches = build_raw_transport_batches(raw_frame_batches)
            result["stats"]["server_raw_transport_build_ms"] = (
                time.monotonic() - transport_start
            ) * 1000.0
            if payload.get("response_transport") == "shared_memory":
                shared_memory_start = time.monotonic()
                transport_batches = store_raw_transport_batches_in_shared_memory(
                    transport_batches
                )
                result["raw_transport_storage"] = "shared_memory"
                result["stats"]["server_shared_memory_write_ms"] = (
                    time.monotonic() - shared_memory_start
                ) * 1000.0
            else:
                result["raw_transport_storage"] = "http"
            result["raw_transport_batches"] = transport_batches
        else:
            result["raw_frame_batches"] = raw_frame_batches
        result["stats"]["server_total_ms"] = (time.monotonic() - total_start) * 1000.0
        logger.info(
            "exact realtime VAE chunk complete: session_id=%s block_idx=%d "
            "decode_ms=%.3f raw_ms=%.3f total_ms=%.3f frames=%d",
            session_id,
            output_block_idx,
            decode_ms,
            raw_ms,
            result["stats"]["server_total_ms"],
            sum(len(sample) for sample in raw_frame_batches),
        )
        if is_final_chunk:
            self.close_session(session_id)
        return result


def create_app(decoder: ExactRealtimeVAEDecoder) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "schema_version": SCHEMA_VERSION,
            "active_session_id": decoder.active_session_id,
        }

    @app.post("/decode")
    async def decode(request: Request) -> Response:
        result = decoder.decode(unpackb(await request.body()))
        return Response(content=packb(result), media_type="application/msgpack")

    return app


def main(argv: list[str] | None = None) -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--vae-decoder-host", default="0.0.0.0")
    pre_parser.add_argument("--vae-decoder-port", type=int, default=31000)
    pre_parser.add_argument("--vae-path", required=True)
    known, remaining = pre_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(parents=[pre_parser])
    ServerArgs.add_cli_args(parser)
    raw_args, unknown = parser.parse_known_args(argv)
    server_args = ServerArgs.from_cli_args(raw_args, unknown)
    set_global_server_args(server_args)
    decoder = ExactRealtimeVAEDecoder(server_args, known.vae_path)
    uvicorn.run(
        create_app(decoder),
        host=known.vae_decoder_host,
        port=known.vae_decoder_port,
        log_level=server_args.log_level,
        ws_per_message_deflate=False,
    )


if __name__ == "__main__":
    main()
