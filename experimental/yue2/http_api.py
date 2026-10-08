# SPDX-License-Identifier: Apache-2.0
"""Direct-curl HTTP route for the YuE2 multimodal pipeline.

This module is intentionally kept out of the core multimodal API until the
audio-generation request schema is upstreamed.
"""
from __future__ import annotations

import json
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    build_sampling_params,
    process_generation_batch,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.configs.sample.sampling_params import generate_request_id
from sglang.srt.observability.trace import extract_trace_headers


router = APIRouter(prefix="/v1/audio", tags=["audio"])


class Yue2AudioGenerationsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    style: str = Field(..., description="style/tags")
    lyrics: str = Field("", description="lyrics text")
    cot: Literal["off", "melody", "full"] = "full"
    abc: Optional[str] = Field(None, description="optional ABC score")
    seed: int = Field(831001, ge=0)
    ode_steps: int = Field(32, ge=1, le=128)
    semantic_max_tokens: int = Field(9000, ge=1)
    abc_max_tokens: int = Field(4096, ge=1)
    vae_core_frames: int = Field(1024, ge=1)
    vae_halo_frames: int = Field(16, ge=0)
    output_path: Optional[str] = Field(None, description="output directory")
    output_file_name: Optional[str] = Field(None, description="output file name")
    cfg_scale: Optional[float] = Field(None, description="classifier-free guidance scale")
    artifacts_dir: Optional[str] = Field(
        None, description="export plan/semantic/latent/request/config artifacts here")


@router.post("/generations")
@router.post("/music/generations")
async def generate_audio(
    request: Yue2AudioGenerationsRequest,
    raw_request: Request,
):
    server_args = get_global_server_args()
    request_id = generate_request_id()
    prompt_payload = {
        "style": request.style,
        "lyrics": request.lyrics,
        "cot": request.cot,
        "seed": request.seed,
        "abc": request.abc,
        "cfg_scale": request.cfg_scale,
        "id": request.output_file_name or request_id,
    }
    if request.artifacts_dir:
        prompt_payload["artifacts_dir"] = request.artifacts_dir
    sampling = build_sampling_params(
        request_id,
        prompt=json.dumps(prompt_payload, ensure_ascii=False),
        style=request.style,
        lyrics=request.lyrics,
        cot=request.cot,
        request_abc=request.abc,
        seed=request.seed,
        cfg_scale=request.cfg_scale,
        num_inference_steps=request.ode_steps,
        ode_steps=request.ode_steps,
        abc_max_tokens=request.abc_max_tokens,
        semantic_max_tokens=request.semantic_max_tokens,
        vae_core_frames=request.vae_core_frames,
        vae_halo_frames=request.vae_halo_frames,
        output_sample_rate=48000,
        output_path=request.output_path or server_args.output_path,
        output_file_name=request.output_file_name or f"{request_id}.wav",
    )
    trace_headers = extract_trace_headers(raw_request.headers)
    batch = prepare_request(
        server_args=server_args,
        sampling_params=sampling,
        external_trace_header=trace_headers,
    )
    save_file_path_list, result = await process_generation_batch(
        async_scheduler_client, batch
    )
    if not save_file_path_list:
        raise HTTPException(status_code=500, detail="YuE2 returned no audio output")
    audio_path = save_file_path_list[0]
    extra = getattr(result, "extra", None) or {}
    return {
        "id": request_id,
        "object": "audio.generation",
        "audio": audio_path,
        "sample_rate": 48000,
        "inference_time_s": (
            result.metrics.total_duration_s
            if result.metrics is not None
            else None
        ),
        "peak_memory_mb": result.peak_memory_mb,
        "yue2_timing": extra.get("yue2_timing"),
        "yue2_stage_seconds": {
            key: extra.get(key)
            for key in ("yue2_nar_seconds", "yue2_vae_seconds", "yue2_audio_seconds")
        },
        "yue2_nar_fast": extra.get("yue2_nar_fast"),
    }
