"""Encode-only HTTP API (``POST /encode``), served with --encode-only."""

from __future__ import annotations

import msgspec
from fastapi import APIRouter, HTTPException, Response

from sglang.multimodal_gen.configs.sample.sampling_params import generate_request_id
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import build_sampling_params
from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    EncodeRequest,
    EncodeResponse,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.utils import (
    _maybe_serialize,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
router = APIRouter(tags=["encode"])


def _build_sampling_kwargs(request: EncodeRequest) -> dict:
    sampling_kwargs: dict = dict(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        seed=request.seed,
        generator_device=request.generator_device,
        width=request.width,
        height=request.height,
        num_frames=request.num_frames,
        fps=request.fps,
        image_path=request.image_path,
        video_path=request.video_path,
        suppress_logs=request.suppress_logs,
        save_output=False,
    )
    if request.extra_sampling_params:
        sampling_kwargs.update(request.extra_sampling_params)
    return {k: v for k, v in sampling_kwargs.items() if v is not None}


@router.post(
    "/encode",
    response_class=Response,
    responses={
        200: {
            "model": EncodeResponse,
            "content": {"application/msgpack": {}},
        }
    },
)
async def encode(request: EncodeRequest):
    request_id = generate_request_id()
    server_args = get_global_server_args()
    if not server_args.encode_only:
        raise HTTPException(
            status_code=400, detail="Server was not launched with --encode-only"
        )
    sampling_kwargs = _build_sampling_kwargs(request)
    try:
        sampling_params = build_sampling_params(request_id, **sampling_kwargs)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid sampling params: {exc}"
        ) from exc
    pipeline_request = prepare_request(
        server_args=server_args, sampling_params=sampling_params
    )
    try:
        output_batch: OutputBatch = await async_scheduler_client.forward(
            pipeline_request
        )
    except Exception as exc:
        logger.error("Encode failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Encode failed: {exc}") from exc
    if output_batch.error:
        raise HTTPException(status_code=500, detail=output_batch.error)
    if output_batch.encoder_output is None:
        raise HTTPException(status_code=500, detail="No encoder output produced")
    inference_time_s = (
        output_batch.metrics.total_duration_s
        if output_batch.metrics and output_batch.metrics.total_duration_s > 0
        else None
    )
    response = EncodeResponse(
        request_id=request_id,
        prompt=request.prompt,
        encoder_output=_maybe_serialize(output_batch.encoder_output),
        inference_time_s=inference_time_s,
    )
    return Response(
        content=msgspec.msgpack.encode(response.model_dump()),
        media_type="application/msgpack",
    )
