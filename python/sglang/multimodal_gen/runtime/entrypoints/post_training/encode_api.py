"""Encode-only HTTP API (``POST /encode``)."""

from __future__ import annotations

import msgspec
from fastapi import APIRouter, HTTPException, Response

from sglang.multimodal_gen.configs.sample.sampling_params import generate_request_id
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import build_sampling_params
from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    EncodeRequest,
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


@router.post(
    "/encode",
    response_class=Response,
    responses={200: {"content": {"application/msgpack": {}}}},
)
async def encode(request: EncodeRequest):
    request_id = generate_request_id()
    sampling_kwargs = dict(
        prompt=request.prompt,
        image_path=request.image_path,
        width=request.width,
        height=request.height,
        save_output=False,
    )
    sampling_kwargs.update(request.extra_sampling_params or {})
    try:
        sampling_params = build_sampling_params(
            request_id, **{k: v for k, v in sampling_kwargs.items() if v is not None}
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid sampling params: {exc}"
        ) from exc
    pipeline_request = prepare_request(
        server_args=get_global_server_args(), sampling_params=sampling_params
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
    payload = {
        "request_id": request_id,
        "encoded": _maybe_serialize(output_batch.encoded),
    }
    return Response(
        content=msgspec.msgpack.encode(payload), media_type="application/msgpack"
    )
