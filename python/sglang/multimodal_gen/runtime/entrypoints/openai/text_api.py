"""Text generation API endpoint for text diffusion models (e.g., Cola-DLM)."""

import time

from fastapi import APIRouter, HTTPException, Request

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    generate_request_id,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    TextGenerationsRequest,
    TextResponse,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    build_sampling_params,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.observability.trace import extract_trace_headers

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/text", tags=["text"])


@router.post("/generations", response_model=TextResponse)
async def create_text_generation(request: Request, body: TextGenerationsRequest):
    """Generate text from a text diffusion model."""
    server_args = get_global_server_args()
    task_type = server_args.pipeline_config.task_type

    if task_type.data_type() != DataType.TEXT:
        raise HTTPException(
            status_code=400,
            detail=f"This endpoint is for text generation models. "
            f"Current model task type is {task_type.name}.",
        )

    request_id = generate_request_id()
    start_time = time.perf_counter()

    # Build sampling params from request fields
    kwargs = {
        "prompt": body.prompt,
        "save_output": False,
    }
    if body.max_new_tokens is not None:
        kwargs["max_new_tokens"] = body.max_new_tokens
    if body.num_inference_steps is not None:
        kwargs["num_inference_steps"] = body.num_inference_steps
    if body.guidance_scale is not None:
        kwargs["guidance_scale"] = body.guidance_scale
    if body.temperature is not None:
        kwargs["temperature"] = body.temperature
    if body.top_k is not None:
        kwargs["top_k"] = body.top_k
    if body.top_p is not None:
        kwargs["top_p"] = body.top_p
    if body.repetition_penalty is not None:
        kwargs["repetition_penalty"] = body.repetition_penalty
    if body.seed is not None:
        kwargs["seed"] = body.seed

    sampling_params = build_sampling_params(request_id, **kwargs)

    # Create request and send to scheduler
    trace_headers = extract_trace_headers(request.headers)
    req = prepare_request(server_args, sampling_params, trace_headers)
    req.prompt = body.prompt

    from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client

    try:
        output_batch = await async_scheduler_client.forward([req])
    except Exception as e:
        logger.error("Text generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    if output_batch.output is None:
        error_msg = output_batch.error or "Unknown error"
        raise HTTPException(
            status_code=500,
            detail=f"Model generation returned no output: {error_msg}",
        )

    # Extract text from output
    text = output_batch.output[0] if output_batch.output else ""
    inference_time = time.perf_counter() - start_time

    return TextResponse(
        id=request_id,
        text=text,
        prompt=body.prompt,
        peak_memory_mb=output_batch.peak_memory_mb,
        inference_time_s=round(inference_time, 3),
    )
