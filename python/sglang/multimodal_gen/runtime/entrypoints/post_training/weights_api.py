"""Weight update API for the diffusion engine."""

from fastapi import APIRouter, Request
from fastapi.responses import ORJSONResponse

from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    GetWeightsChecksumReqInput,
    UpdateWeightFromDiskReqInput,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client

from sglang.srt.managers.io_struct import (
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)

router = APIRouter()


@router.post("/update_weights_from_disk")
async def update_weights_from_disk(request: Request):
    """Update model weights from disk inplace without restarting the server."""
    body = await request.json()
    model_path = body.get("model_path")
    if not model_path:
        return ORJSONResponse(
            {"success": False, "message": "model_path is required"},
            status_code=400,
        )

    req = UpdateWeightFromDiskReqInput(
        model_path=model_path,
        flush_cache=body.get("flush_cache", True),
        target_modules=body.get("target_modules"),
    )

    try:
        response = await async_scheduler_client.forward(req)
    except Exception as e:
        return ORJSONResponse(
            {"success": False, "message": str(e)},
            status_code=500,
        )

    result = response.output
    success = result.get("success", False)
    message = result.get("message", "Unknown status")
    return ORJSONResponse(
        {"success": success, "message": message},
        status_code=200 if success else 400,
    )


@router.post("/get_weights_checksum")
async def get_weights_checksum(request: Request):
    """Return SHA-256 checksum of each requested module's weights."""
    body = await request.json()
    req = GetWeightsChecksumReqInput(
        module_names=body.get("module_names"),
    )

    try:
        response = await async_scheduler_client.forward(req)
    except Exception as e:
        return ORJSONResponse({"error": str(e)}, status_code=500)

    return ORJSONResponse(response.output, status_code=200)

@router.post("/release_memory_occupation")
async def release_memory_occupation(request: Request):
    """Release GPU memory occupation (sleep the engine)."""
    body = await request.json()
    tags = body.get("tags")

    req = ReleaseMemoryOccupationReqInput(tags=tags)

    try:
        response = await async_scheduler_client.forward(req)
    except Exception as e:
        return ORJSONResponse({"success": False, "message": str(e)}, status_code=500)

    # The structure of response.output depends on how your Scheduler/GPUWorker returns it.
    # Perform a robust handling here that is compatible with both dict and non-dict outputs.
    out = response.output
    if isinstance(out, dict):
        return ORJSONResponse(out, status_code=200 if out.get("success", True) else 400)
    return ORJSONResponse({"success": True, "output": out}, status_code=200)


@router.post("/resume_memory_occupation")
async def resume_memory_occupation(request: Request):
    """Resume GPU memory occupation (wake the engine)."""
    body = await request.json()
    tags = body.get("tags")

    req = ResumeMemoryOccupationReqInput(tags=tags)

    try:
        response = await async_scheduler_client.forward(req)
    except Exception as e:
        return ORJSONResponse({"success": False, "message": str(e)}, status_code=500)

    out = response.output
    if isinstance(out, dict):
        return ORJSONResponse(out, status_code=200 if out.get("success", True) else 400)
    return ORJSONResponse({"success": True, "output": out}, status_code=200)
