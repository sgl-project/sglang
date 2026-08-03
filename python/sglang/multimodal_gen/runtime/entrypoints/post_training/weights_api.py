"""Weight update API for the diffusion engine."""

from fastapi import APIRouter, Request

from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    GetWeightsChecksumReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromTensorCheckerReqInput,
    UpdateWeightFromTensorReqInput,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.srt.utils.json_response import orjson_response

router = APIRouter()


@router.post("/update_weights_from_disk")
async def update_weights_from_disk(request: Request):
    """Update model weights from disk inplace without restarting the server."""
    body = await request.json()
    model_path = body.get("model_path")
    if not model_path:
        return orjson_response(
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
        return orjson_response(
            {"success": False, "message": str(e)},
            status_code=500,
        )

    if response.output is None:
        return orjson_response(
            {
                "success": False,
                "message": response.error or "Unknown status",
            },
            status_code=500,
        )

    result = response.output
    return orjson_response(
        result,
        status_code=200 if result["success"] else 400,
    )


@router.post("/update_weights_from_tensor")
async def update_weights_from_tensor(request: Request):
    """Update model weights from serialized tensor payloads."""
    body = await request.json()
    serialized_named_tensors = body.get("serialized_named_tensors")
    if not serialized_named_tensors:
        return orjson_response(
            {"success": False, "message": "serialized_named_tensors is required"},
            status_code=400,
        )

    req = UpdateWeightFromTensorReqInput(
        serialized_named_tensors=serialized_named_tensors,
        load_format=body.get("load_format"),
        target_modules=body.get("target_modules"),
        weight_update_mode=body.get("weight_update_mode"),
        lora_alpha=body.get("lora_alpha"),
        lora_rank=body.get("lora_rank"),
    )

    try:
        response = await async_scheduler_client.forward(req)
    except Exception as e:
        return orjson_response(
            {"success": False, "message": str(e)},
            status_code=500,
        )

    result = response.output
    return orjson_response(
        result,
        status_code=200 if result["success"] else 400,
    )


@router.post("/update_weights_from_tensor_checker")
async def update_weights_from_tensor_checker(request: Request):
    """Verify live module weights against expected SHA-256 values."""
    body = await request.json()
    target_module = body.get("target_module")
    if not target_module:
        return orjson_response(
            {"success": False, "message": "target_module is required"},
            status_code=400,
        )

    expected_named_tensors_sha256 = body.get("expected_named_tensors_sha256")
    if (
        not isinstance(expected_named_tensors_sha256, dict)
        or not expected_named_tensors_sha256
    ):
        return orjson_response(
            {
                "success": False,
                "message": "expected_named_tensors_sha256 is required",
            },
            status_code=400,
        )

    req = UpdateWeightFromTensorCheckerReqInput(
        target_module=target_module,
        expected_named_tensors_sha256=expected_named_tensors_sha256,
    )

    try:
        response = await async_scheduler_client.forward(req)
    except Exception as e:
        return orjson_response(
            {"success": False, "message": str(e)},
            status_code=500,
        )

    result = response.output
    success = result.get("success", False)
    message = result.get("message", "Unknown status")
    return orjson_response(
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
        return orjson_response({"error": str(e)}, status_code=500)

    return orjson_response(response.output, status_code=200)


@router.post("/release_memory_occupation")
async def release_memory_occupation():
    """Release GPU memory occupation (sleep the engine)."""
    try:
        response = await async_scheduler_client.forward(
            ReleaseMemoryOccupationReqInput()
        )
    except Exception as e:
        return orjson_response({"success": False, "message": str(e)}, status_code=500)

    if response.output is None:
        return orjson_response(
            {
                "success": False,
                "message": response.error or "Unknown status",
            },
            status_code=500,
        )

    payload = response.output
    success = bool(payload["success"])
    return orjson_response(payload, status_code=200 if success else 400)


@router.post("/resume_memory_occupation")
async def resume_memory_occupation():
    """Resume GPU memory occupation (wake the engine)."""
    try:
        response = await async_scheduler_client.forward(
            ResumeMemoryOccupationReqInput()
        )
    except Exception as e:
        return orjson_response({"success": False, "message": str(e)}, status_code=500)

    if response.output is None:
        return orjson_response(
            {
                "success": False,
                "message": response.error or "Unknown status",
            },
            status_code=500,
        )

    payload = response.output
    success = bool(payload["success"])
    return orjson_response(payload, status_code=200 if success else 400)
