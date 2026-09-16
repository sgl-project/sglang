"""Elastic EP scaling HTTP endpoints for dp_attention deployments."""

import json
from http import HTTPStatus

from fastapi import APIRouter, Request
from fastapi.responses import ORJSONResponse

from sglang.srt.runtime_context import get_exec
from sglang.srt.utils.auth import AuthLevel, auth_level

router = APIRouter()


@router.post("/scale_elastic_ep")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def scale_elastic_ep(raw_request: Request):
    """Request an asynchronous EP scale-up."""
    try:
        body = await raw_request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return ORJSONResponse(
            {"error": f"Invalid JSON: {e}"},
            status_code=HTTPStatus.BAD_REQUEST,
        )

    if not isinstance(body, dict):
        return ORJSONResponse(
            {"error": "request body must be a JSON object"},
            status_code=HTTPStatus.BAD_REQUEST,
        )

    new_ep_size = body.get("new_ep_size")
    if (
        not isinstance(new_ep_size, int)
        or isinstance(new_ep_size, bool)
        or new_ep_size <= 0
    ):
        return ORJSONResponse(
            {"error": "new_ep_size must be a positive integer"},
            status_code=HTTPStatus.BAD_REQUEST,
        )

    operation_id = body.get("operation_id")
    if operation_id is not None and (
        not isinstance(operation_id, str)
        or not operation_id.strip()
        or len(operation_id) > 128
    ):
        return ORJSONResponse(
            {
                "error": "operation_id must be a non-empty string of at most 128 characters"
            },
            status_code=HTTPStatus.BAD_REQUEST,
        )

    expected_instance_id = body.get("expected_instance_id")
    if expected_instance_id is not None and (
        not isinstance(expected_instance_id, str) or not expected_instance_id.strip()
    ):
        return ORJSONResponse(
            {"error": "expected_instance_id must be a non-empty string"},
            status_code=HTTPStatus.BAD_REQUEST,
        )

    expected_joining_member_ids = body.get("expected_joining_member_ids")
    if expected_joining_member_ids is not None and (
        not isinstance(expected_joining_member_ids, list)
        or len(expected_joining_member_ids) != 1
        or any(
            not isinstance(member_id, str)
            or not member_id.strip()
            or len(member_id) > 256
            for member_id in expected_joining_member_ids
        )
        or len(set(expected_joining_member_ids)) != len(expected_joining_member_ids)
    ):
        return ORJSONResponse(
            {
                "error": (
                    "expected_joining_member_ids must contain exactly one "
                    "non-empty string of at most 256 characters"
                )
            },
            status_code=HTTPStatus.BAD_REQUEST,
        )

    from sglang.srt.entrypoints.http_server import _global_state
    from sglang.srt.managers.io_struct import ScaleElasticEPReqInput

    if get_exec().moe.elastic_ep_backend is None:
        return ORJSONResponse(
            {"error": "elastic EP is not enabled (set --elastic-ep-backend)"},
            status_code=HTTPStatus.NOT_FOUND,
        )

    result = await _global_state.tokenizer_manager.scale_elastic_ep(
        ScaleElasticEPReqInput(
            new_ep_size=new_ep_size,
            operation_id=operation_id,
            expected_instance_id=expected_instance_id,
            expected_joining_member_ids=expected_joining_member_ids,
        )
    )

    if not result.success:
        return ORJSONResponse(
            {
                "error": result.message,
                "instance_id": result.instance_id,
                "operation_id": result.operation_id,
            },
            status_code=(
                HTTPStatus.CONFLICT if result.conflict else HTTPStatus.BAD_REQUEST
            ),
        )

    return ORJSONResponse(
        {
            "message": result.message,
            "old_ep_size": result.old_ep_size,
            "new_ep_size": result.new_ep_size,
            "instance_id": result.instance_id,
            "operation_id": result.operation_id,
            "pending_ep_size": result.pending_ep_size,
            "scale_phase": result.scale_phase,
        }
    )


@router.get("/is_scaling_elastic_ep")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def is_scaling_elastic_ep(raw_request: Request):
    """Return the tokenizer's mirrored Elastic EP scale state."""
    from sglang.srt.entrypoints.http_server import _global_state

    if get_exec().moe.elastic_ep_backend is None:
        return ORJSONResponse(
            {"error": "elastic EP is not enabled (set --elastic-ep-backend)"},
            status_code=HTTPStatus.NOT_FOUND,
        )

    return ORJSONResponse(_global_state.tokenizer_manager.get_elastic_ep_state())
