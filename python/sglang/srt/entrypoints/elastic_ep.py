"""Elastic EP scaling HTTP endpoints for dp_attention deployments."""

import json
from http import HTTPStatus

from fastapi import APIRouter, Request
from fastapi.responses import ORJSONResponse

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

    from sglang.srt.entrypoints.http_server import _global_state
    from sglang.srt.managers.io_struct import ScaleElasticEPReqInput

    if _global_state.tokenizer_manager.server_args.elastic_ep_backend is None:
        return ORJSONResponse(
            {"error": "elastic EP is not enabled (set --elastic-ep-backend)"},
            status_code=HTTPStatus.NOT_FOUND,
        )

    result = await _global_state.tokenizer_manager.scale_elastic_ep(
        ScaleElasticEPReqInput(new_ep_size=new_ep_size)
    )

    if not result.success:
        return ORJSONResponse(
            {"error": result.message},
            status_code=(
                HTTPStatus.CONFLICT
                if result.pending_ep_size is not None
                else HTTPStatus.BAD_REQUEST
            ),
        )

    return ORJSONResponse(
        {
            "message": result.message,
            "old_ep_size": result.old_ep_size,
            "new_ep_size": result.new_ep_size,
        }
    )


@router.get("/is_scaling_elastic_ep")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def is_scaling_elastic_ep(raw_request: Request):
    """Return the tokenizer's mirrored Elastic EP scale state."""
    from sglang.srt.entrypoints.http_server import _global_state

    if _global_state.tokenizer_manager.server_args.elastic_ep_backend is None:
        return ORJSONResponse(
            {"error": "elastic EP is not enabled (set --elastic-ep-backend)"},
            status_code=HTTPStatus.NOT_FOUND,
        )

    return ORJSONResponse(_global_state.tokenizer_manager.get_elastic_ep_state())
