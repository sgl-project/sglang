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

    expected_joining_allocation_ids = body.get("expected_joining_allocation_ids")
    if expected_joining_allocation_ids is not None and (
        not isinstance(expected_joining_allocation_ids, list)
        or len(expected_joining_allocation_ids) != 1
        or any(
            not isinstance(allocation_id, str)
            or not allocation_id.strip()
            or len(allocation_id) > 256
            for allocation_id in expected_joining_allocation_ids
        )
        or len(set(expected_joining_allocation_ids))
        != len(expected_joining_allocation_ids)
    ):
        return ORJSONResponse(
            {
                "error": (
                    "expected_joining_allocation_ids must contain exactly one "
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
            expected_joining_allocation_ids=expected_joining_allocation_ids,
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


@router.post("/recover_elastic_ep")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def recover_elastic_ep(raw_request: Request):
    """Restore one fenced TP1 Elastic EP slot and warm it without user traffic."""
    try:
        body = await raw_request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return ORJSONResponse(
            {"error": f"Invalid JSON: {e}"}, status_code=HTTPStatus.BAD_REQUEST
        )
    if not isinstance(body, dict):
        return ORJSONResponse(
            {"error": "request body must be a JSON object"},
            status_code=HTTPStatus.BAD_REQUEST,
        )
    operation_id = body.get("operation_id")
    allocation_id = body.get("allocation_id")
    topology_generation = body.get("topology_generation")
    rank_offset = body.get("rank_offset")
    if (
        not isinstance(operation_id, str)
        or not operation_id.strip()
        or len(operation_id) > 128
        or not isinstance(allocation_id, str)
        or not allocation_id.strip()
        or len(allocation_id) > 256
        or not isinstance(topology_generation, int)
        or isinstance(topology_generation, bool)
        or topology_generation < 0
        or not isinstance(rank_offset, int)
        or isinstance(rank_offset, bool)
        or rank_offset <= 0
    ):
        return ORJSONResponse(
            {
                "error": (
                    "operation_id, allocation_id, non-negative topology_generation, "
                    "and positive rank_offset are required"
                )
            },
            status_code=HTTPStatus.BAD_REQUEST,
        )
    if get_exec().moe.elastic_ep_backend is None:
        return ORJSONResponse(
            {"error": "elastic EP is not enabled (set --elastic-ep-backend)"},
            status_code=HTTPStatus.NOT_FOUND,
        )

    from sglang.srt.entrypoints.http_server import _global_state
    from sglang.srt.managers.io_struct import RecoverElasticEPReqInput

    result = await _global_state.tokenizer_manager.recover_elastic_ep(
        RecoverElasticEPReqInput(
            operation_id=operation_id,
            runtime_instance_id="",
            topology_generation=topology_generation,
            allocation_id=allocation_id,
            rank_offset=rank_offset,
        )
    )
    if not result.success:
        return ORJSONResponse(
            {
                "error": result.message,
                "operation_id": result.operation_id,
                "recovery_phase": result.recovery_phase,
            },
            status_code=(
                HTTPStatus.CONFLICT if result.conflict else HTTPStatus.BAD_REQUEST
            ),
        )
    return ORJSONResponse(
        {
            "message": result.message,
            "operation_id": result.operation_id,
            "recovery_phase": result.recovery_phase,
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
