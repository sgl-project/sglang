# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, WebSocket

from sglang.multimodal_gen.runtime.entrypoints.vla.protocol import (
    action_generation_response,
    action_metadata,
    action_raw_response,
    infer_action,
    pack_msgpack,
    unpack_msgpack,
)
from sglang.multimodal_gen.runtime.entrypoints.vla.ws_utils import (
    run_action_msgpack_ws,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.srt.utils.json_response import orjson_response

router = APIRouter(prefix="/v1/actions", tags=["actions"])


def _wants_msgpack(request: Request) -> bool:
    content_type = request.headers.get("content-type", "").lower()
    accept = request.headers.get("accept", "").lower()
    return "msgpack" in content_type or "msgpack" in accept


def _response_format(payload: dict) -> str:
    runtime = payload.get("runtime") or {}
    response_format = str(runtime.get("response_format", "envelope")).lower()
    if response_format not in ("envelope", "raw"):
        raise ValueError("runtime.response_format must be 'envelope' or 'raw'")
    return response_format


def _prefer_numpy_output(payload: dict) -> None:
    runtime = payload.setdefault("runtime", {})
    runtime.setdefault("output_format", "numpy")


@router.post("/generations")
async def create_action_generation(request: Request):
    server_args: ServerArgs = request.app.state.server_args
    try:
        if "msgpack" in request.headers.get("content-type", "").lower():
            payload = unpack_msgpack(await request.body())
        else:
            payload = await request.json()
        wants_msgpack = _wants_msgpack(request)
        if wants_msgpack:
            _prefer_numpy_output(payload)
        output = await infer_action(payload, server_args)
        if _response_format(payload) == "raw":
            response = action_raw_response(output, preserve_numpy=wants_msgpack)
        else:
            response = action_generation_response(
                output,
                server_args,
                preserve_numpy=wants_msgpack,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if wants_msgpack:
        return Response(
            content=pack_msgpack(response), media_type="application/msgpack"
        )
    return orjson_response(response)


@router.get("/metadata")
async def action_metadata_endpoint(request: Request):
    return orjson_response(action_metadata(request.app.state.server_args))


@router.websocket("/realtime")
async def action_realtime_ws(websocket: WebSocket):
    server_args: ServerArgs = websocket.app.state.server_args
    await run_action_msgpack_ws(
        websocket,
        server_args,
        prepare_payload=_prefer_numpy_output,
        build_response=lambda output: action_generation_response(
            output,
            server_args,
            preserve_numpy=True,
        ),
    )
