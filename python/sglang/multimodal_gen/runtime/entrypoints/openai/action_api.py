# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
import traceback

from fastapi import APIRouter, Request, Response, WebSocket, WebSocketDisconnect

from sglang.multimodal_gen.runtime.entrypoints.action_utils import (
    action_generation_response,
    action_metadata,
    action_raw_response,
    infer_action,
    pack_msgpack,
    unpack_msgpack,
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
    await websocket.accept()
    server_args: ServerArgs = websocket.app.state.server_args
    await websocket.send_bytes(pack_msgpack(action_metadata(server_args)))

    prev_total_time = None
    while True:
        try:
            start_time = time.monotonic()
            payload = unpack_msgpack(await websocket.receive_bytes())
            _prefer_numpy_output(payload)
            infer_start = time.monotonic()
            output = await infer_action(payload, server_args)
            response = action_generation_response(
                output,
                server_args,
                preserve_numpy=True,
            )
            response.setdefault("server_timing", {})["infer_ms"] = (
                time.monotonic() - infer_start
            ) * 1000
            if prev_total_time is not None:
                response["server_timing"]["prev_total_ms"] = prev_total_time * 1000
            await websocket.send_bytes(pack_msgpack(response))
            prev_total_time = time.monotonic() - start_time
        except WebSocketDisconnect:
            break
        except Exception:
            await websocket.send_text(traceback.format_exc())
            await websocket.close(code=1011, reason="Internal server error")
            raise
