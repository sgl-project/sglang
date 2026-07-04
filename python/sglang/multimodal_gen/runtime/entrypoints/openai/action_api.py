# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
import traceback
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from sglang.multimodal_gen.runtime.entrypoints.action_utils import (
    action_generation_response,
    action_metadata,
    infer_action,
    pack_msgpack,
    unpack_msgpack,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.srt.utils.json_response import orjson_response

router = APIRouter(prefix="/v1/actions", tags=["actions"])


@router.post("/generations")
async def create_action_generation(request: Request, payload: dict[str, Any]):
    server_args: ServerArgs = request.app.state.server_args
    output = await infer_action(payload, server_args)
    return orjson_response(action_generation_response(output, server_args))


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
            infer_start = time.monotonic()
            output = await infer_action(payload, server_args)
            response = action_generation_response(output, server_args)
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
