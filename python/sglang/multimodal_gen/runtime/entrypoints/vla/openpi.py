# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
import traceback

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from sglang.multimodal_gen.runtime.entrypoints.action_utils import (
    action_metadata,
    infer_action,
    pack_msgpack,
    unpack_msgpack,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

router = APIRouter()


@router.websocket("/openpi/policy")
async def openpi_policy_ws(websocket: WebSocket):
    await websocket.accept()
    server_args: ServerArgs = websocket.app.state.server_args
    await websocket.send_bytes(pack_msgpack(action_metadata(server_args)))

    prev_total_time = None
    while True:
        try:
            start_time = time.monotonic()
            observation = unpack_msgpack(await websocket.receive_bytes())
            observation.setdefault("output_format", "numpy")
            infer_start = time.monotonic()
            action = await infer_action(observation, server_args)
            infer_ms = (time.monotonic() - infer_start) * 1000
            action.setdefault("server_timing", {})["infer_ms"] = infer_ms
            if prev_total_time is not None:
                action["server_timing"]["prev_total_ms"] = prev_total_time * 1000
            await websocket.send_bytes(pack_msgpack(action))
            prev_total_time = time.monotonic() - start_time
        except WebSocketDisconnect:
            break
        except Exception:
            await websocket.send_text(traceback.format_exc())
            await websocket.close(code=1011, reason="Internal server error")
            raise
