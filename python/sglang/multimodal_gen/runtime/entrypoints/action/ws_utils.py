# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect

from sglang.multimodal_gen.runtime.entrypoints.action.protocol import (
    action_metadata,
    infer_action,
    pack_msgpack,
    unpack_msgpack,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


async def run_action_msgpack_ws(
    websocket: WebSocket,
    server_args: ServerArgs,
    *,
    prepare_payload: Callable[[dict[str, Any]], None],
    build_response: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    if (
        getattr(server_args.pipeline_config, "action_websocket_sessions", False)
        is not True
    ):
        return await _serve_action_ws(
            websocket, server_args, prepare_payload, build_response
        )

    # A connection owns its episode even if a client supplies a reused session ID.
    session_id = uuid4().hex

    def prepare_episode(payload):
        prepare_payload(payload)
        observation = payload
        if "input" in payload:
            observation = payload["input"].setdefault("observation", {})
        observation["session_id"] = session_id

    try:
        await _serve_action_ws(websocket, server_args, prepare_episode, build_response)
    finally:
        from sglang.multimodal_gen.runtime.entrypoints.control_requests import (
            ReleaseRealtimeSessionReq,
        )
        from sglang.multimodal_gen.runtime.scheduler_client import (
            async_scheduler_client,
        )

        try:
            await async_scheduler_client.forward(
                ReleaseRealtimeSessionReq(session_id=session_id)
            )
        except Exception:
            logging.getLogger(__name__).warning(
                "Could not release action session %s", session_id, exc_info=True
            )


async def _serve_action_ws(
    websocket: WebSocket,
    server_args: ServerArgs,
    prepare_payload: Callable[[dict[str, Any]], None],
    build_response: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    await websocket.accept()
    await websocket.send_bytes(pack_msgpack(action_metadata(server_args)))

    prev_total_time = None
    while True:
        try:
            start_time = time.monotonic()
            payload = unpack_msgpack(await websocket.receive_bytes())
            prepare_payload(payload)
            infer_start = time.monotonic()
            output = await infer_action(payload, server_args)
            response = build_response(output)
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
            try:
                await websocket.send_bytes(
                    pack_msgpack({"error": traceback.format_exc()})
                )
            except Exception:
                pass
            await websocket.close(code=1011, reason="Internal server error")
            raise
