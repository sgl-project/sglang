# SPDX-License-Identifier: Apache-2.0
"""Omni HTTP routes mounted into the standard SRT FastAPI server."""

from __future__ import annotations

import json
from typing import Any, Callable

from fastapi import APIRouter, Depends, Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from sglang.srt.managers.io_struct import OmniGenerateReqInput


def create_srt_omni_router(
    *,
    get_tokenizer_manager: Callable[[], Any],
    validate_json_request: Callable[[Request], Any],
) -> APIRouter:
    router = APIRouter()

    # sessions live in tokenizer manager, so these routes stay in the SRT process
    @router.post("/v1/omni/generate", dependencies=[Depends(validate_json_request)])
    async def omni_generate_request(raw_request: Request):
        payload = await raw_request.json()
        if bool(payload.get("stream", False)):
            request = OmniGenerateReqInput(payload=payload, stream=True)
            return StreamingResponse(
                _stream_omni_generate_events(
                    get_tokenizer_manager(),
                    request,
                    raw_request,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

        result = await get_tokenizer_manager().omni_generate(
            OmniGenerateReqInput(payload=payload),
            raw_request,
        )
        if result.success:
            return ORJSONResponse(content=result.payload, status_code=200)
        return ORJSONResponse(
            content={"error": {"message": result.message}},
            status_code=result.status_code,
        )

    @router.post("/v1/omni/sessions/{session_id}/close")
    async def omni_close_session_request(session_id: str, raw_request: Request):
        payload = {"action": "close_session", "session_id": session_id}
        if raw_request.headers.get("content-length") not in {None, "0"}:
            body = await raw_request.json()
            if isinstance(body, dict):
                payload.update(body)
                payload["session_id"] = session_id
        result = await get_tokenizer_manager().omni_generate(
            OmniGenerateReqInput(payload=payload),
            raw_request,
        )
        if result.success:
            return ORJSONResponse(content=result.payload, status_code=200)
        return ORJSONResponse(
            content={"error": {"message": result.message}},
            status_code=result.status_code,
        )

    return router


async def _stream_omni_generate_events(
    tokenizer_manager: Any,
    request: OmniGenerateReqInput,
    raw_request: Request,
):
    try:
        async for event in tokenizer_manager.omni_generate_stream(
            request,
            raw_request,
        ):
            yield _encode_sse_event(event)
    except ValueError as exc:
        yield _encode_sse_event(
            {
                "type": "error",
                "error": {"message": str(exc)},
                "status_code": 499,
            }
        )
    yield b"data: [DONE]\n\n"


def _encode_sse_event(event: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n\n"
