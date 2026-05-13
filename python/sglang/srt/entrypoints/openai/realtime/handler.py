"""WebSocket entry for realtime transcription.

Handles accept, concurrency, cleanup. Event loop is in session.py.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import WebSocket, WebSocketDisconnect
from openai.types.realtime import RealtimeErrorEvent
from openai.types.realtime.realtime_error import RealtimeError

from sglang.srt.entrypoints.openai.realtime.session import RealtimeConnection
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import random_uuid

logger = logging.getLogger(__name__)


async def _safe_send(websocket: WebSocket, text: str) -> None:
    """Send text; debug-log if the peer is already gone."""
    try:
        await websocket.send_text(text)
    except (WebSocketDisconnect, RuntimeError) as e:
        logger.debug("[realtime] send failed (peer gone): %s", e)


async def _safe_close(websocket: WebSocket) -> None:
    """Close, ignoring already-closed errors (starlette raises RuntimeError)."""
    try:
        await websocket.close()
    except (WebSocketDisconnect, RuntimeError) as e:
        logger.debug("[realtime] close failed (already closed): %s", e)


async def _reject_before_session(
    websocket: WebSocket,
    code: str,
    message: str,
    *,
    error_type: str = "invalid_request_error",
) -> None:
    """Accept, send a wire error envelope, close. Each step is best-effort."""
    try:
        await websocket.accept()
    except (WebSocketDisconnect, RuntimeError) as e:
        logger.debug("[realtime] reject: accept failed: %s", e)
        return
    logger.info("[realtime] rejected (%s)", code)
    envelope = RealtimeErrorEvent(
        event_id=f"event_{random_uuid()}",
        type="error",
        error=RealtimeError(type=error_type, code=code, message=message),
    )
    await _safe_send(websocket, envelope.model_dump_json())
    await _safe_close(websocket)


async def handle_realtime_transcription(
    websocket: WebSocket,
    tokenizer_manager: TokenizerManager,
    adapter: TranscriptionAdapter,
    server_args: ServerArgs,
    session_semaphore: asyncio.Semaphore,
) -> None:
    """Accept the WS, run a RealtimeConnection, release the semaphore.

    `finally` must release: leaked slots block new connections. Single-task,
    so `ws.receive()` returning a disconnect is the only termination trigger.
    """
    # Pre-session rejects come before acquire: unsupported / over-capacity
    # peers shouldn't hold slots.

    # RealtimeConnection.__init__ crashes on a non-chunked adapter; reject early.
    if not adapter.supports_chunked_streaming:
        await _reject_before_session(
            websocket,
            "not_supported",
            "Model does not support streaming ASR",
        )
        return

    # Atomic in asyncio: no `await` between `locked()` and `acquire()`'s
    # fast path, so no other coroutine can slip in between the check and
    # the decrement.
    if session_semaphore.locked():
        await _reject_before_session(
            websocket,
            "too_many_sessions",
            f"Maximum concurrent sessions reached "
            f"({server_args.asr_max_concurrent_sessions}).",
            error_type="rate_limit_exceeded",
        )
        return

    async with session_semaphore:
        try:
            try:
                await websocket.accept()
            except (WebSocketDisconnect, RuntimeError) as e:
                logger.debug("[realtime] accept failed: %s", e)
                return
            connection = RealtimeConnection(
                websocket, tokenizer_manager, adapter, server_args
            )
            await connection.run()
        except WebSocketDisconnect:
            logger.info("[realtime] client disconnected (normal)")
        except Exception:
            logger.exception("[realtime] unexpected error in session")
            envelope = RealtimeErrorEvent(
                event_id=f"event_{random_uuid()}",
                type="error",
                error=RealtimeError(
                    type="server_error",
                    code="inference_failed",
                    message="Internal server error",
                ),
            )
            await _safe_send(websocket, envelope.model_dump_json())
        finally:
            await _safe_close(websocket)
