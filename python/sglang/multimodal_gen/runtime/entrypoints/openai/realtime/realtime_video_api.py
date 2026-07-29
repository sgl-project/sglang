# SPDX-License-Identifier: Apache-2.0

import asyncio
import shutil
import time
from typing import TYPE_CHECKING

import msgspec.msgpack
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
    RealtimeChunkContext,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
    RealtimeFrameSendStats,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.registry import (
    get_realtime_model_adapter,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.timing import (
    RealtimeStageTimer,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    process_generation_batch,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    ReleaseRealtimeSessionReq,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/realtime_video", tags=["realtime"])
_ACTIVE_SESSION_IDS: set[str] = set()
_ACTIVE_SESSION_WAIT_SECONDS = 1.0
_ACTIVE_SESSION_WAIT_INTERVAL_SECONDS = 0.1


def _transport_ms(value: float) -> int:
    return max(0, int(value + 0.5))


async def _wait_for_active_session_slot(
    *,
    timeout_s: float = _ACTIVE_SESSION_WAIT_SECONDS,
    interval_s: float = _ACTIVE_SESSION_WAIT_INTERVAL_SECONDS,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while _ACTIVE_SESSION_IDS and time.monotonic() < deadline:
        await asyncio.sleep(interval_s)
    return not _ACTIVE_SESSION_IDS


def _log_realtime_chunk_timing(
    session: GenerateSession,
    chunk: RealtimeChunkContext,
    batch: "Req",
    request_prepare_ms: float,
    scheduler_forward_ms: float,
    chunk_total_ms: float,
    send_stats: RealtimeFrameSendStats,
) -> None:
    logger.info(
        "realtime chunk timing: session_id=%s request_id=%s "
        "chunk_idx=%s event_id=%s condition_kinds=%s "
        "request_prepare=%.2fms scheduler_forward=%.2fms "
        "output_pace=%.2fms "
        "header_pack=%.2fms "
        "header_write=%.2fms raw_payload_build=%.2fms raw_write=%.2fms "
        "ws_write=%.2fms chunk_total=%.2fms batches=%d frames=%d "
        "frame_shape=%s raw_bytes=%d ws_payload_bytes=%d content_type=%s",
        session.id,
        chunk.request_id,
        batch.block_idx,
        getattr(batch, "realtime_event_id", None),
        sorted(batch.condition_inputs) if batch.condition_inputs else [],
        request_prepare_ms,
        scheduler_forward_ms,
        send_stats["pace_wait_ms"],
        send_stats["header_pack_ms"],
        send_stats["header_write_ms"],
        send_stats["raw_payload_build_ms"],
        send_stats["raw_write_ms"],
        send_stats["ws_write_ms"],
        chunk_total_ms,
        send_stats["num_batches"],
        send_stats["num_frames"],
        send_stats["frame_shape"],
        send_stats["raw_bytes"],
        send_stats["ws_payload_bytes"],
        send_stats["content_type"],
    )


async def _send_realtime_chunk_stats(
    ws: WebSocket,
    session: GenerateSession,
    chunk: RealtimeChunkContext,
    batch: "Req",
    request_prepare_ms: float,
    scheduler_forward_ms: float,
    chunk_total_ms: float,
    send_stats: RealtimeFrameSendStats,
) -> None:
    await ws.send_bytes(
        msgspec.msgpack.encode(
            {
                "type": "chunk_stats",
                "session_id": session.id,
                "request_id": chunk.request_id,
                "chunk_index": batch.block_idx,
                "event_id": getattr(batch, "realtime_event_id", None),
                "request_prepare_ms": _transport_ms(request_prepare_ms),
                "scheduler_forward_ms": _transport_ms(scheduler_forward_ms),
                "pace_wait_ms": _transport_ms(send_stats["pace_wait_ms"]),
                "header_write_ms": _transport_ms(send_stats["header_write_ms"]),
                "raw_payload_build_ms": _transport_ms(
                    send_stats["raw_payload_build_ms"]
                ),
                "raw_write_ms": _transport_ms(send_stats["raw_write_ms"]),
                "ws_write_ms": _transport_ms(send_stats["ws_write_ms"]),
                "chunk_total_ms": _transport_ms(chunk_total_ms),
                "num_batches": send_stats["num_batches"],
                "num_frames": send_stats["num_frames"],
                "raw_bytes": send_stats["raw_bytes"],
                "ws_payload_bytes": send_stats["ws_payload_bytes"],
                "content_type": send_stats["content_type"],
            }
        )
    )


async def _generate_loop(ws: WebSocket, session: GenerateSession):
    adapter = session.adapter
    if adapter is None:
        raise ValueError("realtime adapter is not initialized")

    pending_send_task = None
    while not session.reached_max_chunks():
        try:
            if pending_send_task is not None and pending_send_task.done():
                await pending_send_task
                pending_send_task = None

            # send to scheduler and generate video chunk
            server_args = get_global_server_args()

            await adapter.wait_for_next_chunk(session)

            timer = RealtimeStageTimer()
            chunk_started = time.perf_counter()

            chunk = session.new_chunk()
            batch = adapter.prepare_next_request(
                session,
                server_args,
                chunk,
            )
            if batch.condition_inputs:
                logger.debug(
                    "consume realtime conditions, session_id=%s, block_idx=%s, kinds=%s",
                    session.id,
                    batch.block_idx,
                    sorted(batch.condition_inputs),
                )
            request_prepare_ms = timer.mark_ms()

            _, result = await process_generation_batch(async_scheduler_client, batch)
            scheduler_forward_ms = timer.mark_ms()

            # finish
            adapter.on_chunk_complete(session, result)
            if pending_send_task is not None:
                await pending_send_task
            if batch.realtime_output_pacing:
                await _send_output_and_log(
                    ws,
                    session,
                    chunk,
                    batch,
                    result,
                    request_prepare_ms,
                    scheduler_forward_ms,
                    chunk_started,
                )
                pending_send_task = None
            else:
                pending_send_task = asyncio.create_task(
                    _send_output_and_log(
                        ws,
                        session,
                        chunk,
                        batch,
                        result,
                        request_prepare_ms,
                        scheduler_forward_ms,
                        chunk_started,
                    )
                )

        except asyncio.CancelledError:
            if pending_send_task is not None:
                pending_send_task.cancel()
                await _await_realtime_task(pending_send_task)
            logger.info("generation completed, session_id=%s", session.id)
            break
        except WebSocketDisconnect:
            if pending_send_task is not None:
                pending_send_task.cancel()
                await _await_realtime_task(pending_send_task)
            logger.info(
                "client disconnected during generation, session_id=%s", session.id
            )
            break
        except Exception as e:
            if pending_send_task is not None:
                pending_send_task.cancel()
                await _await_realtime_task(pending_send_task)
            err_msg = str(e).splitlines()[0]
            logger.error("error during generate loop: %s", err_msg)
            try:
                await write_error_msg(f"error during generate loop: {err_msg}", ws)
            except Exception as send_error:
                logger.error(
                    "error during sending complete msg: %s",
                    send_error,
                )
            break
    else:
        if pending_send_task is not None:
            await pending_send_task
        logger.info(
            "generation reached max chunks, session_id=%s, max_chunks=%s",
            session.id,
            session.request.max_chunks if session.request is not None else None,
        )


async def _send_output_and_log(
    ws: WebSocket,
    session: GenerateSession,
    chunk: RealtimeChunkContext,
    batch: "Req",
    result,
    request_prepare_ms: float,
    scheduler_forward_ms: float,
    chunk_started: float,
) -> RealtimeFrameSendStats:
    if session.adapter is None:
        raise ValueError("realtime adapter is not initialized")
    pace_wait_ms = await _wait_for_realtime_output_slot(session, batch, result)
    send_stats = await session.adapter.send_output(
        ws,
        session,
        result,
        batch,
    )
    send_stats["pace_wait_ms"] = pace_wait_ms
    chunk_total_ms = (time.perf_counter() - chunk_started) * 1000
    _log_realtime_chunk_timing(
        session,
        chunk,
        batch,
        request_prepare_ms,
        scheduler_forward_ms,
        chunk_total_ms,
        send_stats,
    )
    await _send_realtime_chunk_stats(
        ws,
        session,
        chunk,
        batch,
        request_prepare_ms,
        scheduler_forward_ms,
        chunk_total_ms,
        send_stats,
    )
    return send_stats


def _result_num_frames(result) -> int:
    if result.raw_frame_batches is None:
        return 0
    return sum(len(frames) for frames in result.raw_frame_batches)


def _output_pacing_fps(batch: "Req") -> float:
    fps = float(batch.fps or 0)
    if batch.enable_frame_interpolation:
        fps *= 2 ** int(batch.frame_interpolation_exp or 1)
    return fps


async def _wait_for_realtime_output_slot(
    session: GenerateSession,
    batch: "Req",
    result,
) -> float:
    if not batch.realtime_output_pacing:
        return 0.0

    frame_count = _result_num_frames(result)
    output_fps = _output_pacing_fps(batch)
    if frame_count <= 0 or output_fps <= 0:
        return 0.0

    now = time.perf_counter()
    next_send_at = session.output_pace_next_send_at
    if next_send_at is None:
        next_send_at = now
    if (
        batch.realtime_event_id is not None
        and batch.realtime_event_id != session.output_pace_last_event_id
    ):
        next_send_at = min(next_send_at, now)
        session.output_pace_last_event_id = batch.realtime_event_id

    wait_s = max(0.0, next_send_at - now)
    if wait_s > 0:
        await asyncio.sleep(wait_s)

    send_started_at = time.perf_counter()
    session.output_pace_next_send_at = (
        max(next_send_at, send_started_at) + frame_count / output_fps
    )
    return wait_s * 1000


async def _await_realtime_task(task: asyncio.Task | None) -> None:
    if task is None:
        return
    try:
        await task
    except (asyncio.CancelledError, WebSocketDisconnect):
        pass
    except Exception as e:
        logger.debug("realtime task exited with error: %s", e)


async def _listen_events(ws: WebSocket, session: GenerateSession):
    """listen for user events: usually condition inputs"""
    async for message in ws.iter_bytes():
        data = None
        try:
            data = msgspec.msgpack.decode(message)
            if not isinstance(data, dict):
                raise ValueError("realtime event must be a map")
            realtime_event = RealtimeEvent.model_validate(data)
            if session.adapter is None:
                raise ValueError("realtime adapter is not initialized")
            event_log = session.adapter.ingest_event(session, realtime_event)
            logger.info(
                "receive realtime event, session_id=%s, event_id=%s, %s",
                session.id,
                realtime_event.event_id,
                event_log,
            )
        except Exception as e:
            event_kind = data.get("kind") if isinstance(data, dict) else None
            logger.warning("invalid event, kind=%s, error=%s", event_kind, e)
            await write_error_msg("invalid event", ws)
            continue


async def _listen_generate_request(ws: WebSocket, session: GenerateSession):
    while True:
        try:
            data = msgspec.msgpack.decode(await ws.receive_bytes())
            if not isinstance(data, dict):
                raise ValueError("generate request must be a map")

            realtime_req = RealtimeVideoGenerationsRequest.model_validate(data)
            adapter = get_realtime_model_adapter(get_global_server_args())
            session.set_adapter(adapter)
            await adapter.on_init(session, realtime_req)

            # Keep session state update atomic with validated request.
            session.set_request(realtime_req)
            break
        except WebSocketDisconnect:
            raise
        except Exception as e:
            logger.warning(
                "invalid generate request, session_id=%s, error=%s",
                session.id,
                e,
            )
            await write_error_msg("invalid generate request", ws)
            continue


async def _cleanup_realtime_session(
    session: GenerateSession,
    generate_task: asyncio.Task | None,
    listen_task: asyncio.Task | None,
) -> None:
    logger.info("terminating session, session_id=%s", session.id)
    for task in (generate_task, listen_task):
        if task and not task.done():
            task.cancel()
    for task in (generate_task, listen_task):
        if task is None:
            continue
        await _await_realtime_task(task)
    try:
        await async_scheduler_client.forward(
            ReleaseRealtimeSessionReq(session_id=session.id)
        )
    except Exception as e:
        logger.warning(
            "failed to release realtime session on scheduler, session_id=%s, error=%s",
            session.id,
            e,
        )
    if session.input_temp_dir is not None:
        shutil.rmtree(session.input_temp_dir, ignore_errors=True)
    session.dispose()


async def _close_realtime_websocket(
    websocket: WebSocket,
    *,
    code: int,
    reason: str,
) -> None:
    try:
        await websocket.close(code=code, reason=reason)
    except (RuntimeError, WebSocketDisconnect):
        pass


@router.websocket("/generate")
async def generate(websocket: WebSocket):
    """endpoint for creating a new realtime session"""
    await websocket.accept()
    if _ACTIVE_SESSION_IDS and not await _wait_for_active_session_slot():
        logger.warning(
            "reject realtime session because another session is active: %s",
            sorted(_ACTIVE_SESSION_IDS),
        )
        try:
            await write_error_msg(
                "another realtime session is already active", websocket
            )
        finally:
            await websocket.close(code=1008)
        return

    session = GenerateSession()
    _ACTIVE_SESSION_IDS.add(session.id)
    generate_task = None
    listen_task = None
    try:
        # receive new generate request
        await _listen_generate_request(websocket, session)

        # continuously generate video chunk
        generate_task = asyncio.create_task(_generate_loop(websocket, session))
        # continuously listen for user events
        listen_task = asyncio.create_task(_listen_events(websocket, session))

        wait_tasks = [generate_task, listen_task]
        await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)
        if generate_task.done() and session.reached_max_chunks():
            await _close_realtime_websocket(
                websocket,
                code=1000,
                reason="generation complete",
            )

    except WebSocketDisconnect:
        logger.info("client disconnected, session_id=%s", session.id)
    finally:
        try:
            await _cleanup_realtime_session(session, generate_task, listen_task)
        finally:
            _ACTIVE_SESSION_IDS.discard(session.id)


async def write_error_msg(error_msg: str, websocket: WebSocket):
    await websocket.send_bytes(
        msgspec.msgpack.encode({"type": "error", "content": error_msg})
    )
