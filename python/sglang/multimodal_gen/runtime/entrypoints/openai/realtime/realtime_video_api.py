# SPDX-License-Identifier: Apache-2.0

import asyncio
import shutil
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from msgpack import packb, unpackb

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.registry import (
    get_realtime_model_adapter,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
    empty_frame_send_stats,
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

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/realtime_video", tags=["realtime"])
_ACTIVE_SESSION_IDS: set[str] = set()


async def _generate_loop(ws: WebSocket, session: GenerateSession):
    while True:
        try:
            start = time.perf_counter()
            timings: dict[str, float] = {}

            # send to scheduler and generate video chunk
            stage_start = time.perf_counter()
            server_args = get_global_server_args()
            if session.adapter is None:
                raise ValueError("realtime adapter is not initialized")
            chunk = session.new_chunk()
            batch = session.adapter.prepare_next_request(
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
            timings["prepare_ms"] = (time.perf_counter() - stage_start) * 1000.0
            stage_start = time.perf_counter()
            _, result = await process_generation_batch(async_scheduler_client, batch)
            timings["scheduler_forward_ms"] = (
                time.perf_counter() - stage_start
            ) * 1000.0

            send_stats = empty_frame_send_stats(result.raw_frame_content_type)
            if session.adapter is not None:
                send_stats = await session.adapter.send_output(
                    ws,
                    session,
                    result,
                    batch,
                )
            timings["header_pack_ms"] = float(send_stats["header_pack_ms"])
            timings["header_write_ms"] = float(send_stats["header_write_ms"])
            timings["raw_payload_build_ms"] = float(send_stats["raw_payload_build_ms"])
            timings["raw_write_ms"] = float(send_stats["raw_write_ms"])
            timings["ws_write_ms"] = float(send_stats["ws_write_ms"])
            timings["total_ms"] = (time.perf_counter() - start) * 1000.0

            # finish
            session.adapter.on_chunk_complete(session, result)

            logger.info(
                "realtime chunk timing: session_id=%s request_id=%s "
                "chunk_idx=%s request_prepare=%.2fms scheduler_forward=%.2fms "
                "header_pack=%.2fms "
                "header_write=%.2fms raw_payload_build=%.2fms raw_write=%.2fms "
                "ws_write=%.2fms chunk_total=%.2fms batches=%d frames=%d "
                "frame_shape=%s raw_bytes=%d ws_payload_bytes=%d content_type=%s",
                session.id,
                chunk.request_id,
                batch.block_idx,
                timings["prepare_ms"],
                timings["scheduler_forward_ms"],
                timings["header_pack_ms"],
                timings["header_write_ms"],
                timings["raw_payload_build_ms"],
                timings["raw_write_ms"],
                timings["ws_write_ms"],
                timings["total_ms"],
                send_stats["num_batches"],
                send_stats["num_frames"],
                send_stats["frame_shape"],
                send_stats["raw_bytes"],
                send_stats["ws_payload_bytes"],
                send_stats["content_type"],
            )

        except asyncio.CancelledError:
            logger.info("generation completed, session_id=%s", session.id)
            break
        except Exception as e:
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
            data = unpackb(message, raw=False)
            if not isinstance(data, dict):
                raise ValueError("realtime event must be a map")
            realtime_event = RealtimeEvent.model_validate(data)
            if session.adapter is None:
                raise ValueError("realtime adapter is not initialized")
            event_log = session.adapter.ingest_event(session, realtime_event)
            logger.debug(
                "receive realtime event, session_id=%s, %s",
                session.id,
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
            data = unpackb(await ws.receive_bytes(), raw=False)
            if not isinstance(data, dict):
                raise ValueError("generate request must be a map")

            realtime_req = RealtimeVideoGenerationsRequest.model_validate(data)
            adapter = get_realtime_model_adapter(get_global_server_args())
            session.set_adapter(adapter)
            await adapter.on_init(session, realtime_req)

            # Keep session state update atomic with validated request.
            session.set_request(realtime_req)
            break
        except Exception as e:
            logger.warning(
                "invalid generate request, session_id=%s, error=%s",
                session.id,
                e,
            )
            await write_error_msg("invalid generate request", ws)
            continue


@router.websocket("/generate")
async def generate(websocket: WebSocket):
    """endpoint for creating a new realtime session"""
    await websocket.accept()
    if _ACTIVE_SESSION_IDS:
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

    except WebSocketDisconnect:
        logger.info("client disconnected, session_id=%s", session.id)
    finally:
        logger.info("terminating session, session_id=%s", session.id)
        _ACTIVE_SESSION_IDS.discard(session.id)
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
        if session:
            session.dispose()


async def write_error_msg(error_msg: str, websocket: WebSocket):
    await websocket.send_bytes(
        packb({"type": "error", "content": error_msg}, use_bin_type=True)
    )
