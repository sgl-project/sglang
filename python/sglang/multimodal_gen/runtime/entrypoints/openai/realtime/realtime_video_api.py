# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
import os
import shutil
import tempfile
import time
from typing import TypedDict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from msgpack import packb, unpackb
from PIL import Image

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeAction,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
    RealtimeVideoMode,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    process_generation_batch,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    ReleaseRealtimeSessionReq,
    prepare_request,
)
from sglang.multimodal_gen.runtime.pipelines_core.realtime_session import (
    REALTIME_SESSION_ID_EXTRA_KEY,
    RETURN_ENCODED_FRAMES_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.realtime_video import (
    RAW_RGB_CHANNELS,
    RAW_RGB_CONTENT_TYPE,
)

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/realtime_video", tags=["realtime"])
_ACTIVE_SESSION_IDS: set[str] = set()


class RealtimeFrameBatchHeader(TypedDict, total=False):
    type: str
    request_id: str
    chunk_index: int
    content_type: str
    num_frames: int
    total_size: int
    format: str
    width: int
    height: int
    channels: int
    bytes_per_frame: int


class RealtimeFrameSendStats(TypedDict):
    msgpack_pack_ms: float
    header_send_ms: float
    raw_join_ms: float
    raw_send_ms: float
    ws_send_ms: float
    raw_bytes: int
    ws_payload_bytes: int
    num_frames: int
    num_batches: int


def _raw_rgb_frame_metadata(batch) -> dict[str, int | str]:
    frame_width = batch.width
    frame_height = batch.height
    if frame_width is None or frame_height is None:
        return {}

    frame_width = int(frame_width)
    frame_height = int(frame_height)
    if batch.enable_upscaling:
        upscaling_scale = int(batch.upscaling_scale or 1)
        frame_width *= upscaling_scale
        frame_height *= upscaling_scale

    return {
        "format": "rgb24",
        "width": frame_width,
        "height": frame_height,
        "channels": RAW_RGB_CHANNELS,
        "bytes_per_frame": frame_width * frame_height * RAW_RGB_CHANNELS,
    }


async def _send_frame_batches(
    ws: WebSocket,
    frame_batches: list[list[bytes]],
    *,
    content_type: str,
    chunk_index_start: int,
    request_id: str,
    frame_metadata: dict[str, int | str] | None = None,
) -> RealtimeFrameSendStats:
    chunk_index = chunk_index_start
    metadata = frame_metadata or {}
    header_pack_ms = 0.0
    header_send_ms = 0.0
    raw_join_ms = 0.0
    raw_send_ms = 0.0
    raw_bytes = 0
    ws_payload_bytes = 0
    num_frames = 0
    num_batches = 0
    for frames in frame_batches:
        frame_bytes = sum(len(frame) for frame in frames)
        header: RealtimeFrameBatchHeader = {
            "type": "frame_batch_header",
            "request_id": request_id,
            "chunk_index": chunk_index,
            "content_type": content_type,
            "num_frames": len(frames),
            "total_size": frame_bytes,
        }
        header.update(metadata)

        stage_start = time.perf_counter()
        header_payload = packb(header, use_bin_type=True)
        header_pack_ms += (time.perf_counter() - stage_start) * 1000.0

        stage_start = time.perf_counter()
        await ws.send_bytes(header_payload)
        header_send_ms += (time.perf_counter() - stage_start) * 1000.0

        stage_start = time.perf_counter()
        raw_payload = b"".join(frames)
        raw_join_ms += (time.perf_counter() - stage_start) * 1000.0

        stage_start = time.perf_counter()
        await ws.send_bytes(raw_payload)
        raw_send_ms += (time.perf_counter() - stage_start) * 1000.0

        raw_bytes += frame_bytes
        ws_payload_bytes += len(header_payload) + len(raw_payload)
        num_frames += len(frames)
        num_batches += 1
        chunk_index += 1

    return {
        "msgpack_pack_ms": header_pack_ms,
        "header_send_ms": header_send_ms,
        "raw_join_ms": raw_join_ms,
        "raw_send_ms": raw_send_ms,
        "ws_send_ms": header_send_ms + raw_send_ms,
        "raw_bytes": raw_bytes,
        "ws_payload_bytes": ws_payload_bytes,
        "num_frames": num_frames,
        "num_batches": num_batches,
    }


async def _generate_loop(ws: WebSocket, session: GenerateSession):
    while True:
        try:
            start = time.perf_counter()
            timings: dict[str, float] = {}
            # For stream-driven v2v (no first_frame),
            # wait until enough frames are buffered for this block.
            stage_start = time.perf_counter()
            if (
                session.request is not None
                and session.is_v2v_enabled()
                and session.request.first_frame is None
            ):
                while not session.has_pending_video_frames():
                    await asyncio.sleep(0.01)
            timings["wait_video_ms"] = (time.perf_counter() - stage_start) * 1000.0

            session.new_request()

            # send to scheduler and generate video chunk
            stage_start = time.perf_counter()
            server_args = get_global_server_args()
            sampling_params = session.build_sampling_params()
            batch = prepare_request(
                server_args=server_args,
                sampling_params=sampling_params,
            )
            batch.session = session.realtime_session
            batch.extra[REALTIME_SESSION_ID_EXTRA_KEY] = session.id
            batch.extra[RETURN_ENCODED_FRAMES_EXTRA_KEY] = True
            batch.block_idx = session.generate_chunk_cnt
            chunk_size = batch.extra.get("chunk_size", 1)
            control_chunk = session.sample_control_chunk(chunk_size)
            if control_chunk is not None:
                logger.debug(
                    "consume realtime control, session_id=%s, block_idx=%s, chunk_size=%s, num_control_frames=%s",
                    session.id,
                    batch.block_idx,
                    chunk_size,
                    len(control_chunk),
                )
                batch.extra["actions"] = control_chunk
            timings["prepare_ms"] = (time.perf_counter() - stage_start) * 1000.0
            stage_start = time.perf_counter()
            _, result = await process_generation_batch(async_scheduler_client, batch)
            timings["process_generation_ms"] = (
                time.perf_counter() - stage_start
            ) * 1000.0

            send_stats: RealtimeFrameSendStats = {
                "msgpack_pack_ms": 0.0,
                "header_send_ms": 0.0,
                "raw_join_ms": 0.0,
                "raw_send_ms": 0.0,
                "ws_send_ms": 0.0,
                "raw_bytes": 0,
                "ws_payload_bytes": 0,
                "num_frames": 0,
                "num_batches": 0,
            }
            frame_metadata = {}
            if result.encoded_frame_batches is not None:
                frame_metadata = (
                    result.encoded_frame_metadata or _raw_rgb_frame_metadata(batch)
                    if result.encoded_frame_content_type == RAW_RGB_CONTENT_TYPE
                    else {}
                )
                send_stats = await _send_frame_batches(
                    ws,
                    result.encoded_frame_batches,
                    content_type=result.encoded_frame_content_type,
                    chunk_index_start=batch.block_idx,
                    request_id=session.request_id,
                    frame_metadata=frame_metadata,
                )
            timings["msgpack_pack_ms"] = float(send_stats["msgpack_pack_ms"])
            timings["header_send_ms"] = float(send_stats["header_send_ms"])
            timings["raw_join_ms"] = float(send_stats["raw_join_ms"])
            timings["raw_send_ms"] = float(send_stats["raw_send_ms"])
            timings["ws_send_ms"] = float(send_stats["ws_send_ms"])
            timings["total_ms"] = (time.perf_counter() - start) * 1000.0

            # finish
            session.generate_chunk_completed()

            logger.info(
                "realtime video stage timing: session_id=%s request_id=%s "
                "chunk_idx=%s wait_video=%.2fms prepare=%.2fms "
                "process_generation=%.2fms msgpack_pack=%.2fms "
                "header_send=%.2fms raw_join=%.2fms raw_send=%.2fms "
                "ws_send=%.2fms total=%.2fms batches=%d frames=%d "
                "frame_shape=%s raw_bytes=%d ws_payload_bytes=%d content_type=%s",
                session.id,
                session.request_id,
                batch.block_idx,
                timings["wait_video_ms"],
                timings["prepare_ms"],
                timings["process_generation_ms"],
                timings["msgpack_pack_ms"],
                timings["header_send_ms"],
                timings["raw_join_ms"],
                timings["raw_send_ms"],
                timings["ws_send_ms"],
                timings["total_ms"],
                send_stats["num_batches"],
                send_stats["num_frames"],
                (
                    (
                        int(frame_metadata["height"]),
                        int(frame_metadata["width"]),
                        int(frame_metadata["channels"]),
                    )
                    if frame_metadata
                    else None
                ),
                send_stats["raw_bytes"],
                send_stats["ws_payload_bytes"],
                result.encoded_frame_content_type,
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


async def _listen_actions(ws: WebSocket, session: GenerateSession):
    async for message in ws.iter_bytes():
        data = None
        try:
            data = unpackb(message, raw=False)
            if not isinstance(data, dict):
                raise ValueError("realtime action must be a map")
            realtime_action = RealtimeAction.model_validate(data)
            if realtime_action.type == "video":
                if not session.is_v2v_enabled():
                    logger.warning(
                        "ignore video action in non-v2v mode, session_id=%s",
                        session.id,
                    )
                    await write_error_msg(
                        "video action requires mode=v2v (or first_frame in auto mode)",
                        ws,
                    )
                    continue

                encoded_frames = list(realtime_action.video_frames or [])
                if realtime_action.video_frame is not None:
                    encoded_frames.append(realtime_action.video_frame)
                if len(encoded_frames) == 0:
                    raise ValueError(
                        "video action requires video_frame or video_frames"
                    )

                frames = []
                for frame_bytes in encoded_frames:
                    image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
                    frames.append(image)
                session.append_video_frames(frames)
                total_bytes = sum(len(frame) for frame in encoded_frames)
                action_log = (
                    f"type=video, num_frames={len(encoded_frames)}, "
                    f"total_bytes={total_bytes}"
                )
            elif realtime_action.type == "control":
                control_chunk = realtime_action.control_chunk
                if control_chunk is not None:
                    session.append_control_chunk(control_chunk)
                    action_log = (
                        f"type=control, mode=chunk, chunk_len={len(control_chunk)}"
                    )
                else:
                    raise ValueError("control action requires control_chunk")
            else:
                if not realtime_action.action_content:
                    raise ValueError("prompt action requires action_content")
                session.append_action(realtime_action)
                action_log = (
                    f"type=prompt, content_len={len(realtime_action.action_content)}"
                )
            logger.debug(
                "receive realtime action, session_id=%s, %s",
                session.id,
                action_log,
            )
        except Exception as e:
            action_type = data.get("type") if isinstance(data, dict) else None
            logger.warning("invalid action, type=%s, error=%s", action_type, e)
            await write_error_msg("invalid action", ws)
            continue


async def _listen_generate_request(ws: WebSocket, session: GenerateSession):
    while True:
        try:
            data = unpackb(await ws.receive_bytes(), raw=False)
            if not isinstance(data, dict):
                raise ValueError("generate request must be a map")

            mode_raw = data.get("mode")
            if mode_raw is None:
                mode = None
            else:
                if isinstance(mode_raw, bytes):
                    try:
                        mode_raw = mode_raw.decode("utf-8")
                    except Exception as e:
                        raise ValueError("mode must be a utf-8 string") from e
                if not isinstance(mode_raw, str):
                    raise ValueError("mode must be one of: t2v, v2v")
                try:
                    mode = RealtimeVideoMode(mode_raw)
                except Exception as e:
                    raise ValueError("mode must be one of: t2v, v2v") from e
            if mode == RealtimeVideoMode.T2V and data.get("first_frame") is not None:
                raise ValueError("first_frame is not allowed when mode=t2v")

            realtime_req = RealtimeVideoGenerationsRequest.model_validate(data)
            # TODO(puf147): convert RGB for krea
            # params.start_frame = Image.open(params.start_frame).convert("RGB")
            if realtime_req.first_frame is not None:
                server_args = get_global_server_args()
                if server_args.input_save_path is not None:
                    uploads_dir = server_args.input_save_path
                    os.makedirs(uploads_dir, exist_ok=True)
                else:
                    if session.input_temp_dir is None:
                        session.input_temp_dir = tempfile.mkdtemp(
                            prefix="sglang_input_"
                        )
                    uploads_dir = session.input_temp_dir

                target_path = os.path.join(uploads_dir, f"{session.id}_first_frame")
                image_path = await save_image_to_path(
                    realtime_req.first_frame, target_path
                )
                realtime_req.first_frame = image_path

            # Keep session state update atomic with validated request.
            session.set_mode(mode)
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

        # generate video chunk
        generate_task = asyncio.create_task(_generate_loop(websocket, session))
        # listen for user actions
        listen_task = asyncio.create_task(_listen_actions(websocket, session))

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
