# SPDX-License-Identifier: Apache-2.0

import asyncio
import shutil
import time
from typing import TYPE_CHECKING, Any

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
    empty_frame_send_stats,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.registry import (
    get_realtime_model_adapter,
)
from sglang.multimodal_gen.runtime.utils.realtime_trace import (
    CLIENT_TRACE_EVENT_KIND,
    compact_client_trace_event,
    log_realtime_trace,
    normalize_trace_id,
    register_realtime_trace_sink,
    unregister_realtime_trace_sink,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.timer import (
    RealtimeStageTimer,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    process_generation_batch,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    ReleaseRealtimeSessionReq,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.realtime.async_vae_client import (
    RealtimeVAEClient,
    RemoteDecodeHandle,
    RemoteFrameBatch,
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
_TRACE_EVENT_QUEUE_LIMIT = 256
_REALTIME_RESULT_STAGE_MARKERS = ("vae", "denois")


class _OrderedDecodeCoordinator:
    """Keep decode completion ordered while the caller denoises the next chunk."""

    def __init__(self) -> None:
        self.pending: asyncio.Task | None = None

    async def submit(self, factory) -> None:
        if self.pending is not None:
            await self.pending
        self.pending = asyncio.create_task(factory())

    async def finish(self) -> None:
        if self.pending is not None:
            await self.pending
            self.pending = None

    async def cancel(self) -> None:
        if self.pending is None:
            return
        self.pending.cancel()
        await asyncio.gather(self.pending, return_exceptions=True)
        self.pending = None


class _LockedRealtimeWebSocket:
    def __init__(self, websocket: WebSocket):
        self._websocket = websocket
        self._send_lock = asyncio.Lock()
        self._send_lock_owner: asyncio.Task | None = None
        self._send_lock_depth = 0

    async def send_bytes(self, payload):
        await self._acquire_send_lock()
        try:
            await self._websocket.send_bytes(payload)
        finally:
            self._release_send_lock()

    async def close(self, *args, **kwargs):
        await self._acquire_send_lock()
        try:
            await self._websocket.close(*args, **kwargs)
        finally:
            self._release_send_lock()

    def send_group(self):
        return _RealtimeWebSocketSendGroup(self)

    def __getattr__(self, name):
        return getattr(self._websocket, name)

    async def _acquire_send_lock(self):
        task = asyncio.current_task()
        if self._send_lock_owner is task:
            self._send_lock_depth += 1
            return
        await self._send_lock.acquire()
        self._send_lock_owner = task
        self._send_lock_depth = 1

    def _release_send_lock(self):
        task = asyncio.current_task()
        if self._send_lock_owner is not task:
            return
        self._send_lock_depth -= 1
        if self._send_lock_depth <= 0:
            self._send_lock_owner = None
            self._send_lock_depth = 0
            self._send_lock.release()


class _RealtimeWebSocketSendGroup:
    def __init__(self, websocket: _LockedRealtimeWebSocket):
        self.websocket = websocket

    async def __aenter__(self):
        await self.websocket._acquire_send_lock()
        return self.websocket

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.websocket._release_send_lock()
        return False


class _NullAsyncContext:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


def _transport_ms(value: float) -> int:
    return max(0, int(value + 0.5))


def _safe_len(value) -> int:
    try:
        return len(value)
    except TypeError:
        return 0


def _make_trace_queue_sink(
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue,
):
    def sink(payload: dict):
        def enqueue():
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass

        loop.call_soon_threadsafe(enqueue)

    return sink


def _install_realtime_trace_sink(session: GenerateSession, sink) -> None:
    trace_id = session.trace_id
    if not trace_id:
        return
    previous_trace_id = getattr(session, "_trace_sink_trace_id", None)
    if previous_trace_id == trace_id:
        return
    if previous_trace_id:
        unregister_realtime_trace_sink(previous_trace_id, sink)
    register_realtime_trace_sink(trace_id, sink)
    session._trace_sink_trace_id = trace_id


def _uninstall_realtime_trace_sink(session: GenerateSession, sink) -> None:
    trace_id = getattr(session, "_trace_sink_trace_id", None)
    if not trace_id:
        return
    unregister_realtime_trace_sink(trace_id, sink)
    session._trace_sink_trace_id = None


async def _send_realtime_trace_events(
    ws: WebSocket,
    queue: asyncio.Queue,
) -> None:
    while True:
        payload = await queue.get()
        await ws.send_bytes(
            msgspec.msgpack.encode(
                {
                    "type": "trace_event",
                    "trace": payload,
                }
            )
        )


async def _wait_for_active_session_slot(
    *,
    timeout_s: float = _ACTIVE_SESSION_WAIT_SECONDS,
    interval_s: float = _ACTIVE_SESSION_WAIT_INTERVAL_SECONDS,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while _ACTIVE_SESSION_IDS and time.monotonic() < deadline:
        await asyncio.sleep(interval_s)
    return not _ACTIVE_SESSION_IDS


def _coerce_metric_ms(value: Any) -> float | None:
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None
    if metric != metric:
        return None
    return metric


def _is_realtime_result_stage_metric(stage_name: str) -> bool:
    normalized = stage_name.lower()
    return any(marker in normalized for marker in _REALTIME_RESULT_STAGE_MARKERS)


def _realtime_stage_event_name(stage_name: str) -> str | None:
    normalized = stage_name.lower()
    if "denois" in normalized:
        return "server.model_denoise_complete"
    if "vae" in normalized and "encod" in normalized:
        return "server.vae_encode_complete"
    if "vae" in normalized and "decod" in normalized:
        return "server.vae_decode_complete"
    if "post" in normalized and "decod" in normalized:
        return "server.post_decode_complete"
    return None


def _iter_realtime_result_stage_metrics(result: Any):
    metrics_candidates = []
    metrics_list = getattr(result, "metrics_list", None)
    if metrics_list:
        metrics_candidates.extend(
            metrics for metrics in metrics_list if metrics is not None
        )
    metrics = getattr(result, "metrics", None)
    if metrics is not None:
        metrics_candidates.append(metrics)

    seen_metrics: set[int] = set()
    seen_stages: set[str] = set()
    for metrics in metrics_candidates:
        metrics_id = id(metrics)
        if metrics_id in seen_metrics:
            continue
        seen_metrics.add(metrics_id)

        stages = getattr(metrics, "stages", None)
        if not isinstance(stages, dict):
            continue
        for stage_name, duration_ms in stages.items():
            stage_name = str(stage_name)
            if stage_name in seen_stages:
                continue
            if not _is_realtime_result_stage_metric(stage_name):
                continue
            metric_ms = _coerce_metric_ms(duration_ms)
            if metric_ms is None:
                continue
            seen_stages.add(stage_name)
            yield stage_name, metric_ms, getattr(metrics, "request_id", None)


def _emit_realtime_result_stage_traces(
    session: GenerateSession,
    chunk: RealtimeChunkContext,
    batch: "Req",
    result: Any,
) -> None:
    if not getattr(session, "trace_id", None):
        return

    for stage_name, duration_ms, metrics_request_id in _iter_realtime_result_stage_metrics(
        result
    ):
        request_id = getattr(chunk, "request_id", None) or metrics_request_id
        log_realtime_trace(
            logger,
            session,
            "server.pipeline_stage_complete",
            request_id=request_id,
            chunk_index=getattr(batch, "block_idx", None),
            event_id=getattr(batch, "realtime_event_id", None),
            stage=stage_name,
            duration_ms=round(duration_ms, 3),
            source="scheduler_result_metrics",
        )
        stage_event_name = _realtime_stage_event_name(stage_name)
        if stage_event_name is not None:
            log_realtime_trace(
                logger,
                session,
                stage_event_name,
                request_id=request_id,
                chunk_index=getattr(batch, "block_idx", None),
                event_id=getattr(batch, "realtime_event_id", None),
                stage=stage_name,
                duration_ms=round(duration_ms, 3),
                source="scheduler_result_metrics",
            )


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
        "realtime chunk timing: trace_id=%s session_id=%s request_id=%s "
        "chunk_idx=%s event_id=%s condition_kinds=%s "
        "request_prepare=%.2fms scheduler_forward=%.2fms "
        "output_pace=%.2fms "
        "header_pack=%.2fms "
        "header_write=%.2fms raw_payload_build=%.2fms raw_write=%.2fms "
        "ws_write=%.2fms chunk_total=%.2fms batches=%d frames=%d "
        "frame_shape=%s raw_bytes=%d ws_payload_bytes=%d content_type=%s",
        session.trace_id,
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
    log_realtime_trace(
        logger,
        session,
        "server.chunk_complete",
        request_id=chunk.request_id,
        chunk_index=batch.block_idx,
        event_id=getattr(batch, "realtime_event_id", None),
        condition_kinds=sorted(batch.condition_inputs) if batch.condition_inputs else [],
        request_prepare_ms=round(request_prepare_ms, 3),
        scheduler_forward_ms=round(scheduler_forward_ms, 3),
        output_pace_ms=round(send_stats["pace_wait_ms"], 3),
        header_pack_ms=round(send_stats["header_pack_ms"], 3),
        header_write_ms=round(send_stats["header_write_ms"], 3),
        raw_payload_build_ms=round(send_stats["raw_payload_build_ms"], 3),
        raw_write_ms=round(send_stats["raw_write_ms"], 3),
        ws_write_ms=round(send_stats["ws_write_ms"], 3),
        chunk_total_ms=round(chunk_total_ms, 3),
        num_batches=send_stats["num_batches"],
        num_frames=send_stats["num_frames"],
        frame_shape=send_stats["frame_shape"],
        raw_bytes=send_stats["raw_bytes"],
        ws_payload_bytes=send_stats["ws_payload_bytes"],
        content_type=send_stats["content_type"],
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
                "trace_id": session.trace_id,
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
    server_args = get_global_server_args()
    if getattr(server_args, "realtime_vae_worker_url", None):
        return await _generate_loop_async_vae(ws, session, server_args)
    return await _generate_loop_local(ws, session)


def _merge_send_stats(
    target: RealtimeFrameSendStats,
    update: RealtimeFrameSendStats,
) -> None:
    for key in (
        "header_pack_ms",
        "header_write_ms",
        "raw_payload_build_ms",
        "raw_write_ms",
        "ws_write_ms",
        "pace_wait_ms",
        "raw_bytes",
        "ws_payload_bytes",
        "num_frames",
        "num_batches",
    ):
        target[key] += update[key]
    if update["frame_shape"] is not None:
        target["frame_shape"] = update["frame_shape"]
    if update["content_type"]:
        target["content_type"] = update["content_type"]


async def _send_remote_frame_batch(
    ws: WebSocket,
    session: GenerateSession,
    batch: "Req",
    frame_batch: RemoteFrameBatch,
    send_stats: RealtimeFrameSendStats,
) -> None:
    if session.adapter is None:
        raise ValueError("realtime adapter is not initialized")
    partial = OutputBatch(
        raw_frame_batches=[list(frame_batch.payloads)],
        raw_frame_content_type=frame_batch.content_type,
        raw_frame_metadata={
            "width": frame_batch.width,
            "height": frame_batch.height,
            "channels": 3,
            "bytes_per_frame": frame_batch.width * frame_batch.height * 3,
        },
        metrics=batch.metrics,
    )
    send_group = getattr(ws, "send_group", None)
    send_context = send_group() if send_group is not None else _NullAsyncContext()
    async with send_context:
        partial_stats = await session.adapter.send_output(ws, session, partial, batch)
    _merge_send_stats(send_stats, partial_stats)


async def _complete_remote_chunk(
    ws: WebSocket,
    session: GenerateSession,
    chunk: RealtimeChunkContext,
    batch: "Req",
    handle: RemoteDecodeHandle,
    send_stats: RealtimeFrameSendStats,
    request_prepare_ms: float,
    scheduler_forward_ms: float,
    chunk_started: float,
) -> None:
    remote_result = await handle.wait()
    chunk_total_ms = (time.perf_counter() - chunk_started) * 1000.0
    log_realtime_trace(
        logger,
        session,
        "server.remote_vae_complete",
        request_id=chunk.request_id,
        chunk_index=chunk.index,
        event_id=getattr(batch, "realtime_event_id", None),
        vae_queue_wait_ms=round(remote_result.queue_wait_ms, 3),
        vae_decode_ms=round(remote_result.decode_ms, 3),
        frame_encode_ms=round(remote_result.encode_ms, 3),
        latent_to_gateway_complete_ms=round(remote_result.transfer_ms, 3),
        num_frames=remote_result.num_frames,
    )
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
    session.generate_chunk_completed(chunk)


async def _generate_loop_async_vae(ws, session: GenerateSession, server_args):
    adapter = session.adapter
    if adapter is None or session.request is None:
        raise ValueError("realtime adapter and request must be initialized")

    session.max_inflight_chunks = 2
    client = RealtimeVAEClient(
        server_args.realtime_vae_worker_url,
        session_id=session.id,
        generation_id=session.generation_id,
        timeout_s=server_args.realtime_vae_timeout_s,
        max_message_bytes=server_args.realtime_vae_max_message_mb * 1024 * 1024,
    )
    session.vae_client = client
    output_format = session.request.realtime_output_format or "webp"
    quality = int(session.request.output_compression or 90)
    await client.open(
        output_format=output_format,
        quality=quality,
        preview_max_width=session.request.realtime_preview_max_width,
    )
    coordinator = _OrderedDecodeCoordinator()

    try:
        while session.can_schedule_chunk():
            if coordinator.pending is not None and coordinator.pending.done():
                await coordinator.finish()

            wait_started = time.perf_counter()
            await adapter.wait_for_next_chunk(session)
            wait_ms = (time.perf_counter() - wait_started) * 1000.0
            log_realtime_trace(
                logger,
                session,
                "server.chunk_wait_done",
                next_chunk_index=session.next_chunk_index,
                wait_ms=round(wait_ms, 3),
            )

            timer = RealtimeStageTimer()
            chunk_started = time.perf_counter()
            chunk = session.new_chunk()
            batch = adapter.prepare_next_request(session, server_args, chunk)
            request_prepare_ms = timer.mark_ms()
            log_realtime_trace(
                logger,
                session,
                "server.scheduler_forward_start",
                request_id=chunk.request_id,
                chunk_index=chunk.index,
                request_prepare_ms=round(request_prepare_ms, 3),
                event_id=getattr(batch, "realtime_event_id", None),
            )
            _, result = await process_generation_batch(async_scheduler_client, batch)
            scheduler_forward_ms = timer.mark_ms()
            _emit_realtime_result_stage_traces(session, chunk, batch, result)
            if result.realtime_latents is None or result.realtime_handoff is None:
                raise RuntimeError("remote VAE path received no latent handoff")

            send_stats = empty_frame_send_stats()

            async def on_frame_batch(
                frame_batch: RemoteFrameBatch,
                *,
                batch=batch,
                send_stats=send_stats,
            ) -> None:
                await _send_remote_frame_batch(
                    ws,
                    session,
                    batch,
                    frame_batch,
                    send_stats,
                )

            handle = await client.submit(
                result.realtime_latents,
                result.realtime_handoff,
                on_frame_batch=on_frame_batch,
            )
            result.realtime_latents = None
            await coordinator.submit(
                lambda chunk=chunk, batch=batch, handle=handle, send_stats=send_stats,
                request_prepare_ms=request_prepare_ms,
                scheduler_forward_ms=scheduler_forward_ms,
                chunk_started=chunk_started: _complete_remote_chunk(
                    ws,
                    session,
                    chunk,
                    batch,
                    handle,
                    send_stats,
                    request_prepare_ms,
                    scheduler_forward_ms,
                    chunk_started,
                )
            )
        await coordinator.finish()
    except asyncio.CancelledError:
        await coordinator.cancel()
        raise
    except WebSocketDisconnect:
        await coordinator.cancel()
        logger.info("client disconnected during async VAE generation: %s", session.id)
    except Exception as exc:
        await coordinator.cancel()
        err_msg = str(exc).splitlines()[0]
        logger.error("error during async VAE generate loop: %s", err_msg)
        try:
            await write_error_msg(f"error during generate loop: {err_msg}", ws)
        except Exception:
            pass
    finally:
        await client.close()
        session.vae_client = None


async def _generate_loop_local(ws: WebSocket, session: GenerateSession):
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

            wait_started = time.perf_counter()
            await adapter.wait_for_next_chunk(session)
            wait_for_next_chunk_ms = (time.perf_counter() - wait_started) * 1000
            log_realtime_trace(
                logger,
                session,
                "server.chunk_wait_done",
                next_chunk_index=session.generate_chunk_cnt,
                wait_ms=round(wait_for_next_chunk_ms, 3),
            )

            timer = RealtimeStageTimer()
            chunk_started = time.perf_counter()

            chunk = session.new_chunk()
            log_realtime_trace(
                logger,
                session,
                "server.chunk_prepare_start",
                request_id=chunk.request_id,
                chunk_index=chunk.index,
            )
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
            log_realtime_trace(
                logger,
                session,
                "server.scheduler_forward_start",
                request_id=chunk.request_id,
                chunk_index=batch.block_idx,
                request_prepare_ms=round(request_prepare_ms, 3),
                event_id=getattr(batch, "realtime_event_id", None),
                condition_kinds=sorted(batch.condition_inputs) if batch.condition_inputs else [],
            )

            _, result = await process_generation_batch(async_scheduler_client, batch)
            scheduler_forward_ms = timer.mark_ms()
            log_realtime_trace(
                logger,
                session,
                "server.scheduler_forward_done",
                request_id=chunk.request_id,
                chunk_index=batch.block_idx,
                scheduler_forward_ms=round(scheduler_forward_ms, 3),
            )
            _emit_realtime_result_stage_traces(session, chunk, batch, result)

            # finish
            adapter.on_chunk_complete(session, result)
            if pending_send_task is not None:
                await pending_send_task
            if getattr(batch, "realtime_output_pacing", False):
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
    log_realtime_trace(
        logger,
        session,
        "server.output_send_start",
        request_id=chunk.request_id,
        chunk_index=batch.block_idx,
        event_id=getattr(batch, "realtime_event_id", None),
    )
    pace_wait_ms = await _wait_for_realtime_output_slot(session, batch, result)
    send_group = getattr(ws, "send_group", None)
    send_context = send_group() if send_group is not None else _NullAsyncContext()
    async with send_context:
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
    log_realtime_trace(
        logger,
        session,
        "server.chunk_stats_sent",
        request_id=chunk.request_id,
        chunk_index=batch.block_idx,
        event_id=getattr(batch, "realtime_event_id", None),
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
    if not getattr(batch, "realtime_output_pacing", False):
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
            if realtime_event.kind == CLIENT_TRACE_EVENT_KIND:
                client_event = compact_client_trace_event(realtime_event.payload)
                event_name = str(client_event.pop("name", "client_trace"))
                log_realtime_trace(logger, session, event_name, **client_event)
                continue
            if session.adapter is None:
                raise ValueError("realtime adapter is not initialized")
            log_realtime_trace(
                logger,
                session,
                "server.event_received",
                event_id=realtime_event.event_id,
                kind=realtime_event.kind,
                client_sent_perf_ms=realtime_event.client_sent_perf_ms,
                client_sent_epoch_ms=realtime_event.client_sent_epoch_ms,
                payload_bytes=_safe_len(message),
            )
            event_log = session.adapter.ingest_event(session, realtime_event)
            log_realtime_trace(
                logger,
                session,
                "server.event_ingested",
                event_id=realtime_event.event_id,
                kind=realtime_event.kind,
                adapter_log=event_log,
            )
            logger.info(
                "receive realtime event, trace_id=%s, session_id=%s, event_id=%s, %s",
                session.trace_id,
                session.id,
                realtime_event.event_id,
                event_log,
            )
        except Exception as e:
            event_kind = data.get("kind") if isinstance(data, dict) else None
            logger.warning("invalid event, kind=%s, error=%s", event_kind, e)
            await write_error_msg("invalid event", ws)
            continue


async def _listen_generate_request(
    ws: WebSocket,
    session: GenerateSession,
    trace_sink=None,
):
    while True:
        try:
            receive_started = time.perf_counter()
            raw_message = await ws.receive_bytes()
            receive_wait_ms = (time.perf_counter() - receive_started) * 1000
            data = msgspec.msgpack.decode(raw_message)
            if not isinstance(data, dict):
                raise ValueError("generate request must be a map")

            realtime_req = RealtimeVideoGenerationsRequest.model_validate(data)
            session.bind_trace(realtime_req)
            if trace_sink is not None:
                _install_realtime_trace_sink(session, trace_sink)
            log_realtime_trace(
                logger,
                session,
                "server.init_received",
                receive_wait_ms=round(receive_wait_ms, 3),
                payload_bytes=len(raw_message),
                model=bool(realtime_req.model),
                size=realtime_req.size,
                fps=realtime_req.fps,
                num_frames=realtime_req.num_frames,
                max_chunks=realtime_req.max_chunks,
                client_trace=session.client_trace,
            )
            adapter = get_realtime_model_adapter(get_global_server_args())
            session.set_adapter(adapter)
            log_realtime_trace(
                logger,
                session,
                "server.adapter_init_start",
                adapter=adapter.__class__.__name__,
            )
            await adapter.on_init(session, realtime_req)
            log_realtime_trace(
                logger,
                session,
                "server.adapter_init_done",
                adapter=adapter.__class__.__name__,
            )

            # Keep session state update atomic with validated request.
            session.set_request(realtime_req)
            log_realtime_trace(
                logger,
                session,
                "server.init_ready",
                adapter=adapter.__class__.__name__,
            )
            break
        except WebSocketDisconnect:
            raise
        except Exception as e:
            log_realtime_trace(
                logger,
                session,
                "server.init_invalid",
                error=str(e).splitlines()[0],
            )
            logger.warning(
                "invalid generate request, trace_id=%s, session_id=%s, error=%s",
                session.trace_id,
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
    log_realtime_trace(logger, session, "server.session_cleanup_start")
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
    log_realtime_trace(logger, session, "server.session_cleanup_done")
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


async def _wait_for_server_warmup(websocket: WebSocket) -> None:
    warmup_done = getattr(websocket.app.state, "server_warmup_done", None)
    if warmup_done is not None and not warmup_done.is_set():
        await warmup_done.wait()


@router.websocket("/generate")
async def generate(websocket: WebSocket):
    """endpoint for creating a new realtime session"""
    await websocket.accept()
    ws = _LockedRealtimeWebSocket(websocket)
    session = GenerateSession()
    session.trace_id = normalize_trace_id(
        websocket.query_params.get("trace_id"), fallback=session.trace_id
    )
    trace_queue: asyncio.Queue = asyncio.Queue(maxsize=_TRACE_EVENT_QUEUE_LIMIT)
    trace_sink = _make_trace_queue_sink(asyncio.get_running_loop(), trace_queue)
    _install_realtime_trace_sink(session, trace_sink)
    trace_task = asyncio.create_task(_send_realtime_trace_events(ws, trace_queue))
    log_realtime_trace(
        logger,
        session,
        "server.ws_accepted",
        client=str(websocket.client) if websocket.client else None,
    )
    warmup_started = time.perf_counter()
    await _wait_for_server_warmup(websocket)
    warmup_wait_ms = (time.perf_counter() - warmup_started) * 1000
    if warmup_wait_ms > 1.0:
        log_realtime_trace(
            logger,
            session,
            "server.warmup_wait_done",
            wait_ms=round(warmup_wait_ms, 3),
        )
    if _ACTIVE_SESSION_IDS and not await _wait_for_active_session_slot():
        logger.warning(
            "reject realtime session because another session is active: %s",
            sorted(_ACTIVE_SESSION_IDS),
        )
        log_realtime_trace(
            logger,
            session,
            "server.session_rejected",
            reason="another realtime session is already active",
            active_sessions=sorted(_ACTIVE_SESSION_IDS),
        )
        try:
            await write_error_msg(
                "another realtime session is already active", ws
            )
        finally:
            await ws.close(code=1008)
            trace_task.cancel()
            await _await_realtime_task(trace_task)
            _uninstall_realtime_trace_sink(session, trace_sink)
        return

    _ACTIVE_SESSION_IDS.add(session.id)
    generate_task = None
    listen_task = None
    try:
        # receive new generate request
        await _listen_generate_request(ws, session, trace_sink)

        # continuously generate video chunk
        generate_task = asyncio.create_task(_generate_loop(ws, session))
        # continuously listen for user events
        listen_task = asyncio.create_task(_listen_events(ws, session))

        wait_tasks = [generate_task, listen_task]
        await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)
        if generate_task.done() and session.reached_max_chunks():
            await _close_realtime_websocket(
                ws,
                code=1000,
                reason="generation complete",
            )

    except WebSocketDisconnect:
        log_realtime_trace(logger, session, "server.client_disconnected")
        logger.info("client disconnected, session_id=%s", session.id)
    finally:
        try:
            await _cleanup_realtime_session(session, generate_task, listen_task)
        finally:
            trace_task.cancel()
            await _await_realtime_task(trace_task)
            _uninstall_realtime_trace_sink(session, trace_sink)
            _ACTIVE_SESSION_IDS.discard(session.id)


async def write_error_msg(error_msg: str, websocket: WebSocket):
    await websocket.send_bytes(
        msgspec.msgpack.encode({"type": "error", "content": error_msg})
    )
