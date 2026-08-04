# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import shutil
import time
from dataclasses import dataclass
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
    calculate_overlap_ms,
    calculate_overlap_ratio,
    log_realtime_trace,
    normalize_trace_id,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.timer import (
    RealtimeStageTimer,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    process_generation_batch,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    ReleaseRealtimeSessionReq,
    ReplaceQueuedRealtimeReq,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.realtime.async_vae_client import (
    RealtimeVAEClient,
    RemoteDecodeHandle,
    RemoteFrameBatch,
)
from sglang.multimodal_gen.runtime.realtime.admission import (
    AdmissionRejected,
    DynamoDBSessionLeaseStore,
    InMemorySessionLeaseStore,
    RealtimeAdmissionController,
    SessionLease,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

logger = init_logger(__name__)

_REALTIME_CONTROL_REFRESH_TIMEOUT_S = 1.0
router = APIRouter(prefix="/v1/realtime_video", tags=["realtime"])
_REALTIME_RESULT_STAGE_MARKERS = ("vae", "denois")
_ADMISSION_CONTROLLER: RealtimeAdmissionController | None = None
_ADMISSION_CONFIG: tuple | None = None


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


@dataclass(frozen=True, slots=True)
class _GatewayManagedConfig:
    session_id: str
    generation_id: str
    coordinator_token: str
    vae_worker_url: str
    output_url: str
    output_token: str


def _gateway_managed_config(websocket: WebSocket) -> _GatewayManagedConfig | None:
    query = websocket.query_params
    if query.get("gateway_managed") != "1":
        return None
    fields = {
        "session_id": query.get("session_id"),
        "generation_id": query.get("generation_id"),
        "coordinator_token": query.get("coordinator_token"),
        "vae_worker_url": query.get("realtime_vae_worker_url"),
        "output_url": query.get("gateway_output_url"),
        "output_token": query.get("gateway_output_token"),
    }
    if not all(fields.values()):
        raise AdmissionRejected("INVALID_GATEWAY_ASSIGNMENT")
    return _GatewayManagedConfig(**fields)


def _log_previous_chunk_overlap(
    session: GenerateSession,
    *,
    current_chunk_index: int,
) -> None:
    previous_chunk_index = current_chunk_index - 1
    vae_interval = session.vae_intervals.get(previous_chunk_index)
    denoise_interval = session.denoise_intervals.get(current_chunk_index)
    if vae_interval is None or denoise_interval is None:
        return
    log_realtime_trace(
        logger,
        session,
        "server.vae_denoise_overlap_complete",
        chunk_index=previous_chunk_index,
        next_chunk_index=current_chunk_index,
        overlap_with_next_denoise_ms=round(
            calculate_overlap_ms(vae_interval, denoise_interval), 3
        ),
        overlap_ratio=round(
            calculate_overlap_ratio(vae_interval, denoise_interval), 4
        ),
    )


def _sync_batch_realtime_metadata(batch: "Req", result: OutputBatch) -> None:
    metadata = getattr(result, "realtime_request_metadata", None) or getattr(
        result, "realtime_handoff", None
    )
    if not isinstance(metadata, dict):
        return
    for batch_field, metadata_field in (
        ("realtime_event_id", "event_id"),
        ("realtime_action_version", "action_version"),
        ("realtime_prompt_version", "prompt_version"),
    ):
        if metadata_field in metadata:
            setattr(batch, batch_field, metadata[metadata_field])


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


def _get_admission_controller(server_args) -> RealtimeAdmissionController:
    global _ADMISSION_CONFIG, _ADMISSION_CONTROLLER
    config = (
        int(server_args.realtime_max_sessions),
        float(server_args.realtime_session_lease_ttl_s),
        float(server_args.realtime_admission_wait_s),
        server_args.realtime_session_lease_table,
    )
    if _ADMISSION_CONTROLLER is not None and _ADMISSION_CONFIG == config:
        return _ADMISSION_CONTROLLER
    if server_args.realtime_session_lease_table:
        store = DynamoDBSessionLeaseStore(
            server_args.realtime_session_lease_table,
            max_active_sessions=server_args.realtime_max_sessions,
            ttl_s=server_args.realtime_session_lease_ttl_s,
        )
    else:
        store = InMemorySessionLeaseStore(
            max_active_sessions=server_args.realtime_max_sessions,
            ttl_s=server_args.realtime_session_lease_ttl_s,
        )
    _ADMISSION_CONTROLLER = RealtimeAdmissionController(
        store,
        wait_timeout_s=server_args.realtime_admission_wait_s,
    )
    _ADMISSION_CONFIG = config
    return _ADMISSION_CONTROLLER


def _resolve_realtime_user_id(
    websocket: WebSocket,
    *,
    require_authenticated: bool = False,
) -> str:
    principal = websocket.scope.get("user")
    if principal is not None:
        for name in ("sub", "id", "username"):
            value = (
                principal.get(name)
                if isinstance(principal, dict)
                else getattr(principal, name, None)
            )
            if value:
                return f"auth:{str(value)[:240]}"
    if require_authenticated:
        raise AdmissionRejected("AUTHENTICATED_USER_REQUIRED")
    query_user = websocket.query_params.get("user_id")
    if query_user:
        return f"query:{query_user[:240]}"
    header_user = websocket.headers.get("x-user-id")
    if header_user:
        return f"header:{header_user[:240]}"
    client_host = websocket.client.host if websocket.client else "unknown"
    return f"client:{client_host}"


def _user_id_fingerprint(user_id: str) -> str:
    return hashlib.blake2s(user_id.encode("utf-8"), digest_size=8).hexdigest()


async def _session_watchdog(
    session: GenerateSession,
    controller: RealtimeAdmissionController,
    lease: SessionLease,
    *,
    idle_timeout_s: float,
    max_lifetime_s: float,
    lease_ttl_s: float,
) -> str:
    interval_s = min(5.0, max(0.1, lease_ttl_s / 3.0))
    while True:
        await asyncio.sleep(interval_s)
        now = time.monotonic()
        if now - session.created_at >= max_lifetime_s:
            return "maximum session lifetime reached"
        if now - session.last_client_activity_at >= idle_timeout_s:
            return "session idle timeout"
        try:
            await controller.renew(lease)
        except AdmissionRejected:
            return "session lease lost"


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
    if session.vae_worker_url or getattr(server_args, "realtime_vae_worker_url", None):
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
            "frame_batch_index": frame_batch.frame_batch_index,
            "num_frame_batches": (
                frame_batch.frame_batch_index + 1 if frame_batch.is_final else 0
            ),
            "is_final_frame_batch": frame_batch.is_final,
        },
        metrics=batch.metrics,
    )
    send_group = getattr(ws, "send_group", None)
    send_context = send_group() if send_group is not None else _NullAsyncContext()
    async with send_context:
        partial_stats = await session.adapter.send_output(ws, session, partial, batch)
    _merge_send_stats(send_stats, partial_stats)


def _make_remote_frame_batch_handler(
    ws: WebSocket,
    session: GenerateSession,
    chunk: RealtimeChunkContext,
    batch: "Req",
    send_stats: RealtimeFrameSendStats,
):
    first_remote_batch = True

    async def on_frame_batch(frame_batch: RemoteFrameBatch) -> None:
        nonlocal first_remote_batch
        if first_remote_batch:
            first_remote_batch = False
            log_realtime_trace(
                logger,
                session,
                "server.remote_first_frame_received",
                request_id=chunk.request_id,
                chunk_index=chunk.index,
                event_id=getattr(batch, "realtime_event_id", None),
                frame_batch_index=frame_batch.frame_batch_index,
            )
        await _send_remote_frame_batch(
            ws,
            session,
            batch,
            frame_batch,
            send_stats,
        )

    return on_frame_batch


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
    vae_started: float,
) -> None:
    remote_result = await handle.wait()
    vae_completed = remote_result.completed_at
    session.vae_intervals[chunk.index] = (vae_started, vae_completed)
    next_denoise = session.denoise_intervals.get(chunk.index + 1)
    overlap_ms = None
    overlap_ratio = None
    if next_denoise is not None:
        overlap_ms = calculate_overlap_ms(
            session.vae_intervals[chunk.index], next_denoise
        )
        overlap_ratio = calculate_overlap_ratio(
            session.vae_intervals[chunk.index], next_denoise
        )
    chunk_total_ms = (time.perf_counter() - chunk_started) * 1000.0
    common = {
        "request_id": chunk.request_id,
        "chunk_index": chunk.index,
        "event_id": getattr(batch, "realtime_event_id", None),
    }
    log_realtime_trace(
        logger,
        session,
        "server.vae_queue_wait_complete",
        **common,
        duration_ms=round(remote_result.queue_wait_ms, 3),
    )
    log_realtime_trace(
        logger,
        session,
        "server.vae_decode_complete",
        **common,
        duration_ms=round(remote_result.decode_ms, 3),
        source="remote_taehv",
    )
    log_realtime_trace(
        logger,
        session,
        "server.frame_encode_complete",
        **common,
        duration_ms=round(remote_result.encode_ms, 3),
    )
    log_realtime_trace(
        logger,
        session,
        "server.frame_transfer_complete",
        **common,
        duration_ms=round(remote_result.transfer_ms, 3),
        first_frame_ms=(
            round(remote_result.first_frame_ms, 3)
            if remote_result.first_frame_ms is not None
            else None
        ),
    )
    remote_fields = {
        "request_id": chunk.request_id,
        "chunk_index": chunk.index,
        "event_id": getattr(batch, "realtime_event_id", None),
        "vae_queue_wait_ms": round(remote_result.queue_wait_ms, 3),
        "vae_decode_ms": round(remote_result.decode_ms, 3),
        "frame_encode_ms": round(remote_result.encode_ms, 3),
        "latent_to_gateway_complete_ms": round(remote_result.transfer_ms, 3),
        "latent_serialize_ms": round(remote_result.serialize_ms, 3),
        "latent_send_ms": round(remote_result.latent_send_ms, 3),
        "vae_credit_wait_ms": round(remote_result.credit_wait_ms, 3),
        "first_frame_ms": (
            round(remote_result.first_frame_ms, 3)
            if remote_result.first_frame_ms is not None
            else None
        ),
        "num_frames": remote_result.num_frames,
    }
    if overlap_ms is not None and overlap_ratio is not None:
        remote_fields.update(
            overlap_with_next_denoise_ms=round(overlap_ms, 3),
            overlap_ratio=round(overlap_ratio, 4),
        )
    log_realtime_trace(
        logger,
        session,
        "server.remote_vae_complete",
        **remote_fields,
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
    vae_worker_url = (
        getattr(session, "vae_worker_url", None)
        or server_args.realtime_vae_worker_url
    )
    if not vae_worker_url:
        raise ValueError("realtime VAE worker URL is required")
    client = RealtimeVAEClient(
        vae_worker_url,
        session_id=session.id,
        generation_id=session.generation_id,
        timeout_s=server_args.realtime_vae_timeout_s,
        max_message_bytes=server_args.realtime_vae_max_message_mb * 1024 * 1024,
    )
    session.vae_client = client
    output_format = session.request.realtime_output_format or "webp"
    quality = int(session.request.output_compression or 90)
    coordinator = _OrderedDecodeCoordinator()

    try:
        await client.open(
            output_format=output_format,
            quality=quality,
            preview_max_width=session.request.realtime_preview_max_width,
            output_url=getattr(session, "gateway_output_url", None),
            output_token=getattr(session, "gateway_output_token", None),
            trace_id=session.trace_id,
        )
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
            session.bind_chunk_request(chunk, batch)
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
            denoise_started = time.perf_counter()
            _, result = await process_generation_batch(async_scheduler_client, batch)
            _sync_batch_realtime_metadata(batch, result)
            denoise_completed = time.perf_counter()
            session.denoise_intervals[chunk.index] = (
                denoise_started,
                denoise_completed,
            )
            _log_previous_chunk_overlap(
                session,
                current_chunk_index=chunk.index,
            )
            scheduler_forward_ms = timer.mark_ms()
            _emit_realtime_result_stage_traces(session, chunk, batch, result)
            if result.realtime_latents is None or result.realtime_handoff is None:
                raise RuntimeError("remote VAE path received no latent handoff")

            send_stats = empty_frame_send_stats()
            on_frame_batch = _make_remote_frame_batch_handler(
                ws,
                session,
                chunk,
                batch,
                send_stats,
            )

            vae_started = time.perf_counter()
            handle = await client.submit(
                result.realtime_latents,
                result.realtime_handoff,
                on_frame_batch=on_frame_batch,
            )
            log_realtime_trace(
                logger,
                session,
                "server.latent_transfer_accepted",
                request_id=chunk.request_id,
                chunk_index=chunk.index,
                event_id=getattr(batch, "realtime_event_id", None),
                latent_serialize_ms=round(handle.serialize_ms, 3),
                latent_send_ms=round(handle.latent_send_ms, 3),
                vae_credit_wait_ms=round(handle.credit_wait_ms, 3),
            )
            result.realtime_latents = None
            await coordinator.submit(
                lambda chunk=chunk, batch=batch, handle=handle, send_stats=send_stats,
                request_prepare_ms=request_prepare_ms,
                scheduler_forward_ms=scheduler_forward_ms,
                chunk_started=chunk_started,
                vae_started=vae_started: _complete_remote_chunk(
                    ws,
                    session,
                    chunk,
                    batch,
                    handle,
                    send_stats,
                    request_prepare_ms,
                    scheduler_forward_ms,
                    chunk_started,
                    vae_started,
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
            session.bind_chunk_request(chunk, batch)
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
            _sync_batch_realtime_metadata(batch, result)
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


async def _refresh_latest_queued_controls(
    session: GenerateSession,
    event_kind: str,
    event_id: int | None,
) -> None:
    chunk = session.latest_active_chunk
    if chunk is None or session.adapter is None:
        return
    current_batch = session.active_batches.get(chunk.index)
    if current_batch is None:
        return
    replacement = session.adapter.refresh_queued_request(
        session,
        get_global_server_args(),
        chunk,
        current_batch,
        event_kind,
    )
    if replacement is None:
        return
    try:
        output = await asyncio.wait_for(
            async_scheduler_client.forward(
                ReplaceQueuedRealtimeReq(
                    session_id=session.id,
                    generation_id=session.generation_id,
                    chunk_index=chunk.index,
                    request_id=chunk.request_id,
                    replacement=replacement,
                )
            ),
            timeout=_REALTIME_CONTROL_REFRESH_TIMEOUT_S,
        )
    except TimeoutError:
        output = OutputBatch(
            output={
                "replaced": False,
                "buffered": False,
                "too_late": False,
                "invalid": False,
                "timeout": True,
            }
        )
    result = output.output if isinstance(output, OutputBatch) else None
    result = result if isinstance(result, dict) else {}
    replaced = bool(result.get("replaced"))
    buffered = bool(result.get("buffered"))
    if (
        (replaced or buffered)
        and session.active_chunks.get(chunk.index) == chunk
        and session.active_batches.get(chunk.index) is current_batch
    ):
        session.active_batches[chunk.index] = replacement
    log_realtime_trace(
        logger,
        session,
        "server.queued_controls_refresh",
        request_id=chunk.request_id,
        chunk_index=chunk.index,
        event_id=event_id,
        kind=event_kind,
        replaced=replaced,
        buffered=buffered,
        too_late=bool(result.get("too_late")),
        invalid=bool(result.get("invalid")),
        timeout=bool(result.get("timeout")),
    )


async def _drain_queued_control_refreshes(session: GenerateSession) -> None:
    while session.pending_control_refresh is not None:
        event_kind, event_id = session.pending_control_refresh
        session.pending_control_refresh = None
        try:
            await _refresh_latest_queued_controls(session, event_kind, event_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "failed to refresh queued realtime controls, session_id=%s, "
                "event_id=%s, error=%s",
                session.id,
                event_id,
                exc,
            )


def _schedule_queued_control_refresh(
    session: GenerateSession,
    event_kind: str,
    event_id: int | None,
) -> None:
    if event_kind != "camera_actions":
        return
    session.pending_control_refresh = (event_kind, event_id)
    task = session.control_refresh_task
    if task is None or task.done():
        session.control_refresh_task = asyncio.create_task(
            _drain_queued_control_refreshes(session)
        )


async def _listen_events(ws: WebSocket, session: GenerateSession):
    """listen for user events: usually condition inputs"""
    async for message in ws.iter_bytes():
        data = None
        try:
            data = msgspec.msgpack.decode(message)
            if not isinstance(data, dict):
                raise ValueError("realtime event must be a map")
            realtime_event = RealtimeEvent.model_validate(data)
            if realtime_event.kind == "heartbeat":
                session.mark_client_activity()
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
            session.mark_event_version(realtime_event.kind)
            session.mark_client_activity()
            _schedule_queued_control_refresh(
                session, realtime_event.kind, realtime_event.event_id
            )
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
):
    while True:
        try:
            receive_started = time.perf_counter()
            raw_message = await ws.receive_bytes()
            session.mark_client_activity()
            receive_wait_ms = (time.perf_counter() - receive_started) * 1000
            data = msgspec.msgpack.decode(raw_message)
            if not isinstance(data, dict):
                raise ValueError("generate request must be a map")

            realtime_req = RealtimeVideoGenerationsRequest.model_validate(data)
            session.bind_trace(realtime_req)
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


async def _release_scheduler_realtime_session(
    session_id: str,
    *,
    attempts: int = 3,
    retry_delay_s: float = 0.05,
    timeout_s: float = 2.0,
) -> bool:
    for attempt in range(max(1, attempts)):
        try:
            await asyncio.wait_for(
                async_scheduler_client.forward(
                    ReleaseRealtimeSessionReq(session_id=session_id)
                ),
                timeout=max(0.001, timeout_s),
            )
            return True
        except Exception as exc:
            if attempt + 1 >= max(1, attempts):
                logger.warning(
                    "failed to release realtime session on scheduler after %d "
                    "attempts, session_id=%s, error=%s",
                    attempt + 1,
                    session_id,
                    exc,
                )
                return False
            await asyncio.sleep(max(0.0, retry_delay_s) * (attempt + 1))
    return False


async def _cleanup_realtime_session(
    session: GenerateSession,
    generate_task: asyncio.Task | None,
    listen_task: asyncio.Task | None,
) -> None:
    log_realtime_trace(logger, session, "server.session_cleanup_start")
    logger.info("terminating session, session_id=%s", session.id)
    refresh_task = session.control_refresh_task
    for task in (generate_task, listen_task, refresh_task):
        if task and not task.done():
            task.cancel()
    for task in (generate_task, listen_task, refresh_task):
        if task is None:
            continue
        await _await_realtime_task(task)
    await _release_scheduler_realtime_session(session.id)
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


async def _wait_for_initialization_or_watchdog(
    initialization_task: asyncio.Task,
    watchdog_task: asyncio.Task,
) -> str | None:
    done, _ = await asyncio.wait(
        (initialization_task, watchdog_task),
        return_when=asyncio.FIRST_COMPLETED,
    )
    if initialization_task in done:
        await initialization_task
        return None

    reason = await watchdog_task
    initialization_task.cancel()
    await _await_realtime_task(initialization_task)
    return reason


@router.websocket("/generate")
async def generate(websocket: WebSocket):
    """endpoint for creating a new realtime session"""
    await websocket.accept()
    ws = _LockedRealtimeWebSocket(websocket)
    server_args = get_global_server_args()
    try:
        gateway_config = _gateway_managed_config(websocket)
        if gateway_config is None:
            user_id = _resolve_realtime_user_id(
                websocket,
                require_authenticated=server_args.realtime_require_authenticated_user,
            )
        else:
            user_id = "gateway-managed"
    except AdmissionRejected as exc:
        await write_error_msg(f"realtime admission rejected: {exc.reason}", ws)
        await _close_realtime_websocket(ws, code=1008, reason=exc.reason)
        return
    session = GenerateSession(
        session_id=gateway_config.session_id if gateway_config else None,
        generation_id=gateway_config.generation_id if gateway_config else None,
    )
    if gateway_config is not None:
        session.vae_worker_url = gateway_config.vae_worker_url
        session.gateway_output_url = gateway_config.output_url
        session.gateway_output_token = gateway_config.output_token
    session.trace_id = normalize_trace_id(
        websocket.query_params.get("trace_id"), fallback=session.trace_id
    )
    log_realtime_trace(
        logger,
        session,
        "server.ws_accepted",
        client=str(websocket.client) if websocket.client else None,
    )
    controller = (
        None if gateway_config is not None else _get_admission_controller(server_args)
    )
    lease = None
    initialization_task = None
    generate_task = None
    listen_task = None
    watchdog_task = None
    try:
        if controller is not None:
            try:
                lease = await controller.admit(
                    user_id,
                    session.id,
                    session.generation_id,
                    wait_for_capacity=False,
                )
            except AdmissionRejected as exc:
                log_realtime_trace(
                    logger,
                    session,
                    "server.session_rejected",
                    reason=exc.reason,
                    retry_after_s=exc.retry_after_s,
                )
                await write_error_msg(f"realtime admission rejected: {exc.reason}", ws)
                await _close_realtime_websocket(
                    ws,
                    code=1008,
                    reason=exc.reason,
                )
                return
            watchdog_task = asyncio.create_task(
                _session_watchdog(
                    session,
                    controller,
                    lease,
                    idle_timeout_s=server_args.realtime_session_idle_timeout_s,
                    max_lifetime_s=server_args.realtime_session_max_lifetime_s,
                    lease_ttl_s=server_args.realtime_session_lease_ttl_s,
                )
            )
        log_realtime_trace(
            logger,
            session,
            "server.session_admitted",
            user_key_hash=_user_id_fingerprint(user_id),
            gateway_managed=gateway_config is not None,
        )

        async def initialize_session() -> None:
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
            await _listen_generate_request(ws, session)

        initialization_task = asyncio.create_task(initialize_session())
        if watchdog_task is None:
            await initialization_task
            init_close_reason = None
        else:
            init_close_reason = await _wait_for_initialization_or_watchdog(
                initialization_task,
                watchdog_task,
            )
        if init_close_reason is not None:
            log_realtime_trace(
                logger,
                session,
                "server.session_watchdog_closed",
                reason=init_close_reason,
                phase="initialization",
            )
            await _close_realtime_websocket(
                ws,
                code=1000,
                reason=init_close_reason,
            )
            return

        # continuously generate video chunk
        generate_task = asyncio.create_task(_generate_loop(ws, session))
        # continuously listen for user events
        listen_task = asyncio.create_task(_listen_events(ws, session))
        wait_tasks = [generate_task, listen_task]
        if watchdog_task is not None:
            wait_tasks.append(watchdog_task)
        await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)
        if generate_task.done() and session.reached_max_chunks():
            await _close_realtime_websocket(
                ws,
                code=1000,
                reason="generation complete",
            )
        elif watchdog_task is not None and watchdog_task.done():
            reason = await watchdog_task
            log_realtime_trace(
                logger,
                session,
                "server.session_watchdog_closed",
                reason=reason,
            )
            await _close_realtime_websocket(ws, code=1000, reason=reason)

    except WebSocketDisconnect:
        log_realtime_trace(logger, session, "server.client_disconnected")
        logger.info("client disconnected, session_id=%s", session.id)
    finally:
        try:
            if watchdog_task is not None:
                watchdog_task.cancel()
                await _await_realtime_task(watchdog_task)
            if initialization_task is not None and not initialization_task.done():
                initialization_task.cancel()
                await _await_realtime_task(initialization_task)
            await _cleanup_realtime_session(session, generate_task, listen_task)
        finally:
            if lease is not None and controller is not None:
                await controller.release(lease)


async def write_error_msg(error_msg: str, websocket: WebSocket):
    await websocket.send_bytes(
        msgspec.msgpack.encode({"type": "error", "content": error_msg})
    )
